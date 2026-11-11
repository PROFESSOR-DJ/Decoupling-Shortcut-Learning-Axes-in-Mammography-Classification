"""Earlier version of core_pipeline.py with the same model, strategies and evaluation.

Kept because external_validation.py imports it for the INbreast runs.
Differences: no resume from partial results, and Grad-CAM examples are sampled at random.
"""

import os, glob, re, random, copy, pickle
import numpy as np
import pandas as pd
import cv2

cv2.setNumThreads(0)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import models, transforms
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import wilcoxon, pointbiserialr, norm as _norm
import matplotlib.pyplot as plt
import warnings

from paths import CBIS_DDSM_DIR, OUTPUT_DIR

warnings.filterwarnings('ignore', category=ConvergenceWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# Seeds and strategies
SEED_LIST = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
ABLATIONS = ['baseline', 'grl_only', 'distcorr', 'mi_min', 'camreg', 'full']

# Data and output folders (set in src/paths.py)
DATA_ROOT = CBIS_DDSM_DIR
CSV_DIR = os.path.join(DATA_ROOT, "csv")
JPEG_DIR = os.path.join(DATA_ROOT, "jpeg")

OUT_DIR = OUTPUT_DIR
CACHE_FILE = os.path.join(OUT_DIR, "cbis_image_cache.pkl")

# Training settings
IMG_SIZE = 224
BATCH_SIZE = 32
MAX_EPOCHS = 20
PATIENCE = 5
LR = 1e-4
NUM_WORKERS = 0

# Loss weights, ramped in over LAMBDA_RAMP_EPOCHS epochs
MAX_LAMBDA_GRL = 5.0
GRL_ADV_WEIGHT = 2.0
GRL_REINIT_EVERY = 4
MAX_WEIGHT_DISTCORR = 8.0
MAX_WEIGHT_CAMREG = 2.0
MAX_WEIGHT_MI = 1.0
CLUB_CRITIC_STEPS = 5
LAMBDA_RAMP_EPOCHS = 8

# Extra runs inside main(): distcorr weight sweep, Grad-CAM figure, MI diagnostic
SENSITIVITY_WEIGHTS = [3.0, 4.0, 5.0, 6.0, 7.0]
SENSITIVITY_SEEDS = SEED_LIST

QUALITATIVE_SEED = SEED_LIST[0]
N_QUALITATIVE_EXAMPLES = 4

MI_DIAGNOSTIC_SEEDS = [42, 43, 44]

# Shortcut injection: P(dense | malignant) = BIAS_STRENGTH in the training fold
BIAS_TRAINING = True
BIAS_STRENGTH = 0.85

IMAGE_CACHE = {}


def build_metadata():
    """Read the CBIS-DDSM CSVs, resolve image paths and derive malignancy and density labels."""
    print("=== CSV files found ===")
    csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
    for f in csv_files:
        print(" ", os.path.basename(f))

    case_csvs = [f for f in csv_files if 'case_description' in os.path.basename(f).lower()]
    dfs = []
    for f in case_csvs:
        df = pd.read_csv(f)
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        df['source_file'] = os.path.basename(f)
        dfs.append(df)

    meta_raw = pd.concat(dfs, ignore_index=True)
    print(f"Total case rows across all CSVs: {len(meta_raw)}")

    jpeg_folder_index = {}
    for entry in os.listdir(JPEG_DIR):
        full_path = os.path.join(JPEG_DIR, entry)
        if os.path.isdir(full_path):
            jpeg_folder_index[entry] = sorted(os.listdir(full_path))

    UID_PATTERN = re.compile(r'1\.3\.6\.1\.4\.1\.9590\.100\.1\.2\.\d+')

    def find_uid_matches(path_str):
        if not isinstance(path_str, str):
            return []
        return UID_PATTERN.findall(path_str)

    path_col_candidates = [c for c in meta_raw.columns if 'path' in c]

    def resolve_full_image(row):
        for col in path_col_candidates:
            uids = find_uid_matches(row.get(col, None))
            for uid in uids:
                if uid in jpeg_folder_index:
                    files = jpeg_folder_index[uid]
                    full_candidates = [f for f in files if f.startswith('1-')]
                    chosen = full_candidates[0] if full_candidates else files[0]
                    return os.path.join(JPEG_DIR, uid, chosen)
        return None

    meta_raw['filepath'] = meta_raw.apply(resolve_full_image, axis=1)
    match_rate = meta_raw['filepath'].notna().mean()
    print(f"MATCH RATE: {match_rate:.1%}")
    meta_raw = meta_raw.dropna(subset=['filepath']).reset_index(drop=True)

    meta_raw['malignant'] = meta_raw['pathology'].str.upper().str.contains('MALIGNANT').astype(int)
    meta_raw['density'] = pd.to_numeric(meta_raw['breast_density'], errors='coerce')
    meta_raw = meta_raw.dropna(subset=['density']).reset_index(drop=True)
    meta_raw['density_binary'] = (meta_raw['density'] >= meta_raw['density'].median()).astype(int)

    view_col = next((c for c in meta_raw.columns if c == 'image_view'), None)
    if view_col is not None:
        meta_raw['view'] = meta_raw[view_col].astype(str).str.upper().str.extract(r'(CC|MLO)')[0]
    else:
        meta_raw['view'] = np.nan
    n_with_view = meta_raw['view'].notna().sum()
    print(f"View column found: {view_col is not None} ({n_with_view}/{len(meta_raw)} rows have CC/MLO)")

    print(f"Final usable rows: {len(meta_raw)}")
    print("Malignant / Benign counts:\n", meta_raw['malignant'].value_counts())
    return meta_raw, 'patient_id'


def segment_breast(img_gray):
    """Breast mask: Gaussian blur, Otsu threshold, largest connected component."""
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)
    return clean_mask


def build_or_load_cache(filepaths):
    """Resize every image to 224x224 with its mask and cache the result on disk."""
    global IMAGE_CACHE
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached images+masks from {CACHE_FILE} ...")
        with open(CACHE_FILE, 'rb') as f:
            IMAGE_CACHE = pickle.load(f)
        print(f"Loaded {len(IMAGE_CACHE)} cached entries.")
        return
    print(f"Precomputing {len(filepaths)} images+masks (one-time cost, cached to disk after)...")
    for i, fp in enumerate(filepaths):
        img_gray = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        img_gray = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
        mask = segment_breast(img_gray)
        IMAGE_CACHE[fp] = (img_gray.astype(np.uint8), mask.astype(np.uint8))
        if (i + 1) % 500 == 0:
            print(f"  cached {i+1}/{len(filepaths)}")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(IMAGE_CACHE, f)
    print(f"Cache built and saved to {CACHE_FILE}")


class CBISDataset(Dataset):
    """Dataset returning (image, label, mask, density) for one split."""
    def __init__(self, df, img_size=IMG_SIZE, augment=False):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.augment = augment
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fp = row['filepath']
        if fp in IMAGE_CACHE:
            img_gray, mask = IMAGE_CACHE[fp]
        else:
            img_gray = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            img_gray = cv2.resize(img_gray, (self.img_size, self.img_size))
            mask = segment_breast(img_gray)

        img_gray = img_gray.copy()
        mask = mask.copy()
        if self.augment and random.random() < 0.5:
            img_gray = np.fliplr(img_gray).copy()
            mask = np.fliplr(mask).copy()

        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        img_t = self.norm(img_t)
        mask_t = torch.from_numpy((mask > 0).astype(np.float32))

        return (img_t, mask_t, int(row['malignant']), int(row['density_binary']))


def bias_split_by_correlation(df, seed, bias_strength=BIAS_STRENGTH):
    """Subsample the training fold so density and malignancy are correlated (shortcut injection)."""
    def bias_group(group_df, target_frac_pos, rng):
        pos = group_df[group_df.density_binary == 1]
        neg = group_df[group_df.density_binary == 0]
        if len(pos) == 0 or len(neg) == 0:
            return group_df
        n_total = int(min(len(pos) / target_frac_pos, len(neg) / (1 - target_frac_pos)))
        n_pos = int(round(n_total * target_frac_pos))
        n_neg = n_total - n_pos
        pos_s = pos.sample(n=min(n_pos, len(pos)), random_state=rng)
        neg_s = neg.sample(n=min(n_neg, len(neg)), random_state=rng)
        return pd.concat([pos_s, neg_s])

    mal1 = df[df.malignant == 1]
    mal0 = df[df.malignant == 0]
    biased = pd.concat([
        bias_group(mal1, bias_strength, seed),
        bias_group(mal0, 1 - bias_strength, seed + 1),
    ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return biased


def group_masks(df):
    """Aligned = density agrees with the injected direction; conflicting = it does not."""
    aligned = ((df.malignant == 1) & (df.density_binary == 1)) | \
              ((df.malignant == 0) & (df.density_binary == 0))
    return aligned.values, (~aligned).values


def _pairwise_dist(x):
    x = x.view(x.size(0), -1)
    sq = (x ** 2).sum(1, keepdim=True)
    d2 = torch.clamp(sq + sq.t() - 2 * x @ x.t(), min=0.0)
    return torch.sqrt(d2 + 1e-9)


def distance_correlation(x, y):
    """Differentiable batch distance correlation between features and density."""
    n = x.size(0)
    a = _pairwise_dist(x)
    b = _pairwise_dist(y)
    A = a - a.mean(0, keepdim=True) - a.mean(1, keepdim=True) + a.mean()
    B = b - b.mean(0, keepdim=True) - b.mean(1, keepdim=True) + b.mean()
    dcov2_xy = (A * B).sum() / (n * n)
    dcov2_xx = (A * A).sum() / (n * n)
    dcov2_yy = (B * B).sum() / (n * n)
    denom = torch.clamp(dcov2_xx * dcov2_yy, min=1e-9)
    return torch.sqrt(torch.clamp(dcov2_xy, min=0.0) / torch.sqrt(denom) + 1e-12)


def club_mi_upper_bound(club_logits_diag, density):
    """CLUB upper bound on mutual information between features and density (mi_min)."""
    logsig_pos = F.logsigmoid(club_logits_diag)
    logsig_neg = F.logsigmoid(-club_logits_diag)
    term1 = density * logsig_pos + (1 - density) * logsig_neg
    y_bar = density.mean()
    term2 = y_bar * logsig_pos + (1 - y_bar) * logsig_neg
    return (term1 - term2).mean()


class GradReverse(torch.autograd.Function):
    """Gradient reversal layer used by the GRL adversary."""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)


class DCCAMNet(nn.Module):
    """ResNet-18 encoder + malignancy head + density head, with closed-form CAM."""
    def __init__(self, use_grl=False):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        backbone.avgpool = nn.Identity()
        self.backbone = backbone
        self.main_head = nn.Linear(self.feat_dim, 1)
        self.nuisance_head = nn.Linear(self.feat_dim, 1)
        self.use_grl = use_grl

    def _forward_spatial(self, x):
        b = self.backbone
        x = b.conv1(x); x = b.bn1(x); x = b.relu(x); x = b.maxpool(x)
        x = b.layer1(x); x = b.layer2(x); x = b.layer3(x)
        conv_feats = b.layer4(x)
        pooled = F.adaptive_avg_pool2d(conv_feats, 1).flatten(1)
        return pooled, conv_feats

    def forward(self, x, lambda_=1.0):
        pooled, conv_feats = self._forward_spatial(x)
        main_logit = self.main_head(pooled).squeeze(1)
        if self.use_grl:
            reversed_feats = grad_reverse(pooled, lambda_)
            nuisance_logit = self.nuisance_head(reversed_feats).squeeze(1)
        else:
            nuisance_logit = self.nuisance_head(pooled.detach()).squeeze(1)
        return main_logit, nuisance_logit, pooled, conv_feats

    def cam(self, conv_feats):
        w = self.main_head.weight.view(1, -1, 1, 1)
        cam = F.relu((w * conv_feats).sum(1))
        return cam

    def club_logits(self, pooled):
        w = self.nuisance_head.weight.detach()
        b = self.nuisance_head.bias.detach() if self.nuisance_head.bias is not None else None
        return F.linear(pooled, w, b).squeeze(1)


class GradCAM:
    """Grad-CAM on the last convolutional block, used at test time."""
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        self.fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self.bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x):
        self.model.zero_grad()
        main_logit, _, _, _ = self.model(x)
        main_logit.sum().backward(retain_graph=False)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1))
        cam = F.interpolate(cam.unsqueeze(1), size=(IMG_SIZE, IMG_SIZE),
                             mode='bilinear', align_corners=False).squeeze(1)
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


def spurious_attribution_score(cam, mask):
    """SAS: share of attribution that falls outside the breast mask."""
    outside = (cam * (1 - mask)).sum(dim=(1, 2))
    total = cam.sum(dim=(1, 2)) + 1e-8
    return (outside / total).cpu().numpy()


def make_loader(dataset, batch_size, shuffle):
    """DataLoader with the project defaults."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                       num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))


def strategy_flags(strategy):
    """Which penalties are switched on for each strategy name."""
    return {
        'baseline':  dict(use_grl=False, use_distcorr=False, use_mi=False, use_camreg=False),
        'grl_only':  dict(use_grl=True,  use_distcorr=False, use_mi=False, use_camreg=False),
        'distcorr':  dict(use_grl=False, use_distcorr=True,  use_mi=False, use_camreg=False),
        'mi_min':    dict(use_grl=False, use_distcorr=False, use_mi=True,  use_camreg=False),
        'camreg':    dict(use_grl=False, use_distcorr=False, use_mi=False, use_camreg=True),
        'full':      dict(use_grl=False, use_distcorr=True,  use_mi=False, use_camreg=True),
    }[strategy]


def train_model(strategy, train_df, val_df, seed, log_diagnostics=False):
    """Train one model for one strategy and seed, with early stopping on validation AUROC."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    flags = strategy_flags(strategy)
    model = DCCAMNet(use_grl=flags['use_grl']).to(device)

    if flags['use_mi']:
        optimizer = torch.optim.Adam(
            list(model.backbone.parameters()) + list(model.main_head.parameters()), lr=LR)
        critic_optimizer = torch.optim.Adam(model.nuisance_head.parameters(), lr=LR)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        critic_optimizer = None
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

    train_loader = make_loader(CBISDataset(train_df, augment=True), BATCH_SIZE, shuffle=True)
    val_loader = make_loader(CBISDataset(val_df, augment=False), BATCH_SIZE, shuffle=False)

    best_val_auroc, best_state, patience_ct = -np.inf, None, 0
    any_ramped_component = flags['use_grl'] or flags['use_distcorr'] or flags['use_mi'] or flags['use_camreg']
    checkpoint_eligible_from = LAMBDA_RAMP_EPOCHS if any_ramped_component else 0
    diagnostic_history = []

    for epoch in range(MAX_EPOCHS):
        model.train()
        ramp = min(1.0, epoch / LAMBDA_RAMP_EPOCHS)
        lambda_grl = MAX_LAMBDA_GRL * ramp
        w_dc = MAX_WEIGHT_DISTCORR * ramp
        w_cam = MAX_WEIGHT_CAMREG * ramp
        w_mi = MAX_WEIGHT_MI * ramp

        if flags['use_grl'] and GRL_REINIT_EVERY and epoch > 0 and epoch % GRL_REINIT_EVERY == 0:
            model.nuisance_head.reset_parameters()

        total_loss = 0.0
        total_club_estimate = 0.0
        for imgs, masks, labels, density in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            density = density.float().to(device, non_blocking=True)

            if flags['use_mi']:
                with torch.no_grad():
                    pooled_cached, _ = model._forward_spatial(imgs)
                for _ in range(CLUB_CRITIC_STEPS):
                    critic_optimizer.zero_grad()
                    critic_logit = model.nuisance_head(pooled_cached).squeeze(1)
                    critic_loss = F.binary_cross_entropy_with_logits(critic_logit, density)
                    critic_loss.backward()
                    critic_optimizer.step()

                optimizer.zero_grad()
                with autocast('cuda', enabled=(device.type == 'cuda')):
                    main_logit, _, pooled, conv_feats = model(imgs, lambda_=lambda_grl)
                    main_loss = F.binary_cross_entropy_with_logits(main_logit, labels)
                    club_logit_diag = model.club_logits(pooled)
                    club_estimate = club_mi_upper_bound(club_logit_diag, density)
                    loss = main_loss + w_mi * club_estimate
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_club_estimate += club_estimate.item()

            else:
                optimizer.zero_grad()
                with autocast('cuda', enabled=(device.type == 'cuda')):
                    main_logit, nuisance_logit, pooled, conv_feats = model(imgs, lambda_=lambda_grl)
                    loss = F.binary_cross_entropy_with_logits(main_logit, labels)

                    if flags['use_grl']:
                        nuisance_loss = F.binary_cross_entropy_with_logits(nuisance_logit, density)
                        loss = loss + GRL_ADV_WEIGHT * nuisance_loss

                    if flags['use_distcorr']:
                        dc = distance_correlation(pooled.float(), density.unsqueeze(1).float())
                        loss = loss + w_dc * dc

                    if flags['use_camreg']:
                        cam = model.cam(conv_feats)
                        mask_ds = F.adaptive_avg_pool2d(masks.unsqueeze(1), cam.shape[-2:]).squeeze(1)
                        cam_flat = cam.flatten(1)
                        cam_norm = cam_flat / (cam_flat.sum(1, keepdim=True) + 1e-8)
                        outside_frac = (cam_norm * (1 - mask_ds.flatten(1))).sum(1)
                        loss = loss + w_cam * outside_frac.mean()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, masks, labels, density in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                main_logit, _, _, _ = model(imgs)
                val_preds.append(torch.sigmoid(main_logit).cpu().numpy())
                val_targets.append(labels.numpy())
        val_preds = np.concatenate(val_preds); val_targets = np.concatenate(val_targets)
        val_auroc = roc_auc_score(val_targets, val_preds) if len(np.unique(val_targets)) > 1 else 0.5

        if flags['use_mi']:
            print(f"  [{strategy}] epoch {epoch+1} — w_mi {w_mi:.2f} — "
                  f"raw CLUB MI estimate {total_club_estimate/len(train_loader):.4f} — "
                  f"loss {total_loss/len(train_loader):.4f} — val AUROC {val_auroc:.4f}")

            if log_diagnostics:
                val_feats, val_density = extract_features_and_labels(model, val_loader)
                probe_acc = np.nan
                if len(np.unique(val_density)) >= 2 and len(val_feats) >= 10:
                    n = len(val_feats)
                    rng_ep = np.random.RandomState(epoch)
                    idx = rng_ep.permutation(n)
                    split = int(n * 0.7)
                    tr, te = idx[:split], idx[split:]
                    if len(np.unique(val_density[tr])) >= 2 and len(np.unique(val_density[te])) >= 2:
                        probe = LogisticRegression(max_iter=1000).fit(val_feats[tr], val_density[tr])
                        probe_acc = probe.score(val_feats[te], val_density[te])
                diagnostic_history.append({
                    'epoch': epoch + 1,
                    'raw_club_estimate': total_club_estimate / len(train_loader),
                    'val_auroc': val_auroc,
                    'val_probe_acc': probe_acc,
                })
        else:
            print(f"  [{strategy}] epoch {epoch+1} — grl_lambda {lambda_grl:.2f} — "
                  f"w_dc {w_dc:.2f} — w_cam {w_cam:.2f} — loss {total_loss/len(train_loader):.4f} "
                  f"— val AUROC {val_auroc:.4f}")

        if epoch >= checkpoint_eligible_from:
            if val_auroc > best_val_auroc:
                best_val_auroc, best_state, patience_ct = val_auroc, copy.deepcopy(model.state_dict()), 0
            else:
                patience_ct += 1
                if patience_ct >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    if log_diagnostics:
        return model, diagnostic_history
    return model


@torch.no_grad()
def get_auroc(model, loader):
    """AUROC of the malignancy head on a loader."""
    model.eval()
    preds, targets = [], []
    for imgs, masks, labels, density in loader:
        imgs = imgs.to(device, non_blocking=True)
        main_logit, _, _, _ = model(imgs)
        preds.append(torch.sigmoid(main_logit).cpu().numpy())
        targets.append(labels.numpy())
    preds, targets = np.concatenate(preds), np.concatenate(targets)
    if len(np.unique(targets)) < 2:
        return np.nan
    return roc_auc_score(targets, preds)


def get_mean_sas(model, loader):
    """Mean SAS over a loader."""
    gradcam = GradCAM(model, model.backbone.layer4[-1])
    scores = []
    for imgs, masks, labels, density in loader:
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        cam = gradcam.generate(imgs)
        scores.append(spurious_attribution_score(cam, masks))
    gradcam.remove()
    return np.mean(np.concatenate(scores))


def evaluate_view_subgroups(model, test_df):
    """AUROC and gap separately for CC and MLO views."""
    out = {'auroc_cc': np.nan, 'auroc_mlo': np.nan}
    if 'view' not in test_df.columns:
        return out
    for view_name, key in [('CC', 'auroc_cc'), ('MLO', 'auroc_mlo')]:
        sub_df = test_df[test_df['view'] == view_name].reset_index(drop=True)
        if len(sub_df) < 10 or sub_df['malignant'].nunique() < 2:
            continue
        loader = make_loader(CBISDataset(sub_df, augment=False), BATCH_SIZE, shuffle=False)
        out[key] = get_auroc(model, loader)
    return out


@torch.no_grad()
def extract_features_and_labels(model, loader):
    """Pooled features, malignancy and density labels for a loader."""
    model.eval()
    feats_all, density_all = [], []
    for imgs, masks, labels, density in loader:
        imgs = imgs.to(device, non_blocking=True)
        _, _, pooled, _ = model(imgs)
        feats_all.append(pooled.cpu().numpy())
        density_all.append(density.numpy())
    return np.concatenate(feats_all), np.concatenate(density_all)


def bootstrap_probe_accuracy(model, loader, n_boot=20, seed=0):
    """Linear and MLP density probes on frozen features, averaged over bootstrap splits."""
    feats, density = extract_features_and_labels(model, loader)
    out = {'linear_mean': np.nan, 'linear_std': np.nan, 'mlp_mean': np.nan, 'mlp_std': np.nan}
    if len(np.unique(density)) < 2:
        return out
    rng = np.random.RandomState(seed)
    n = len(feats)
    lin_scores, mlp_scores = [], []
    for b in range(n_boot):
        idx = rng.permutation(n)
        split = int(n * 0.7)
        tr, te = idx[:split], idx[split:]
        if len(np.unique(density[tr])) < 2 or len(np.unique(density[te])) < 2:
            continue
        lin = LogisticRegression(max_iter=1000).fit(feats[tr], density[tr])
        lin_scores.append(lin.score(feats[te], density[te]))

        scaler = StandardScaler().fit(feats[tr])
        feats_tr_scaled = scaler.transform(feats[tr])
        feats_te_scaled = scaler.transform(feats[te])
        mlp = MLPClassifier(
            hidden_layer_sizes=(64,), max_iter=2000, random_state=b,
            early_stopping=True, n_iter_no_change=15, validation_fraction=0.15,
        ).fit(feats_tr_scaled, density[tr])
        mlp_scores.append(mlp.score(feats_te_scaled, density[te]))
    if lin_scores:
        out['linear_mean'], out['linear_std'] = np.mean(lin_scores), np.std(lin_scores)
    if mlp_scores:
        out['mlp_mean'], out['mlp_std'] = np.mean(mlp_scores), np.std(mlp_scores)
    return out


def bootstrap_ci(diffs, n_boot=5000, seed=0):
    """Percentile bootstrap 95% CI of the mean paired difference."""
    rng = np.random.RandomState(seed)
    diffs = np.asarray(diffs)
    boots = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n_boot)]
    return np.mean(diffs), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def required_n_for_power(diffs, alpha=0.05, power_target=0.80):
    """Number of seeds needed for 80% power, from the observed paired differences."""
    diffs = np.asarray(diffs)
    if diffs.std(ddof=1) == 0 or len(diffs) < 2:
        return np.nan, np.nan
    d = diffs.mean() / diffs.std(ddof=1)
    if d == 0:
        return np.nan, 0.0
    z_alpha = _norm.ppf(1 - alpha / 2)
    z_power = _norm.ppf(power_target)
    n_required = ((z_alpha + z_power) / d) ** 2
    n_current = len(diffs)
    z_achieved = abs(d) * np.sqrt(n_current) - z_alpha
    achieved_power = _norm.cdf(z_achieved)
    return n_required, achieved_power


def power_analysis_table(results_df, ablations, metrics=('worst_group_gap', 'sas', 'probe_linear_mean', 'auroc_all')):
    """Power analysis for every strategy and metric."""
    print(f"\n{'='*60}\nPOWER ANALYSIS (required N for 80% power at alpha=0.05)")
    print("=" * 60)
    baseline_df = results_df[results_df.strategy == 'baseline'].set_index('seed')
    rows = []
    for strategy in ablations:
        if strategy == 'baseline':
            continue
        arm_df = results_df[results_df.strategy == strategy].set_index('seed')
        common_seeds = baseline_df.index.intersection(arm_df.index)
        for metric in metrics:
            b = baseline_df.loc[common_seeds, metric].values
            a = arm_df.loc[common_seeds, metric].values
            valid = ~(np.isnan(a) | np.isnan(b))
            if valid.sum() < 2:
                continue
            diffs = a[valid] - b[valid]
            n_req, achieved_power = required_n_for_power(diffs)
            current_n = valid.sum()
            adequate = "YES" if (not np.isnan(n_req) and n_req <= current_n) else "no -- need more seeds"
            print(f"  {strategy:>10} | {metric:<18} | current n={current_n} | "
                  f"achieved power={achieved_power:.2f} | N needed for 80% power={n_req:.1f} | adequate: {adequate}")
            rows.append({'strategy': strategy, 'metric': metric, 'current_n': current_n,
                          'achieved_power': achieved_power, 'n_required_80pct': n_req})
    return pd.DataFrame(rows)


def plot_forest(results_df, ablations, out_path):
    """Forest plot of paired differences against the baseline."""
    metrics = [('worst_group_gap', 'Worst-group gap'), ('sas', 'SAS'),
               ('probe_linear_mean', 'Density probe (linear)'), ('auroc_all', 'AUROC (all)')]
    baseline_df = results_df[results_df.strategy == 'baseline'].set_index('seed')
    arms = [s for s in ablations if s != 'baseline']

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 0.6 * len(arms) + 1.5), sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        ys = np.arange(len(arms))
        for y, strategy in zip(ys, arms):
            arm_df = results_df[results_df.strategy == strategy].set_index('seed')
            common_seeds = baseline_df.index.intersection(arm_df.index)
            b = baseline_df.loc[common_seeds, metric].values
            a = arm_df.loc[common_seeds, metric].values
            valid = ~(np.isnan(a) | np.isnan(b))
            if valid.sum() < 2:
                ax.plot([], [])
                continue
            diffs = a[valid] - b[valid]
            mean_diff, lo, hi = bootstrap_ci(diffs)
            color = 'tab:red' if (lo > 0 or hi < 0) else 'tab:gray'
            ax.errorbar(mean_diff, y, xerr=[[mean_diff - lo], [hi - mean_diff]],
                        fmt='o', color=color, capsize=4)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.set_yticks(ys)
        ax.set_yticklabels(arms)
        ax.set_title(f"{title}\n(vs baseline)")
        ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Forest plot saved to: {out_path}")


def generate_qualitative_figure(qualitative_models, example_df, out_path, n_examples=N_QUALITATIVE_EXAMPLES):
    """Grad-CAM grid for selected conflicting-group test cases."""
    if not qualitative_models or len(example_df) == 0:
        print("Skipping qualitative figure: no models or no conflicting-group examples available.")
        return
    n_examples = min(n_examples, len(example_df))
    sample_df = example_df.sample(n=n_examples, random_state=0).reset_index(drop=True)
    dataset = CBISDataset(sample_df, augment=False)

    strategies = list(qualitative_models.keys())
    fig, axes = plt.subplots(n_examples, len(strategies) + 1,
                              figsize=(2.6 * (len(strategies) + 1), 2.6 * n_examples))
    if n_examples == 1:
        axes = axes[np.newaxis, :]

    for row in range(n_examples):
        img_t, mask_t, label, density = dataset[row]
        img_disp = img_t.permute(1, 2, 0).numpy()
        img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)

        axes[row, 0].imshow(img_disp)
        axes[row, 0].contour(mask_t.numpy(), colors='lime', linewidths=1)
        axes[row, 0].set_ylabel(f"malig={label}\ndense={density}", fontsize=8)
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        if row == 0:
            axes[row, 0].set_title("image + mask", fontsize=9)

        for col, strategy in enumerate(strategies, start=1):
            model = qualitative_models[strategy]
            gradcam = GradCAM(model, model.backbone.layer4[-1])
            imgs = img_t.unsqueeze(0).to(device)
            cam = gradcam.generate(imgs)[0].detach().cpu().numpy()
            gradcam.remove()

            axes[row, col].imshow(img_disp)
            axes[row, col].imshow(cam, cmap='jet', alpha=0.45)
            axes[row, col].contour(mask_t.numpy(), colors='lime', linewidths=1)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(strategy, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Qualitative GradCAM comparison saved to: {out_path}")


def run_mi_diagnostic(meta_raw, patient_col):
    """Diagnostic run for mi_min: CLUB estimate against a fresh density probe per epoch."""
    print(f"\n{'='*60}\nMI_MIN DIAGNOSTIC: raw CLUB estimate vs. fresh validation probe")
    print(f"seeds={MI_DIAGNOSTIC_SEEDS} (small and separate from the main SEED_LIST on purpose)")
    print("=" * 60)

    all_history = []
    for seed in MI_DIAGNOSTIC_SEEDS:
        splitter1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        trainval_idx, test_idx = next(splitter1.split(meta_raw, groups=meta_raw[patient_col]))
        trainval_df = meta_raw.iloc[trainval_idx].reset_index(drop=True)
        splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, val_idx = next(splitter2.split(trainval_df, groups=trainval_df[patient_col]))
        train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
        val_df = trainval_df.iloc[val_idx].reset_index(drop=True)
        train_df_used = bias_split_by_correlation(train_df, seed, BIAS_STRENGTH) if BIAS_TRAINING else train_df

        print(f"\n  --- seed {seed} ---")
        _, history = train_model('mi_min', train_df_used, val_df, seed, log_diagnostics=True)
        for row in history:
            row['seed'] = seed
        all_history.extend(history)

    hist_df = pd.DataFrame(all_history)
    hist_df.to_csv(os.path.join(OUT_DIR, 'mi_min_club_diagnostic.csv'), index=False)

    print(f"\n{'='*60}\nVERDICT (first-half vs second-half mean, per seed)")
    print("=" * 60)
    for seed in MI_DIAGNOSTIC_SEEDS:
        seed_hist = hist_df[hist_df['seed'] == seed].sort_values('epoch')
        if len(seed_hist) < 4:
            print(f"  seed {seed}: too few epochs logged -- skipped")
            continue
        mid = len(seed_hist) // 2
        club_first, club_second = seed_hist['raw_club_estimate'].iloc[:mid].mean(), seed_hist['raw_club_estimate'].iloc[mid:].mean()
        probe_first, probe_second = seed_hist['val_probe_acc'].iloc[:mid].mean(), seed_hist['val_probe_acc'].iloc[mid:].mean()
        club_trend = "DOWN" if club_second < club_first else "UP"
        probe_trend = "DOWN" if probe_second < probe_first else "UP"
        agree = (club_trend == "DOWN" and probe_trend == "DOWN")
        print(f"  seed {seed}: CLUB estimate {club_first:.4f} -> {club_second:.4f} ({club_trend})   "
              f"fresh probe acc {probe_first:.4f} -> {probe_second:.4f} ({probe_trend})   "
              f"{'CONSISTENT (both down)' if agree else 'DIVERGENT -- estimator and probe disagree'}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for seed in MI_DIAGNOSTIC_SEEDS:
        seed_hist = hist_df[hist_df['seed'] == seed].sort_values('epoch')
        axes[0].plot(seed_hist['epoch'], seed_hist['raw_club_estimate'], marker='o', label=f'seed {seed}')
        axes[1].plot(seed_hist['epoch'], seed_hist['val_probe_acc'], marker='s', label=f'seed {seed}')
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("raw CLUB MI estimate (training-time)")
    axes[0].set_title("Training-time CLUB estimate"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("fresh linear probe accuracy (val)")
    axes[1].set_title("Independent validation probe (ground truth check)"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'mi_min_club_diagnostic.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\nSaved: mi_min_club_diagnostic.csv, mi_min_club_diagnostic.png")
    return hist_df


def run_sensitivity_sweep(meta_raw, patient_col):
    """Distance-correlation weight sweep (3 to 7) over all seeds."""
    print(f"\n{'='*60}\nSENSITIVITY SWEEP: distcorr weight over {SENSITIVITY_WEIGHTS}, "
          f"seeds={SENSITIVITY_SEEDS}\n{'='*60}")
    global MAX_WEIGHT_DISTCORR
    original_weight = MAX_WEIGHT_DISTCORR
    rows = []
    try:
        for weight in SENSITIVITY_WEIGHTS:
            MAX_WEIGHT_DISTCORR = weight
            for seed in SENSITIVITY_SEEDS:
                splitter1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
                trainval_idx, test_idx = next(splitter1.split(meta_raw, groups=meta_raw[patient_col]))
                trainval_df = meta_raw.iloc[trainval_idx].reset_index(drop=True)
                test_df = meta_raw.iloc[test_idx].reset_index(drop=True)
                splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                train_idx, val_idx = next(splitter2.split(trainval_df, groups=trainval_df[patient_col]))
                train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
                val_df = trainval_df.iloc[val_idx].reset_index(drop=True)
                train_df_used = bias_split_by_correlation(train_df, seed, BIAS_STRENGTH) if BIAS_TRAINING else train_df

                aligned_mask, conflict_mask = group_masks(test_df)
                aligned_df = test_df[aligned_mask].reset_index(drop=True)
                conflict_df = test_df[conflict_mask].reset_index(drop=True)
                aligned_loader = make_loader(CBISDataset(aligned_df, augment=False), BATCH_SIZE, shuffle=False)
                conflict_loader = make_loader(CBISDataset(conflict_df, augment=False), BATCH_SIZE, shuffle=False)
                test_loader = make_loader(CBISDataset(test_df, augment=False), BATCH_SIZE, shuffle=False)

                print(f"  weight={weight}, seed={seed} ...")
                model = train_model('distcorr', train_df_used, val_df, seed)
                auroc = get_auroc(model, test_loader)
                auroc_aligned = get_auroc(model, aligned_loader)
                auroc_conflict = get_auroc(model, conflict_loader)
                gap = auroc_aligned - auroc_conflict
                rows.append({'weight': weight, 'seed': seed, 'auroc_all': auroc, 'worst_group_gap': gap})
                print(f"    -> AUROC(all)={auroc:.4f}  gap={gap:.4f}")
    finally:
        MAX_WEIGHT_DISTCORR = original_weight

    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(os.path.join(OUT_DIR, 'distcorr_sensitivity_sweep.csv'), index=False)

    summary = sweep_df.groupby('weight').agg(
        gap_mean=('worst_group_gap', 'mean'), gap_std=('worst_group_gap', 'std'),
        auroc_mean=('auroc_all', 'mean'), auroc_std=('auroc_all', 'std'),
    ).reset_index()
    print("\nSensitivity sweep summary:")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].errorbar(summary['weight'], summary['gap_mean'], yerr=summary['gap_std'], marker='o', capsize=4)
    axes[0].set_xlabel("distcorr weight (MAX_WEIGHT_DISTCORR)"); axes[0].set_ylabel("Worst-group gap")
    axes[0].set_title("Sensitivity: worst-group gap"); axes[0].grid(alpha=0.3)
    axes[1].errorbar(summary['weight'], summary['auroc_mean'], yerr=summary['auroc_std'], marker='s', color='tab:green', capsize=4)
    axes[1].set_xlabel("distcorr weight (MAX_WEIGHT_DISTCORR)"); axes[1].set_ylabel("AUROC (all)")
    axes[1].set_title("Sensitivity: AUROC"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'distcorr_sensitivity_sweep.png'), dpi=150)
    plt.close(fig)
    print("Sensitivity sweep plot saved.")
    return sweep_df


def load_data():
    """Build metadata, print the natural density-malignancy correlation and load the image cache."""
    print("Device:", device)
    meta_raw, patient_col = build_metadata()

    corr, p = pointbiserialr(meta_raw['malignant'], meta_raw['density'])
    print(f"\n{'='*60}\nDIAGNOSTIC 1: Point-biserial correlation (malignant vs density)")
    print(f"r={corr:.3f}, p={p:.4f}")
    print("=> Natural confound is weak; training-set correlation will be amplified "
          f"in a controlled way (BIAS_TRAINING={BIAS_TRAINING}, strength={BIAS_STRENGTH}) "
          "so deconfounding methods have a real, measurable effect to remove.")
    print("=" * 60)

    build_or_load_cache(meta_raw['filepath'].unique().tolist())
    return meta_raw, patient_col


def main():
    """Full sweep over strategies and seeds, paired statistics, power analysis and figures."""
    meta_raw, patient_col = load_data()

    all_results = []
    qualitative_models = {}
    qualitative_conflict_df = None
    for seed in SEED_LIST:
        print(f"\n{'#'*60}\nSEED {seed}\n{'#'*60}")
        splitter1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        trainval_idx, test_idx = next(splitter1.split(meta_raw, groups=meta_raw[patient_col]))
        trainval_df = meta_raw.iloc[trainval_idx].reset_index(drop=True)
        test_df = meta_raw.iloc[test_idx].reset_index(drop=True)
        splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, val_idx = next(splitter2.split(trainval_df, groups=trainval_df[patient_col]))
        train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
        val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

        if BIAS_TRAINING:
            train_df_used = bias_split_by_correlation(train_df, seed, BIAS_STRENGTH)
            r_biased, _ = pointbiserialr(train_df_used['malignant'], train_df_used['density_binary'])
            print(f"Train {len(train_df_used)} (biased from {len(train_df)}, "
                  f"achieved point-biserial r={r_biased:.3f}) / Val {len(val_df)} / Test {len(test_df)}")
        else:
            train_df_used = train_df
            print(f"Train {len(train_df_used)} / Val {len(val_df)} / Test {len(test_df)}")

        aligned_mask, conflict_mask = group_masks(test_df)
        aligned_df = test_df[aligned_mask].reset_index(drop=True)
        conflict_df = test_df[conflict_mask].reset_index(drop=True)
        print(f"  Test worst-group split: aligned={len(aligned_df)}  conflicting={len(conflict_df)}")
        if seed == QUALITATIVE_SEED:
            qualitative_conflict_df = conflict_df

        test_loader = make_loader(CBISDataset(test_df, augment=False), BATCH_SIZE, shuffle=False)
        aligned_loader = make_loader(CBISDataset(aligned_df, augment=False), BATCH_SIZE, shuffle=False) if len(aligned_df) else None
        conflict_loader = make_loader(CBISDataset(conflict_df, augment=False), BATCH_SIZE, shuffle=False) if len(conflict_df) else None

        for strategy in ABLATIONS:
            model = train_model(strategy, train_df_used, val_df, seed)

            auroc = get_auroc(model, test_loader)
            auroc_aligned = get_auroc(model, aligned_loader) if aligned_loader else np.nan
            auroc_conflict = get_auroc(model, conflict_loader) if conflict_loader else np.nan
            worst_group_gap = auroc_aligned - auroc_conflict if not (np.isnan(auroc_aligned) or np.isnan(auroc_conflict)) else np.nan
            sas = get_mean_sas(model, test_loader)
            probe = bootstrap_probe_accuracy(model, test_loader, seed=seed)
            view_scores = evaluate_view_subgroups(model, test_df)

            print(f"  [{strategy}] AUROC(all)={auroc:.4f}  AUROC(aligned)={auroc_aligned:.4f}  "
                  f"AUROC(conflict)={auroc_conflict:.4f}  gap={worst_group_gap:.4f}  SAS={sas:.4f}  "
                  f"probe(linear)={probe['linear_mean']:.4f}±{probe['linear_std']:.4f}  "
                  f"probe(mlp)={probe['mlp_mean']:.4f}±{probe['mlp_std']:.4f}  "
                  f"AUROC(CC)={view_scores['auroc_cc']:.4f}  AUROC(MLO)={view_scores['auroc_mlo']:.4f}")

            all_results.append({
                'seed': seed, 'strategy': strategy,
                'auroc_all': auroc, 'auroc_aligned': auroc_aligned, 'auroc_conflict': auroc_conflict,
                'worst_group_gap': worst_group_gap, 'sas': sas,
                'probe_linear_mean': probe['linear_mean'], 'probe_linear_std': probe['linear_std'],
                'probe_mlp_mean': probe['mlp_mean'], 'probe_mlp_std': probe['mlp_std'],
                'auroc_cc': view_scores['auroc_cc'], 'auroc_mlo': view_scores['auroc_mlo'],
            })

            if seed == QUALITATIVE_SEED and strategy in ('baseline', 'distcorr', 'camreg', 'full'):
                qualitative_models[strategy] = copy.deepcopy(model)

    results_df = pd.DataFrame(all_results)
    print("\n\nFULL RESULTS ACROSS SEEDS AND STRATEGIES")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(OUT_DIR, 'dccam_cbis_results.csv'), index=False)

    print(f"\n{'='*60}\nPAIRED COMPARISONS vs. baseline (same seeds)")
    print("Note: with n<=5 seeds, Wilcoxon cannot reach p<0.05 regardless of true")
    print("effect size (min achievable two-sided p = 1/16 = 0.0625 at n=5).")
    print("The bootstrap CI on the mean paired difference is the primary evidence.")
    print("=" * 60)
    baseline_df = results_df[results_df.strategy == 'baseline'].set_index('seed')
    for strategy in ABLATIONS:
        if strategy == 'baseline':
            continue
        arm_df = results_df[results_df.strategy == strategy].set_index('seed')
        common_seeds = baseline_df.index.intersection(arm_df.index)
        for metric, better in [('worst_group_gap', 'lower'), ('sas', 'lower'),
                                ('probe_linear_mean', 'lower'), ('auroc_all', 'higher (or unchanged)')]:
            b = baseline_df.loc[common_seeds, metric].values
            a = arm_df.loc[common_seeds, metric].values
            valid = ~(np.isnan(a) | np.isnan(b))
            if valid.sum() < 3:
                continue
            diffs = a[valid] - b[valid]
            mean_diff, lo, hi = bootstrap_ci(diffs)
            try:
                _, p_val = wilcoxon(a[valid], b[valid])
            except ValueError:
                p_val = np.nan
            print(f"  {strategy:>10} vs baseline | {metric:<18} | want {better:<20} | "
                  f"mean_diff={mean_diff:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  wilcoxon p={p_val:.4f}")

    metrics_to_plot = [('auroc_all', 'Main task AUROC (all test)'),
                        ('worst_group_gap', 'Worst-group gap (aligned - conflicting)'),
                        ('sas', 'Spurious Attribution Score'),
                        ('probe_linear_mean', 'Density probe accuracy (linear)')]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 4))
    for ax, (col, title) in zip(axes, metrics_to_plot):
        means = [results_df[results_df.strategy == s][col].mean() for s in ABLATIONS]
        stds = [results_df[results_df.strategy == s][col].std() for s in ABLATIONS]
        ax.bar(ABLATIONS, means, yerr=stds, capsize=4)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'dccam_cbis_comparison.png'), dpi=150)
    plt.show()

    if results_df['auroc_cc'].notna().any():
        print(f"\n{'='*60}\nVIEW-SUBGROUP CONSISTENCY CHECK (CC vs MLO)")
        print("Not a full domain-shift validation -- both views were present in")
        print("training. A real external-dataset check remains necessary future work.")
        print("=" * 60)
        view_summary = results_df.groupby('strategy')[['auroc_cc', 'auroc_mlo']].mean()
        view_summary['cc_mlo_gap'] = (view_summary['auroc_cc'] - view_summary['auroc_mlo']).abs()
        print(view_summary.reindex(ABLATIONS).to_string())
    else:
        print("\nView column not found/insufficient in this data -- skipped the CC/MLO check.")

    power_df = power_analysis_table(results_df, ABLATIONS)
    power_df.to_csv(os.path.join(OUT_DIR, 'power_analysis.csv'), index=False)

    plot_forest(results_df, ABLATIONS, os.path.join(OUT_DIR, 'dccam_forest_plot.png'))

    if qualitative_conflict_df is not None:
        generate_qualitative_figure(
            qualitative_models, qualitative_conflict_df,
            os.path.join(OUT_DIR, 'dccam_qualitative_gradcam.png'),
        )

    print(f"\nSaved results to: {OUT_DIR}")

    RUN_SENSITIVITY_SWEEP = True
    if RUN_SENSITIVITY_SWEEP:
        run_sensitivity_sweep(meta_raw, patient_col)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'mi_diagnostic':
        meta_raw, patient_col = load_data()
        run_mi_diagnostic(meta_raw, patient_col)
    else:
        main()
