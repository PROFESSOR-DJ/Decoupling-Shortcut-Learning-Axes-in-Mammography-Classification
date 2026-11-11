"""Shared helpers for the extra experiments: splits, shortcut injection by condition,
train/evaluate one run, saved probabilities, result store and paired statistics.
"""

import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import io
import re
import sys
import gc
import json
import time
import platform
import contextlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from scipy.stats import pointbiserialr, wilcoxon

import core_pipeline
from paths import OUTPUT_DIR


def apply_determinism():
    """Deterministic cuDNN / cuBLAS settings (call after importing the pipeline)."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


apply_determinism()

ORIG_DATASET = core_pipeline.CBISDataset
ORIG_NET = core_pipeline.DCCAMNet

HERE = OUTPUT_DIR
OUT_ROOT = HERE

METRICS = ('auroc_all', 'worst_group_gap', 'sas', 'probe_linear_mean')
METRIC_LABELS = {
    'auroc_all': 'AUROC (all)', 'worst_group_gap': 'Worst-group gap',
    'sas': 'SAS', 'probe_linear_mean': 'Density probe acc.',
}


def make_out_dir(study, smoke=False):
    """Create the output folders for one study."""
    name = study + ('_SMOKE' if smoke else '')
    root = os.path.join(OUT_ROOT, name)
    d = {'root': root}
    for sub in ('probs', 'ckpt', 'logs'):
        d[sub] = os.path.join(root, sub)
        os.makedirs(d[sub], exist_ok=True)
    return d


def determinism_snapshot():
    return {
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def write_manifest(out, script, args, extra=None):
    """Save seeds, arguments and library versions next to the results."""
    manifest = {
        'script': script,
        'run_utc': datetime.now(timezone.utc).isoformat(),
        'args': vars(args) if hasattr(args, '__dict__') else args,
        'pipeline_constants': {
            k: getattr(core_pipeline, k) for k in (
                'IMG_SIZE', 'BATCH_SIZE', 'MAX_EPOCHS', 'PATIENCE', 'LR', 'MAX_LAMBDA_GRL',
                'GRL_ADV_WEIGHT', 'GRL_REINIT_EVERY', 'MAX_WEIGHT_DISTCORR', 'MAX_WEIGHT_CAMREG',
                'LAMBDA_RAMP_EPOCHS', 'BIAS_TRAINING', 'BIAS_STRENGTH')},
        'determinism': determinism_snapshot(),
    }
    if extra:
        manifest.update(extra)
    path = os.path.join(out['root'], f"run_manifest_{script.replace('.py', '')}.json")
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    return path


def add_common_args(p, default_seeds=None):
    """Command-line flags shared by all experiment scripts."""
    p.add_argument('--seeds', type=int, default=default_seeds,
                   help="use the first N seeds of core_pipeline.SEED_LIST (42..54). Default: %s"
                        % (default_seeds if default_seeds else 'all 13'))
    p.add_argument('--seed-list', type=int, nargs='+', default=None,
                   help="explicit seeds (overrides --seeds), e.g. --seed-list 42 43 44")
    p.add_argument('--no-probe', action='store_true',
                   help="skip the bootstrapped density probes (saves a few minutes per run)")
    p.add_argument('--smoke', action='store_true',
                   help="tiny pipeline test: 1 seed, 2 epochs, outputs go to <study>_SMOKE/")
    p.add_argument('--dry-run', action='store_true',
                   help="print the planned runs and exit without training")
    return p


def resolve_seeds(args):
    if args.smoke:
        return core_pipeline.SEED_LIST[:1]
    if args.seed_list:
        return list(args.seed_list)
    if args.seeds:
        return core_pipeline.SEED_LIST[:args.seeds]
    return list(core_pipeline.SEED_LIST)


def apply_smoke(args):
    if args.smoke:
        core_pipeline.MAX_EPOCHS = 2
        core_pipeline.LAMBDA_RAMP_EPOCHS = 1
        core_pipeline.PATIENCE = 1
        args.no_probe = True
        print("[smoke] MAX_EPOCHS=2, LAMBDA_RAMP_EPOCHS=1, 1 seed, probes off -- pipeline test only; "
              "numbers are meaningless.")


def load_cbis_meta(density_split='median', need_images=True):
    """CBIS-DDSM metadata (median or clinical density split), optionally with images."""
    meta, pcol = core_pipeline.load_data() if need_images else core_pipeline.build_metadata()
    if density_split == 'clinical':
        meta = meta.copy()
        meta['density_binary'] = (meta['density'] >= 3).astype(int)
    elif density_split != 'median':
        raise ValueError(f"unknown density_split {density_split!r}")
    return meta, pcol


def make_split(meta, pcol, seed):
    """Patient-grouped 70/30 train+val/test split, then 80/20 train/val, as in the main run."""
    s1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    trainval_idx, test_idx = next(s1.split(meta, groups=meta[pcol]))
    trainval_df = meta.iloc[trainval_idx].reset_index(drop=True)
    test_df = meta.iloc[test_idx].reset_index(drop=True)
    s2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(s2.split(trainval_df, groups=trainval_df[pcol]))
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def phi(df):
    """Point-biserial correlation between malignancy and binary density."""
    if df['malignant'].nunique() < 2 or df['density_binary'].nunique() < 2:
        return float('nan')
    return float(pointbiserialr(df['malignant'], df['density_binary'])[0])


def _calibrate_bias(train_df, seed, target_r, iters=16):
    """Find the bias strength that gives a target training correlation."""
    lo, hi = 0.5, 0.97
    best = (1e9, None, None)
    for _ in range(iters):
        s = (lo + hi) / 2
        b = core_pipeline.bias_split_by_correlation(train_df, seed, s)
        r = phi(b)
        if not np.isnan(r) and abs(r - target_r) < best[0]:
            best = (abs(r - target_r), s, b)
        if np.isnan(r) or r < target_r:
            lo = s
        else:
            hi = s
    return best[1], best[2]


def build_training_set(train_df, seed, condition):
    """Training fold for a condition: 'natural', 'main' (r ~ 0.6) or 'r0.3'."""
    if condition == 'natural':
        used, s = train_df.copy(), None
    elif condition == 'main':
        s = core_pipeline.BIAS_STRENGTH
        used = core_pipeline.bias_split_by_correlation(train_df, seed, s)
    elif condition.startswith('r'):
        s, used = _calibrate_bias(train_df, seed, float(condition[1:]))
    else:
        raise ValueError(f"unknown condition {condition!r}")
    info = {
        'condition': condition, 'bias_strength': s,
        'n_train_before': len(train_df), 'n_train': len(used),
        'n_train_malignant': int(used['malignant'].sum()),
        'r_train': phi(used), 'r_train_before': phi(train_df),
    }
    return used, info


def group_counts(test_df):
    """Sizes of the aligned and conflicting test groups."""
    a_mask, c_mask = core_pipeline.group_masks(test_df)
    conflict = test_df[c_mask]
    return {
        'n_test': len(test_df), 'n_aligned': int(a_mask.sum()), 'n_conflict': int(c_mask.sum()),
        'n_conflict_malignant': int(conflict['malignant'].sum()),
        'n_conflict_benign': int((1 - conflict['malignant']).sum()),
    }, a_mask, c_mask


@contextlib.contextmanager
def override(**kw):
    """Temporarily change pipeline constants (loss weights, GRL settings)."""
    old = {}
    for k, v in kw.items():
        if not hasattr(core_pipeline, k):
            raise AttributeError(f"core_pipeline has no module-level constant {k!r}")
        old[k] = getattr(core_pipeline, k)
        setattr(core_pipeline, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(core_pipeline, k, v)


class AdvMLP(nn.Sequential):
    """Two-layer MLP adversary for the GRL variants."""
    def __init__(self, d):
        super().__init__(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, 2 * d), nn.ReLU(), nn.Linear(2 * d, 1))


_WEIGHTS = {'resnet18': 'ResNet18_Weights', 'resnet34': 'ResNet34_Weights', 'resnet50': 'ResNet50_Weights'}


def _make_net_class(backbone='resnet18', adv='linear'):
    class Net(ORIG_NET):
        def __init__(self, use_grl=False):
            nn.Module.__init__(self)
            ctor = getattr(models, backbone)
            weights = getattr(models, _WEIGHTS[backbone]).IMAGENET1K_V1
            bb = ctor(weights=weights)
            self.feat_dim = bb.fc.in_features
            bb.fc = nn.Identity()
            bb.avgpool = nn.Identity()
            self.backbone = bb
            self.main_head = nn.Linear(self.feat_dim, 1)
            self.nuisance_head = nn.Linear(self.feat_dim, 1) if adv == 'linear' else AdvMLP(self.feat_dim)
            self.use_grl = use_grl
    return Net


@contextlib.contextmanager
def use_net(backbone='resnet18', adv='linear'):
    """Temporarily swap the backbone or adversary used by the pipeline."""
    if backbone == 'resnet18' and adv == 'linear':
        yield
        return
    core_pipeline.DCCAMNet = _make_net_class(backbone, adv)
    try:
        yield
    finally:
        core_pipeline.DCCAMNet = ORIG_NET


@contextlib.contextmanager
def ordinal_density(on=True):
    """Use the ordinal BI-RADS value as the density signal during training only."""
    if not on:
        yield
        return
    class OrdinalDS(ORIG_DATASET):
        def __getitem__(self, idx):
            img_t, mask_t, label, _ = super().__getitem__(idx)
            d = float(self.df.iloc[idx]['density'])
            return img_t, mask_t, label, torch.tensor((d - 1.0) / 3.0, dtype=torch.float32)
    core_pipeline.CBISDataset = OrdinalDS
    try:
        yield
    finally:
        core_pipeline.CBISDataset = ORIG_DATASET


def add_grl_tuned_args(p):
    g = p.add_argument_group("tuned GRL arm ('grl_tuned'; use the setting chosen by weight_and_grl_sweeps.py --study grl)")
    g.add_argument('--grl-lambda', type=float, default=1.0)
    g.add_argument('--grl-weight', type=float, default=1.0)
    g.add_argument('--grl-reinit', type=int, default=0)
    g.add_argument('--grl-adv', choices=['linear', 'mlp'], default='linear')


def resolve_arm(arm, args=None):
    """Map an arm name to (strategy, overrides, network settings)."""
    if arm == 'grl_tuned':
        return 'grl_only', dict(MAX_LAMBDA_GRL=args.grl_lambda, GRL_ADV_WEIGHT=args.grl_weight,
                                GRL_REINIT_EVERY=args.grl_reinit), dict(adv=args.grl_adv)
    return arm, {}, {}


class ResultStore:
    """Results CSV that is appended after every run, so a stopped run can resume."""
    def __init__(self, path, key_cols=('tag', 'strategy', 'seed')):
        self.path, self.key_cols = path, key_cols
        self.rows, self.done = [], set()
        if os.path.exists(path):
            df = pd.read_csv(path)
            self.rows = df.to_dict('records')
            self.done = {tuple(r[k] for k in key_cols) for r in self.rows}

    def has(self, *key):
        return tuple(key) in self.done

    def add(self, row):
        self.rows.append(row)
        self.done.add(tuple(row[k] for k in self.key_cols))
        pd.DataFrame([row]).to_csv(self.path, mode='a', index=False, header=not os.path.exists(self.path))

    def df(self):
        return pd.DataFrame(self.rows)


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except UnicodeEncodeError:
                st.write(s.encode('ascii', 'replace').decode())
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


_EPOCH_RE = re.compile(r'epoch (\d+) .*?loss (\S+) .*?val AUROC (\S+)')
_LAMBDA_RE = re.compile(r'grl_lambda ([\d.]+) .*?w_dc ([\d.]+) .*?w_cam ([\d.]+)')


def parse_history(log_text):
    rows = []
    for line in log_text.splitlines():
        m = _EPOCH_RE.search(line)
        if not m:
            continue
        rec = {'epoch': int(m.group(1)), 'loss': float(m.group(2)), 'val_auroc': float(m.group(3)),
               'grl_lambda': np.nan, 'w_dc': np.nan, 'w_cam': np.nan}
        m2 = _LAMBDA_RE.search(line)
        if m2:
            rec['grl_lambda'], rec['w_dc'], rec['w_cam'] = map(float, m2.groups())
        rows.append(rec)
    return rows


@torch.no_grad()
def predict(model, loader):
    """Predicted probabilities and labels for a loader."""
    model.eval()
    probs, logits = [], []
    for imgs, masks, labels, density in loader:
        imgs = imgs.to(core_pipeline.device, non_blocking=True)
        logit, _, _, _ = model(imgs)
        probs.append(torch.sigmoid(logit).float().cpu().numpy())
        logits.append(logit.float().cpu().numpy())
    return np.concatenate(probs), np.concatenate(logits)


def _auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float('nan')


def sas_with_mask_variants(model, loader, ks=(3, 6)):
    """Per-image SAS with the original mask and with dilated / eroded masks."""
    gradcam = core_pipeline.GradCAM(model, model.backbone.layer4[-1])
    out = {'orig': []}
    for k in ks:
        out[f'dil{k}'], out[f'ero{k}'] = [], []
    try:
        for imgs, masks, labels, density in loader:
            imgs = imgs.to(core_pipeline.device, non_blocking=True)
            masks = masks.to(core_pipeline.device, non_blocking=True)
            cam = gradcam.generate(imgs)
            out['orig'].append(core_pipeline.spurious_attribution_score(cam, masks))
            m4 = masks.unsqueeze(1)
            for k in ks:
                dil = F.max_pool2d(m4, 2 * k + 1, stride=1, padding=k).squeeze(1)
                ero = (-F.max_pool2d(-m4, 2 * k + 1, stride=1, padding=k)).squeeze(1)
                out[f'dil{k}'].append(core_pipeline.spurious_attribution_score(cam, dil))
                out[f'ero{k}'].append(core_pipeline.spurious_attribution_score(cam, ero))
    finally:
        gradcam.remove()
    return {k: np.concatenate(v) for k, v in out.items()}


def _safe(s):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(s))


def train_and_evaluate(label, train_strategy, train_df_used, val_df, test_df, seed, out, *,
                       tag, ref_tag=None, overrides=None, net=None, ordinal=False,
                       save_ckpt=False, mask_variants=False, run_probe=True,
                       info=None, extra=None):
    """Train one run and save metrics, probabilities, history and checkpoint."""
    overrides, net = overrides or {}, net or {}
    run_id = f"{_safe(tag)}__{_safe(label)}__seed{seed}"
    t0 = time.time()

    buf = io.StringIO()
    with override(**overrides), use_net(**net), ordinal_density(ordinal):
        with contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
            model = core_pipeline.train_model(train_strategy, train_df_used, val_df, seed)
    train_seconds = time.time() - t0
    log_text = buf.getvalue()
    with open(os.path.join(out['logs'], run_id + '.log'), 'w', encoding='utf-8') as f:
        f.write(log_text)
    hist = parse_history(log_text)
    if hist:
        h = pd.DataFrame(hist)
        h.insert(0, 'seed', seed); h.insert(0, 'strategy', label); h.insert(0, 'tag', tag)
        hp = os.path.join(out['root'], 'epoch_history.csv')
        h.to_csv(hp, mode='a', index=False, header=not os.path.exists(hp))

    gc_, a_mask, c_mask = group_counts(test_df)
    mk = lambda df: core_pipeline.make_loader(ORIG_DATASET(df, augment=False), core_pipeline.BATCH_SIZE, shuffle=False)
    test_loader, val_loader = mk(test_df), mk(val_df)

    p_test, z_test = predict(model, test_loader)
    y_test = test_df['malignant'].to_numpy()
    auroc_all = _auc(y_test, p_test)
    auroc_a = _auc(y_test[a_mask], p_test[a_mask]) if a_mask.any() else float('nan')
    auroc_c = _auc(y_test[c_mask], p_test[c_mask]) if c_mask.any() else float('nan')
    gap = auroc_a - auroc_c if not (np.isnan(auroc_a) or np.isnan(auroc_c)) else float('nan')

    p_val, _ = predict(model, val_loader)
    y_val = val_df['malignant'].to_numpy()
    va, vc = core_pipeline.group_masks(val_df)
    val_auroc = _auc(y_val, p_val)
    v_a = _auc(y_val[va], p_val[va]) if va.any() else float('nan')
    v_c = _auc(y_val[vc], p_val[vc]) if vc.any() else float('nan')
    val_gap = v_a - v_c if not (np.isnan(v_a) or np.isnan(v_c)) else float('nan')

    sas_img = sas_with_mask_variants(model, test_loader) if mask_variants else None
    if sas_img is None:
        sas_mean = float(core_pipeline.get_mean_sas(model, test_loader))
        sas_extra = {}
    else:
        sas_mean = float(np.mean(sas_img['orig']))
        sas_extra = {f'sas_{k}': float(np.mean(v)) for k, v in sas_img.items() if k != 'orig'}

    if run_probe:
        probe = core_pipeline.bootstrap_probe_accuracy(model, test_loader, seed=seed)
    else:
        probe = {'linear_mean': np.nan, 'linear_std': np.nan, 'mlp_mean': np.nan, 'mlp_std': np.nan}
    views = core_pipeline.evaluate_view_subgroups(model, test_df)

    tdf = pd.DataFrame({
        'row': np.arange(len(test_df)),
        'filepath': test_df['filepath'].to_numpy(),
        'patient_id': test_df['patient_id'].to_numpy() if 'patient_id' in test_df else np.nan,
        'view': test_df['view'].to_numpy() if 'view' in test_df else np.nan,
        'malignant': y_test, 'density_binary': test_df['density_binary'].to_numpy(),
        'group': np.where(a_mask, 'aligned', 'conflict'), 'prob': p_test, 'logit': z_test,
    })
    if sas_img is not None:
        for k, v in sas_img.items():
            tdf[f'sas_{k}'] = v
    tdf.to_csv(os.path.join(out['probs'], run_id + '__test.csv'), index=False)
    pd.DataFrame({'row': np.arange(len(val_df)), 'malignant': y_val,
                  'density_binary': val_df['density_binary'].to_numpy(),
                  'group': np.where(va, 'aligned', 'conflict'), 'prob': p_val}
                 ).to_csv(os.path.join(out['probs'], run_id + '__val.csv'), index=False)
    if save_ckpt:
        torch.save(model.state_dict(), os.path.join(out['ckpt'], run_id + '.pt'))

    best_val = max((r['val_auroc'] for r in hist), default=float('nan'))
    row = {
        'tag': tag, 'ref_tag': ref_tag if ref_tag is not None else tag,
        'strategy': label, 'seed': seed,
        'auroc_all': auroc_all, 'auroc_aligned': auroc_a, 'auroc_conflict': auroc_c,
        'worst_group_gap': gap, 'sas': sas_mean,
        'probe_linear_mean': probe['linear_mean'], 'probe_linear_std': probe['linear_std'],
        'probe_mlp_mean': probe['mlp_mean'], 'probe_mlp_std': probe['mlp_std'],
        'auroc_cc': views['auroc_cc'], 'auroc_mlo': views['auroc_mlo'],
        'val_auroc': val_auroc, 'val_gap': val_gap, 'best_epoch_val_auroc': best_val,
        'epochs_run': len(hist), 'train_seconds': round(train_seconds, 1),
        'n_val': len(val_df), 'n_val_aligned': int(va.sum()), 'n_val_conflict': int(vc.sum()),
    }
    row.update(gc_)
    row.update(sas_extra)
    row.update(info or {})
    row.update(extra or {})

    print(f"  [{tag} | {label} | seed {seed}] AUROC={auroc_all:.4f} aligned={auroc_a:.4f} "
          f"conflict={auroc_c:.4f} gap={gap:.4f} SAS={sas_mean:.4f} "
          f"probe={probe['linear_mean']:.4f} ({train_seconds/60:.1f} min train)")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def paired_vs_baseline(df, metrics=METRICS, baseline_label='baseline'):
    """Paired difference against the same-seed baseline: bootstrap CI and Wilcoxon p."""
    rows = []
    for (tag, strat), g in df.groupby(['tag', 'strategy']):
        if strat == baseline_label:
            continue
        ref_tag = g['ref_tag'].iloc[0] if 'ref_tag' in g else tag
        base = df[(df['tag'] == ref_tag) & (df['strategy'] == baseline_label)].set_index('seed')
        arm = g.set_index('seed')
        common = base.index.intersection(arm.index)
        if len(common) < 2:
            continue
        for m in metrics:
            b, a = base.loc[common, m].to_numpy(float), arm.loc[common, m].to_numpy(float)
            ok = ~(np.isnan(a) | np.isnan(b))
            if ok.sum() < 2:
                continue
            d = a[ok] - b[ok]
            mean_d, lo, hi = core_pipeline.bootstrap_ci(d)
            try:
                p = wilcoxon(d)[1]
            except ValueError:
                p = float('nan')
            rows.append({'tag': tag, 'strategy': strat, 'ref_tag': ref_tag, 'metric': m, 'n': int(ok.sum()),
                         'mean_diff_vs_baseline': mean_d, 'ci_lo': lo, 'ci_hi': hi, 'wilcoxon_p': p,
                         'min_possible_p': 2 * 0.5 ** int(ok.sum())})
    return pd.DataFrame(rows)


def summary_by_tag(df, metrics=METRICS + ('auroc_aligned', 'auroc_conflict', 'val_auroc', 'val_gap')):
    """Mean and SD per tag and strategy."""
    agg = {m: ['mean', 'std'] for m in metrics if m in df.columns}
    for c in ('n_train', 'r_train', 'n_aligned', 'n_conflict', 'epochs_run'):
        if c in df.columns:
            agg[c] = ['mean']
    s = df.groupby(['tag', 'strategy']).agg(agg)
    s.columns = ['_'.join(c) for c in s.columns]
    s['n_seeds'] = df.groupby(['tag', 'strategy']).size()
    return s.reset_index()


def write_summaries(df, out):
    """Write summary_by_tag.csv and paired_vs_baseline.csv."""
    s = summary_by_tag(df)
    s.to_csv(os.path.join(out['root'], 'summary_by_tag.csv'), index=False)
    p = paired_vs_baseline(df)
    p.to_csv(os.path.join(out['root'], 'paired_vs_baseline.csv'), index=False)
    with pd.option_context('display.width', 220, 'display.max_columns', 50, 'display.max_rows', 500):
        print("\n==== SUMMARY (mean / sd over seeds) ====")
        print(s.round(4).to_string(index=False))
        if len(p):
            print("\n==== PAIRED vs baseline (same seeds; 95% bootstrap CI; Wilcoxon) ====")
            print(p.round(4).to_string(index=False))
    return s, p


def plan_banner(n_runs, minutes_per_run=None):
    """Print the number of runs and a rough time estimate."""
    msg = f"Planned training runs: {n_runs}"
    if minutes_per_run:
        msg += f"  (~{n_runs * minutes_per_run / 60:.1f} h at {minutes_per_run} min/run)"
    print(msg)
