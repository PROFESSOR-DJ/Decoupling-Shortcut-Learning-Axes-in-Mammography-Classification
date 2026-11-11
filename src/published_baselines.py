"""Published comparison methods trained on the same splits: MINE penalty, non-linear
adversary (Muller et al.), Group DRO and JTT. Writes the combined 10-arm table.
"""

import os, sys, copy, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from sklearn.model_selection import GroupShuffleSplit

try:
    from core_pipeline import (
        device, OUT_DIR, BATCH_SIZE, MAX_EPOCHS, PATIENCE, LR,
        LAMBDA_RAMP_EPOCHS, SEED_LIST, BIAS_TRAINING, BIAS_STRENGTH,
        CBISDataset, bias_split_by_correlation,
        group_masks, make_loader, DCCAMNet,
        get_auroc, get_mean_sas, evaluate_view_subgroups, bootstrap_probe_accuracy,
        bootstrap_ci, power_analysis_table, plot_forest,
        load_data,
    )
except ImportError as e:
    raise ImportError(
        "Could not import from core_pipeline.py. Make sure published_baselines.py is in the "
        "SAME directory as core_pipeline.py (rename if your local copy has a different "
        f"filename). Original error: {e}"
    )

BENCHMARK_ARMS = ['mine', 'adv_nonlinear', 'group_dro', 'jtt']

MINE_CRITIC_STEPS = 5
MAX_WEIGHT_MINE = 0.55
MINE_CRITIC_LR = 1e-3

GROUP_DRO_STEP_SIZE = 0.01

JTT_PHASE1_EPOCHS = 3
JTT_UPWEIGHT = 20


class MINEStatisticsNetwork(nn.Module):
    """Statistics network T(x, y) for the MINE estimator."""
    def __init__(self, in_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 400, bias=False), nn.BatchNorm1d(400), nn.ReLU(inplace=True),
            nn.Linear(400, 400, bias=False), nn.BatchNorm1d(400), nn.ReLU(inplace=True),
            nn.Linear(400, 400, bias=False), nn.BatchNorm1d(400), nn.ReLU(inplace=True),
            nn.Linear(400, 1),
        )

    def forward(self, x):
        return self.layers(x)


class MINEEstimator(nn.Module):
    """MINE lower bound on mutual information (Belghazi et al., 2018)."""
    def __init__(self, feature_dim):
        super().__init__()
        self.stats_network = MINEStatisticsNetwork(feature_dim + 1)

    def forward(self, x, y):
        if y.dim() == 1:
            y = y.unsqueeze(1)
        batch_size = x.shape[0]
        xy = torch.cat([x.repeat_interleave(batch_size, dim=0), y.tile(batch_size, 1)], -1)
        stats = self.stats_network(xy).reshape(batch_size, batch_size)
        diag = torch.diagonal(stats).mean()
        off_diag = stats - torch.diag(torch.full((batch_size,), float('inf'), device=stats.device))
        logsumexp = torch.logsumexp(off_diag, dim=0) - torch.log(torch.tensor(float(batch_size), device=stats.device))
        return diag - logsumexp.mean()


class AdvClassifierNonlinear(nn.Module):
    """Non-linear adversary head from Muller et al. (2025)."""
    def __init__(self, z_shape=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(z_shape, z_shape * 2), nn.ReLU(),
            nn.Linear(z_shape * 2, z_shape * 2), nn.ReLU(),
            nn.Linear(z_shape * 2, 1),
        )

    def forward(self, x):
        return self.layers(x)


def train_mine(train_df, val_df, seed):
    """Train with a MINE mutual-information penalty."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = DCCAMNet(use_grl=False).to(device)
    mine_net = MINEEstimator(feature_dim=model.feat_dim).to(device)
    optimizer = torch.optim.Adam(list(model.backbone.parameters()) + list(model.main_head.parameters()), lr=LR)
    critic_optimizer = torch.optim.Adam(mine_net.parameters(), lr=MINE_CRITIC_LR)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

    train_loader = make_loader(CBISDataset(train_df, augment=True), BATCH_SIZE, shuffle=True)
    val_loader = make_loader(CBISDataset(val_df, augment=False), BATCH_SIZE, shuffle=False)
    best_val_auroc, best_state, patience_ct = -np.inf, None, 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        w_mine = MAX_WEIGHT_MINE * min(1.0, epoch / LAMBDA_RAMP_EPOCHS)
        total_loss = 0.0
        for imgs, masks, labels, density in train_loader:
            imgs = imgs.to(device); labels = labels.float().to(device); density = density.float().to(device)
            if imgs.size(0) < 4:
                continue

            with torch.no_grad():
                pooled_cached, _ = model._forward_spatial(imgs)
            for _ in range(MINE_CRITIC_STEPS):
                critic_optimizer.zero_grad()
                mi_est = mine_net(pooled_cached, density)
                (-mi_est).backward()
                critic_optimizer.step()

            optimizer.zero_grad()
            with autocast('cuda', enabled=(device.type == 'cuda')):
                main_logit, _, pooled, _ = model(imgs)
                main_loss = F.binary_cross_entropy_with_logits(main_logit, labels)
                mi_est_encoder = mine_net(pooled, density)
                loss = main_loss + w_mine * mi_est_encoder
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, masks, labels, density in val_loader:
                imgs = imgs.to(device)
                main_logit, _, _, _ = model(imgs)
                val_preds.append(torch.sigmoid(main_logit).cpu().numpy())
                val_targets.append(labels.numpy())
        val_preds = np.concatenate(val_preds); val_targets = np.concatenate(val_targets)
        from sklearn.metrics import roc_auc_score
        val_auroc = roc_auc_score(val_targets, val_preds) if len(np.unique(val_targets)) > 1 else 0.5
        print(f"  [mine] epoch {epoch+1} — w_mine {w_mine:.2f} — loss {total_loss/len(train_loader):.4f} — val AUROC {val_auroc:.4f}")

        if epoch >= LAMBDA_RAMP_EPOCHS:
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
    return model


def train_adv_nonlinear(train_df, val_df, seed):
    """Train with the non-linear adversary behind the gradient reversal layer."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = DCCAMNet(use_grl=True).to(device)
    model.nuisance_head = AdvClassifierNonlinear(z_shape=model.feat_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

    train_loader = make_loader(CBISDataset(train_df, augment=True), BATCH_SIZE, shuffle=True)
    val_loader = make_loader(CBISDataset(val_df, augment=False), BATCH_SIZE, shuffle=False)
    best_val_auroc, best_state, patience_ct = -np.inf, None, 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        lambda_grl = 5.0 * min(1.0, epoch / LAMBDA_RAMP_EPOCHS)
        total_loss = 0.0
        for imgs, masks, labels, density in train_loader:
            imgs = imgs.to(device); labels = labels.float().to(device); density = density.float().to(device)
            optimizer.zero_grad()
            with autocast('cuda', enabled=(device.type == 'cuda')):
                main_logit, nuisance_logit, pooled, _ = model(imgs, lambda_=lambda_grl)
                main_loss = F.binary_cross_entropy_with_logits(main_logit, labels)
                nuisance_loss = F.binary_cross_entropy_with_logits(nuisance_logit.squeeze(-1), density)
                loss = main_loss + 2.0 * nuisance_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, masks, labels, density in val_loader:
                imgs = imgs.to(device)
                main_logit, _, _, _ = model(imgs)
                val_preds.append(torch.sigmoid(main_logit).cpu().numpy())
                val_targets.append(labels.numpy())
        val_preds = np.concatenate(val_preds); val_targets = np.concatenate(val_targets)
        from sklearn.metrics import roc_auc_score
        val_auroc = roc_auc_score(val_targets, val_preds) if len(np.unique(val_targets)) > 1 else 0.5
        print(f"  [adv_nonlinear] epoch {epoch+1} — lambda {lambda_grl:.2f} — loss {total_loss/len(train_loader):.4f} — val AUROC {val_auroc:.4f}")

        if epoch >= LAMBDA_RAMP_EPOCHS:
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
    return model


def train_group_dro(train_df, val_df, seed):
    """Group DRO over the four malignancy x density groups (Sagawa et al., 2020)."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = DCCAMNet(use_grl=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

    train_loader = make_loader(CBISDataset(train_df, augment=True), BATCH_SIZE, shuffle=True)
    val_loader = make_loader(CBISDataset(val_df, augment=False), BATCH_SIZE, shuffle=False)
    best_val_auroc, best_state, patience_ct = -np.inf, None, 0
    n_groups = 4
    group_weights = torch.ones(n_groups, device=device) / n_groups

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0.0
        for imgs, masks, labels, density in train_loader:
            imgs = imgs.to(device); labels = labels.float().to(device); density = density.float().to(device)
            group_ids = (labels.long() * 2 + density.long())

            optimizer.zero_grad()
            with autocast('cuda', enabled=(device.type == 'cuda')):
                main_logit, _, _, _ = model(imgs)
                per_sample_loss = F.binary_cross_entropy_with_logits(main_logit, labels, reduction='none')

                group_losses = torch.zeros(n_groups, device=device)
                group_present = torch.zeros(n_groups, dtype=torch.bool, device=device)
                for g in range(n_groups):
                    mask_g = group_ids == g
                    if mask_g.any():
                        group_losses[g] = per_sample_loss[mask_g].mean()
                        group_present[g] = True

                with torch.no_grad():
                    new_weights = group_weights.clone()
                    new_weights[group_present] = group_weights[group_present] * torch.exp(
                        GROUP_DRO_STEP_SIZE * group_losses[group_present].detach())
                    new_weights = new_weights / new_weights.sum()
                    group_weights = new_weights

                loss = (group_weights[group_present] * group_losses[group_present]).sum() / group_weights[group_present].sum()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, masks, labels, density in val_loader:
                imgs = imgs.to(device)
                main_logit, _, _, _ = model(imgs)
                val_preds.append(torch.sigmoid(main_logit).cpu().numpy())
                val_targets.append(labels.numpy())
        val_preds = np.concatenate(val_preds); val_targets = np.concatenate(val_targets)
        from sklearn.metrics import roc_auc_score
        val_auroc = roc_auc_score(val_targets, val_preds) if len(np.unique(val_targets)) > 1 else 0.5
        print(f"  [group_dro] epoch {epoch+1} — group_weights {group_weights.cpu().numpy().round(3)} — "
              f"loss {total_loss/len(train_loader):.4f} — val AUROC {val_auroc:.4f}")

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
    return model


def _train_erm(train_df, val_df, seed, max_epochs, sample_weights=None):
    """Plain ERM training loop with optional per-example weights (used by JTT)."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = DCCAMNet(use_grl=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

    ds = CBISDataset(train_df, augment=True)
    if sample_weights is not None:
        from torch.utils.data import DataLoader, WeightedRandomSampler
        weight_array = train_df.index.map(lambda i: sample_weights.get(i, 1.0)).to_numpy().astype(np.float64)
        sampler = WeightedRandomSampler(weights=weight_array, num_samples=len(weight_array), replacement=True)
        train_loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                                   num_workers=0, pin_memory=(device.type == 'cuda'))
    else:
        train_loader = make_loader(ds, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(CBISDataset(val_df, augment=False), BATCH_SIZE, shuffle=False)
    best_val_auroc, best_state, patience_ct = -np.inf, None, 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for imgs, masks, labels, density in train_loader:
            imgs = imgs.to(device); labels = labels.float().to(device)
            optimizer.zero_grad()
            with autocast('cuda', enabled=(device.type == 'cuda')):
                main_logit, _, _, _ = model(imgs)
                loss = F.binary_cross_entropy_with_logits(main_logit, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, masks, labels, density in val_loader:
                imgs = imgs.to(device)
                main_logit, _, _, _ = model(imgs)
                val_preds.append(torch.sigmoid(main_logit).cpu().numpy())
                val_targets.append(labels.numpy())
        val_preds = np.concatenate(val_preds); val_targets = np.concatenate(val_targets)
        from sklearn.metrics import roc_auc_score
        val_auroc = roc_auc_score(val_targets, val_preds) if len(np.unique(val_targets)) > 1 else 0.5
        print(f"    epoch {epoch+1} — loss {total_loss/len(train_loader):.4f} — val AUROC {val_auroc:.4f}")

        if epoch >= 3:
            if val_auroc > best_val_auroc:
                best_val_auroc, best_state, patience_ct = val_auroc, copy.deepcopy(model.state_dict()), 0
            else:
                patience_ct += 1
                if patience_ct >= PATIENCE:
                    break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


def train_jtt(train_df, val_df, seed):
    """Just Train Twice: short ERM run, then retrain with misclassified examples upweighted."""
    print(f"  [jtt] --- phase 1: {JTT_PHASE1_EPOCHS}-epoch ERM to identify hard examples ---")
    phase1_model = _train_erm(train_df, val_df, seed, max_epochs=JTT_PHASE1_EPOCHS)

    phase1_loader = make_loader(CBISDataset(train_df, augment=False), BATCH_SIZE, shuffle=False)
    phase1_model.eval()
    all_preds, row_indices = [], list(train_df.index)
    with torch.no_grad():
        for imgs, masks, labels, density in phase1_loader:
            imgs = imgs.to(device)
            main_logit, _, _, _ = phase1_model(imgs)
            preds = (torch.sigmoid(main_logit) > 0.5).float().cpu().numpy()
            all_preds.append(preds)
    all_preds = np.concatenate(all_preds)
    true_labels = train_df['malignant'].to_numpy()
    misclassified = all_preds != true_labels
    n_misclassified = misclassified.sum()
    print(f"  [jtt] phase 1 identified {n_misclassified}/{len(train_df)} misclassified training examples "
          f"({n_misclassified/len(train_df):.1%}) -- upweighting these by {JTT_UPWEIGHT}x in phase 2")

    sample_weights = {idx: (JTT_UPWEIGHT if mis else 1.0) for idx, mis in zip(row_indices, misclassified)}
    print("  [jtt] --- phase 2: fresh model, upweighted retrain ---")
    phase2_model = _train_erm(train_df, val_df, seed + 10000, max_epochs=MAX_EPOCHS, sample_weights=sample_weights)
    return phase2_model


ARM_TRAIN_FUNCS = {
    'mine': train_mine,
    'adv_nonlinear': train_adv_nonlinear,
    'group_dro': train_group_dro,
    'jtt': train_jtt,
}


def run_benchmark(meta_raw, patient_col, seeds):
    """Train every benchmark method on every seed and evaluate like the main sweep."""
    all_results = []
    for seed in seeds:
        print(f"\n{'#'*60}\nSEED {seed}\n{'#'*60}")
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
        test_loader = make_loader(CBISDataset(test_df, augment=False), BATCH_SIZE, shuffle=False)
        aligned_loader = make_loader(CBISDataset(aligned_df, augment=False), BATCH_SIZE, shuffle=False) if len(aligned_df) else None
        conflict_loader = make_loader(CBISDataset(conflict_df, augment=False), BATCH_SIZE, shuffle=False) if len(conflict_df) else None

        for arm in BENCHMARK_ARMS:
            print(f"\n--- {arm}, seed {seed} ---")
            model = ARM_TRAIN_FUNCS[arm](train_df_used, val_df, seed)

            auroc = get_auroc(model, test_loader)
            auroc_aligned = get_auroc(model, aligned_loader) if aligned_loader else np.nan
            auroc_conflict = get_auroc(model, conflict_loader) if conflict_loader else np.nan
            gap = auroc_aligned - auroc_conflict if not (np.isnan(auroc_aligned) or np.isnan(auroc_conflict)) else np.nan
            sas = get_mean_sas(model, test_loader)
            probe = bootstrap_probe_accuracy(model, test_loader, seed=seed)
            view_scores = evaluate_view_subgroups(model, test_df)

            print(f"  [{arm}] AUROC(all)={auroc:.4f}  gap={gap:.4f}  SAS={sas:.4f}  "
                  f"probe(linear)={probe['linear_mean']:.4f}")

            all_results.append({
                'seed': seed, 'strategy': arm,
                'auroc_all': auroc, 'auroc_aligned': auroc_aligned, 'auroc_conflict': auroc_conflict,
                'worst_group_gap': gap, 'sas': sas,
                'probe_linear_mean': probe['linear_mean'], 'probe_linear_std': probe['linear_std'],
                'probe_mlp_mean': probe['mlp_mean'], 'probe_mlp_std': probe['mlp_std'],
                'auroc_cc': view_scores['auroc_cc'], 'auroc_mlo': view_scores['auroc_mlo'],
            })

    return pd.DataFrame(all_results)


def main():
    """Run the benchmark, compare with the baseline and write the combined 10-arm table."""
    n_seeds_arg = None
    if len(sys.argv) > 1 and sys.argv[1] == '--seeds':
        n_seeds_arg = int(sys.argv[2])
    seeds = SEED_LIST[:n_seeds_arg] if n_seeds_arg else SEED_LIST
    print(f"Running benchmark arms {BENCHMARK_ARMS} on seeds {seeds}")

    meta_raw, patient_col = load_data()
    new_results_df = run_benchmark(meta_raw, patient_col, seeds)
    new_results_df.to_csv(os.path.join(OUT_DIR, 'benchmark_baselines_results.csv'), index=False)

    existing_path = os.path.join(OUT_DIR, 'dccam_cbis_results.csv')
    if os.path.exists(existing_path):
        existing_df = pd.read_csv(existing_path)
        baseline_rows = existing_df[(existing_df['strategy'] == 'baseline') & (existing_df['seed'].isin(seeds))]
        combined_for_stats = pd.concat([baseline_rows, new_results_df], ignore_index=True)

        metrics = ['worst_group_gap', 'sas', 'probe_linear_mean', 'auroc_all']
        better_map = {
            'worst_group_gap': 'lower', 'sas': 'lower',
            'probe_linear_mean': 'lower', 'auroc_all': 'higher (or unchanged)',
        }
        print(f"\n{'='*70}\nPAIRED COMPARISONS: new benchmark arms vs. baseline (n={len(seeds)} seeds)\n{'='*70}")
        stat_rows = []
        for arm in BENCHMARK_ARMS:
            arm_df = combined_for_stats[combined_for_stats.strategy == arm].set_index('seed')
            base_df = combined_for_stats[combined_for_stats.strategy == 'baseline'].set_index('seed')
            common = base_df.index.intersection(arm_df.index)
            for metric in metrics:
                b = base_df.loc[common, metric].values
                a = arm_df.loc[common, metric].values
                valid = ~(np.isnan(a) | np.isnan(b))
                if valid.sum() < 2:
                    continue
                diffs = a[valid] - b[valid]
                mean_diff, lo, hi = bootstrap_ci(diffs)
                from scipy.stats import wilcoxon
                try:
                    _, p_val = wilcoxon(a[valid], b[valid])
                except ValueError:
                    p_val = np.nan
                print(f"  {arm:>14} | {metric:<18} | want {better_map[metric]:<20} | "
                      f"mean_diff={mean_diff:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  wilcoxon p={p_val:.4f}")
                stat_rows.append({'arm': arm, 'metric': metric, 'mean_diff': mean_diff,
                                   'ci_lo': lo, 'ci_hi': hi, 'wilcoxon_p': p_val})
        stats_df = pd.DataFrame(stat_rows)
        stats_df.to_csv(os.path.join(OUT_DIR, 'benchmark_baselines_stats.csv'), index=False)
        plot_forest(combined_for_stats, ['baseline'] + BENCHMARK_ARMS,
                    os.path.join(OUT_DIR, 'benchmark_baselines_forest_plot.png'))
        power_df = power_analysis_table(combined_for_stats, ['baseline'] + BENCHMARK_ARMS)
        power_df.to_csv(os.path.join(OUT_DIR, 'benchmark_baselines_power_analysis.csv'), index=False)

        full_existing = existing_df[existing_df['seed'].isin(seeds)]
        combined_10arm = pd.concat([full_existing, new_results_df], ignore_index=True)
        combined_10arm.to_csv(os.path.join(OUT_DIR, 'combined_10arm_results.csv'), index=False)
        summary = combined_10arm.groupby('strategy').agg(
            auroc_mean=('auroc_all', 'mean'), auroc_std=('auroc_all', 'std'),
            gap_mean=('worst_group_gap', 'mean'), gap_std=('worst_group_gap', 'std'),
            sas_mean=('sas', 'mean'), sas_std=('sas', 'std'),
            probe_mean=('probe_linear_mean', 'mean'), probe_std=('probe_linear_mean', 'std'),
        )
        order = ['baseline', 'grl_only', 'adv_nonlinear', 'distcorr', 'mi_min', 'mine',
                 'camreg', 'full', 'group_dro', 'jtt']
        summary = summary.reindex([a for a in order if a in summary.index])
        summary.to_csv(os.path.join(OUT_DIR, 'combined_10arm_summary.csv'))
        print("\n\nCOMBINED 10-ARM SUMMARY (mean ± std)")
        print(summary.round(4).to_string())
    else:
        print(f"\nNote: {existing_path} not found -- skipping merge with main sweep results. "
              f"benchmark_baselines_results.csv still saved on its own.")

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
