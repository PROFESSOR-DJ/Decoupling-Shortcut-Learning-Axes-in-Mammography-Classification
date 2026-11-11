"""Calibration, decision threshold and SAS-under-perturbed-mask analysis of saved probabilities.

Run: python analysis/calibration_analysis.py calibration --dir <results folder>
"""
import os
import sys

# Make src/ and analysis/ importable when the script is run from any folder.
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', d) for d in ('src', 'analysis')]

import re
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression

from paths import OUTPUT_DIR

HERE = OUTPUT_DIR


def _safe(s):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(s))


def ece_score(y, p, n_bins=10):
    """Expected calibration error (10 bins)."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def calib_slope_intercept(y, p):
    """Calibration slope and intercept from logistic recalibration."""
    if len(np.unique(y)) < 2:
        return float('nan'), float('nan')
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def sens_spec(y, p, thr):
    """Sensitivity and specificity at a threshold."""
    pred = p >= thr
    pos, neg = y == 1, y == 0
    sens = float(pred[pos].mean()) if pos.any() else float('nan')
    spec = float((~pred[neg]).mean()) if neg.any() else float('nan')
    return sens, spec


def youden_threshold(y, p):
    """Threshold that maximises Youden's J."""
    if len(np.unique(y)) < 2:
        return float('nan')
    fpr, tpr, thr = roc_curve(y, p)
    k = int(np.argmax(tpr - fpr))
    return float(thr[k]) if np.isfinite(thr[k]) else 1.0


def threshold_at_sensitivity(y, p, target=0.90):
    if len(np.unique(y)) < 2:
        return float('nan')
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    return float(thr[ok[0]]) if len(ok) else float('nan')


def run_metrics(test, val):
    """All calibration and threshold metrics for one run."""
    y, p = test['malignant'].to_numpy().astype(int), test['prob'].to_numpy()
    yv, pv = val['malignant'].to_numpy().astype(int), val['prob'].to_numpy()
    slope, icpt = calib_slope_intercept(y, p)
    t_val = youden_threshold(yv, pv)
    t90 = threshold_at_sensitivity(yv, pv, 0.90)
    se05, sp05 = sens_spec(y, p, 0.5)
    seV, spV = sens_spec(y, p, t_val)
    se90, sp90 = sens_spec(y, p, t90)
    r = {
        'ece': ece_score(y, p), 'brier': float(brier_score_loss(y, p)),
        'log_loss': float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        'calib_slope': slope, 'calib_intercept': icpt,
        'mean_prob': float(p.mean()), 'prevalence': float(y.mean()),
        'calib_in_the_large': float(p.mean() - y.mean()),
        'val_youden_threshold': t_val, 'val_thr_sens90': t90,
        'sens_at_0.5': se05, 'spec_at_0.5': sp05,
        'sens_at_val_thr': seV, 'spec_at_val_thr': spV,
        'sens_at_sens90_thr': se90, 'spec_at_sens90_thr': sp90,
        'frac_pred_pos_at_0.5': float((p >= 0.5).mean()),
    }
    for g in ('aligned', 'conflict'):
        m = (test['group'] == g).to_numpy()
        if m.sum() > 0:
            r[f'ece_{g}'] = ece_score(y[m], p[m])
            r[f'mean_prob_{g}'] = float(p[m].mean())
            r[f'prevalence_{g}'] = float(y[m].mean())
            s, c = sens_spec(y[m], p[m], 0.5)
            r[f'sens_{g}_at_0.5'], r[f'spec_{g}_at_0.5'] = s, c
            s, c = sens_spec(y[m], p[m], t_val)
            r[f'sens_{g}_at_val_thr'], r[f'spec_{g}_at_val_thr'] = s, c
    return r


CAL_METRICS = ['ece', 'brier', 'log_loss', 'calib_slope', 'calib_intercept', 'calib_in_the_large',
               'val_youden_threshold', 'sens_at_0.5', 'spec_at_0.5', 'sens_at_val_thr', 'spec_at_val_thr',
               'ece_aligned', 'ece_conflict', 'sens_conflict_at_0.5', 'spec_conflict_at_0.5',
               'sens_conflict_at_val_thr', 'spec_conflict_at_val_thr']


def load_runs(d):
    """Load results.csv and the saved test / validation probabilities."""
    res = pd.read_csv(os.path.join(d, 'results.csv'))
    rows, pooled = [], {}
    for _, r in res.iterrows():
        rid = f"{_safe(r['tag'])}__{_safe(r['strategy'])}__seed{int(r['seed'])}"
        ft, fv = (os.path.join(d, 'probs', rid + s) for s in ('__test.csv', '__val.csv'))
        if not (os.path.exists(ft) and os.path.exists(fv)):
            print(f"  [skip] probabilities missing for {rid}")
            continue
        test, val = pd.read_csv(ft), pd.read_csv(fv)
        m = run_metrics(test, val)
        m.update({'tag': r['tag'], 'ref_tag': r.get('ref_tag', r['tag']), 'strategy': r['strategy'], 'seed': int(r['seed'])})
        rows.append(m)
        pooled.setdefault((r['tag'], r['strategy']), []).append(test[['malignant', 'prob']])
    return res, pd.DataFrame(rows), pooled


def reliability_plot(pooled, tag, path, n_bins=10):
    """Reliability diagram per strategy."""
    strategies = [s for (t, s) in pooled if t == tag]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={'width_ratios': [3, 2]})
    ax, ax2 = axes
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='perfect')
    for s in strategies:
        d = pd.concat(pooled[(tag, s)])
        y, p = d['malignant'].to_numpy(), d['prob'].to_numpy()
        b = np.minimum((p * n_bins).astype(int), n_bins - 1)
        xs = [p[b == k].mean() for k in range(n_bins) if (b == k).sum() >= 10]
        ys = [y[b == k].mean() for k in range(n_bins) if (b == k).sum() >= 10]
        ax.plot(xs, ys, marker='o', ms=4, label=f"{s} (ECE {ece_score(y, p):.3f})")
        ax2.hist(p, bins=20, range=(0, 1), histtype='step', label=s)
    ax.set_xlabel('mean predicted probability'); ax.set_ylabel('observed malignant fraction')
    ax.set_title(f'Reliability ({tag}; test folds pooled over seeds)'); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax2.set_xlabel('predicted probability'); ax2.set_title('distribution of predictions'); ax2.grid(alpha=.3)
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def threshold_plot(cal, tag, path):
    """Validation threshold and sensitivity / specificity per strategy."""
    g = cal[cal['tag'] == tag].groupby('strategy')[
        ['val_youden_threshold', 'sens_at_0.5', 'spec_at_0.5', 'sens_at_val_thr', 'spec_at_val_thr']].agg(['mean', 'std'])
    if g.empty:
        return
    strat = list(g.index)
    x = np.arange(len(strat))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(x, g[('val_youden_threshold', 'mean')], yerr=g[('val_youden_threshold', 'std')], capsize=3)
    axes[0].axhline(0.5, color='gray', ls='--'); axes[0].set_title('validation-optimal (Youden) threshold')
    for ax, m, name in ((axes[1], 'sens', 'Sensitivity'), (axes[2], 'spec', 'Specificity')):
        ax.bar(x - 0.2, g[(f'{m}_at_0.5', 'mean')], 0.4, yerr=g[(f'{m}_at_0.5', 'std')], capsize=2, label='threshold 0.5')
        ax.bar(x + 0.2, g[(f'{m}_at_val_thr', 'mean')], 0.4, yerr=g[(f'{m}_at_val_thr', 'std')], capsize=2, label='validation threshold')
        ax.set_title(f'{name} on test'); ax.legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(strat, rotation=25, ha='right'); ax.grid(alpha=.3, axis='y')
    fig.suptitle(f'Decision-threshold shift ({tag}), mean +/- SD over seeds')
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def sas_sensitivity(res, d):
    """SAS under dilated / eroded masks, paired against the baseline."""
    import experiment_utils as rc
    cols = [c for c in ('sas', 'sas_dil3', 'sas_dil6', 'sas_ero3', 'sas_ero6') if c in res.columns]
    if len(cols) < 2:
        print("  (no dilated/eroded SAS columns in results.csv -- skipped)")
        return None
    if 'ref_tag' not in res:
        res = res.assign(ref_tag=res['tag'])
    summ = res.groupby(['tag', 'strategy'])[cols].agg(['mean', 'std'])
    summ.columns = ['_'.join(c) for c in summ.columns]
    pair = rc.paired_vs_baseline(res, metrics=tuple(cols))
    if len(pair):
        pair['sign'] = np.sign(pair['mean_diff_vs_baseline'])
        cons = pair.groupby(['tag', 'strategy'])['sign'].agg(lambda s: bool(s.nunique() == 1)).rename('direction_same_for_all_masks')
        pair = pair.merge(cons.reset_index(), on=['tag', 'strategy'])
    summ.reset_index().to_csv(os.path.join(d, 'sas_mask_sensitivity_summary.csv'), index=False)
    pair.to_csv(os.path.join(d, 'sas_mask_sensitivity.csv'), index=False)
    print("\n==== SAS under dilated / eroded masks (paired vs baseline) ====")
    with pd.option_context('display.width', 200, 'display.max_columns', 30):
        print(pair.round(4).to_string(index=False))
    return pair


def run_calibration(d):
    """Calibration and mask analysis for one results folder."""
    import experiment_utils as rc
    print(f"\n=== calibration / threshold analysis for {d} ===")
    res, cal, pooled = load_runs(d)
    if cal.empty:
        print("no probability files found -- nothing to analyse")
        return {}
    cal.to_csv(os.path.join(d, 'calibration_per_run.csv'), index=False)
    cols = [c for c in CAL_METRICS if c in cal.columns]
    s = cal.groupby(['tag', 'strategy'])[cols].agg(['mean', 'std'])
    s.columns = ['_'.join(c) for c in s.columns]
    s['n_seeds'] = cal.groupby(['tag', 'strategy']).size()
    s.reset_index().to_csv(os.path.join(d, 'calibration_summary.csv'), index=False)
    pair = rc.paired_vs_baseline(cal, metrics=tuple(cols))
    pair.to_csv(os.path.join(d, 'calibration_paired_vs_baseline.csv'), index=False)
    with pd.option_context('display.width', 220, 'display.max_columns', 40, 'display.max_rows', 300):
        print(s.reset_index()[['tag', 'strategy', 'ece_mean', 'ece_std', 'brier_mean', 'calib_slope_mean',
                               'calib_intercept_mean', 'val_youden_threshold_mean', 'n_seeds']].round(4).to_string(index=False))
        if len(pair):
            print("\nPaired vs baseline:")
            print(pair[pair['metric'].isin(['ece', 'calib_slope', 'val_youden_threshold', 'sens_at_0.5', 'spec_at_0.5'])]
                  .round(4).to_string(index=False))
    for tag in sorted(cal['tag'].unique()):
        reliability_plot(pooled, tag, os.path.join(d, f'reliability_diagram_{_safe(tag)}.png'))
        threshold_plot(cal, tag, os.path.join(d, f'threshold_shift_{_safe(tag)}.png'))
    sas_sensitivity(res, d)
    return {'per_run': cal, 'summary': s, 'paired': pair}


def run_reproduce(d, tag='main', main_csv=None):
    """Compare a rerun with the main-run results seed by seed."""
    res = pd.read_csv(os.path.join(d, 'results.csv'))
    res = res[res['tag'] == tag]
    cands = [main_csv] if main_csv else [os.path.join(HERE, 'main_experiment', 'dccam_cbis_results.csv'),
                                         os.path.join(HERE, 'dccam_cbis_results.csv')]
    path = next((c for c in cands if c and os.path.exists(c)), None)
    if path is None:
        sys.exit("could not find dccam_cbis_results.csv -- pass --main-csv <path>")
    old = pd.read_csv(path)
    m = ['auroc_all', 'auroc_aligned', 'auroc_conflict', 'worst_group_gap', 'sas', 'probe_linear_mean']
    mg = res.merge(old, on=['seed', 'strategy'], suffixes=('_new', '_old'))
    if mg.empty:
        sys.exit("no overlapping (seed, strategy) rows between the new run and the main CSV")
    for k in m:
        mg[f'{k}_diff'] = mg[f'{k}_new'] - mg[f'{k}_old']
    keep = ['seed', 'strategy'] + [c for k in m for c in (f'{k}_old', f'{k}_new', f'{k}_diff')]
    mg[keep].to_csv(os.path.join(d, 'reproduce_vs_main.csv'), index=False)
    rows = []
    for strat, g in mg.groupby('strategy'):
        r = {'strategy': strat, 'n_seeds': len(g)}
        for k in m:
            r[f'{k}_old_mean'], r[f'{k}_new_mean'] = g[f'{k}_old'].mean(), g[f'{k}_new'].mean()
            r[f'{k}_old_sd'], r[f'{k}_new_sd'] = g[f'{k}_old'].std(ddof=1), g[f'{k}_new'].std(ddof=1)
            r[f'{k}_mean_abs_diff'] = g[f'{k}_diff'].abs().mean()
            r[f'{k}_max_abs_diff'] = g[f'{k}_diff'].abs().max()
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(d, 'reproduce_summary.csv'), index=False)
    print(f"\n=== new run ('{tag}') vs {path} ===")
    with pd.option_context('display.width', 220, 'display.max_columns', 40):
        print(out[['strategy', 'n_seeds', 'auroc_all_old_mean', 'auroc_all_new_mean', 'auroc_all_mean_abs_diff',
                   'worst_group_gap_old_mean', 'worst_group_gap_new_mean', 'worst_group_gap_mean_abs_diff',
                   'worst_group_gap_old_sd', 'worst_group_gap_new_sd']].round(4).to_string(index=False))
    print("\nInterpretation: a mean |diff| near 0 means the rerun is bit-for-bit reproducible on your machine;"
          "\nsmall (< ~0.01) differences are GPU non-determinism; large ones mean the paper's table came from different code/settings.")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['calibration', 'reproduce', 'all'])
    ap.add_argument('--dir', required=True, help="results folder (contains results.csv and probs/)")
    ap.add_argument('--tag', default='main', help="reproduce: which tag to compare with the main CSV")
    ap.add_argument('--main-csv', default=None)
    a = ap.parse_args(argv)
    if a.cmd in ('calibration', 'all'):
        run_calibration(a.dir)
    if a.cmd in ('reproduce', 'all'):
        run_reproduce(a.dir, a.tag, a.main_csv)


if __name__ == '__main__':
    main()
