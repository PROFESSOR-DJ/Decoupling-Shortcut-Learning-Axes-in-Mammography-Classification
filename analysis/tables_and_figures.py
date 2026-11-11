"""Builds LaTeX/CSV tables and summary figures from the result CSVs.

Run: python analysis/tables_and_figures.py
"""
import os
import sys

# Make src/ and analysis/ importable when the script is run from any folder.
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', d) for d in ('src', 'analysis')]

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import OUTPUT_DIR, SRC_DIR

HERE = OUTPUT_DIR
OUT = os.path.join(HERE, 'tables_and_figures')
ORDER = ['baseline', 'grl_only', 'adv_nonlinear', 'group_dro', 'jtt', 'mine', 'mi_min', 'distcorr', 'camreg', 'full']
LABEL = {'baseline': 'Baseline', 'grl_only': 'GRL', 'adv_nonlinear': 'Adv. (non-linear)', 'group_dro': 'Group DRO',
         'jtt': 'JTT', 'mine': 'MINE', 'mi_min': 'MI-min (CLUB)', 'distcorr': 'DistCorr', 'camreg': 'CamReg', 'full': 'Full'}
COLOR = {'baseline': '#555555', 'grl_only': '#d62728', 'adv_nonlinear': '#ff7f0e', 'group_dro': '#8c564b',
         'jtt': '#e377c2', 'mine': '#bcbd22', 'mi_min': '#17becf', 'distcorr': '#1f77b4', 'camreg': '#2ca02c', 'full': '#9467bd'}
METRICS = [('auroc_all', 'AUROC', 'higher'), ('worst_group_gap', 'Gap', 'lower'), ('sas', 'SAS', 'lower'),
           ('probe_linear_mean', 'Probe', 'lower')]


def find(explicit, names):
    """First existing file among the candidates."""
    cands = ([explicit] if explicit else []) + [os.path.join(HERE, n) for n in names]
    return next((c for c in cands if c and os.path.exists(c)), None)


def boot_ci(d, n=5000, seed=0):
    """Bootstrap 95% CI of a mean."""
    rng = np.random.RandomState(seed)
    d = np.asarray(d, float)
    b = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired(df, base, strat, metric):
    """Paired difference against the baseline for one metric."""
    a = df[df.strategy == strat].drop_duplicates('seed').set_index('seed')[metric]
    b = base.drop_duplicates('seed').set_index('seed')[metric]
    c = a.index.intersection(b.index)
    d = (a.loc[c] - b.loc[c]).dropna().to_numpy()
    if len(d) < 3:
        return None
    m, lo, hi = boot_ci(d)
    try:
        p = float(wilcoxon(d).pvalue)
    except ValueError:
        p = float('nan')
    return dict(n=len(d), diff=m, lo=lo, hi=hi, p=p, lower=int((d < 0).sum()))


def pfmt(p):
    return 'p n/a' if np.isnan(p) else ('p<0.001' if p < 0.001 else f'p={p:.3f}')


def tex_escape(s):
    return str(s).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def write_table(name, header, rows, caption, label, colspec=None):
    """Write a table as CSV and as a booktabs LaTeX file."""
    pd.DataFrame(rows, columns=header).to_csv(os.path.join(OUT, name + '.csv'), index=False)
    colspec = colspec or 'l' + 'c' * (len(header) - 1)
    L = [r'\begin{table}[htbp]', r'\centering', r'\small', rf'\begin{{tabular}}{{{colspec}}}', r'\toprule',
         ' & '.join(header) + r' \\', r'\midrule']
    for r in rows:
        L.append(' & '.join(str(x) for x in r) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', rf'\caption{{{caption}}}', rf'\label{{{label}}}', r'\end{table}']
    open(os.path.join(OUT, name + '.tex'), 'w', encoding='utf-8').write("\n".join(L) + "\n")
    print(f"  wrote {name}.tex / .csv")


def sig_mark(r):
    return '*' if (r['hi'] < 0 or r['lo'] > 0) else ''


def table_setup(pipeline_consts, manifest_gpu):
    """Experimental setup table (constants read from the pipeline)."""
    c = pipeline_consts
    g = lambda k, d='n/a': c.get(k, d)
    rows = [
        ('Dataset (primary)', 'CBIS-DDSM; 3,568 lesion cases (1,457 malignant / 2,111 benign), CC and MLO views'),
        ('Dataset (external)', 'INbreast, full-field digital; BI-RADS 3 and unparseable assessments excluded'),
        ('Density variable', 'BI-RADS breast density, binarised at the cohort median (CBIS-DDSM) / ACR C-D (INbreast)'),
        ('Backbone', 'ResNet-18, ImageNet-pretrained, global average pooling, one linear head'),
        ('Input', f"{g('IMG_SIZE')} x {g('IMG_SIZE')} greyscale replicated to 3 channels; horizontal flip augmentation"),
        ('Optimiser', f"Adam, learning rate {g('LR')}, batch size {g('BATCH_SIZE')}"),
        ('Schedule', f"up to {g('MAX_EPOCHS')} epochs, early stopping on validation AUROC (patience {g('PATIENCE')}); "
                     f"penalty weights ramped linearly over {g('LAMBDA_RAMP_EPOCHS')} epochs"),
        ('Penalty weights (maximum)', f"GRL lambda {g('MAX_LAMBDA_GRL')} (adversary weight {g('GRL_ADV_WEIGHT')}, reset every "
                                      f"{g('GRL_REINIT_EVERY')} epochs); DistCorr {g('MAX_WEIGHT_DISTCORR')}; CAM {g('MAX_WEIGHT_CAMREG')}"),
        ('Shortcut injection', f"training fold only: correlation-targeted subsampling, P(dense | malignant) = {g('BIAS_STRENGTH')} "
                               f"(achieved r about 0.6); validation and test folds untouched"),
        ('Splits', 'patient-grouped 70/30 train+val / test, then 80/20 train / val; no patient crosses a split'),
        ('Seeds', f"{len(c.get('SEED_LIST', []))} (" + (f"{min(c['SEED_LIST'])}--{max(c['SEED_LIST'])}" if c.get('SEED_LIST') else 'n/a')
                  + "); paired comparisons on seed"),
        ('Worst-group evaluation', 'aligned = (malignant and dense) or (benign and not dense); conflicting = the rest; '
                                   'gap = AUROC aligned - AUROC conflicting'),
        ('SAS', 'share of Grad-CAM mass outside the Otsu breast mask'),
        ('Density probe', 'linear and MLP probes on the pooled 512-d features, 20 bootstrap 70/30 splits (accuracy)'),
        ('Statistics', 'paired difference vs baseline, 95% percentile bootstrap CI (5,000 resamples), two-sided Wilcoxon signed-rank'),
        ('Hardware / determinism', (manifest_gpu or 'NVIDIA GPU') + '; cudnn.benchmark off, cudnn.deterministic on'),
    ]
    write_table('table_setup', ['Item', 'Setting'], [(tex_escape(a), tex_escape(b).replace('\\_', '_').replace('_', r'\_')) for a, b in rows],
                'Experimental setup shared by all strategies.', 'tab:setup', colspec='lp{0.72\\textwidth}')


def table_main(df, name, caption, label):
    """Mean +- SD per strategy."""
    rows = []
    for s in [x for x in ORDER if x in set(df.strategy)]:
        d = df[df.strategy == s]
        cells = [LABEL[s]] + [f"{d[m].mean():.3f} $\\pm$ {d[m].std(ddof=1):.3f}" for m, _, _ in METRICS]
        rows.append(cells + [str(d.seed.nunique())])
    write_table(name, ['Strategy'] + [h for _, h, _ in METRICS] + ['n'], rows, caption, label, colspec='lcccc c')


def table_paired(df, name, caption, label, ref=None, strategies=None):
    """Paired differences against the baseline."""
    base = ref if ref is not None else df[df.strategy == 'baseline']
    rows = []
    for s in [x for x in (strategies or ORDER) if x != 'baseline' and x in set(df.strategy)]:
        cells = [LABEL[s]]
        for m, _, _ in METRICS:
            r = paired(df, base, s, m)
            cells.append('--' if r is None else f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]{sig_mark(r)} ({pfmt(r['p'])})")
        rows.append(cells)
    write_table(name, ['Strategy'] + [h for _, h, _ in METRICS], rows,
                caption + r' Mean paired difference [95\% bootstrap CI]; * = CI excludes 0; Wilcoxon p in brackets.',
                label, colspec='lllll')


def group_sizes_table(counts_dir):
    """Training and test group sizes."""
    p = os.path.join(counts_dir, 'group_counts_summary.csv')
    if not os.path.exists(p):
        print(f"  [skip] {p} not found"); return
    g = pd.read_csv(p, index_col=0)
    pick = [('n_train_before_subsampling', 'Training images before subsampling'),
            ('n_train_after_subsampling', 'Training images after subsampling (effective)'),
            ('n_val', 'Validation images'), ('n_test', 'Test images'),
            ('n_test_aligned', 'Test, aligned group'), ('n_test_conflict', 'Test, conflicting group'),
            ('conflict_malignant(fatty)', 'Conflicting: malignant (non-dense)'), ('conflict_benign(dense)', 'Conflicting: benign (dense)')]
    rows = [[lab, f"{g.loc[k, 'mean']:.0f}", f"{g.loc[k, 'std']:.0f}", f"{g.loc[k, 'min']:.0f}--{g.loc[k, 'max']:.0f}"]
            for k, lab in pick if k in g.index]
    write_table('table_group_sizes', ['Quantity', 'Mean', 'SD', 'Range over seeds'], rows,
                'Effective sample sizes over the 13 seeds.', 'tab:groupsizes', colspec='lccc')


def fig_two_axes(frames, path):
    """SAS against worst-group gap, one point per seed."""
    fig, axes = plt.subplots(1, len(frames), figsize=(6.2 * len(frames), 4.6), squeeze=False)
    for ax, (title, df) in zip(axes[0], frames):
        for s in [x for x in ORDER if x in set(df.strategy)]:
            d = df[df.strategy == s]
            ax.scatter(d.sas, d.worst_group_gap, s=16, alpha=.45, color=COLOR[s])
            ax.scatter(d.sas.mean(), d.worst_group_gap.mean(), s=90, marker='X', color=COLOR[s], edgecolor='k', linewidth=.6, label=LABEL[s])
        ax.axhline(0, color='k', lw=.6, ls=':')
        ax.set_xlabel('SAS (attribution outside the breast mask)  -- explanation axis')
        ax.set_ylabel('worst-group gap  -- shortcut-performance axis')
        ax.set_title(title, fontsize=10); ax.grid(alpha=.3)
    for ax in axes[0]:
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle('Worst-group gap against SAS: one dot per seed, X = strategy mean', fontsize=11)
    plt.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    print(f"  wrote {os.path.basename(path)}")


def fig_group_auroc(df, path):
    """Aligned against conflicting AUROC per strategy."""
    ss = [x for x in ORDER if x in set(df.strategy)]
    x = np.arange(len(ss))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for off, col, lab, c in ((-0.2, 'auroc_aligned', 'aligned group', '#1f77b4'), (0.2, 'auroc_conflict', 'conflicting group', '#d62728')):
        m = [df[df.strategy == s][col].mean() for s in ss]
        sd = [df[df.strategy == s][col].std(ddof=1) for s in ss]
        ax.bar(x + off, m, 0.4, yerr=sd, capsize=3, color=c, label=lab, alpha=.85)
    ax.axhline(0.5, color='k', ls=':', lw=.8)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[s] for s in ss], rotation=25, ha='right'); ax.set_ylabel('AUROC')
    ax.set_title('AUROC in the aligned and conflicting groups (mean $\\pm$ SD over seeds)'); ax.legend(); ax.grid(alpha=.3, axis='y')
    plt.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    print(f"  wrote {os.path.basename(path)}")


def fig_benchmark(cbis, bench, path):
    """Forest plot of the benchmark methods and the studied strategies."""
    base = cbis[cbis.strategy == 'baseline']
    pool = pd.concat([bench, cbis[cbis.strategy.isin(['distcorr', 'full'])]])
    ss = [s for s in ['adv_nonlinear', 'group_dro', 'jtt', 'mine', 'distcorr', 'full'] if s in set(pool.strategy)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, (m, title) in zip(axes, (('worst_group_gap', 'worst-group gap vs baseline'), ('auroc_all', 'AUROC vs baseline'))):
        for i, s in enumerate(ss):
            r = paired(pool, base, s, m)
            if r is None:
                continue
            ax.errorbar(r['diff'], i, xerr=[[r['diff'] - r['lo']], [r['hi'] - r['diff']]], fmt='o', color=COLOR[s], capsize=3,
                        mfc='white' if not (r['hi'] < 0 or r['lo'] > 0) else COLOR[s])
        ax.axvline(0, color='k', lw=.7, ls='--'); ax.set_title(title, fontsize=10); ax.grid(alpha=.3)
        ax.set_yticks(range(len(ss))); ax.set_yticklabels([LABEL[s] for s in ss])
    fig.suptitle('Paired difference vs baseline (95% bootstrap CI; hollow = CI includes 0)', fontsize=10)
    plt.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    print(f"  wrote {os.path.basename(path)}")


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cbis-csv'); ap.add_argument('--bench-csv'); ap.add_argument('--inbreast-csv', nargs='*', default=[])
    ap.add_argument('--counts-dir'); ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    OUT = a.out
    os.makedirs(OUT, exist_ok=True)

    consts = {}
    try:
        sys.path.insert(0, SRC_DIR)
        import core_pipeline
        for k in ('IMG_SIZE', 'BATCH_SIZE', 'MAX_EPOCHS', 'PATIENCE', 'LR', 'LAMBDA_RAMP_EPOCHS', 'MAX_LAMBDA_GRL', 'GRL_ADV_WEIGHT',
                  'GRL_REINIT_EVERY', 'MAX_WEIGHT_DISTCORR', 'MAX_WEIGHT_CAMREG', 'BIAS_STRENGTH', 'SEED_LIST'):
            if hasattr(core_pipeline, k):
                consts[k] = getattr(core_pipeline, k)
    except Exception as e:
        print(f"  (core_pipeline not importable: {e!r}; setup table will show n/a for constants)")
    gpu = None
    import json
    for mp in (os.path.join(HERE, 'main_experiment', 'run_manifest.json'), os.path.join(HERE, 'method_comparison', 'run_manifest.json')):
        if os.path.exists(mp):
            gpu = json.load(open(mp)).get('determinism', {}).get('cuda_device_name') or json.load(open(mp)).get('determinism', {}).get('gpu')
            if gpu:
                break
    print("== tables ==")
    table_setup(consts, gpu)

    cbis_p = find(a.cbis_csv, ['main_experiment/dccam_cbis_results.csv', 'dccam_cbis_results.csv', 'method_comparison/dccam_cbis_results.csv'])
    cbis = None
    if cbis_p:
        print(f"  CBIS-DDSM results: {cbis_p}")
        cbis = pd.read_csv(cbis_p).drop_duplicates(['seed', 'strategy'], keep='last')
        table_main(cbis, 'table_main_results', 'CBIS-DDSM: mean $\\pm$ SD over seeds.', 'tab:main')
        table_paired(cbis, 'table_paired', 'CBIS-DDSM: paired comparison against the baseline.', 'tab:paired')
        fig_two_axes([('CBIS-DDSM', cbis)], os.path.join(OUT, 'fig_two_axes.png'))
        fig_group_auroc(cbis, os.path.join(OUT, 'fig_group_auroc.png'))
    else:
        print("  [skip] CBIS results csv not found (--cbis-csv)")

    bench_p = find(a.bench_csv, ['method_comparison/benchmark_baselines_results.csv', 'benchmark_baselines_results.csv'])
    if bench_p and cbis is not None:
        print(f"  benchmark results: {bench_p}")
        bench = pd.read_csv(bench_p).drop_duplicates(['seed', 'strategy'], keep='last')
        base = bench[bench.strategy == 'baseline']
        ref_note = 'the benchmark file' if len(base) else 'the CBIS-DDSM baseline of the main ablation'
        if not len(base):
            base = cbis[cbis.strategy == 'baseline']
        pool = pd.concat([bench[bench.strategy != 'baseline'], cbis[cbis.strategy.isin(['distcorr', 'full'])]])
        table_paired(pool, 'table_benchmark', f'Comparison with published group-robustness baselines (reference: {ref_note}, paired on seed).',
                     'tab:benchmark', ref=base, strategies=['adv_nonlinear', 'group_dro', 'jtt', 'mine', 'distcorr', 'full'])
        print(f"  NOTE: benchmark methods are compared against {ref_note}; if that baseline came from a different run, say so in the caption.")
        fig_benchmark(cbis, bench, os.path.join(OUT, 'fig_benchmark_forest.png'))
    elif not bench_p:
        print("  [skip] benchmark csv not found (--bench-csv)")

    inb = [pd.read_csv(p) for p in a.inbreast_csv if os.path.exists(p)]
    if inb:
        idf = pd.concat(inb).drop_duplicates(['seed', 'strategy'], keep='last')
        print(f"  INbreast: {idf.seed.nunique()} seeds from {len(inb)} file(s)")
        table_main(idf, 'table_inbreast_main', f'INbreast ({idf.seed.nunique()} seeds): mean $\\pm$ SD.', 'tab:inbreast')
        table_paired(idf, 'table_inbreast_paired', f'INbreast ({idf.seed.nunique()} seeds): paired comparison against the baseline.', 'tab:inbreastpaired')
        if cbis is not None:
            fig_two_axes([('CBIS-DDSM', cbis), (f'INbreast ({idf.seed.nunique()} seeds)', idf)], os.path.join(OUT, 'fig_two_axes.png'))
    if a.counts_dir:
        group_sizes_table(a.counts_dir)
    print(f"\nDone. Everything is in {OUT}")


if __name__ == '__main__':
    main()
