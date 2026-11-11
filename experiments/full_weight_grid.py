"""Full model on a grid of distance-correlation x CAM weights, against a same-seed baseline.

Run: python experiments/full_weight_grid.py
"""
import os
import sys

# Make src/ and analysis/ importable when the script is run from any folder.
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', d) for d in ('src', 'analysis')]

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import experiment_utils as rc
import core_pipeline

STUDY_DIR = 'full_weight_grid'
METRICS = ('worst_group_gap', 'sas', 'auroc_all')
AUROC_COST_TOLERANCE = 0.03


def cell_tag(dc, cam):
    return f'full_dc{dc:g}_cam{cam:g}'


def build_plan(dc_weights, cam_weights):
    """One run per (distance-correlation weight, CAM weight) cell."""
    plan = []
    for dc in dc_weights:
        for cam in cam_weights:
            plan.append(dict(tag=cell_tag(dc, cam), dc=float(dc), cam=float(cam),
                             ov=dict(MAX_WEIGHT_DISTCORR=float(dc), MAX_WEIGHT_CAMREG=float(cam))))
    return plan


def conclusion_check(df, plan, out_root):
    """For each cell: does the full model lower the gap and SAS, and at what AUROC cost."""
    paired = rc.paired_vs_baseline(df, metrics=METRICS)
    rows = []
    for it in plan:
        p = paired[(paired.tag == it['tag']) & (paired.strategy == 'full')]
        if p.empty:
            continue
        rec = {'w_distcorr': it['dc'], 'w_camreg': it['cam'],
               'is_paper_setting': (it['dc'] == core_pipeline.MAX_WEIGHT_DISTCORR and it['cam'] == core_pipeline.MAX_WEIGHT_CAMREG),
               'n_seeds': int(p['n'].iloc[0])}
        for m in METRICS:
            r = p[p.metric == m]
            if r.empty:
                continue
            r = r.iloc[0]
            rec[f'{m}_diff'], rec[f'{m}_ci_lo'], rec[f'{m}_ci_hi'] = r.mean_diff_vs_baseline, r.ci_lo, r.ci_hi
            rec[f'{m}_wilcoxon_p'] = r.wilcoxon_p
        rec['lowers_gap'] = 'yes' if rec.get('worst_group_gap_ci_hi', 1) < 0 else 'NO'
        rec['lowers_sas'] = 'yes' if rec.get('sas_ci_hi', 1) < 0 else 'NO'
        rec['auroc_cost_small'] = 'yes' if rec.get('auroc_all_diff', -1) > -AUROC_COST_TOLERANCE else 'NO'
        rec['auroc_cost_significant'] = 'yes' if rec.get('auroc_all_ci_hi', 1) < 0 else 'no'
        rows.append(rec)
    chk = pd.DataFrame(rows)
    chk.to_csv(os.path.join(out_root, 'grid_conclusion_check.csv'), index=False)
    if len(chk):
        n = len(chk)
        print("\n==== CLAIM CHECK per grid cell (95% bootstrap CI of paired diff vs same-seed baseline) ====")
        show = ['w_distcorr', 'w_camreg', 'worst_group_gap_diff', 'worst_group_gap_ci_lo', 'worst_group_gap_ci_hi',
                'sas_diff', 'auroc_all_diff', 'lowers_gap', 'lowers_sas', 'auroc_cost_small']
        with pd.option_context('display.width', 220, 'display.max_columns', 30):
            print(chk[[c for c in show if c in chk.columns]].round(3).to_string(index=False))
        print(f"\ncells where 'full' lowers the gap (CI < 0): {(chk.lowers_gap == 'yes').sum()}/{n}")
        print(f"cells where 'full' lowers SAS      (CI < 0): {(chk.lowers_sas == 'yes').sum()}/{n}")
        print(f"cells with AUROC change > -{AUROC_COST_TOLERANCE}: {(chk.auroc_cost_small == 'yes').sum()}/{n}")
    return chk


def heatmaps(chk, path, dc_weights, cam_weights):
    """Heat maps of the paired changes over the weight grid."""
    if chk.empty:
        return
    titles = {'worst_group_gap': 'Worst-group gap: full - baseline', 'sas': 'SAS: full - baseline',
              'auroc_all': 'AUROC (all): full - baseline'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, m in zip(axes, METRICS):
        grid = np.full((len(cam_weights), len(dc_weights)), np.nan)
        sig = np.zeros_like(grid, dtype=bool)
        for _, r in chk.iterrows():
            i, j = cam_weights.index(r.w_camreg), dc_weights.index(r.w_distcorr)
            grid[i, j] = r[f'{m}_diff']
            sig[i, j] = (r[f'{m}_ci_hi'] < 0) or (r[f'{m}_ci_lo'] > 0)
        vmax = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1
        im = ax.imshow(grid, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower', aspect='auto')
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:+.3f}" + ("*" if sig[i, j] else ""), ha='center', va='center', fontsize=9)
        ax.set_xticks(range(len(dc_weights))); ax.set_xticklabels([f'{w:g}' for w in dc_weights])
        ax.set_yticks(range(len(cam_weights))); ax.set_yticklabels([f'{w:g}' for w in cam_weights])
        ax.set_xlabel('DistCorr weight (MAX_WEIGHT_DISTCORR)'); ax.set_ylabel('CAM weight (MAX_WEIGHT_CAMREG)')
        ax.set_title(titles[m], fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Full model over the weight grid (mean paired difference vs same-seed baseline; * = 95% bootstrap CI excludes 0)",
                 fontsize=10)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap, default_seeds=5)
    ap.add_argument('--dc-weights', type=float, nargs='+', default=[2.0, 8.0, 16.0])
    ap.add_argument('--cam-weights', type=float, nargs='+', default=[0.5, 2.0, 8.0])
    ap.add_argument('--summarize-only', action='store_true', help="rebuild tables and the figure from results.csv, no training")
    ap.add_argument('--save-ckpt', action='store_true')
    args = ap.parse_args(argv)
    rc.apply_smoke(args)
    if args.smoke:
        args.dc_weights, args.cam_weights = args.dc_weights[:1], args.cam_weights[:1]

    seeds = rc.resolve_seeds(args)
    plan = build_plan(args.dc_weights, args.cam_weights)
    print(f"grid: DistCorr weights {args.dc_weights} x CAM weights {args.cam_weights}; seeds {seeds}")
    print(f"paper setting = DistCorr {core_pipeline.MAX_WEIGHT_DISTCORR:g} x CAM {core_pipeline.MAX_WEIGHT_CAMREG:g}"
          + ("" if any(p['dc'] == core_pipeline.MAX_WEIGHT_DISTCORR and p['cam'] == core_pipeline.MAX_WEIGHT_CAMREG for p in plan)
             else "  (NOT in this grid -- add it so the paper's own setting is included)"))
    rc.plan_banner((len(plan) + 1) * len(seeds))
    if args.dry_run:
        return

    out = rc.make_out_dir(STUDY_DIR, smoke=args.smoke)
    rc.write_manifest(out, 'full_weight_grid.py', args, {'seeds': seeds, 'plan': [p['tag'] for p in plan]})
    store = rc.ResultStore(os.path.join(out['root'], 'results.csv'))

    if not args.summarize_only:
        meta, pcol = rc.load_cbis_meta()
        for seed in seeds:
            train_df, val_df, test_df = rc.make_split(meta, pcol, seed)
            train_used, info = rc.build_training_set(train_df, seed, 'main')
            print(f"\n#### seed {seed}: train {info['n_train_before']} -> {info['n_train']} (r={info['r_train']:.3f})")
            items = [dict(tag='baseline_ref', label='baseline', ov={}, dc=np.nan, cam=np.nan)] + \
                    [dict(tag=p['tag'], label='full', ov=p['ov'], dc=p['dc'], cam=p['cam']) for p in plan]
            for it in items:
                if store.has(it['tag'], it['label'], seed):
                    print(f"  [{it['tag']} | {it['label']} | seed {seed}] already done -- skipped")
                    continue
                row = rc.train_and_evaluate(
                    it['label'], it['label'], train_used, val_df, test_df, seed, out,
                    tag=it['tag'], ref_tag='baseline_ref', overrides=it['ov'],
                    save_ckpt=args.save_ckpt, run_probe=not args.no_probe, info=info,
                    extra={'study': 'full_weight_grid', 'w_distcorr': it['dc'], 'w_camreg': it['cam']})
                store.add(row)

    df = store.df()
    if df.empty:
        print("no results yet")
        return
    rc.write_summaries(df, out)
    chk = conclusion_check(df, plan, out['root'])
    heatmaps(chk, os.path.join(out['root'], 'fig_full_weight_grid.png'), args.dc_weights, args.cam_weights)
    print(f"\nDone. Results in: {out['root']}")


if __name__ == '__main__':
    main()
