"""Loss-weight sweeps for distcorr and camreg, and the GRL settings study (5 seeds).

Run: python experiments/weight_and_grl_sweeps.py --study distcorr|camreg|grl
"""
import os
import sys

# Make src/ and analysis/ importable when the script is run from any folder.
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', d) for d in ('src', 'analysis')]

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import experiment_utils as rc

GRL_CONFIGS = {
    'grl_main':      (5.0, 2.0, 4, 'linear'),
    'grl_noreinit':  (5.0, 2.0, 0, 'linear'),
    'grl_l1':        (1.0, 1.0, 0, 'linear'),
    'grl_l03':       (0.3, 1.0, 0, 'linear'),
    'grl_l01':       (0.1, 1.0, 0, 'linear'),
    'grl_l1_reinit': (1.0, 1.0, 4, 'linear'),
    'grl_mlp_l1':    (1.0, 1.0, 0, 'mlp'),
    'grl_mlp_l03':   (0.3, 1.0, 0, 'mlp'),
}


def build_plan(args):
    """List of (tag, strategy, overrides) runs for the chosen study."""
    plan = []
    if args.study == 'distcorr':
        for w in args.weights or [1.0, 2.0, 4.0, 8.0, 12.0, 16.0]:
            plan.append(dict(tag=f'distcorr_w{w:g}', label='distcorr', strat='distcorr',
                             ov=dict(MAX_WEIGHT_DISTCORR=float(w)), net={}, param=float(w), config=f'w={w:g}'))
    elif args.study == 'camreg':
        for w in args.weights or [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
            plan.append(dict(tag=f'{args.cam_strategy}_cam{w:g}', label=args.cam_strategy, strat=args.cam_strategy,
                             ov=dict(MAX_WEIGHT_CAMREG=float(w)), net={}, param=float(w), config=f'cam={w:g}'))
    else:
        names = args.configs or list(GRL_CONFIGS)
        for n in names:
            lam, wt, rein, adv = GRL_CONFIGS[n]
            plan.append(dict(tag=n, label='grl_only', strat='grl_only',
                             ov=dict(MAX_LAMBDA_GRL=lam, GRL_ADV_WEIGHT=wt, GRL_REINIT_EVERY=rein),
                             net=dict(adv=adv), param=float('nan'), config=n))
    return plan


def plot_sweep(df, study, path):
    """Paired change against the baseline as the setting varies."""
    metrics = ['auroc_all', 'worst_group_gap', 'sas', 'probe_linear_mean']
    base = df[df.tag == 'baseline_ref']
    arm = df[df.tag != 'baseline_ref']
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
    if study in ('distcorr', 'camreg'):
        for ax, m in zip(axes, metrics):
            g = arm.groupby('param')[m].agg(['mean', 'std']).reset_index().sort_values('param')
            ax.errorbar(g['param'], g['mean'], yerr=g['std'], marker='o', capsize=3)
            ax.axhline(base[m].mean(), color='gray', ls='--', label='baseline (same seeds)')
            ax.set_xscale('log'); ax.set_xlabel('weight (log scale)'); ax.set_title(rc.METRIC_LABELS[m]); ax.grid(alpha=.3)
        axes[0].legend(fontsize=8)
    else:
        for ax, m in zip(axes, metrics):
            g = arm.groupby('config')[m].agg(['mean', 'std']).reindex(list(GRL_CONFIGS)).dropna(how='all')
            ax.bar(range(len(g)), g['mean'], yerr=g['std'], capsize=3)
            ax.axhline(base[m].mean(), color='gray', ls='--')
            ax.set_xticks(range(len(g))); ax.set_xticklabels(g.index, rotation=60, ha='right', fontsize=7)
            ax.set_title(rc.METRIC_LABELS[m]); ax.grid(alpha=.3, axis='y')
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def grl_selection(df, out, tol=0.02):
    """Pick the GRL setting on validation data only (AUROC cost <= tol, lowest gap)."""
    base = df[df.tag == 'baseline_ref']
    b_val = base['val_auroc'].mean()
    g = df[df.tag != 'baseline_ref'].groupby('config').agg(
        val_auroc=('val_auroc', 'mean'), val_gap=('val_gap', 'mean'),
        test_auroc=('auroc_all', 'mean'), test_gap=('worst_group_gap', 'mean'),
        test_sas=('sas', 'mean'), test_probe=('probe_linear_mean', 'mean'), n_seeds=('seed', 'nunique')).reset_index()
    g['eligible(val_auroc>=baseline-%.2f)' % tol] = g['val_auroc'] >= b_val - tol
    elig = g[g.iloc[:, -1]]
    g['selected_by_validation'] = False
    if len(elig):
        best = elig.sort_values('val_gap').iloc[0]['config']
        g.loc[g['config'] == best, 'selected_by_validation'] = True
        lam, wt, rein, adv = GRL_CONFIGS[best]
        print(f"\nGRL config selected on VALIDATION worst-group gap: {best}")
        print(f"  -> pass to calibration_experiment.py / inbreast_validation.py:  --grl-lambda {lam} --grl-weight {wt} --grl-reinit {rein} --grl-adv {adv}")
    else:
        print("\nNo GRL config stayed within the validation-AUROC tolerance of the baseline.")
    g['baseline_val_auroc'] = b_val
    g.to_csv(os.path.join(out['root'], 'grl_selection.csv'), index=False)
    print(g.round(4).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap, default_seeds=5)
    ap.add_argument('--study', choices=['distcorr', 'camreg', 'grl'], required=True)
    ap.add_argument('--weights', type=float, nargs='+', default=None, help="override the weight grid (distcorr / camreg)")
    ap.add_argument('--cam-strategy', choices=['camreg', 'full'], default='camreg',
                    help="camreg study: sweep the CAM weight inside 'camreg' (default) or inside 'full'")
    ap.add_argument('--configs', nargs='+', choices=list(GRL_CONFIGS), default=None, help="grl study: subset of configs")
    ap.add_argument('--save-ckpt', action='store_true')
    args = ap.parse_args(argv)
    rc.apply_smoke(args)

    seeds = rc.resolve_seeds(args)
    plan = build_plan(args)
    n_runs = (len(plan) + 1) * len(seeds)
    print(f"study={args.study} seeds={seeds}\nplan: " + ', '.join(p['tag'] for p in plan))
    rc.plan_banner(n_runs)
    if args.dry_run:
        return

    study_dir = {'distcorr': 'distcorr_weight_sweep', 'camreg': 'camreg_weight_sweep', 'grl': 'grl_settings'}[args.study]
    out = rc.make_out_dir(study_dir, smoke=args.smoke)
    rc.write_manifest(out, 'weight_and_grl_sweeps.py', args, {'seeds': seeds, 'plan': [p['tag'] for p in plan]})
    meta, pcol = rc.load_cbis_meta()
    store = rc.ResultStore(os.path.join(out['root'], 'results.csv'))

    for seed in seeds:
        train_df, val_df, test_df = rc.make_split(meta, pcol, seed)
        train_used, info = rc.build_training_set(train_df, seed, 'main')
        print(f"\n#### seed {seed}: train {info['n_train_before']} -> {info['n_train']} (r={info['r_train']:.3f})")
        items = [dict(tag='baseline_ref', label='baseline', strat='baseline', ov={}, net={},
                      param=float('nan'), config='baseline')] + plan
        for it in items:
            if store.has(it['tag'], it['label'], seed):
                print(f"  [{it['tag']} | {it['label']} | seed {seed}] already done -- skipped")
                continue
            row = rc.train_and_evaluate(
                it['label'], it['strat'], train_used, val_df, test_df, seed, out,
                tag=it['tag'], ref_tag='baseline_ref', overrides=it['ov'], net=it['net'],
                save_ckpt=args.save_ckpt, run_probe=not args.no_probe, info=info,
                extra={'study': args.study, 'param': it['param'], 'config': it['config']})
            store.add(row)

    df = store.df()
    rc.write_summaries(df, out)
    plot_sweep(df, args.study, os.path.join(out['root'], f'fig_{args.study}_sweep.png'))
    if args.study == 'grl':
        grl_selection(df, out)
    print(f"\nDone. Results in: {out['root']}")


if __name__ == '__main__':
    main()
