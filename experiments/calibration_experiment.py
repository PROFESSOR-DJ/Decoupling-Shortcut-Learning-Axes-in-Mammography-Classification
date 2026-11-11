"""13-seed rerun of the main experiment that also saves probabilities, checkpoints and
SAS under dilated/eroded masks (calibration table, mask check, group sizes).

Run: python experiments/calibration_experiment.py
"""
import os
import sys

# Make src/ and analysis/ importable when the script is run from any folder.
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', d) for d in ('src', 'analysis')]

import argparse

import experiment_utils as rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap, default_seeds=None)
    ap.add_argument('--arms', nargs='+', default=['baseline', 'distcorr', 'camreg', 'full'],
                    help="any of baseline grl_only distcorr camreg full grl_tuned (mi_min runs only in method_comparison.py)")
    ap.add_argument('--no-ckpt', action='store_true', help="do not save model checkpoints")
    ap.add_argument('--no-analysis', action='store_true', help="skip the automatic calibration analysis at the end")
    rc.add_grl_tuned_args(ap)
    args = ap.parse_args(argv)
    rc.apply_smoke(args)

    seeds = rc.resolve_seeds(args)
    print(f"arms={args.arms} seeds={seeds}")
    rc.plan_banner(len(args.arms) * len(seeds))
    if args.dry_run:
        return

    out = rc.make_out_dir('calibration_experiment', smoke=args.smoke)
    rc.write_manifest(out, 'calibration_experiment.py', args, {'seeds': seeds})
    meta, pcol = rc.load_cbis_meta()
    store = rc.ResultStore(os.path.join(out['root'], 'results.csv'))

    for seed in seeds:
        train_df, val_df, test_df = rc.make_split(meta, pcol, seed)
        train_used, info = rc.build_training_set(train_df, seed, 'main')
        print(f"\n#### seed {seed}: train {info['n_train_before']} -> {info['n_train']} (r={info['r_train']:.3f})")
        for arm in args.arms:
            if store.has('main', arm, seed):
                print(f"  [main | {arm} | seed {seed}] already done -- skipped")
                continue
            strat, ov, net = rc.resolve_arm(arm, args)
            row = rc.train_and_evaluate(
                arm, strat, train_used, val_df, test_df, seed, out, tag='main', ref_tag='main',
                overrides=ov, net=net, save_ckpt=not args.no_ckpt, mask_variants=True,
                run_probe=not args.no_probe, info=info)
            store.add(row)

    df = store.df()
    rc.write_summaries(df, out)
    if not args.no_analysis:
        try:
            import calibration_analysis
            calibration_analysis.run_calibration(out['root'])
        except Exception as e:
            print(f"[warning] automatic analysis failed ({e!r}); results.csv and probs/ are safe -- "
                  f"run: python calibration_analysis.py calibration --dir \"{out['root']}\"")
    print(f"\nDone. Results in: {out['root']}")


if __name__ == '__main__':
    main()
