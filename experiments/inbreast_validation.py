"""INbreast (full-field digital mammography) replication of the main experiment, 13 seeds.

Run: python experiments/inbreast_validation.py --data-root <INbreast folder>
"""
import os
import sys

# Make src/ and analysis/ importable when the script is run from any folder.
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', d) for d in ('src', 'analysis')]

import argparse

import pandas as pd

import experiment_utils as rc
import core_pipeline
from paths import INBREAST_DIR


def find_data_root(cli_value):
    """INbreast folder from --data-root, otherwise the default under data/."""
    if cli_value:
        return cli_value
    return INBREAST_DIR


def find_cache(out_root):
    """Reuse an existing INbreast image cache if one is found."""
    cands = [os.path.join(rc.HERE, 'inbreast_image_cache.pkl'),
             os.path.join(rc.HERE, 'method_comparison', 'inbreast_image_cache.pkl'),
             os.path.join(rc.HERE, 'main_experiment', 'inbreast_image_cache.pkl'),
             os.path.join(out_root, 'inbreast_image_cache.pkl')]
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[-1]


def cohort_tables(meta, pcol, seeds, out):
    """Cohort summary and per-seed split / group sizes (no training)."""
    lines = []
    lines.append(f"INbreast usable images/cases : {len(meta)}")
    lines.append(f"Distinct patients            : {meta[pcol].nunique()}")
    lines.append(f"Malignant / benign           : {int(meta['malignant'].sum())} / {int((1 - meta['malignant']).sum())}")
    lines.append(f"Density binary (>=3 dense)   : dense {int(meta['density_binary'].sum())} / not dense {int((1 - meta['density_binary']).sum())}")
    lines.append(f"Natural point-biserial r (malignant vs density_binary) = {rc.phi(meta):.3f}")
    lines.append("Acquisition                  : full-field DIGITAL mammography (not digitized film)")
    dens = pd.crosstab(meta['density'], meta['malignant']).rename(columns={0: 'benign', 1: 'malignant'})
    dens.to_csv(os.path.join(out['root'], 'cohort_density_counts.csv'))
    lines.append("\nBI-RADS/ACR density x outcome:\n" + dens.to_string())
    rows = []
    for s in seeds:
        tr, va, te = rc.make_split(meta, pcol, s)
        used, info = rc.build_training_set(tr, s, 'main')
        gc_, _, _ = rc.group_counts(te)
        rows.append({'seed': s, 'n_train_before': len(tr), 'n_train_used': len(used), 'r_train': info['r_train'],
                     'n_val': len(va), **gc_})
    pc = pd.DataFrame(rows)
    pc.to_csv(os.path.join(out['root'], 'per_seed_group_counts.csv'), index=False)
    lines.append("\nPer-seed split sizes after the r~0.6 subsampling (mean over seeds):\n"
                 + pc.drop(columns='seed').mean().round(1).to_string())
    txt = "\n".join(lines)
    with open(os.path.join(out['root'], 'cohort_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(txt)
    print("\n" + txt + "\n")
    small = pc['n_conflict_malignant'].min()
    if small < 5:
        print(f"[note] the smallest conflicting-malignant group has only {int(small)} cases -> AUROC_conflict is very noisy "
              f"on INbreast; report group sizes next to every INbreast number.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_common_args(ap, default_seeds=None)
    ap.add_argument('--data-root', default=None, help="folder containing AllDICOMs/ and INbreast.csv/.xls")
    ap.add_argument('--arms', nargs='+', default=['baseline', 'grl_only', 'distcorr', 'camreg', 'full'],
                    help="any of baseline grl_only distcorr camreg full grl_tuned (mi_min runs only in method_comparison.py)")
    ap.add_argument('--conditions', nargs='+', default=['main'], help="main (r~0.6, the paper) and/or natural (no subsampling)")
    ap.add_argument('--summary-only', action='store_true', help="write cohort tables and stop")
    ap.add_argument('--no-ckpt', action='store_true')
    ap.add_argument('--no-analysis', action='store_true')
    rc.add_grl_tuned_args(ap)
    args = ap.parse_args(argv)
    rc.apply_smoke(args)

    seeds = rc.resolve_seeds(args)
    print(f"arms={args.arms} conditions={args.conditions} seeds={seeds}")
    rc.plan_banner(len(args.arms) * len(args.conditions) * len(seeds))
    if args.dry_run:
        return

    data_root = find_data_root(args.data_root)
    if not data_root or not os.path.isdir(data_root):
        sys.exit("STOPPED: give --data-root \"<...>/INbreast Release 1.0\" (the folder with AllDICOMs/ and INbreast.csv/.xls).")

    out = rc.make_out_dir('inbreast_validation', smoke=args.smoke)
    rc.write_manifest(out, 'inbreast_validation.py', args, {'seeds': seeds, 'data_root': data_root})

    import external_validation as ev
    rc.apply_determinism()

    meta, pcol = ev.build_metadata_inbreast(data_root)
    meta = meta.reset_index(drop=True)
    cache_file = find_cache(out['root'])
    print(f"INbreast image cache: {cache_file}")
    ev.build_or_load_external_cache(meta['filepath'].unique().tolist(), cache_file)
    core_pipeline.IMAGE_CACHE.update(ev.legacy_pipeline.IMAGE_CACHE)
    missing = [f for f in meta['filepath'].unique() if f not in core_pipeline.IMAGE_CACHE]
    if missing:
        sys.exit(f"STOPPED: {len(missing)} INbreast images are missing from the cache (first: {missing[0]}).")

    cohort_tables(meta, pcol, seeds, out)
    if args.summary_only:
        return

    store = rc.ResultStore(os.path.join(out['root'], 'results.csv'))
    for seed in seeds:
        train_df, val_df, test_df = rc.make_split(meta, pcol, seed)
        for cond in args.conditions:
            train_used, info = rc.build_training_set(train_df, seed, cond)
            print(f"\n#### INbreast | seed {seed} | {cond}: train {info['n_train_before']} -> {info['n_train']} "
                  f"(r={info['r_train']:.3f}, malignant {info['n_train_malignant']})")
            for arm in args.arms:
                tag = f'inbreast_{cond}'
                if store.has(tag, arm, seed):
                    print(f"  [{tag} | {arm} | seed {seed}] already done -- skipped")
                    continue
                strat, ov, net = rc.resolve_arm(arm, args)
                row = rc.train_and_evaluate(
                    arm, strat, train_used, val_df, test_df, seed, out, tag=tag, ref_tag=tag,
                    overrides=ov, net=net, save_ckpt=not args.no_ckpt, mask_variants=True,
                    run_probe=not args.no_probe, info=info, extra={'dataset': 'INbreast'})
                store.add(row)

    df = store.df()
    rc.write_summaries(df, out)
    if not args.no_analysis:
        try:
            import calibration_analysis
            calibration_analysis.run_calibration(out['root'])
        except Exception as e:
            print(f"[warning] automatic analysis failed ({e!r}); run: python calibration_analysis.py calibration --dir \"{out['root']}\"")
    print(f"\nDone. Results in: {out['root']}")


if __name__ == '__main__':
    main()
