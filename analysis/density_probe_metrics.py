"""Balanced accuracy and AUROC of the density probe, recomputed from saved checkpoints.

Run: python analysis/density_probe_metrics.py --dir outputs/calibration_experiment
"""
import os
import sys

# Make src/ and analysis/ importable when the script is run from any folder.
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', d) for d in ('src', 'analysis')]

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def probe_metrics(feats, density, seed, n_boot=20, mlp=True):
    """Linear and MLP density probes with the same bootstrap splits as the pipeline."""
    feats, density = np.asarray(feats), np.asarray(density).astype(int)
    rng = np.random.RandomState(seed)
    n = len(feats)
    acc = {k: [] for k in ('lin_acc', 'lin_bal', 'lin_auc', 'floor', 'mlp_bal', 'mlp_auc')}
    for b in range(n_boot):
        idx = rng.permutation(n)
        split = int(n * 0.7)
        tr, te = idx[:split], idx[split:]
        if len(np.unique(density[tr])) < 2 or len(np.unique(density[te])) < 2:
            continue
        lin = LogisticRegression(max_iter=1000).fit(feats[tr], density[tr])
        pred = lin.predict(feats[te])
        acc['lin_acc'].append(float((pred == density[te]).mean()))
        acc['lin_bal'].append(float(balanced_accuracy_score(density[te], pred)))
        acc['lin_auc'].append(float(roc_auc_score(density[te], lin.predict_proba(feats[te])[:, 1])))
        frac = float(density[te].mean())
        acc['floor'].append(max(frac, 1 - frac))
        if mlp:
            sc = StandardScaler().fit(feats[tr])
            m = MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000, random_state=b, early_stopping=True,
                              n_iter_no_change=15, validation_fraction=0.15).fit(sc.transform(feats[tr]), density[tr])
            acc['mlp_bal'].append(float(balanced_accuracy_score(density[te], m.predict(sc.transform(feats[te])))))
            acc['mlp_auc'].append(float(roc_auc_score(density[te], m.predict_proba(sc.transform(feats[te]))[:, 1])))
    mean = lambda v: float(np.mean(v)) if v else float('nan')
    out = {'probe_lin_acc': mean(acc['lin_acc']), 'probe_lin_bal_acc': mean(acc['lin_bal']),
           'probe_lin_auc': mean(acc['lin_auc']), 'probe_majority_floor': mean(acc['floor']),
           'probe_mlp_bal_acc': mean(acc['mlp_bal']), 'probe_mlp_auc': mean(acc['mlp_auc'])}
    out['probe_lin_acc_minus_floor'] = out['probe_lin_acc'] - out['probe_majority_floor']
    return out


def self_test():
    """Check the probe code on synthetic data."""
    rng = np.random.RandomState(0)
    n, d = 1000, 32
    dens = (rng.rand(n) < 0.85).astype(int)
    informative = rng.randn(n, d) + 1.2 * dens[:, None]
    noise = rng.randn(n, d)
    a, b = probe_metrics(informative, dens, 0), probe_metrics(noise, dens, 0)
    print("informative features :", {k: round(v, 3) for k, v in a.items()})
    print("pure-noise features  :", {k: round(v, 3) for k, v in b.items()})
    assert b['probe_lin_auc'] < 0.6 and abs(b['probe_lin_bal_acc'] - 0.5) < 0.12, "noise probe should be near chance"
    assert a['probe_lin_auc'] > 0.9 and a['probe_lin_bal_acc'] > b['probe_lin_bal_acc']
    assert abs(b['probe_lin_acc'] - b['probe_majority_floor']) < 0.06, "noise probe accuracy should sit at the floor"
    print("self-test OK: plain accuracy of the noise probe is at the majority floor, balanced accuracy / AUROC are at chance.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=None, help="results folder with results.csv and ckpt/*.pt (calibration_experiment.py output)")
    ap.add_argument('--no-mlp', action='store_true')
    ap.add_argument('--n-boot', type=int, default=20, help="bootstrap resamples (20 = same as core_pipeline)")
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return
    if not args.dir:
        ap.error('--dir is required (or use --self-test)')

    import torch
    import experiment_utils as rc
    import core_pipeline

    res = pd.read_csv(os.path.join(args.dir, 'results.csv'))
    meta, pcol = rc.load_cbis_meta()
    rows = []
    for _, r in res.iterrows():
        run_id = f"{rc._safe(r['tag'])}__{rc._safe(r['strategy'])}__seed{int(r['seed'])}"
        ck = os.path.join(args.dir, 'ckpt', run_id + '.pt')
        if not os.path.exists(ck):
            print(f"  [skip] no checkpoint for {run_id}")
            continue
        if str(r['strategy']) == 'grl_tuned':
            print(f"  [skip] {run_id}: tuned GRL may use an MLP adversary head (state dict differs)")
            continue
        seed = int(r['seed'])
        _, _, test_df = rc.make_split(meta, pcol, seed)
        model = core_pipeline.DCCAMNet(use_grl=False).to(core_pipeline.device)
        model.load_state_dict(torch.load(ck, map_location=core_pipeline.device))
        model.eval()
        loader = core_pipeline.make_loader(core_pipeline.CBISDataset(test_df, augment=False), core_pipeline.BATCH_SIZE, shuffle=False)
        feats, density = core_pipeline.extract_features_and_labels(model, loader)
        m = probe_metrics(feats, density, seed, n_boot=args.n_boot, mlp=not args.no_mlp)
        m.update({'tag': r['tag'], 'ref_tag': r.get('ref_tag', r['tag']), 'strategy': r['strategy'], 'seed': seed,
                  'results_csv_probe_linear_mean': r['probe_linear_mean']})
        rows.append(m)
        print(f"  {run_id}: acc {m['probe_lin_acc']:.4f} (csv {r['probe_linear_mean']:.4f})  "
              f"bal.acc {m['probe_lin_bal_acc']:.4f}  AUROC {m['probe_lin_auc']:.4f}  floor {m['probe_majority_floor']:.4f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        print("nothing computed -- no checkpoints found (calibration_experiment.py saves them unless run with --no-ckpt)")
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.dir, 'probe_balanced_per_run.csv'), index=False)
    d = (df['probe_lin_acc'] - df['results_csv_probe_linear_mean']).abs().max()
    print(f"\nCHECK: max |recomputed plain accuracy - results.csv probe_linear_mean| = {d:.5f}  "
          f"({'OK, checkpoints match results.csv' if d < 0.005 else 'MISMATCH: checkpoints are not the models behind results.csv'})")
    cols = [c for c in df.columns if c.startswith('probe_')]
    s = df.groupby(['tag', 'strategy'])[cols].agg(['mean', 'std'])
    s.columns = ['_'.join(c) for c in s.columns]
    s['n_seeds'] = df.groupby(['tag', 'strategy']).size()
    s.reset_index().to_csv(os.path.join(args.dir, 'probe_balanced_summary.csv'), index=False)
    paired = rc.paired_vs_baseline(df, metrics=tuple(cols))
    paired.to_csv(os.path.join(args.dir, 'probe_balanced_paired.csv'), index=False)
    with pd.option_context('display.width', 220, 'display.max_columns', 30):
        print(s.reset_index()[['tag', 'strategy', 'probe_lin_acc_mean', 'probe_majority_floor_mean',
                               'probe_lin_bal_acc_mean', 'probe_lin_auc_mean', 'n_seeds']].round(4).to_string(index=False))
        print("\nPaired vs baseline (balanced accuracy / AUROC):")
        print(paired[paired.metric.isin(['probe_lin_bal_acc', 'probe_lin_auc', 'probe_lin_acc_minus_floor'])]
              .round(4).to_string(index=False))


if __name__ == '__main__':
    main()
