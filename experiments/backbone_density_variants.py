"""Robustness: DenseNet-121, ResNet-50, clinical (C/D) density split and ordinal density.

Run: python experiments/backbone_density_variants.py --seeds 5
"""
import os
import sys

# Make src/ and analysis/ importable when the script is run from any folder.
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', d) for d in ('src', 'analysis')]

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import json
import time
import argparse
import platform
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import pointbiserialr, wilcoxon
from sklearn.model_selection import GroupShuffleSplit

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

import core_pipeline

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

warnings.filterwarnings('ignore', message='.*Precision loss occurred.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, module=r'scipy\.stats\._wilcoxon')

OUTPUT_SUBDIR = 'backbone_density_variants'
REFERENCE = 'resnet18_ref'

VARIANTS = {
    'densenet121':     dict(label='DenseNet-121',            backbone='densenet121',
                            split='median',   ordinal=False),
    'resnet50':        dict(label='ResNet-50',               backbone='resnet50',
                            split='median',   ordinal=False),
    'resnet34':        dict(label='ResNet-34',               backbone='resnet34',
                            split='median',   ordinal=False),
    'clinical_split':  dict(label='Clinical split (BI-RADS C/D)', backbone='resnet18',
                            split='clinical', ordinal=False),
    'ordinal_density': dict(label='Ordinal density signal',  backbone='resnet18',
                            split='median',   ordinal=True),
}
VARIANT_ORDER = ['densenet121', 'resnet50', 'resnet34', 'clinical_split', 'ordinal_density']
DEFAULT_VARIANTS = ['densenet121', 'resnet50', 'clinical_split', 'ordinal_density']
DEFAULT_ARMS = ['baseline', 'distcorr', 'camreg', 'full']
ORDINAL_ARMS = {'baseline', 'distcorr', 'full'}
PLOT_LABELS = dict(VARIANTS)
PLOT_LABELS[REFERENCE] = dict(label='ResNet-18 (paper)')

METRICS = [
    ('worst_group_gap',   'Worst-group gap',  'lower'),
    ('sas',               'SAS',              'lower'),
    ('probe_linear_mean', 'Density probe',    'lower'),
    ('auroc_all',         'AUROC (all test)', 'higher'),
]
METRIC_COLS = ['auroc_all', 'auroc_aligned', 'auroc_conflict', 'worst_group_gap', 'sas',
               'probe_linear_mean', 'probe_linear_std', 'probe_mlp_mean', 'probe_mlp_std',
               'auroc_cc', 'auroc_mlo']
CHECKPOINT_COLS = (['variant', 'variant_label', 'backbone', 'split', 'ordinal', 'seed',
                    'strategy', 'n_train_used', 'achieved_r', 'train_malignant_frac']
                   + METRIC_COLS + ['train_minutes', 'source'])


def determinism_snapshot():
    return {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


class _DenseBackbone(nn.Module):

    """DenseNet-121 trunk with its last block exposed as layer4."""
    def __init__(self, pretrained):
        super().__init__()
        w = tv_models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        f = tv_models.densenet121(weights=w).features
        self.stem = nn.Sequential(f.conv0, f.norm0, f.relu0, f.pool0,
                                  f.denseblock1, f.transition1,
                                  f.denseblock2, f.transition2,
                                  f.denseblock3, f.transition3)
        self.layer4 = nn.Sequential(f.denseblock4, f.norm5, nn.ReLU(inplace=False))
        self.feat_dim = 1024

    def forward(self, x):
        return self.layer4(self.stem(x))


_ACTIVE = {'backbone': 'resnet18', 'pretrained': True}


class VariantNet(core_pipeline.DCCAMNet):

    """DCCAMNet with a selectable GAP + linear-head backbone."""
    def __init__(self, use_grl=False):
        nn.Module.__init__(self)
        name, pre = _ACTIVE['backbone'], _ACTIVE['pretrained']
        if name == 'densenet121':
            self.backbone = _DenseBackbone(pre)
            self.feat_dim = self.backbone.feat_dim
        else:
            ctor = {'resnet34': (tv_models.resnet34, tv_models.ResNet34_Weights.IMAGENET1K_V1),
                    'resnet50': (tv_models.resnet50, tv_models.ResNet50_Weights.IMAGENET1K_V1),
                    }[name]
            b = ctor[0](weights=ctor[1] if pre else None)
            self.feat_dim = b.fc.in_features
            b.fc = nn.Identity()
            b.avgpool = nn.Identity()
            self.backbone = b
        self.main_head = nn.Linear(self.feat_dim, 1)
        self.nuisance_head = nn.Linear(self.feat_dim, 1)
        self.use_grl = use_grl

    def _forward_spatial(self, x):
        if isinstance(self.backbone, _DenseBackbone):
            conv_feats = self.backbone(x)
            return F.adaptive_avg_pool2d(conv_feats, 1).flatten(1), conv_feats
        return super()._forward_spatial(x)


class _OrdinalTrainDataset(core_pipeline.CBISDataset):

    """Training dataset that returns ordinal density instead of binary density."""
    def __getitem__(self, idx):
        img, mask, y, d = super().__getitem__(idx)
        if self.augment:
            dens = float(self.df.iloc[idx]['density'])
            d = float(np.clip((dens - 1.0) / 3.0, 0.0, 1.0))
        return img, mask, y, d


class _patched:

    """Temporarily swap pipeline globals for one training call."""
    def __init__(self, backbone, ordinal):
        self.backbone, self.ordinal = backbone, ordinal

    def __enter__(self):
        self._net, self._ds = core_pipeline.DCCAMNet, core_pipeline.CBISDataset
        _ACTIVE['backbone'] = self.backbone
        if self.backbone != 'resnet18':
            core_pipeline.DCCAMNet = VariantNet
        if self.ordinal:
            core_pipeline.CBISDataset = _OrdinalTrainDataset
        return self

    def __exit__(self, *exc):
        core_pipeline.DCCAMNet, core_pipeline.CBISDataset = self._net, self._ds
        _ACTIVE['backbone'] = 'resnet18'


def make_splits(meta_raw, patient_col, seed):
    """Same patient-grouped splits as the main run."""
    sp1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    trainval_idx, test_idx = next(sp1.split(meta_raw, groups=meta_raw[patient_col]))
    trainval_df = meta_raw.iloc[trainval_idx].reset_index(drop=True)
    test_df = meta_raw.iloc[test_idx].reset_index(drop=True)
    sp2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(sp2.split(trainval_df, groups=trainval_df[patient_col]))
    return (trainval_df.iloc[train_idx].reset_index(drop=True),
            trainval_df.iloc[val_idx].reset_index(drop=True), test_df)


def variant_meta(meta_raw, split):
    """Metadata for a density definition (median or clinical split)."""
    if split == 'clinical':
        m = meta_raw.copy()
        m['density_binary'] = (m['density'] >= 3).astype(int)
        return m
    return meta_raw


def injected_fold(train_df, seed):
    """Training fold with the paper injection."""
    used = core_pipeline.bias_split_by_correlation(train_df, seed)
    if used['malignant'].nunique() < 2 or used['density_binary'].nunique() < 2:
        return used, np.nan, np.nan
    r, p = pointbiserialr(used['malignant'], used['density_binary'])
    return used, float(r), float(p)


def arms_for(variant, arms):
    return [a for a in arms if not VARIANTS[variant]['ordinal'] or a in ORDINAL_ARMS]


def split_log_record(variant, seed, train_df, used, val_df, test_df, r, p):
    ta, tc = core_pipeline.group_masks(used)
    ea, ec = core_pipeline.group_masks(test_df)
    return {
        'variant': variant, 'seed': seed,
        'n_train_available': len(train_df), 'n_train_used': len(used),
        'n_val': len(val_df), 'n_test': len(test_df),
        'achieved_r': r, 'achieved_p': p,
        'train_malignant_frac': float(used['malignant'].mean()),
        'train_dense_frac': float(used['density_binary'].mean()),
        'natural_malignant_frac': float(train_df['malignant'].mean()),
        'natural_dense_frac': float(train_df['density_binary'].mean()),
        'train_aligned': int(ta.sum()), 'train_conflicting': int(tc.sum()),
        'test_aligned': int(ea.sum()), 'test_conflicting': int(ec.sum()),
    }


def check_backbones(backbones):
    print('\nBACKBONE CHECK (random weights, CPU, no download)')
    print('%-12s %9s %8s %-18s %-12s %s' % ('backbone', 'params(M)', 'feat_dim',
                                          'conv_feats shape', 'CAM shape', 'Grad-CAM'))
    ok_all = True
    for name in backbones:
        pre_old = _ACTIVE['pretrained']
        _ACTIVE['pretrained'] = False
        try:
            with _patched(name, ordinal=False):
                net = core_pipeline.DCCAMNet(use_grl=False)
            net.eval()
            x = torch.randn(2, 3, core_pipeline.IMG_SIZE, core_pipeline.IMG_SIZE)
            logit, _, pooled, conv = net(x)
            cam = net.cam(conv)
            gc = core_pipeline.GradCAM(net, net.backbone.layer4[-1])
            g = gc.generate(x)
            gc.remove()
            ok = (pooled.shape[1] == net.feat_dim and conv.shape[-1] == cam.shape[-1]
                  and g.shape[-2:] == (core_pipeline.IMG_SIZE, core_pipeline.IMG_SIZE)
                  and bool(torch.isfinite(g).all()))
            print('%-12s %9.1f %8d %-18s %-12s %s'
                  % (name, sum(p.numel() for p in net.parameters()) / 1e6, net.feat_dim,
                     tuple(conv.shape[1:]), tuple(cam.shape[1:]), 'ok' if ok else 'FAILED'))
            ok_all = ok_all and ok
        except Exception as exc:
            print('%-12s FAILED: %r' % (name, exc))
            ok_all = False
        finally:
            _ACTIVE['pretrained'] = pre_old
    return ok_all


def find_published_csv(explicit):
    cands = [explicit] if explicit else [
        os.path.join(core_pipeline.OUT_DIR, 'main_experiment', 'dccam_cbis_results.csv'),
        os.path.join(core_pipeline.OUT_DIR, 'method_comparison', 'dccam_cbis_results.csv'),
        os.path.join(core_pipeline.OUT_DIR, 'dccam_cbis_results.csv'),
    ]
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def load_reference(path, seeds, arms):
    """ResNet-18 rows from the main run, used as the reference variant."""
    df = pd.read_csv(path)
    dup = df[df.duplicated(['seed', 'strategy'], keep=False)]
    if len(dup):
        raise ValueError('%s has duplicate (seed, strategy) rows -- use the final '
                         'dccam_cbis_results.csv, not the .partial.csv' % path)
    df = df[df['seed'].isin(seeds) & df['strategy'].isin(arms)].copy()
    rows = []
    for _, r in df.iterrows():
        row = {c: r.get(c, np.nan) for c in METRIC_COLS}
        row.update({'variant': REFERENCE, 'variant_label': 'ResNet-18 (paper, imported)',
                    'backbone': 'resnet18', 'split': 'median', 'ordinal': False,
                    'seed': int(r['seed']), 'strategy': r['strategy'],
                    'n_train_used': np.nan, 'achieved_r': np.nan,
                    'train_malignant_frac': np.nan, 'train_minutes': np.nan,
                    'source': 'imported'})
        rows.append(row)
    return pd.DataFrame(rows)


def append_checkpoint(row, path):
    """Append one result row with a fixed column order."""
    pd.DataFrame([row]).reindex(columns=CHECKPOINT_COLS).to_csv(
        path, mode='a', index=False, header=not os.path.exists(path))


def run(args, out_dir):
    """Train every variant x arm x seed (resumable)."""
    meta_raw, patient_col = core_pipeline.load_data()
    seeds = list(core_pipeline.SEED_LIST)[:args.seeds]
    split_log_path = os.path.join(out_dir, 'split_log.csv')
    ck_path = os.path.join(out_dir, 'results.partial.csv')

    if args.dry_run:
        rows = []
        for v in args.variants:
            meta_v = variant_meta(meta_raw, VARIANTS[v]['split'])
            for seed in seeds:
                tr, va, te = make_splits(meta_v, patient_col, seed)
                used, r, p = injected_fold(tr, seed)
                rows.append(split_log_record(v, seed, tr, used, va, te, r, p))
        log = pd.DataFrame(rows)
        log.to_csv(os.path.join(out_dir, 'split_log_dryrun.csv'), index=False)
        print('\n' + '=' * 100)
        print('DRY RUN -- training folds (mean over %d seeds). NOTHING TRAINED.' % len(seeds))
        print('=' * 100)
        print('%-16s %-9s %8s %8s %10s %10s %9s %9s' % (
            'variant', 'backbone', 'r', 'n_train', 'malig.frac', 'dense.frac',
            'test_al', 'test_cf'))
        for v in args.variants:
            s = log[log.variant == v]
            print('%-16s %-9s %8.3f %8.0f %10.3f %10.3f %9.0f %9.0f' % (
                v, VARIANTS[v]['backbone'], s.achieved_r.mean(), s.n_train_used.mean(),
                s.train_malignant_frac.mean(), s.train_dense_frac.mean(),
                s.test_aligned.mean(), s.test_conflicting.mean()))
        print('\nnatural malignant fraction %.3f; natural dense fraction: median split %.3f'
              % (log.natural_malignant_frac.mean(),
                 log[log.variant != 'clinical_split'].natural_dense_frac.mean()
                 if (log.variant != 'clinical_split').any() else float('nan')))
        if (log.variant == 'clinical_split').any():
            print('                                        clinical split %.3f'
                  % log[log.variant == 'clinical_split'].natural_dense_frac.mean())
        n_runs = sum(len(arms_for(v, args.arms)) * len(seeds) for v in args.variants)
        print('\nplanned training runs: %d' % n_runs)
        ok = check_backbones(sorted({VARIANTS[v]['backbone'] for v in args.variants
                                     if VARIANTS[v]['backbone'] != 'resnet18'}))
        if not ok:
            print('\nBACKBONE CHECK FAILED -- do not start the full run; send me this output.')
        return None

    results, done = [], set()
    if os.path.exists(ck_path):
        prior = pd.read_csv(ck_path).reindex(columns=CHECKPOINT_COLS)
        results.extend(prior.to_dict('records'))
        done = set(zip(prior['variant'], prior['seed'].astype(int), prior['strategy']))
        print('Resuming: %d results already on disk; they will be skipped.' % len(prior))
    split_rows = pd.read_csv(split_log_path).to_dict('records') if os.path.exists(split_log_path) else []

    total = sum(len(arms_for(v, args.arms)) * len(seeds) for v in args.variants)
    t_start, n_done = time.time(), 0
    print('Planned: %d training runs (%d already done).'
          % (total, sum(1 for k in done if k[0] != REFERENCE)))

    for v in args.variants:
        cfg = VARIANTS[v]
        meta_v = variant_meta(meta_raw, cfg['split'])
        print('\n' + '=' * 72)
        print('VARIANT: %-16s %s | backbone=%s | split=%s | ordinal=%s'
              % (v, cfg['label'], cfg['backbone'], cfg['split'], cfg['ordinal']))
        print('=' * 72)
        for seed in seeds:
            tr, va, te = make_splits(meta_v, patient_col, seed)
            used, r, p = injected_fold(tr, seed)
            al_df = te[core_pipeline.group_masks(te)[0]].reset_index(drop=True)
            cf_df = te[core_pipeline.group_masks(te)[1]].reset_index(drop=True)
            print('\n--- seed %d | n_train=%d (of %d) | r=%.3f | malignant frac=%.3f | '
                  'test aligned=%d conflicting=%d'
                  % (seed, len(used), len(tr), r, used['malignant'].mean(),
                     len(al_df), len(cf_df)))
            split_rows = [x for x in split_rows
                          if not (x.get('variant') == v and x.get('seed') == seed)]
            split_rows.append(split_log_record(v, seed, tr, used, va, te, r, p))
            pd.DataFrame(split_rows).to_csv(split_log_path, index=False)

            loaders = None
            for arm in arms_for(v, args.arms):
                if (v, seed, arm) in done:
                    print('    [%s] already done -- skipping.' % arm)
                    continue
                if loaders is None:
                    mk = lambda d: (core_pipeline.make_loader(core_pipeline.CBISDataset(d, augment=False),
                                                      core_pipeline.BATCH_SIZE, shuffle=False)
                                    if len(d) else None)
                    loaders = (mk(te), mk(al_df), mk(cf_df))
                te_l, al_l, cf_l = loaders

                t0 = time.time()
                with _patched(cfg['backbone'], cfg['ordinal'] and arm != 'baseline'):
                    model = core_pipeline.train_model(arm, used, va, seed)
                auroc = core_pipeline.get_auroc(model, te_l)
                a_al = core_pipeline.get_auroc(model, al_l) if al_l else np.nan
                a_cf = core_pipeline.get_auroc(model, cf_l) if cf_l else np.nan
                gap = a_al - a_cf if not (np.isnan(a_al) or np.isnan(a_cf)) else np.nan
                sas = core_pipeline.get_mean_sas(model, te_l)
                probe = core_pipeline.bootstrap_probe_accuracy(model, te_l, seed=seed)
                views = core_pipeline.evaluate_view_subgroups(model, te)
                mins = (time.time() - t0) / 60.0
                print('    [%-8s] AUROC=%.4f gap=%.4f SAS=%.4f probe=%.4f (%.1f min)'
                      % (arm, auroc, gap, sas, probe['linear_mean'], mins))
                row = {
                    'variant': v, 'variant_label': cfg['label'], 'backbone': cfg['backbone'],
                    'split': cfg['split'], 'ordinal': cfg['ordinal'] and arm != 'baseline',
                    'seed': seed, 'strategy': arm, 'n_train_used': len(used),
                    'achieved_r': r, 'train_malignant_frac': float(used['malignant'].mean()),
                    'auroc_all': auroc, 'auroc_aligned': a_al, 'auroc_conflict': a_cf,
                    'worst_group_gap': gap, 'sas': sas,
                    'probe_linear_mean': probe['linear_mean'], 'probe_linear_std': probe['linear_std'],
                    'probe_mlp_mean': probe['mlp_mean'], 'probe_mlp_std': probe['mlp_std'],
                    'auroc_cc': views['auroc_cc'], 'auroc_mlo': views['auroc_mlo'],
                    'train_minutes': mins, 'source': 'trained',
                }
                results.append(row)
                done.add((v, seed, arm))
                append_checkpoint(row, ck_path)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                n_done += 1
                el = (time.time() - t_start) / 60.0
                print('        progress %d/%d  elapsed %.0f min  (~%.0f min remaining)'
                      % (n_done, total, el, el / n_done * (total - n_done)))
    return pd.DataFrame(results)


def paired_table(df, variants, arms):
    """Paired change of each arm against its own variant baseline."""
    rows = []
    for v in variants:
        sub = df[df.variant == v]
        base = sub[sub.strategy == 'baseline'].set_index('seed')
        for arm in arms:
            if arm == 'baseline':
                continue
            a = sub[sub.strategy == arm].set_index('seed')
            common = sorted(set(a.index) & set(base.index))
            if len(common) < 2:
                continue
            for m, _, want in METRICS:
                d = (a.loc[common, m] - base.loc[common, m]).dropna().values
                if len(d) < 2:
                    continue
                mean, lo, hi = core_pipeline.bootstrap_ci(d)
                if np.allclose(d, 0):
                    pv = 1.0
                else:
                    try:
                        pv = float(wilcoxon(d).pvalue)
                    except ValueError:
                        pv = np.nan
                rows.append({'variant': v, 'strategy': arm, 'metric': m, 'want': want,
                             'n_pairs': len(d), 'mean_diff': mean, 'ci_lo': lo, 'ci_hi': hi,
                             'wilcoxon_p': pv, 'ci_excludes_zero': bool(lo > 0 or hi < 0),
                             'seeds_below_baseline': int((d < 0).sum())})
    cols = ['variant', 'strategy', 'metric', 'want', 'n_pairs', 'mean_diff', 'ci_lo', 'ci_hi',
            'wilcoxon_p', 'ci_excludes_zero', 'seeds_below_baseline']
    return pd.DataFrame(rows, columns=cols)


def pattern_check(pt, df, variants):
    """Check the two-axis pattern (gap vs SAS) for every variant."""
    def get(v, arm, m):
        r = pt[(pt.variant == v) & (pt.strategy == arm) & (pt.metric == m)]
        return None if r.empty else r.iloc[0]

    rows = []
    for v in variants:
        base = df[(df.variant == v) & (df.strategy == 'baseline')]['worst_group_gap'].dropna()
        entry = {'variant': v, 'n_seeds': len(base),
                 'baseline_gap_mean': base.mean() if len(base) else np.nan,
                 'baseline_gap_seeds_positive': int((base > 0).sum())}

        def crit(key, arm, m, test):
            r = get(v, arm, m)
            if r is None:
                entry[key] = 'n/a'
                entry[key + '_detail'] = ''
                return
            entry[key] = 'yes' if test(r) else 'NO'
            entry[key + '_detail'] = '%+.3f [%+.3f, %+.3f]' % (r.mean_diff, r.ci_lo, r.ci_hi)

        crit('distcorr_lowers_gap', 'distcorr', 'worst_group_gap', lambda r: r.ci_hi < 0)
        crit('full_lowers_gap', 'full', 'worst_group_gap', lambda r: r.ci_hi < 0)
        crit('camreg_lowers_sas', 'camreg', 'sas', lambda r: r.ci_hi < 0)
        crit('full_lowers_sas', 'full', 'sas', lambda r: r.ci_hi < 0)
        crit('camreg_does_NOT_lower_gap', 'camreg', 'worst_group_gap', lambda r: r.ci_hi >= 0)
        crit('distcorr_does_NOT_lower_sas', 'distcorr', 'sas', lambda r: r.ci_hi >= 0)
        rows.append(entry)
    return pd.DataFrame(rows)


def summarize(df, out_dir, variants, arms):
    """Summary tables for all variants."""
    df = df.copy()
    order = [REFERENCE] + [v for v in VARIANT_ORDER if v in variants]
    order = [v for v in order if v in set(df.variant)]

    summ = []
    for v in order:
        for arm in arms:
            s = df[(df.variant == v) & (df.strategy == arm)]
            if s.empty:
                continue
            row = {'variant': v, 'strategy': arm, 'n_seeds': len(s),
                   'n_train_mean': s['n_train_used'].mean(),
                   'achieved_r_mean': s['achieved_r'].mean()}
            for m, _, _ in METRICS:
                row[m + '_mean'] = s[m].mean()
                row[m + '_sd'] = s[m].std(ddof=1)
            summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(out_dir, 'variant_summary.csv'), index=False)

    pt = paired_table(df, order, arms)
    pt.to_csv(os.path.join(out_dir, 'paired_vs_baseline.csv'), index=False)
    pc = pattern_check(pt, df, order)
    pc.to_csv(os.path.join(out_dir, 'pattern_check.csv'), index=False)

    print('\n' + '=' * 100)
    print('VARIANT SUMMARY (mean +- SD across seeds)')
    print('=' * 100)
    print('%-16s %-9s %3s %6s %6s | %-15s %-15s %-15s %-15s' % (
        'variant', 'strategy', 'n', 'n_tr', 'r', 'gap', 'SAS', 'probe', 'AUROC'))
    for _, r in summ.iterrows():
        print('%-16s %-9s %3d %6.0f %6.3f | %5.3f +- %5.3f   %5.3f +- %5.3f   %5.3f +- %5.3f   '
              '%5.3f +- %5.3f' % (
                  r.variant, r.strategy, r.n_seeds,
                  r.n_train_mean if pd.notna(r.n_train_mean) else 0,
                  r.achieved_r_mean if pd.notna(r.achieved_r_mean) else 0,
                  r.worst_group_gap_mean, r.worst_group_gap_sd, r.sas_mean, r.sas_sd,
                  r.probe_linear_mean_mean, r.probe_linear_mean_sd,
                  r.auroc_all_mean, r.auroc_all_sd))

    print('\n' + '=' * 100)
    print('PATTERN CHECK -- are the paper\'s qualitative findings reproduced?')
    print('  (difference vs the same variant\'s baseline, mean [95% bootstrap CI])')
    print('=' * 100)
    keys = ['distcorr_lowers_gap', 'full_lowers_gap', 'camreg_lowers_sas', 'full_lowers_sas',
            'camreg_does_NOT_lower_gap', 'distcorr_does_NOT_lower_sas']
    for _, r in pc.iterrows():
        print('\n%s  (n=%d seeds; baseline gap %.3f, positive in %d seeds)'
              % (r.variant, r.n_seeds, r.baseline_gap_mean, r.baseline_gap_seeds_positive))
        for k in keys:
            print('   %-30s %-4s %s' % (k, r[k], r[k + '_detail']))
    print('\n  "NO" is a finding, not an error. Report it, and say which criterion failed.')
    print('  With few seeds the CI is the primary evidence (Wilcoxon cannot go below '
          'p=0.0625 at n=5).')
    return order, pt


def plot_effects(pt, order, arms, out_dir):
    """Effects by variant figure."""
    if pt.empty:
        return
    arms_p = [a for a in arms if a != 'baseline']
    colors = {a: c for a, c in zip(arms_p, plt.rcParams['axes.prop_cycle'].by_key()['color'])}
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.8 * len(METRICS), 0.9 + 0.62 * len(order) * len(arms_p) / 2 + 2.2),
                             sharey=True)
    ys = {v: i for i, v in enumerate(order)}
    off = {a: (j - (len(arms_p) - 1) / 2) * 0.17 for j, a in enumerate(arms_p)}
    for ax, (m, title, _) in zip(axes, METRICS):
        for a in arms_p:
            for v in order:
                r = pt[(pt.variant == v) & (pt.strategy == a) & (pt.metric == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.errorbar(r.mean_diff, ys[v] + off[a],
                            xerr=[[r.mean_diff - r.ci_lo], [r.ci_hi - r.mean_diff]],
                            fmt='o', ms=5, color=colors[a], capsize=2.5, lw=1.4,
                            mfc=colors[a] if r.ci_excludes_zero else 'white')
        ax.axvline(0, color='k', lw=0.8, ls='--')
        ax.set_title('%s\n(vs baseline)' % title, fontsize=10)
        ax.grid(alpha=0.3, axis='x', ls=':')
    axes[0].set_yticks(list(ys.values()))
    axes[0].set_yticklabels([PLOT_LABELS[v]['label'] for v in order], fontsize=9)
    axes[0].invert_yaxis()
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='o', ls='', color=colors[a], label=a) for a in arms_p]
    h.append(Line2D([], [], marker='o', ls='', mfc='white', color='gray',
                    label='hollow = CI includes 0'))
    fig.legend(handles=h, loc='lower center', ncol=len(h), fontsize=8, frameon=False)
    fig.suptitle('Effect of each strategy relative to its own baseline, by backbone / '
                 'density representation (mean paired difference, 95% bootstrap CI)', fontsize=10.5)
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
    p = os.path.join(out_dir, 'effects_by_variant.png')
    fig.savefig(p, dpi=220)
    plt.close(fig)
    print('\nWrote %s' % p)


def parse_args():
    ap = argparse.ArgumentParser(
        description='Backbone and density-definition robustness on CBIS-DDSM')
    ap.add_argument('--seeds', type=int, default=5,
                    help='first N seeds of core_pipeline.SEED_LIST (default 5)')
    ap.add_argument('--variants', nargs='+', default=None,
                    help='subset of %s (default %s)' % (VARIANT_ORDER, DEFAULT_VARIANTS))
    ap.add_argument('--arms', nargs='+', default=None,
                    help='subset of %s (default %s)' % (core_pipeline.ABLATIONS, DEFAULT_ARMS))
    ap.add_argument('--smoke', action='store_true',
                    help='1 seed, densenet121, baseline + distcorr; separate output folder')
    ap.add_argument('--dry-run', action='store_true',
                    help='build folds, check backbones, train nothing')
    ap.add_argument('--summarize-only', action='store_true')
    ap.add_argument('--import-reference', action=argparse.BooleanOptionalAction, default=True,
                    help="import the paper's ResNet-18 rows as a reference column (default on)")
    ap.add_argument('--published-csv', default=None)
    args = ap.parse_args()
    args.variants = args.variants or list(DEFAULT_VARIANTS)
    args.arms = args.arms or list(DEFAULT_ARMS)
    if args.smoke:
        args.seeds, args.variants, args.arms = 1, ['densenet121'], ['baseline', 'distcorr']
    bad = [v for v in args.variants if v not in VARIANTS]
    if bad:
        ap.error('unknown variant(s) %s (valid: %s)' % (bad, VARIANT_ORDER))
    bad = [a for a in args.arms if a not in core_pipeline.ABLATIONS]
    if bad:
        ap.error('unknown arm(s) %s (core_pipeline knows %s)' % (bad, core_pipeline.ABLATIONS))
    if 'baseline' not in args.arms:
        ap.error('--arms must include baseline (every comparison is against it)')
    args.variants = [v for v in VARIANT_ORDER if v in args.variants]
    return args


def main():
    args = parse_args()
    sub = OUTPUT_SUBDIR + ('_smoke' if args.smoke else '')
    out_dir = os.path.join(core_pipeline.OUT_DIR, sub)
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 78)
    print('BACKBONE AND DENSITY-DEFINITION ROBUSTNESS')
    print('=' * 78)
    print('core_pipeline OUT_DIR (untouched) : %s' % core_pipeline.OUT_DIR)
    print('image cache               : %s (exists=%s)'
          % (core_pipeline.CACHE_FILE, os.path.exists(core_pipeline.CACHE_FILE)))
    print('writing results to        : %s' % out_dir)
    print('seeds                     : %s' % list(core_pipeline.SEED_LIST)[:args.seeds])
    print('variants                  : %s' % args.variants)
    print('arms                      : %s' % args.arms)
    if args.dry_run:
        print('MODE                      : DRY RUN -- nothing will be trained')
    print('=' * 78)
    t0 = time.time()

    ck = os.path.join(out_dir, 'results.partial.csv')
    full = os.path.join(out_dir, 'results.csv')
    if args.summarize_only:
        src = full if os.path.exists(full) else ck
        if not os.path.exists(src):
            print('Nothing to summarize: no results in %s' % out_dir)
            return
        df = pd.read_csv(src)
        df = df[df.variant != REFERENCE].drop_duplicates(['variant', 'seed', 'strategy'], keep='last')
    else:
        df = run(args, out_dir)
        if df is None:
            return
        df = df.drop_duplicates(['variant', 'seed', 'strategy'], keep='last')

    ref_path = None
    if args.import_reference:
        ref_path = find_published_csv(args.published_csv)
        if ref_path is None:
            print("\nNOTE: dccam_cbis_results.csv not found -- no ResNet-18 reference column.")
        else:
            try:
                ref = load_reference(ref_path, list(core_pipeline.SEED_LIST)[:args.seeds], args.arms)
                print('\nReference ResNet-18 rows imported from %s (modified %s)'
                      % (ref_path, datetime.fromtimestamp(os.path.getmtime(ref_path))
                         .isoformat(timespec='minutes')))
                df = pd.concat([ref, df], ignore_index=True)
            except ValueError as exc:
                print('\nERROR importing the reference rows:\n  %s' % exc)
                sys.exit(2)

    df.to_csv(full, index=False)
    variants_present = [v for v in args.variants if v in set(df.variant)]
    order, pt = summarize(df, out_dir, variants_present, args.arms)
    plot_effects(pt, order, args.arms, out_dir)

    manifest = {
        'script': 'backbone_density_variants.py',
        'purpose': 'backbones and density definitions on CBIS-DDSM',
        'run_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_minutes': round((time.time() - t0) / 60.0, 1),
        'seeds': list(core_pipeline.SEED_LIST)[:args.seeds],
        'variants': {v: VARIANTS[v] for v in args.variants},
        'arms': args.arms,
        'reference_csv': ref_path,
        'n_rows': int(len(df)),
        'pipeline_config': {k: getattr(core_pipeline, k, None) for k in
                         ['IMG_SIZE', 'BATCH_SIZE', 'MAX_EPOCHS', 'PATIENCE', 'LR',
                          'LAMBDA_RAMP_EPOCHS', 'MAX_LAMBDA_GRL', 'MAX_WEIGHT_DISTCORR',
                          'MAX_WEIGHT_CAMREG', 'BIAS_STRENGTH', 'DATA_ROOT', 'CACHE_FILE']},
        'determinism': determinism_snapshot(),
    }
    with open(os.path.join(out_dir, 'run_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print('\nDone in %.1f min. Results in %s' % ((time.time() - t0) / 60.0, out_dir))


if __name__ == '__main__':
    main()
