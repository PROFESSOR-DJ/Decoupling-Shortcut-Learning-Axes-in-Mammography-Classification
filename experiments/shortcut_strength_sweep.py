"""Shortcut-strength sweep: injected correlation r = 0 to 0.62, with training-size and
class-prior controls (5 seeds).

Run: python experiments/shortcut_strength_sweep.py --seeds 5
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

import core_pipeline

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

warnings.filterwarnings('ignore', message='.*Precision loss occurred.*')
warnings.filterwarnings('ignore', category=RuntimeWarning,
                        module=r'scipy\.stats\._wilcoxon')

OUTPUT_SUBDIR = 'shortcut_strength_sweep'

CONDITIONS = {
    'natural':         dict(inject=False, strength=None, mode='none',
                            label='natural (no injection)'),
    'subsample_r00':   dict(inject=True,  strength=0.50, mode='pipeline',
                            label='subsampled, r~0 (size control)'),
    'weak_r03':        dict(inject=True,  strength=0.65, mode='pipeline',
                            label='weak, r~0.30'),
    'moderate_r045':   dict(inject=True,  strength=0.75, mode='pipeline',
                            label='moderate, r~0.45'),
    'published_r06':   dict(inject=True,  strength=0.85, mode='pipeline',
                            label='published, r~0.62'),
    'balanced_r06':    dict(inject=True,  strength=None, mode='balanced', target_r=0.62,
                            label='class-balanced, r~0.62 (prior control)'),
    'balanced_r00':    dict(inject=True,  strength=None, mode='balanced', target_r=0.0,
                            n_match='balanced_r06',
                            label='class-balanced, r~0 (null for the prior control)'),
    'sizematched_r06': dict(inject=True,  strength=0.85, mode='sizematched',
                            label='published injection, n-matched (size control)'),
}
CONDITION_ORDER = ['natural', 'subsample_r00', 'weak_r03', 'moderate_r045',
                   'published_r06', 'sizematched_r06', 'balanced_r00', 'balanced_r06']
DOSE_FAMILY = ['natural', 'subsample_r00', 'weak_r03', 'moderate_r045', 'published_r06']
CONTROL_CONDITIONS = ['sizematched_r06', 'balanced_r00', 'balanced_r06']
PUBLISHED_CONDITION = 'published_r06'

DEFAULT_STRATEGIES = ['baseline', 'distcorr', 'camreg', 'full']

METRICS = [
    ('worst_group_gap',   'Worst-group gap (aligned - conflicting)', 'lower'),
    ('sas',               'Spurious Attribution Score',              'lower'),
    ('probe_linear_mean', 'Density probe accuracy (linear)',         'lower'),
    ('auroc_all',         'Main-task AUROC (all test)',              'higher'),
]


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


CELLS = [('mal_dense', 1, 1), ('mal_nondense', 1, 0),
         ('ben_dense', 0, 1), ('ben_nondense', 0, 0)]


def cell_counts(df):
    """Counts of the four malignancy x density training cells."""
    return {name: int(((df['malignant'] == m) & (df['density_binary'] == d)).sum())
            for name, m, d in CELLS}


def _phi_for(m, s):
    p_dense = m * s + (1.0 - m) * (1.0 - s)
    return (2.0 * s - 1.0) * np.sqrt(m * (1.0 - m)) / np.sqrt(p_dense * (1.0 - p_dense))


def solve_strength_for_r(m, target_r):
    """Bias strength that gives a target correlation at a given class prior."""
    lo, hi = 0.5, 0.999
    if _phi_for(m, hi) < target_r:
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _phi_for(m, mid) < target_r:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def balanced_injection(train_df, seed, target_r, n_cap=None):
    """Inject a target correlation while keeping the natural class prior."""
    m = float(train_df['malignant'].mean())
    s = solve_strength_for_r(m, target_r)
    props = {'mal_dense': m * s, 'mal_nondense': m * (1 - s),
             'ben_dense': (1 - m) * (1 - s), 'ben_nondense': (1 - m) * s}
    avail = cell_counts(train_df)
    n_max = min(avail[k] / props[k] for k in props if props[k] > 0)
    n_total = int(np.floor(n_max))
    if n_cap is not None:
        n_total = min(n_total, int(n_cap))
    if n_total < 50:
        return None, {'m': m, 's': s, 'n_total': n_total, 'avail': avail}
    parts = []
    for i, (name, mal, dense) in enumerate(CELLS):
        k = min(int(round(n_total * props[name])), avail[name])
        cell = train_df[(train_df['malignant'] == mal) & (train_df['density_binary'] == dense)]
        parts.append(cell.sample(n=k, random_state=seed + 101 * (i + 1)))
    used = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return used, {'m': m, 's': s, 'n_total': len(used), 'avail': avail}


def sizematched_injection(train_df, seed, strength, n_target):
    """Paper injection, then a stratified subsample to a fixed size."""
    pub = core_pipeline.bias_split_by_correlation(train_df, seed, strength)
    if n_target is None or n_target >= len(pub):
        return pub.reset_index(drop=True)
    frac = n_target / float(len(pub))
    parts = []
    for i, (name, mal, dense) in enumerate(CELLS):
        cell = pub[(pub['malignant'] == mal) & (pub['density_binary'] == dense)]
        k = int(round(len(cell) * frac))
        if k > 0:
            parts.append(cell.sample(n=min(k, len(cell)), random_state=seed + 211 * (i + 1)))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_condition_train_df(train_df, condition, seed):
    """Training fold for one sweep condition."""
    cfg = CONDITIONS[condition]
    mode = cfg.get('mode', 'pipeline')
    note = ''
    if mode == 'none':
        used = train_df.reset_index(drop=True)
    elif mode == 'pipeline':
        used = core_pipeline.bias_split_by_correlation(train_df, seed, cfg['strength'])
    elif mode == 'balanced':
        n_cap = None
        if cfg.get('n_match'):
            ref, ref_info = balanced_injection(train_df, seed,
                                               CONDITIONS[cfg['n_match']]['target_r'])
            n_cap = len(ref) if ref is not None else None
        used, info = balanced_injection(train_df, seed, cfg['target_r'], n_cap=n_cap)
        if used is None:
            raise RuntimeError(
                '%s is infeasible for seed %d: the scarcest cell supports only '
                'n=%d (available cells: %s)' % (condition, seed, info['n_total'], info['avail']))
        note = 'prior m=%.3f, solved strength s=%.3f%s' % (
            info['m'], info['s'], (', n capped to match %s' % cfg['n_match']) if n_cap else '')
    elif mode == 'sizematched':
        bal, _ = balanced_injection(train_df, seed, CONDITIONS['balanced_r06']['target_r'])
        n_target = len(bal) if bal is not None else None
        used = sizematched_injection(train_df, seed, cfg['strength'], n_target)
        note = 'n matched to balanced_r06 (n=%s)' % n_target
    else:
        raise ValueError('unknown mode %r' % mode)

    if used['malignant'].nunique() < 2 or used['density_binary'].nunique() < 2:
        return used, np.nan, np.nan, note
    r, p = pointbiserialr(used['malignant'], used['density_binary'])
    return used, float(r), float(p), note


IMPORT_METRIC_COLS = ['auroc_all', 'auroc_aligned', 'auroc_conflict', 'worst_group_gap',
                      'sas', 'probe_linear_mean', 'probe_linear_std', 'probe_mlp_mean',
                      'probe_mlp_std', 'auroc_cc', 'auroc_mlo']
REPRO_COLS = ['auroc_all', 'worst_group_gap', 'sas', 'probe_linear_mean']
REPRO_TOL = 5e-3


CHECKPOINT_COLS = (['condition', 'condition_label', 'bias_strength', 'achieved_r',
                    'n_train_used', 'seed', 'strategy'] + IMPORT_METRIC_COLS +
                   ['train_minutes', 'source'])


def append_checkpoint(row, path):
    """Append one result row with a fixed column order."""
    pd.DataFrame([row]).reindex(columns=CHECKPOINT_COLS).to_csv(
        path, mode='a', index=False, header=not os.path.exists(path))


def find_published_csv(explicit):
    """Locate the main-run results CSV."""
    cands = [explicit] if explicit else [
        os.path.join(core_pipeline.OUT_DIR, 'main_experiment', 'dccam_cbis_results.csv'),
        os.path.join(core_pipeline.OUT_DIR, 'method_comparison', 'dccam_cbis_results.csv'),
        os.path.join(core_pipeline.OUT_DIR, 'dccam_cbis_results.csv'),
    ]
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def load_published_index(path, seeds, strategies):
    """Main-run results indexed by (seed, strategy)."""
    df = pd.read_csv(path)
    need = {'seed', 'strategy'} | set(REPRO_COLS)
    if not need.issubset(df.columns):
        raise ValueError('%s lacks columns %s' % (path, sorted(need - set(df.columns))))
    dup = df[df.duplicated(['seed', 'strategy'], keep=False)]
    if len(dup):
        raise ValueError(
            '%s has duplicate (seed, strategy) rows:\n%s\nUse the final dccam_cbis_results.csv '
            '(not the .partial.csv) or pass --no-import-published.'
            % (path, dup[['seed', 'strategy']].drop_duplicates().to_string(index=False)))
    df = df[df['seed'].isin(seeds) & df['strategy'].isin(strategies)]
    return {(int(r['seed']), r['strategy']): r.to_dict() for _, r in df.iterrows()}


def reproducibility_check(prior_rows, pub_index, out_dir):
    """Compare rows retrained here with the main-run results."""
    recs = []
    for r in prior_rows:
        if r.get('condition') != PUBLISHED_CONDITION or r.get('source') == 'imported':
            continue
        ref = pub_index.get((int(r['seed']), r['strategy']))
        if ref is None:
            continue
        rec = {'seed': int(r['seed']), 'strategy': r['strategy']}
        worst = 0.0
        for c in REPRO_COLS:
            d = float(r[c]) - float(ref[c])
            rec['%s_trained' % c] = float(r[c])
            rec['%s_paper' % c] = float(ref[c])
            rec['%s_diff' % c] = d
            worst = max(worst, abs(d))
        rec['max_abs_diff'] = worst
        rec['within_tol'] = bool(worst <= REPRO_TOL)
        recs.append(rec)
    if not recs:
        return
    out = pd.DataFrame(recs)
    out.to_csv(os.path.join(out_dir, 'import_reproducibility.csv'), index=False)
    print('\n' + '-' * 72)
    print('REPRODUCIBILITY CHECK: published_r06 trained here vs the paper CSV '
          '(tol %.0e)' % REPRO_TOL)
    for _, r in out.iterrows():
        print('  seed %-3d %-9s  d(AUROC)=%+.4f  d(gap)=%+.4f  d(SAS)=%+.4f  d(probe)=%+.4f  %s'
              % (r['seed'], r['strategy'], r['auroc_all_diff'], r['worst_group_gap_diff'],
                 r['sas_diff'], r['probe_linear_mean_diff'],
                 'OK' if r['within_tol'] else '*** MISMATCH ***'))
    if not out['within_tol'].all():
        print('  MISMATCH: the paper CSV and a fresh training run disagree. Do NOT trust the\n'
              '  import until you know why (different core_pipeline.py, GPU, or cudnn settings).\n'
              '  Re-run with --no-import-published to train the published arm here instead.')
    print('-' * 72)


def make_splits(meta_raw, patient_col, seed):
    """Same patient-grouped splits as the main run."""
    sp1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    trainval_idx, test_idx = next(sp1.split(meta_raw, groups=meta_raw[patient_col]))
    trainval_df = meta_raw.iloc[trainval_idx].reset_index(drop=True)
    test_df = meta_raw.iloc[test_idx].reset_index(drop=True)
    sp2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(sp2.split(trainval_df, groups=trainval_df[patient_col]))
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def split_log_record(condition, cfg, seed, train_df, train_used, val_df, test_df,
                     r_ach, p_ach, note):
    tr_al, tr_cf = core_pipeline.group_masks(train_used)
    te_al, te_cf = core_pipeline.group_masks(test_df)
    rec = {
        'condition': condition,
        'condition_label': cfg['label'],
        'bias_strength': cfg.get('strength'),
        'seed': seed,
        'n_train_available': len(train_df),
        'n_train_used': len(train_used),
        'n_val': len(val_df),
        'n_test': len(test_df),
        'achieved_r': r_ach,
        'achieved_p': p_ach,
        'train_aligned': int(tr_al.sum()),
        'train_conflicting': int(tr_cf.sum()),
        'test_aligned': int(te_al.sum()),
        'test_conflicting': int(te_cf.sum()),
        'train_malignant_frac': float(train_used['malignant'].mean()),
        'train_dense_frac': float(train_used['density_binary'].mean()),
        'natural_malignant_frac': float(train_df['malignant'].mean()),
        'note': note,
    }
    for k, v in cell_counts(train_used).items():
        rec['train_' + k] = v
    return rec


def run_sweep(args, out_dir):
    """Train every condition x strategy x seed (resumable)."""
    meta_raw, patient_col = core_pipeline.load_data()

    seed_list = list(core_pipeline.SEED_LIST)[:args.seeds]
    conditions = args.conditions
    strategies = args.strategies

    checkpoint_path = os.path.join(out_dir, 'results.partial.csv')
    split_log_path = os.path.join(out_dir, 'split_log.csv')

    if args.dry_run:
        rows = []
        for condition in conditions:
            cfg = CONDITIONS[condition]
            for seed in seed_list:
                train_df, val_df, test_df = make_splits(meta_raw, patient_col, seed)
                used, r_ach, p_ach, note = make_condition_train_df(train_df, condition, seed)
                rows.append(split_log_record(condition, cfg, seed, train_df, used, val_df,
                                             test_df, r_ach, p_ach, note))
        log = pd.DataFrame(rows)
        log.to_csv(os.path.join(out_dir, 'split_log_dryrun.csv'), index=False)
        print('\n' + '=' * 100)
        print('DRY RUN -- training folds per condition (mean over %d seeds). NOTHING TRAINED.'
              % len(seed_list))
        print('=' * 100)
        print('%-16s %7s %9s %9s %9s  %-34s' % ('condition', 'r', 'n_train', 'malig.frac',
                                              'dense.frac', 'training cells (mal-d/mal-nd/ben-d/ben-nd)'))
        for condition in conditions:
            s = log[log.condition == condition]
            cells = '%5.0f /%5.0f /%5.0f /%5.0f' % (
                s.train_mal_dense.mean(), s.train_mal_nondense.mean(),
                s.train_ben_dense.mean(), s.train_ben_nondense.mean())
            print('%-16s %7.3f %9.0f %9.3f %9.3f  %s' % (
                condition, s.achieved_r.mean(), s.n_train_used.mean(),
                s.train_malignant_frac.mean(), s.train_dense_frac.mean(), cells))
        print('\nnatural class prior (train split): malignant fraction %.3f'
              % log.natural_malignant_frac.mean())
        for condition in CONTROL_CONDITIONS:
            if condition in conditions:
                s = log[log.condition == condition]
                print('%s: n_train min %d / max %d across seeds; achieved r min %.3f / max %.3f'
                      % (condition, s.n_train_used.min(), s.n_train_used.max(),
                         s.achieved_r.min(), s.achieved_r.max()))
        print('\nWrote %s' % os.path.join(out_dir, 'split_log_dryrun.csv'))
        return None

    all_results, completed = [], set()
    if os.path.exists(checkpoint_path):
        prior = pd.read_csv(checkpoint_path)
        if 'source' not in prior.columns:
            prior['source'] = 'trained'
        prior['source'] = prior['source'].fillna('trained')
        prior = prior.reindex(columns=CHECKPOINT_COLS)
        prior.to_csv(checkpoint_path, index=False)
        all_results.extend(prior.to_dict('records'))
        completed = set(zip(prior['condition'], prior['seed'].astype(int), prior['strategy']))
        print('Resuming: %d (condition, seed, strategy) results already on disk; '
              'they will be skipped.' % len(prior))

    split_rows = []
    if os.path.exists(split_log_path):
        split_rows = pd.read_csv(split_log_path).to_dict('records')

    pub_index = {}
    if args.import_published and PUBLISHED_CONDITION in conditions:
        pub_path = find_published_csv(args.published_csv)
        if pub_path is None:
            print('\nNOTE: no dccam_cbis_results.csv found (looked in main_experiment/, '
                  'method_comparison/ and the core_pipeline folder) -- published_r06 will be TRAINED.')
        else:
            try:
                pub_index = load_published_index(pub_path, seed_list, strategies)
            except ValueError as exc:
                print('\nERROR importing the paper CSV:\n  %s' % exc)
                sys.exit(2)
            args._published_csv_used = pub_path
            print('\nIMPORTING published_r06 from %s' % pub_path)
            print('  file modified %s ; %d of %d requested (seed, strategy) rows available'
                  % (datetime.fromtimestamp(os.path.getmtime(pub_path)).isoformat(timespec='minutes'),
                     len(pub_index), len(seed_list) * len(strategies)))
            reproducibility_check(all_results, pub_index, out_dir)

    n_import = sum(1 for s in seed_list for st in strategies
                   if (PUBLISHED_CONDITION, s, st) not in completed and (s, st) in pub_index
                   and PUBLISHED_CONDITION in conditions)
    total = len(conditions) * len(seed_list) * len(strategies)
    done_n = len(completed)
    t_start = time.time()
    print('Planned: %d cells in total, %d already done, %d will be imported, %d to train.'
          % (total, len(completed), n_import, max(0, total - len(completed) - n_import)))

    for condition in conditions:
        cfg = CONDITIONS[condition]
        print('\n' + '=' * 72)
        print('CONDITION: %-16s  %s' % (condition, cfg['label']))
        print('=' * 72)

        for seed in seed_list:
            train_df, val_df, test_df = make_splits(meta_raw, patient_col, seed)
            train_used, r_ach, p_ach, note = make_condition_train_df(train_df, condition, seed)

            aligned_mask, conflict_mask = core_pipeline.group_masks(test_df)
            aligned_df = test_df[aligned_mask].reset_index(drop=True)
            conflict_df = test_df[conflict_mask].reset_index(drop=True)
            tr_aligned, tr_conflict = core_pipeline.group_masks(train_used)

            print('\n--- seed %d | n_train=%d (from %d available) | achieved r=%.3f (p=%.4g) '
                  '| malignant frac=%.3f%s'
                  % (seed, len(train_used), len(train_df), r_ach, p_ach,
                     train_used['malignant'].mean(), ('  [' + note + ']') if note else ''))
            print('    train groups: aligned=%d conflicting=%d | '
                  'test groups: aligned=%d conflicting=%d'
                  % (int(tr_aligned.sum()), int(tr_conflict.sum()),
                     len(aligned_df), len(conflict_df)))

            split_rows = [r for r in split_rows
                          if not (r.get('condition') == condition and r.get('seed') == seed)]
            split_rows.append(split_log_record(condition, cfg, seed, train_df, train_used,
                                               val_df, test_df, r_ach, p_ach, note))
            pd.DataFrame(split_rows).to_csv(split_log_path, index=False)

            test_loader = aligned_loader = conflict_loader = None

            for strategy in strategies:
                key = (condition, seed, strategy)
                if key in completed:
                    print('    [%s] already done -- skipping.' % strategy)
                    continue

                base_row = {
                    'condition': condition,
                    'condition_label': cfg['label'],
                    'bias_strength': cfg.get('strength'),
                    'achieved_r': r_ach,
                    'n_train_used': len(train_used),
                    'seed': seed,
                    'strategy': strategy,
                }

                if condition == PUBLISHED_CONDITION and (seed, strategy) in pub_index:
                    ref = pub_index[(seed, strategy)]
                    row = dict(base_row)
                    for c in IMPORT_METRIC_COLS:
                        row[c] = ref.get(c, np.nan)
                    row['train_minutes'] = np.nan
                    row['source'] = 'imported'
                    all_results.append(row)
                    completed.add(key)
                    append_checkpoint(row, checkpoint_path)
                    done_n += 1
                    print('    [%-8s] imported  AUROC=%.4f  gap=%.4f  SAS=%.4f  probe=%.4f'
                          % (strategy, row['auroc_all'], row['worst_group_gap'],
                             row['sas'], row['probe_linear_mean']))
                    continue

                if test_loader is None:
                    test_loader = core_pipeline.make_loader(
                        core_pipeline.CBISDataset(test_df, augment=False), core_pipeline.BATCH_SIZE, shuffle=False)
                    aligned_loader = core_pipeline.make_loader(
                        core_pipeline.CBISDataset(aligned_df, augment=False), core_pipeline.BATCH_SIZE,
                        shuffle=False) if len(aligned_df) else None
                    conflict_loader = core_pipeline.make_loader(
                        core_pipeline.CBISDataset(conflict_df, augment=False), core_pipeline.BATCH_SIZE,
                        shuffle=False) if len(conflict_df) else None

                t0 = time.time()
                model = core_pipeline.train_model(strategy, train_used, val_df, seed)

                auroc = core_pipeline.get_auroc(model, test_loader)
                a_al = core_pipeline.get_auroc(model, aligned_loader) if aligned_loader else np.nan
                a_cf = core_pipeline.get_auroc(model, conflict_loader) if conflict_loader else np.nan
                gap = a_al - a_cf if not (np.isnan(a_al) or np.isnan(a_cf)) else np.nan
                sas = core_pipeline.get_mean_sas(model, test_loader)
                probe = core_pipeline.bootstrap_probe_accuracy(model, test_loader, seed=seed)
                views = core_pipeline.evaluate_view_subgroups(model, test_df)
                mins = (time.time() - t0) / 60.0

                print('    [%-8s] AUROC=%.4f  gap=%.4f  SAS=%.4f  probe=%.4f  (%.1f min)'
                      % (strategy, auroc, gap, sas, probe['linear_mean'], mins))

                row = dict(base_row)
                row.update({
                    'auroc_all': auroc,
                    'auroc_aligned': a_al,
                    'auroc_conflict': a_cf,
                    'worst_group_gap': gap,
                    'sas': sas,
                    'probe_linear_mean': probe['linear_mean'],
                    'probe_linear_std': probe['linear_std'],
                    'probe_mlp_mean': probe['mlp_mean'],
                    'probe_mlp_std': probe['mlp_std'],
                    'auroc_cc': views['auroc_cc'],
                    'auroc_mlo': views['auroc_mlo'],
                    'train_minutes': mins,
                    'source': 'trained',
                })
                all_results.append(row)
                completed.add(key)

                append_checkpoint(row, checkpoint_path)

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                done_n += 1
                elapsed = (time.time() - t_start) / 60.0
                print('        progress %d/%d  elapsed %.0f min' % (done_n, total, elapsed))

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(out_dir, 'results.csv'), index=False)
    return df


def summarize(df, out_dir, strategies, conditions):
    """Summary tables and figures from the results."""
    df = df[df['condition'].isin(conditions) & df['strategy'].isin(strategies)].copy()
    if df.empty:
        print('No results to summarize.')
        return

    order = [c for c in CONDITION_ORDER if c in conditions]
    n_seeds = df['seed'].nunique()

    split_path = os.path.join(out_dir, 'split_log.csv')
    split = pd.read_csv(split_path) if os.path.exists(split_path) else pd.DataFrame()
    prior_by_cond = (split.groupby('condition')['train_malignant_frac'].mean().to_dict()
                     if len(split) else {})
    natural_prior = (float(split['natural_malignant_frac'].mean())
                     if len(split) and 'natural_malignant_frac' in split else np.nan)

    rows = []
    for cond in order:
        for strat in strategies:
            sub = df[(df.condition == cond) & (df.strategy == strat)]
            if sub.empty:
                continue
            rec = {
                'condition': cond,
                'condition_label': CONDITIONS[cond]['label'],
                'achieved_r_mean': sub['achieved_r'].mean(),
                'n_train_mean': sub['n_train_used'].mean(),
                'train_malignant_frac_mean': prior_by_cond.get(cond, np.nan),
                'strategy': strat,
                'n_seeds': sub['seed'].nunique(),
            }
            for m, _, _ in METRICS:
                rec['%s_mean' % m] = sub[m].mean()
                rec['%s_sd' % m] = sub[m].std(ddof=1)
            rows.append(rec)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, 'condition_summary.csv'), index=False)

    print('\n' + '=' * 78)
    print('PER-CONDITION SUMMARY (mean +/- SD across seeds)')
    print('=' * 78)
    for cond in order:
        s = summary[summary.condition == cond]
        if s.empty:
            continue
        print('\n%-16s %s   achieved r=%.3f   n_train=%.0f   malignant frac=%.3f'
              % (cond, CONDITIONS[cond]['label'],
                 s['achieved_r_mean'].iloc[0], s['n_train_mean'].iloc[0],
                 s['train_malignant_frac_mean'].iloc[0]))
        print('  %-10s %18s %18s %18s %18s'
              % ('strategy', 'gap', 'SAS', 'probe(lin)', 'AUROC'))
        for _, r in s.iterrows():
            print('  %-10s %9.4f+-%-7.4f %9.4f+-%-7.4f %9.4f+-%-7.4f %9.4f+-%-7.4f'
                  % (r['strategy'],
                     r['worst_group_gap_mean'], r['worst_group_gap_sd'],
                     r['sas_mean'], r['sas_sd'],
                     r['probe_linear_mean_mean'], r['probe_linear_mean_sd'],
                     r['auroc_all_mean'], r['auroc_all_sd']))

    dose_rows = []
    for cond in order:
        for strat in strategies:
            sub = df[(df.condition == cond) & (df.strategy == strat)]
            if sub.empty:
                continue
            rec = {'condition': cond,
                   'strategy': strat,
                   'achieved_r': sub['achieved_r'].mean(),
                   'n_train': sub['n_train_used'].mean(),
                   'train_malignant_frac': prior_by_cond.get(cond, np.nan)}
            for m, _, _ in METRICS:
                rec[m] = sub[m].mean()
            dose_rows.append(rec)
    dose = pd.DataFrame(dose_rows)
    dose.to_csv(os.path.join(out_dir, 'dose_response.csv'), index=False)

    ctab = dose.drop_duplicates('condition').set_index('condition')
    print('\n' + '=' * 78)
    print('WHAT EACH CONDITION\'S TRAINING FOLD ACTUALLY CONTAINS')
    print('=' * 78)
    print('%-16s %8s %9s %12s' % ('condition', 'r', 'n_train', 'malignant frac'))
    for cond in order:
        if cond in ctab.index:
            r = ctab.loc[cond]
            print('%-16s %8.3f %9.0f %12.3f' % (cond, r['achieved_r'], r['n_train'],
                                                r['train_malignant_frac']))
    if not np.isnan(natural_prior):
        print('(natural malignant fraction in the training split: %.3f)' % natural_prior)
    fam = [c for c in DOSE_FAMILY if c in ctab.index]
    if len(fam) >= 3:
        cc = np.corrcoef(ctab.loc[fam, 'achieved_r'], ctab.loc[fam, 'train_malignant_frac'])[0, 1]
        print('corr(achieved r, malignant fraction) across the dose-response conditions = %+.3f' % cc)
        if cc > 0.8:
            print('  WARNING: in the core_pipeline-sampler conditions the class prior rises with r, so a\n'
                  '  rising gap is NOT by itself evidence of a density shortcut. The decisive\n'
                  '  evidence is the balanced_r06 vs sizematched_r06 contrast below.')

    base = dose[(dose.strategy == 'baseline') & (dose.condition.isin(DOSE_FAMILY))] \
        .sort_values('achieved_r')
    if len(base) >= 3:
        print('\n' + '=' * 78)
        print('DOSE-RESPONSE, BASELINE ONLY  -- does shortcut reliance track correlation?')
        print('=' * 78)
        print('%-16s %8s %9s %9s %9s %9s'
              % ('condition', 'r', 'gap', 'SAS', 'probe', 'AUROC'))
        for _, r in base.iterrows():
            print('%-16s %8.3f %9.4f %9.4f %9.4f %9.4f'
                  % (r['condition'], r['achieved_r'], r['worst_group_gap'],
                     r['sas'], r['probe_linear_mean'], r['auroc_all']))
        for m, name, _ in METRICS:
            v = base[['achieved_r', m]].dropna()
            if len(v) >= 3:
                rr = np.corrcoef(v['achieved_r'], v[m])[0, 1]
                print('  Pearson r(achieved_r, %-18s) across conditions = %+.3f' % (m, rr))
        print('\n  How to read this. Worst-group gap and probe accuracy are EXPECTED to rise')
        print('  if the model is using density. SAS is different: it measures attribution')
        print('  OUTSIDE the breast mask, but density information sits INSIDE the breast, so')
        print('  SAS has no reason to rise and may fall -- report whichever direction it takes')
        print('  and say why. Main-task AUROC is confounded by n_train and class prior.')
        print('  These are %d condition means, so the Pearson values are descriptive only.'
              % len(base))

    control_comparison(df, out_dir, strategies)

    pair_rows = []
    min_p = 2 * (0.5 ** n_seeds)
    for cond in order:
        cdf = df[df.condition == cond]
        bl = cdf[cdf.strategy == 'baseline'].set_index('seed')
        if bl.empty:
            continue
        for strat in strategies:
            if strat == 'baseline':
                continue
            arm = cdf[cdf.strategy == strat].set_index('seed')
            common = bl.index.intersection(arm.index)
            if len(common) < 3:
                continue
            for m, _, better in METRICS:
                b = bl.loc[common, m].values
                a = arm.loc[common, m].values
                ok = ~(np.isnan(a) | np.isnan(b))
                if ok.sum() < 3:
                    continue
                diffs = a[ok] - b[ok]
                mean_d, lo, hi = core_pipeline.bootstrap_ci(diffs)
                if np.allclose(diffs, 0.0):
                    pv = 1.0
                else:
                    try:
                        with np.errstate(invalid='ignore', divide='ignore'):
                            _, pv = wilcoxon(a[ok], b[ok])
                    except ValueError:
                        pv = np.nan
                pair_rows.append({
                    'condition': cond,
                    'achieved_r': cdf['achieved_r'].mean(),
                    'strategy': strat,
                    'metric': m,
                    'want': better,
                    'n_pairs': int(ok.sum()),
                    'mean_diff': mean_d,
                    'ci_lo': lo,
                    'ci_hi': hi,
                    'wilcoxon_p': pv,
                    'ci_excludes_zero': bool((lo > 0 and hi > 0) or (lo < 0 and hi < 0)),
                })
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(os.path.join(out_dir, 'paired_vs_baseline.csv'), index=False)

    if not pairs.empty:
        print('\n' + '=' * 78)
        print('PAIRED vs BASELINE, WITHIN EACH CONDITION (same seeds)')
        print('With n=%d paired seeds the MINIMUM achievable Wilcoxon two-sided p'
              % n_seeds)
        print('is %.4f. Treat the bootstrap CI as the primary evidence.' % min_p)
        print('=' * 78)
        for cond in order:
            sub = pairs[pairs.condition == cond]
            if sub.empty:
                continue
            print('\n%s (r=%.3f)' % (cond, sub['achieved_r'].iloc[0]))
            for _, r in sub.iterrows():
                flag = '  *' if r['ci_excludes_zero'] else '   '
                print('  %-9s %-18s diff=%+.4f  CI=[%+.4f, %+.4f]  p=%.4f%s'
                      % (r['strategy'], r['metric'], r['mean_diff'],
                         r['ci_lo'], r['ci_hi'], r['wilcoxon_p'], flag))
        print('\n  * = bootstrap 95% CI excludes zero.')

    plot_dose_response(dose, strategies, order, out_dir)
    plot_strategy_by_condition(summary, strategies, order, out_dir)


def control_comparison(df, out_dir, strategies):
    """Paired contrasts that separate correlation, training size and class prior."""
    have = set(df['condition'].unique())
    contrasts = [
        ('size effect (class prior held)', 'published_r06', 'sizematched_r06'),
        ('class-prior effect (n held)', 'sizematched_r06', 'balanced_r06'),
        ('CORRELATION effect (class prior AND n held)', 'balanced_r00', 'balanced_r06'),
    ]
    rows = []
    for label, a_cond, b_cond in contrasts:
        if a_cond not in have or b_cond not in have:
            continue
        for strat in strategies:
            A = df[(df.condition == a_cond) & (df.strategy == strat)].set_index('seed')
            B = df[(df.condition == b_cond) & (df.strategy == strat)].set_index('seed')
            common = A.index.intersection(B.index)
            if len(common) < 3:
                continue
            for m, _, _ in METRICS:
                a = A.loc[common, m].values.astype(float)
                b = B.loc[common, m].values.astype(float)
                ok = ~(np.isnan(a) | np.isnan(b))
                if ok.sum() < 3:
                    continue
                diffs = b[ok] - a[ok]
                mean_d, lo, hi = core_pipeline.bootstrap_ci(diffs)
                rows.append({
                    'contrast': label, 'from': a_cond, 'to': b_cond, 'strategy': strat,
                    'metric': m, 'n_pairs': int(ok.sum()),
                    'mean_from': float(a[ok].mean()), 'mean_to': float(b[ok].mean()),
                    'mean_diff': mean_d, 'ci_lo': lo, 'ci_hi': hi,
                    'ci_excludes_zero': bool((lo > 0 and hi > 0) or (lo < 0 and hi < 0)),
                })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, 'control_comparison.csv'), index=False)
    if out.empty:
        return

    print('\n' + '=' * 78)
    print('CONTROLS AT r ~ 0.62: SEPARATING TRAINING-SET SIZE FROM CLASS PRIOR')
    print('=' * 78)
    for label, a_cond, b_cond in contrasts:
        sub = out[(out.contrast == label) & (out.strategy == 'baseline')]
        if sub.empty:
            continue
        print('\n%s :  %s -> %s   (baseline arm, paired by seed)' % (label, a_cond, b_cond))
        for _, r in sub.iterrows():
            flag = ' *' if r['ci_excludes_zero'] else '  '
            print('  %-18s %8.4f -> %8.4f   diff=%+.4f  CI=[%+.4f, %+.4f]%s'
                  % (r['metric'], r['mean_from'], r['mean_to'], r['mean_diff'],
                     r['ci_lo'], r['ci_hi'], flag))

    def _base_mean(cond, m):
        v = df[(df.condition == cond) & (df.strategy == 'baseline')][m]
        return float(v.mean()) if len(v) else np.nan

    core = out[(out.contrast.str.startswith('CORRELATION')) & (out.strategy == 'baseline')]
    g0, g6 = (_base_mean(c, 'worst_group_gap') for c in ('balanced_r00', 'balanced_r06'))
    gap_row = core[core.metric == 'worst_group_gap']
    if len(gap_row) and not (np.isnan(g0) or np.isnan(g6)):
        gr = gap_row.iloc[0]
        print('\nKEY RESULT (baseline arm, natural class prior, n held fixed):')
        print('  raising the training correlation from r~0 to r~0.62 moves the worst-group gap '
              '%.3f -> %.3f' % (g0, g6))
        print('  (paired diff %+.3f, 95%% CI [%+.3f, %+.3f]).' % (gr['mean_diff'], gr['ci_lo'],
                                                                gr['ci_hi']))
        if gr['ci_excludes_zero'] and gr['mean_diff'] > 0:
            print('  The CI excludes zero: density is used as a shortcut even when the class prior\n'
                  '  and the training-set size are held fixed.')
        else:
            print('  The CI does not show a positive effect once class prior and n are held\n'
                  '  fixed, so part of the main-run gap may come from class prior or sample size.')
    print('  * = bootstrap 95%% CI excludes zero (n=%d seeds; CI is the primary evidence).'
          % out['n_pairs'].max())


def plot_dose_response(dose, strategies, order, out_dir):
    """Worst-group gap against the injected correlation."""
    if dose.empty:
        return
    from matplotlib.lines import Line2D
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fam = [c for c in order if c in DOSE_FAMILY and c != 'natural']
    has_nat = 'natural' in order
    ctrl = [c for c in order if c in CONTROL_CONDITIONS]
    ctrl_style = {'balanced_r06': dict(marker='*', s=150),
                  'balanced_r00': dict(marker='h', s=80),
                  'sizematched_r06': dict(marker='D', s=55)}
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5.4 * len(METRICS), 4.4))
    if len(METRICS) == 1:
        axes = [axes]
    markers = ['o', 's', '^', 'v', 'P', 'X']
    for ax, (m, title, _) in zip(axes, METRICS):
        for i, strat in enumerate(strategies):
            col = colors[i % len(colors)]
            sub = dose[(dose.strategy == strat) & (dose.condition.isin(fam))] \
                .sort_values('achieved_r')
            if not sub.empty:
                ax.plot(sub['achieved_r'], sub[m], marker=markers[i % len(markers)],
                        linewidth=1.8, markersize=6, color=col, label=strat)
            if has_nat:
                ns = dose[(dose.strategy == strat) & (dose.condition == 'natural')]
                if not ns.empty:
                    ax.scatter(ns['achieved_r'], ns[m], marker='+', s=130, linewidths=2.2,
                               color=col, zorder=6)
            for c in ctrl:
                cs = dose[(dose.strategy == strat) & (dose.condition == c)]
                if cs.empty:
                    continue
                ax.scatter(cs['achieved_r'], cs[m], facecolors='none', edgecolors=col,
                           linewidths=1.6, zorder=5, **ctrl_style[c])
        ax.set_xlabel('Achieved training-set point-biserial r')
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, linestyle=':')
    axes[0].set_ylabel('Metric value (mean over seeds)')
    handles, labels = axes[-1].get_legend_handles_labels()
    if has_nat:
        handles.append(Line2D([], [], linestyle='', marker='+', color='k', markersize=10,
                              markeredgewidth=2))
        labels.append('natural data (n~1980, no injection; not on the line)')
    for c in ctrl:
        st = ctrl_style[c]
        handles.append(Line2D([], [], linestyle='', marker=st['marker'], markerfacecolor='none',
                              markeredgecolor='k', markersize=9 if st['marker'] == '*' else 6))
        labels.append({'balanced_r06': 'class-balanced, r~0.62 (natural prior)',
                       'balanced_r00': 'class-balanced, r~0 (null for the above)',
                       'sizematched_r06': 'size-matched (published prior, same n)'}[c])
    fig.legend(handles, labels, fontsize=8, frameon=False, loc='lower center',
               ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Shortcut reliance vs training-set correlation  '
                 '(lines = core_pipeline-sampler arms, n~570-970, class prior rises with r; '
                 'hollow markers = controls, n~360)', fontsize=10.5)
    fig.tight_layout(rect=[0, 0.14, 1, 0.93])
    p = os.path.join(out_dir, 'dose_response.png')
    fig.savefig(p, dpi=220)
    plt.close(fig)
    print('\nWrote %s' % p)


def plot_strategy_by_condition(summary, strategies, order, out_dir):
    """Each strategy across all conditions."""
    if summary.empty:
        return
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5.2 * len(METRICS), 4.4))
    if len(METRICS) == 1:
        axes = [axes]
    width = 0.8 / max(len(strategies), 1)
    x = np.arange(len(order))
    for ax, (m, title, _) in zip(axes, METRICS):
        for i, strat in enumerate(strategies):
            means, sds = [], []
            for cond in order:
                r = summary[(summary.condition == cond) & (summary.strategy == strat)]
                means.append(r['%s_mean' % m].iloc[0] if not r.empty else np.nan)
                sds.append(r['%s_sd' % m].iloc[0] if not r.empty else np.nan)
            ax.bar(x + i * width - 0.4 + width / 2, means, width,
                   yerr=sds, capsize=3, label=strat)
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITIONS[c]['label'].split('(')[0].strip() for c in order],
                           rotation=30, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis='y', linestyle=':')
    axes[-1].legend(fontsize=8, frameon=False)
    fig.suptitle('Each strategy across correlation conditions '
                 '(bars = mean, whiskers = SD across seeds)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(out_dir, 'strategy_by_condition.png')
    fig.savefig(p, dpi=220)
    plt.close(fig)
    print('Wrote %s' % p)


def parse_args():
    ap = argparse.ArgumentParser(
        description='Shortcut-strength sweep on CBIS-DDSM')
    ap.add_argument('--seeds', type=int, default=5,
                    help='how many seeds from the front of core_pipeline.SEED_LIST (default 5)')
    ap.add_argument('--smoke', action='store_true',
                    help='1 seed / 2 conditions / 2 strategies -- plumbing test')
    ap.add_argument('--strategies', nargs='+', default=None,
                    help='subset of %s' % core_pipeline.ABLATIONS)
    ap.add_argument('--conditions', nargs='+', default=None,
                    help='subset of %s' % CONDITION_ORDER)
    ap.add_argument('--summarize-only', action='store_true',
                    help='rebuild tables and figures from the checkpoint, no training')
    ap.add_argument('--dry-run', action='store_true',
                    help='build and log every training fold, print n / r / class prior, '
                         'then stop -- trains nothing')
    ap.add_argument('--import-published', action=argparse.BooleanOptionalAction, default=True,
                    help="import published_r06 from the paper's dccam_cbis_results.csv "
                         'instead of retraining it (default on; --no-import-published to retrain)')
    ap.add_argument('--published-csv', default=None,
                    help="explicit path to the paper's dccam_cbis_results.csv")
    args = ap.parse_args()

    args.strategies = args.strategies or list(DEFAULT_STRATEGIES)
    args.conditions = args.conditions or list(CONDITION_ORDER)
    if args.smoke:
        args.seeds = 1
        args.conditions = ['natural', 'published_r06']
        args.strategies = ['baseline', 'distcorr']

    bad = [c for c in args.conditions if c not in CONDITIONS]
    if bad:
        ap.error('unknown condition(s): %s  (valid: %s)' % (bad, CONDITION_ORDER))
    bad = [s for s in args.strategies if s not in core_pipeline.ABLATIONS]
    if bad:
        ap.error('unknown strategy(ies): %s -- core_pipeline knows %s' % (bad, core_pipeline.ABLATIONS))
    args.conditions = [c for c in CONDITION_ORDER if c in args.conditions]
    return args


def main():
    args = parse_args()

    out_dir = os.path.join(core_pipeline.OUT_DIR, OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 78)
    print('SHORTCUT-STRENGTH SWEEP')
    print('=' * 78)
    print('core_pipeline OUT_DIR (untouched) : %s' % core_pipeline.OUT_DIR)
    print('image cache               : %s (exists=%s)'
          % (core_pipeline.CACHE_FILE, os.path.exists(core_pipeline.CACHE_FILE)))
    print('writing results to        : %s' % out_dir)
    print('seeds                     : %s' % list(core_pipeline.SEED_LIST)[:args.seeds])
    print('conditions                : %s' % args.conditions)
    print('strategies                : %s' % args.strategies)
    if not args.summarize_only and not args.dry_run:
        n_cells = len(args.conditions) * args.seeds * len(args.strategies)
        n_pub = (args.seeds * len(args.strategies)
                 if (args.import_published and PUBLISHED_CONDITION in args.conditions) else 0)
        print('result cells              : %d  (up to %d imported from the paper CSV, '
              'so at most %d trained)' % (n_cells, n_pub, n_cells - n_pub))
    if args.dry_run:
        print('MODE                      : DRY RUN -- nothing will be trained')
    if not os.path.exists(core_pipeline.CACHE_FILE):
        print('\nNOTE: no image cache found. The first condition will build it '
              '(one-time cost,\n      then reused by every later run).')
    print('=' * 78)

    t0 = time.time()

    if args.summarize_only:
        ck = os.path.join(out_dir, 'results.partial.csv')
        full = os.path.join(out_dir, 'results.csv')
        src = full if os.path.exists(full) else ck
        if not os.path.exists(src):
            print('Nothing to summarize: no results file in %s' % out_dir)
            return
        print('Summarizing from %s' % src)
        df = pd.read_csv(src).drop_duplicates(
            subset=['condition', 'seed', 'strategy'], keep='last')
        df.to_csv(os.path.join(out_dir, 'results.csv'), index=False)
    else:
        df = run_sweep(args, out_dir)
        if args.dry_run:
            print('\nDry run finished. If the table above looks right, run without --dry-run.')
            return

    summarize(df, out_dir, args.strategies, args.conditions)

    manifest = {
        'script': os.path.basename(__file__),
        'purpose': 'shortcut-strength sweep on CBIS-DDSM',
        'run_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_minutes': round((time.time() - t0) / 60.0, 2),
        'summarize_only': args.summarize_only,
        'seeds': list(core_pipeline.SEED_LIST)[:args.seeds],
        'conditions': {c: CONDITIONS[c] for c in args.conditions},
        'strategies': args.strategies,
        'n_rows': int(len(df)),
        'import_published': bool(args.import_published),
        'published_csv_used': getattr(args, '_published_csv_used', None),
        'rows_by_source': (df['source'].value_counts().to_dict()
                           if 'source' in df.columns else {}),
        'pipeline_config': {
            'IMG_SIZE': core_pipeline.IMG_SIZE,
            'BATCH_SIZE': core_pipeline.BATCH_SIZE,
            'MAX_EPOCHS': core_pipeline.MAX_EPOCHS,
            'PATIENCE': core_pipeline.PATIENCE,
            'LR': core_pipeline.LR,
            'LAMBDA_RAMP_EPOCHS': core_pipeline.LAMBDA_RAMP_EPOCHS,
            'MAX_LAMBDA_GRL': core_pipeline.MAX_LAMBDA_GRL,
            'GRL_ADV_WEIGHT': core_pipeline.GRL_ADV_WEIGHT,
            'MAX_WEIGHT_DISTCORR': core_pipeline.MAX_WEIGHT_DISTCORR,
            'MAX_WEIGHT_CAMREG': core_pipeline.MAX_WEIGHT_CAMREG,
            'DATA_ROOT': core_pipeline.DATA_ROOT,
            'CACHE_FILE': core_pipeline.CACHE_FILE,
        },
        'determinism': determinism_snapshot(),
    }
    with open(os.path.join(out_dir, 'run_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    print('\n' + '=' * 78)
    print('DONE in %.1f min. Everything is in:\n  %s'
          % ((time.time() - t0) / 60.0, out_dir))
    print('=' * 78)
    print('Key output files:')
    print('  dose_response.png / .csv  -- shortcut reliance vs correlation strength')
    print('  split_log.csv             -- n_train, achieved r, class prior per seed')
    print('  control_comparison.csv    -- size vs class-prior contrasts at r~0.62')


if __name__ == '__main__':
    main()
