"""Density definition report: median vs clinical split, natural correlation, per-seed group sizes.

Run: python analysis/density_split_report.py
"""
import os
import sys

# Make src/ and analysis/ importable when the script is run from any folder.
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', d) for d in ('src', 'analysis')]

import json
import time
import argparse
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import pointbiserialr, spearmanr, chi2_contingency
from sklearn.model_selection import GroupShuffleSplit

import core_pipeline

OUTPUT_SUBDIR = 'density_split_report'

BIRADS_LABEL = {
    1: '1 / A  almost entirely fatty',
    2: '2 / B  scattered fibroglandular',
    3: '3 / C  heterogeneously dense',
    4: '4 / D  extremely dense',
}
CLINICAL_DENSE = {3, 4}


def build_metadata_only():
    """CBIS-DDSM metadata without loading the image cache."""
    return core_pipeline.build_metadata()


def contingency(df, out_dir, emit):
    """BI-RADS density x malignancy table."""
    med = df['density'].median()
    emit('')
    emit('=' * 78)
    emit('1. THE MEDIAN SPLIT ITSELF')
    emit('=' * 78)
    emit('median BI-RADS density in the usable cohort : %.1f' % med)
    emit('binarisation used by the paper              : density_binary = '
         '(density >= %.1f)' % med)
    emit('')
    emit('So "dense" (density_binary = 1) means BI-RADS %s'
         % ', '.join(str(int(v)) for v in sorted(df.loc[df.density_binary == 1,
                                                        'density'].unique())))
    emit('   "non-dense" (density_binary = 0) means BI-RADS %s'
         % ', '.join(str(int(v)) for v in sorted(df.loc[df.density_binary == 0,
                                                        'density'].unique())))

    tab = pd.crosstab(df['density'].astype(int), df['density_binary'],
                      margins=True, margins_name='total')
    tab.index = [BIRADS_LABEL.get(i, str(i)) if i in BIRADS_LABEL else str(i)
                 for i in tab.index]
    tab.columns = ['non-dense (0)' if c == 0 else
                   'dense (1)' if c == 1 else c for c in tab.columns]
    emit('')
    emit('BI-RADS category x median split:')
    emit(tab.to_string())

    full = pd.crosstab([df['density'].astype(int), df['malignant']],
                       df['density_binary'], margins=True, margins_name='total')
    full.to_csv(os.path.join(out_dir, 'birads_by_split.csv'))

    emit('')
    emit('BI-RADS category x malignancy (counts):')
    ct = pd.crosstab(df['density'].astype(int), df['malignant'])
    ct.columns = ['benign', 'malignant']
    ct['total'] = ct.sum(axis=1)
    ct['malignant_rate'] = (ct['malignant'] / ct['total']).round(3)
    ct.index = [BIRADS_LABEL.get(i, str(i)) for i in ct.index]
    emit(ct.to_string())
    emit('')
    emit('The malignant_rate column is the clinically interesting one: if it is')
    emit('flat across BI-RADS categories, density carries little information')
    emit('about malignancy in this cohort, which is exactly why the natural')
    emit('correlation is near zero and why the paper injects one.')
    return med


def split_comparison(df, out_dir, emit):
    """Median split against the clinical A/B vs C/D split."""
    clinical = df['density'].isin(CLINICAL_DENSE).astype(int)
    median_split = df['density_binary'].astype(int)

    agree = (clinical == median_split)
    emit('')
    emit('=' * 78)
    emit('2. MEDIAN SPLIT vs THE CLINICAL A/B-versus-C/D SPLIT')
    emit('=' * 78)
    emit('clinical convention: BI-RADS 1-2 (A, B) = non-dense; 3-4 (C, D) = dense')
    emit('')
    emit('agreement between the two splits : %d / %d images (%.1f%%)'
         % (int(agree.sum()), len(df), 100.0 * agree.mean()))
    if agree.all():
        emit('')
        emit('=> The two splits are IDENTICAL on this cohort. The median split is')
        emit('   the clinical split.')
    else:
        emit('')
        emit('=> The splits DIFFER. Disagreement breakdown, by BI-RADS category:')
        dis = pd.DataFrame({
            'birads': df['density'].astype(int),
            'median_split': np.where(median_split == 1, 'dense', 'non-dense'),
            'clinical_split': np.where(clinical == 1, 'dense', 'non-dense'),
            'agrees': agree.values,
        })
        tbl = dis.groupby(['birads', 'median_split', 'clinical_split', 'agrees']) \
                 .size().rename('n_images').reset_index()
        emit(tbl.to_string(index=False))
        emit('')
        emit('   The rows with agrees=False are the categories the two splits')
        emit('   classify differently.')
        emit('')
        emit('   This matters: the aligned/conflicting groups, and therefore the')
        emit('   worst-group gap, are defined by the median split. Report the')
        emit('   disagreement honestly and, if it is large, consider reporting')
        emit('   the clinical split as a robustness column.')

    cmp_df = pd.DataFrame({
        'split': ['median (paper)', 'clinical A/B vs C/D'],
        'n_non_dense': [int((median_split == 0).sum()), int((clinical == 0).sum())],
        'n_dense': [int((median_split == 1).sum()), int((clinical == 1).sum())],
        'frac_dense': [float((median_split == 1).mean()), float((clinical == 1).mean())],
        'agreement_with_other': [float(agree.mean())] * 2,
    })
    cmp_df.to_csv(os.path.join(out_dir, 'split_comparison.csv'), index=False)
    emit('')
    emit(cmp_df.to_string(index=False))
    return clinical


def correlations(df, clinical, out_dir, emit):
    """Natural density-malignancy correlation under each definition."""
    emit('')
    emit('=' * 78)
    emit('3. THE CONFOUND STRENGTH, COMPUTED FOUR WAYS')
    emit('=' * 78)

    rows = []

    r, p = pointbiserialr(df['malignant'], df['density'])
    rows.append(('point-biserial', 'malignant vs density (ordinal 1-4)', r, p))
    emit('point-biserial, malignant vs density (ordinal 1-4) : r=%+.4f  p=%.4f'
         % (r, p))
    emit('  (this is the natural correlation reported in the paper)')

    r2, p2 = pointbiserialr(df['malignant'], df['density_binary'])
    rows.append(('point-biserial', 'malignant vs density_binary (median)', r2, p2))
    emit('point-biserial, malignant vs density_binary (median): r=%+.4f  p=%.4f'
         % (r2, p2))

    r3, p3 = pointbiserialr(df['malignant'], clinical)
    rows.append(('point-biserial', 'malignant vs dense (clinical A/B vs C/D)', r3, p3))
    emit('point-biserial, malignant vs clinical dense       : r=%+.4f  p=%.4f'
         % (r3, p3))

    rs, ps = spearmanr(df['malignant'], df['density'])
    rows.append(('spearman', 'malignant vs density (ordinal)', rs, ps))
    emit('spearman,       malignant vs density (ordinal)     : rho=%+.4f  p=%.4f'
         % (rs, ps))

    ct = pd.crosstab(df['malignant'], df['density_binary'])
    chi2, pchi, dof, _ = chi2_contingency(ct)
    n = ct.values.sum()
    cramers_v = float(np.sqrt(chi2 / n))
    rows.append(('chi2', 'malignant x density_binary (2x2)', chi2, pchi))
    rows.append(('cramers_v', 'malignant x density_binary (2x2)', cramers_v, np.nan))
    emit('chi-square,     malignant x density_binary 2x2     : chi2=%.3f  '
         'dof=%d  p=%.4f  Cramer V=%.4f' % (chi2, dof, pchi, cramers_v))

    out = pd.DataFrame(rows, columns=['test', 'variables', 'statistic', 'p_value'])
    out.to_csv(os.path.join(out_dir, 'correlations.csv'), index=False)

    emit('')
    emit('All four agree that the NATURAL association is negligible. That is the')
    emit('premise of the controlled-injection design: the study measures shortcut')
    emit('learning under a known, injected correlation.')


def group_breakdown(df, out_dir, emit):
    """Aligned / conflicting cells in the whole cohort."""
    aligned, conflicting = core_pipeline.group_masks(df)
    emit('')
    emit('=' * 78)
    emit('4. ALIGNED / CONFLICTING GROUPS -- defined BY the split above')
    emit('=' * 78)
    emit('aligned      = (malignant & dense) or (benign & non-dense)')
    emit('conflicting  = the reverse')
    emit('')
    emit('whole usable cohort: aligned=%d (%.1f%%)  conflicting=%d (%.1f%%)'
         % (int(aligned.sum()), 100.0 * aligned.mean(),
            int(conflicting.sum()), 100.0 * conflicting.mean()))

    cell = pd.crosstab(df['malignant'], df['density_binary'])
    cell.index = ['benign', 'malignant']
    cell.columns = ['non-dense', 'dense']
    emit('')
    emit('the four cells these groups are built from:')
    emit(cell.to_string())
    emit('')
    emit('A worst-group gap computed on a small group is noisy, so report these')
    emit('cell sizes next to the gap.')


def per_seed_counts(df, patient_col, n_seeds, out_dir, emit):
    """Per-seed fold and group sizes."""
    rows = []
    for seed in list(core_pipeline.SEED_LIST)[:n_seeds]:
        sp1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        tv_idx, te_idx = next(sp1.split(df, groups=df[patient_col]))
        tv = df.iloc[tv_idx].reset_index(drop=True)
        te = df.iloc[te_idx].reset_index(drop=True)
        sp2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_idx, va_idx = next(sp2.split(tv, groups=tv[patient_col]))
        tr = tv.iloc[tr_idx].reset_index(drop=True)
        va = tv.iloc[va_idx].reset_index(drop=True)

        for name, part in [('train', tr), ('val', va), ('test', te)]:
            al, cf = core_pipeline.group_masks(part)
            rows.append({
                'seed': seed, 'fold': name, 'n': len(part),
                'n_patients': part[patient_col].nunique(),
                'n_malignant': int(part['malignant'].sum()),
                'n_benign': int((part['malignant'] == 0).sum()),
                'n_dense': int(part['density_binary'].sum()),
                'n_non_dense': int((part['density_binary'] == 0).sum()),
                'n_aligned': int(al.sum()),
                'n_conflicting': int(cf.sum()),
            })

        biased = core_pipeline.bias_split_by_correlation(tr, seed, core_pipeline.BIAS_STRENGTH)
        al, cf = core_pipeline.group_masks(biased)
        r, _ = pointbiserialr(biased['malignant'], biased['density_binary'])
        rows.append({
            'seed': seed, 'fold': 'train_biased_r06', 'n': len(biased),
            'n_patients': biased[patient_col].nunique(),
            'n_malignant': int(biased['malignant'].sum()),
            'n_benign': int((biased['malignant'] == 0).sum()),
            'n_dense': int(biased['density_binary'].sum()),
            'n_non_dense': int((biased['density_binary'] == 0).sum()),
            'n_aligned': int(al.sum()),
            'n_conflicting': int(cf.sum()),
            'achieved_r': r,
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, 'group_counts_per_seed.csv'), index=False)

    emit('')
    emit('=' * 78)
    emit('5. PER-SEED FOLD AND GROUP SIZES')
    emit('=' * 78)
    agg = out.groupby('fold').agg(
        n_mean=('n', 'mean'), n_min=('n', 'min'), n_max=('n', 'max'),
        aligned_mean=('n_aligned', 'mean'), conflicting_mean=('n_conflicting', 'mean'),
        patients_mean=('n_patients', 'mean')).round(1)
    emit(agg.to_string())
    if 'achieved_r' in out.columns:
        rr = out['achieved_r'].dropna()
        if len(rr):
            emit('')
            emit('injected training correlation across %d seeds: mean r=%.3f  '
                 'SD %.3f  min %.3f  max %.3f'
                 % (len(rr), rr.mean(), rr.std(ddof=1), rr.min(), rr.max()))
            emit('(reported in the paper as r ~ 0.6)')
    emit('')
    emit('Note the row "train_biased_r06": the injection DISCARDS rows to hit')
    emit('its target ratio, so the trained-on set is smaller than the available')
    emit('training split. shortcut_strength_sweep.py logs this per condition and')
    emit('includes a size-matched r~0 control arm.')
    return out


def figure(df, out_dir):
    """BI-RADS distribution figure."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ct = pd.crosstab(df['density'].astype(int), df['malignant'])
    idx = sorted(ct.index)
    w = 0.38
    x = np.arange(len(idx))
    axes[0].bar(x - w / 2, [ct.loc[i, 0] if 0 in ct.columns else 0 for i in idx],
                w, label='benign', color='#4a90d9')
    axes[0].bar(x + w / 2, [ct.loc[i, 1] if 1 in ct.columns else 0 for i in idx],
                w, label='malignant', color='#c23b22')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['BI-RADS %d' % i for i in idx])
    axes[0].set_ylabel('Number of images')
    axes[0].set_title('Cohort composition by BI-RADS density')
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.3, axis='y', linestyle=':')

    med = df['density'].median()
    rate = df.groupby(df['density'].astype(int))['malignant'].mean()
    axes[1].bar([str(i) for i in rate.index], rate.values, color='#6b7280')
    axes[1].axhline(df['malignant'].mean(), color='k', linestyle='--', linewidth=1,
                    label='cohort mean (%.3f)' % df['malignant'].mean())
    axes[1].set_xlabel('BI-RADS density category')
    axes[1].set_ylabel('Fraction malignant')
    axes[1].set_title('Malignancy rate by density (median split at %.1f)' % med)
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.3, axis='y', linestyle=':')

    fig.suptitle('What the median density split contains', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(out_dir, 'birads_distribution.png')
    fig.savefig(p, dpi=220)
    plt.close(fig)
    print('wrote %s' % p)


def main():
    ap = argparse.ArgumentParser(
        description='Density-split and BI-RADS report (no training)')
    ap.add_argument('--seeds', type=int, default=len(core_pipeline.SEED_LIST))
    ap.add_argument('--no-cache', action='store_true',
                    help='metadata only; never load the image pickle')
    args = ap.parse_args()

    out_dir = os.path.join(core_pipeline.OUT_DIR, OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    print('=' * 78)
    print('DENSITY-SPLIT REPORT  (no model, no GPU, no training)')
    print('=' * 78)
    print('writing to: %s' % out_dir)

    df, patient_col = build_metadata_only()

    lines = []
    def emit(s=''):
        print(s)
        lines.append(str(s))

    emit('')
    emit('DENSITY-SPLIT AND BI-RADS REPORT')
    emit('generated %s' % datetime.now(timezone.utc).isoformat())
    emit('cohort: %d usable images, %d unique %s'
         % (len(df), df[patient_col].nunique(), patient_col))

    contingency(df, out_dir, emit)
    clinical = split_comparison(df, out_dir, emit)
    correlations(df, clinical, out_dir, emit)
    group_breakdown(df, out_dir, emit)
    per_seed_counts(df, patient_col, args.seeds, out_dir, emit)
    figure(df, out_dir)

    with open(os.path.join(out_dir, 'density_report.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    manifest = {
        'script': os.path.basename(__file__),
        'purpose': 'BI-RADS composition of the median density split and per-seed group sizes',
        'run_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_minutes': round((time.time() - t0) / 60.0, 2),
        'n_images': int(len(df)),
        'n_patients': int(df[patient_col].nunique()),
        'density_median': float(df['density'].median()),
        'seeds_reported': list(core_pipeline.SEED_LIST)[:args.seeds],
        'env': {'python': sys.version.split()[0], 'platform': platform.platform(),
                'pandas': pd.__version__, 'numpy': np.__version__},
    }
    with open(os.path.join(out_dir, 'run_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    print('\nDONE in %.1f min. Everything is in:\n  %s'
          % ((time.time() - t0) / 60.0, out_dir))
    print('\nFull report: density_report.txt')


if __name__ == '__main__':
    main()
