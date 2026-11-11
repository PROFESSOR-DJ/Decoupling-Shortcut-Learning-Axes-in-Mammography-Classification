"""Otsu breast-mask audit: area, objects, contrast, height span and native-resolution Dice.

Run: python analysis/mask_audit.py
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
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import core_pipeline

OUTPUT_SUBDIR = 'mask_audit'

AREA_MIN_FRAC = 0.15
AREA_MAX_FRAC = 0.92
DISCARD_FRAC = 0.08
TOP_BOTTOM_FRAC = 0.80
MORPH_RADII = [3, 5, 9]


def connected_component_stats(img_gray):
    """Otsu components and the share discarded by the largest-component step."""
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, binm = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_fg = float((binm > 0).sum())
    if not contours:
        return dict(n_components=0, otsu_fg_frac=total_fg / binm.size,
                    discarded_frac=0.0, largest_frac=0.0)
    areas = sorted((cv2.contourArea(c) for c in contours), reverse=True)
    largest = areas[0]
    others = float(sum(areas[1:]))
    denom = largest + others
    return dict(
        n_components=len(contours),
        otsu_fg_frac=total_fg / binm.size,
        discarded_frac=(others / denom) if denom > 0 else 0.0,
        largest_frac=largest / binm.size,
    )


def border_contact(mask):
    """Fraction of each image edge covered by the mask."""
    m = mask > 0
    return dict(
        edge_top=float(m[0, :].mean()),
        edge_bottom=float(m[-1, :].mean()),
        edge_left=float(m[:, 0].mean()),
        edge_right=float(m[:, -1].mean()),
    )


def audit_one(fp, img_gray, mask):
    """All mask statistics and flags for one image."""
    rec = {'filepath': fp}
    rec['area_frac'] = float((mask > 0).mean())
    rec.update(connected_component_stats(img_gray))
    rec.update(border_contact(mask))

    m = mask > 0
    rec['mean_in'] = float(img_gray[m].mean()) if m.any() else np.nan
    rec['mean_out'] = float(img_gray[~m].mean()) if (~m).any() else np.nan
    rec['intensity_contrast'] = rec['mean_in'] - rec['mean_out']

    rec['flag_area_small'] = rec['area_frac'] < AREA_MIN_FRAC
    rec['flag_area_large'] = rec['area_frac'] > AREA_MAX_FRAC
    rec['flag_multi_object'] = rec['discarded_frac'] > DISCARD_FRAC
    rec['flag_spans_height'] = (rec['edge_top'] > TOP_BOTTOM_FRAC and
                                rec['edge_bottom'] > TOP_BOTTOM_FRAC)
    rec['flag_low_contrast'] = (not np.isnan(rec['intensity_contrast'])
                                and rec['intensity_contrast'] < 20.0)
    rec['flag_empty'] = rec['area_frac'] < 1e-6

    flags = [k for k in rec if k.startswith('flag_') and rec[k]]
    rec['n_flags'] = len(flags)
    rec['flags'] = ';'.join(f[5:] for f in flags)
    return rec


def perturbation_areas(mask):
    """Mask area after dilation and erosion."""
    out = {'area_frac_r0': float((mask > 0).mean())}
    for r in MORPH_RADII:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        out['area_frac_dilate%d' % r] = float((cv2.dilate(mask, k) > 0).mean())
        out['area_frac_erode%d' % r] = float((cv2.erode(mask, k) > 0).mean())
    return out


def contact_sheet(records, cache, title, out_path, ncols=6, nrows=4):
    """Grid of images with the mask contour and the flags that fired."""
    n = min(len(records), ncols * nrows)
    if n == 0:
        return False
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.75 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis('off')
    for ax, rec in zip(axes, records[:n]):
        fp = rec['filepath']
        if fp not in cache:
            continue
        img_gray, mask = cache[fp]
        ax.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
        cs, _ = cv2.findContours((mask > 0).astype(np.uint8),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            c = c.squeeze(1)
            if c.ndim == 2 and len(c) > 2:
                ax.plot(np.append(c[:, 0], c[0, 0]),
                        np.append(c[:, 1], c[0, 1]),
                        linewidth=1.3, color='#19c37d')
        lab = 'area=%.2f  comp=%d' % (rec['area_frac'], rec['n_components'])
        if rec['flags']:
            lab += '\n' + rec['flags'].replace(';', ', ')
        ax.set_title(lab, fontsize=6.5,
                     color='#c23b22' if rec['n_flags'] else '#333333')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('  wrote %s' % out_path)
    return True


def area_histogram(df, out_path):
    """Histogram of mask area."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ok = df[df.n_flags == 0]['area_frac']
    bad = df[df.n_flags > 0]['area_frac']
    bins = np.linspace(0, 1, 51)
    ax.hist([ok, bad], bins=bins, stacked=True,
            label=['no flags (n=%d)' % len(ok), 'flagged (n=%d)' % len(bad)],
            color=['#4a90d9', '#c23b22'])
    ax.axvline(AREA_MIN_FRAC, color='k', linestyle='--', linewidth=1)
    ax.axvline(AREA_MAX_FRAC, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Breast-mask area as a fraction of the 224x224 frame')
    ax.set_ylabel('Number of images')
    ax.set_title('Otsu breast-mask area distribution')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, axis='y', linestyle=':')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('  wrote %s' % out_path)


def full_res_check(filepaths, rng, n=40):
    """Dice between the 224x224 mask and a mask made at native resolution."""
    rows = []
    sample = list(rng.choice(filepaths, size=min(n, len(filepaths)), replace=False))
    for fp in sample:
        img_full = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img_full is None:
            continue
        mask_full = core_pipeline.segment_breast(img_full)
        mask_full_ds = cv2.resize(mask_full, (core_pipeline.IMG_SIZE, core_pipeline.IMG_SIZE),
                                  interpolation=cv2.INTER_NEAREST)
        img_small = cv2.resize(img_full, (core_pipeline.IMG_SIZE, core_pipeline.IMG_SIZE))
        mask_small = core_pipeline.segment_breast(img_small)

        a = mask_full_ds > 0
        b = mask_small > 0
        inter = float((a & b).sum())
        union = float((a | b).sum())
        rows.append({
            'filepath': fp,
            'native_h': img_full.shape[0],
            'native_w': img_full.shape[1],
            'area_frac_native_downscaled': float(a.mean()),
            'area_frac_masked_at_224': float(b.mean()),
            'iou': inter / union if union > 0 else np.nan,
            'dice': (2 * inter / (a.sum() + b.sum())) if (a.sum() + b.sum()) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def parse_args():
    ap = argparse.ArgumentParser(description='Otsu breast-mask audit (no training)')
    ap.add_argument('--n-random', type=int, default=48)
    ap.add_argument('--n-worst', type=int, default=48)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--full-res', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.join(core_pipeline.OUT_DIR, OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    print('=' * 78)
    print('BREAST-MASK AUDIT  (no model, no GPU, no training)')
    print('=' * 78)
    print('writing to: %s' % out_dir)

    meta_raw, patient_col = core_pipeline.load_data()
    cache = core_pipeline.IMAGE_CACHE
    print('\nimage cache entries: %d' % len(cache))

    filepaths = meta_raw['filepath'].unique().tolist()
    if args.limit:
        filepaths = filepaths[:args.limit]
    filepaths = [fp for fp in filepaths if fp in cache]
    print('auditing %d cached images' % len(filepaths))
    if not filepaths:
        print('\nERROR: no cohort filepaths are present in the image cache.')
        print('Run the main pipeline once so %s exists.' % core_pipeline.CACHE_FILE)
        return

    recs, pert = [], []
    for i, fp in enumerate(filepaths):
        img_gray, mask = cache[fp]
        recs.append(audit_one(fp, img_gray, mask))
        p = perturbation_areas(mask)
        p['filepath'] = fp
        pert.append(p)
        if (i + 1) % 500 == 0:
            print('  audited %d/%d' % (i + 1, len(filepaths)))

    df = pd.DataFrame(recs)
    meta_small = meta_raw[['filepath', 'malignant', 'density', 'density_binary', 'view']] \
        .drop_duplicates('filepath')
    df = df.merge(meta_small, on='filepath', how='left')
    df.to_csv(os.path.join(out_dir, 'mask_metrics.csv'), index=False)

    pdf = pd.DataFrame(pert)
    pdf.to_csv(os.path.join(out_dir, 'mask_perturbation_areas.csv'), index=False)

    flag_cols = [c for c in df.columns if c.startswith('flag_')]
    counts = pd.DataFrame({
        'flag': [c[5:] for c in flag_cols],
        'n_images': [int(df[c].sum()) for c in flag_cols],
        'pct': [100.0 * df[c].mean() for c in flag_cols],
    }).sort_values('n_images', ascending=False)
    counts.to_csv(os.path.join(out_dir, 'flag_counts.csv'), index=False)

    lines = []
    def emit(s=''):
        print(s)
        lines.append(s)

    emit('')
    emit('=' * 78)
    emit('MASK AUDIT SUMMARY')
    emit('=' * 78)
    emit('images audited                 : %d' % len(df))
    emit('images with ZERO flags         : %d (%.1f%%)'
         % (int((df.n_flags == 0).sum()), 100.0 * (df.n_flags == 0).mean()))
    emit('images with >=1 flag           : %d (%.1f%%)'
         % (int((df.n_flags > 0).sum()), 100.0 * (df.n_flags > 0).mean()))
    emit('')
    emit('mask area as fraction of frame : mean %.3f  SD %.3f  median %.3f'
         % (df.area_frac.mean(), df.area_frac.std(ddof=1), df.area_frac.median()))
    emit('                                 p1 %.3f  p5 %.3f  p95 %.3f  p99 %.3f'
         % tuple(df.area_frac.quantile([0.01, 0.05, 0.95, 0.99])))
    emit('inside-minus-outside intensity : mean %.1f  (higher = mask really is '
         'on the bright breast region)' % df.intensity_contrast.mean())
    emit('Otsu components found          : mean %.2f  max %d  '
         '(>1 means labels/markers/other tissue were present and discarded)'
         % (df.n_components.mean(), int(df.n_components.max())))
    emit('foreground discarded by the    : mean %.3f  p95 %.3f'
         % (df.discarded_frac.mean(), df.discarded_frac.quantile(0.95)))
    emit('  largest-component cleanup')
    emit('')
    emit('FLAG BREAKDOWN (flags are conservative and over-report by design):')
    for _, r in counts.iterrows():
        emit('  %-16s %6d  (%5.1f%%)' % (r['flag'], r['n_images'], r['pct']))
    emit('')
    emit('MASK-AREA STABILITY UNDER MORPHOLOGY (mean area fraction):')
    emit('  %-10s %8s' % ('radius', 'area'))
    emit('  %-10s %8.3f' % ('r=0', pdf['area_frac_r0'].mean()))
    for r in MORPH_RADII:
        emit('  %-10s %8.3f   (erode %d px)'
             % ('-%d px' % r, pdf['area_frac_erode%d' % r].mean(), r))
    for r in MORPH_RADII:
        emit('  %-10s %8.3f   (dilate %d px)'
             % ('+%d px' % r, pdf['area_frac_dilate%d' % r].mean(), r))
    emit('')
    emit('Interpretation: these area numbers bound how much SAS could move if')
    emit('the boundary were off by that many pixels. The SAS-under-perturbation')
    emit('test itself needs Grad-CAMs from trained models; it is run by')
    emit('calibration_experiment.py and calibration_analysis.py.')

    if args.full_res:
        emit('')
        emit('FULL-RESOLUTION CHECK (mask at native res, downscaled, vs mask at 224):')
        rng = np.random.RandomState(args.seed)
        fr = full_res_check(filepaths, rng)
        if len(fr):
            fr.to_csv(os.path.join(out_dir, 'full_res_agreement.csv'), index=False)
            emit('  n=%d images   Dice mean %.3f (SD %.3f)   IoU mean %.3f'
                 % (len(fr), fr.dice.mean(), fr.dice.std(ddof=1), fr.iou.mean()))
            emit('  A high Dice means the 224x224 working resolution is not what')
            emit('  determines the mask.')
        else:
            emit('  could not read any full-resolution files')

    with open(os.path.join(out_dir, 'mask_summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print('\nbuilding contact sheets...')
    rng = np.random.RandomState(args.seed)

    rand_idx = rng.choice(len(df), size=min(args.n_random, len(df)), replace=False)
    rand_recs = df.iloc[rand_idx].to_dict('records')
    for i in range(0, len(rand_recs), 24):
        contact_sheet(rand_recs[i:i + 24], cache,
                      'Random sample of breast masks (unbiased) -- sheet %d'
                      % (i // 24 + 1),
                      os.path.join(out_dir, 'contact_random_%02d.png' % (i // 24 + 1)))

    worst = df.sort_values(['n_flags', 'discarded_frac'], ascending=[False, False])
    worst_recs = worst.head(args.n_worst).to_dict('records')
    if worst_recs and worst_recs[0]['n_flags'] > 0:
        for i in range(0, len(worst_recs), 24):
            contact_sheet(worst_recs[i:i + 24], cache,
                          'Most-suspect masks in the whole cohort -- sheet %d'
                          % (i // 24 + 1),
                          os.path.join(out_dir, 'contact_worst_%02d.png' % (i // 24 + 1)))
    else:
        print('  no flagged images -- worst-case sheet skipped')

    by_area = df.sort_values('area_frac')
    ext = pd.concat([by_area.head(12), by_area.tail(12)]).to_dict('records')
    contact_sheet(ext, cache,
                  'Smallest 12 and largest 12 masks by area',
                  os.path.join(out_dir, 'contact_extremes_01.png'))

    area_histogram(df, os.path.join(out_dir, 'area_histogram.png'))

    manifest = {
        'script': os.path.basename(__file__),
        'purpose': 'Otsu breast-mask verification (no training)',
        'run_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_minutes': round((time.time() - t0) / 60.0, 2),
        'n_images_audited': int(len(df)),
        'thresholds': {
            'AREA_MIN_FRAC': AREA_MIN_FRAC, 'AREA_MAX_FRAC': AREA_MAX_FRAC,
            'DISCARD_FRAC': DISCARD_FRAC, 'TOP_BOTTOM_FRAC': TOP_BOTTOM_FRAC,
            'MORPH_RADII': MORPH_RADII,
        },
        'full_res_check': bool(args.full_res),
        'pipeline': {'IMG_SIZE': core_pipeline.IMG_SIZE, 'CACHE_FILE': core_pipeline.CACHE_FILE,
                  'DATA_ROOT': core_pipeline.DATA_ROOT},
        'env': {'python': sys.version.split()[0], 'platform': platform.platform(),
                'opencv': cv2.__version__, 'numpy': np.__version__},
    }
    with open(os.path.join(out_dir, 'run_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    print('\nDONE in %.1f min. Everything is in:\n  %s'
          % ((time.time() - t0) / 60.0, out_dir))
    print('\nStart with contact_worst_01.png: it shows the most suspect masks.')


if __name__ == '__main__':
    main()
