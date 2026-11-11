"""INbreast (and optional VinDr-Mammo / EMBED) metadata, DICOM reading and image cache,
plus the external-dataset version of the main sweep.
"""

import os, re, glob, argparse
import numpy as np
import pandas as pd
import cv2

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
except ImportError:
    raise ImportError(
        "This script needs pydicom for DICOM reading. Install with:\n"
        "  pip install pydicom --break-system-packages\n"
        "(or without --break-system-packages depending on your environment)"
    )

try:
    from core_pipeline_legacy import (
        OUT_DIR, IMG_SIZE, SEED_LIST, ABLATIONS, BIAS_STRENGTH,
        segment_breast, CBISDataset, make_loader, train_model,
        bias_split_by_correlation, group_masks, get_auroc, get_mean_sas,
        bootstrap_probe_accuracy, bootstrap_ci,
        power_analysis_table, plot_forest, BATCH_SIZE,
    )
    import core_pipeline_legacy as legacy_pipeline
except ImportError as e:
    raise ImportError(
        "Could not import from core_pipeline_legacy.py. Make sure external_validation.py is in the "
        f"SAME directory as core_pipeline_legacy.py. Original error: {e}"
    )

from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import wilcoxon


def read_dicom_as_gray(filepath, img_size=IMG_SIZE):
    """Read a DICOM, apply VOI LUT, fix MONOCHROME1 and resize to 224x224."""
    ds = pydicom.dcmread(filepath)
    arr = ds.pixel_array.astype(np.float32)
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass

    if getattr(ds, 'PhotometricInterpretation', '') == 'MONOCHROME1':
        arr = arr.max() - arr

    lo, hi = np.percentile(arr, [0.5, 99.5])
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-8) * 255.0
    arr = arr.astype(np.uint8)
    arr = cv2.resize(arr, (img_size, img_size))
    return arr


def _extract_leading_int(s):
    m = re.search(r'\d+', str(s))
    return int(m.group()) if m else None


def _extract_density_letter(s):
    matches = re.findall(r'\b[A-Da-d]\b', str(s))
    return matches[-1].upper() if matches else None


def build_metadata_vindr(data_root, require_images=True):
    """VinDr-Mammo metadata (malignancy from BI-RADS, density from ACR)."""
    csv_path = os.path.join(data_root, 'breast-level_annotations.csv')
    if not os.path.exists(csv_path):
        candidates = glob.glob(os.path.join(data_root, '**', 'breast-level_annotations.csv'), recursive=True)
        if not candidates:
            raise FileNotFoundError(
                f"Could not find breast-level_annotations.csv.\n"
                f"  Looked for: {os.path.abspath(csv_path)}\n"
                f"  Current working directory: {os.getcwd()}\n"
                f"  If you're calling this directly (not via prepare_external_subset.py, which "
                f"downloads this file automatically), make sure the CSV is actually at that path.")
        csv_path = candidates[0]
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"Loaded {len(df)} rows from {csv_path}")
    print("Raw breast_birads value counts:\n", df['breast_birads'].value_counts())
    print("Raw breast_density value counts:\n", df['breast_density'].value_counts())

    df['birads_num'] = df['breast_birads'].apply(_extract_leading_int)
    n_before = len(df)
    df = df[df['birads_num'].notna() & (df['birads_num'] != 0)].copy()
    print(f"Excluded {n_before - len(df)} rows with BI-RADS 0 (incomplete) or unparseable assessment")
    df['malignant'] = (df['birads_num'] >= 4).astype(int)

    density_letter = df['breast_density'].apply(_extract_density_letter)
    density_ordinal = density_letter.map({'A': 1, 'B': 2, 'C': 3, 'D': 4})
    df = df[density_ordinal.notna()].copy()
    df['density'] = density_ordinal.loc[df.index]
    df['density_binary'] = (df['density'] >= 3).astype(int)

    image_dirs = glob.glob(os.path.join(data_root, '**', 'images'), recursive=True)
    image_root = image_dirs[0] if image_dirs else os.path.join(data_root, 'images')

    def find_dicom(row):
        if require_images:
            candidates = glob.glob(os.path.join(image_root, str(row['study_id']), f"{row['image_id']}.*"))
            return candidates[0] if candidates else None
        else:
            return os.path.join(image_root, str(row['study_id']), f"{row['image_id']}.dicom")

    df['filepath'] = df.apply(find_dicom, axis=1)
    if require_images:
        n_missing = df['filepath'].isna().sum()
        if n_missing:
            print(f"WARNING: {n_missing}/{len(df)} rows had no matching DICOM file found under {image_root} "
                  f"-- check the folder structure matches study_id/image_id.ext")
        df = df.dropna(subset=['filepath']).reset_index(drop=True)

    df['view'] = df['view_position'].astype(str).str.upper().str.extract(r'(CC|MLO)')[0]
    df['patient_id'] = df['study_id']
    df['relative_url'] = 'images/' + df['study_id'].astype(str) + '/' + df['image_id'].astype(str) + '.dicom'

    print(f"Final usable rows: {len(df)}  |  malignant={df['malignant'].sum()}  benign={(1-df['malignant']).sum()}")
    return df[['filepath', 'malignant', 'density', 'density_binary', 'patient_id', 'view', 'relative_url']], 'patient_id'


EMBED_MALIGNANT_KEYWORDS = ['invasive', 'situ']
EMBED_BENIGN_KEYWORDS = ['benign', 'negative', 'non-breast', 'non breast']
EMBED_EXCLUDE_KEYWORDS = ['high-risk', 'high risk', 'borderline']
EMBED_ABBREV_MALIGNANT = {'IBC', 'ISC'}
EMBED_ABBREV_BENIGN = {'B', 'N', 'NBC'}
EMBED_ABBREV_EXCLUDE = {'HRL', 'BLL'}


def _classify_embed_severity(raw_value):
    exact = str(raw_value).strip().upper()
    if exact in EMBED_ABBREV_MALIGNANT:
        return 'malignant'
    if exact in EMBED_ABBREV_BENIGN:
        return 'benign'
    if exact in EMBED_ABBREV_EXCLUDE:
        return 'excluded'
    s = str(raw_value).lower()
    if any(k in s for k in EMBED_MALIGNANT_KEYWORDS):
        return 'malignant'
    if any(k in s for k in EMBED_BENIGN_KEYWORDS):
        return 'benign'
    if any(k in s for k in EMBED_EXCLUDE_KEYWORDS):
        return 'excluded'
    return 'unmatched'


def build_metadata_embed(data_root, metadata_csv, require_images=True):
    """EMBED metadata from its exported CSV."""
    df = pd.read_csv(metadata_csv, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    print(f"Loaded {len(df)} rows from {metadata_csv}")
    print(f"Columns available: {list(df.columns)[:30]}{'...' if len(df.columns) > 30 else ''}")

    if 'path_severity' not in df.columns or 'tissueden' not in df.columns:
        raise KeyError(
            "Expected columns 'path_severity' and 'tissueden' not found. Column names in your "
            "actual downloaded CSV may differ from what's documented -- check "
            "Emory-HITI/EMBED_Open_Data's README/tables/clinical_legend.csv and adjust the "
            "column names referenced in this function accordingly."
        )
    print("Raw path_severity value counts (before mapping -- verify keyword matching below against this):\n",
          df['path_severity'].value_counts(dropna=False))

    severity_class = df['path_severity'].apply(_classify_embed_severity)
    unmatched = df.loc[severity_class == 'unmatched', 'path_severity']
    if len(unmatched):
        print(f"WARNING: {len(unmatched)} rows had a path_severity value not recognized by either the "
              f"exact-abbreviation or substring matchers -- these are DROPPED. Unmatched value counts:\n"
              f"{unmatched.value_counts()}\n"
              f"If these are a real category, add them to EMBED_ABBREV_* or EMBED_*_KEYWORDS above.")
    df = df[severity_class.isin(['malignant', 'benign'])].copy()
    df['malignant'] = (severity_class.loc[df.index] == 'malignant').astype(int)

    density_num = pd.to_numeric(df['tissueden'], errors='coerce')
    df = df[density_num.notna()].copy()
    df['density'] = density_num.loc[df.index].clip(1, 4)
    df['density_binary'] = (df['density'] >= 3).astype(int)

    patient_col_candidates = [c for c in df.columns if re.search(r'empi|patient|mrn', c, re.IGNORECASE)]
    path_col_candidates = [c for c in df.columns if re.search(r'path|file', c, re.IGNORECASE)
                            and c not in ('path_severity',)]
    if not patient_col_candidates or not path_col_candidates:
        raise KeyError(
            f"Auto-detection found patient-id candidates {patient_col_candidates} and "
            f"file-path candidates {path_col_candidates} -- if either is empty or wrong, "
            "pass the correct column names explicitly by editing PATIENT_COL/PATH_COL below."
        )
    patient_col_name, path_col_name = patient_col_candidates[0], path_col_candidates[0]
    print(f"Using '{patient_col_name}' as patient identifier and '{path_col_name}' as image file path "
          f"-- VERIFY these are correct for your actual file.")

    df['patient_id'] = df[patient_col_name]
    df['s3_key'] = df[path_col_name].astype(str)
    df['filepath'] = df[path_col_name].apply(
        lambda p: p if os.path.isabs(str(p)) else os.path.join(data_root, str(p)))
    if require_images:
        df = df[df['filepath'].apply(os.path.exists)].reset_index(drop=True)
    df['view'] = np.nan

    print(f"Final usable rows: {len(df)}  |  malignant={df['malignant'].sum()}  benign={(1-df['malignant']).sum()}")
    return df[['filepath', 'malignant', 'density', 'density_binary', 'patient_id', 'view', 's3_key']], 'patient_id'


_INBREAST_FILENAME_RE = re.compile(
    r'^(?P<file_id>\d+)_(?P<patient_id>[0-9a-fA-F]+)_MG_(?P<laterality>[LR])_(?P<view>CC|MLO)_ANON',
)


def _scan_inbreast_dicoms(data_root):
    """List INbreast DICOMs and parse patient / file ids from the file names."""
    image_dirs = glob.glob(os.path.join(data_root, '**', 'AllDICOMs'), recursive=True)
    image_root = image_dirs[0] if image_dirs else os.path.join(data_root, 'AllDICOMs')
    all_files = glob.glob(os.path.join(image_root, '*'))
    rows = []
    n_unparsed = 0
    for fp in all_files:
        m = _INBREAST_FILENAME_RE.match(os.path.basename(fp))
        if not m:
            n_unparsed += 1
            continue
        rows.append({
            'file_id': m.group('file_id'),
            'patient_id': m.group('patient_id'),
            'laterality': m.group('laterality'),
            'view': m.group('view'),
            'filepath': fp,
        })
    if n_unparsed:
        print(f"Note: {n_unparsed} file(s) in AllDICOMs/ didn't match the expected "
              f"<file_id>_<patient_id>_MG_<L/R>_<CC/MLO>_ANON naming pattern -- skipped "
              f"(likely a stray non-image file; {len(rows)} DICOMs matched and parsed).")
    return pd.DataFrame(rows)


def build_metadata_inbreast(data_root, require_images=True):
    """INbreast metadata: BI-RADS 4-6 malignant, 1-2 benign, 3 excluded; ACR >= 3 dense."""
    csv_candidates = glob.glob(os.path.join(data_root, '**', 'INbreast.csv'), recursive=True)
    xls_candidates = glob.glob(os.path.join(data_root, '**', 'INbreast.xls'), recursive=True)
    if csv_candidates:
        meta = pd.read_csv(csv_candidates[0], sep=None, engine='python')
        print(f"Loaded {len(meta)} rows from {csv_candidates[0]}")
    elif xls_candidates:
        meta = pd.read_excel(xls_candidates[0])
        print(f"Loaded {len(meta)} rows from {xls_candidates[0]}")
    else:
        raise FileNotFoundError(f"Could not find INbreast.csv or INbreast.xls under {data_root}")
    meta.columns = [c.strip() for c in meta.columns]

    birads_col = next((c for c in meta.columns if 'birads' in c.lower() or 'bi-rads' in c.lower()), None)
    density_col = next((c for c in meta.columns if 'density' in c.lower() or c.lower() == 'acr'), None)
    pathology_col = next((c for c in meta.columns if 'patholog' in c.lower() or 'biopsy' in c.lower()), None)
    if not birads_col or not density_col:
        raise KeyError(
            f"Auto-detected columns: BI-RADS={birads_col}, density={density_col}. "
            f"Full column list: {list(meta.columns)}. If either is None, the actual column "
            f"names in your downloaded metadata file differ from what this function expects "
            f"-- edit the candidate lists above accordingly."
        )
    print(f"Using columns: BI-RADS={birads_col!r}, density={density_col!r}"
          + (f", pathology/biopsy={pathology_col!r} (will prefer this over the BI-RADS proxy where present)"
             if pathology_col else " (no explicit pathology/biopsy column found -- using BI-RADS proxy only)"))
    print(f"Raw {birads_col} value counts:\n{meta[birads_col].value_counts(dropna=False)}")
    print(f"Raw {density_col} value counts:\n{meta[density_col].value_counts(dropna=False)}")

    files_df = _scan_inbreast_dicoms(data_root) if require_images else pd.DataFrame()
    join_col = None
    if len(files_df):
        file_ids = set(files_df['file_id'].astype(str))
        best_overlap, best_col = 0, None
        for c in meta.columns:
            overlap = meta[c].astype(str).str.extract(r'(\d+)')[0].isin(file_ids).sum()
            if overlap > best_overlap:
                best_overlap, best_col = overlap, c
        join_col = best_col
        print(f"Matching metadata to files via column {join_col!r} "
              f"({best_overlap}/{len(meta)} rows matched a scanned DICOM file_id)")
        meta['_file_id'] = meta[join_col].astype(str).str.extract(r'(\d+)')[0]

    meta['birads_num'] = meta[birads_col].apply(_extract_leading_int)
    n_before = len(meta)
    meta = meta[meta['birads_num'].notna() & (meta['birads_num'] != 3)].copy()
    print(f"Excluded {n_before - len(meta)} rows with BI-RADS 3 or unparseable assessment")

    if pathology_col is not None and meta[pathology_col].notna().any():
        sev = meta[pathology_col].astype(str).str.lower()
        meta['malignant'] = sev.str.contains('malig|cancer|invasive', na=False).astype(int)
        has_result = sev.str.contains('malig|cancer|invasive|benign', na=False)
        print(f"Used explicit pathology/biopsy column for {has_result.sum()}/{len(meta)} rows; "
              f"remaining rows fall back to the BI-RADS proxy.")
        meta.loc[~has_result, 'malignant'] = (meta.loc[~has_result, 'birads_num'] >= 4).astype(int)
    else:
        meta['malignant'] = (meta['birads_num'] >= 4).astype(int)

    meta['density'] = pd.to_numeric(meta[density_col], errors='coerce').clip(1, 4)
    meta = meta[meta['density'].notna()].copy()
    meta['density_binary'] = (meta['density'] >= 3).astype(int)

    if require_images:
        if join_col is None:
            raise RuntimeError("Could not determine which metadata column matches the scanned "
                                "DICOM filenames -- inspect the printed column list and file_id "
                                "overlap counts above, then hardcode join_col if needed.")
        merged = meta.merge(files_df, left_on='_file_id', right_on='file_id', how='inner')
        n_lost = len(meta) - len(merged)
        if n_lost:
            print(f"WARNING: {n_lost}/{len(meta)} metadata rows had no matching DICOM file after the join.")
        df = merged
    else:
        df = meta.copy()
        df['filepath'] = df.get('_file_id', meta.index.astype(str))
        df['patient_id'] = df['filepath']
        df['view'] = np.nan

    print(f"Final usable rows: {len(df)}  |  malignant={df['malignant'].sum()}  benign={(1-df['malignant']).sum()}")
    return df[['filepath', 'malignant', 'density', 'density_binary', 'patient_id', 'view']], 'patient_id'


def _validate_cache_entries(cache_dict, source_label):
    """Check that every cached image and mask is a non-empty array."""
    bad = []
    for fp, (img, mask) in cache_dict.items():
        if img is None or getattr(img, "size", 0) == 0:
            bad.append(fp)
    if bad:
        preview = "\n  ".join(bad[:10])
        more = f"\n  ...and {len(bad) - 10} more" if len(bad) > 10 else ""
        raise RuntimeError(
            f"Found {len(bad)} unreadable/empty image(s) in the {source_label} "
            f"cache -- these would otherwise crash training with a cryptic "
            f"'cv2.resize ... !ssize.empty()' error. Affected file(s):\n  "
            f"{preview}{more}\n"
            f"Delete the stale cache file and rerun, or inspect these specific "
            f"files directly (they may be corrupted, non-standard DICOMs, or "
            f"the file extension/format didn't match what the reader expected)."
        )
    print(f"Validated {len(cache_dict)} cached entries in {source_label} -- all readable, none empty.")


def build_or_load_external_cache(filepaths, cache_file):
    """Build or load the image + mask cache for an external dataset."""
    import pickle
    needed = set(filepaths)

    if os.path.exists(cache_file):
        print(f"Loading cached images+masks from {cache_file} ...")
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        missing_from_disk_cache = needed - set(cached.keys())
        if missing_from_disk_cache:
            print(f"WARNING: existing cache file is STALE -- it's missing "
                  f"{len(missing_from_disk_cache)}/{len(needed)} of the filepaths "
                  f"this run actually needs (likely left over from an earlier run with "
                  f"a different file list or path convention). Rebuilding fresh instead "
                  f"of silently trusting an incomplete cache.")
        else:
            _validate_cache_entries(cached, f"existing cache file ({cache_file})")
            for k, v in cached.items():
                legacy_pipeline.IMAGE_CACHE[k] = v
            print(f"Loaded {len(cached)} cached entries -- covers all {len(needed)} needed filepaths.")
            return

    print(f"Precomputing {len(filepaths)} images+masks (one-time cost, cached to disk after)...")
    new_cache = {}
    failed = []
    for i, fp in enumerate(filepaths):
        ext = os.path.splitext(fp)[1].lower()
        try:
            if ext in ('.dcm', '.dicom', ''):
                img_gray = read_dicom_as_gray(fp)
            else:
                img_gray = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                if img_gray is None:
                    raise ValueError(f"cv2.imread returned None (unreadable/unsupported format) for: {fp}")
                img_gray = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
            if img_gray is None or img_gray.size == 0:
                raise ValueError(f"Reader produced an empty image for: {fp}")
            mask = segment_breast(img_gray)
        except Exception as e:
            failed.append((fp, str(e)))
            print(f"  [SKIPPED -- unreadable] {fp}: {e}")
            continue
        new_cache[fp] = (img_gray.astype(np.uint8), mask.astype(np.uint8))
        legacy_pipeline.IMAGE_CACHE[fp] = new_cache[fp]
        if (i + 1) % 200 == 0:
            print(f"  cached {i+1}/{len(filepaths)}")

    if failed:
        print(f"\nWARNING: {len(failed)}/{len(filepaths)} file(s) could not be read and were "
              f"excluded from the cache (and therefore from training/evaluation). "
              f"If this number is large relative to your dataset, investigate before "
              f"trusting downstream results -- see the [SKIPPED] lines above for exactly "
              f"which files and why.\n")

    _validate_cache_entries(new_cache, "newly built cache")
    with open(cache_file, 'wb') as f:
        pickle.dump(new_cache, f)
    print(f"Cache built and saved to {cache_file} ({len(new_cache)}/{len(filepaths)} files succeeded)")


def run_external_sweep(dataset_name, meta_raw, patient_col, seeds, ablations):
    """Same strategy x seed sweep as the main pipeline, on an external dataset."""
    cache_file = os.path.join(OUT_DIR, f'{dataset_name}_image_cache.pkl')
    build_or_load_external_cache(meta_raw['filepath'].unique().tolist(), cache_file)

    all_results = []
    for seed in seeds:
        print(f"\n{'#'*60}\n[{dataset_name}] SEED {seed}\n{'#'*60}")
        splitter1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        trainval_idx, test_idx = next(splitter1.split(meta_raw, groups=meta_raw[patient_col]))
        trainval_df = meta_raw.iloc[trainval_idx].reset_index(drop=True)
        test_df = meta_raw.iloc[test_idx].reset_index(drop=True)
        splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, val_idx = next(splitter2.split(trainval_df, groups=trainval_df[patient_col]))
        train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
        val_df = trainval_df.iloc[val_idx].reset_index(drop=True)
        train_df_used = bias_split_by_correlation(train_df, seed, BIAS_STRENGTH)

        aligned_mask, conflict_mask = group_masks(test_df)
        aligned_df = test_df[aligned_mask].reset_index(drop=True)
        conflict_df = test_df[conflict_mask].reset_index(drop=True)
        test_loader = make_loader(CBISDataset(test_df, augment=False), BATCH_SIZE, shuffle=False)
        aligned_loader = make_loader(CBISDataset(aligned_df, augment=False), BATCH_SIZE, shuffle=False) if len(aligned_df) else None
        conflict_loader = make_loader(CBISDataset(conflict_df, augment=False), BATCH_SIZE, shuffle=False) if len(conflict_df) else None

        needed = set(train_df_used['filepath']) | set(val_df['filepath']) | set(test_df['filepath'])
        missing = needed - set(legacy_pipeline.IMAGE_CACHE.keys())
        if missing:
            preview = "\n  ".join(list(missing)[:10])
            more = f"\n  ...and {len(missing) - 10} more" if len(missing) > 10 else ""
            raise RuntimeError(
                f"[{dataset_name}] seed {seed}: {len(missing)} filepath(s) needed for "
                f"training/validation/test are NOT present in IMAGE_CACHE, even though "
                f"caching completed without error. This means the cache was built with a "
                f"different string form of these paths than what ended up in the "
                f"dataframe (e.g. path separators, case, or normalization differing) -- "
                f"a real mismatch worth fixing at the source, not silently working around. "
                f"Missing example(s):\n  {preview}{more}\n"
                f"Compare these exact strings against the keys actually in IMAGE_CACHE "
                f"(e.g. print(list(legacy_pipeline.IMAGE_CACHE.keys())[:5])) to see how they differ."
            )
        print(f"[{dataset_name}] seed {seed}: pre-flight check passed -- all "
              f"{len(needed)} needed filepaths are present in IMAGE_CACHE.")

        for strategy in ablations:
            print(f"\n--- [{dataset_name}] {strategy}, seed {seed} ---")
            model = train_model(strategy, train_df_used, val_df, seed)

            auroc = get_auroc(model, test_loader)
            auroc_aligned = get_auroc(model, aligned_loader) if aligned_loader else np.nan
            auroc_conflict = get_auroc(model, conflict_loader) if conflict_loader else np.nan
            gap = auroc_aligned - auroc_conflict if not (np.isnan(auroc_aligned) or np.isnan(auroc_conflict)) else np.nan
            sas = get_mean_sas(model, test_loader)
            probe = bootstrap_probe_accuracy(model, test_loader, seed=seed)

            print(f"  [{strategy}] AUROC(all)={auroc:.4f}  gap={gap:.4f}  SAS={sas:.4f}  probe={probe['linear_mean']:.4f}")
            all_results.append({
                'seed': seed, 'strategy': strategy, 'auroc_all': auroc,
                'auroc_aligned': auroc_aligned, 'auroc_conflict': auroc_conflict,
                'worst_group_gap': gap, 'sas': sas,
                'probe_linear_mean': probe['linear_mean'], 'probe_linear_std': probe['linear_std'],
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUT_DIR, f'{dataset_name}_external_validation_results.csv'), index=False)

    print(f"\n{'='*70}\n[{dataset_name}] PAIRED COMPARISONS vs. baseline\n{'='*70}")
    metrics = ['worst_group_gap', 'sas', 'probe_linear_mean', 'auroc_all']
    better_map = {'worst_group_gap': 'lower', 'sas': 'lower',
                  'probe_linear_mean': 'lower', 'auroc_all': 'higher (or unchanged)'}
    stat_rows = []
    baseline_df = results_df[results_df.strategy == 'baseline'].set_index('seed')
    for strategy in ablations:
        if strategy == 'baseline':
            continue
        arm_df = results_df[results_df.strategy == strategy].set_index('seed')
        common = baseline_df.index.intersection(arm_df.index)
        if len(common) < 2:
            continue
        for metric in metrics:
            b = baseline_df.loc[common, metric].values
            a = arm_df.loc[common, metric].values
            valid = ~(np.isnan(a) | np.isnan(b))
            if valid.sum() < 2:
                continue
            diffs = a[valid] - b[valid]
            mean_diff, lo, hi = bootstrap_ci(diffs)
            try:
                _, p_val = wilcoxon(a[valid], b[valid])
            except ValueError:
                p_val = np.nan
            print(f"  {strategy:>10} | {metric:<18} | want {better_map[metric]:<20} | "
                  f"mean_diff={mean_diff:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  wilcoxon p={p_val:.4f}")
            stat_rows.append({'arm': strategy, 'metric': metric, 'mean_diff': mean_diff, 'ci_lo': lo, 'ci_hi': hi, 'wilcoxon_p': p_val})

    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(os.path.join(OUT_DIR, f'{dataset_name}_external_validation_stats.csv'), index=False)
    if len(stats_df):
        plot_forest(results_df, ablations, os.path.join(OUT_DIR, f'{dataset_name}_external_validation_forest.png'))
        power_df = power_analysis_table(results_df, ablations)
        power_df.to_csv(os.path.join(OUT_DIR, f'{dataset_name}_external_validation_power.csv'), index=False)

    summary = results_df.groupby('strategy').agg(
        auroc_mean=('auroc_all', 'mean'), auroc_std=('auroc_all', 'std'),
        gap_mean=('worst_group_gap', 'mean'), gap_std=('worst_group_gap', 'std'),
        sas_mean=('sas', 'mean'), sas_std=('sas', 'std'),
        probe_mean=('probe_linear_mean', 'mean'), probe_std=('probe_linear_mean', 'std'),
    ).reindex([a for a in ablations if a in results_df.strategy.unique()])
    summary.to_csv(os.path.join(OUT_DIR, f'{dataset_name}_external_validation_summary.csv'))
    print(f"\n[{dataset_name}] SUMMARY\n{summary.round(4).to_string()}")
    print(f"\nAll {dataset_name} outputs saved to: {OUT_DIR}")
    return results_df


def main():
    """Command-line entry point for an external dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['vindr', 'embed', 'inbreast'])
    parser.add_argument('--manifest_csv', default=None,
                         help="Output of prepare_external_subset.py (e.g. vindr_subset_manifest.csv). "
                              "RECOMMENDED path -- needs no raw image files on disk at all.")
    parser.add_argument('--cache_pkl', default=None,
                         help="Output of prepare_external_subset.py (e.g. vindr_subset_cache.pkl). "
                              "Required together with --manifest_csv.")
    parser.add_argument('--data_root', default=None,
                         help="FALLBACK mode only: a local folder already containing raw DICOM/image "
                              "files plus the original metadata CSV. Not needed if you used "
                              "prepare_external_subset.py.")
    parser.add_argument('--metadata_csv', default=None, help='Fallback mode, --dataset embed only')
    parser.add_argument('--seeds', type=int, default=None, help='Use first N seeds of SEED_LIST (default: all)')
    parser.add_argument('--ablations', default=None, help='Comma-separated (default: same as main sweep)')
    args = parser.parse_args()

    seeds = SEED_LIST[:args.seeds] if args.seeds else SEED_LIST
    ablations = args.ablations.split(',') if args.ablations else ABLATIONS

    if args.manifest_csv and args.cache_pkl:
        print(f"Loading pre-built manifest from {args.manifest_csv} and cache from {args.cache_pkl}")
        meta_raw = pd.read_csv(args.manifest_csv)
        import pickle
        with open(args.cache_pkl, 'rb') as f:
            cached = pickle.load(f)
        for k, v in cached.items():
            legacy_pipeline.IMAGE_CACHE[k] = v
        print(f"Loaded {len(cached)} cached image+mask entries; {len(meta_raw)} metadata rows.")
        patient_col = 'patient_id'
    elif args.data_root:
        if args.dataset == 'vindr':
            meta_raw, patient_col = build_metadata_vindr(args.data_root)
        elif args.dataset == 'inbreast':
            meta_raw, patient_col = build_metadata_inbreast(args.data_root)
        else:
            if not args.metadata_csv:
                raise ValueError("--metadata_csv is required for --dataset embed in --data_root mode")
            meta_raw, patient_col = build_metadata_embed(args.data_root, args.metadata_csv)
        cache_file = os.path.join(OUT_DIR, f'{args.dataset}_image_cache.pkl')
        build_or_load_external_cache(meta_raw['filepath'].unique().tolist(), cache_file)
    else:
        raise ValueError("Provide either (--manifest_csv AND --cache_pkl) from prepare_external_subset.py "
                          "[recommended], or --data_root for the fallback raw-files-on-disk mode.")

    print(f"\nRunning external validation: dataset={args.dataset}, seeds={seeds}, ablations={ablations}")
    run_external_sweep(args.dataset, meta_raw, patient_col, seeds, ablations)


if __name__ == "__main__":
    main()
