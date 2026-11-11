"""All arms + MI diagnostic + published methods + INbreast sweep, in one run.

Outputs: outputs/method_comparison/ (combined_10arm_results.csv is used for Table 7).
Run: python experiments/method_comparison.py [--seeds 3]
"""
import os
import sys

# Make src/ and analysis/ importable when the script is run from any folder.
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', d) for d in ('src', 'analysis')]

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import json
import platform
from datetime import datetime, timezone

import torch

import core_pipeline

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


import published_baselines
import external_validation
from paths import INBREAST_DIR

OUTPUT_SUBDIR = 'method_comparison'

INBREAST_DATA_ROOT = INBREAST_DIR


def _determinism_snapshot():
    """Settings and versions saved in run_manifest.json."""
    return {
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'use_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }


def main():
    n_seeds_arg = None
    if len(sys.argv) > 2 and sys.argv[1] == '--seeds':
        n_seeds_arg = int(sys.argv[2])

    out_dir = os.path.join(core_pipeline.OUT_DIR, OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    core_pipeline.OUT_DIR = out_dir
    published_baselines.OUT_DIR = out_dir
    external_validation.OUT_DIR = out_dir

    if n_seeds_arg:
        core_pipeline.SEED_LIST = core_pipeline.SEED_LIST[:n_seeds_arg]
        print(f"Smoke-test mode: using only the first {n_seeds_arg} seed(s): {core_pipeline.SEED_LIST}")

    print(f"Method comparison -- arms (full ablation): {core_pipeline.ABLATIONS}")
    print(f"Method comparison -- seeds: {core_pipeline.SEED_LIST}")
    print(f"Method comparison -- outputs will be written to: {out_dir}")
    print(f"Method comparison -- determinism settings: {_determinism_snapshot()}\n")

    print("=" * 70)
    print("STEP 1 / 4 -- full 6-arm CBIS-DDSM ablation (core_pipeline.main())")
    print("=" * 70)
    core_pipeline.main()

    print("\n" + "=" * 70)
    print("STEP 2 / 4 -- mi_min CLUB-vs-probe diagnostic (core_pipeline.run_mi_diagnostic())")
    print("=" * 70)
    meta_raw, patient_col = core_pipeline.load_data()
    core_pipeline.run_mi_diagnostic(meta_raw, patient_col)

    print("\n" + "=" * 70)
    print("STEP 3 / 4 -- external-method benchmark (benchmark.main())")
    print("=" * 70)
    published_baselines.main()

    print("\n" + "=" * 70)
    print("STEP 4 / 4 -- INbreast external validation")
    print("=" * 70)
    inbreast_ran = False
    if not INBREAST_DATA_ROOT or not os.path.isdir(INBREAST_DATA_ROOT):
        print("Skipped: set INBREAST_DATA_ROOT near the top of this file to your local "
              "INbreast folder first (the one containing AllDICOMs/ and INbreast.csv/.xls), "
              "then re-run. See external_validation.py's own module docstring for details.")
    else:
        meta_raw_inb, patient_col_inb = external_validation.build_metadata_inbreast(INBREAST_DATA_ROOT)
        external_validation.run_external_sweep(
            'inbreast', meta_raw_inb, patient_col_inb, core_pipeline.SEED_LIST, core_pipeline.ABLATIONS,
        )
        inbreast_ran = True

    manifest = {
        'script': 'method_comparison.py',
        'run_utc': datetime.now(timezone.utc).isoformat(),
        'ablations': core_pipeline.ABLATIONS,
        'seed_list': core_pipeline.SEED_LIST,
        'out_dir': out_dir,
        'inbreast_step_ran': inbreast_ran,
        'determinism': _determinism_snapshot(),
    }
    manifest_path = os.path.join(out_dir, 'run_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nMethod comparison complete. All figures/tables are in: {out_dir}")
    print(f"Run manifest (seeds/versions/determinism settings) saved to: {manifest_path}")


if __name__ == "__main__":
    main()
