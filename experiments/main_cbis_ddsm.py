"""Main CBIS-DDSM experiment (5 strategies x 13 seeds) with deterministic GPU settings.

Outputs: outputs/main_experiment/ (main results, forest plot, Grad-CAM figure).
Run: python experiments/main_cbis_ddsm.py [--seeds 3] [--skip-sensitivity]
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

MAIN_STRATEGIES = ['baseline', 'grl_only', 'distcorr', 'camreg', 'full']
OUTPUT_SUBDIR = 'main_experiment'

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


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
    skip_sensitivity = '--skip-sensitivity' in sys.argv
    if len(sys.argv) > 2 and sys.argv[1] == '--seeds':
        n_seeds_arg = int(sys.argv[2])

    out_dir = os.path.join(core_pipeline.OUT_DIR, OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    core_pipeline.ABLATIONS = MAIN_STRATEGIES
    core_pipeline.OUT_DIR = out_dir

    if skip_sensitivity:
        core_pipeline.RUN_SENSITIVITY_SWEEP = False
        print("--skip-sensitivity: sensitivity sweep will NOT be rerun; "
              "reusing whatever distcorr_sensitivity_sweep.csv/.png already exists in OUT_DIR.")

    if n_seeds_arg:
        core_pipeline.SEED_LIST = core_pipeline.SEED_LIST[:n_seeds_arg]
        print(f"Smoke-test mode: using only the first {n_seeds_arg} seed(s): {core_pipeline.SEED_LIST}")

    print(f"Main experiment -- arms: {core_pipeline.ABLATIONS}")
    print(f"Main experiment -- seeds: {core_pipeline.SEED_LIST}")
    print(f"Main experiment -- outputs will be written to: {out_dir}")
    print(f"Main experiment -- determinism settings: {_determinism_snapshot()}\n")

    core_pipeline.main()

    manifest = {
        'script': 'main_cbis_ddsm.py',
        'run_utc': datetime.now(timezone.utc).isoformat(),
        'ablations': core_pipeline.ABLATIONS,
        'seed_list': core_pipeline.SEED_LIST,
        'out_dir': out_dir,
        'determinism': _determinism_snapshot(),
    }
    manifest_path = os.path.join(out_dir, 'run_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nMain experiment complete. All figures/tables are in: {out_dir}")
    print(f"Run manifest (seeds/versions/determinism settings) saved to: {manifest_path}")


if __name__ == "__main__":
    main()
