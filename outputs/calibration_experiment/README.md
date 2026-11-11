# calibration_experiment

13-seed run that also saves probabilities and checkpoints: calibration, decision threshold, SAS under perturbed masks and the balanced density probe.

Created by: `python experiments/calibration_experiment.py, then analysis/calibration_analysis.py and analysis/density_probe_metrics.py`

| File | Contents |
|---|---|
| `results.csv` | one row per run, with group sizes and SAS under dilated/eroded masks |
| `summary_by_tag.csv` |  |
| `paired_vs_baseline.csv` |  |
| `calibration_per_run.csv` |  |
| `calibration_summary.csv` | Table 8 |
| `calibration_paired_vs_baseline.csv` |  |
| `sas_mask_sensitivity.csv` | SAS under dilated / eroded masks |
| `sas_mask_sensitivity_summary.csv` |  |
| `probe_balanced_per_run.csv` |  |
| `probe_balanced_summary.csv` | balanced accuracy and AUROC of the density probe |
| `probe_balanced_paired.csv` |  |
| `reproduce_vs_main.csv` |  |
| `reproduce_summary.csv` |  |
| `reliability_diagram_main.png` |  |
| `threshold_shift_main.png` |  |
| `epoch_history.csv` |  |
| `probs/` | saved test / validation probabilities |
| `logs/` | training logs |
| `ckpt/` | model checkpoints, not uploaded (see .gitignore) |
| `run_manifest_calibration_experiment.json` |  |
