"""
generate_partial_results.py
============================
Reads dccam_cbis_results.csv and produces tabular summaries + graphs for
exactly the strategies you specify (default: baseline, grl_only, distcorr).

This is a general-purpose script -- it works whether your CSV has only
these 3 strategies or the full 5 (camreg/full rows are simply ignored if
present). It computes nothing that isn't directly in the data: no
invented numbers, no simulated results.

USAGE
-----
    python generate_partial_results.py
    python generate_partial_results.py --csv path/to/dccam_cbis_results.csv
    python generate_partial_results.py --strategies baseline grl_only distcorr camreg

OUTPUTS (written to ./results_output/ by default)
-----
    summary_table.csv           -- mean/std/n per strategy per metric
    paired_comparisons.csv      -- each strategy vs baseline: mean diff,
                                    bootstrap 95% CI, Wilcoxon p-value
    summary_bars.png            -- 2x2 grid: AUROC, Gap, SAS, Probe accuracy
    paired_forest_plot.png      -- forest-plot style view of paired diffs
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# ---------- palette ----------
NAVY = "#0B3D5C"
STEEL = "#4F8FA6"
CORAL = "#E4572E"
TEAL = "#2CA58D"
MUTE_GRAY = "#8C8C8C"
INK = "#1B2735"
MUTE = "#5B6670"
GRID = "#D3DDE2"

COLOR_CYCLE = [MUTE_GRAY, MUTE, STEEL, CORAL, TEAL]  # extends if you add more strategies

METRICS = {
    "auroc_all": "AUROC (overall)",
    "worst_group_gap": "Worst-group gap",
    "sas": "Spurious Attribution Score",
    "probe_linear_mean": "Density probe accuracy",
}


def bootstrap_ci(diffs, n_resamples=5000, ci=95, seed=0):
    """Percentile bootstrap CI on the mean of paired differences."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs)
    n = len(diffs)
    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lo, hi


def build_summary_table(df, strategies, outdir):
    subset = df[df["strategy"].isin(strategies)]
    available_metrics = [m for m in METRICS if m in subset.columns]

    rows = []
    for strat in strategies:
        strat_df = subset[subset["strategy"] == strat]
        if strat_df.empty:
            print(f"  [warning] no rows found for strategy '{strat}' -- skipping")
            continue
        row = {"strategy": strat, "n_seeds": len(strat_df)}
        for m in available_metrics:
            row[f"{m}_mean"] = strat_df[m].mean()
            row[f"{m}_std"] = strat_df[m].std()
        rows.append(row)

    summary = pd.DataFrame(rows)
    path = os.path.join(outdir, "summary_table.csv")
    summary.to_csv(path, index=False)
    print(f"Saved: {path}")
    print()
    print(summary.round(4).to_string(index=False))
    print()
    return summary, available_metrics


def build_paired_comparisons(df, strategies, baseline_name, outdir):
    """For every strategy != baseline, compute paired diff vs baseline,
    matched by seed. Requires both strategies to share the same seeds."""
    if baseline_name not in strategies:
        print(f"  [note] '{baseline_name}' not in requested strategies -- "
              f"skipping paired comparisons (nothing to compare against).")
        return None

    base_df = df[df["strategy"] == baseline_name].set_index("seed")
    available_metrics = [m for m in METRICS if m in df.columns]
    rows = []

    for strat in strategies:
        if strat == baseline_name:
            continue
        strat_df = df[df["strategy"] == strat].set_index("seed")
        common_seeds = base_df.index.intersection(strat_df.index)
        if len(common_seeds) == 0:
            print(f"  [warning] no overlapping seeds between '{strat}' and "
                  f"'{baseline_name}' -- skipping")
            continue

        for m in available_metrics:
            diffs = (strat_df.loc[common_seeds, m] - base_df.loc[common_seeds, m]).values
            mean_diff = diffs.mean()
            ci_lo, ci_hi = bootstrap_ci(diffs)
            try:
                _, p = wilcoxon(diffs)
            except ValueError:
                p = float("nan")  # happens if all diffs are exactly zero
            rows.append({
                "strategy": strat, "metric": m, "n": len(common_seeds),
                "mean_diff_vs_baseline": mean_diff,
                "ci_95_lo": ci_lo, "ci_95_hi": ci_hi,
                "wilcoxon_p": p,
            })

    result = pd.DataFrame(rows)
    path = os.path.join(outdir, "paired_comparisons.csv")
    result.to_csv(path, index=False)
    print(f"Saved: {path}")
    print()
    print(result.round(4).to_string(index=False))
    print()
    return result


def plot_summary_bars(summary, available_metrics, outdir):
    n_metrics = len(available_metrics)
    ncols = 2
    nrows = (n_metrics + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.5 * nrows), dpi=200)
    axes = np.atleast_1d(axes).flatten()
    fig.patch.set_facecolor("white")

    x = np.arange(len(summary))
    colors = COLOR_CYCLE[: len(summary)]

    for ax, metric in zip(axes, available_metrics):
        means = summary[f"{metric}_mean"].values
        stds = summary[f"{metric}_std"].values
        ax.bar(x, means, yerr=stds, capsize=5, color=colors, width=0.55,
               edgecolor="none", error_kw={"ecolor": INK, "elinewidth": 1.2})
        ax.set_xticks(x)
        ax.set_xticklabels(summary["strategy"].str.replace("_", " ").str.title(), fontsize=11)
        ax.set_title(METRICS[metric], fontsize=13, fontweight="bold", color=NAVY)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for xi, m, s in zip(x, means, stds):
            ax.text(xi, m + s + max(means) * 0.04, f"{m:.3f}", ha="center",
                    fontsize=10, fontweight="bold", color=INK)
        ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.22)

    # hide any unused subplot slots
    for ax in axes[n_metrics:]:
        ax.axis("off")

    n_seeds = int(summary["n_seeds"].max())
    fig.suptitle(f"Results Summary -- {', '.join(summary['strategy'].str.replace('_',' ').str.title())} "
                 f"(n = {n_seeds} seeds each)",
                 fontsize=14, fontweight="bold", color=NAVY, y=1.01)
    plt.tight_layout()
    path = os.path.join(outdir, "summary_bars.png")
    plt.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_paired_forest(paired, outdir):
    if paired is None or paired.empty:
        print("  [note] no paired comparison data -- skipping forest plot")
        return

    metrics_present = paired["metric"].unique().tolist()
    strategies_present = paired["strategy"].unique().tolist()

    fig, axes = plt.subplots(1, len(metrics_present), figsize=(4.2 * len(metrics_present), 3.5), dpi=200)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    for ax, metric in zip(axes, metrics_present):
        sub = paired[paired["metric"] == metric]
        y_pos = np.arange(len(sub))
        significant = (sub["ci_95_lo"] > 0) | (sub["ci_95_hi"] < 0)
        colors = [CORAL if s else MUTE_GRAY for s in significant]

        for yi, (_, row) in zip(y_pos, sub.iterrows()):
            ax.plot([row["ci_95_lo"], row["ci_95_hi"]], [yi, yi],
                    color=colors[yi], linewidth=2)
            ax.scatter([row["mean_diff_vs_baseline"]], [yi], color=colors[yi], s=60, zorder=3)

        ax.axvline(0, color=INK, linestyle="--", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["strategy"].str.replace("_", " ").str.title(), fontsize=10)
        ax.set_title(METRICS.get(metric, metric), fontsize=11, fontweight="bold", color=NAVY)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color=GRID, linewidth=0.6)

    fig.suptitle("Paired Differences vs. Baseline (95% bootstrap CI)",
                 fontsize=13, fontweight="bold", color=NAVY, y=1.05)
    plt.tight_layout()
    path = os.path.join(outdir, "paired_forest_plot.png")
    plt.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="dccam_cbis_results.csv",
                         help="Path to the results CSV (default: dccam_cbis_results.csv)")
    parser.add_argument("--strategies", nargs="+", default=["baseline", "grl_only", "distcorr"],
                         help="Strategies to include (default: baseline grl_only distcorr)")
    parser.add_argument("--baseline", default="baseline",
                         help="Which strategy name is the baseline for paired comparisons")
    parser.add_argument("--outdir", default="results_output",
                         help="Output directory (default: results_output)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: file not found: {args.csv}")
        print("Run this script from the folder containing your results CSV, "
              "or pass --csv with the correct path.")
        return

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    missing = set(args.strategies) - set(df["strategy"].unique())
    if missing:
        print(f"[warning] these requested strategies are not in the CSV: {missing}")
    present_strategies = [s for s in args.strategies if s in df["strategy"].unique()]

    if not present_strategies:
        print("None of the requested strategies were found in the CSV. Nothing to do.")
        return

    print(f"Building results for: {present_strategies}\n")

    summary, available_metrics = build_summary_table(df, present_strategies, args.outdir)
    paired = build_paired_comparisons(df, present_strategies, args.baseline, args.outdir)

    plot_summary_bars(summary, available_metrics, args.outdir)
    plot_paired_forest(paired, args.outdir)

    print(f"\nAll outputs saved to: {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()