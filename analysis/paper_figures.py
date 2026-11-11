"""Draws the forest plot, the sweep figure and the GRL figure of the paper from the result CSVs.

Outputs: outputs/paper_figures/forest_plot_main.png, shortcut_and_weight_sweeps.png, grl_configurations.png
Run: python analysis/paper_figures.py
"""
import os
import sys

# Make src/ and analysis/ importable when the script is run from any folder.
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', d) for d in ('src', 'analysis')]

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from paths import OUTPUT_DIR

HERE = OUTPUT_DIR
OUT_DIR = os.path.join(HERE, "paper_figures")
# Input CSVs written by the experiment scripts
PATHS = {
    "shortcut_summary": os.path.join(HERE, "shortcut_strength_sweep", "condition_summary.csv"),
    "dc_paired":    os.path.join(HERE, "distcorr_weight_sweep", "paired_vs_baseline.csv"),
    "cam_paired":   os.path.join(HERE, "camreg_weight_sweep", "paired_vs_baseline.csv"),
    "grid_paired":  os.path.join(HERE, "full_weight_grid", "paired_vs_baseline.csv"),
    "grl_paired":   os.path.join(HERE, "grl_settings", "paired_vs_baseline.csv"),
    "grl_results":  os.path.join(HERE, "grl_settings", "results.csv"),
    "dccam": next((p for p in [os.path.join(HERE, "main_experiment", "dccam_cbis_results.csv"),
                               os.path.join(HERE, "method_comparison", "dccam_cbis_results.csv"),
                               os.path.join(HERE, "dccam_cbis_results.csv")] if os.path.exists(p)),
                  os.path.join(HERE, "dccam_cbis_results.csv")),
}
# Figures are drawn at the paper text width (468 pt), so no text is smaller than 7 pt
PAGE_W = 468 / 72.0
_missing = [p for p in PATHS.values() if not os.path.exists(p)]
if _missing:
    raise SystemExit("Cannot find these input files (edit PATHS at the top of the script):\n  " + "\n  ".join(_missing))
os.makedirs(OUT_DIR, exist_ok=True)

# Colours, markers and labels for each strategy
GRAY = "#7a7974"
C = {"baseline": GRAY, "distcorr": "#2a78d6", "camreg": "#eb6834", "full": "#1baf7a"}
MK = {"baseline": "o", "distcorr": "s", "camreg": "^", "full": "D"}
LAB = {"baseline": "Baseline", "distcorr": "DistCorr", "camreg": "CamReg", "full": "Full"}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.5, "lines.linewidth": 1.6,
    "savefig.dpi": 600, "savefig.bbox": None, "lines.markersize": 4,
})


def zero_line(ax):
    """Horizontal reference line at zero."""
    ax.axhline(0, color="#52514e", lw=0.7, zorder=1)


def fig_shortcut_strength():
    """Optional larger figure (not in the paper): gap and SAS against the training correlation."""
    s = pd.read_csv(PATHS["shortcut_summary"])
    family = ["natural", "weak_r03", "moderate_r045", "published_r06"]
    s = s[s.condition.isin(family)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, (m, title) in zip(axes, [("worst_group_gap", "(a) Worst-group gap"),
                                     ("sas", "(b) Spurious attribution score (SAS)")]):
        for k in ["baseline", "distcorr", "camreg", "full"]:
            d = s[s.strategy == k].set_index("condition").loc[family]
            ax.errorbar(d.achieved_r_mean, d[m + "_mean"], yerr=d[m + "_sd"], color=C[k],
                        marker=MK[k], ms=5, lw=1.4, capsize=2, elinewidth=0.8,
                        markeredgecolor="white", markeredgewidth=0.6, label=LAB[k])
        ax.set_xlabel("Achieved density-malignancy correlation in training set ($r$)")
        ax.set_title(title, loc="left")
        ax.set_xticks([0.0, 0.2, 0.4, 0.6])
        if m == "worst_group_gap":
            zero_line(ax)
            ax.set_ylabel("AUROC$_a$ $-$ AUROC$_c$")
        else:
            ax.set_ylabel("Share of attribution outside breast")
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(h, l, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.savefig(os.path.join(OUT_DIR, "shortcut_strength_sweep.png"))
    plt.close(fig)


def _sel(p, tag, metric):
    """Mean difference and 95% CI for one tag and metric from a paired_vs_baseline table."""
    r = p[(p.tag == tag) & (p.metric == metric)].iloc[0]
    return r.mean_diff_vs_baseline, r.ci_lo, r.ci_hi


def _plot_ci(ax, x, vals, color, marker, label, offset=0.0):
    """Means with CI error bars, slightly offset on the x axis so series do not overlap."""
    m = np.array([v[0] for v in vals]); lo = np.array([v[1] for v in vals]); hi = np.array([v[2] for v in vals])
    xx = np.array(x) * (1 + offset)
    ax.errorbar(xx, m, yerr=[m - lo, hi - m], color=color, marker=marker, ms=4.5, lw=1.4,
                capsize=2, elinewidth=0.8, markeredgecolor="white", markeredgewidth=0.6, label=label)


def fig_weight_sweeps():
    """Optional larger figure (not in the paper): distcorr weight, CAM weight and full-model grid."""
    dc = pd.read_csv(PATHS["dc_paired"]); cam = pd.read_csv(PATHS["cam_paired"]); gr = pd.read_csv(PATHS["grid_paired"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    ax = axes[0]; w = [1, 2, 4, 8, 12, 16]
    _plot_ci(ax, w, [_sel(dc, f"distcorr_w{x}", "worst_group_gap") for x in w], C["distcorr"], "s", "$\\Delta$ gap", -0.04)
    _plot_ci(ax, w, [_sel(dc, f"distcorr_w{x}", "auroc_all") for x in w], "#52514e", "o", "$\\Delta$ AUROC", 0.04)
    _plot_ci(ax, w, [_sel(dc, f"distcorr_w{x}", "sas") for x in w], C["camreg"], "^", "$\\Delta$ SAS", 0.0)
    ax.set_xscale("log", base=2); ax.set_xticks(w); ax.set_xticklabels([str(x) for x in w])
    ax.axvline(8, color="#b5b3ad", lw=0.8, ls="--"); zero_line(ax)
    ax.set_xlabel("DistCorr weight (paper: 8)"); ax.set_ylabel("Paired difference vs baseline")
    ax.set_title("(a) DistCorr only", loc="left"); ax.set_ylim(-0.6, 0.2)
    ax.legend(frameon=False, loc="lower left")
    ax = axes[1]; w = [0.25, 0.5, 1, 2, 4, 8]; tags = ["0.25", "0.5", "1", "2", "4", "8"]
    _plot_ci(ax, w, [_sel(cam, f"camreg_cam{t}", "worst_group_gap") for t in tags], C["distcorr"], "s", "$\\Delta$ gap", -0.04)
    _plot_ci(ax, w, [_sel(cam, f"camreg_cam{t}", "auroc_all") for t in tags], "#52514e", "o", "$\\Delta$ AUROC", 0.04)
    _plot_ci(ax, w, [_sel(cam, f"camreg_cam{t}", "sas") for t in tags], C["camreg"], "^", "$\\Delta$ SAS", 0.0)
    ax.set_xscale("log", base=2); ax.set_xticks(w); ax.set_xticklabels(tags)
    ax.axvline(2, color="#b5b3ad", lw=0.8, ls="--"); zero_line(ax)
    ax.set_xlabel("CAM weight (paper: 2)"); ax.set_title("(b) CamReg only", loc="left"); ax.set_ylim(-0.6, 0.2)
    ax = axes[2]; dcw = [2, 8, 16]; cw = ["0.5", "2", "8"]
    A = np.array([[_sel(gr, f"full_dc{d}_cam{c}", "auroc_all")[0] for c in cw] for d in dcw])
    G = np.array([[_sel(gr, f"full_dc{d}_cam{c}", "worst_group_gap")[0] for c in cw] for d in dcw])
    ax.imshow(-A, cmap="Blues", vmin=0, vmax=0.12, origin="lower", aspect="auto")
    for i in range(3):
        for j in range(3):
            txt_col = "white" if -A[i, j] > 0.07 else "#0b0b0b"
            ax.text(j, i, f"{A[i, j]:+.3f}\n({G[i, j]:+.2f})".replace("-", "\u2212"), ha="center", va="center", fontsize=7, color=txt_col)
    ax.set_xticks(range(3)); ax.set_xticklabels(cw); ax.set_yticks(range(3)); ax.set_yticklabels([str(x) for x in dcw])
    ax.set_xlabel("CAM weight"); ax.set_ylabel("DistCorr weight"); ax.grid(False)
    ax.set_title("(c) Full: $\\Delta$AUROC ($\\Delta$gap)", loc="left")
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "loss_weight_sweeps.png"))
    plt.close(fig)


# GRL settings in the order shown in the GRL figure
GRL_ORDER = ["grl_main", "grl_l1_reinit", "grl_noreinit", "grl_l1", "grl_l03", "grl_l01", "grl_mlp_l1", "grl_mlp_l03"]
GRL_LABEL = {
    "grl_main": "$\\lambda$=5, w=2, re-init/4 ep (default)",
    "grl_l1_reinit": "$\\lambda$=1, w=1, re-init/4 ep",
    "grl_noreinit": "$\\lambda$=5, w=2, no re-init (selected)",
    "grl_l1": "$\\lambda$=1, w=1, no re-init",
    "grl_l03": "$\\lambda$=0.3, w=1, no re-init",
    "grl_l01": "$\\lambda$=0.1, w=1, no re-init",
    "grl_mlp_l1": "$\\lambda$=1, w=1, MLP adversary",
    "grl_mlp_l03": "$\\lambda$=0.3, w=1, MLP adversary",
}


def fig_grl():
    """GRL figure: paired effect of each GRL setting, and aligned vs conflicting AUROC per seed."""
    p = pd.read_csv(PATHS["grl_paired"]); r = pd.read_csv(PATHS["grl_results"])
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.75), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]; y = np.arange(len(GRL_ORDER))[::-1]
    for k, (metric, col, mk, lab, dy) in enumerate([("worst_group_gap", C["distcorr"], "s", "$\\Delta$ gap", 0.15),
                                                    ("auroc_all", "#52514e", "o", "$\\Delta$ AUROC", -0.15)]):
        v = [_sel(p, t, metric) for t in GRL_ORDER]
        m = np.array([a[0] for a in v]); lo = np.array([a[1] for a in v]); hi = np.array([a[2] for a in v])
        ax.errorbar(m, y + dy, xerr=[m - lo, hi - m], fmt=mk, color=col, ms=4.5, capsize=2, elinewidth=0.8,
                    markeredgecolor="white", markeredgewidth=0.6, label=lab)
    ax.axvline(0, color="#52514e", lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels([GRL_LABEL[t] for t in GRL_ORDER], fontsize=7)
    ax.set_xlabel("Paired difference vs baseline (5 seeds, 95% CI)")
    ax.set_title("(a) GRL settings", loc="left"); ax.legend(frameon=False, loc="lower right"); ax.grid(axis="y", visible=False)
    ax = axes[1]
    styles = {"baseline_ref": (GRAY, "o", "Baseline"), "grl_main": (C["camreg"], "^", "GRL, default"),
              "grl_noreinit": (C["distcorr"], "s", "GRL, no re-init")}
    for tag, (col, mk, lab) in styles.items():
        d = r[r.tag == tag]
        ax.scatter(d.auroc_conflict, d.auroc_aligned, color=col, marker=mk, s=28, edgecolor="white", linewidth=0.6,
                   label=lab, zorder=3)
    ax.plot([0.1, 1.0], [0.1, 1.0], color="#b5b3ad", lw=0.8, ls="--", zorder=1)
    ax.set_xlim(0.1, 1.0); ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("AUROC$_c$ (conflicting group)"); ax.set_ylabel("AUROC$_a$ (aligned group)")
    ax.set_title("(b) Per seed (above dashed line = positive gap)", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=7,
              handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "grl_configurations.png"))
    plt.close(fig)


def fig_sweeps_combined():
    """Sweep figure: shortcut strength, distcorr weight, CAM weight and full-model weight grid."""
    s = pd.read_csv(PATHS["shortcut_summary"])
    family = ["natural", "weak_r03", "moderate_r045", "published_r06"]
    s = s[s.condition.isin(family)]
    dc = pd.read_csv(PATHS["dc_paired"]); cam = pd.read_csv(PATHS["cam_paired"]); gr = pd.read_csv(PATHS["grid_paired"])
    fig, axes = plt.subplots(1, 4, figsize=(PAGE_W, 2.35), gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    ax = axes[0]
    for k in ["baseline", "distcorr", "camreg", "full"]:
        d = s[s.strategy == k].set_index("condition").loc[family]
        ax.errorbar(d.achieved_r_mean, d["worst_group_gap_mean"], yerr=d["worst_group_gap_sd"], color=C[k],
                    marker=MK[k], ms=4.5, lw=1.4, capsize=2, elinewidth=0.8,
                    markeredgecolor="white", markeredgewidth=0.6, label=LAB[k])
    zero_line(ax); ax.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax.set_xlabel("Training correlation $r$"); ax.set_ylabel("Worst-group gap")
    ax.set_title("(a) Shortcut strength", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=7, handlelength=1.2, borderaxespad=0.1)
    ax = axes[1]; w = [1, 2, 4, 8, 12, 16]
    _plot_ci(ax, w, [_sel(dc, f"distcorr_w{x}", "worst_group_gap") for x in w], C["distcorr"], "s", "$\\Delta$ gap", -0.04)
    _plot_ci(ax, w, [_sel(dc, f"distcorr_w{x}", "auroc_all") for x in w], "#52514e", "o", "$\\Delta$ AUROC", 0.04)
    _plot_ci(ax, w, [_sel(dc, f"distcorr_w{x}", "sas") for x in w], C["camreg"], "^", "$\\Delta$ SAS", 0.0)
    ax.set_xscale("log", base=2); ax.set_xticks(w); ax.set_xticklabels([str(x) for x in w])
    ax.axvline(8, color="#b5b3ad", lw=0.8, ls="--"); zero_line(ax)
    ax.set_xlabel("DistCorr weight"); ax.set_ylabel("Paired difference")
    ax.set_title("(b) DistCorr weight", loc="left"); ax.set_ylim(-0.6, 0.2)
    ax.legend(frameon=False, loc="lower left", fontsize=7, handlelength=1.2, borderaxespad=0.1)
    ax = axes[2]; w = [0.25, 0.5, 1, 2, 4, 8]; tags = ["0.25", "0.5", "1", "2", "4", "8"]
    _plot_ci(ax, w, [_sel(cam, f"camreg_cam{t}", "worst_group_gap") for t in tags], C["distcorr"], "s", "$\\Delta$ gap", -0.04)
    _plot_ci(ax, w, [_sel(cam, f"camreg_cam{t}", "auroc_all") for t in tags], "#52514e", "o", "$\\Delta$ AUROC", 0.04)
    _plot_ci(ax, w, [_sel(cam, f"camreg_cam{t}", "sas") for t in tags], C["camreg"], "^", "$\\Delta$ SAS", 0.0)
    ax.set_xscale("log", base=2); ax.set_xticks(w); ax.set_xticklabels(tags)
    ax.axvline(2, color="#b5b3ad", lw=0.8, ls="--"); zero_line(ax)
    ax.set_xlabel("CAM weight"); ax.set_title("(c) CAM weight", loc="left"); ax.set_ylim(-0.6, 0.2)
    ax = axes[3]; dcw = [2, 8, 16]; cw = ["0.5", "2", "8"]
    A = np.array([[_sel(gr, f"full_dc{d}_cam{c}", "auroc_all")[0] for c in cw] for d in dcw])
    G = np.array([[_sel(gr, f"full_dc{d}_cam{c}", "worst_group_gap")[0] for c in cw] for d in dcw])
    ax.imshow(-A, cmap="Blues", vmin=0, vmax=0.12, origin="lower", aspect="auto")
    for i in range(3):
        for j in range(3):
            txt_col = "white" if -A[i, j] > 0.07 else "#0b0b0b"
            ax.text(j, i, f"{A[i, j]:+.3f}\n({G[i, j]:+.2f})".replace("-", "\u2212"), ha="center", va="center", fontsize=7, color=txt_col)
    ax.set_xticks(range(3)); ax.set_xticklabels(cw); ax.set_yticks(range(3)); ax.set_yticklabels([str(x) for x in dcw])
    ax.set_xlabel("CAM weight"); ax.set_ylabel("DistCorr weight"); ax.grid(False)
    ax.set_title("(d) Full model", loc="left")
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(w_pad=0.6, pad=0.3)
    fig.savefig(os.path.join(OUT_DIR, "shortcut_and_weight_sweeps.png"))
    plt.close(fig)


def _bootstrap_ci(diffs, n_boot=5000, seed=0):
    """Same percentile bootstrap as core_pipeline.bootstrap_ci."""
    rng = np.random.RandomState(seed)
    diffs = np.asarray(diffs)
    boots = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n_boot)]
    return np.mean(diffs), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def fig_forest():
    """Forest plot of the 13-seed paired differences against the baseline."""
    d = pd.read_csv(PATHS["dccam"])
    base = d[d.strategy == "baseline"].set_index("seed")
    arms = ["grl_only", "distcorr", "camreg", "full"]
    names_ = {"grl_only": "GRL", "distcorr": "Distcorr", "camreg": "Camreg", "full": "Full"}
    metrics = [("worst_group_gap", "(a) Worst-group gap"), ("sas", "(b) SAS"),
               ("probe_linear_mean", "(c) Density probe"), ("auroc_all", "(d) AUROC")]
    fig, axes = plt.subplots(1, 4, figsize=(PAGE_W, 1.6), gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    ys = np.arange(len(arms))[::-1]
    for ax, (m, title) in zip(axes, metrics):
        for y, a in zip(ys, arms):
            arm = d[d.strategy == a].set_index("seed")
            common = base.index.intersection(arm.index)
            mean, lo, hi = _bootstrap_ci(arm.loc[common, m].values - base.loc[common, m].values)
            col = "#e34948" if (lo > 0 or hi < 0) else GRAY
            ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="o", color=col, ms=4, capsize=2,
                        elinewidth=0.9, markeredgecolor="white", markeredgewidth=0.5)
        ax.axvline(0, color="#52514e", lw=0.7, ls="--")
        ax.set_title(title, loc="left"); ax.grid(axis="y", visible=False)
        ax.set_yticks(ys)
        ax.set_yticklabels([names_[a] for a in arms] if ax is axes[0] else [])
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    fig.supxlabel("Paired difference against the baseline (13 seeds, 95% bootstrap CI)", fontsize=7.5, y=0.02)
    fig.tight_layout(w_pad=0.8, pad=0.3, rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(OUT_DIR, "forest_plot_main.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_forest()
    fig_sweeps_combined()
    fig_grl()
    print("Figures written to", OUT_DIR)
