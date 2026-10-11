#!/usr/bin/env python3
"""Render the formal trusted-identity non-interference result."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "identity"
OUTPUT = ROOT / "figs" / "fig8_identity_noninterference"
MANUSCRIPT_OUTPUT = OUTPUT

LABELS = {
    "public_hash": "Public hash",
    "same_detector_low": "Low score",
    "same_detector_high": "High score",
    "multiscore_extreme": "Multi-score",
    "shadow_frequency": "Shadow freq.",
    "shadow_ridge": "Shadow ridge",
    "oracle_key": "Oracle key",
}
MARKERS = ["o", "s", "^", "D", "P", "X"]
BLUE = "#0072B2"
ORANGE = "#D55E00"
GOLD = "#E69F00"
GREY = "#5F6368"


def main() -> None:
    cells = pd.read_csv(RESULTS / "identity_inference_primary_cells.csv")
    pooled = pd.read_csv(RESULTS / "identity_inference_primary_pooled.csv")
    methods = [value for value in pooled["method"] if value != "oracle_key"]
    y = np.arange(len(methods))[::-1]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2, "axes.titlesize": 9.2,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.8,
        "legend.fontsize": 7.5, "axes.linewidth": 0.7,
    })
    fig, axes = plt.subplots(
        1, 3, figsize=(7.15, 2.55),
        gridspec_kw={"wspace": 0.50, "width_ratios": [1.15, 1.15, 1.28]},
    )

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#D9DDE2", linewidth=0.55, zorder=0)
        ax.tick_params(length=3, width=0.7)

    ax = axes[0]
    ax.axvspan(0.975, 1.025, color=BLUE, alpha=0.10, zorder=0)
    ax.axvline(1.0, color=GREY, linestyle="--", linewidth=0.9)
    for idx, method in enumerate(methods):
        subset = cells[cells.method == method]
        point = float(pooled.loc[pooled.method == method, "mean_precision_lift"].iloc[0])
        lo = float(subset.precision_lift_ci_lo.min()); hi = float(subset.precision_lift_ci_hi.max())
        ax.errorbar(point, y[idx], xerr=[[point - lo], [hi - point]], fmt=MARKERS[idx],
                    color=BLUE, markerfacecolor="white", markeredgewidth=1.0,
                    markersize=4.8, capsize=2.2, linewidth=0.9, zorder=3)
    ax.set_yticks(y, [LABELS[m] for m in methods])
    ax.set_xlim(0.97, 1.03); ax.set_xticks([0.975, 1.0, 1.025])
    ax.set_xlabel("Precision lift over random")
    ax.set_title("(a) Identity-ranking lift", loc="left", fontweight="bold")

    ax = axes[1]
    ax.axvspan(0.49, 0.51, color=BLUE, alpha=0.10, zorder=0)
    ax.axvline(0.5, color=GREY, linestyle="--", linewidth=0.9)
    for idx, method in enumerate(methods):
        subset = cells[cells.method == method]
        point = float(pooled.loc[pooled.method == method, "mean_auroc"].iloc[0])
        lo = float(subset.auroc_ci_lo.min()); hi = float(subset.auroc_ci_hi.max())
        ax.errorbar(point, y[idx], xerr=[[point - lo], [hi - point]], fmt=MARKERS[idx],
                    color=BLUE, markerfacecolor="white", markeredgewidth=1.0,
                    markersize=4.8, capsize=2.2, linewidth=0.9, zorder=3)
    ax.set_yticks(y, [])
    ax.set_xlim(0.485, 0.515); ax.set_xticks([0.49, 0.50, 0.51])
    ax.set_xlabel("Trusted-identity AUROC")
    ax.set_title("(b) Identity AUROC", loc="left", fontweight="bold")

    ax = axes[2]
    ax.grid(False)
    ax.grid(axis="y", color="#D9DDE2", linewidth=0.55, zorder=0)
    plot_methods = methods + ["oracle_key"]
    values = [100 * float(pooled.loc[pooled.method == method, "mean_target_corruption"].iloc[0]) for method in plot_methods]
    x = np.arange(len(plot_methods))
    colors = [BLUE] * len(methods) + [ORANGE]
    hatches = ["//", "..", "xx", "--", "++", "oo", "\\\\"]
    bars = ax.bar(x, values, color=["white"] * len(x), edgecolor=colors, linewidth=1.05,
                  hatch=hatches, zorder=2)
    ax.axhline(10.0, color=GOLD, linestyle=(0, (4, 2)), linewidth=1.2)
    ax.text(-0.42, 10.8, "Stress ceiling 10%", ha="left", va="bottom",
            color="#8A5A00", fontsize=6.9, fontweight="bold")
    ax.text(-0.42, 13.4, "Non-oracle max: 7.95%", ha="left",
            va="bottom", fontsize=6.9, color=BLUE, fontweight="bold")
    ax.text(bars[-1].get_x() + bars[-1].get_width() / 2, values[-1] + 0.7,
            f"{values[-1]:.1f}%", ha="center", va="bottom", fontsize=7.0,
            color=ORANGE, fontweight="bold")
    ax.set_xticks(x, ["Hash", "Low", "High", "Multi", "S-freq", "S-ridge", "Oracle"],
                  rotation=48, ha="right")
    ax.set_ylim(0, 30); ax.set_ylabel("Trusted set corrupted")
    ax.set_title("(c) Trusted-set corruption", loc="left", fontweight="bold", fontsize=8.7)

    fig.text(0.01, 0.01,
             "Points are pooled means. Whiskers in (a,b) span the extrema of 10 Bonferroni-adjusted cell CIs per attack; shaded bands are preregistered equivalence regions. Oracle AUROC = 1.00.",
             ha="left", va="bottom", fontsize=6.8, color=GREY)
    fig.subplots_adjust(left=0.14, right=0.995, top=0.90, bottom=0.30)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for output in (OUTPUT, MANUSCRIPT_OUTPUT):
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
