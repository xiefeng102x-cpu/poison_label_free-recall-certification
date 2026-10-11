#!/usr/bin/env python3
"""Create the confirmation member-versus-fresh calibration figure."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/bloodmnist_fresh_query_calibration_confirmation_v1/per_seed_view_metrics.csv"
OUTPUT = ROOT / "results/top_journal_program_v1/figures"
VIEWS = (
    ("suspect_hidden64", "Suspect hidden64"),
    ("paired_hidden64_drift", "Paired hidden64 drift (oracle clean checkpoint)"),
)


def read_rows() -> list[dict[str, str]]:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = read_rows()
    seeds = sorted({int(row["seed"]) for row in rows})
    if seeds != [74, 75, 76, 77, 78] or len(rows) != 10:
        raise RuntimeError("unexpected frozen confirmation rows")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), sharey=True)
    colors = {"member": "#326a9b", "fresh": "#e1832f"}
    for axis, (view, title) in zip(axes, VIEWS):
        selected = {int(row["seed"]): row for row in rows if row["view"] == view}
        member = np.asarray([float(selected[seed]["member_fpr"]) for seed in seeds])
        fresh = np.asarray([float(selected[seed]["fresh_fpr"]) for seed in seeds])
        x = np.arange(len(seeds) + 1, dtype=float)
        member_values = np.r_[member, member.mean()]
        fresh_values = np.r_[fresh, fresh.mean()]
        for index in range(len(x)):
            axis.plot(
                [x[index], x[index]],
                [member_values[index] * 100, fresh_values[index] * 100],
                color="#aab3ba",
                linewidth=1.2,
                zorder=1,
            )
        axis.scatter(
            x - 0.08,
            member_values * 100,
            color=colors["member"],
            marker="o",
            s=34,
            label="Training-member clean",
            zorder=3,
        )
        axis.scatter(
            x + 0.08,
            fresh_values * 100,
            color=colors["fresh"],
            marker="s",
            s=32,
            label="Fresh test clean",
            zorder=3,
        )
        axis.axhline(5.0, color="#333333", linestyle="--", linewidth=1.0, label="Nominal 5%")
        axis.axhline(100 * 4 / 97, color="#777777", linestyle=":", linewidth=1.0, label="Rank-grid null 4/97")
        axis.axvline(len(seeds) - 0.5, color="#c4c9cd", linewidth=0.8)
        axis.set_xticks(x, [str(seed) for seed in seeds] + ["Mean"])
        axis.set_title(title)
        axis.set_xlabel("Confirmation seed")
        axis.grid(axis="y", color="#e8ebed", linewidth=0.7)
        axis.set_axisbelow(True)
        gap = 100 * (member.mean() - fresh.mean())
        axis.text(
            0.02,
            0.97,
            f"Mean member − fresh = {gap:+.2f} pp",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
        )
    axes[0].set_ylabel("Empirical false-positive rate (%)")
    axes[0].set_ylim(0.0, 8.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(
        "Fresh queries do not remove calibration shift; its direction depends on the observable view",
        y=1.13,
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUTPUT / f"fig07_fresh_query_calibration.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
