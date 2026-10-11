#!/usr/bin/env python3
"""Make the theory-probe and multiclass-applicability summary figure."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
THEORY = ROOT / "results/checkpoint_auditability_theory_probe_v1"
RECEIPT = ROOT / "results/top_journal_program_v1/multiclass_backdoorbench_program_receipt.json"
OUTPUT = ROOT / "results/top_journal_program_v1/figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    aggregate = read_csv(THEORY / "aggregate_auc_deltas.csv")
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    summary = receipt["confirmation"]["eligible_summary"]
    OUTPUT.mkdir(parents=True, exist_ok=True)

    colors = {"suspect_hidden64": "#3265a8", "clean_to_suspect_hidden64_drift": "#e07a2f"}
    labels = {
        "backdoorbench_spectral": "Spectral",
        "backdoorbench_scan": "SCAn",
        "backdoorbench_spectre_oracle_target": "SPECTRE\noracle",
        "backdoorbench_spectre_uniform_5pct": "SPECTRE\n5%",
    }
    methods = list(labels)
    views = list(colors)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))

    ax = axes[0]
    positions = np.arange(len(aggregate))
    values = np.asarray([float(row["mean_auc_delta"]) for row in aggregate])
    low = np.asarray([float(row["minimum_auc_delta"]) for row in aggregate])
    high = np.asarray([float(row["maximum_auc_delta"]) for row in aggregate])
    bar_colors = ["#8f6db0" if value >= 0 else "#9a9a9a" for value in values]
    ax.bar(positions, values, color=bar_colors, width=0.72)
    ax.errorbar(
        positions,
        values,
        yerr=np.vstack([values - low, high - values]),
        fmt="none",
        ecolor="#333333",
        capsize=3,
        linewidth=1,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(
        positions,
        [f"{row['checkpoint'][0].upper()}-{row['representation']}" for row in aggregate],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("Drift minus absolute test AUC")
    ax.set_title("(a) No general drift advantage")
    ax.grid(axis="y", alpha=0.2)

    width = 0.35
    x = np.arange(len(methods))
    for view_index, view in enumerate(views):
        selected = {row["method"]: row for row in summary if row["view"] == view}
        offset = (view_index - 0.5) * width
        axes[1].bar(
            x + offset,
            [100 * selected[method]["pooled_tpr"] for method in methods],
            width,
            color=colors[view],
            label="Suspect" if view == "suspect_hidden64" else "Paired drift (oracle)",
        )
        axes[2].bar(
            x + offset,
            [100 * selected[method]["pooled_fpr_all_clean"] for method in methods],
            width,
            color=colors[view],
        )
    for ax, title, ylabel in (
        (axes[1], "(b) Five-seed poison recall", "Pooled TPR (%)"),
        (axes[2], "(c) Five-seed clean false positives", "All-clean FPR (%)"),
    ):
        ax.set_xticks(x, [labels[method] for method in methods])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
    axes[1].set_ylim(0, 105)
    axes[2].set_ylim(0, 14)
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        "Checkpoint-difference boundary and multiclass BackdoorBench applicability",
        fontsize=11,
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUTPUT / f"fig06_auditability_multiclass.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
