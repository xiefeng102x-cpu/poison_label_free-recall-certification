#!/usr/bin/env python3
"""Render the verified P6-C QML exact/shot/noise bridge figure."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAPER14 = Path(__file__).resolve().parents[1]
RESULT = PAPER14 / "results/p6c_qml_noise_bridge_v1"
OUT = PAPER14 / "results/top_journal_program_v1/figures"


def main() -> None:
    with (RESULT / "paired_summary.json").open() as handle:
        summary = json.load(handle)

    with (RESULT / "diagnostic_2048/per_seed_metrics.csv").open(newline="") as handle:
        diagnostic_rows = [
            row for row in csv.DictReader(handle) if row["victim_kind"] == "backdoor"
        ]
    diagnostic_auc = {row["seed"]: float(row["auc"]) for row in diagnostic_rows}
    diagnostic_tpr = np.mean([float(row["tpr"]) for row in diagnostic_rows])
    diagnostic_fpr = np.mean([float(row["fpr"]) for row in diagnostic_rows])

    keys = [
        "exact",
        "finite_shot_512",
        "light_noise_512",
        "nominal_noise_512",
        "nominal_noise_2048",
        "heavy_noise_512",
    ]
    labels = ["Exact", "512 shots", "Light", "Nominal\n512", "Nominal\n2048 (diag.)", "Heavy"]
    cond = summary["conditions"]
    cond["nominal_noise_2048"] = {
        "mean_auc": float(np.mean(list(diagnostic_auc.values()))),
        "per_seed_auc": diagnostic_auc,
        "pooled_tpr": float(diagnostic_tpr),
        "pooled_fpr": float(diagnostic_fpr),
    }
    x = np.arange(len(keys))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9), constrained_layout=True)

    ax = axes[0]
    for seed, marker in zip(("42", "43", "44"), ("o", "s", "^")):
        values = [cond[key]["per_seed_auc"][seed] for key in keys]
        ax.plot(x, values, marker=marker, linewidth=1.15, markersize=5,
                alpha=0.75, label=f"seed {seed}")
    means = [cond[key]["mean_auc"] for key in keys]
    ax.plot(x, means, color="black", marker="D", linewidth=2.1,
            markersize=5.5, label="mean")
    ax.axhline(0.95, color="#b22222", linestyle="--", linewidth=1.2,
               label="nominal AUC gate")
    ax.axvspan(3.65, 4.35, color="#777777", alpha=0.08, linewidth=0)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylim(0.80, 1.01)
    ax.set_ylabel("Backdoor-victim test AUC")
    ax.set_title("(a) Rank separation largely survives nominal noise")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")

    ax = axes[1]
    width = 0.36
    tpr = [cond[key]["pooled_tpr"] for key in keys]
    fpr = [cond[key]["pooled_fpr"] for key in keys]
    ax.bar(x - width / 2, tpr, width, label="pooled TPR", color="#356a9a")
    ax.bar(x + width / 2, fpr, width, label="pooled FPR", color="#d0773c")
    ax.axhline(0.90, color="#356a9a", linestyle=":", linewidth=1.2,
               label="TPR gate (0.90)")
    ax.axhline(0.05, color="#b22222", linestyle="--", linewidth=1.2,
               label="FPR gate (0.05)")
    ax.axvspan(3.65, 4.35, color="#777777", alpha=0.08, linewidth=0)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Frozen-threshold rate")
    ax.set_title("(b) The exact-calibrated operating point does not transfer")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle("Three-seed QML simulator-to-noise bridge (frozen detector and threshold)",
                 fontsize=11)
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = OUT / f"fig08_qml_noise_bridge.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
