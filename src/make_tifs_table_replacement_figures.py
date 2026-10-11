#!/usr/bin/env python3
"""Generate the TIFS figures that replace two dense result tables.

The script reads the archived CSV outputs used by the manuscript rather than
copying values from LaTeX.  It also checks the expected scenario/method/grid
coverage before writing publication-ready PDF and PNG files.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SCENARIOS = [
    ("fixed_attack_local_angle", "Fixed local QML"),
    ("whitebox_representation_matching_local_angle", "White-box local QML"),
    ("fixed_attack_residual_reupload", "Residual-reupload QML"),
]

METHODS = [
    ("cf_cmd", "CF", "o", "#0072B2", True),
    ("cfu", "CFU (development-selected)", "D", "#D55E00", False),
    ("spectral_signatures", "Spectral", "^", "#E69F00", True),
    ("spectre_que", "SPECTRE/QUE", "s", "#009E73", True),
    ("activation_clustering", "Activation Clustering", "P", "#56B4E9", True),
    ("scan_style", "SCAn-style", "X", "#CC79A7", True),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def load_baseline_points(root: Path) -> dict[tuple[str, str], tuple[float, float]]:
    strong_path = root / "results/bloodmnist_strong_baselines_v1/eligible_summary.csv"
    cfu_path = root / "results/bloodmnist_no_whitening_development_v1/eligible_summary.csv"

    points: dict[tuple[str, str], tuple[float, float]] = {}
    for row in read_csv(strong_path):
        if row["view"] != "checkpoint_drift_xz16":
            continue
        key = (row["scenario"], row["method"])
        points[key] = (100.0 * float(row["pooled_fpr"]), 100.0 * float(row["pooled_tpr"]))

    for row in read_csv(cfu_path):
        key = (row["scenario"], "cfu")
        points[key] = (100.0 * float(row["pooled_fpr"]), 100.0 * float(row["pooled_tpr"]))

    expected = {(scenario, method) for scenario, _ in SCENARIOS for method, *_ in METHODS}
    missing = sorted(expected - points.keys())
    if missing:
        raise ValueError(f"Missing baseline points: {missing}")
    return points


def make_baseline_figure(root: Path, output_dir: Path) -> None:
    points = load_baseline_points(root)
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.72), sharex=True, sharey=True)

    for index, ((scenario, title), ax) in enumerate(zip(SCENARIOS, axes)):
        ax.axhspan(90.0, 97.5, color="#E8F4EA", zorder=0)
        ax.axhline(90.0, color="#4D4D4D", linewidth=0.85, linestyle="--", zorder=1)
        ax.axvline(5.0, color="#777777", linewidth=0.75, linestyle=":", zorder=1)

        for method, label, marker, color, filled in METHODS:
            fpr, tpr = points[(scenario, method)]
            ax.scatter(
                fpr,
                tpr,
                s=43,
                marker=marker,
                facecolors=color if filled else "white",
                edgecolors=color,
                linewidths=1.25,
                label=label,
                zorder=3,
            )

        ax.set_title(title, pad=5)
        ax.set_xlim(0.0, 5.35)
        ax.set_ylim(40.0, 97.5)
        ax.set_xticks(np.arange(0, 5.1, 1.0))
        ax.set_yticks([40, 50, 60, 70, 80, 90])
        ax.grid(axis="both", color="#D9D9D9", linewidth=0.5, alpha=0.8, zorder=0)
        ax.set_xlabel("False-positive rate (%)")
        if index == 0:
            ax.set_ylabel("True-positive rate (%)")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
        columnspacing=1.25,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.84, bottom=0.26, wspace=0.18)

    for suffix, dpi in (("pdf", None), ("png", 300)):
        fig.savefig(
            output_dir / f"fig03_matched_baseline_tradeoff.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def load_budget_grid(root: Path) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    source = root / "results/reference_calibration_budget_sensitivity_v1/eligible_grid_summary.csv"
    rows = read_csv(source)
    references = sorted({int(row["reference_count"]) for row in rows})
    calibrations = sorted({int(row["calibration_count"]) for row in rows})
    if references != [8, 16, 32, 64, 120] or calibrations != [20, 40, 80]:
        raise ValueError(
            f"Unexpected budget grid: references={references}, calibrations={calibrations}"
        )

    lookup = {
        (int(row["reference_count"]), int(row["calibration_count"])): row for row in rows
    }
    expected = {(reference, calibration) for reference in references for calibration in calibrations}
    missing = sorted(expected - lookup.keys())
    if missing:
        raise ValueError(f"Missing budget cells: {missing}")

    tpr = np.array(
        [[100.0 * float(lookup[(reference, calibration)]["pooled_tpr"]) for calibration in calibrations]
         for reference in references]
    )
    fpr = np.array(
        [[100.0 * float(lookup[(reference, calibration)]["pooled_fpr"]) for calibration in calibrations]
         for reference in references]
    )
    return tpr, fpr, references, calibrations


def annotate_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    cmap: str,
    lower: float,
    upper: float,
    threshold: float | None = None,
) -> None:
    color_map = mpl.colormaps[cmap]
    normalizer = mpl.colors.Normalize(vmin=lower, vmax=upper)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            red, green, blue, _ = color_map(normalizer(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            color = "#111111" if luminance >= 0.48 else "white"
            weight = "bold" if threshold is not None and value >= threshold else "normal"
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8.0,
                fontweight=weight,
            )
            if threshold is not None and value >= threshold:
                ax.add_patch(
                    mpl.patches.Rectangle(
                        (column - 0.49, row - 0.49),
                        0.98,
                        0.98,
                        fill=False,
                        edgecolor="black",
                        linewidth=1.25,
                    )
                )


def make_budget_figure(root: Path, output_dir: Path) -> None:
    tpr, fpr, references, calibrations = load_budget_grid(root)
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.12), constrained_layout=True)

    specifications = [
        (axes[0], tpr, "(a) True-positive rate (%)", "YlGnBu", 40.0, 100.0, 90.0),
        (
            axes[1],
            fpr,
            "(b) False-positive rate (%) - lower is better",
            "cividis",
            0.0,
            5.0,
            None,
        ),
    ]
    for ax, values, title, cmap, lower, upper, threshold in specifications:
        image = ax.imshow(values, cmap=cmap, vmin=lower, vmax=upper, aspect="auto")
        ax.set_title(title, pad=7)
        ax.set_xticks(range(len(calibrations)), labels=[str(value) for value in calibrations])
        ax.set_yticks(range(len(references)), labels=[str(value) for value in references])
        ax.set_xlabel("Calibration rows")
        ax.set_ylabel("Trusted-reference rows")
        ax.set_xticks(np.arange(-0.5, len(calibrations), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(references), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        annotate_heatmap(
            ax,
            values,
            cmap=cmap,
            lower=lower,
            upper=upper,
            threshold=threshold,
        )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
        colorbar.ax.tick_params(labelsize=7)

    for suffix, dpi in (("pdf", None), ("png", 300)):
        fig.savefig(
            output_dir / f"fig04_reference_calibration_heatmaps.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="paper14 root directory",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = root / "results/top_journal_program_v1/figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    make_baseline_figure(root, output_dir)
    make_budget_figure(root, output_dir)
    print(f"Wrote four figure files to {output_dir}")


if __name__ == "__main__":
    main()
