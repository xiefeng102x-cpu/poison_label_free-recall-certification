#!/usr/bin/env python3
"""Generate publication figures directly from frozen paper14 result tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BLUE = "#2F5D8A"
ORANGE = "#D9822B"
GOLD = "#C6A645"
OLIVE = "#68783D"
PINK = "#B75D69"
PURPLE = "#6F4E7C"
CHARCOAL = "#263238"
GREY = "#A7AFB5"
LIGHT_GREY = "#E7EAED"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": CHARCOAL,
            "axes.labelcolor": CHARCOAL,
            "xtick.color": CHARCOAL,
            "ytick.color": CHARCOAL,
            "text.color": CHARCOAL,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": LIGHT_GREY,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.85,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def percent_axis(ax: plt.Axes, *, upper: float = 1.0) -> None:
    ax.set_xlim(0.0, upper)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))


def audit_pipeline(out: Path) -> None:
    """Draw a five-stage audit pipeline with explicit privilege lanes."""

    fig, ax = plt.subplots(figsize=(11.6, 4.55))
    fig.subplots_adjust(left=0.012, right=0.988, top=0.985, bottom=0.035)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        body: str,
        footer: str,
        *,
        face: str,
        edge: str = CHARCOAL,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=1.25,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(
            x + 0.012,
            y + h - 0.04,
            title,
            ha="left",
            va="top",
            fontsize=9.8,
            fontweight="bold",
            color=edge,
        )
        ax.text(
            x + 0.012,
            y + h - 0.105,
            body,
            ha="left",
            va="top",
            fontsize=8.6,
            linespacing=1.24,
            color=CHARCOAL,
        )
        ax.plot(
            [x + 0.012, x + w - 0.012],
            [y + 0.064, y + 0.064],
            color=edge,
            linewidth=0.55,
            alpha=0.55,
        )
        ax.text(
            x + w / 2,
            y + 0.031,
            footer,
            ha="center",
            va="center",
            fontsize=7.7,
            fontstyle="italic",
            color=edge,
        )

    def arrow(x1: float, y1: float, x2: float, y2: float, *, color: str = CHARCOAL,
              linestyle: str = "-", connectionstyle: str = "arc3") -> None:
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=11,
            linewidth=1.25, color=color, linestyle=linestyle,
            connectionstyle=connectionstyle,
        ))

    attack_fill = "#F8E9E6"
    trust_fill = "#E8F1F8"
    process_fill = "#F4F6F7"
    observe_fill = "#ECF3F8"
    score_fill = "#F7F2E5"
    calibration_fill = "#EEF4EA"
    decision_fill = "#E6EFDE"
    attack_edge = "#9C3D35"

    # Attacker control and knowledge are separate lanes.
    attack_patch = FancyBboxPatch(
        (0.02, 0.765),
        0.18,
        0.17,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.2,
        edgecolor=attack_edge,
        facecolor=attack_fill,
    )
    ax.add_patch(attack_patch)
    ax.text(
        0.032,
        0.9,
        "ATTACKER CONTROL",
        fontsize=9.3,
        fontweight="bold",
        color=attack_edge,
        ha="left",
        va="top",
    )
    ax.text(
        0.032,
        0.845,
        "poisoned training subset\nfixed budget + trigger/labels",
        fontsize=8.3,
        color=CHARCOAL,
        ha="left",
        va="top",
        linespacing=1.2,
    )
    ax.plot([0.245, 0.955], [0.885, 0.885], color=attack_edge, linestyle="--", linewidth=1.2)
    arrow(0.93, 0.885, 0.958, 0.885, color=attack_edge, linestyle="--")
    ax.text(
        0.60,
        0.92,
        "ADAPTIVE KNOWLEDGE: observation + scoring + calibration + threshold",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color=attack_edge,
    )

    # Five-stage decision spine.
    xs = [0.02, 0.216, 0.412, 0.608, 0.804]
    width = 0.166
    y = 0.335
    height = 0.34
    stages = [
        ("1  AUDIT INPUTS", "suspect model M1\ncandidate batch C\nfixed access level", "victim eligible?", process_fill, CHARCOAL),
        ("2  OBSERVE", "absolute: h(M1,x)\npaired: h(M1,x)-h(M0,x)", "channel identifiable?", observe_fill, BLUE),
        ("3  CROSS-FIT SCORE", "seeded A/B split\nfit on opposite half\nexclude query from fit", "direction stable?", score_fill, "#8A6A16"),
        ("4  CALIBRATE", "clean rank on K\np-value at alpha\nfrozen threshold", "law transports?", calibration_fill, OLIVE),
        ("5  DECIDE", "flag candidate\npass or abstain\nrecord provenance", "bounded action", decision_fill, OLIVE),
    ]
    for x, (title, body, footer, face, edge) in zip(xs, stages):
        box(x, y, width, height, title, body, footer, face=face, edge=edge)
    for left, right in zip(xs[:-1], xs[1:]):
        arrow(left + width + 0.006, y + height / 2, right - 0.006, y + height / 2)
    arrow(0.20, 0.80, xs[0] + width / 2, y + height + 0.006, color=attack_edge)

    # Trusted state feeds only the stages that consume it.
    trusted = FancyBboxPatch(
        (0.20, 0.04),
        0.60,
        0.19,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.2,
        edgecolor=BLUE,
        facecolor=trust_fill,
    )
    ax.add_patch(trusted)
    ax.text(
        0.215,
        0.205,
        "DEFENDER TRUSTED ROOT - immutable to attacker",
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color=BLUE,
        zorder=5,
        bbox={"facecolor": trust_fill, "edgecolor": "none", "pad": 1.2},
    )
    trusted_items = [
        (xs[1] + width / 2, "optional clean M0", xs[1] + width / 2),
        (xs[2] + width / 2, "clean reference R", xs[2] + width / 2),
        (xs[3] + width / 2, "clean calibration K", xs[3] + width / 2),
    ]
    for label_x, label, target_x in trusted_items:
        ax.text(
            label_x,
            0.12,
            label,
            ha="center",
            va="center",
            fontsize=8.1,
            color=BLUE,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": BLUE, "linewidth": 0.8},
        )
        arrow(target_x, 0.16, target_x, y - 0.008, color=BLUE)

    ax.text(
        0.955,
        0.075,
        "Red = attacker\nBlue = trusted state\nDark arrows = audit flow",
        ha="right",
        va="bottom",
        fontsize=7.6,
        color=CHARCOAL,
        linespacing=1.25,
    )

    save(fig, out, "fig00_threat_observation_pipeline")


def strong_baselines(results: Path, out: Path) -> None:
    df = pd.read_csv(results / "bloodmnist_strong_baselines_v1" / "eligible_summary.csv")
    df = df[df["view"].eq("checkpoint_drift_xz16")].copy()
    scenario_order = [
        "fixed_attack_local_angle",
        "whitebox_representation_matching_local_angle",
        "fixed_attack_residual_reupload",
    ]
    scenario_labels = ["Fixed attack\nlocal QML", "White-box adaptive\nlocal QML", "Fixed attack\nresidual QML"]
    method_order = ["cf_cmd", "spectral_signatures", "spectre_que", "activation_clustering", "scan_style"]
    method_labels = ["CF-CMD", "Spectral", "SPECTRE/QUE", "Activation clustering", "SCAn-style"]
    colors = [BLUE, ORANGE, GOLD, OLIVE, PINK]
    assert set(scenario_order).issubset(set(df["scenario"]))
    assert set(method_order).issubset(set(df["method"]))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), gridspec_kw={"wspace": 0.16})
    fig.subplots_adjust(top=0.72, bottom=0.17, left=0.08, right=0.98)
    width = 0.15
    x = np.arange(len(scenario_order))
    for metric, ax, gate, title in [
        ("pooled_tpr", axes[0], 0.90, "Detection rate (higher is better)"),
        ("pooled_fpr", axes[1], 0.05, "False-positive rate (lower is better)"),
    ]:
        for index, (method, label, color) in enumerate(zip(method_order, method_labels, colors)):
            values = []
            for scenario in scenario_order:
                row = df[(df["scenario"].eq(scenario)) & (df["method"].eq(method))]
                assert len(row) == 1
                values.append(float(row.iloc[0][metric]))
            offset = (index - 2) * width
            ax.bar(x + offset, values, width=width, color=color, edgecolor="white", linewidth=0.5, label=label)
        ax.axhline(gate, color=CHARCOAL, linestyle="--", linewidth=1.1, label="Gate" if metric == "pooled_tpr" else None)
        ax.set_xticks(x, scenario_labels)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_ylim(0, 1.02 if metric == "pooled_tpr" else 0.10)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc="upper center", bbox_to_anchor=(0.53, 0.86), fontsize=8.2)
    fig.suptitle("Strong baselines expose architecture and adaptive-attack gaps", x=0.03, y=0.98, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.03, 0.025, "Matched XZ16 checkpoint-drift view; pooled over attack-eligible seeds. Dashed lines show the pre-registered gates.", fontsize=8)
    save(fig, out, "fig01_strong_baseline_comparison")


def adaptive_forest(results: Path, out: Path) -> None:
    adaptive = pd.read_csv(results / "bloodmnist_whitebox_adaptive_confirmation_v1" / "eligible_summary.csv")
    primary = adaptive[adaptive["representation"].eq("coherent_bonferroni_input_qml_xz")].copy()
    residual = pd.read_csv(results / "bloodmnist_residual_reupload_confirmation_v1" / "eligible_summary.csv")
    residual = residual[residual["representation"].eq("coherent_bonferroni_input_residual_reupload_xz")]
    revised = pd.read_csv(results / "bloodmnist_covariance_free_confirmation_v1" / "eligible_summary.csv")
    revised = revised[revised["scenario"].eq("fixed_pca_additive")]
    final_observable = json.loads(
        (results / "p6d_final_observable_adaptive_v4" / "confirmation" / "confirmation_summary.json").read_text(
            encoding="utf-8"
        )
    )

    rows = [
        ("Fixed local attack (CFU)", revised.iloc[0]["pooled_tpr"], revised.iloc[0]["tpr_ci_low"], revised.iloc[0]["tpr_ci_high"], 7, BLUE, "D", False),
        ("Representation matching (CF)", primary[primary["mode"].eq("checkpoint_drift_representation_matching")].iloc[0]["pooled_tpr"], primary[primary["mode"].eq("checkpoint_drift_representation_matching")].iloc[0]["tpr_seed_bootstrap_ci_low"], primary[primary["mode"].eq("checkpoint_drift_representation_matching")].iloc[0]["tpr_seed_bootstrap_ci_high"], 7, ORANGE, "o", True),
        ("Multi-direction adaptive (CF)", primary[primary["mode"].eq("multi_direction_coherence_breaking")].iloc[0]["pooled_tpr"], primary[primary["mode"].eq("multi_direction_coherence_breaking")].iloc[0]["tpr_seed_bootstrap_ci_low"], primary[primary["mode"].eq("multi_direction_coherence_breaking")].iloc[0]["tpr_seed_bootstrap_ci_high"], 5, GOLD, "o", True),
        ("Joint input/internal adaptive (CF)", primary[primary["mode"].eq("joint_input_internal_detector_evasion")].iloc[0]["pooled_tpr"], primary[primary["mode"].eq("joint_input_internal_detector_evasion")].iloc[0]["tpr_seed_bootstrap_ci_low"], primary[primary["mode"].eq("joint_input_internal_detector_evasion")].iloc[0]["tpr_seed_bootstrap_ci_high"], 7, PINK, "o", True),
        ("Final-observable adaptive (CFU)\N{DAGGER}", final_observable["mean_seed_tpr"], final_observable["mean_seed_tpr_ci_low"], final_observable["mean_seed_tpr_ci_high"], final_observable["attack_viable_count"], PURPLE, "^", False),
        ("Residual-reupload QML (CF)", residual.iloc[0]["pooled_tpr"], residual.iloc[0]["tpr_seed_bootstrap_ci_low"], residual.iloc[0]["tpr_seed_bootstrap_ci_high"], 8, OLIVE, "o", True),
    ]
    labels = [r[0] for r in rows]
    estimates = np.array([float(r[1]) for r in rows])
    lows = np.array([float(r[2]) for r in rows])
    highs = np.array([float(r[3]) for r in rows])
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 4.15))
    fig.subplots_adjust(top=0.93, bottom=0.23, left=0.31, right=0.97)
    ax.axvspan(0.90, 1.0, color="#E8F4EA", zorder=0)
    for yi, est, lo, hi, row in zip(y, estimates, lows, highs, rows):
        color, marker, filled = row[5], row[6], row[7]
        ax.errorbar(
            est,
            yi,
            xerr=[[est - lo], [hi - est]],
            fmt=marker,
            color=color,
            ecolor=color,
            markerfacecolor=color if filled else "white",
            markeredgecolor=color,
            markeredgewidth=1.4,
            capsize=3.5,
            markersize=7,
            linewidth=1.7,
        )
        if hi > 0.85:
            label_x, alignment = lo - 0.018, "right"
        else:
            label_x, alignment = hi + 0.025, "left"
        ax.text(label_x, yi, f"{est:.1%}  (n={row[4]})", va="center", ha=alignment, fontsize=9)
    ax.axvline(0.90, color=CHARCOAL, linestyle="--", linewidth=1.1)
    ax.text(0.895, len(rows) - 0.67, "90% gate", ha="right", va="center", fontsize=8.0, color=CHARCOAL)
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.2, len(rows) - 0.5)
    percent_axis(ax)
    ax.set_xlabel("Pooled true-positive rate (95% seed-block bootstrap interval)")
    ax.grid(axis="y", visible=False)
    fig.text(
        0.31,
        0.035,
        "\N{DAGGER} Conditional on 7/10 attack-viable victims; the separate 8/10 viability gate was not met.",
        fontsize=7.8,
    )
    save(fig, out, "fig02_adaptive_architecture_forest")


def ablations(results: Path, out: Path) -> None:
    df = pd.read_csv(results / "bloodmnist_mechanism_ablations_v1" / "eligible_summary.csv")
    order = [
        "candidate_reference_xz16",
        "no_whitening_xz16",
        "no_sample_exclusion_xz16",
        "no_conformal_gaussian_xz16",
        "dimension_z_plus_zz36",
        "internal_xz16_only",
        "dimension_pauli128",
        "dimension_z8",
        "bonferroni_input_plus_xz16",
        "input_only",
        "no_checkpoint_difference_absolute_xz16",
    ]
    labels = {
        "candidate_reference_xz16": "Candidate-as-reference (historical)",
        "no_whitening_xz16": "No whitening (CF-CMD-U)",
        "no_sample_exclusion_xz16": "No sample exclusion",
        "no_conformal_gaussian_xz16": "Gaussian instead of conformal",
        "dimension_z_plus_zz36": "Z + ZZ (36D)",
        "internal_xz16_only": "Frozen XZ16",
        "dimension_pauli128": "Pauli (128D)",
        "dimension_z8": "Z only (8D)",
        "bonferroni_input_plus_xz16": "Input + XZ16",
        "input_only": "Input only",
        "no_checkpoint_difference_absolute_xz16": "No checkpoint difference",
    }
    indexed = df.set_index("representation").loc[order]
    y = np.arange(len(order))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2), sharey=True, gridspec_kw={"width_ratios": [1.7, 1.0], "wspace": 0.09})
    fig.subplots_adjust(top=0.84, bottom=0.15, left=0.27, right=0.97)
    tpr_colors = [BLUE if name == "internal_xz16_only" else ORANGE if name == "no_whitening_xz16" else GREY for name in order]
    axes[0].barh(y, indexed["pooled_tpr"], color=tpr_colors, edgecolor="white")
    axes[0].axvline(0.90, color=CHARCOAL, linestyle="--", linewidth=1.1)
    axes[0].set_yticks(y, [labels[name] for name in order])
    axes[0].set_xlabel("True-positive rate")
    percent_axis(axes[0])
    fpr_colors = [BLUE if name == "internal_xz16_only" else ORANGE if name == "no_whitening_xz16" else GREY for name in order]
    axes[1].barh(y, indexed["pooled_fpr"], color=fpr_colors, edgecolor="white")
    axes[1].axvline(0.05, color=CHARCOAL, linestyle="--", linewidth=1.1)
    axes[1].set_xlabel("False-positive rate")
    percent_axis(axes[1], upper=0.08)
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    fig.suptitle("Local CF-CMD depends on drift; higher dimension does not rescue transfer", x=0.03, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.03, 0.025, "Fixed local attack, eight eligible seeds. Blue is frozen XZ16; orange is the covariance-free revision. Candidate-as-reference is a labeled historical variant.", fontsize=8)
    save(fig, out, "fig03_mechanism_dimension_ablations")


def boundary_heatmaps(results: Path, out: Path) -> None:
    df = pd.read_csv(results / "bloodmnist_boundary_study_v1" / "pooled_boundary_grid.csv")
    df = df[df["shots"].astype(str).eq("exact")].copy()
    methods = ["covariance_free_cf_cmd_u", "original_whitened_cf_cmd"]
    titles = ["CF-CMD-U (no whitening)", "Original CF-CMD (whitened)"]
    fractions = [0.01, 0.03, 0.05, 0.10, 0.20, 0.30]
    contaminations = [0.00, 0.01, 0.03, 0.05, 0.10]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), constrained_layout=True)
    image = None
    for ax, method, title in zip(axes, methods, titles):
        subset = df[df["method"].eq(method)]
        pivot = subset.pivot(index="trusted_contamination", columns="poison_fraction", values="pooled_tpr").loc[contaminations, fractions]
        fpr = subset.pivot(index="trusted_contamination", columns="poison_fraction", values="pooled_fpr").loc[contaminations, fractions]
        image = ax.imshow(pivot.values, vmin=0, vmax=1, cmap=mpl.colors.LinearSegmentedColormap.from_list("blue_gold", ["#F4F6F7", GOLD, BLUE]), aspect="auto")
        for i in range(len(contaminations)):
            for j in range(len(fractions)):
                tpr = float(pivot.iloc[i, j])
                passed = tpr >= 0.90 and float(fpr.iloc[i, j]) <= 0.05
                ax.text(j, i, f"{tpr:.0%}" + (" ✓" if passed else ""), ha="center", va="center", fontsize=7.5, color="white" if tpr > 0.68 else CHARCOAL, fontweight="bold" if passed else "normal")
        ax.set_xticks(np.arange(len(fractions)), [f"{v:.0%}" for v in fractions])
        ax.set_yticks(np.arange(len(contaminations)), [f"{v:.0%}" for v in contaminations])
        ax.set_xlabel("Triggered-candidate prevalence")
        ax.set_ylabel("Trusted-reference contamination")
        ax.set_title(title, loc="left")
        ax.grid(False)
    assert image is not None
    cbar = fig.colorbar(image, ax=axes, shrink=0.78, pad=0.02)
    cbar.set_label("Pooled true-positive rate")
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    fig.suptitle("Detection collapses at low prevalence and under trusted-reference contamination", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, -0.01, "Exact-state stress test under fixed victims; ✓ requires TPR ≥90% and FPR ≤5%. Prevalence is the evaluation candidate pool, not the training poison ratio.", fontsize=8)
    save(fig, out, "fig04_boundary_heatmaps")


def post_sanitization(results: Path, out: Path) -> None:
    df = pd.read_csv(results / "bloodmnist_post_sanitization_v1" / "summary.csv").set_index("condition")
    conditions = ["detected_removal", "random_matched_removal", "oracle_poison_removal"]
    labels = ["Detected removal", "Random matched", "Oracle poison"]
    x = np.arange(len(conditions))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.3), gridspec_kw={"wspace": 0.18})
    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.08, right=0.98)
    series = [
        ("QML balanced CA", "mean_qml_balanced_ca", BLUE),
        ("Classical balanced CA", "mean_classical_balanced_ca", OLIVE),
    ]
    for idx, (label, col, color) in enumerate(series):
        axes[0].bar(x + (idx - 0.5) * width, df.loc[conditions, col], width=width, color=color, label=label)
    axes[0].axhline(0.90, color=CHARCOAL, linestyle="--", linewidth=1.1)
    axes[0].set_ylim(0.85, 0.98)
    axes[0].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_title("Clean balanced accuracy remains above 90%", loc="left")
    axes[0].set_xticks(x, labels)
    axes[0].legend(loc="upper left")
    series = [
        ("QML ASR", "mean_qml_asr", ORANGE),
        ("Classical ASR", "mean_classical_asr", PINK),
    ]
    for idx, (label, col, color) in enumerate(series):
        axes[1].bar(x + (idx - 0.5) * width, df.loc[conditions, col], width=width, color=color, label=label)
    axes[1].axhline(0.80, color=CHARCOAL, linestyle="--", linewidth=1.1)
    axes[1].set_ylim(0, 1.0)
    axes[1].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    axes[1].set_title("Detected removal reduces attack success", loc="left")
    axes[1].set_xticks(x, labels)
    axes[1].legend(loc="upper left")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    fig.suptitle("Screening has practical sanitization value despite robustness failures", x=0.03, y=0.98, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.03, 0.035, "Mean over seven attack-eligible seeds; models are retrained from scratch after removal. Random matched removal preserves substantially higher ASR.", fontsize=8)
    save(fig, out, "fig05_post_sanitization")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.paper_root.resolve()
    results = root / "results"
    out = results / "top_journal_program_v1" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    configure_style()
    audit_pipeline(out)
    strong_baselines(results, out)
    adaptive_forest(results, out)
    ablations(results, out)
    boundary_heatmaps(results, out)
    post_sanitization(results, out)
    print(f"Wrote 12 figure files to {out}")


if __name__ == "__main__":
    main()
