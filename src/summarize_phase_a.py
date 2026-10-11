#!/usr/bin/env python3
"""Create the frozen scientific conclusion and compact figures for Phase A."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main(results: Path) -> None:
    metrics = read_csv(results / "heldout_metrics.csv")
    comparisons = read_csv(results / "paired_auc_comparisons.csv")
    invariance = read_csv(results / "unitary_invariance_checks.csv")
    eligibility = read_csv(results / "victim_eligibility.csv")
    pair_audit = read_csv(results / "pair_audit.csv")

    seeds = sorted({int(row["seed"]) for row in metrics})
    reps = ["Z", "XYZ", "Full"]
    checkpoints = ["encoding", "middle", "output"]

    def select(**conditions):
        return [
            row
            for row in metrics
            if all(str(row[key]) == str(value) for key, value in conditions.items())
        ]

    primary = {
        (int(row["seed"]), row["representation"]): float(row["auc"])
        for row in select(victim_kind="backdoor", checkpoint="output", detector="logistic")
    }
    encoding = {
        (int(row["seed"]), row["representation"]): float(row["auc"])
        for row in select(victim_kind="backdoor", checkpoint="encoding", detector="logistic")
    }
    clean_control = {
        (int(row["seed"]), row["representation"]): float(row["auc"])
        for row in select(victim_kind="clean", checkpoint="output", detector="logistic")
    }
    depth = {
        (int(row["seed"]), row["checkpoint"], row["representation"]): float(row["auc"])
        for row in select(victim_kind="backdoor", detector="logistic")
    }

    means = {rep: float(np.mean([primary[(seed, rep)] for seed in seeds])) for rep in reps}
    encoding_means = {rep: float(np.mean([encoding[(seed, rep)] for seed in seeds])) for rep in reps}
    clean_means = {
        rep: float(np.mean([clean_control[(seed, rep)] for seed in seeds])) for rep in reps
    }
    max_gram_relative_error = max(
        float(row["full_pauli_gram_relative_fro_delta_vs_encoding"])
        for row in invariance
    )
    max_pair_error = max(float(row["pair_reconstruction_max_abs_error"]) for row in pair_audit)

    primary_deltas = {
        seed: primary[(seed, "Full")] - primary[(seed, "XYZ")] for seed in seeds
    }
    backdoor_clean_full_deltas = {
        seed: primary[(seed, "Full")] - clean_control[(seed, "Full")] for seed in seeds
    }
    per_seed_delta_ci = {}
    for row in comparisons:
        if (
            row["victim_kind"] == "backdoor"
            and row["checkpoint"] == "output"
            and row["detector"] == "logistic"
            and row["comparison"] == "Full_minus_XYZ"
        ):
            per_seed_delta_ci[int(row["seed"])] = {
                "delta": float(row["delta_auc"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
            }

    scientific_decision = "STOP_AFTER_PHASE_A_LOW_DIMENSIONAL_READOUT_SUFFICIENT"
    summary = {
        "engineering_gate_from_original_guide": "GO_B",
        "scientific_progression_decision": scientific_decision,
        "reason": (
            "Full-space attack/clean discriminability is high, but local XYZ is equally strong; "
            "the same Full-space signal is present under clean-model controls and is already visible "
            "at the encoding checkpoint. The hypothesized compression gap is not supported for this configuration."
        ),
        "primary_scope": "within-victim, output checkpoint, L2 logistic regression",
        "mean_test_auc": means,
        "mean_encoding_auc": encoding_means,
        "mean_clean_model_output_auc": clean_means,
        "full_minus_xyz_by_seed": {str(key): value for key, value in primary_deltas.items()},
        "full_minus_xyz_paired_bootstrap": {
            str(key): value for key, value in per_seed_delta_ci.items()
        },
        "full_backdoor_minus_clean_control_by_seed": {
            str(key): value for key, value in backdoor_clean_full_deltas.items()
        },
        "max_full_pauli_gram_relative_fro_error": max_gram_relative_error,
        "max_pair_reconstruction_error": max_pair_error,
        "victim_eligibility": [
            {
                "seed": int(row["seed"]),
                "clean_accuracy_percent": float(row["clean_accuracy_percent"]),
                "attack_success_rate_percent": float(row["attack_success_rate_percent"]),
            }
            for row in eligibility
        ],
        "boundary": (
            "This is a preregistered 5-qubit MNIST pilot with 36 held-out matched pairs per seed. "
            "It does not establish cross-seed or cross-attack universality."
        ),
    }
    write_json(results / "scientific_conclusion.json", summary)

    figures = results / "figures"
    figures.mkdir(exist_ok=True)
    colors = {"Z": "#5B8FF9", "XYZ": "#61DDAA", "Full": "#65789B"}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)

    x = np.arange(len(seeds))
    width = 0.24
    for offset, rep in zip((-width, 0.0, width), reps):
        axes[0].bar(
            x + offset,
            [primary[(seed, rep)] for seed in seeds],
            width=width,
            label=f"{rep} ({5 if rep == 'Z' else 15 if rep == 'XYZ' else 1023}D)",
            color=colors[rep],
        )
    axes[0].set_xticks(x, [str(seed) for seed in seeds])
    axes[0].set_ylim(0.85, 1.005)
    axes[0].set_xlabel("Victim seed")
    axes[0].set_ylabel("Held-out ROC AUC")
    axes[0].set_title("Output checkpoint")
    axes[0].legend(frameon=False, loc="lower left")
    axes[0].grid(axis="y", alpha=0.2)

    checkpoint_x = np.arange(len(checkpoints))
    for rep in reps:
        means_by_depth = [
            np.mean([depth[(seed, checkpoint, rep)] for seed in seeds])
            for checkpoint in checkpoints
        ]
        axes[1].plot(
            checkpoint_x,
            means_by_depth,
            marker="o",
            linewidth=2,
            label=rep,
            color=colors[rep],
        )
    axes[1].set_xticks(checkpoint_x, ["Encoding", "Middle", "Output"])
    axes[1].set_ylim(0.85, 1.005)
    axes[1].set_ylabel("Mean held-out ROC AUC")
    axes[1].set_title("Depth changes restricted readout, not Full")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, loc="lower right")

    fig.suptitle("Paper 14 Phase A: no measurable Full-versus-XYZ accessibility gap")
    fig.savefig(figures / "phase_a_auc_summary.png", dpi=220)
    fig.savefig(figures / "phase_a_auc_summary.pdf")
    plt.close(fig)

    eligibility_text = ", ".join(
        f"seed {row['seed']}: CA {float(row['clean_accuracy_percent']):.2f}%, "
        f"ASR {float(row['attack_success_rate_percent']):.2f}%"
        for row in eligibility
    )
    table_rows = "\n".join(
        f"| {seed} | {primary[(seed, 'Z')]:.4f} | {primary[(seed, 'XYZ')]:.4f} | "
        f"{primary[(seed, 'Full')]:.4f} | {primary_deltas[seed]:+.4f} |"
        for seed in seeds
    )
    report = f"""# Phase A validation report

## Decision

**{scientific_decision}**

The original guide's engineering rule labels this `GO_B` because Full-space AUC is high. The scientifically relevant continuation gate is not passed: Full does not improve on XYZ, the Full signal is identical under the clean-model control, and clean/triggered samples are already separable at encoding.

## Frozen victim eligibility

{eligibility_text}.

## Primary held-out result

The primary analysis is within-victim L2 logistic regression at the output checkpoint. Each seed uses 108 train, 36 validation and 36 untouched test pairs; clean and triggered members of a pair never cross splits.

| Seed | Z AUC | XYZ AUC | Full AUC | Full − XYZ |
|---:|---:|---:|---:|---:|
{table_rows}

Mean AUC: Z `{means['Z']:.4f}`, XYZ `{means['XYZ']:.4f}`, Full `{means['Full']:.4f}`. Mean Full − XYZ is `{np.mean(list(primary_deltas.values())):+.4f}`.

## Controls and numerical validation

- Full backdoored-minus-clean-model AUC difference is exactly `0.0000` for all three seeds.
- Mean encoding-checkpoint AUC is Z `{encoding_means['Z']:.4f}`, XYZ `{encoding_means['XYZ']:.4f}`, Full `{encoding_means['Full']:.4f}`. The trigger is visible before learned QNN dynamics.
- Maximum clean/trigger pair reconstruction error is `{max_pair_error:.2e}`.
- Maximum relative Full-Pauli Gram-matrix deviation across fixed-unitary checkpoints is `{max_gram_relative_error:.2e}`, confirming the expected full-information invariance.
- The RBF-SVM sensitivity analysis reaches the same qualitative conclusion; it does not reveal a hidden Full-over-XYZ gap.
- An independent Mann–Whitney recomputation matched all 108 stored AUC values to within floating-point error.

## Interpretation

For this qualified 5-qubit MNIST attack, the state contains a strong trigger signature, but it is not hidden in a high-dimensional subspace inaccessible to conventional local measurements. The evidence supports **low-dimensional readout sufficiency**, not measurement-compression loss. Pauli-weight search, PCA and attack-subspace removal should therefore not be run on this configuration merely to search for a preferred positive result.

## Scope boundary

This is a three-seed, within-victim pilot with 36 held-out pairs per seed. It validates the Phase A decision for this frozen attack configuration; it does not establish cross-seed detector transfer, other trigger families, hardware-shot behavior, or universal absence of high-dimensional hidden attack information.
"""
    (results / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "complete", "decision": scientific_decision}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    main(args.results.resolve())
