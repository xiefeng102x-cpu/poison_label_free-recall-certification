#!/usr/bin/env python3
"""Summarize adaptive training of the replacement BloodMNIST encoder."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "paper14/results/bloodmnist_new_encoder_adaptive_v1"
CLEAN_SUMMARY = (
    PROJECT_ROOT
    / "paper14/results/bloodmnist_clean_encoder_v1/clean_encoder_summary.json"
)
PREVIOUS_RECEIPT = (
    PROJECT_ROOT
    / "paper14/results/bloodmnist_ca_improvement_v1/improvement_receipt.json"
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_directory(path: Path) -> list[dict]:
    return [load(Path(filename)) for filename in sorted(glob.glob(str(path / "*.json")))]


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": [float(value) for value in array],
        "mean": float(np.mean(array)),
        "population_std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    development = load_directory(RESULT_ROOT / "development")
    delayed = load_directory(RESULT_ROOT / "delayed_test")
    confirmation = sorted(
        load_directory(RESULT_ROOT / "confirmation"), key=lambda row: int(row["seed"])
    )
    if len(delayed) != 1:
        raise RuntimeError(f"Expected one delayed-test artifact, found {len(delayed)}")
    clean_summary = load(CLEAN_SUMMARY)
    previous = load(PREVIOUS_RECEIPT)
    clean_by_seed = {
        int(row["seed"]): float(row["new_clean_accuracy"])
        for row in clean_summary["confirmation"]["per_seed"]
    }

    development_rows = []
    for row in development:
        point = row["selected"]
        development_rows.append(
            {
                "learning_rate": float(row["configuration"]["learning_rate"]),
                "selected_epoch": int(point["epoch"]),
                "passes_validation_gates": bool(point["passes_validation_gates"]),
                "validation_ca": float(point["validation"]["clean"]["accuracy"]),
                "validation_asr": float(point["validation"]["attack"]["asr"]),
                "validation_separability": float(
                    point["validation"]["detector"][
                        "separability_xyz_independent_reference"
                    ]
                ),
                "test_evaluated": bool(point["test_evaluated"]),
            }
        )
    selected_development = max(
        development_rows,
        key=lambda row: (
            row["passes_validation_gates"],
            row["validation_ca"],
            -row["validation_separability"],
            row["validation_asr"],
        ),
    )

    per_seed = []
    for row in confirmation:
        seed = int(row["seed"])
        point = row["selected"]
        test = point["test"]
        adaptive_ca = float(test["clean"]["accuracy"])
        clean_ca = clean_by_seed[seed]
        per_seed.append(
            {
                "seed": seed,
                "selected_epoch": int(point["epoch"]),
                "clean_checkpoint_ca": clean_ca,
                "adaptive_ca": adaptive_ca,
                "paired_ca_change": adaptive_ca - clean_ca,
                "heldout_asr": float(test["attack"]["asr"]),
                "heldout_triggered_count": int(test["attack"]["triggered_count"]),
                "independent_xyz_auc": float(
                    test["detector"]["auc_xyz_independent_reference"]
                ),
                "independent_xyz_separability": float(
                    test["detector"][
                        "separability_xyz_independent_reference"
                    ]
                ),
                "archived_poison_asr": float(point["archived_poison"]["asr"]),
                "projector_l2_drift": float(row["projector_l2_drift"]),
                "passes_validation_gates": bool(point["passes_validation_gates"]),
            }
        )

    def values(name: str) -> list[float]:
        return [float(row[name]) for row in per_seed]

    prior_confirmation = next(
        row for row in previous["groups"] if row["group"] == "confirmation"
    )
    configuration = confirmation[0]["configuration"]
    max_reconstruction = max(
        max(
            float(row["data_receipt"]["reconstruction_error"]["target_max_abs_error"]),
            float(
                row["data_receipt"]["reconstruction_error"][
                    "non_target_max_abs_error"
                ]
            ),
        )
        for row in confirmation
    )
    max_trigger_reconstruction = max(
        float(row["data_receipt"]["triggered_reconstruction_max_abs_error"])
        for row in confirmation
    )
    summary = {
        "status": "PASS",
        "protocol": {
            "development_seed": 42,
            "confirmation_seeds": [43, 44, 45, 46],
            "adapted_component": "PCA256-to-8 projector only",
            "frozen_component": "8-to-32-to-2 classifier",
            "test_trigger_scope": "all held-out non-target samples plus the archived seed-specific trigger",
            "detector_reference_scope": "current adapted train-target XYZ features",
            "selection_scope": "training and validation only",
        },
        "development_sweep": sorted(
            development_rows, key=lambda row: row["learning_rate"]
        ),
        "selected_configuration": {
            **configuration,
            "development_choice": selected_development,
        },
        "development_delayed_test": delayed[0]["selected"]["test"],
        "confirmation": {
            "per_seed": per_seed,
            "adaptive_ca": distribution(values("adaptive_ca")),
            "clean_checkpoint_ca": distribution(values("clean_checkpoint_ca")),
            "paired_ca_change": distribution(values("paired_ca_change")),
            "heldout_asr": distribution(values("heldout_asr")),
            "independent_xyz_auc": distribution(values("independent_xyz_auc")),
            "independent_xyz_separability": distribution(
                values("independent_xyz_separability")
            ),
            "archived_poison_asr": distribution(values("archived_poison_asr")),
            "projector_l2_drift": distribution(values("projector_l2_drift")),
            "all_test_ca_at_least_90_percent": all(
                row["adaptive_ca"] >= 0.90 for row in per_seed
            ),
            "all_test_ca_drops_at_most_5_points": all(
                row["paired_ca_change"] >= -0.05 for row in per_seed
            ),
            "all_heldout_asr_at_least_80_percent": all(
                row["heldout_asr"] >= 0.80 for row in per_seed
            ),
            "all_independent_separability_at_most_065": all(
                row["independent_xyz_separability"] <= 0.65 for row in per_seed
            ),
        },
        "previous_old_encoder_context": {
            "adaptive_ca_mean": prior_confirmation["new_adaptive_ca_mean"],
            "adaptive_ca_population_std": prior_confirmation[
                "new_adaptive_ca_population_sd"
            ],
            "archived_poison_asr_mean": prior_confirmation["new_asr_mean"],
            "recalibrated_xyz_separability_mean": prior_confirmation[
                "new_separability_xyz_mean"
            ],
            "comparison_warning": "The prior ASR used the archived 60-sample trigger pool and its detector used a different calibration protocol; only CA is definition-matched to the new held-out clean test result.",
        },
        "integrity": {
            "development_test_access_flags": [
                row["test_evaluated"] for row in development_rows
            ],
            "delayed_test_count": len(delayed),
            "confirmation_artifact_count": len(confirmation),
            "max_clean_reconstruction_error": max_reconstruction,
            "max_triggered_reconstruction_error": max_trigger_reconstruction,
            "reconstruction_threshold": 5e-3,
            "confirmation_validation_gates_pass": [
                row["passes_validation_gates"] for row in per_seed
            ],
        },
    }
    if not all(
        [
            summary["confirmation"]["all_test_ca_at_least_90_percent"],
            summary["confirmation"]["all_test_ca_drops_at_most_5_points"],
            summary["confirmation"]["all_heldout_asr_at_least_80_percent"],
            summary["confirmation"]["all_independent_separability_at_most_065"],
        ]
    ):
        summary["status"] = "FAIL"
    output = RESULT_ROOT / "adaptive_summary.json"
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
