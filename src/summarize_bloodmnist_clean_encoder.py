#!/usr/bin/env python3
"""Summarize the replacement BloodMNIST clean-encoder experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "paper14/results/bloodmnist_clean_encoder_v1"
DIAGNOSIS_ROOT = PROJECT_ROOT / "paper14/results/bloodmnist_ca_diagnosis_v1"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    development = load_json(RESULT_ROOT / "development/run_summary.json")
    delayed_test = load_json(
        RESULT_ROOT / "development/seed_42_selected_delayed_test.json"
    )
    confirmation = load_json(RESULT_ROOT / "confirmation/run_summary.json")
    checkpoint_rows = load_csv(
        DIAGNOSIS_ROOT / "heldout_checkpoint_audit/checkpoint_ca.csv"
    )
    baseline_rows = load_csv(
        DIAGNOSIS_ROOT / "heldout_checkpoint_audit/heldout_feature_baselines.csv"
    )

    confirmation = sorted(confirmation, key=lambda row: int(row["seed"]))
    seeds = [int(row["seed"]) for row in confirmation]
    old_by_seed = {
        int(row["seed"]): float(row["accuracy"])
        for row in checkpoint_rows
        if int(row["seed"]) in seeds
        and int(row["epoch"]) == 200
        and row["split"] == "test"
    }
    pca_logreg_by_seed = {
        int(row["seed"]): float(row["accuracy"])
        for row in baseline_rows
        if int(row["seed"]) in seeds and row["model"] == "pca256_logreg"
    }
    new_by_seed = {int(row["seed"]): float(row["test"]["accuracy"]) for row in confirmation}

    new_values = [new_by_seed[seed] for seed in seeds]
    old_values = [old_by_seed[seed] for seed in seeds]
    pca_values = [pca_logreg_by_seed[seed] for seed in seeds]
    improvement = [new_by_seed[seed] - old_by_seed[seed] for seed in seeds]
    classical_gap = [new_by_seed[seed] - pca_logreg_by_seed[seed] for seed in seeds]
    validation_values = [float(row["val"]["accuracy"]) for row in confirmation]
    auc_values = [float(row["test"]["roc_auc"]) for row in confirmation]
    balanced_values = [float(row["test"]["balanced_accuracy"]) for row in confirmation]
    train_val_gaps = [float(row["train_val_accuracy_gap"]) for row in confirmation]
    reconstruction_errors = [
        max(
            float(row["data_receipt"]["reconstruction_error"]["target_max_abs_error"]),
            float(row["data_receipt"]["reconstruction_error"]["non_target_max_abs_error"]),
        )
        for row in confirmation
    ]

    development_candidates = [
        {
            "encoder": row["encoder"],
            "layers": int(row["layers"]),
            "best_epoch": int(row["best_epoch"]),
            "parameters": int(row["parameter_count"]),
            "train_accuracy": float(row["train"]["accuracy"]),
            "val_accuracy": float(row["val"]["accuracy"]),
            "val_auc": float(row["val"]["roc_auc"]),
            "test_evaluated_during_selection": bool(row["test_evaluated"]),
        }
        for row in development
    ]
    selected = max(
        development_candidates,
        key=lambda row: (row["val_accuracy"], row["val_auc"]),
    )
    summary = {
        "protocol": {
            "development_seed": 42,
            "confirmation_seeds": seeds,
            "selection_scope": "train and validation only",
            "test_scope": "one delayed development evaluation and one evaluation per frozen confirmation run",
            "target_clean_accuracy": 0.90,
        },
        "selected_encoder": {
            **selected,
            "description": "supervised PCA256-to-8 projection, bounded local RY angle encoding, eight Z expectations, 8-32-2 classifier",
            "projector_parameters": int(confirmation[0]["projector_parameter_count"]),
            "quantum_trainable_parameters": int(
                confirmation[0]["quantum_parameter_count"]
            ),
            "classifier_parameters": int(
                confirmation[0]["classifier_parameter_count"]
            ),
        },
        "development_candidates": development_candidates,
        "development_delayed_test": delayed_test["metrics"],
        "confirmation": {
            "per_seed": [
                {
                    "seed": seed,
                    "new_clean_accuracy": new_by_seed[seed],
                    "historical_epoch200_accuracy": old_by_seed[seed],
                    "paired_improvement": new_by_seed[seed] - old_by_seed[seed],
                    "pca256_logreg_accuracy": pca_logreg_by_seed[seed],
                    "gap_to_pca256_logreg": new_by_seed[seed]
                    - pca_logreg_by_seed[seed],
                }
                for seed in seeds
            ],
            "clean_accuracy": distribution(new_values),
            "balanced_accuracy": distribution(balanced_values),
            "roc_auc": distribution(auc_values),
            "validation_accuracy": distribution(validation_values),
            "historical_epoch200_accuracy": distribution(old_values),
            "paired_improvement": distribution(improvement),
            "pca256_logreg_accuracy": distribution(pca_values),
            "gap_to_pca256_logreg": distribution(classical_gap),
            "train_validation_accuracy_gap": distribution(train_val_gaps),
            "all_seeds_meet_90_percent": all(value >= 0.90 for value in new_values),
        },
        "integrity": {
            "max_reconstruction_error": max(reconstruction_errors),
            "reconstruction_threshold": 5e-3,
            "all_development_candidates_kept_test_unread": all(
                not row["test_evaluated_during_selection"]
                for row in development_candidates
            ),
            "delayed_checkpoint_matches_training_record": delayed_test[
                "training_record_match"
            ],
            "scaler_checkpoint_match": delayed_test["scaler_checkpoint_match"],
            "state_norm_max_abs_error": max(
                float(row["state_norm_max_abs_error"]) for row in confirmation
            ),
        },
    }
    output = RESULT_ROOT / "clean_encoder_summary.json"
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
