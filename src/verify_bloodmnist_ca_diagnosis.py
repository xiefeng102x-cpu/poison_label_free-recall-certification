#!/usr/bin/env python3
"""Verify completeness and headline numbers of the BloodMNIST CA diagnosis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPO_ROOT / "Blackdoor_1/paper14/results/bloodmnist_ca_diagnosis_v1"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, tolerance: float = 5e-7) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"Expected {expected}, got {actual}")


def selected_mean(rows: list[dict], field: str, **filters) -> float:
    values = [
        float(row[field])
        for row in rows
        if all(str(row[key]) == str(value) for key, value in filters.items())
    ]
    if len(values) != 5:
        raise AssertionError(f"Expected five rows for {filters}, got {len(values)}")
    return float(np.mean(values))


def main() -> None:
    checkpoint = read_csv(ROOT / "heldout_checkpoint_audit/checkpoint_ca.csv")
    baselines = read_csv(ROOT / "heldout_checkpoint_audit/heldout_feature_baselines.csv")
    readouts = read_csv(ROOT / "heldout_checkpoint_audit/qnn_feature_readouts.csv")
    adaptive = read_csv(ROOT / "heldout_checkpoint_audit/adaptive_qnn_heldout_ca.csv")
    classical = read_csv(
        ROOT / "matched_classical_heldout_5seeds/matched_classical_per_run.csv"
    )
    reconstruction = json.loads(
        (ROOT / "heldout_checkpoint_audit/reconstruction_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    adaptive_receipt = json.loads(
        (ROOT / "heldout_checkpoint_audit/adaptive_qnn_heldout_ca_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    gradient = json.loads((ROOT / "diagnostic_summary.json").read_text(encoding="utf-8"))[
        "gradient_audit"
    ]

    if len(checkpoint) != 60:
        raise AssertionError(f"Expected 60 checkpoint rows, got {len(checkpoint)}")
    if len(baselines) != 30:
        raise AssertionError(f"Expected 30 baseline rows, got {len(baselines)}")
    if len(readouts) != 100:
        raise AssertionError(f"Expected 100 readout rows, got {len(readouts)}")
    if len(adaptive) != 40:
        raise AssertionError(f"Expected 40 adaptive held-out rows, got {len(adaptive)}")

    max_reconstruction_error = max(
        max(
            receipt["saved_training_tensor_match"]["target_max_abs_error"],
            receipt["saved_training_tensor_match"]["non_target_max_abs_error"],
        )
        for receipt in reconstruction["receipts"]
    )
    if max_reconstruction_error > 5e-3:
        raise AssertionError(f"Reconstruction error too large: {max_reconstruction_error}")
    if adaptive_receipt["max_abs_difference_from_archived_clean_pool_ca"] > 1e-4:
        raise AssertionError("Adaptive replay does not reproduce archived CA")

    close(
        selected_mean(checkpoint, "accuracy", epoch="200", split="test"),
        0.7876010781671159,
    )
    close(
        selected_mean(baselines, "accuracy", model="l2norm256_rbf_svm"),
        0.9859838274932614,
    )
    close(
        selected_mean(readouts, "accuracy", epoch="200", readout="refit_logreg_z8"),
        0.7935309973045823,
    )
    close(
        selected_mean(
            adaptive, "accuracy", lambda_r="1.0", split="untouched_test"
        ),
        0.6501347708894879,
    )
    close(
        selected_mean(classical, "final_heldout_ca", lambda_r="1.0"),
        0.910512129380054,
    )

    lambda_one = gradient["summaries"]["lambda_1"]
    historical = lambda_one["historical_loss_shift_vs_finite_difference"]
    corrected = lambda_one["chain_rule_parameter_shift_vs_finite_difference"]
    if historical["relative_l2_error"] < 0.8:
        raise AssertionError("Historical gradient defect is not reproduced")
    if corrected["relative_l2_error"] > 0.002:
        raise AssertionError("Corrected gradient does not match finite difference")

    corrected_artifacts = [
        ROOT
        / "corrected_qnn_lambda0_seed42/corrected_seed=42_lambda=0.0.json",
        ROOT
        / "corrected_qnn_lambda1_seed42/corrected_seed=42_lambda=1.0.json",
        ROOT
        / "corrected_qnn_lambda1_lr0.001_seed42/corrected_seed=42_lambda=1.0.json",
    ]
    simulator_errors = []
    for path in corrected_artifacts:
        payload = json.loads(path.read_text(encoding="utf-8"))
        simulator_errors.append(float(payload["simulator_max_abs_z_error"]))
    if max(simulator_errors) > 1e-7:
        raise AssertionError("Differentiable simulator check failed")

    report = ROOT / "CA_ROOT_CAUSE_REPORT.md"
    if not report.exists() or report.stat().st_size < 5000:
        raise AssertionError("Root-cause report is missing or incomplete")

    receipt = {
        "status": "PASS",
        "checkpoint_rows": len(checkpoint),
        "baseline_rows": len(baselines),
        "readout_rows": len(readouts),
        "adaptive_rows": len(adaptive),
        "max_reconstruction_error": max_reconstruction_error,
        "max_adaptive_archive_ca_replay_error": adaptive_receipt[
            "max_abs_difference_from_archived_clean_pool_ca"
        ],
        "max_differentiable_simulator_z_error": max(simulator_errors),
    }
    (ROOT / "verification_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
