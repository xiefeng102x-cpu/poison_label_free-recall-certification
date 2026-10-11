#!/usr/bin/env python3
"""Verification gate for the BloodMNIST CA-improvement experiment."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPO_ROOT / "Blackdoor_1/paper14/results/bloodmnist_ca_improvement_v1"


def close(actual: float, expected: float, tolerance: float = 5e-7) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"Expected {expected}, got {actual}")


def main() -> None:
    receipt = json.loads((ROOT / "improvement_receipt.json").read_text(encoding="utf-8"))
    groups = {row["group"]: row for row in receipt["groups"]}
    confirmation = groups["confirmation"]
    descriptive = groups["all_descriptive"]
    if confirmation["seeds"] != [43, 44, 45, 46]:
        raise AssertionError("Confirmation seed boundary changed")
    close(confirmation["new_adaptive_ca_mean"], 0.7796495956873315)
    close(confirmation["paired_ca_gain_mean"], 0.1394878706199461)
    close(confirmation["new_asr_mean"], 0.9291666666666667)
    close(confirmation["new_auc_xyz_mean"], 0.6054166666666667)
    close(descriptive["new_adaptive_ca_mean"], 0.7741239892183288)

    max_simulator_error = 0.0
    max_reconstruction_error = 0.0
    for seed in range(42, 47):
        artifact = ROOT / "phase2_adaptive" / (
            f"seed={seed}_a=10.0_s=0.0_d=20.0_g=0.001_lr=0.01.json"
        )
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if len(payload["history"]) != 6 or payload["history"][-1]["epoch"] != 5:
            raise AssertionError(f"Incomplete trajectory for seed {seed}")
        max_simulator_error = max(
            max_simulator_error, float(payload["simulator_max_abs_z_error"])
        )
        errors = payload["reconstruction_error"]
        max_reconstruction_error = max(
            max_reconstruction_error,
            float(errors["target_max_abs_error"]),
            float(errors["non_target_max_abs_error"]),
        )
    if max_simulator_error > 1e-7:
        raise AssertionError(f"Simulator error too large: {max_simulator_error}")
    if max_reconstruction_error > 5e-3:
        raise AssertionError(f"Reconstruction error too large: {max_reconstruction_error}")

    phase1 = receipt["phase1_best_frozen_observable_readout"]
    if phase1["feature_set"] != "XYZ" or phase1["family"] != "rbf_svm":
        raise AssertionError("Unexpected Phase-1 winner")
    close(float(phase1["test_accuracy_mean"]), 0.7951482479784366)

    report = ROOT / "EXPERIMENT_REPORT.md"
    if not report.exists() or report.stat().st_size < 4000:
        raise AssertionError("Experiment report missing or incomplete")

    verification = {
        "status": "PASS",
        "confirmation_seeds": confirmation["seeds"],
        "confirmation_ca_mean": confirmation["new_adaptive_ca_mean"],
        "confirmation_paired_ca_gain_mean": confirmation["paired_ca_gain_mean"],
        "confirmation_asr_mean": confirmation["new_asr_mean"],
        "confirmation_auc_xyz_mean": confirmation["new_auc_xyz_mean"],
        "max_simulator_z_error": max_simulator_error,
        "max_reconstruction_error": max_reconstruction_error,
    }
    (ROOT / "verification_receipt.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
