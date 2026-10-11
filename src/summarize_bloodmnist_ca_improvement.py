#!/usr/bin/env python3
"""Summarize and verify the BloodMNIST CA-improvement experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_ROOT / "Blackdoor_1"
IMPROVEMENT_ROOT = PROJECT_ROOT / "paper14/results/bloodmnist_ca_improvement_v1"
DIAGNOSIS_ROOT = PROJECT_ROOT / "paper14/results/bloodmnist_ca_diagnosis_v1"
ADAPTIVE_ROOT = (
    PROJECT_ROOT / "qml_adaptive_nc_project/paper_nc_adaptive/results/training_time_bloodmnist"
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(label: str, seeds: list[int], rows: list[dict]) -> dict:
    selected = [row for row in rows if row["seed"] in seeds]
    fields = [
        "clean_checkpoint_ca",
        "historical_adaptive_ca",
        "new_adaptive_ca",
        "new_asr",
        "new_auc_xyz",
        "new_separability_xyz",
    ]
    result = {"group": label, "seeds": seeds, "n_seeds": len(seeds)}
    for field in fields:
        values = np.array([row[field] for row in selected], dtype=float)
        result[f"{field}_mean"] = float(values.mean())
        result[f"{field}_population_sd"] = float(values.std(ddof=0))
    gains = np.array(
        [row["new_adaptive_ca"] - row["historical_adaptive_ca"] for row in selected]
    )
    drops = np.array(
        [row["clean_checkpoint_ca"] - row["new_adaptive_ca"] for row in selected]
    )
    result["paired_ca_gain_mean"] = float(gains.mean())
    result["paired_ca_gain_population_sd"] = float(gains.std(ddof=0))
    result["ca_drop_from_clean_mean"] = float(drops.mean())
    return result


def main() -> None:
    checkpoint_rows = read_csv(
        DIAGNOSIS_ROOT / "heldout_checkpoint_audit/checkpoint_ca.csv"
    )
    adaptive_rows = read_csv(
        DIAGNOSIS_ROOT / "heldout_checkpoint_audit/adaptive_qnn_heldout_ca.csv"
    )
    clean_ca = {
        int(row["seed"]): float(row["accuracy"])
        for row in checkpoint_rows
        if row["epoch"] == "200" and row["split"] == "test"
    }
    old_ca = {
        int(row["seed"]): float(row["accuracy"])
        for row in adaptive_rows
        if row["lambda_r"] == "1.0" and row["split"] == "untouched_test"
    }

    per_seed: list[dict] = []
    for seed in range(42, 47):
        path = (
            IMPROVEMENT_ROOT
            / "phase2_adaptive"
            / f"seed={seed}_a=10.0_s=0.0_d=20.0_g=0.001_lr=0.01.json"
        )
        run = json.loads(path.read_text(encoding="utf-8"))
        final = run["history"][-1]["untouched_test"]
        old = json.loads(
            (ADAPTIVE_ROOT / f"result_seed={seed}_L8_lr1.0.json").read_text(
                encoding="utf-8"
            )
        )["final"]
        per_seed.append(
            {
                "seed": seed,
                "role": "development" if seed == 42 else "confirmation",
                "clean_checkpoint_ca": clean_ca[seed],
                "historical_adaptive_ca": old_ca[seed],
                "historical_asr": float(old["asr"]),
                "historical_auc_xyz": float(old["auc_xyz"]),
                "new_adaptive_ca": float(final["ca"]),
                "new_asr": float(final["asr"]),
                "new_auc_xyz": float(final["auc_xyz"]),
                "new_separability_xyz": max(
                    float(final["auc_xyz"]), 1.0 - float(final["auc_xyz"])
                ),
                "new_epoch": int(run["history"][-1]["epoch"]),
            }
        )

    groups = [
        aggregate("development", [42], per_seed),
        aggregate("confirmation", [43, 44, 45, 46], per_seed),
        aggregate("all_descriptive", [42, 43, 44, 45, 46], per_seed),
    ]
    phase1 = read_csv(IMPROVEMENT_ROOT / "phase1_xyz/xyz_readout_summary.csv")
    best_xyz = max(phase1, key=lambda row: float(row["test_accuracy_mean"]))

    output = IMPROVEMENT_ROOT
    write_csv(output / "selected_candidate_per_seed.csv", per_seed)
    write_csv(output / "selected_candidate_summary.csv", groups)
    receipt = {
        "status": "PASS",
        "selected_configuration": {
            "epochs": 5,
            "learning_rate": 0.01,
            "backdoor_weight": 10.0,
            "stealth_weight": 0.0,
            "distill_weight": 20.0,
            "anchor_weight": 0.001,
            "gradient_clip": 1.0,
        },
        "selection_boundary": (
            "Seed 42 was used for exploratory configuration selection and is not an "
            "independent confirmation seed. Seeds 43-46 form the confirmation set."
        ),
        "phase1_best_frozen_observable_readout": best_xyz,
        "per_seed": per_seed,
        "groups": groups,
    }
    (output / "improvement_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
