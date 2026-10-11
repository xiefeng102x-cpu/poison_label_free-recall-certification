#!/usr/bin/env python3
"""Build and verify the manuscript's non-overlapping evaluation-scale receipt."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/top_journal_program_v1/experimental_coverage_receipt_v1.json"


BLOCKS = (
    {
        "id": "mnist_trigger_direction_archive",
        "dataset": "MNIST",
        "role": "retrospective fixed-seed archive",
        "architectures": ["archived MNIST victims"],
        "attacks": ["nine source-target trigger directions"],
        "condition_path": "results/mnist_group_coherent_drift_10seeds_v1/deployment_scope_summary.csv",
        "condition_mode": "mnist_all_archived_configs",
        "score_path": "results/mnist_group_coherent_drift_10seeds_v1/per_sample_scores.csv",
    },
    {
        "id": "blood_covariance_free_confirmation",
        "dataset": "BloodMNIST",
        "role": "fresh confirmation",
        "architectures": ["classical MLP", "local-angle QML"],
        "attacks": ["additive", "representation matching"],
        "condition_path": "results/bloodmnist_covariance_free_confirmation_v1/victim_metrics.csv",
        "condition_mode": "csv_rows",
        "score_path": "results/bloodmnist_covariance_free_confirmation_v1/per_sample_scores.csv",
    },
    {
        "id": "blood_cross_attack_confirmation",
        "dataset": "BloodMNIST",
        "role": "fresh confirmation",
        "architectures": ["classical MLP", "local-angle QML"],
        "attacks": ["Blended", "SIG"],
        "condition_path": "results/bloodmnist_cross_attack_confirmation_v1/victim_metrics.csv",
        "condition_mode": "csv_rows",
        "score_path": "results/bloodmnist_cross_attack_confirmation_v1/per_sample_scores.csv",
    },
    {
        "id": "blood_cnn_badnets_confirmation",
        "dataset": "BloodMNIST",
        "role": "fresh confirmation",
        "architectures": ["image-native CNN"],
        "attacks": ["BadNets"],
        "condition_path": "results/bloodmnist_cnn_badnets_confirmation_v1/victim_metrics.csv",
        "condition_mode": "csv_rows",
        "score_path": "results/bloodmnist_cnn_badnets_confirmation_v1/per_sample_scores.csv",
    },
    {
        "id": "blood_input_aware_confirmation",
        "dataset": "BloodMNIST",
        "role": "fresh confirmation",
        "architectures": ["image-native CNN"],
        "attacks": ["Input-Aware-style"],
        "condition_path": "results/bloodmnist_input_aware_confirmation_v1/victim_metrics.csv",
        "condition_mode": "csv_rows",
        "score_path": "results/bloodmnist_input_aware_confirmation_v1/per_sample_scores.csv",
    },
    {
        "id": "blood_whitebox_confirmation",
        "dataset": "BloodMNIST",
        "role": "fresh white-box confirmation",
        "architectures": ["local-angle QML"],
        "attacks": [
            "representation matching",
            "multi-direction coherence breaking",
            "joint input/internal evasion",
        ],
        "condition_path": "results/bloodmnist_whitebox_adaptive_confirmation_v1/victim_metrics.csv",
        "condition_mode": "csv_rows",
        "score_path": "results/bloodmnist_whitebox_adaptive_confirmation_v1/per_sample_scores.csv",
    },
    {
        "id": "blood_residual_reupload_confirmation",
        "dataset": "BloodMNIST",
        "role": "fresh architecture confirmation",
        "architectures": ["residual reupload QML"],
        "attacks": ["representation matching"],
        "condition_path": "results/bloodmnist_residual_reupload_confirmation_v1/victim_metrics.csv",
        "condition_mode": "csv_rows",
        "score_path": "results/bloodmnist_residual_reupload_confirmation_v1/per_sample_scores.csv",
    },
    {
        "id": "blood_multiclass_confirmation",
        "dataset": "BloodMNIST",
        "role": "fresh multiclass confirmation",
        "architectures": ["image-native CNN"],
        "attacks": ["BadNets"],
        "condition_path": "results/bloodmnist_multiclass_backdoorbench_confirmation_v1/victim_metrics.csv",
        "condition_mode": "csv_rows",
        "score_path": "results/bloodmnist_multiclass_backdoorbench_confirmation_v1/per_sample_flags.csv",
    },
    {
        "id": "blood_final_observable_extension",
        "dataset": "BloodMNIST",
        "role": "preregistered confirmation extension",
        "architectures": ["local-angle QML"],
        "attacks": ["final-observable adaptive"],
        "condition_path": "results/p6d_final_observable_adaptive_v4/confirmation/victim_metrics.csv",
        "condition_mode": "csv_rows",
        "score_path": "results/p6d_final_observable_adaptive_v4/confirmation/per_sample_scores.csv",
    },
    {
        "id": "pathmnist_official_weight_confirmation",
        "dataset": "PathMNIST",
        "role": "fresh external-domain confirmation",
        "architectures": ["official-weight ResNet-18"],
        "attacks": ["BadNets"],
        "condition_path": "results/pathmnist_official_weight_detector_confirmation_v1/detector_metrics.csv",
        "condition_mode": "unique_seed",
        "score_path": "results/pathmnist_official_weight_detector_confirmation_v1/scores.csv",
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_receipt() -> dict[str, Any]:
    records = []
    for spec in BLOCKS:
        condition_path = ROOT / spec["condition_path"]
        score_path = ROOT / spec["score_path"]
        condition_rows = read_csv(condition_path)
        score_rows = read_csv(score_path)
        if spec["condition_mode"] == "csv_rows":
            condition_count = len(condition_rows)
        elif spec["condition_mode"] == "unique_seed":
            condition_count = len({row["seed"] for row in condition_rows})
        elif spec["condition_mode"] == "mnist_all_archived_configs":
            selected = [row for row in condition_rows if row["scope"] == "all_archived_configs"]
            if len(selected) != 1:
                raise AssertionError("MNIST all-archive summary row is not unique")
            condition_count = int(selected[0]["n_configs"])
        else:
            raise AssertionError(f"unknown condition mode: {spec['condition_mode']}")
        records.append({
            **spec,
            "condition_seed_evaluations": condition_count,
            "score_or_flag_records": len(score_rows),
            "condition_file_sha256": sha256_file(condition_path),
            "score_file_sha256": sha256_file(score_path),
        })

    totals = {
        "datasets": 3,
        "victim_architecture_families": 5,
        "named_attack_constructions": 11,
        "confirmation_grade_attack_constructions": 9,
        "pilot_only_attack_constructions": 2,
        "condition_seed_evaluations": sum(row["condition_seed_evaluations"] for row in records),
        "score_or_flag_records": sum(row["score_or_flag_records"] for row in records),
    }
    expected = {
        "condition_seed_evaluations": 210,
        "score_or_flag_records": 3_333_660,
    }
    for key, value in expected.items():
        if totals[key] != value:
            raise AssertionError(f"coverage total drift for {key}: {totals[key]} != {value}")
    return {
        "receipt_id": "experimental_coverage_receipt_v1",
        "scope": "Non-overlapping manuscript evidence blocks; development-only duplicates and sensitivity replays are excluded from totals.",
        "inferential_unit": "The attack-trained victim/condition indexed by seed; rows within a victim are correlated decisions, not independent replicates.",
        "counting_note": "Condition-seed evaluations may reuse a seed under distinct attacks or architectures and must not be described as 210 independent trained models.",
        "attack_inventory": {
            "confirmation_grade": [
                "additive",
                "BadNets",
                "Blended",
                "SIG",
                "Input-Aware-style",
                "representation matching",
                "multi-direction coherence breaking",
                "joint input/internal evasion",
                "final-observable adaptive",
            ],
            "pilot_only": ["WaNet", "label-consistent"],
        },
        "architecture_inventory": [
            "classical MLP",
            "image-native CNN",
            "local-angle QML",
            "residual reupload QML",
            "official-weight ResNet-18",
        ],
        "totals": totals,
        "blocks": records,
    }


def stable_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    receipt = build_receipt()
    if args.verify:
        stored = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if stored != receipt:
            raise AssertionError("stored coverage receipt does not match source files")
        print(json.dumps({"status": "valid", "totals": receipt["totals"]}, indent=2, sort_keys=True))
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(stable_bytes(receipt))
    print(json.dumps({"status": "written", "output": str(OUTPUT), "totals": receipt["totals"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
