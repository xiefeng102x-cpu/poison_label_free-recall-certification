#!/usr/bin/env python3
"""Replay P6-A multiclass BackdoorBench development/confirmation artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import run_bloodmnist_multiclass_backdoorbench_pilot as runner


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEVELOPMENT = runner.DEFAULT_OUTPUT
DEFAULT_CONFIRMATION = runner.DEFAULT_CONFIRMATION_OUTPUT
DEFAULT_RECEIPT = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/multiclass_backdoorbench_program_receipt.json"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=tolerance))


def verify_stage(
    root: Path,
    expected_stage: str,
    expected_seeds: tuple[int, ...],
    dataset_root: Path,
) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment"] == "bloodmnist_multiclass_backdoorbench_pilot_v1"
    assert manifest["stage"] == expected_stage
    assert manifest["seeds"] == list(expected_seeds)
    assert manifest["protocol_sha256"] == runner.sha256_file(runner.PROTOCOL)
    assert manifest["runner_sha256"] == runner.sha256_file(Path(runner.__file__))
    assert manifest["upstream_commit"] == runner.official.UPSTREAM_COMMIT
    assert manifest["dataset_npz_sha256"] == runner.sha256_file(
        dataset_root / "bloodmnist.npz"
    )

    configs = read_csv(root / "per_config_metrics.csv")
    expected_config_count = len(expected_seeds) * len(runner.VIEWS) * len(runner.METHODS)
    assert len(configs) == expected_config_count
    config_lookup = {
        (int(row["seed"]), row["view"], row["method"]): row for row in configs
    }
    assert len(config_lookup) == expected_config_count

    candidate_count = 11959
    group_count = 0
    sample_count = 0
    mapping_digest_by_seed: dict[int, str] = {}
    current_key: tuple[int, str, str] | None = None
    current = None

    def finish_group() -> None:
        nonlocal group_count, current
        if current is None:
            return
        assert current["rows"] == candidate_count
        assert current["poison"] == runner.POISON_COUNT
        assert current["tp"] + current["fn"] == runner.POISON_COUNT
        assert current["fp"] + current["tn"] == candidate_count - runner.POISON_COUNT
        row = config_lookup[current["key"]]
        for field in ("tp", "fn", "fp", "tn", "flagged_count"):
            assert int(row[field]) == current[field]
        assert close(row["tpr"], current["tp"] / (current["tp"] + current["fn"]))
        assert close(row["fpr_all_clean"], current["fp"] / (current["fp"] + current["tn"]))
        assert close(
            row["target_clean_fpr"],
            current["target_fp"] / current["target_clean"],
        )
        digest = current["mapping_hash"].hexdigest()
        prior = mapping_digest_by_seed.setdefault(current["key"][0], digest)
        assert prior == digest
        group_count += 1

    with (root / "per_sample_flags.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["seed"]), row["view"], row["method"])
            if key != current_key:
                finish_group()
                current_key = key
                current = {
                    "key": key,
                    "rows": 0,
                    "poison": 0,
                    "tp": 0,
                    "fn": 0,
                    "fp": 0,
                    "tn": 0,
                    "flagged_count": 0,
                    "target_fp": 0,
                    "target_clean": 0,
                    "mapping_hash": hashlib.sha256(),
                }
            assert current is not None
            index = current["rows"]
            assert int(row["sample_index"]) == index
            assert int(row["original_index"]) == index
            original_label = int(row["original_label"])
            assigned_label = int(row["assigned_label"])
            poison = bool(int(row["is_poison"]))
            flagged = bool(int(row["flagged"]))
            if poison:
                assert original_label != runner.TARGET_CLASS
                assert assigned_label == runner.TARGET_CLASS
            else:
                assert assigned_label == original_label
            current["mapping_hash"].update(
                np.asarray(
                    [index, original_label, assigned_label, int(poison)], dtype=np.int64
                ).tobytes()
            )
            current["rows"] += 1
            current["poison"] += int(poison)
            current["flagged_count"] += int(flagged)
            current["tp"] += int(flagged and poison)
            current["fn"] += int((not flagged) and poison)
            current["fp"] += int(flagged and not poison)
            current["tn"] += int((not flagged) and (not poison))
            is_target_clean = (not poison) and original_label == runner.TARGET_CLASS
            current["target_clean"] += int(is_target_clean)
            current["target_fp"] += int(is_target_clean and flagged)
            sample_count += 1
    finish_group()
    assert group_count == expected_config_count
    assert sample_count == expected_config_count * candidate_count

    victims = read_csv(root / "victim_metrics.csv")
    assert len(victims) == len(expected_seeds)
    assert [int(row["seed"]) for row in victims] == list(expected_seeds)
    eligible = {int(row["seed"]) for row in victims if int(row["eligible"]) == 1}
    for row in victims:
        expected = float(row["balanced_clean_accuracy"]) >= 0.80 and float(
            row["attack_success_rate"]
        ) >= 0.80
        assert bool(int(row["eligible"])) is expected

    scans = read_csv(root / "scan_applicability.csv")
    assert len(scans) == len(expected_seeds) * len(runner.VIEWS)
    scan_valid = all(
        row["finite"] == "True"
        and row["nondegenerate"] == "True"
        and float(row["class_score_range"]) > 1e-8
        for row in scans
    )
    assert scan_valid

    receipts = json.loads((root / "seed_receipts.json").read_text(encoding="utf-8"))
    assert [int(row["seed"]) for row in receipts] == list(expected_seeds)
    for receipt in receipts:
        assert receipt["candidate_count"] == candidate_count
        assert receipt["pool"]["poison_count"] == runner.POISON_COUNT
        for artifact in receipt["checkpoints"].values():
            path = Path(artifact["path"])
            assert path.exists()
            assert runner.sha256_file(path) == artifact["file_sha256"]

    summary = []
    for view in runner.VIEWS:
        for method in runner.METHODS:
            selected = [
                row
                for row in configs
                if row["view"] == view
                and row["method"] == method
                and int(row["seed"]) in eligible
            ]
            if not selected:
                continue
            tp = sum(int(row["tp"]) for row in selected)
            fn = sum(int(row["fn"]) for row in selected)
            fp = sum(int(row["fp"]) for row in selected)
            tn = sum(int(row["tn"]) for row in selected)
            summary.append(
                {
                    "view": view,
                    "method": method,
                    "eligible_seed_count": len(selected),
                    "pooled_tpr": tp / (tp + fn),
                    "pooled_fpr_all_clean": fp / (fp + tn),
                    "mean_target_clean_fpr": float(
                        np.mean([float(row["target_clean_fpr"]) for row in selected])
                    ),
                    "minimum_seed_tpr": min(float(row["tpr"]) for row in selected),
                    "maximum_seed_tpr": max(float(row["tpr"]) for row in selected),
                }
            )

    return {
        "stage": expected_stage,
        "status": "PASS",
        "eligible_seeds": sorted(eligible),
        "config_rows": len(configs),
        "sample_rows": sample_count,
        "scan_applicability_pass": scan_valid,
        "gate": manifest["gate"],
        "eligible_summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data")
    )
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    development = verify_stage(
        args.development,
        "development",
        runner.DEVELOPMENT_SEEDS,
        args.dataset_root,
    )
    confirmation = verify_stage(
        args.confirmation,
        "confirmation",
        runner.CONFIRMATION_SEEDS,
        args.dataset_root,
    )
    receipt = {
        "status": "PASS",
        "protocol_sha256": runner.sha256_file(runner.PROTOCOL),
        "development": development,
        "confirmation": confirmation,
    }
    runner.write_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
