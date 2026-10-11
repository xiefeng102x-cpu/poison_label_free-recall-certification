#!/usr/bin/env python3
"""Freeze GTSRB confirmation settings after a qualified victim-only smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-protocol",
        type=Path,
        default=ROOT / "configs/tifs_native_gtsrb_development_v1.json",
    )
    parser.add_argument(
        "--development-output",
        type=Path,
        default=ROOT / "results/tifs_native_cifar_gtsrb_v1/gtsrb_development",
    )
    parser.add_argument(
        "--runner", type=Path, default=ROOT / "src/run_tifs_native_gtsrb.py"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "configs/tifs_native_gtsrb_confirmation_v1.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite frozen protocol: {args.output}")

    development = read_json(args.development_protocol)
    with (args.development_output / "victim_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        victim_rows = list(csv.DictReader(handle))
    if len(victim_rows) != 1 or int(victim_rows[0]["seed"]) != 100:
        raise RuntimeError("expected exactly the seed-100 development smoke")
    victim = victim_rows[0]
    gate = development["smoke_gate"]
    if float(victim["balanced_clean_accuracy"]) < float(
        gate["test_balanced_clean_accuracy_min"]
    ):
        raise RuntimeError("development smoke missed the clean BCA gate")
    if float(victim["attack_success_rate"]) < float(gate["attack_success_rate_min"]):
        raise RuntimeError("development smoke missed the ASR gate")
    if int(victim["eligible"]) != 1:
        raise RuntimeError("development smoke was not marked eligible")

    checkpoint = Path(victim["checkpoint_path"])
    smoke_receipt = args.development_output / "receipts/seed_100.json"
    mps_receipt_path = args.development_output / "device_smoke_mps.json"
    mps_receipt = read_json(mps_receipt_path)
    if mps_receipt.get("status") != "pass" or mps_receipt.get("actual_device") != "mps":
        raise RuntimeError("a passing actual-MPS checkpoint smoke is required")
    if mps_receipt.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise RuntimeError("MPS receipt does not identify the qualified checkpoint")

    planned = development["planned_confirmation"]
    confirmation = {
        "protocol_id": "tifs_native_gtsrb_confirmation_v1",
        "stage": "confirmation",
        "frozen_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "freeze_method": (
            "programmatic datetime.now(timezone.utc).astimezone().isoformat(); "
            "refuse overwrite"
        ),
        "development_provenance": {
            "protocol_path": str(args.development_protocol.resolve()),
            "protocol_sha256": sha256_file(args.development_protocol),
            "smoke_manifest_path": str(
                (args.development_output / "manifest.json").resolve()
            ),
            "smoke_manifest_sha256": sha256_file(
                args.development_output / "manifest.json"
            ),
            "smoke_receipt_path": str(smoke_receipt.resolve()),
            "smoke_receipt_sha256": sha256_file(smoke_receipt),
            "smoke_checkpoint_path": str(checkpoint.resolve()),
            "smoke_checkpoint_sha256": sha256_file(checkpoint),
            "mps_smoke_receipt_path": str(mps_receipt_path.resolve()),
            "mps_smoke_receipt_sha256": sha256_file(mps_receipt_path),
            "qualified_metrics": {
                "clean_accuracy": float(victim["clean_accuracy"]),
                "balanced_clean_accuracy": float(victim["balanced_clean_accuracy"]),
                "attack_success_rate": float(victim["attack_success_rate"]),
            },
        },
        "runner": {
            "path": str(args.runner.resolve()),
            "sha256": sha256_file(args.runner),
        },
        "dataset": development["dataset"],
        "seeds": planned["seeds"],
        "split": development["split"],
        "attack": development["attack"],
        "victim": development["victim"],
        "trusted_split": development["trusted_split"],
        "detectors": development["detectors"],
        "execution": {
            "formal_device": "mps",
            "fallback": "fail",
            "cli_requirement": "--device mps",
            "checkpoint_state_storage": "cpu",
            "actual_device_must_be_recorded_per_seed": True,
        },
        "victim_eligibility": planned["victim_eligibility"],
        "minimum_eligible_victims": planned["minimum_eligible_victims"],
        "primary_detector_gate": planned["primary_detector_gate"],
        "no_seed_replacement": bool(planned["no_seed_replacement"]),
        "stopping_rule": {
            "attempt_all_frozen_seeds": True,
            "resume_only_at_completed_seed_boundaries": True,
            "aggregate_detector_metrics_over_eligible_victims_only": True,
            "no_hyperparameter_or_threshold_retuning_after_any_confirmation_outcome": True,
            "report_ineligible_and_failed_seeds_without_replacement": True,
            "qualification": (
                "confirmation is complete after all 10 frozen seeds are attempted; "
                "primary gate is evaluated only if at least 8 victims are eligible"
            ),
        },
    }
    if len(confirmation["seeds"]) != 10 or len(set(confirmation["seeds"])) != 10:
        raise RuntimeError("confirmation seed list must contain 10 unique seeds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(confirmation, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "frozen",
                "path": str(args.output.resolve()),
                "sha256": sha256_file(args.output),
                "frozen_at": confirmation["frozen_at"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
