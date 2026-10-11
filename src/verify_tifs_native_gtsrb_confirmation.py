#!/usr/bin/env python3
"""Independently verify the frozen native-GTSRB confirmation artifacts.

This verifier does not retrain a victim or change a detector decision.  It
checks the frozen protocol/source binding, model and per-sample hashes, row
count invariants, per-seed confusion matrices, eligibility decisions, and the
eligible-victim pooled primary gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/tifs_native_gtsrb_confirmation_v1_1.json"
DEFAULT_OUTPUT = ROOT / "results/tifs_native_cifar_gtsrb_v1/gtsrb"
DEFAULT_RECEIPT = DEFAULT_OUTPUT / "verification_receipt.json"

SEEDS = list(range(101, 111))
METHODS = ("cf_cmd_u", "target_centroid_distance", "crossfit_spectral")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float, tol: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tol, abs_tol=tol)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify(protocol_path: Path, output: Path) -> dict[str, Any]:
    errors: list[str] = []
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    victims = read_csv(output / "victim_metrics.csv")
    detectors = read_csv(output / "detector_metrics.csv")

    protocol_sha = sha256_file(protocol_path)
    runner = ROOT / "src/run_tifs_native_gtsrb.py"
    if protocol.get("protocol_id") != "tifs_native_gtsrb_confirmation_v1_1":
        errors.append("unexpected protocol id")
    if protocol.get("seeds") != SEEDS:
        errors.append("frozen seed block mismatch")
    if sha256_file(output / "frozen_protocol.json") != protocol_sha:
        errors.append("archived protocol differs from invoked protocol")
    if protocol.get("runner", {}).get("sha256") != sha256_file(runner):
        errors.append("runner source hash differs from frozen protocol")
    if manifest.get("status") != "complete" or manifest.get("stage") != "confirmation":
        errors.append("manifest is not a completed confirmation")
    if manifest.get("protocol_sha256") != protocol_sha:
        errors.append("manifest protocol hash mismatch")
    if manifest.get("runner_sha256") != sha256_file(runner):
        errors.append("manifest runner hash mismatch")
    if manifest.get("completed_seeds") != SEEDS:
        errors.append("manifest completed-seed block mismatch")
    if manifest.get("requested_device") != "mps":
        errors.append("formal run did not request MPS")
    devices = manifest.get("actual_device_by_seed", {})
    if devices != {str(seed): "mps" for seed in SEEDS}:
        errors.append("not every seed records MPS as the actual device")

    victim_by_seed = {int(row["seed"]): row for row in victims}
    if sorted(victim_by_seed) != SEEDS or len(victims) != len(SEEDS):
        errors.append("victim table is not exactly the frozen ten-seed block")
    detector_by_key = {(int(row["seed"]), row["method"]): row for row in detectors}
    if len(detectors) != len(SEEDS) * len(METHODS) or set(detector_by_key) != {
        (seed, method) for seed in SEEDS for method in METHODS
    }:
        errors.append("detector table is not the complete 10 x 3 block")

    replayed: dict[tuple[int, str], dict[str, int]] = {}
    eligible: list[int] = []
    for seed in SEEDS:
        victim = victim_by_seed.get(seed)
        if victim is None:
            continue
        expected_eligible = (
            float(victim["balanced_clean_accuracy"])
            >= float(protocol["victim_eligibility"]["test_balanced_clean_accuracy_min"])
            and float(victim["attack_success_rate"])
            >= float(protocol["victim_eligibility"]["attack_success_rate_min"])
        )
        if bool(int(victim["eligible"])) is not expected_eligible:
            errors.append(f"seed {seed}: eligibility decision mismatch")
        if expected_eligible:
            eligible.append(seed)
        expected_counts = {
            "clean_training_count": 23976,
            "poison_count": 2264,
            "candidate_count": 26240,
            "defined_target_candidate_count": 3614,
            "clean_test_count": 12630,
            "asr_test_count": 11880,
        }
        for key, expected in expected_counts.items():
            if int(victim[key]) != expected:
                errors.append(f"seed {seed}: {key} mismatch")
        if victim.get("actual_device") != "mps":
            errors.append(f"seed {seed}: actual device is not MPS")

        checkpoint = Path(victim["checkpoint_path"])
        if not checkpoint.is_file() or sha256_file(checkpoint) != victim["checkpoint_sha256"]:
            errors.append(f"seed {seed}: checkpoint hash mismatch")
        receipt_path = output / "receipts" / f"seed_{seed}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("protocol_sha256") != protocol_sha:
            errors.append(f"seed {seed}: receipt protocol hash mismatch")
        if receipt.get("checkpoint_sha256") != victim["checkpoint_sha256"]:
            errors.append(f"seed {seed}: receipt checkpoint hash mismatch")
        if receipt.get("actual_device") != "mps":
            errors.append(f"seed {seed}: receipt device mismatch")
        sample_path = Path(receipt["per_sample_path"])
        if not sample_path.is_file() or sha256_file(sample_path) != receipt["per_sample_sha256"]:
            errors.append(f"seed {seed}: per-sample file hash mismatch")
            continue
        rows = read_csv(sample_path)
        if len(rows) != 3614 * len(METHODS):
            errors.append(f"seed {seed}: per-sample row count mismatch")
            continue
        for method in METHODS:
            block = [row for row in rows if row["method"] == method]
            if len(block) != 3614:
                errors.append(f"seed {seed}/{method}: detector-defined row count mismatch")
                continue
            if len({row["record_id"] for row in block}) != 3614:
                errors.append(f"seed {seed}/{method}: record ids are not unique")
            truth = [int(row["poison_truth"]) for row in block]
            flags = [int(row["flag"]) for row in block]
            if sum(truth) != 2264 or len(truth) - sum(truth) != 1350:
                errors.append(f"seed {seed}/{method}: truth-count invariant mismatch")
            if any(int(row["assigned_label"]) != 2 or int(row["defined"]) != 1 for row in block):
                errors.append(f"seed {seed}/{method}: target-domain invariant mismatch")
            try:
                scores = [float(row["score"]) for row in block]
                pvalues = [float(row["p_value"]) for row in block]
                if not all(math.isfinite(value) for value in scores):
                    raise ValueError("non-finite score")
                if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in pvalues):
                    raise ValueError("invalid p-value")
            except ValueError as exc:
                errors.append(f"seed {seed}/{method}: {exc}")
            tp = sum(flag and poison for flag, poison in zip(flags, truth))
            fp = sum(flag and not poison for flag, poison in zip(flags, truth))
            replayed[(seed, method)] = {
                "tp": tp,
                "fn": 2264 - tp,
                "fp": fp,
                "tn_target": 1350 - fp,
                "tn_all_clean": 23976 - fp,
                "flagged_count": sum(flags),
            }

    for key, counts in replayed.items():
        row = detector_by_key.get(key)
        if row is None:
            continue
        seed, method = key
        for name, expected in counts.items():
            if int(row[name]) != expected:
                errors.append(f"seed {seed}/{method}: {name} does not replay")
        expected_tpr = counts["tp"] / 2264
        expected_target_fpr = counts["fp"] / 1350
        expected_all_fpr = counts["fp"] / 23976
        for name, expected in (
            ("tpr", expected_tpr),
            ("target_clean_fpr", expected_target_fpr),
            ("all_clean_fpr", expected_all_fpr),
        ):
            if not close(float(row[name]), expected):
                errors.append(f"seed {seed}/{method}: {name} does not replay")
        if int(row["victim_eligible"]) != int(seed in eligible):
            errors.append(f"seed {seed}/{method}: victim eligibility join mismatch")

    pooled: dict[str, dict[str, float | int | None]] = {}
    for method in METHODS:
        rows = [detector_by_key[(seed, method)] for seed in eligible if (seed, method) in detector_by_key]
        tp = sum(int(row["tp"]) for row in rows)
        fn = sum(int(row["fn"]) for row in rows)
        fp = sum(int(row["fp"]) for row in rows)
        tn = sum(int(row["tn_target"]) for row in rows)
        pooled[method] = {
            "eligible_seed_count": len(rows),
            "pooled_tpr": tp / (tp + fn) if tp + fn else None,
            "pooled_target_clean_fpr": fp / (fp + tn) if fp + tn else None,
        }
        recorded = summary.get("methods", {}).get(method, {})
        for name in ("eligible_seed_count", "pooled_tpr", "pooled_target_clean_fpr"):
            if recorded.get(name) is None or pooled[method][name] is None:
                if recorded.get(name) != pooled[method][name]:
                    errors.append(f"summary {method}/{name} mismatch")
            elif not close(float(recorded[name]), float(pooled[method][name])):
                errors.append(f"summary {method}/{name} mismatch")

    enough = len(eligible) >= int(protocol["minimum_eligible_victims"])
    primary = pooled["cf_cmd_u"]
    gate = bool(
        enough
        and primary["pooled_tpr"] is not None
        and primary["pooled_target_clean_fpr"] is not None
        and float(primary["pooled_tpr"]) >= float(protocol["primary_detector_gate"]["pooled_tpr_min"])
        and float(primary["pooled_target_clean_fpr"])
        <= float(protocol["primary_detector_gate"]["pooled_target_clean_fpr_max"])
    )
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "formal_evidence": not errors,
        "verified_seeds": SEEDS,
        "eligible_seeds": eligible,
        "eligible_seed_count": len(eligible),
        "minimum_eligible_victims": int(protocol["minimum_eligible_victims"]),
        "primary_gate_evaluable": enough,
        "primary_gate_pass": gate,
        "pooled_metrics_eligible_victims": pooled,
        "protocol_sha256": protocol_sha,
        "runner_sha256": sha256_file(runner),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    result = verify(args.protocol.resolve(), args.output.resolve())
    atomic_json(args.receipt.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
