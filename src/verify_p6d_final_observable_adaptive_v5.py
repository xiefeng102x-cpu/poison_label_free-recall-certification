#!/usr/bin/env python3
"""Independent integrity and decision replay for the P6-D v5 development stop."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "results/p6d_final_observable_adaptive_v5/development"
DEFAULT_RECEIPT = ROOT / "results/top_journal_program_v1/p6d_v5_development_stop_receipt.json"
PROTOCOL = ROOT / "results/top_journal_program_v1/p6d_final_observable_adaptive_protocol_v5.json"
RUNNER = ROOT / "src/run_p6d_final_observable_adaptive_v5.py"
DEVELOPMENT_SEEDS = (134, 135, 136)
CONFIRMATION_SEEDS = tuple(range(137, 147))
RULES = ("hard60", "uncertain60", "stratified60")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(payload: Any) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def margin(row: Mapping[str, Any]) -> float:
    return min(
        (float(row["validation_balanced_ca"]) - 0.92) / 0.02,
        (float(row["validation_asr"]) - 0.90) / 0.05,
    )


def replay_rule_summary(rule: str, rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    chosen = [row for row in rows if row["configuration_id"] == rule and as_bool(row["selected_epoch"])]
    if sorted(int(row["seed"]) for row in chosen) != list(DEVELOPMENT_SEEDS):
        raise AssertionError(f"{rule} does not contain exactly one selected epoch per development seed")
    guarded = [
        row for row in chosen
        if float(row["validation_balanced_ca"]) >= 0.92 and float(row["validation_asr"]) >= 0.90
    ]
    return {
        "configuration_id": rule,
        "guarded_seed_count": len(guarded),
        "worst_normalized_utility_margin": min(margin(row) for row in chosen),
        "mean_tpr_guarded": sum(float(row["tpr"]) for row in guarded) / len(guarded) if guarded else None,
        "worst_validation_asr": min(float(row["validation_asr"]) for row in chosen),
        "worst_validation_balanced_ca": min(float(row["validation_balanced_ca"]) for row in chosen),
    }


def rule_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -int(row["guarded_seed_count"]),
        -float(row["worst_normalized_utility_margin"]),
        float(row["mean_tpr_guarded"]) if row["mean_tpr_guarded"] is not None else float("inf"),
        -float(row["worst_validation_asr"]),
        RULES.index(str(row["configuration_id"])),
    )


def verify(bundle: Path) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["source_audit"]["development_runner_sha256"] != sha256_file(RUNNER):
        raise AssertionError("Frozen runner hash mismatch")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise AssertionError("Development manifest is not complete")
    if manifest["protocol_sha256"] != sha256_file(PROTOCOL) or manifest["runner_sha256"] != sha256_file(RUNNER):
        raise AssertionError("Manifest source/protocol hash mismatch")
    for relative, expected in manifest["artifacts"].items():
        path = bundle / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise AssertionError(f"Artifact hash mismatch: {relative}")

    selection_path = bundle / "development_selection_receipt.json"
    sidecar = bundle / "development_selection_receipt.json.sha256"
    if sidecar.read_text(encoding="utf-8").strip() != sha256_file(selection_path):
        raise AssertionError("Development selection sidecar mismatch")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("selection_uses_test") is not False:
        raise AssertionError("Development selection is not marked validation-only")
    epoch_rows = read_csv(bundle / "epoch_history.csv")
    forbidden = [key for key in epoch_rows[0] if key.startswith("test_")]
    if forbidden:
        raise AssertionError("Test field present in validation selection table")
    replayed = [replay_rule_summary(rule, epoch_rows) for rule in RULES]
    selected_rule = min(replayed, key=rule_key)["configuration_id"]
    if selected_rule != selection["selected_configuration_id"]:
        raise AssertionError("Independent validation-only selection replay disagrees")

    victim_rows = read_csv(bundle / "victim_metrics.csv")
    metric_rows = read_csv(bundle / "per_seed_metrics.csv")
    if sorted(int(row["seed"]) for row in victim_rows) != list(DEVELOPMENT_SEEDS):
        raise AssertionError("Victim table seed mismatch")
    if sorted(int(row["seed"]) for row in metric_rows) != list(DEVELOPMENT_SEEDS):
        raise AssertionError("Detector table seed mismatch")
    viable = 0
    for row in victim_rows:
        expected = float(row["test_balanced_ca"]) >= 0.90 and float(row["test_asr"]) >= 0.80
        if as_bool(row["eligible"]) != expected or row["configuration_id"] != selected_rule:
            raise AssertionError("Victim eligibility/configuration mismatch")
        viable += int(expected)
    evasive = sum(
        as_bool(row["eligible"]) and float(row["tpr"]) < 0.90 and float(row["fpr"]) <= 0.05
        for row in metric_rows
    )
    summary = json.loads((bundle / "development_summary.json").read_text(encoding="utf-8"))
    expected_decision = "AUTHORIZE_CONFIRMATION_FREEZE" if viable == 3 and evasive >= 2 else "STOP_NO_CONFIRMATION"
    if (
        summary["selected_configuration_id"] != selected_rule
        or int(summary["attack_viable_count"]) != viable
        or int(summary["evasive_and_viable_count"]) != evasive
        or summary["decision"] != expected_decision
        or bool(summary["confirmation_authorized"]) != (expected_decision == "AUTHORIZE_CONFIRMATION_FREEZE")
    ):
        raise AssertionError("Development decision replay mismatch")

    receipts = json.loads((bundle / "confidence_selection_receipts.json").read_text(encoding="utf-8"))
    if len(receipts) != 9:
        raise AssertionError("Expected nine confidence selection receipts")
    for receipt in receipts:
        indices = [int(value) for value in receipt["selected_train_indices"]]
        if len(indices) != 60 or len(set(indices)) != 60:
            raise AssertionError("Confidence selector receipt is not 60 unique members")
        if stable_json_hash(indices) != receipt["selected_indices_sha256"]:
            raise AssertionError("Confidence selector index hash mismatch")
        if int(receipt["seed"]) not in DEVELOPMENT_SEEDS or receipt["configuration_id"] not in RULES:
            raise AssertionError("Confidence selector seed/configuration mismatch")

    names = "\n".join(path.name for path in bundle.rglob("*"))
    if any(str(seed) in names for seed in CONFIRMATION_SEEDS):
        raise AssertionError("Confirmation seed identifier found in development bundle")
    return {
        "status": "VALID_DEVELOPMENT_STOP",
        "bundle": str(bundle.resolve()),
        "manifest_sha256": sha256_file(bundle / "manifest.json"),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(RUNNER),
        "selected_configuration_id": selected_rule,
        "development_seed_count": len(DEVELOPMENT_SEEDS),
        "attack_viable_count": viable,
        "evasive_and_viable_count": evasive,
        "decision": expected_decision,
        "confirmation_seeds_consumed": False,
        "integrity_replay": True,
        "full_numeric_score_replay": False,
        "numeric_replay_boundary": "The development bundle stores outcome counts and checkpoints but not the full observable matrix needed for independent score recomputation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--write-receipt", type=Path)
    args = parser.parse_args()
    result = verify(args.bundle.resolve())
    if args.write_receipt:
        target = args.write_receipt.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
