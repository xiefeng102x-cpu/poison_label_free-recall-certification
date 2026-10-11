#!/usr/bin/env python3
"""Independently verify the immutable P6-D v2 pre-test failure."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from staged_provenance import effective_sha256

PAPER14_ROOT = Path(__file__).resolve().parents[1]
PROGRAM = PAPER14_ROOT / "results/top_journal_program_v1"
FAILED = PAPER14_ROOT / "results/p6d_final_observable_adaptive_v2/development"
RECEIPT = PROGRAM / "p6d_v2_formal_failure_receipt.json"


def sha256(path: Path) -> str:
    return effective_sha256(path, PAPER14_ROOT)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_p6d_v2_formal_failure(root: Path = PAPER14_ROOT) -> dict[str, Any]:
    program = root / "results/top_journal_program_v1"
    failed = root / "results/p6d_final_observable_adaptive_v2/development"
    receipt = json.loads((program / RECEIPT.name).read_text(encoding="utf-8"))
    expected_sources = {
        "protocol_v2": root / "results/top_journal_program_v1/p6d_final_observable_adaptive_protocol_v2.json",
        "runner_v2": root / "src/run_p6d_final_observable_adaptive_v2.py",
        "verifier_v2": root / "src/verify_p6d_final_observable_adaptive_v2.py",
        "focused_test_v2": root / "tests/test_p6d_final_observable_adaptive_v2.py",
    }
    for key, path in expected_sources.items():
        if sha256(path) != receipt["source_hashes"][key]:
            raise AssertionError(f"V2 source hash mismatch: {key}")
    expected_files = set(receipt["partial_artifact_hashes"])
    actual_files = {
        path.relative_to(failed).as_posix()
        for path in failed.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise AssertionError("V2 partial artifact inventory changed")
    for relative, digest in receipt["partial_artifact_hashes"].items():
        if sha256(failed / relative) != digest:
            raise AssertionError(f"V2 partial artifact hash mismatch: {relative}")

    counts = {
        "epoch_history.csv": len(csv_rows(failed / "epoch_history.csv")),
        "per_sample_scores.csv": len(csv_rows(failed / "per_sample_scores.csv")),
        "observable_rows.csv": len(csv_rows(failed / "observable_rows.csv")),
        "fold_receipts.json": len(json.loads((failed / "fold_receipts.json").read_text())),
    }
    if counts != receipt["surviving_row_counts"]:
        raise AssertionError("V2 surviving row counts changed")
    history = csv_rows(failed / "epoch_history.csv")
    groups = {(row["seed"], row["configuration_id"]) for row in history}
    if groups != {(str(seed), config) for seed in (108, 109, 110)
                  for config in ("w0.5", "w2", "w8")}:
        raise AssertionError("V2 did not preserve all nine validation-only trainings")

    forbidden = {
        "split_receipts.json", "per_seed_metrics.csv", "victim_metrics.csv",
        "checkpoint_receipts.json", "EXPERIMENT_REPORT.md",
        "confirmation_lock_receipt.json",
    }
    if any((failed / name).exists() for name in forbidden):
        raise AssertionError("A post-failure/test artifact unexpectedly exists")
    if list(failed.glob("seed_*/w*/attacked_projector.pt")):
        raise AssertionError("A selected attacked projector exists despite pre-test failure")

    source = (root / "src/run_p6d_final_observable_adaptive.py").read_text(encoding="utf-8")
    split_write = source.index('atomic_json(stage_root / "split_receipts.json", split_receipts)',
                               source.index("def run_development_launcher"))
    authorization = source.index("authorize_development_test_read", split_write)
    test_read = source.index("test_split = deferred_tests[seed]()", authorization)
    if not split_write < authorization < test_read:
        raise AssertionError("Inherited launcher no longer proves the pre-test failure order")
    if receipt["development_seeds_opened_and_consumed"] != [108, 109, 110]:
        raise AssertionError("V2 development seed disposition mismatch")
    if receipt["confirmation_seeds_permanently_closed"] != [111, 112, 113, 114, 115]:
        raise AssertionError("V2 confirmation seed disposition mismatch")
    return {
        "status": "PASS",
        "epoch_rows": counts["epoch_history.csv"],
        "score_rows": counts["per_sample_scores.csv"],
        "observable_rows": counts["observable_rows.csv"],
        "fold_receipts": counts["fold_receipts.json"],
        "validation_training_groups": len(groups),
        "test_split_read": False,
        "victim_metrics_present": False,
        "detector_efficacy_claim_allowed": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify_p6d_v2_formal_failure(), indent=2, sort_keys=True))
