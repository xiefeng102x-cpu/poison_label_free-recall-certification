#!/usr/bin/env python3
"""Read-only verification of the frozen P6-D v1 formal development failure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from staged_provenance import effective_sha256, effective_size


ROOT = Path(__file__).resolve().parents[1]
PROGRAM_REL = Path("results/top_journal_program_v1")
DEVELOPMENT_REL = Path("results/p6d_final_observable_adaptive_v1/development")
RUNNER_SHA256 = "a9afe435f41c05605820b1692b178ec9289e4ba178f7c8a80caf00986d796d09"
PROTOCOL_SHA256 = "0eb9316274c428c7893407fbd7bba6f7e1d9f379b9d6e5b28aaa172e56678225"


def sha256(path: Path) -> str:
    return effective_sha256(path, ROOT)


def csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def verify_p6d_v1_formal_failure(root: Path = ROOT) -> dict[str, Any]:
    """Verify receipt, narrative, frozen source identities and surviving artifacts."""
    program = root / PROGRAM_REL
    development = root / DEVELOPMENT_REL
    receipt_path = program / "p6d_v1_formal_failure_receipt.json"
    report_path = program / "P6D_V1_FORMAL_FAILURE.md"
    protocol_path = program / "p6d_final_observable_adaptive_protocol_v1.json"
    runner_path = root / "src/run_p6d_final_observable_adaptive.py"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest_path = development / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection_path = development / "development_selection_receipt.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    assert receipt["receipt_id"] == "paper14_p6d_v1_formal_failure_2026-10-01"
    assert receipt["status"] == "FORMAL_DEVELOPMENT_FAILED_PRE_TEST_ARTIFACT_SCHEMA"
    assert receipt["protocol_id"] == manifest["protocol_id"]
    assert receipt["failure_phase"].startswith("after all nine validation-only attack trainings")
    assert "before observable_rows.csv was written" in receipt["failure_phase"]
    assert "before any test split was read" in receipt["failure_phase"]
    assert receipt["exception"] == "ValueError: CSV rows do not share one ordered schema"
    assert "trusted_sample_index" in receipt["root_cause"]
    assert receipt["scientific_parameters_changed_before_failure"] is False
    assert receipt["protocol_sha256"] == PROTOCOL_SHA256 == sha256(protocol_path)
    assert receipt["formal_runner_sha256"] == RUNNER_SHA256 == sha256(runner_path)
    assert manifest["status"] == "initialized"
    assert manifest["run_signature"]["stage"] == "development"
    assert manifest["run_signature"]["seeds"] == [95, 96, 97]
    assert manifest["run_signature"]["runner_sha256"] == RUNNER_SHA256
    assert manifest["run_signature"]["protocol_sha256"] == PROTOCOL_SHA256
    assert manifest["immutable_hashes"]["src/run_p6d_final_observable_adaptive.py"] == RUNNER_SHA256
    assert manifest["immutable_hashes"]["protocol"] == PROTOCOL_SHA256

    partial_hashes = receipt["partial_artifacts"]
    for relative_path, expected_hash in partial_hashes.items():
        artifact = development / relative_path
        assert artifact.is_file(), artifact
        assert sha256(artifact) == expected_hash, relative_path
    for relative_path, metadata in manifest["artifacts"].items():
        artifact = development / relative_path
        assert artifact.is_file(), artifact
        assert effective_size(artifact, ROOT) == metadata["bytes"], relative_path
        assert sha256(artifact) == metadata["sha256"], relative_path
    assert partial_hashes["manifest.json"] == sha256(manifest_path)
    assert partial_hashes["development_selection_receipt.json"] == sha256(selection_path)

    expected_rows = receipt["partial_row_counts_excluding_header"]
    assert set(expected_rows) == {"epoch_history.csv", "per_sample_scores.csv"}
    for filename, expected_count in expected_rows.items():
        assert csv_row_count(development / filename) == expected_count, filename
    assert selection["selection_uses_test"] is False
    assert selection["selected_configuration_id"] == "w0.5"
    assert selection["selected_evasion_weight"] == 0.5

    assert receipt["development_seeds_opened_and_consumed"] == [95, 96, 97]
    assert receipt["confirmation_seeds_not_opened"] == [98, 99, 100, 101, 102]
    assert receipt["test_split_read"] is False
    assert receipt["victim_metrics_created"] is False
    assert receipt["detector_efficacy_claim_allowed"] is False
    assert receipt["frozen_decision"].startswith("Do not rerun or repair v1 in place")
    assert receipt["selected_configuration_diagnostic_only"]["selection_uses_test"] is False
    assert "victim_metrics.csv" in receipt["required_missing_artifacts"]
    assert {
        "observable_rows.csv",
        "fold_receipts.json",
        "split_receipts.json",
        "per_seed_metrics.csv",
        "victim_metrics.csv",
        "checkpoint_receipts.json",
    }.issubset(set(receipt["required_missing_artifacts"]))
    assert not list(development.rglob("victim_metrics*"))
    assert not list(development.rglob("*confirmation*lock*"))
    assert not list(development.rglob("confirmation_lock_receipt.json"))

    report = " ".join(report_path.read_text(encoding="utf-8").split())
    for phrase in (
        "failed while writing `observable_rows.csv`",
        "extra `trusted_sample_index` field",
        "before any test split was constructed or read",
        "Seeds 95--97 are consumed",
        "Confirmation seeds 98--102 were never opened",
        "no victim eligibility or detector-efficacy conclusion is permitted",
        "A continuation must use a v2 protocol and entirely new seed blocks",
    ):
        assert phrase in report, phrase
    return {
        "status": "PASS",
        "receipt_status": receipt["status"],
        "runner_sha256": RUNNER_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "verified_partial_artifacts": len(partial_hashes),
        "verified_manifest_artifacts": len(manifest["artifacts"]),
        "epoch_history_rows": expected_rows["epoch_history.csv"],
        "per_sample_score_rows": expected_rows["per_sample_scores.csv"],
        "victim_metrics_present": False,
        "confirmation_lock_present": False,
        "test_split_read": False,
        "detector_efficacy_claim_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="paper14 root")
    args = parser.parse_args()
    print(json.dumps(verify_p6d_v1_formal_failure(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
