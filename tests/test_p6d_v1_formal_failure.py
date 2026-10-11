from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import verify_p6d_v1_formal_failure as verifier


def test_frozen_incident_receipt_and_partial_bundle_verify():
    result = verifier.verify_p6d_v1_formal_failure(ROOT)
    assert result["status"] == "PASS"
    assert result["verified_partial_artifacts"] == 6
    assert result["verified_manifest_artifacts"] == 9
    assert result["epoch_history_rows"] == 117
    assert result["per_sample_score_rows"] == 5940
    assert result["victim_metrics_present"] is False
    assert result["confirmation_lock_present"] is False
    assert result["test_split_read"] is False
    assert result["detector_efficacy_claim_allowed"] is False


def test_partial_hash_mismatch_is_rejected(monkeypatch):
    original_sha256 = verifier.sha256
    target = ROOT / verifier.DEVELOPMENT_REL / "per_sample_scores.csv"

    def altered_sha256(path):
        if path == target:
            return "0" * 64
        return original_sha256(path)

    monkeypatch.setattr(verifier, "sha256", altered_sha256)
    with pytest.raises(AssertionError):
        verifier.verify_p6d_v1_formal_failure(ROOT)


def test_partial_row_count_mismatch_is_rejected(monkeypatch):
    original_count = verifier.csv_row_count

    def altered_count(path):
        count = original_count(path)
        return count + 1 if path.name == "epoch_history.csv" else count

    monkeypatch.setattr(verifier, "csv_row_count", altered_count)
    with pytest.raises(AssertionError):
        verifier.verify_p6d_v1_formal_failure(ROOT)
