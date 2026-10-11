from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import verify_p6d_v2_formal_failure as verifier


def test_v2_failure_receipt_replays_complete_pre_test_state():
    result = verifier.verify_p6d_v2_formal_failure(ROOT)
    assert result == {
        "status": "PASS", "epoch_rows": 117, "score_rows": 5940,
        "observable_rows": 7740, "fold_receipts": 9,
        "validation_training_groups": 9, "test_split_read": False,
        "victim_metrics_present": False, "detector_efficacy_claim_allowed": False,
    }


def test_v2_failure_verifier_rejects_hash_drift(tmp_path):
    fake = tmp_path / "paper14"
    (fake / "results/top_journal_program_v1").mkdir(parents=True)
    receipt = json.loads(verifier.RECEIPT.read_text())
    receipt["partial_artifact_hashes"] = {}
    (fake / "results/top_journal_program_v1/p6d_v2_formal_failure_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8")
    with pytest.raises((AssertionError, FileNotFoundError)):
        verifier.verify_p6d_v2_formal_failure(fake)


def test_v2_failure_receipt_closes_all_old_seed_blocks():
    receipt = json.loads(verifier.RECEIPT.read_text())
    assert receipt["development_seeds_opened_and_consumed"] == [108, 109, 110]
    assert receipt["confirmation_seeds_permanently_closed"] == [111, 112, 113, 114, 115]
    assert receipt["scientific_interpretation"]["detector_efficacy_claim_allowed"] is False
