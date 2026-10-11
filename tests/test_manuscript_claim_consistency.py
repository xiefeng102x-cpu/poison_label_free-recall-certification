from __future__ import annotations

import sys
from pathlib import Path


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import verify_manuscript_claim_consistency as audit  # noqa: E402


def test_percentage_formatter_uses_declared_precision() -> None:
    assert audit.pct(0.4266666666666667) == "42.67"
    assert audit.pct(3.521436746192447e-05, 4) == "0.0035"


def test_single_sample_summary_is_recomputed_from_completed_outputs() -> None:
    summary = audit.primary_single_sample_summary()
    assert round(summary["tpr_min"], 6) == round(0.5266666666666666, 6)
    assert round(summary["tpr_max"], 6) == round(0.6071428571428571, 6)
    assert round(summary["fpr_min"], 6) == round(0.0064285714285714285, 6)
    assert round(summary["fpr_max"], 6) == round(0.015, 6)


def test_full_bilingual_claim_consistency_audit_passes() -> None:
    # The test must remain read-only inside a hash-closed reviewer package.
    receipt = audit.run(receipt_path=None)
    assert receipt["status"] == "PASS"
    assert receipt["errors"] == []
    assert receipt["pathmnist_terminal_capsule_read"] is True
    assert receipt["pathmnist_raw_or_partial_performance_artifacts_read"] is False
    assert receipt["authoritative_claim_families_recomputed"] == 7
    assert receipt["frozen_configuration_tokens_checked"] == 24
    assert receipt["documentation_consistency_tokens_checked"] == 11
    assert set(receipt["documentation_hashes"]) == {
        "status",
        "reproduce",
        "artifact_index",
    }
