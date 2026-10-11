from __future__ import annotations

import json
from pathlib import Path
import sys


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import verify_tifs_terminal_manuscript_v2 as verifier  # noqa: E402


def test_tifs_terminal_verifier_preparation_is_outcome_blind() -> None:
    result = verifier.preterminal_self_check(write_receipt=False)
    assert result["status"] == "PASS_PRETERMINAL_TIFS_TERMINAL_VERIFIER_PREP"
    assert result["errors"] == []
    assert result["configuration_tokens_checked"] == 35
    assert result["frozen_protocol_hashes_checked"] == 9
    assert result["pathmnist_terminal_receipts_opened"] is False
    assert result["pathmnist_performance_artifacts_parsed"] is False
    assert result["submission_ready"] is False


def test_terminal_tifs_surface_requires_authorized_claim_and_metrics(tmp_path: Path) -> None:
    route = "R3_ELIGIBLE_PRIMARY_GATE_PASS"
    summary = {
        "pooled_tpr_pct": 92.0,
        "pooled_all_clean_fpr_pct": 2.0,
        "pooled_target_clean_fpr_pct": 3.0,
    }
    capsule = {
        "selected_route": route,
        "main_text_en": "Bounded terminal claim.",
        "detector_summaries": {"primary_cf_cmd_u": summary},
    }
    capsule_path = tmp_path / "capsule.json"
    capsule_path.write_text(json.dumps(capsule), encoding="utf-8")
    capsule_sha = verifier.sha256(capsule_path)
    main = capsule["main_text_en"]
    supplement = "92.00 & 2.00 & 3.00"
    assert verifier.terminal_route_surface_errors(
        main, supplement, capsule, capsule_sha
    ) == []
    failed = verifier.terminal_route_surface_errors(
        main, supplement.replace("92.00", ""), capsule, capsule_sha
    )
    assert "TIFS supplement lacks capsule endpoint values: primary_cf_cmd_u" in failed


def test_r1_tifs_surface_rejects_pooled_metric_marker(tmp_path: Path) -> None:
    route = "R1_INSUFFICIENT_ELIGIBLE_VICTIMS"
    capsule = {
        "selected_route": route,
        "main_text_en": "No pooled external-domain efficacy endpoint.",
        "detector_summaries": None,
    }
    capsule_path = tmp_path / "capsule.json"
    capsule_path.write_text(json.dumps(capsule), encoding="utf-8")
    capsule_sha = verifier.sha256(capsule_path)
    main = capsule["main_text_en"]
    supplement = "Frozen PathMNIST detector endpoints"
    errors = verifier.terminal_route_surface_errors(
        main, supplement, capsule, capsule_sha
    )
    assert "ineligible route exposes pooled PathMNIST endpoints" in errors


def test_terminal_tifs_surface_rejects_internal_markers(tmp_path: Path) -> None:
    capsule = {
        "selected_route": "R1_INSUFFICIENT_ELIGIBLE_VICTIMS",
        "main_text_en": "No pooled external-domain efficacy endpoint.",
        "detector_summaries": None,
    }
    capsule_path = tmp_path / "capsule.json"
    capsule_path.write_text(json.dumps(capsule), encoding="utf-8")
    errors = verifier.terminal_route_surface_errors(
        capsule["main_text_en"] + "\nPATHMNIST_TERMINAL_ROUTE=R1",
        "",
        capsule,
        verifier.sha256(capsule_path),
    )
    assert any("internal terminal marker leaked" in error for error in errors)
