from __future__ import annotations

import json
from pathlib import Path
import sys


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import verify_pathmnist_terminal_integration_execution_contract_v2 as verifier  # noqa: E402


def test_terminal_execution_contract_covers_v2_closure_without_results() -> None:
    result = json.loads(verifier.RECEIPT.read_text(encoding="utf-8"))
    assert result["status"] == "PASS_PRETERMINAL_EXECUTION"
    assert result["errors"] == []
    assert result["routes_checked"] == 4
    assert result["mutable_targets_covered"] == 19
    assert result["preterminal_mutable_hashes_verified"] == 19
    assert result["protected_history_artifacts"] == 4
    assert result["protected_history_hashes_verified"] == 4
    assert result["terminal_surface_actions_verified"] == 19
    assert result["preterminal_surface_anchors_verified"] == 19
    assert result["preterminal_route_marker_absence_checks"] == 19
    assert result["preterminal_capsule_marker_absence_checks"] == 19
    assert result["preterminal_metric_marker_absence_checks"] == 19
    assert result["required_terminal_inputs"] == 2
    assert result["required_new_terminal_outputs"] == 9
    assert result["mandatory_terminal_steps"] == 12
    assert result["route_independent_invariants"] == 8
    assert result["pathmnist_result_artifacts_read"] is False
    assert result["submission_ready"] is False

    contract = json.loads(verifier.CONTRACT.read_text(encoding="utf-8"))
    route_marker = "PATHMNIST_TERMINAL_ROUTE=R3_ELIGIBLE_PRIMARY_GATE_PASS"
    capsule_marker = "PATHMNIST_CLAIM_CAPSULE_SHA256="
    anonymous_submission_sources = {
        "manuscript_tifs_en/main_tifs.tex",
        "manuscript_tifs_en/supplement_tifs.tex",
    }
    for relative in contract["mutable_integration_targets"]:
        text = (PAPER14 / relative).read_text(encoding="utf-8")
        if relative in anonymous_submission_sources:
            assert route_marker not in text
            assert capsule_marker not in text
        else:
            assert route_marker in text
            assert capsule_marker in text
    for relative, expected in contract["protected_history_hashes"].items():
        assert verifier.sha256(PAPER14 / relative) == expected


def test_legacy_final_receipt_cannot_be_made_mutable(tmp_path: Path) -> None:
    contract = json.loads(verifier.CONTRACT.read_text(encoding="utf-8"))
    legacy = "results/top_journal_program_v1/final_bundle_receipt.json"
    contract["protected_history"].remove(legacy)
    contract["mutable_integration_targets"].append(legacy)
    tampered = tmp_path / "contract.json"
    tampered.write_text(json.dumps(contract), encoding="utf-8")
    try:
        verifier.run(contract_path=tampered, write_receipt=False)
    except RuntimeError as exc:
        assert "legacy final-bundle receipt is not protected history" in str(exc)
    else:
        raise AssertionError("legacy V1 receipt was accepted as a mutable target")


def test_preterminal_surface_hash_drift_is_rejected(tmp_path: Path) -> None:
    contract = json.loads(verifier.CONTRACT.read_text(encoding="utf-8"))
    target = "manuscript_tifs_en/main_tifs.tex"
    contract["preterminal_mutable_hashes"][target] = "0" * 64
    tampered = tmp_path / "contract.json"
    tampered.write_text(json.dumps(contract), encoding="utf-8")
    try:
        verifier.run(contract_path=tampered, write_receipt=False)
    except RuntimeError as exc:
        assert "preterminal mutable baseline drift:" in str(exc)
    else:
        raise AssertionError("preterminal claim-surface drift was accepted")


def test_terminal_surface_action_coverage_is_complete(tmp_path: Path) -> None:
    contract = json.loads(verifier.CONTRACT.read_text(encoding="utf-8"))
    contract["terminal_surface_actions"].pop()
    tampered = tmp_path / "contract.json"
    tampered.write_text(json.dumps(contract), encoding="utf-8")
    try:
        verifier.run(contract_path=tampered, write_receipt=False)
    except RuntimeError as exc:
        assert "terminal surface actions do not cover every mutable target" in str(exc)
    else:
        raise AssertionError("incomplete terminal surface action map was accepted")
