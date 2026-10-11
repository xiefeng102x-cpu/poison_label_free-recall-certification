from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import select_pathmnist_terminal_route_v2 as selector  # noqa: E402
import verify_top_journal_final_bundle_v2 as verifier  # noqa: E402


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preterminal_final_verifier_is_closed_after_terminal_integration() -> None:
    with pytest.raises(RuntimeError, match="preterminal mutable baseline drift"):
        verifier.preterminal_self_check(write_receipt=False)

    contract = json.loads(verifier.EXECUTION_CONTRACT.read_text(encoding="utf-8"))
    for relative, baseline in contract["preterminal_mutable_hashes"].items():
        assert file_hash(PAPER14 / relative) != baseline
    for relative, baseline in contract["protected_history_hashes"].items():
        assert file_hash(PAPER14 / relative) == baseline


def test_missing_replay_receipt_never_parses_performance(tmp_path: Path) -> None:
    result = verifier.verify_terminal_control(
        tmp_path / "missing-replay.json",
        tmp_path / "missing-route.json",
        tmp_path / "formal",
    )
    assert result["status"] == "WAITING_REPLAY_RECEIPT"
    assert result["replay_receipt_opened"] is False
    assert result["route_receipt_opened"] is False
    assert result["performance_artifacts_parsed"] is False


def test_invalid_replay_receipt_never_opens_route_or_manifest(tmp_path: Path) -> None:
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")
    result = verifier.verify_terminal_control(
        replay,
        tmp_path / "route.json",
        tmp_path / "formal",
    )
    assert result["status"] == "REPLAY_CONTROL_REJECTED"
    assert result["replay_receipt_opened"] is True
    assert result["route_receipt_opened"] is False
    assert result["performance_artifacts_parsed"] is False


def test_valid_synthetic_receipts_unlock_exact_route_parse(tmp_path: Path) -> None:
    output = tmp_path / "formal"
    output.mkdir()
    manifest = {
        "status": "COMPLETE_FORMAL",
        "completed_seeds": [103, 104, 105, 106, 107],
        "victim_eligibility_by_seed": {
            str(seed): True for seed in [103, 104, 105, 106, 107]
        },
        "all_pairs_eligible": True,
        "primary_point_gate_pass": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in ("seed_receipts.json", "detector_metrics.csv", "scores.csv"):
        (output / name).write_text("synthetic\n", encoding="utf-8")

    acquisition = json.loads(selector.ACQUISITION_RECEIPT.read_text(encoding="utf-8"))
    replay_payload = {
        "status": "PASS",
        "formal_evidence": True,
        "failures": [],
        "verified_seeds": [103, 104, 105, 106, 107],
        "verified_method_rows": 25,
        "path_only_staging": True,
        "dataset_bound_before_replay": True,
        "dataset_reopened_after_replay": False,
        "evidence_files_modified": False,
        "performance_endpoints_inspected_before_replay": False,
        "formal_output": str(output.resolve()),
        "frozen_verifier_sha256": file_hash(selector.FROZEN_VERIFIER),
        "staging_wrapper_v2_sha256": file_hash(selector.V2_WRAPPER),
        "amendment_sha256": file_hash(selector.V2_AMENDMENT),
        "dataset_sha256": acquisition["sha256"],
        "dataset_byte_count": acquisition["byte_count"],
        "acquisition_receipt_sha256": file_hash(selector.ACQUISITION_RECEIPT),
        "evidence_hashes": {
            name: file_hash(output / name) for name in selector.TOP_LEVEL_FILES
        },
    }
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps(replay_payload), encoding="utf-8")
    expected_route = selector.select(replay, output, None)
    route = tmp_path / "route.json"
    route.write_text(json.dumps(expected_route), encoding="utf-8")

    result = verifier.verify_terminal_control(replay, route, output)
    assert result["status"] == "PASS_TERMINAL_CONTROL"
    assert result["errors"] == []
    assert result["selected_route"] == "R3_ELIGIBLE_PRIMARY_GATE_PASS"
    assert result["claim_decision"] == "SUPPORTED_CONDITIONALLY"
    assert result["performance_artifacts_parsed"] is True


def test_exact_terminal_route_marker_is_required_on_every_surface(tmp_path: Path) -> None:
    contract = {
        "required_terminal_route_marker_prefix": "PATHMNIST_TERMINAL_ROUTE=",
        "mutable_integration_targets": ["a.md", "b.tex", "c.json"],
    }
    route = "R2_ELIGIBLE_PRIMARY_GATE_FAIL"
    marker = f"PATHMNIST_TERMINAL_ROUTE={route}"
    for name in contract["mutable_integration_targets"]:
        (tmp_path / name).write_text(marker + "\n", encoding="utf-8")
    result = verifier.verify_route_markers(contract, route, root=tmp_path)
    assert result["errors"] == []
    assert len(result["verified_targets"]) == 3

    (tmp_path / "b.tex").write_text(
        "PATHMNIST_TERMINAL_ROUTE=R3_ELIGIBLE_PRIMARY_GATE_PASS\n",
        encoding="utf-8",
    )
    failed = verifier.verify_route_markers(contract, route, root=tmp_path)
    assert "selected route marker missing: b.tex" in failed["errors"]


def test_capsule_hash_sentences_and_metric_markers_bind_surfaces(tmp_path: Path) -> None:
    paths = ["en.tex", "tifs.tex", "zh.tex", "table.md", "supp.tex", "status.md"]
    contract = {
        "required_terminal_capsule_hash_marker_prefix": "PATHMNIST_CLAIM_CAPSULE_SHA256=",
        "required_terminal_metric_marker_prefix": "PATHMNIST_METRIC=",
        "mutable_integration_targets": paths,
        "terminal_claim_binding_surfaces": {
            "english_sentence_targets": ["en.tex", "tifs.tex"],
            "chinese_sentence_targets": ["zh.tex"],
            "metric_targets": ["table.md", "supp.tex"],
        },
    }
    route = "R3_ELIGIBLE_PRIMARY_GATE_PASS"
    summary = {
        "pooled_tpr_pct": 92.0,
        "pooled_all_clean_fpr_pct": 2.0,
        "pooled_target_clean_fpr_pct": 3.0,
    }
    capsule = {
        "status": "PASS",
        "selected_route": route,
        "main_text_en": "Bounded English terminal claim.",
        "main_text_zh": "有边界的中文终局主张。",
        "detector_summaries": {"primary_cf_cmd_u": summary},
    }
    capsule_path = tmp_path / "capsule.json"
    capsule_path.write_text(json.dumps(capsule), encoding="utf-8")
    capsule_marker = f"PATHMNIST_CLAIM_CAPSULE_SHA256={verifier.sha256(capsule_path)}"
    metric = verifier.metric_marker("primary_cf_cmd_u", summary)
    for relative in paths:
        content = capsule_marker + "\n"
        if relative in ("en.tex", "tifs.tex"):
            content += capsule["main_text_en"] + "\n"
        if relative == "zh.tex":
            content += capsule["main_text_zh"] + "\n"
        if relative in ("table.md", "supp.tex"):
            content += metric + "\n"
        (tmp_path / relative).write_text(content, encoding="utf-8")
    result = verifier.verify_capsule_surface_binding(
        contract, route, capsule_path, root=tmp_path
    )
    assert result["errors"] == []
    assert len(result["capsule_marked_targets"]) == len(paths)
    assert len(result["sentence_targets_verified"]) == 3
    assert len(result["metric_targets_verified"]) == 2

    (tmp_path / "supp.tex").write_text(capsule_marker + "\n", encoding="utf-8")
    failed = verifier.verify_capsule_surface_binding(
        contract, route, capsule_path, root=tmp_path
    )
    assert "terminal metric markers missing: supp.tex" in failed["errors"]
