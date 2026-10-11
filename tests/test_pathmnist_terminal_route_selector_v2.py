from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import select_pathmnist_terminal_route_v2 as selector  # noqa: E402


def make_output(tmp_path: Path, eligible: int, primary_pass: bool) -> Path:
    output = tmp_path / "formal"
    output.mkdir()
    eligibility = {
        str(seed): index < eligible for index, seed in enumerate(selector.SEEDS)
    }
    manifest = {
        "status": "COMPLETE_FORMAL",
        "completed_seeds": selector.SEEDS,
        "victim_eligibility_by_seed": eligibility,
        "all_pairs_eligible": eligible == 5,
        "primary_point_gate_pass": primary_pass,
    }
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in selector.TOP_LEVEL_FILES[1:]:
        (output / name).write_text(name + "\n", encoding="utf-8")
    return output


def make_replay(tmp_path: Path, output: Path, *, valid: bool = True) -> Path:
    acquisition = json.loads(
        selector.ACQUISITION_RECEIPT.read_text(encoding="utf-8")
    )
    replay = {
        "status": "PASS" if valid else "FAIL",
        "formal_evidence": valid,
        "failures": [] if valid else ["synthetic invalid replay"],
        "verified_seeds": selector.SEEDS,
        "verified_method_rows": 25,
        "path_only_staging": True,
        "dataset_bound_before_replay": True,
        "dataset_reopened_after_replay": False,
        "evidence_files_modified": False,
        "performance_endpoints_inspected_before_replay": False,
        "formal_output": str(output.resolve()),
        "dataset_sha256": acquisition["sha256"],
        "dataset_byte_count": acquisition["byte_count"],
        "acquisition_receipt_sha256": selector.sha256(
            selector.ACQUISITION_RECEIPT
        ),
        "frozen_verifier_sha256": selector.sha256(selector.FROZEN_VERIFIER),
        "staging_wrapper_v2_sha256": selector.sha256(selector.V2_WRAPPER),
        "amendment_sha256": selector.sha256(selector.V2_AMENDMENT),
        "evidence_hashes": {
            name: selector.sha256(output / name) for name in selector.TOP_LEVEL_FILES
        },
    }
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(replay), encoding="utf-8")
    return path


def test_invalid_replay_selects_r0_without_parsing_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    output = make_output(tmp_path, eligible=5, primary_pass=True)
    replay = make_replay(tmp_path, output, valid=False)
    monkeypatch.setattr(
        selector.json,
        "loads",
        selector.json.loads,
    )
    result = selector.select(replay, output)
    assert result["selected_route"] == "R0_INVALID_OR_INCOMPLETE"
    assert result["performance_artifacts_parsed"] is False
    assert result["pooling_allowed"] is False


@pytest.mark.parametrize(
    ("eligible", "primary_pass", "expected"),
    [
        (4, False, "R1_INSUFFICIENT_ELIGIBLE_VICTIMS"),
        (5, False, "R2_ELIGIBLE_PRIMARY_GATE_FAIL"),
        (5, True, "R3_ELIGIBLE_PRIMARY_GATE_PASS"),
    ],
)
def test_valid_replay_selects_frozen_route(
    tmp_path: Path, eligible: int, primary_pass: bool, expected: str
) -> None:
    output = make_output(tmp_path, eligible=eligible, primary_pass=primary_pass)
    replay = make_replay(tmp_path, output)
    result = selector.select(replay, output)
    assert result["selected_route"] == expected
    assert result["performance_artifacts_parsed"] is True
    assert result["performance_endpoints_inspected_before_replay_pass"] is False
