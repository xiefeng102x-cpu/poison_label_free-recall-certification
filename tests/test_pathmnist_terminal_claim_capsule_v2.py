from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import build_pathmnist_terminal_claim_capsule_v2 as builder  # noqa: E402
import select_pathmnist_terminal_route_v2 as selector  # noqa: E402
import verify_pathmnist_terminal_claim_capsule_v2 as capsule_verifier  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_bundle(tmp_path: Path, *, eligible: int, gate: bool) -> tuple[Path, Path, Path]:
    output = tmp_path / f"formal-{eligible}-{int(gate)}"
    output.mkdir()
    eligibility = {
        str(seed): index < eligible for index, seed in enumerate(builder.SEEDS)
    }
    summaries = {}
    for method_index, method in enumerate(builder.METHODS):
        primary_tpr = 0.92 if gate else 0.85
        summaries[method] = {
            "pooled_tpr": primary_tpr - 0.01 * method_index,
            "pooled_all_clean_fpr": 0.02 + 0.005 * method_index,
            "pooled_target_clean_fpr": 0.03 + 0.005 * method_index,
            "seed_block_bootstrap_95pct_ci": {
                "pooled_tpr": [0.88, 0.96],
                "pooled_all_clean_fpr": [0.01, 0.04],
                "pooled_target_clean_fpr": [0.02, 0.05],
            },
            "per_seed": [{"seed": seed, "method": method} for seed in builder.SEEDS],
        }
    manifest = {
        "status": "COMPLETE_FORMAL",
        "completed_seeds": builder.SEEDS,
        "victim_eligibility_by_seed": eligibility,
        "all_pairs_eligible": eligible == 5,
        "primary_point_gate_pass": bool(eligible == 5 and gate),
        "detector_summaries": summaries,
    }
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    seed_rows = [
        {
            "seed": seed,
            "eligible": eligibility[str(seed)],
            "clean_test": {"balanced_accuracy": 0.86},
            "suspect_test": {"balanced_accuracy": 0.84, "asr": 0.95},
        }
        for seed in builder.SEEDS
    ]
    (output / "seed_receipts.json").write_text(json.dumps(seed_rows), encoding="utf-8")
    for name in ("detector_metrics.csv", "scores.csv"):
        (output / name).write_text("synthetic\n", encoding="utf-8")

    acquisition = json.loads(selector.ACQUISITION_RECEIPT.read_text(encoding="utf-8"))
    replay_payload = {
        "status": "PASS",
        "formal_evidence": True,
        "failures": [],
        "verified_seeds": builder.SEEDS,
        "verified_method_rows": 25,
        "path_only_staging": True,
        "dataset_bound_before_replay": True,
        "dataset_reopened_after_replay": False,
        "evidence_files_modified": False,
        "performance_endpoints_inspected_before_replay": False,
        "formal_output": str(output.resolve()),
        "frozen_verifier_sha256": digest(selector.FROZEN_VERIFIER),
        "staging_wrapper_v2_sha256": digest(selector.V2_WRAPPER),
        "amendment_sha256": digest(selector.V2_AMENDMENT),
        "dataset_sha256": acquisition["sha256"],
        "dataset_byte_count": acquisition["byte_count"],
        "acquisition_receipt_sha256": digest(selector.ACQUISITION_RECEIPT),
        "evidence_hashes": {
            name: digest(output / name) for name in selector.TOP_LEVEL_FILES
        },
    }
    replay = tmp_path / f"replay-{eligible}-{int(gate)}.json"
    replay.write_text(json.dumps(replay_payload), encoding="utf-8")
    route_payload = selector.select(replay, output, None)
    route = tmp_path / f"route-{eligible}-{int(gate)}.json"
    route.write_text(json.dumps(route_payload), encoding="utf-8")
    return replay, route, output


@pytest.mark.parametrize(
    ("eligible", "gate", "expected_route", "has_detector_summary"),
    [
        (4, False, "R1_INSUFFICIENT_ELIGIBLE_VICTIMS", False),
        (5, False, "R2_ELIGIBLE_PRIMARY_GATE_FAIL", True),
        (5, True, "R3_ELIGIBLE_PRIMARY_GATE_PASS", True),
    ],
)
def test_capsule_obeys_route_specific_information_budget(
    tmp_path: Path,
    eligible: int,
    gate: bool,
    expected_route: str,
    has_detector_summary: bool,
) -> None:
    replay, route, output = synthetic_bundle(tmp_path, eligible=eligible, gate=gate)
    capsule_path = tmp_path / f"capsule-{eligible}-{int(gate)}.json"
    capsule = builder.build(
        replay_receipt=replay,
        route_receipt=route,
        formal_output=output,
        capsule_path=capsule_path,
    )
    assert capsule["status"] == "PASS"
    assert capsule["selected_route"] == expected_route
    assert capsule["eligible_seed_count"] == eligible
    assert (capsule["detector_summaries"] is not None) is has_detector_summary
    assert "{{" not in capsule["main_text_en"]
    assert capsule["route_marker"] == f"PATHMNIST_TERMINAL_ROUTE={expected_route}"
    if eligible < 5:
        assert "pooled_tpr" not in json.dumps(capsule["detector_summaries"])
        assert "不合并外部域检测效力端点" in capsule["main_text_zh"]


def test_capsule_rejects_missing_terminal_replay(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="V2 replay receipt is absent"):
        builder.build(
            replay_receipt=tmp_path / "missing.json",
            route_receipt=tmp_path / "route.json",
            formal_output=tmp_path / "formal",
            capsule_path=tmp_path / "capsule.json",
        )


@pytest.mark.parametrize(("eligible", "gate"), [(4, False), (5, False), (5, True)])
def test_independent_capsule_replay_accepts_all_terminal_routes(
    tmp_path: Path, eligible: int, gate: bool
) -> None:
    replay, route, output = synthetic_bundle(tmp_path, eligible=eligible, gate=gate)
    capsule_path = tmp_path / f"verified-capsule-{eligible}-{int(gate)}.json"
    builder.build(
        replay_receipt=replay,
        route_receipt=route,
        formal_output=output,
        capsule_path=capsule_path,
    )
    result = capsule_verifier.verify(
        replay_receipt=replay,
        route_receipt=route,
        formal_output=output,
        capsule_path=capsule_path,
        receipt_path=tmp_path / "unused.json",
        write_receipt=False,
    )
    assert result["status"] == "PASS"
    assert result["victim_rows_replayed"] == 5
    assert result["detector_summaries_replayed"] == (0 if eligible < 5 else 5)


def test_independent_capsule_replay_rejects_tampered_metric(tmp_path: Path) -> None:
    replay, route, output = synthetic_bundle(tmp_path, eligible=5, gate=True)
    capsule_path = tmp_path / "tampered-capsule.json"
    capsule = builder.build(
        replay_receipt=replay,
        route_receipt=route,
        formal_output=output,
        capsule_path=capsule_path,
    )
    capsule["detector_summaries"]["primary_cf_cmd_u"]["pooled_tpr_pct"] = 99.99
    capsule_path.write_text(json.dumps(capsule), encoding="utf-8")
    with pytest.raises(RuntimeError, match="detector summaries do not replay"):
        capsule_verifier.verify(
            replay_receipt=replay,
            route_receipt=route,
            formal_output=output,
            capsule_path=capsule_path,
            receipt_path=tmp_path / "unused.json",
            write_receipt=False,
        )
