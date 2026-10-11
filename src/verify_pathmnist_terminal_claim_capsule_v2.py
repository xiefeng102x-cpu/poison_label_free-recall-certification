#!/usr/bin/env python3
"""Independently replay the route-authorized PathMNIST claim capsule V2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import select_pathmnist_terminal_route_v2 as selector
import verify_top_journal_final_bundle_v2 as final_v2


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "results/top_journal_program_v1"
FORMAL_OUTPUT = ROOT / "results/pathmnist_official_weight_detector_confirmation_v1"
DEFAULT_CAPSULE = PROGRAM / "pathmnist_terminal_claim_capsule_v2_2026-10-04.json"
DEFAULT_RECEIPT = PROGRAM / (
    "pathmnist_terminal_claim_capsule_verification_v2_2026-10-04.json"
)
BUILDER = ROOT / "src/build_pathmnist_terminal_claim_capsule_v2.py"
SEEDS = [103, 104, 105, 106, 107]
METHODS = [
    "primary_cf_cmd_u",
    "paired_drift_cf_cmd_u",
    "centroid",
    "scan",
    "spectral",
]


def pct(value: float) -> float:
    return round(100.0 * float(value), 2)


def expected_victims(seed_receipts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if [int(row.get("seed", -1)) for row in seed_receipts] != SEEDS:
        raise RuntimeError("independent capsule replay found a seed mismatch")
    return [
        {
            "seed": int(row["seed"]),
            "eligible": bool(row["eligible"]),
            "clean_bca_pct": pct(row["clean_test"]["balanced_accuracy"]),
            "suspect_bca_pct": pct(row["suspect_test"]["balanced_accuracy"]),
            "suspect_asr_pct": pct(row["suspect_test"]["asr"]),
        }
        for row in seed_receipts
    ]


def expected_summaries(manifest: dict[str, Any]) -> dict[str, Any]:
    source = manifest.get("detector_summaries", {})
    result: dict[str, Any] = {}
    for method in METHODS:
        if method not in source:
            raise RuntimeError(f"independent capsule replay lacks method {method}")
        summary = source[method]
        intervals = summary.get("seed_block_bootstrap_95pct_ci")
        if not isinstance(intervals, dict):
            raise RuntimeError(f"independent capsule replay lacks intervals for {method}")
        result[method] = {
            "pooled_tpr_pct": pct(summary["pooled_tpr"]),
            "pooled_all_clean_fpr_pct": pct(summary["pooled_all_clean_fpr"]),
            "pooled_target_clean_fpr_pct": pct(summary["pooled_target_clean_fpr"]),
            "seed_block_bootstrap_95pct_ci_pct": {
                name: [pct(value) for value in intervals[name]]
                for name in (
                    "pooled_tpr",
                    "pooled_all_clean_fpr",
                    "pooled_target_clean_fpr",
                )
            },
            "per_seed": summary["per_seed"],
        }
    return result


def expected_chinese(route: str, eligible: int, primary: dict[str, Any] | None) -> str:
    if route == "R1_INSUFFICIENT_ELIGIBLE_VICTIMS":
        return (
            f"独立重放的五 seed PathMNIST 区块只有 {eligible}/5 个攻击合格的 "
            "victim pair；检测输出仅作为逐 seed 诊断，不合并外部域检测效力端点。"
        )
    if primary is None:
        raise RuntimeError("independent R2/R3 replay lacks primary metrics")
    tpr = primary["pooled_tpr_pct"]
    fpr = primary["pooled_all_clean_fpr_pct"]
    if route == "R2_ELIGIBLE_PRIMARY_GATE_FAIL":
        return (
            "五个 official-weight PathMNIST ResNet-18 victim pair 均通过独立重放，"
            f"但冻结 primary detector 的 TPR/全体干净 FPR 为 {tpr:.2f}%/{fpr:.2f}%，"
            "未通过点门槛；这是已确认的外部域局限，而不是正面效力证据。"
        )
    if route == "R3_ELIGIBLE_PRIMARY_GATE_PASS":
        return (
            "五个 official-weight PathMNIST ResNet-18 victim pair 均通过独立重放，"
            f"冻结 primary detector 的 TPR/全体干净 FPR 为 {tpr:.2f}%/{fpr:.2f}%。"
            "这只是在该数据集、架构、攻击、reference budget 与 calibration law 下的"
            "有条件外部域确认。"
        )
    raise RuntimeError(f"unsupported route in independent capsule replay: {route}")


def verify(
    *,
    replay_receipt: Path = selector.DEFAULT_REPLAY_RECEIPT,
    route_receipt: Path = selector.DEFAULT_SELECTION_RECEIPT,
    formal_output: Path = FORMAL_OUTPUT,
    capsule_path: Path = DEFAULT_CAPSULE,
    receipt_path: Path = DEFAULT_RECEIPT,
    write_receipt: bool = True,
) -> dict[str, Any]:
    control = final_v2.verify_terminal_control(
        replay_receipt, route_receipt, formal_output
    )
    if control["status"] != "PASS_TERMINAL_CONTROL":
        raise RuntimeError("; ".join(control["errors"]))
    if not capsule_path.is_file():
        raise RuntimeError("terminal claim capsule is absent")

    route_record = json.loads(route_receipt.read_text(encoding="utf-8"))
    manifest_path = formal_output / "manifest.json"
    seeds_path = formal_output / "seed_receipts.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seeds = json.loads(seeds_path.read_text(encoding="utf-8"))
    capsule = json.loads(capsule_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    route = route_record["selected_route"]
    victims = expected_victims(seeds)
    eligible = sum(bool(row["eligible"]) for row in victims)

    expected_scalars = {
        "status": "PASS",
        "schema": "pathmnist_terminal_claim_capsule_v2",
        "selected_route": route,
        "route_marker": f"PATHMNIST_TERMINAL_ROUTE={route}",
        "claim_decision": route_record["claim_decision"],
        "pooling_allowed": route_record["pooling_allowed"],
        "eligible_seed_count": eligible,
        "victim_rows": victims,
        "allowed_numeric_fields": route_record["allowed_numeric_fields"],
        "formal_manifest_sha256": final_v2.sha256(manifest_path),
        "seed_receipts_sha256": final_v2.sha256(seeds_path),
        "replay_receipt_sha256": final_v2.sha256(replay_receipt),
        "route_receipt_sha256": final_v2.sha256(route_receipt),
        "builder_sha256": final_v2.sha256(BUILDER),
        "performance_parsed_only_after_terminal_control_pass": True,
    }
    for key, expected in expected_scalars.items():
        if capsule.get(key) != expected:
            errors.append(f"claim capsule mismatch: {key}")

    expected_detector: dict[str, Any] | None
    if route == "R1_INSUFFICIENT_ELIGIBLE_VICTIMS":
        expected_detector = None
        if eligible >= 5 or route_record["pooling_allowed"] is not False:
            errors.append("R1 independent replay violates eligibility/pooling")
    else:
        expected_detector = expected_summaries(manifest)
        if eligible != 5 or route_record["pooling_allowed"] is not True:
            errors.append("R2/R3 independent replay violates eligibility/pooling")
        primary = expected_detector["primary_cf_cmd_u"]
        gate = bool(
            primary["pooled_tpr_pct"] >= 90.0
            and primary["pooled_all_clean_fpr_pct"] <= 5.0
        )
        if gate is not route_record["primary_point_gate_pass"]:
            errors.append("independent primary gate differs from route receipt")
    if capsule.get("detector_summaries") != expected_detector:
        errors.append("claim capsule detector summaries do not replay")

    primary = None if expected_detector is None else expected_detector["primary_cf_cmd_u"]
    english = route_record["allowed_main_text_template"].replace(
        "{{eligible_seed_count}}", str(eligible)
    )
    if primary is not None:
        english = english.replace(
            "{{primary_tpr_pct}}", f'{primary["pooled_tpr_pct"]:.2f}'
        ).replace(
            "{{primary_all_clean_fpr_pct}}",
            f'{primary["pooled_all_clean_fpr_pct"]:.2f}',
        )
    if capsule.get("main_text_en") != english:
        errors.append("claim capsule English text does not replay")
    if capsule.get("main_text_zh") != expected_chinese(route, eligible, primary):
        errors.append("claim capsule Chinese text does not replay")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "selected_route": route,
        "victim_rows_replayed": len(victims),
        "detector_summaries_replayed": 0 if expected_detector is None else len(expected_detector),
        "pooling_allowed": route_record["pooling_allowed"],
        "capsule_sha256": final_v2.sha256(capsule_path),
        "builder_sha256": final_v2.sha256(BUILDER),
        "verifier_sha256": final_v2.sha256(Path(__file__)),
        "performance_parsed_only_after_terminal_control_pass": True,
    }
    if write_receipt:
        final_v2.atomic_json(receipt_path, result)
    if errors:
        raise RuntimeError("; ".join(errors))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-receipt", type=Path, default=selector.DEFAULT_REPLAY_RECEIPT)
    parser.add_argument("--route-receipt", type=Path, default=selector.DEFAULT_SELECTION_RECEIPT)
    parser.add_argument("--formal-output", type=Path, default=FORMAL_OUTPUT)
    parser.add_argument("--capsule", type=Path, default=DEFAULT_CAPSULE)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                replay_receipt=args.replay_receipt,
                route_receipt=args.route_receipt,
                formal_output=args.formal_output,
                capsule_path=args.capsule,
                receipt_path=args.receipt,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
