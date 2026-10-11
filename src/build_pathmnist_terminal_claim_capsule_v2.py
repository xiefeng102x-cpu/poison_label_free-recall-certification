#!/usr/bin/env python3
"""Build the route-authorized PathMNIST terminal claim capsule.

The capsule is the only normalized performance payload used for manuscript
integration.  A replay and route receipt must first pass the V2 final control
gate.  R1 exports victim feasibility only; R2/R3 export the complete frozen
primary/baseline summaries.  The builder is never a polling command.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import select_pathmnist_terminal_route_v2 as selector
import verify_top_journal_final_bundle_v2 as final_v2


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "results/top_journal_program_v1"
FORMAL_OUTPUT = ROOT / "results/pathmnist_official_weight_detector_confirmation_v1"
DEFAULT_CAPSULE = PROGRAM / "pathmnist_terminal_claim_capsule_v2_2026-10-04.json"
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


def victim_rows(seed_receipts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if [int(row.get("seed", -1)) for row in seed_receipts] != SEEDS:
        raise RuntimeError("terminal seed receipts do not match 103--107")
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


def authorized_detector_summaries(manifest: dict[str, Any]) -> dict[str, Any]:
    source = manifest.get("detector_summaries", {})
    if not all(method in source for method in METHODS):
        raise RuntimeError("formal manifest lacks a complete detector summary set")
    result: dict[str, Any] = {}
    for method in METHODS:
        summary = source[method]
        intervals = summary.get("seed_block_bootstrap_95pct_ci")
        if not isinstance(intervals, dict):
            raise RuntimeError(f"eligible route lacks seed-block intervals: {method}")
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


def chinese_text(route: str, eligible: int, primary: dict[str, Any] | None) -> str:
    if route == "R1_INSUFFICIENT_ELIGIBLE_VICTIMS":
        return (
            f"独立重放的五 seed PathMNIST 区块只有 {eligible}/5 个攻击合格的 "
            "victim pair；检测输出仅作为逐 seed 诊断，不合并外部域检测效力端点。"
        )
    if primary is None:
        raise RuntimeError("R2/R3 claim text requires primary metrics")
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
    raise RuntimeError(f"unsupported terminal claim route: {route}")


def build(
    *,
    replay_receipt: Path = selector.DEFAULT_REPLAY_RECEIPT,
    route_receipt: Path = selector.DEFAULT_SELECTION_RECEIPT,
    formal_output: Path = FORMAL_OUTPUT,
    capsule_path: Path = DEFAULT_CAPSULE,
) -> dict[str, Any]:
    control = final_v2.verify_terminal_control(
        replay_receipt, route_receipt, formal_output
    )
    if control["status"] != "PASS_TERMINAL_CONTROL":
        raise RuntimeError("; ".join(control["errors"]))
    route_record = json.loads(route_receipt.read_text(encoding="utf-8"))
    route = route_record["selected_route"]
    manifest_path = formal_output / "manifest.json"
    seeds_path = formal_output / "seed_receipts.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seeds = json.loads(seeds_path.read_text(encoding="utf-8"))
    victims = victim_rows(seeds)
    eligible = sum(bool(row["eligible"]) for row in victims)
    if eligible != route_record["eligible_seed_count"]:
        raise RuntimeError("capsule eligibility count differs from route receipt")

    detector_summaries: dict[str, Any] | None = None
    primary: dict[str, Any] | None = None
    if route in {
        "R2_ELIGIBLE_PRIMARY_GATE_FAIL",
        "R3_ELIGIBLE_PRIMARY_GATE_PASS",
    }:
        if eligible != 5 or route_record["pooling_allowed"] is not True:
            raise RuntimeError("pooled route is not backed by five eligible victims")
        detector_summaries = authorized_detector_summaries(manifest)
        primary = detector_summaries["primary_cf_cmd_u"]
        recomputed_gate = bool(
            primary["pooled_tpr_pct"] >= 90.0
            and primary["pooled_all_clean_fpr_pct"] <= 5.0
        )
        if recomputed_gate is not route_record["primary_point_gate_pass"]:
            raise RuntimeError("capsule primary metrics disagree with the route gate")
    elif route == "R1_INSUFFICIENT_ELIGIBLE_VICTIMS":
        if eligible >= 5 or route_record["pooling_allowed"] is not False:
            raise RuntimeError("R1 pooling/eligibility contract mismatch")
    else:
        raise RuntimeError("R0 cannot produce a scientific claim capsule")

    english = route_record["allowed_main_text_template"]
    english = english.replace("{{eligible_seed_count}}", str(eligible))
    if primary is not None:
        english = english.replace(
            "{{primary_tpr_pct}}", f'{primary["pooled_tpr_pct"]:.2f}'
        ).replace(
            "{{primary_all_clean_fpr_pct}}",
            f'{primary["pooled_all_clean_fpr_pct"]:.2f}',
        )
    if "{{" in english or "}}" in english:
        raise RuntimeError("terminal English claim template is not fully rendered")

    capsule = {
        "status": "PASS",
        "schema": "pathmnist_terminal_claim_capsule_v2",
        "selected_route": route,
        "route_marker": f"PATHMNIST_TERMINAL_ROUTE={route}",
        "claim_decision": route_record["claim_decision"],
        "pooling_allowed": route_record["pooling_allowed"],
        "eligible_seed_count": eligible,
        "victim_rows": victims,
        "detector_summaries": detector_summaries,
        "allowed_numeric_fields": route_record["allowed_numeric_fields"],
        "main_text_en": english,
        "main_text_zh": chinese_text(route, eligible, primary),
        "formal_manifest_sha256": final_v2.sha256(manifest_path),
        "seed_receipts_sha256": final_v2.sha256(seeds_path),
        "replay_receipt_sha256": final_v2.sha256(replay_receipt),
        "route_receipt_sha256": final_v2.sha256(route_receipt),
        "builder_sha256": final_v2.sha256(Path(__file__)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "performance_parsed_only_after_terminal_control_pass": True,
    }
    final_v2.atomic_json(capsule_path, capsule)
    return capsule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-receipt", type=Path, default=selector.DEFAULT_REPLAY_RECEIPT)
    parser.add_argument("--route-receipt", type=Path, default=selector.DEFAULT_SELECTION_RECEIPT)
    parser.add_argument("--formal-output", type=Path, default=FORMAL_OUTPUT)
    parser.add_argument("--capsule", type=Path, default=DEFAULT_CAPSULE)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                replay_receipt=args.replay_receipt,
                route_receipt=args.route_receipt,
                formal_output=args.formal_output,
                capsule_path=args.capsule,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
