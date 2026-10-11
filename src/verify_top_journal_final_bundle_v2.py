#!/usr/bin/env python3
"""V2 terminal gate and final-bundle verifier for paper14.

Preterminal self-check mode reads only control contracts and preterminal file
hashes.  Terminal mode first validates the atomic V2 replay receipt; only then
does it validate the route receipt and permit the frozen selector to parse the
formal PathMNIST manifest.  The V1 verifier and receipt remain immutable
history.  This module is prepared pre-outcome but must not be run in terminal
mode until route integration and the required terminal receipts exist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import select_pathmnist_terminal_route_v2 as selector


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "results/top_journal_program_v1"
EXECUTION_CONTRACT = PROGRAM / "pathmnist_terminal_integration_execution_contract_v2.json"
MIGRATION_CONTRACT = PROGRAM / "final_bundle_v2_migration_contract_2026-10-04.json"
PREPARATION_RECEIPT = PROGRAM / "final_bundle_v2_verifier_preparation_2026-10-04.json"
FINAL_RECEIPT = PROGRAM / "final_bundle_v2_receipt_2026-10-04.json"
FORMAL_OUTPUT = ROOT / "results/pathmnist_official_weight_detector_confirmation_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.split())


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_execution_contract() -> dict[str, Any]:
    contract = json.loads(EXECUTION_CONTRACT.read_text(encoding="utf-8"))
    if contract.get("status") != "FROZEN_PRETERMINAL_EXECUTION":
        raise RuntimeError("terminal execution contract is not frozen")
    if contract.get("performance_endpoints_inspected") is not False:
        raise RuntimeError("terminal execution contract violates endpoint sealing")
    return contract


def preterminal_self_check(*, write_receipt: bool = True) -> dict[str, Any]:
    """Prove V2 verifier readiness without opening either terminal receipt."""

    contract = load_execution_contract()
    errors: list[str] = []
    mutable_hashes = contract["preterminal_mutable_hashes"]
    protected_hashes = contract["protected_history_hashes"]
    for relative, expected in mutable_hashes.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"preterminal mutable baseline drift: {relative}")
    for relative, expected in protected_hashes.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"protected history drift: {relative}")

    required_inputs = contract["required_terminal_inputs"]
    if required_inputs != [
        str(selector.DEFAULT_REPLAY_RECEIPT.relative_to(ROOT)),
        str(selector.DEFAULT_SELECTION_RECEIPT.relative_to(ROOT)),
    ]:
        errors.append("terminal receipt paths differ from the frozen selector")
    required_outputs = contract["required_new_terminal_outputs"]
    if str(FINAL_RECEIPT.relative_to(ROOT)) not in required_outputs:
        errors.append("V2 final receipt is absent from terminal outputs")
    migration = json.loads(MIGRATION_CONTRACT.read_text(encoding="utf-8"))
    if migration.get("status") != "FROZEN_PRETERMINAL_MIGRATION":
        errors.append("V2 migration contract is not frozen")
    stale = migration["stale_v1"]
    if stale.get("authorized_as_current") is not False:
        errors.append("V1 final bundle is authorized as current")

    result = {
        "status": "PASS_PRETERMINAL_FINAL_VERIFIER_PREP" if not errors else "FAIL",
        "errors": errors,
        "mutable_baseline_hashes_verified": len(mutable_hashes),
        "protected_history_hashes_verified": len(protected_hashes),
        "terminal_inputs_declared": len(required_inputs),
        "terminal_outputs_declared": len(required_outputs),
        "terminal_surface_actions_declared": len(
            contract.get("terminal_surface_actions", [])
        ),
        "terminal_capsule_marker_targets_declared": len(
            contract.get("mutable_integration_targets", [])
        ),
        "terminal_claim_sentence_targets_declared": sum(
            len(contract.get("terminal_claim_binding_surfaces", {}).get(key, []))
            for key in ("english_sentence_targets", "chinese_sentence_targets")
        ),
        "terminal_metric_targets_declared": len(
            contract.get("terminal_claim_binding_surfaces", {}).get(
                "metric_targets", []
            )
        ),
        "verifier_sha256": sha256(Path(__file__)),
        "execution_contract_sha256": sha256(EXECUTION_CONTRACT),
        "migration_contract_sha256": sha256(MIGRATION_CONTRACT),
        "pathmnist_terminal_receipts_opened": False,
        "pathmnist_performance_artifacts_parsed": False,
        "submission_ready": False,
    }
    if write_receipt:
        atomic_json(PREPARATION_RECEIPT, result)
    if errors:
        raise RuntimeError("; ".join(errors))
    return result


def verify_terminal_control(
    replay_receipt: Path,
    route_receipt: Path,
    formal_output: Path,
) -> dict[str, Any]:
    """Validate terminal control before and after the first performance parse."""

    if not replay_receipt.is_file():
        return {
            "status": "WAITING_REPLAY_RECEIPT",
            "errors": ["V2 replay receipt is absent"],
            "replay_receipt_opened": False,
            "route_receipt_opened": False,
            "performance_artifacts_parsed": False,
        }
    replay = json.loads(replay_receipt.read_text(encoding="utf-8"))
    failures = selector.replay_control_failures(replay, formal_output)
    if failures:
        return {
            "status": "REPLAY_CONTROL_REJECTED",
            "errors": failures,
            "replay_receipt_opened": True,
            "route_receipt_opened": False,
            "performance_artifacts_parsed": False,
        }
    if not route_receipt.is_file():
        return {
            "status": "WAITING_ROUTE_RECEIPT",
            "errors": ["route receipt is absent after a valid replay receipt"],
            "replay_receipt_opened": True,
            "route_receipt_opened": False,
            "performance_artifacts_parsed": False,
        }

    # This call is the first permitted performance-bearing parse.  The selector
    # revalidates the complete replay control gate before opening manifest.json.
    expected = selector.select(replay_receipt, formal_output, None)
    observed = json.loads(route_receipt.read_text(encoding="utf-8"))
    comparable = (
        "status",
        "selected_route",
        "claim_decision",
        "pooling_allowed",
        "allowed_main_text_template",
        "allowed_numeric_fields",
        "required_surfaces",
        "eligible_seed_count",
        "all_pairs_eligible",
        "primary_point_gate_pass",
        "replay_control_failures",
        "performance_artifacts_parsed",
        "performance_endpoints_inspected_before_replay_pass",
        "replay_receipt_sha256",
        "selector_sha256",
        "selector_contract_sha256",
        "base_contract_sha256",
        "tifs_template_sha256",
        "v2_replay_amendment_sha256",
        "selector_frozen_before_terminal_receipt",
    )
    errors = [
        f"route receipt mismatch: {key}"
        for key in comparable
        if observed.get(key) != expected.get(key)
    ]
    route = observed.get("selected_route")
    if route == "R0_INVALID_OR_INCOMPLETE":
        errors.append("R0 cannot produce a submission-ready final bundle")
    return {
        "status": "PASS_TERMINAL_CONTROL" if not errors else "FAIL",
        "errors": errors,
        "selected_route": route,
        "claim_decision": observed.get("claim_decision"),
        "pooling_allowed": observed.get("pooling_allowed"),
        "replay_receipt_opened": True,
        "route_receipt_opened": True,
        "performance_artifacts_parsed": True,
        "replay_receipt_sha256": sha256(replay_receipt),
        "route_receipt_sha256": sha256(route_receipt),
    }


def verify_changed_surfaces_and_history(contract: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    changed: list[str] = []
    for relative, baseline in contract["preterminal_mutable_hashes"].items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"terminal mutable target missing: {relative}")
        elif sha256(path) == baseline:
            errors.append(f"terminal mutable target was not updated: {relative}")
        else:
            changed.append(relative)
    preserved: list[str] = []
    for relative, baseline in contract["protected_history_hashes"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != baseline:
            errors.append(f"protected history changed: {relative}")
        else:
            preserved.append(relative)
    return {
        "errors": errors,
        "changed_mutable_targets": changed,
        "preserved_history_targets": preserved,
    }


def verify_route_markers(
    contract: dict[str, Any], selected_route: str, *, root: Path = ROOT
) -> dict[str, Any]:
    """Require one exact selected-route marker on every mutable surface."""

    prefix = contract.get("required_terminal_route_marker_prefix")
    expected = f"{prefix}{selected_route}"
    errors: list[str] = []
    verified: list[str] = []
    for relative in contract["mutable_integration_targets"]:
        path = root / relative
        if not path.is_file():
            errors.append(f"route-marker target missing: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        if expected not in text:
            errors.append(f"selected route marker missing: {relative}")
            continue
        other_markers = [
            f"{prefix}{route}"
            for route in (
                "R0_INVALID_OR_INCOMPLETE",
                "R1_INSUFFICIENT_ELIGIBLE_VICTIMS",
                "R2_ELIGIBLE_PRIMARY_GATE_FAIL",
                "R3_ELIGIBLE_PRIMARY_GATE_PASS",
            )
            if route != selected_route and f"{prefix}{route}" in text
        ]
        if other_markers:
            errors.append(f"conflicting terminal route marker: {relative}")
            continue
        verified.append(relative)
    return {
        "errors": errors,
        "expected_marker": expected,
        "verified_targets": verified,
    }


def metric_marker(method: str, summary: dict[str, Any]) -> str:
    return (
        f"PATHMNIST_METRIC={method}"
        f"|TPR={summary['pooled_tpr_pct']:.2f}"
        f"|ALL_CLEAN_FPR={summary['pooled_all_clean_fpr_pct']:.2f}"
        f"|TARGET_CLEAN_FPR={summary['pooled_target_clean_fpr_pct']:.2f}"
    )


def verify_capsule_surface_binding(
    contract: dict[str, Any],
    selected_route: str,
    capsule_path: Path,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Bind every surface to one capsule and its route-authorized claims."""

    errors: list[str] = []
    if not capsule_path.is_file():
        return {
            "errors": ["terminal claim capsule is missing"],
            "capsule_sha256": None,
            "capsule_marked_targets": [],
            "sentence_targets_verified": [],
            "metric_targets_verified": [],
        }
    capsule = json.loads(capsule_path.read_text(encoding="utf-8"))
    if capsule.get("status") != "PASS":
        errors.append("terminal claim capsule is not PASS")
    if capsule.get("selected_route") != selected_route:
        errors.append("terminal claim capsule route mismatch")
    capsule_sha = sha256(capsule_path)
    marker = f"{contract['required_terminal_capsule_hash_marker_prefix']}{capsule_sha}"
    marked: list[str] = []
    for relative in contract["mutable_integration_targets"]:
        path = root / relative
        if path.is_file() and marker in path.read_text(encoding="utf-8"):
            marked.append(relative)
        else:
            errors.append(f"claim capsule hash marker missing: {relative}")

    bindings = contract["terminal_claim_binding_surfaces"]
    sentence_targets: list[str] = []
    for relative in bindings["english_sentence_targets"]:
        path = root / relative
        if path.is_file() and normalized(capsule["main_text_en"]) in normalized(
            path.read_text(encoding="utf-8")
        ):
            sentence_targets.append(relative)
        else:
            errors.append(f"English terminal claim sentence missing: {relative}")
    for relative in bindings["chinese_sentence_targets"]:
        path = root / relative
        if path.is_file() and normalized(capsule["main_text_zh"]) in normalized(
            path.read_text(encoding="utf-8")
        ):
            sentence_targets.append(relative)
        else:
            errors.append(f"Chinese terminal claim sentence missing: {relative}")

    metric_targets: list[str] = []
    summaries = capsule.get("detector_summaries")
    if summaries is not None:
        required_markers = [
            metric_marker(method, summary) for method, summary in summaries.items()
        ]
        for relative in bindings["metric_targets"]:
            path = root / relative
            text = path.read_text(encoding="utf-8") if path.is_file() else ""
            missing = [value for value in required_markers if value not in text]
            if missing:
                errors.append(f"terminal metric markers missing: {relative}")
            else:
                metric_targets.append(relative)
    else:
        for relative in bindings["metric_targets"]:
            path = root / relative
            text = path.read_text(encoding="utf-8") if path.is_file() else ""
            if contract["required_terminal_metric_marker_prefix"] in text:
                errors.append(f"R1 surface contains pooled detector marker: {relative}")
            else:
                metric_targets.append(relative)
    return {
        "errors": errors,
        "capsule_sha256": capsule_sha,
        "capsule_marked_targets": marked,
        "sentence_targets_verified": sentence_targets,
        "metric_targets_verified": metric_targets,
    }


def verify_c27_decision(route: dict[str, Any]) -> dict[str, Any]:
    matrix = PROGRAM / "FINAL_CLAIM_MATRIX.csv"
    with matrix.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row.get("claim_id") == "C27"]
    errors: list[str] = []
    if len(matches) != 1:
        errors.append("FINAL_CLAIM_MATRIX does not contain exactly one C27 row")
        decision = None
    else:
        decision = matches[0].get("decision")
        if decision != route.get("claim_decision"):
            errors.append("C27 decision does not match the terminal route receipt")
    return {"errors": errors, "c27_decision": decision}


def terminal_run(
    *,
    replay_receipt: Path = selector.DEFAULT_REPLAY_RECEIPT,
    route_receipt: Path = selector.DEFAULT_SELECTION_RECEIPT,
    formal_output: Path = FORMAL_OUTPUT,
    final_receipt: Path = FINAL_RECEIPT,
) -> dict[str, Any]:
    """Run only after terminal integration; never use this as a preterminal poll."""

    control = verify_terminal_control(replay_receipt, route_receipt, formal_output)
    if control["status"] != "PASS_TERMINAL_CONTROL":
        raise RuntimeError("; ".join(control["errors"]))
    contract = load_execution_contract()
    surface = verify_changed_surfaces_and_history(contract)
    route = json.loads(route_receipt.read_text(encoding="utf-8"))
    markers = verify_route_markers(contract, control["selected_route"])
    capsule_relative = next(
        relative
        for relative in contract["required_new_terminal_outputs"]
        if "terminal_claim_capsule" in relative
    )
    capsule_binding = verify_capsule_surface_binding(
        contract, control["selected_route"], ROOT / capsule_relative
    )
    c27 = verify_c27_decision(route)
    errors = (
        surface["errors"]
        + markers["errors"]
        + capsule_binding["errors"]
        + c27["errors"]
    )

    terminal_receipts: dict[str, dict[str, Any]] = {}
    for relative in contract["required_new_terminal_outputs"]:
        path = ROOT / relative
        if path == final_receipt:
            continue
        if not path.is_file():
            errors.append(f"required terminal receipt missing: {relative}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        terminal_receipts[relative] = {
            "status": payload.get("status"),
            "sha256": sha256(path),
        }
        if not str(payload.get("status", "")).startswith("PASS"):
            errors.append(f"terminal receipt is not PASS: {relative}")

    result = {
        "status": "PASS_TERMINAL_BUNDLE" if not errors else "FAIL",
        "errors": errors,
        "selected_route": control["selected_route"],
        "claim_decision": control["claim_decision"],
        "pooling_allowed": control["pooling_allowed"],
        "terminal_control": control,
        "changed_mutable_targets": surface["changed_mutable_targets"],
        "preserved_history_targets": surface["preserved_history_targets"],
        "terminal_route_marker": markers["expected_marker"],
        "route_marked_targets": markers["verified_targets"],
        "claim_capsule_sha256": capsule_binding["capsule_sha256"],
        "capsule_marked_targets": capsule_binding["capsule_marked_targets"],
        "claim_sentence_targets": capsule_binding["sentence_targets_verified"],
        "metric_marker_targets": capsule_binding["metric_targets_verified"],
        "c27_decision": c27["c27_decision"],
        "terminal_receipts": terminal_receipts,
        "final_verifier_sha256": sha256(Path(__file__)),
        "execution_contract_sha256": sha256(EXECUTION_CONTRACT),
        "submission_ready": False,
        "submission_ready_reason": (
            "mechanical bundle closure does not infer author-controlled licences, "
            "accessions, identities, funding, conflicts, CRediT or AI disclosure"
        ),
    }
    atomic_json(final_receipt, result)
    if errors:
        raise RuntimeError("; ".join(errors))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("preterminal-self-check", "terminal"), required=True
    )
    args = parser.parse_args()
    result = (
        preterminal_self_check()
        if args.mode == "preterminal-self-check"
        else terminal_run()
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
