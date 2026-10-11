#!/usr/bin/env python3
"""Verify outcome-blind coverage for PathMNIST terminal integration V2.

The verifier reads control contracts and filenames only.  It does not open the
V2 replay receipt, route receipt, formal manifest, or any performance table.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "results/top_journal_program_v1"
CONTRACT = PROGRAM / "pathmnist_terminal_integration_execution_contract_v2.json"
BASE = PROGRAM / "pathmnist_terminal_claim_integration_contract_v1.json"
TEMPLATE = PROGRAM / "tifs_pathmnist_terminal_integration_template_v1.json"
RECEIPT = PROGRAM / (
    "pathmnist_terminal_integration_execution_readiness_v2_2026-10-04.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.split())


def run(*, contract_path: Path = CONTRACT, write_receipt: bool = True) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    base = json.loads(BASE.read_text(encoding="utf-8"))
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    errors: list[str] = []

    if contract.get("status") != "FROZEN_PRETERMINAL_EXECUTION":
        errors.append("execution contract is not frozen preterminal")
    if contract.get("performance_endpoints_inspected") is not False:
        errors.append("execution contract does not preserve endpoint sealing")
    if contract.get("scientific_decision_rules_changed") is not False:
        errors.append("execution contract changes scientific decision rules")

    for relative, expected in contract.get("bindings", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"control binding drift: {relative}")

    expected_routes = [item["route"] for item in base["ordered_routes"]]
    observed_routes = [item.get("route") for item in contract.get("ordered_routes", [])]
    if observed_routes != expected_routes:
        errors.append("execution route order differs from frozen base contract")
    expected_pooling = [
        item["efficacy_pooling_allowed"] for item in base["ordered_routes"]
    ]
    observed_pooling = [
        item.get("pooling_allowed") for item in contract.get("ordered_routes", [])
    ]
    if observed_pooling != expected_pooling:
        errors.append("execution pooling permissions differ from frozen base contract")

    mutable = contract.get("mutable_integration_targets", [])
    protected = contract.get("protected_history", [])
    mutable_hashes = contract.get("preterminal_mutable_hashes", {})
    protected_hashes = contract.get("protected_history_hashes", {})
    surface_actions = contract.get("terminal_surface_actions", [])
    marker_prefix = contract.get("required_terminal_route_marker_prefix")
    capsule_marker_prefix = contract.get(
        "required_terminal_capsule_hash_marker_prefix"
    )
    metric_marker_prefix = contract.get("required_terminal_metric_marker_prefix")
    binding_surfaces = contract.get("terminal_claim_binding_surfaces", {})
    if len(mutable) != len(set(mutable)):
        errors.append("duplicate mutable integration target")
    if set(mutable) & set(protected):
        errors.append("a protected history artifact is listed as mutable")
    if set(mutable_hashes) != set(mutable):
        errors.append("preterminal mutable-hash inventory does not match targets")
    if set(protected_hashes) != set(protected):
        errors.append("protected-history hash inventory does not match targets")
    action_paths = [item.get("path") for item in surface_actions]
    if len(action_paths) != len(set(action_paths)):
        errors.append("duplicate terminal surface action")
    if set(action_paths) != set(mutable):
        errors.append("terminal surface actions do not cover every mutable target")
    if marker_prefix != "PATHMNIST_TERMINAL_ROUTE=":
        errors.append("terminal route marker prefix is not frozen")
    if capsule_marker_prefix != "PATHMNIST_CLAIM_CAPSULE_SHA256=":
        errors.append("terminal capsule marker prefix is not frozen")
    if metric_marker_prefix != "PATHMNIST_METRIC=":
        errors.append("terminal metric marker prefix is not frozen")
    binding_paths = {
        path
        for paths in binding_surfaces.values()
        for path in paths
    }
    if not binding_paths.issubset(set(mutable)):
        errors.append("terminal claim-binding surfaces are outside mutable targets")
    if [len(binding_surfaces.get(key, [])) for key in (
        "english_sentence_targets",
        "chinese_sentence_targets",
        "metric_targets",
    )] != [2, 1, 2]:
        errors.append("terminal claim-binding surface counts are incomplete")
    missing = [relative for relative in mutable + protected if not (ROOT / relative).is_file()]
    if missing:
        errors.append(f"integration inventory path missing: {missing}")
    for relative, expected in mutable_hashes.items():
        path = ROOT / relative
        if path.is_file() and sha256(path) != expected:
            errors.append(f"preterminal mutable baseline drift: {relative}")
    for relative, expected in protected_hashes.items():
        path = ROOT / relative
        if path.is_file() and sha256(path) != expected:
            errors.append(f"protected history drift: {relative}")
    anchor_checks = 0
    marker_absence_checks = 0
    capsule_marker_absence_checks = 0
    metric_marker_absence_checks = 0
    for item in surface_actions:
        relative = item.get("path")
        anchor = item.get("preterminal_anchor")
        action = item.get("terminal_action")
        if not isinstance(anchor, str) or not anchor:
            errors.append(f"missing preterminal anchor: {relative}")
            continue
        if not isinstance(action, str) or not action:
            errors.append(f"missing terminal action: {relative}")
            continue
        path = ROOT / str(relative)
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            if normalized(anchor) not in normalized(text):
                errors.append(f"preterminal surface anchor missing: {relative}")
            if marker_prefix in text:
                errors.append(f"terminal route marker leaked preterminally: {relative}")
            marker_absence_checks += 1
            if capsule_marker_prefix in text:
                errors.append(f"terminal capsule marker leaked preterminally: {relative}")
            capsule_marker_absence_checks += 1
            if metric_marker_prefix in text:
                errors.append(f"terminal metric marker leaked preterminally: {relative}")
            metric_marker_absence_checks += 1
        anchor_checks += 1

    base_targets = set(base["mandatory_integration_targets"])
    legacy_receipt = "results/top_journal_program_v1/final_bundle_receipt.json"
    expected_existing_coverage = base_targets - {legacy_receipt}
    expected_existing_coverage.update(
        target["path"] for target in template["source_targets"]
    )
    if not expected_existing_coverage.issubset(set(mutable)):
        absent = sorted(expected_existing_coverage - set(mutable))
        errors.append(f"base/template integration targets not covered: {absent}")
    if legacy_receipt not in protected:
        errors.append("legacy final-bundle receipt is not protected history")
    if "src/verify_top_journal_final_bundle.py" not in protected:
        errors.append("legacy final-bundle verifier is not protected history")

    required_new = contract.get("required_new_terminal_outputs", [])
    if len(required_new) != len(set(required_new)):
        errors.append("duplicate required terminal output")
    if legacy_receipt in required_new:
        errors.append("legacy final-bundle receipt is reused as a V2 output")
    if not any(path.endswith("final_bundle_v2_receipt_2026-10-04.json") for path in required_new):
        errors.append("new V2 final-bundle receipt is not required")
    integration_receipt = (
        "results/top_journal_program_v1/"
        "pathmnist_terminal_surface_integration_v2_2026-10-04.json"
    )
    if integration_receipt not in required_new:
        errors.append("terminal surface-integration receipt is not required")
    integrator = "src/integrate_pathmnist_terminal_surfaces_v2.py"
    if integrator not in contract.get("bindings", {}):
        errors.append("terminal surface integrator is not hash-bound")

    if len(contract.get("mandatory_sequence", [])) != 12:
        errors.append("terminal execution sequence is incomplete")
    if len(contract.get("route_independent_invariants", [])) != 8:
        errors.append("route-independent invariant set is incomplete")

    result = {
        "status": "PASS_PRETERMINAL_EXECUTION" if not errors else "FAIL",
        "errors": errors,
        "routes_checked": len(expected_routes),
        "mutable_targets_covered": len(mutable),
        "preterminal_mutable_hashes_verified": len(mutable_hashes),
        "protected_history_artifacts": len(protected),
        "protected_history_hashes_verified": len(protected_hashes),
        "terminal_surface_actions_verified": len(surface_actions),
        "preterminal_surface_anchors_verified": anchor_checks,
        "preterminal_route_marker_absence_checks": marker_absence_checks,
        "preterminal_capsule_marker_absence_checks": capsule_marker_absence_checks,
        "preterminal_metric_marker_absence_checks": metric_marker_absence_checks,
        "required_terminal_inputs": len(contract.get("required_terminal_inputs", [])),
        "required_new_terminal_outputs": len(required_new),
        "mandatory_terminal_steps": len(contract.get("mandatory_sequence", [])),
        "route_independent_invariants": len(
            contract.get("route_independent_invariants", [])
        ),
        "contract_sha256": sha256(contract_path),
        "performance_endpoints_inspected": False,
        "pathmnist_result_artifacts_read": False,
        "submission_ready": False,
    }
    if write_receipt:
        RECEIPT.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if errors:
        raise RuntimeError("; ".join(errors))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
