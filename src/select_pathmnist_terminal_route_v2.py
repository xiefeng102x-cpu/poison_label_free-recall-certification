#!/usr/bin/env python3
"""Mechanically select the frozen PathMNIST R0--R3 terminal route.

No performance artifact is parsed until the V2 replay receipt proves that the
unchanged verifier passed over the complete formal block.  An invalid replay
selects R0 without opening the formal manifest as JSON.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "results/top_journal_program_v1"
DEFAULT_OUTPUT = ROOT / "results/pathmnist_official_weight_detector_confirmation_v1"
DEFAULT_REPLAY_RECEIPT = PROGRAM / (
    "pathmnist_terminal_staged_replay_v2_receipt_2026-10-04.json"
)
DEFAULT_SELECTION_RECEIPT = PROGRAM / (
    "pathmnist_terminal_route_selection_v2_receipt.json"
)
BASE_CONTRACT = PROGRAM / "pathmnist_terminal_claim_integration_contract_v1.json"
TIFS_TEMPLATE = PROGRAM / "tifs_pathmnist_terminal_integration_template_v1.json"
V2_AMENDMENT = PROGRAM / (
    "pathmnist_terminal_replay_dataset_lifetime_amendment_2026-10-04.json"
)
SELECTOR_CONTRACT = PROGRAM / "pathmnist_terminal_route_selector_contract_v1.json"
ACQUISITION_RECEIPT = PROGRAM / "pathmnist_acquisition_receipt.json"
FROZEN_VERIFIER = ROOT / "src/verify_pathmnist_official_weight_detector_confirmation.py"
V2_WRAPPER = ROOT / "src/run_pathmnist_terminal_staged_replay_v2.py"
TOP_LEVEL_FILES = (
    "manifest.json",
    "seed_receipts.json",
    "detector_metrics.csv",
    "scores.csv",
)
SEEDS = [103, 104, 105, 106, 107]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_control_contracts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = json.loads(BASE_CONTRACT.read_text(encoding="utf-8"))
    template = json.loads(TIFS_TEMPLATE.read_text(encoding="utf-8"))
    selector = json.loads(SELECTOR_CONTRACT.read_text(encoding="utf-8"))
    if selector.get("status") != "FROZEN_BEFORE_V2_TERMINAL_RECEIPT":
        raise RuntimeError("route-selector contract is not frozen")
    expected = {
        "selector_sha256": sha256(Path(__file__).resolve()),
        "base_contract_sha256": sha256(BASE_CONTRACT),
        "tifs_template_sha256": sha256(TIFS_TEMPLATE),
        "v2_replay_amendment_sha256": sha256(V2_AMENDMENT),
    }
    for key, value in expected.items():
        if selector.get(key) != value:
            raise RuntimeError(f"route-selector binding mismatch: {key}")
    if selector.get("performance_endpoints_inspected") is not False:
        raise RuntimeError("route-selector contract does not preserve sealing")
    if selector.get("scientific_decision_rules_changed") is not False:
        raise RuntimeError("route-selector contract changes decision rules")
    return contract, template, selector


def replay_control_failures(replay: dict[str, Any], output: Path) -> list[str]:
    failures: list[str] = []
    expected_scalars = {
        "status": "PASS",
        "formal_evidence": True,
        "failures": [],
        "verified_seeds": SEEDS,
        "verified_method_rows": 25,
        "path_only_staging": True,
        "dataset_bound_before_replay": True,
        "dataset_reopened_after_replay": False,
        "evidence_files_modified": False,
        "performance_endpoints_inspected_before_replay": False,
        "formal_output": str(output.resolve()),
        "frozen_verifier_sha256": sha256(FROZEN_VERIFIER),
        "staging_wrapper_v2_sha256": sha256(V2_WRAPPER),
        "amendment_sha256": sha256(V2_AMENDMENT),
    }
    for key, expected in expected_scalars.items():
        if replay.get(key) != expected:
            failures.append(f"replay control mismatch: {key}")

    acquisition = json.loads(ACQUISITION_RECEIPT.read_text(encoding="utf-8"))
    if replay.get("dataset_sha256") != acquisition.get("sha256"):
        failures.append("replay dataset SHA-256 mismatch")
    if replay.get("dataset_byte_count") != acquisition.get("byte_count"):
        failures.append("replay dataset byte-count mismatch")
    if replay.get("acquisition_receipt_sha256") != sha256(ACQUISITION_RECEIPT):
        failures.append("replay acquisition-receipt binding mismatch")

    evidence = replay.get("evidence_hashes")
    if not isinstance(evidence, dict) or set(evidence) != set(TOP_LEVEL_FILES):
        failures.append("replay evidence-hash inventory mismatch")
    else:
        for name in TOP_LEVEL_FILES:
            path = output / name
            if not path.is_file() or evidence.get(name) != sha256(path):
                failures.append(f"replay evidence hash mismatch: {name}")
    return failures


def route_record(template: dict[str, Any], route: str) -> dict[str, Any]:
    matches = [item for item in template["ordered_routes"] if item["route"] == route]
    if len(matches) != 1:
        raise RuntimeError(f"terminal template lacks unique route {route}")
    return matches[0]


def select(
    replay_receipt: Path,
    output: Path,
    selection_receipt: Path | None = None,
) -> dict[str, Any]:
    contract, template, selector_contract = load_control_contracts()
    replay = json.loads(replay_receipt.read_text(encoding="utf-8"))
    control_failures = replay_control_failures(replay, output)
    performance_artifacts_parsed = False

    if control_failures:
        route = "R0_INVALID_OR_INCOMPLETE"
        eligible_seed_count = None
        primary_point_gate_pass = None
        all_pairs_eligible = None
    else:
        # This is the first performance-bearing parse.  It is reachable only
        # after a complete, hash-bound PASS receipt from the frozen verifier.
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        performance_artifacts_parsed = True
        if manifest.get("status") != "COMPLETE_FORMAL":
            raise RuntimeError("PASS receipt is bound to a nonterminal manifest")
        if manifest.get("completed_seeds") != SEEDS:
            raise RuntimeError("formal manifest seed block mismatch")
        eligibility = manifest.get("victim_eligibility_by_seed", {})
        if set(eligibility) != {str(seed) for seed in SEEDS} or not all(
            isinstance(value, bool) for value in eligibility.values()
        ):
            raise RuntimeError("formal manifest eligibility map is invalid")
        eligible_seed_count = sum(bool(value) for value in eligibility.values())
        all_pairs_eligible = manifest.get("all_pairs_eligible")
        primary_point_gate_pass = manifest.get("primary_point_gate_pass")
        if all_pairs_eligible is not (eligible_seed_count == 5):
            raise RuntimeError("formal manifest all-pairs gate mismatch")
        if not isinstance(primary_point_gate_pass, bool):
            raise RuntimeError("formal manifest primary gate is not Boolean")
        if eligible_seed_count < 5:
            if primary_point_gate_pass is not False:
                raise RuntimeError("ineligible block cannot pass the primary gate")
            route = "R1_INSUFFICIENT_ELIGIBLE_VICTIMS"
        elif primary_point_gate_pass:
            route = "R3_ELIGIBLE_PRIMARY_GATE_PASS"
        else:
            route = "R2_ELIGIBLE_PRIMARY_GATE_FAIL"

    ordered = [item["route"] for item in contract["ordered_routes"]]
    if ordered != [
        "R0_INVALID_OR_INCOMPLETE",
        "R1_INSUFFICIENT_ELIGIBLE_VICTIMS",
        "R2_ELIGIBLE_PRIMARY_GATE_FAIL",
        "R3_ELIGIBLE_PRIMARY_GATE_PASS",
    ]:
        raise RuntimeError("base terminal route order drift")
    selected = route_record(template, route)
    result: dict[str, Any] = {
        "status": "TERMINAL_ROUTE_SELECTED",
        "selected_route": route,
        "claim_decision": selected["claim_decision"],
        "pooling_allowed": selected["pooling_allowed"],
        "allowed_main_text_template": selected["main_text"],
        "allowed_numeric_fields": selected["allowed_numeric_fields"],
        "required_surfaces": selected["required_surfaces"],
        "eligible_seed_count": eligible_seed_count,
        "all_pairs_eligible": all_pairs_eligible,
        "primary_point_gate_pass": primary_point_gate_pass,
        "replay_control_failures": control_failures,
        "performance_artifacts_parsed": performance_artifacts_parsed,
        "performance_endpoints_inspected_before_replay_pass": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "replay_receipt_sha256": sha256(replay_receipt),
        "selector_sha256": sha256(Path(__file__).resolve()),
        "selector_contract_sha256": sha256(SELECTOR_CONTRACT),
        "base_contract_sha256": sha256(BASE_CONTRACT),
        "tifs_template_sha256": sha256(TIFS_TEMPLATE),
        "v2_replay_amendment_sha256": sha256(V2_AMENDMENT),
        "selector_frozen_before_terminal_receipt": selector_contract[
            "frozen_before_v2_terminal_receipt"
        ],
    }
    if selection_receipt is not None:
        atomic_json(selection_receipt, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-receipt", type=Path, default=DEFAULT_REPLAY_RECEIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection-receipt", type=Path, default=DEFAULT_SELECTION_RECEIPT)
    args = parser.parse_args()
    result = select(args.replay_receipt, args.output, args.selection_receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["selected_route"] != "R0_INVALID_OR_INCOMPLETE" else 1)


if __name__ == "__main__":
    main()
