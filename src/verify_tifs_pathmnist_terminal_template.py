#!/usr/bin/env python3
"""Verify the pre-outcome TIFS PathMNIST terminal-integration template."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "results/top_journal_program_v1/tifs_pathmnist_terminal_integration_template_v1.json"
BASE_CONTRACT = ROOT / "results/top_journal_program_v1/pathmnist_terminal_claim_integration_contract_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run() -> dict[str, object]:
    errors: list[str] = []
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    if template.get("status") != "FROZEN_BEFORE_CORRECTED_TERMINAL_REPLAY":
        errors.append("template is not frozen pre-outcome")
    if template.get("performance_endpoints_inspected") is not False:
        errors.append("template does not preserve endpoint sealing")
    if template.get("scientific_decision_rules_changed") is not False:
        errors.append("template changes scientific decision rules")
    if template.get("base_contract_sha256") != sha256(BASE_CONTRACT):
        errors.append("base terminal contract hash mismatch")

    routes = template.get("ordered_routes", [])
    expected_routes = [
        "R0_INVALID_OR_INCOMPLETE",
        "R1_INSUFFICIENT_ELIGIBLE_VICTIMS",
        "R2_ELIGIBLE_PRIMARY_GATE_FAIL",
        "R3_ELIGIBLE_PRIMARY_GATE_PASS",
    ]
    if [item.get("route") for item in routes] != expected_routes:
        errors.append("terminal route order mismatch")
    if [item.get("pooling_allowed") for item in routes] != [False, False, True, True]:
        errors.append("route pooling permissions mismatch")

    for target in template.get("source_targets", []):
        path = ROOT / target["path"]
        if not path.is_file():
            errors.append(f"missing integration target: {target['path']}")
            continue
        source = path.read_text(encoding="utf-8")
        for anchor in target.get("anchors", []):
            if anchor not in source:
                errors.append(f"missing anchor in {target['path']}: {anchor}")

    main = (ROOT / "manuscript_tifs_en/main_tifs.tex").read_text(encoding="utf-8")
    for unit in re.split(r"\\\\|\n\s*\n", main):
        if re.search(r"PathMNIST", unit, flags=re.IGNORECASE) and re.search(
            r"\b(?:TPR|FPR|AUC)\b", unit
        ):
            errors.append("preterminal TIFS source contains a PathMNIST detector endpoint")
            break
    return {
        "status": "PASS_PRETERMINAL_TEMPLATE" if not errors else "FAIL",
        "errors": errors,
        "routes": expected_routes,
        "source_targets": len(template.get("source_targets", [])),
        "global_prohibitions": len(template.get("global_prohibitions", [])),
        "mandatory_post_selection_actions": len(template.get("mandatory_post_selection_actions", [])),
        "template_sha256": sha256(TEMPLATE),
        "base_contract_sha256": sha256(BASE_CONTRACT),
        "performance_endpoints_inspected": False,
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS_PRETERMINAL_TEMPLATE" else 1)
