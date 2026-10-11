#!/usr/bin/env python3
"""Verify the outcome-blind migration contract for the final bundle V2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import run_full_source_regression_terminal_v2 as terminal_regression
import build_terminal_release_profile_v2 as terminal_release_profile
import run_terminal_reviewer_package_v2 as terminal_reviewer_package
import prepare_tifs_terminal_visual_qa_v2 as tifs_terminal_visual_qa
import integrate_pathmnist_terminal_surfaces_v2 as terminal_surface_integrator


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "results/top_journal_program_v1"
CONTRACT = PROGRAM / "final_bundle_v2_migration_contract_2026-10-04.json"
RECEIPT = PROGRAM / "final_bundle_v2_migration_readiness_preterminal_2026-10-04.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(*, write_receipt: bool = True) -> dict[str, Any]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    errors: list[str] = []
    if contract.get("status") != "FROZEN_PRETERMINAL_MIGRATION":
        errors.append("migration contract is not frozen")
    if contract.get("performance_endpoints_inspected") is not False:
        errors.append("migration contract does not preserve endpoint sealing")
    if contract.get("scientific_decision_rules_changed") is not False:
        errors.append("migration contract changes scientific decisions")

    stale = contract["stale_v1"]
    for key in ("verifier", "receipt"):
        path = ROOT / stale[key]
        if not path.is_file() or sha256(path) != stale[f"{key}_sha256"]:
            errors.append(f"stale V1 {key} history drift")
    if stale.get("authorized_as_current") is not False:
        errors.append("stale V1 is incorrectly authorized as current")

    for relative, expected in contract["frozen_sources"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"frozen migration source drift: {relative}")

    missing = [
        relative
        for relative in contract["mandatory_v2_inventory_additions"]
        if not (ROOT / relative).is_file()
    ]
    if missing:
        errors.append(f"missing V2 inventory additions: {missing}")

    bilingual = json.loads(
        (PROGRAM / "bilingual_manuscript_receipt.json").read_text(encoding="utf-8")
    )
    claims = json.loads(
        (PROGRAM / "manuscript_claim_consistency_receipt_2026-10-02.json").read_text(
            encoding="utf-8"
        )
    )
    tifs = json.loads(
        (PROGRAM / "tifs_main_preterminal_receipt_2026-10-03.json").read_text(
            encoding="utf-8"
        )
    )
    theory = json.loads(
        (PROGRAM / "theory_claim_surface_consistency_receipt_2026-10-04.json").read_text(
            encoding="utf-8"
        )
    )
    threat = json.loads(
        (PROGRAM / "threat_model_surface_consistency_receipt_2026-10-04.json").read_text(
            encoding="utf-8"
        )
    )
    integration = json.loads(
        (
            PROGRAM
            / "pathmnist_terminal_integration_execution_readiness_v2_2026-10-04.json"
        ).read_text(encoding="utf-8")
    )
    final_v2_prep = json.loads(
        (PROGRAM / "final_bundle_v2_verifier_preparation_2026-10-04.json").read_text(
            encoding="utf-8"
        )
    )
    tifs_terminal_prep = json.loads(
        (PROGRAM / "tifs_terminal_verifier_preparation_2026-10-04.json").read_text(
            encoding="utf-8"
        )
    )
    terminal_regression_prep = terminal_regression.preterminal_self_check()
    terminal_release_profile_prep = terminal_release_profile.preterminal_self_check()
    terminal_reviewer_package_prep = terminal_reviewer_package.preterminal_self_check()
    tifs_terminal_visual_qa_prep = tifs_terminal_visual_qa.preterminal_self_check()
    terminal_surface_integrator_prep = terminal_surface_integrator.preterminal_self_check()
    current = contract["current_preterminal_contract"]
    observed = {
        "chinese_pages": bilingual["chinese"]["pages"],
        "english_pages": bilingual["english"]["pages"],
        "bilingual_claim_configuration_tokens": claims[
            "required_bilingual_tokens_checked"
        ],
        "frozen_configuration_tokens": claims[
            "frozen_configuration_tokens_checked"
        ],
        "documentation_consistency_tokens": claims[
            "documentation_consistency_tokens_checked"
        ],
        "tifs_main_pages": tifs["page_count"],
        "tifs_supplement_pages": tifs["supplement_page_count"],
        "tifs_configuration_protocols": tifs[
            "configuration_ledger_protocol_count"
        ],
        "tifs_configuration_assertions": tifs[
            "configuration_ledger_assertion_count"
        ],
        "theory_propositions_per_surface": theory["propositions_per_surface"],
        "theory_assumption_nonimplication_tokens": theory[
            "assumption_and_nonimplication_tokens_checked"
        ],
        "theory_hostile_audit_interfaces": theory[
            "hostile_audit_interfaces_checked"
        ],
        "theory_forbidden_positive_forms": theory[
            "forbidden_positive_forms_checked"
        ],
        "threat_model_surfaces": threat["surfaces_checked"],
        "threat_model_boundary_tokens": threat[
            "threat_model_and_boundary_tokens_checked"
        ],
        "threat_model_privilege_rows": threat["privilege_ledger_rows_checked"],
        "threat_model_forbidden_positive_forms": threat[
            "forbidden_positive_forms_checked"
        ],
        "terminal_integration_mutable_targets": integration[
            "mutable_targets_covered"
        ],
        "terminal_integration_mutable_hashes": integration[
            "preterminal_mutable_hashes_verified"
        ],
        "terminal_integration_protected_history": integration[
            "protected_history_artifacts"
        ],
        "terminal_integration_protected_hashes": integration[
            "protected_history_hashes_verified"
        ],
        "terminal_integration_new_outputs": integration[
            "required_new_terminal_outputs"
        ],
        "terminal_integration_steps": integration["mandatory_terminal_steps"],
        "terminal_integration_surface_actions": integration[
            "terminal_surface_actions_verified"
        ],
        "terminal_integration_surface_anchors": integration[
            "preterminal_surface_anchors_verified"
        ],
        "terminal_integration_route_marker_absence": integration[
            "preterminal_route_marker_absence_checks"
        ],
        "terminal_integration_capsule_marker_absence": integration[
            "preterminal_capsule_marker_absence_checks"
        ],
        "terminal_integration_metric_marker_absence": integration[
            "preterminal_metric_marker_absence_checks"
        ],
        "final_v2_mutable_baseline_hashes": final_v2_prep[
            "mutable_baseline_hashes_verified"
        ],
        "final_v2_protected_history_hashes": final_v2_prep[
            "protected_history_hashes_verified"
        ],
        "final_v2_terminal_inputs": final_v2_prep["terminal_inputs_declared"],
        "final_v2_terminal_outputs": final_v2_prep["terminal_outputs_declared"],
        "final_v2_surface_actions": final_v2_prep[
            "terminal_surface_actions_declared"
        ],
        "final_v2_capsule_marker_targets": final_v2_prep[
            "terminal_capsule_marker_targets_declared"
        ],
        "final_v2_claim_sentence_targets": final_v2_prep[
            "terminal_claim_sentence_targets_declared"
        ],
        "final_v2_metric_targets": final_v2_prep[
            "terminal_metric_targets_declared"
        ],
        "tifs_terminal_configuration_tokens": tifs_terminal_prep[
            "configuration_tokens_checked"
        ],
        "tifs_terminal_frozen_protocol_hashes": tifs_terminal_prep[
            "frozen_protocol_hashes_checked"
        ],
        "terminal_regression_source_inventory_files": terminal_regression_prep[
            "source_inventory"
        ]["file_count"],
        "terminal_regression_output_declared": int(
            terminal_regression_prep["receipt_target"]
            in json.loads(
                (
                    PROGRAM
                    / "pathmnist_terminal_integration_execution_contract_v2.json"
                ).read_text(encoding="utf-8")
            )["required_new_terminal_outputs"]
        ),
        "terminal_release_profile_manifest_targets": len(
            terminal_release_profile_prep["terminal_manifest_targets"]
        ),
        "terminal_release_profile_author_blockers": len(
            terminal_release_profile_prep["author_blockers_preserved"]
        ),
        "terminal_release_profile_output_declared": int(
            terminal_release_profile_prep["terminal_receipt_target"]
            in json.loads(
                (
                    PROGRAM
                    / "pathmnist_terminal_integration_execution_contract_v2.json"
                ).read_text(encoding="utf-8")
            )["required_new_terminal_outputs"]
        ),
        "terminal_reviewer_package_expected_policy_skips": (
            terminal_reviewer_package_prep["expected_policy_skips"]
        ),
        "terminal_reviewer_package_allowed_skip_prefixes": len(
            terminal_reviewer_package_prep["allowed_skip_prefixes"]
        ),
        "terminal_reviewer_package_output_declared": int(
            terminal_reviewer_package_prep["terminal_receipt_target"]
            in json.loads(
                (
                    PROGRAM
                    / "pathmnist_terminal_integration_execution_contract_v2.json"
                ).read_text(encoding="utf-8")
            )["required_new_terminal_outputs"]
        ),
        "tifs_terminal_visual_qa_confirmation_tokens": int(
            bool(tifs_terminal_visual_qa_prep["confirmation_token_sha256"])
        ),
        "tifs_terminal_visual_qa_output_declared": int(
            tifs_terminal_visual_qa_prep["receipt_target"]
            in json.loads(
                (
                    PROGRAM
                    / "pathmnist_terminal_integration_execution_contract_v2.json"
                ).read_text(encoding="utf-8")
            )["required_new_terminal_outputs"]
        ),
        "terminal_surface_integrator_routes_simulated": (
            terminal_surface_integrator_prep["routes_simulated"]
        ),
        "terminal_surface_integrator_targets_per_route": (
            terminal_surface_integrator_prep["targets_per_route"]
        ),
        "terminal_surface_integrator_synthetic_renders": (
            terminal_surface_integrator_prep["synthetic_renders_checked"]
        ),
        "terminal_surface_integrator_output_declared": int(
            str(terminal_surface_integrator.DEFAULT_RECEIPT.relative_to(ROOT))
            in json.loads(
                (
                    PROGRAM
                    / "pathmnist_terminal_integration_execution_contract_v2.json"
                ).read_text(encoding="utf-8")
            )["required_new_terminal_outputs"]
        ),
    }
    for key, value in observed.items():
        if current.get(key) != value:
            errors.append(f"preterminal migration value drift: {key}")
    if claims.get("pathmnist_partial_artifacts_read") is not False:
        errors.append("claim verifier read a sealed PathMNIST endpoint")
    if tifs.get("pathmnist_partial_artifacts_read") is not False:
        errors.append("TIFS verifier read a sealed PathMNIST endpoint")
    if tifs.get("submission_ready") is not False:
        errors.append("preterminal TIFS receipt is incorrectly submission-ready")
    if theory.get("pathmnist_artifacts_read") is not False:
        errors.append("theory surface verifier read a PathMNIST artifact")
    if threat.get("pathmnist_artifacts_read") is not False:
        errors.append("threat-model surface verifier read a PathMNIST artifact")
    if integration.get("pathmnist_result_artifacts_read") is not False:
        errors.append("terminal integration verifier read a PathMNIST result artifact")
    if integration.get("submission_ready") is not False:
        errors.append("preterminal integration receipt is incorrectly submission-ready")
    if final_v2_prep.get("pathmnist_terminal_receipts_opened") is not False:
        errors.append("V2 final-verifier preparation opened a terminal receipt")
    if final_v2_prep.get("pathmnist_performance_artifacts_parsed") is not False:
        errors.append("V2 final-verifier preparation parsed PathMNIST performance")
    if final_v2_prep.get("submission_ready") is not False:
        errors.append("V2 final-verifier preparation is incorrectly submission-ready")
    if tifs_terminal_prep.get("pathmnist_terminal_receipts_opened") is not False:
        errors.append("TIFS terminal-verifier preparation opened a terminal receipt")
    if tifs_terminal_prep.get("pathmnist_performance_artifacts_parsed") is not False:
        errors.append("TIFS terminal-verifier preparation parsed PathMNIST performance")
    if tifs_terminal_prep.get("submission_ready") is not False:
        errors.append("TIFS terminal-verifier preparation is incorrectly submission-ready")
    if terminal_regression_prep.get("terminal_receipts_opened") is not False:
        errors.append("terminal regression preparation opened a terminal receipt")
    if terminal_regression_prep.get("pathmnist_performance_artifacts_parsed") is not False:
        errors.append("terminal regression preparation parsed PathMNIST performance")
    if terminal_regression_prep.get("submission_ready") is not False:
        errors.append("terminal regression preparation is incorrectly submission-ready")
    if terminal_release_profile_prep.get("terminal_receipts_opened") is not False:
        errors.append("terminal release-profile preparation opened a terminal receipt")
    if terminal_release_profile_prep.get("pathmnist_performance_artifacts_parsed") is not False:
        errors.append("terminal release-profile preparation parsed PathMNIST performance")
    if terminal_release_profile_prep.get("submission_ready") is not False:
        errors.append("terminal release-profile preparation is incorrectly submission-ready")
    if terminal_reviewer_package_prep.get("terminal_receipts_opened") is not False:
        errors.append("terminal reviewer-package preparation opened a terminal receipt")
    if terminal_reviewer_package_prep.get("pathmnist_performance_artifacts_parsed") is not False:
        errors.append("terminal reviewer-package preparation parsed PathMNIST performance")
    if terminal_reviewer_package_prep.get("package_built") is not False:
        errors.append("terminal reviewer-package preparation built a package preterminally")
    if terminal_reviewer_package_prep.get("submission_ready") is not False:
        errors.append("terminal reviewer-package preparation is incorrectly submission-ready")
    if tifs_terminal_visual_qa_prep.get("terminal_receipts_opened") is not False:
        errors.append("TIFS visual-QA preparation opened a terminal receipt")
    if tifs_terminal_visual_qa_prep.get("pathmnist_performance_artifacts_parsed") is not False:
        errors.append("TIFS visual-QA preparation parsed PathMNIST performance")
    if tifs_terminal_visual_qa_prep.get("visual_inspection_claimed") is not False:
        errors.append("TIFS visual-QA preparation claimed a visual inspection")
    if tifs_terminal_visual_qa_prep.get("submission_ready") is not False:
        errors.append("TIFS visual-QA preparation is incorrectly submission-ready")
    if terminal_surface_integrator_prep.get("terminal_receipts_opened") is not False:
        errors.append("terminal surface-integrator preparation opened a terminal receipt")
    if terminal_surface_integrator_prep.get("pathmnist_performance_artifacts_parsed") is not False:
        errors.append("terminal surface-integrator preparation parsed PathMNIST performance")
    if terminal_surface_integrator_prep.get("files_modified") is not False:
        errors.append("terminal surface-integrator preparation modified claim surfaces")
    if terminal_surface_integrator_prep.get("submission_ready") is not False:
        errors.append("terminal surface-integrator preparation is incorrectly submission-ready")

    result = {
        "status": "PASS_PRETERMINAL_MIGRATION" if not errors else "FAIL",
        "errors": errors,
        "contract_sha256": sha256(CONTRACT),
        "stale_v1_preserved_not_current": not errors,
        "v2_inventory_additions_verified": len(
            contract["mandatory_v2_inventory_additions"]
        ),
        "terminal_only_inputs_required": len(contract["terminal_only_inputs"]),
        "mandatory_terminal_steps": len(contract["mandatory_terminal_sequence"]),
        "forbidden_preterminal_actions": len(
            contract["forbidden_preterminal_actions"]
        ),
        "current_values": observed,
        "pathmnist_partial_artifacts_read": False,
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
