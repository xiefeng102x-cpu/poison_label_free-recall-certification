from __future__ import annotations

import json
from pathlib import Path
import sys


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import verify_final_bundle_v2_migration_preterminal as verifier  # noqa: E402


def test_final_bundle_v2_migration_is_frozen_and_outcome_blind() -> None:
    result = json.loads(verifier.RECEIPT.read_text(encoding="utf-8"))
    assert result["status"] == "PASS_PRETERMINAL_MIGRATION"
    assert result["errors"] == []
    assert result["stale_v1_preserved_not_current"] is True
    assert result["v2_inventory_additions_verified"] == 43
    assert result["terminal_only_inputs_required"] == 2
    assert result["mandatory_terminal_steps"] == 8
    assert result["forbidden_preterminal_actions"] == 5
    assert result["current_values"]["threat_model_surfaces"] == 5
    assert result["current_values"]["threat_model_boundary_tokens"] == 50
    assert result["current_values"]["threat_model_privilege_rows"] == 5
    assert result["current_values"]["threat_model_forbidden_positive_forms"] == 25
    assert result["current_values"]["terminal_integration_mutable_targets"] == 19
    assert result["current_values"]["terminal_integration_mutable_hashes"] == 19
    assert result["current_values"]["terminal_integration_protected_history"] == 4
    assert result["current_values"]["terminal_integration_protected_hashes"] == 4
    assert result["current_values"]["terminal_integration_new_outputs"] == 9
    assert result["current_values"]["terminal_integration_steps"] == 12
    assert result["current_values"]["terminal_integration_surface_actions"] == 19
    assert result["current_values"]["terminal_integration_surface_anchors"] == 19
    assert result["current_values"]["terminal_integration_route_marker_absence"] == 19
    assert result["current_values"]["terminal_integration_capsule_marker_absence"] == 19
    assert result["current_values"]["terminal_integration_metric_marker_absence"] == 19
    assert result["current_values"]["final_v2_mutable_baseline_hashes"] == 19
    assert result["current_values"]["final_v2_protected_history_hashes"] == 4
    assert result["current_values"]["final_v2_terminal_inputs"] == 2
    assert result["current_values"]["final_v2_terminal_outputs"] == 9
    assert result["current_values"]["final_v2_surface_actions"] == 19
    assert result["current_values"]["final_v2_capsule_marker_targets"] == 19
    assert result["current_values"]["final_v2_claim_sentence_targets"] == 3
    assert result["current_values"]["final_v2_metric_targets"] == 2
    assert result["current_values"]["tifs_terminal_configuration_tokens"] == 35
    assert result["current_values"]["tifs_terminal_frozen_protocol_hashes"] == 9
    assert result["current_values"]["terminal_regression_source_inventory_files"] == 191
    assert result["current_values"]["terminal_regression_output_declared"] == 1
    assert result["current_values"]["terminal_release_profile_manifest_targets"] == 4
    assert result["current_values"]["terminal_release_profile_author_blockers"] == 4
    assert result["current_values"]["terminal_release_profile_output_declared"] == 1
    assert result["current_values"]["terminal_reviewer_package_expected_policy_skips"] == 5
    assert result["current_values"]["terminal_reviewer_package_allowed_skip_prefixes"] == 4
    assert result["current_values"]["terminal_reviewer_package_output_declared"] == 1
    assert result["current_values"]["tifs_terminal_visual_qa_confirmation_tokens"] == 1
    assert result["current_values"]["tifs_terminal_visual_qa_output_declared"] == 1
    assert result["current_values"]["terminal_surface_integrator_routes_simulated"] == 3
    assert result["current_values"]["terminal_surface_integrator_targets_per_route"] == 19
    assert result["current_values"]["terminal_surface_integrator_synthetic_renders"] == 57
    assert result["current_values"]["terminal_surface_integrator_output_declared"] == 1
    assert result["pathmnist_partial_artifacts_read"] is False
    assert result["submission_ready"] is False
