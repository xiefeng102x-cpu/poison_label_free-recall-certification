#!/usr/bin/env python3
"""Independent replay checks for the paper14 top-journal evidence bundle."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import platform
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from staged_provenance import effective_sha256, effective_size


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROGRAM = RESULTS / "top_journal_program_v1"
RELEASE_MANIFEST = PROGRAM / "release_archive_manifest_preterminal_2026-10-02.csv"
RELEASE_PREFLIGHT = PROGRAM / "release_archive_preflight_preterminal_2026-10-02.json"


def staged_ledger() -> dict[str, object] | None:
    path = ROOT / "SANITIZATION_LEDGER.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "paper14_blind_review_sanitization_audit_v1"
    assert payload["status"] == "STAGED_COPY_VERIFIED"
    return payload


def staged_ledger_index() -> dict[str, dict[str, object]]:
    ledger = staged_ledger()
    if ledger is None:
        return {}
    return {item["path"]: item for item in ledger["files"]}


def resolve_workspace_path(label: str) -> Path:
    path = Path(label)
    if path.parts and path.parts[0] == "paper14":
        return ROOT.joinpath(*path.parts[1:])
    return ROOT.parent / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return effective_sha256(path, ROOT)


def final_verifier_provenance() -> dict[str, str]:
    path = Path(__file__).resolve()
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
    }


def verify_reproducibility_environment() -> dict[str, object]:
    snapshot_path = PROGRAM / "reproducibility_environment_2026-10-02.json"
    requirements_path = ROOT / "requirements-replay.txt"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["schema"] == "paper14_reproducibility_environment_v1"
    dependency = snapshot["dependency_file"]
    assert dependency["path"] == "requirements-replay.txt"
    assert dependency["sha256"] == sha256(requirements_path)
    backdoorbench = snapshot["source_dependencies"]["BackdoorBench"]
    assert backdoorbench["commit"] == "f02e3534645f0ee63d6848653062cd6c0d6c400d"
    assert (ROOT / backdoorbench["archived_path"]).is_dir()
    assert snapshot["source_dependencies"]["MedMNIST"]["version"] == "3.0.2"
    assert snapshot["torch_backend"]["formal_pathmnist_primary_device"] == "Apple MPS"
    assert snapshot["torch_backend"]["mixed_device_confirmation_allowed"] is False
    observed: dict[str, str | None] = {}
    mismatches: dict[str, dict[str, str | None]] = {}
    for distribution, expected in snapshot["top_level_packages"].items():
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        observed[distribution] = version
        if version != expected:
            mismatches[distribution] = {"expected": expected, "observed": version}
    host = snapshot["host"]
    runtime = {
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
    }
    if runtime["python_version"] != host["python_version"]:
        mismatches["python"] = {
            "expected": host["python_version"],
            "observed": runtime["python_version"],
        }
    if runtime["architecture"] != host["architecture"]:
        mismatches["architecture"] = {
            "expected": host["architecture"],
            "observed": runtime["architecture"],
        }
    return {
        "status": "MATCH_RECORDED_EVIDENCE_HOST" if not mismatches else "RUNTIME_DEVIATION_RECORDED",
        "snapshot_sha256": sha256(snapshot_path),
        "requirements_sha256": sha256(requirements_path),
        "package_count": len(observed),
        "runtime": runtime,
        "mismatches": mismatches,
        "cross_platform_bit_identity_claimed": False,
    }


def verify_mnist_source_provenance() -> dict[str, object]:
    receipt_path = PROGRAM / "mnist_source_provenance_receipt_2026-10-02.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == "paper14_mnist_source_provenance_v1"
    assert receipt["raw_data_redistributed_in_paper14"] is False
    assert receipt["status"] == (
        "PARTIAL_PROVENANCE_RAW_IDENTITY_VERIFIED_GENERATION_ENV_UNRECORDED"
    )
    official_md5 = {
        "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
        "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
        "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
    }
    rows = receipt["raw_gzip_files"]
    assert len(rows) == 4
    assert {Path(row["path"]).name: row["md5"] for row in rows} == official_md5

    workspace_root = ROOT.parent
    raw_present = all((workspace_root / row["path"]).is_file() for row in rows)
    if raw_present:
        for row in rows:
            path = workspace_root / row["path"]
            assert path.stat().st_size == row["bytes"]
            assert sha256(path) == row["sha256"]

    chain = receipt["phase_a_direct_input_chain"]
    for path_key, hash_key in (
        ("config_path", "config_sha256"),
        ("runner_path", "runner_sha256"),
        ("manifest_path", "manifest_sha256"),
    ):
        path = resolve_workspace_path(chain[path_key])
        assert path.is_file()
        assert sha256(path) == chain[hash_key]
    assert chain["source_checkpoint_count"] == 6
    assert chain["source_checkpoint_hashes_recorded_in_manifest"] is True
    assert len(receipt["unresolved_before_public_deposition"]) == 3
    return {
        "status": "RAW_AND_DIRECT_INPUTS_VERIFIED" if raw_present else "RECEIPT_ONLY",
        "raw_resource_count": len(rows),
        "official_md5_match": True,
        "direct_checkpoint_count": chain["source_checkpoint_count"],
        "generation_environment_fully_recorded": False,
        "dataset_redistribution_terms_confirmed": False,
        "receipt_sha256": sha256(receipt_path),
    }


def verify_mnist_historical_environment_recovery() -> dict[str, object]:
    receipt_path = (
        PROGRAM / "mnist_historical_environment_recovery_receipt_2026-10-03.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == (
        "paper14_mnist_historical_environment_recovery_audit_v1"
    )
    assert receipt["status"] == "CLOSED_DOCUMENTED_IRRECOVERABLE_RAW_DATA_EXCLUDED"
    decision = receipt["decision"]
    assert decision["exact_generation_environment_recoverable"] is False
    assert decision["historical_torchvision_version_recoverable"] is False
    assert decision["modern_replay_pin_used_as_historical_version"] is False
    assert decision["raw_mnist_in_public_or_reviewer_profile"] is False
    boundary = receipt["evidence_boundaries"]
    assert boundary["replay_torchvision_version"] == "0.22.1"
    assert boundary["replay_version_scope"] == "current replay environment only"
    assert boundary["scientific_input_chain_hash_bound"] is True
    assert boundary["phase_a_manifest_environment_fields"] == {
        "python_recorded": True,
        "torch_recorded": False,
        "torchvision_recorded": False,
    }
    for item in receipt["hashes"].values():
        assert sha256(ROOT / item["path"]) == item["sha256"]
    return {
        "status": receipt["status"],
        "historical_torchvision_version_recoverable": False,
        "raw_mnist_in_release_profiles": False,
        "scientific_input_chain_hash_bound": True,
    }


def close(actual: float, expected: float, atol: float = 1e-12) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol)


def confusion(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    counts = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for row in rows:
        label = int(row["label"])
        flagged = int(row["flagged"])
        if label == 1 and flagged == 1:
            counts["tp"] += 1
        elif label == 1:
            counts["fn"] += 1
        elif flagged == 1:
            counts["fp"] += 1
        else:
            counts["tn"] += 1
    return counts


def replay_sample_table(
    directory: Path,
    sample_name: str,
    metric_name: str,
    key_fields: tuple[str, ...],
    allowed_methods: set[str] | None = None,
) -> dict[str, int]:
    samples = read_csv(directory / sample_name)
    metrics = read_csv(directory / metric_name)
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in samples:
        if allowed_methods is not None and row.get("method") not in allowed_methods:
            continue
        grouped[tuple(row[field] for field in key_fields)].append(row)
    metric_lookup = {tuple(row[field] for field in key_fields): row for row in metrics}
    errors: list[str] = []
    for key, rows in grouped.items():
        metric = metric_lookup.get(key)
        if metric is None:
            errors.append(f"missing metric row {key}")
            continue
        observed = confusion(rows)
        for field, value in observed.items():
            if int(metric[field]) != value:
                errors.append(f"{key} {field}: samples={value}, metric={metric[field]}")
        expected_tpr = observed["tp"] / max(1, observed["tp"] + observed["fn"])
        expected_fpr = observed["fp"] / max(1, observed["fp"] + observed["tn"])
        if not close(expected_tpr, float(metric["tpr"])):
            errors.append(f"{key} TPR mismatch")
        if not close(expected_fpr, float(metric["fpr"])):
            errors.append(f"{key} FPR mismatch")
    if errors:
        raise AssertionError("; ".join(errors[:12]))
    return {"sample_rows": len(samples), "replayed_configs": len(grouped)}


def verify_protocol_hashes() -> int:
    manifests = [
        RESULTS / "bloodmnist_residual_reupload_confirmation_v1" / "manifest.json",
        RESULTS / "bloodmnist_strong_baselines_v1" / "manifest.json",
        RESULTS / "bloodmnist_mechanism_ablations_v1" / "manifest.json",
        RESULTS / "bloodmnist_covariance_free_confirmation_v1" / "manifest.json",
        RESULTS / "bloodmnist_boundary_study_v1" / "manifest.json",
        RESULTS / "bloodmnist_post_sanitization_v1" / "manifest.json",
        RESULTS / "bloodmnist_official_backdoorbench_confirmation_v1" / "manifest.json",
        RESULTS / "checkpoint_auditability_theory_probe_v1" / "manifest.json",
        RESULTS / "auditability_geometry_diagnostic_v1" / "manifest.json",
        RESULTS / "bloodmnist_multiclass_backdoorbench_pilot_v1" / "manifest.json",
        RESULTS / "bloodmnist_multiclass_backdoorbench_confirmation_v1" / "manifest.json",
        RESULTS / "bloodmnist_multiclass_agpd_adapter_v1" / "manifest.json",
        RESULTS / "bloodmnist_multiclass_agpd_adapter_confirmation_v1" / "manifest.json",
        RESULTS / "bloodmnist_fresh_query_calibration_v1" / "manifest.json",
        RESULTS / "bloodmnist_fresh_query_calibration_confirmation_v1" / "manifest.json",
        RESULTS / "pathmnist_external_validation_v1" / "manifest.json",
        RESULTS / "pathmnist_external_validation_v2" / "manifest.json",
        RESULTS / "pathmnist_warmstart_pilot_v1" / "manifest.json",
    ]
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        protocol = Path(manifest["protocol"])
        if not protocol.exists():
            protocol = PROGRAM / protocol.name
        assert protocol.exists(), protocol
        assert sha256(protocol) == manifest["protocol_sha256"], path
    acquisition = json.loads(
        (PROGRAM / "pathmnist_acquisition_receipt.json").read_text(encoding="utf-8")
    )
    acquisition_protocol = PROGRAM / "pathmnist_acquisition_protocol_v1.json"
    assert acquisition["status"] == "PASS"
    assert acquisition["protocol_sha256"] == sha256(acquisition_protocol)
    return len(manifests) + 1


def verify_new_program_receipts() -> dict[str, int | str]:
    theory = json.loads(
        (PROGRAM / "checkpoint_auditability_theory_receipt.json").read_text(encoding="utf-8")
    )
    multiclass = json.loads(
        (PROGRAM / "multiclass_backdoorbench_program_receipt.json").read_text(encoding="utf-8")
    )
    agpd = json.loads(
        (PROGRAM / "multiclass_agpd_adapter_verification.json").read_text(encoding="utf-8")
    )
    calibration = json.loads(
        (PROGRAM / "fresh_query_calibration_verification.json").read_text(encoding="utf-8")
    )
    path_acquisition = json.loads(
        (PROGRAM / "pathmnist_acquisition_receipt.json").read_text(encoding="utf-8")
    )
    path_v1 = json.loads(
        (PROGRAM / "pathmnist_external_validation_verification.json").read_text(encoding="utf-8")
    )
    path_v2 = json.loads(
        (PROGRAM / "pathmnist_external_validation_verification_v2.json").read_text(encoding="utf-8")
    )
    path_warm = json.loads(
        (PROGRAM / "pathmnist_warmstart_verification.json").read_text(encoding="utf-8")
    )
    assert theory["status"] == "PASS"
    assert theory["replayed_view_configurations"] == 36
    assert theory["replayed_comparisons"] == 18
    assert theory["decision"]["proceed_to_fresh_warm_start_pilot"] is False
    assert multiclass["status"] == "PASS"
    assert multiclass["development"]["status"] == "PASS"
    assert multiclass["confirmation"]["status"] == "PASS"
    assert multiclass["development"]["sample_rows"] == 191344
    assert multiclass["confirmation"]["sample_rows"] == 478360
    assert multiclass["confirmation"]["eligible_seeds"] == [74, 75, 76, 77, 78]
    assert agpd["verification"] == "PASS"
    assert [stage["stage"] for stage in agpd["stages"]] == ["development", "confirmation"]
    assert agpd["stages"][0]["pooled_counts"] == {"tp": 2, "fn": 1198, "fp": 219, "tn": 22499}
    assert agpd["stages"][1]["pooled_counts"] == {"tp": 160, "fn": 2840, "fp": 151, "tn": 56644}
    assert calibration["verification"] == "PASS"
    assert [stage["query_rows"] for stage in calibration["stages"]] == [11984, 29960]
    confirmation_calibration = calibration["stages"][1]["pooled"]
    assert close(confirmation_calibration["suspect_hidden64"]["member_fpr"], 474 / 11650)
    assert close(confirmation_calibration["suspect_hidden64"]["fresh_fpr"], 105 / 3330)
    assert close(confirmation_calibration["paired_hidden64_drift"]["member_fpr"], 502 / 11650)
    assert close(confirmation_calibration["paired_hidden64_drift"]["fresh_fpr"], 181 / 3330)
    assert path_acquisition["status"] == "PASS"
    assert path_acquisition["md5"] == "a8b06965200029087d5bd730944a56c1"
    assert path_acquisition["sha256"] == "81823f52dc622e69db2db4c72f8e8e617938dd6864d3c1f23d4e49724a28ea72"
    assert path_v1["status"] == path_v2["status"] == path_warm["status"] == "PASS"
    assert path_v1["stages"][0]["sample_decisions_replayed"] == 324000
    assert path_v1["stages"][0]["gate"]["both_victims_eligible"] is False
    assert path_v2["stages"][0]["sample_decisions_replayed"] == 324000
    assert path_v2["stages"][0]["eligible_seeds"] == []
    assert path_warm["replay"]["sample_decisions_replayed"] == 324000
    assert path_warm["replay"]["eligible_seeds"] == []
    assert path_warm["replay"]["gate"]["coordinate_alignment_verified"] is True
    assert path_warm["replay"]["gate"]["authorize_fresh_seed_confirmation_protocol"] is False
    return {
        "status": "PASS",
        "theory_view_configurations": theory["replayed_view_configurations"],
        "theory_paired_comparisons": theory["replayed_comparisons"],
        "multiclass_sample_rows": (
            multiclass["development"]["sample_rows"]
            + multiclass["confirmation"]["sample_rows"]
        ),
        "multiclass_config_rows": (
            multiclass["development"]["config_rows"]
            + multiclass["confirmation"]["config_rows"]
        ),
        "agpd_adapter_sample_rows": 83713,
        "agpd_adapter_seeds": 7,
        "fresh_query_rows": 41944,
        "fresh_query_seeds": 7,
        "pathmnist_acquisition": "PASS",
        "pathmnist_method_level_decisions": 972000,
        "pathmnist_experiment_protocols": 3,
        "pathmnist_from_scratch_eligible_victims": 1,
        "pathmnist_warmstart_eligible_victims": 0,
    }


def verify_auditability_geometry_diagnostic() -> dict[str, object]:
    protocol_path = PROGRAM / "auditability_geometry_diagnostic_protocol_v1.json"
    receipt_path = PROGRAM / "auditability_geometry_diagnostic_receipt.json"
    result = RESULTS / "auditability_geometry_diagnostic_v1"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    decision = json.loads((result / "decision.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["errors"] == []
    assert receipt["protocol_sha256"] == sha256(protocol_path)
    assert receipt["analysis_manifest_sha256"] == sha256(result / "manifest.json")
    assert receipt["verifier_sha256"] == sha256(ROOT / receipt["verifier"])
    assert manifest["runner_sha256"] == sha256(ROOT / manifest["runner"])
    for item in manifest["output_hashes"].values():
        assert item["sha256"] == sha256(ROOT / item["path"])
    assert receipt["phase_a_configurations_replayed"] == 36
    assert receipt["bloodmnist_victims_replayed"] == 30
    assert receipt["population_summaries_replayed"] == 48
    assert receipt["paired_contrasts_replayed"] == 5
    assert decision == receipt["decision"] == manifest["decision"]
    assert decision["label"] == "NO_CLEAN_CANCELLATION_DIAGNOSIS"
    assert decision["common_eligible_seeds"] == [55, 56, 58, 60, 61]
    assert decision["first_order_cancellation_gate"] is False
    assert decision["residual_nonlinear_information_gate"] is True
    assert close(decision["multi_to_representation_mean_shift_ratio"], 0.9736528100241794)
    assert close(decision["multi_minus_representation_mean_resultant_length"], 0.027295748570908617)
    assert close(decision["multi_direction_crossfit_rbf_auc_ci_low"], 0.9947055555555556)
    claim_rows = {row["claim_id"]: row for row in read_csv(PROGRAM / "FINAL_CLAIM_MATRIX.csv")}
    assert claim_rows["C28"]["decision"] == "REJECTED"
    assert "42.67\\%" in (ROOT / "manuscript_en" / "main_en.tex").read_text(encoding="utf-8")
    assert "实际抵消机制被否定" in (ROOT / "manuscript_zh" / "main_zh.tex").read_text(encoding="utf-8")
    return {
        "status": "PASS",
        "phase_a_configurations": 36,
        "bloodmnist_victims": 30,
        "population_summaries": 48,
        "paired_contrasts": 5,
        "decision": decision["label"],
    }


def verify_single_sample_reference_audit() -> dict[str, object]:
    protocol_path = PROGRAM / "single_sample_reference_audit_protocol_v1.json"
    receipt_path = PROGRAM / "single_sample_reference_audit_receipt.json"
    result = RESULTS / "single_sample_reference_audit_v1"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    decision = json.loads((result / "decision.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["errors"] == []
    assert receipt["protocol_sha256"] == sha256(protocol_path)
    assert receipt["analysis_manifest_sha256"] == sha256(result / "manifest.json")
    assert receipt["verifier_sha256"] == sha256(ROOT / receipt["verifier"])
    assert manifest["runner_sha256"] == sha256(ROOT / manifest["runner"])
    for item in manifest["output_hashes"].values():
        assert item["sha256"] == sha256(ROOT / item["path"])
    assert receipt["victims_replayed"] == manifest["victim_count"] == 30
    assert receipt["sample_scores_replayed"] == manifest["sample_score_rows"] == 25800
    assert receipt["population_summaries_replayed"] == 6
    assert decision == receipt["decision"] == manifest["decision"]
    assert decision["label"] == "NO_RELIABLE_SINGLE_SAMPLE_CHANNEL_IN_THIS_ARCHIVE"
    assert decision["fresh_confirmation_authorized"] is False
    assert decision["strong_gate_modes"] == []
    assert decision["partial_gate_modes"] == []
    for endpoint in decision["mode_endpoints"].values():
        assert endpoint["pooled_poison_tpr"] < 0.8
        assert endpoint["pooled_member_clean_fpr"] <= 0.075
        assert endpoint["pooled_fresh_clean_fpr"] <= 0.075
    claim_rows = {row["claim_id"]: row for row in read_csv(PROGRAM / "FINAL_CLAIM_MATRIX.csv")}
    assert claim_rows["C29"]["decision"] == "REJECTED"
    assert "52.67\\%--60.71\\%" in (ROOT / "manuscript_en" / "main_en.tex").read_text(encoding="utf-8")
    assert "52.67\\%--60.71\\%" in (ROOT / "manuscript_zh" / "main_zh.tex").read_text(encoding="utf-8")
    return {
        "status": "PASS",
        "victims": 30,
        "sample_scores": 25800,
        "population_summaries": 6,
        "decision": decision["label"],
    }


def verify_reference_calibration_budget_sensitivity() -> dict[str, object]:
    protocol_path = PROGRAM / "reference_calibration_budget_sensitivity_protocol_v1.json"
    receipt_path = PROGRAM / "reference_calibration_budget_sensitivity_verification.json"
    result = RESULTS / "reference_calibration_budget_sensitivity_v1"
    runner = ROOT / "src/run_reference_calibration_budget_sensitivity.py"
    verifier = ROOT / "src/verify_reference_calibration_budget_sensitivity.py"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    assert receipt["verification"] == "PASS"
    assert receipt["protocol_sha256"] == manifest["protocol_sha256"] == sha256(protocol_path)
    assert receipt["runner_sha256"] == manifest["runner_sha256"] == sha256(runner)
    assert receipt["verifier_sha256"] == sha256(verifier)
    assert receipt["dataset_npz_sha256"] == manifest["dataset_npz_sha256"]
    assert receipt["grid_cells"] == 15
    assert receipt["seed_grid_instances"] == receipt["metric_rows"] == 150
    assert receipt["sample_rows"] == 99000
    assert manifest["original_120_80_cell_exact_replay"] is True
    assert manifest["eligible_seeds"] == [64, 66, 67, 68, 69, 70, 71]
    assert manifest["reference_counts"] == [8, 16, 32, 64, 120]
    assert manifest["calibration_counts"] == [20, 40, 80]
    for name, expected in receipt["artifact_sha256"].items():
        assert sha256(result / name) == expected
    summaries = {
        (int(row["reference_count"]), int(row["calibration_count"])): row
        for row in receipt["summary"]
    }
    assert len(summaries) == 15
    assert close(summaries[(8, 20)]["pooled_tpr"], 0.4380952380952381)
    assert close(summaries[(16, 80)]["pooled_tpr"], 0.8404761904761905)
    assert close(summaries[(120, 80)]["pooled_tpr"], 0.9428571428571428)
    assert close(summaries[(120, 80)]["pooled_fpr"], 0.04738095238095238)
    claim_rows = {row["claim_id"]: row for row in read_csv(PROGRAM / "FINAL_CLAIM_MATRIX.csv")}
    assert claim_rows["C30"]["decision"] == "REJECTED"
    return {
        "status": "PASS",
        "budget_cells": 15,
        "seed_grid_instances": 150,
        "sample_rows": 99000,
        "original_cell_exact_replay": True,
    }


def verify_manuscript_claim_consistency() -> dict[str, object]:
    receipt_path = PROGRAM / "manuscript_claim_consistency_receipt_2026-10-02.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["errors"] == []
    assert receipt["pathmnist_partial_artifacts_read"] is False
    assert receipt["authoritative_claim_families_recomputed"] == 7
    assert receipt["required_bilingual_tokens_checked"] == 47
    assert receipt["claim_matrix_decisions_checked"] == 7
    assert receipt["forbidden_positive_forms_checked"] == 8
    assert receipt["verifier_sha256"] == sha256(ROOT / receipt["verifier"])
    expected = {
        "english_source": ROOT / "manuscript_en/main_en.tex",
        "english_pdf": ROOT / "manuscript_en/output/pdf/main_en.pdf",
        "chinese_source": ROOT / "manuscript_zh/main_zh.tex",
        "chinese_pdf": ROOT / "manuscript_zh/output/pdf/main_zh.pdf",
    }
    for key, path in expected.items():
        assert receipt["manuscript_hashes"][key] == sha256(path)
    grid = receipt["single_sample_operating_grid"]
    assert grid["calibration_count"] == 100
    assert grid["scorer_count"] == 3
    assert close(grid["smallest_fused_rank"], 3.0 / 101.0)
    assert close(grid["next_fused_rank"], 6.0 / 101.0)
    return {
        "status": "PASS",
        "claim_families": 7,
        "bilingual_tokens": 47,
        "claim_decisions": 7,
        "forbidden_positive_forms": 8,
        "pathmnist_partial_artifacts_read": False,
    }


def verify_pathmnist_weight_redistribution_audit() -> dict[str, object]:
    receipt_path = (
        PROGRAM / "pathmnist_weight_redistribution_audit_receipt_2026-10-02.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == "paper14_pathmnist_weight_redistribution_audit_v1"
    assert receipt["status"] == "SAFE_RELEASE_POLICY_FIXED_TERMS_UNCONFIRMED"
    decision = receipt["decision"]
    assert decision["terms_confirmed"] is False
    assert decision["official_binary_redistribution_authorized"] is False
    assert decision["derived_binary_redistribution_authorized"] is False
    assert decision["dataset_license_applied_to_weights"] is False
    assert decision["private_evidence_workspace_retained"] is True
    assert decision["policy"] == (
        "EXCLUDE_OFFICIAL_AND_DERIVED_WEIGHT_BINARIES_FROM_REVIEWER_AND_"
        "PUBLIC_RELEASE_PROFILES_UNTIL_EXPLICIT_AUTHORIZATION"
    )
    assert len(receipt["primary_sources"]) == 3
    ledger_index = staged_ledger_index()
    for item in receipt["binary_inventory"]:
        path = ROOT / item["path"]
        if path.is_file():
            assert path.stat().st_size == item["size_bytes"]
            assert sha256(path) == item["sha256"]
        else:
            detail = ledger_index.get(item["path"])
            assert detail is not None
            assert detail["profile_action"] == "EXCLUDED_WEIGHT_BINARY"
            assert detail["source_size_bytes"] == item["size_bytes"]
            assert detail["source_sha256"] == item["sha256"]
    provenance = receipt["provenance_receipt"]
    assert sha256(ROOT / provenance["path"]) == provenance["sha256"]
    return {
        "status": receipt["status"],
        "terms_confirmed": False,
        "binary_count": len(receipt["binary_inventory"]),
        "public_binary_redistribution_authorized": False,
        "release_policy": decision["policy"],
    }


def verify_public_code_profile() -> dict[str, object]:
    from build_public_code_profile import (
        BINARY_ARTIFACT_SUFFIXES,
        MANIFEST,
        PREFLIGHT,
        project_code_files,
        protocol_files,
        selected_files,
        vendor_files,
    )

    rows = read_csv(MANIFEST)
    expected_paths = [str(path.relative_to(ROOT)) for path in selected_files()]
    assert [row["path"] for row in rows] == expected_paths
    assert len(expected_paths) == len(set(expected_paths))
    for row in rows:
        path = ROOT / row["path"]
        assert effective_size(path, ROOT) == int(row["size_bytes"])
        assert sha256(path) == row["sha256"]
        assert path.suffix.lower() not in BINARY_ARTIFACT_SUFFIXES
        assert row["residual_path_risks"] == ""
        assert row["secret_flags"] == ""
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    assert preflight["schema"] == "paper14_public_code_profile_preflight_v1"
    assert preflight["status"] == "PRETERMINAL_PROJECT_LICENSE_REQUIRED"
    assert preflight["manifest"]["sha256"] == sha256(MANIFEST)
    assert preflight["file_count"] == len(rows)
    assert preflight["complete_project_code_closure"] is True
    assert preflight["project_code_file_count"] == len(project_code_files())
    assert preflight["protocol_file_count"] == len(protocol_files())
    assert preflight["backdoorbench_file_count"] == len(vendor_files())
    assert preflight["residual_path_file_count"] == 0
    assert preflight["secret_flag_file_count"] == 0
    assert preflight["binary_artifact_count"] == 0
    assert preflight["project_code_license_selected"] is False
    assert preflight["publication_authorized"] is False
    return {
        "status": preflight["status"],
        "file_count": len(rows),
        "project_code_file_count": preflight["project_code_file_count"],
        "protocol_file_count": preflight["protocol_file_count"],
        "backdoorbench_file_count": preflight["backdoorbench_file_count"],
        "binary_artifact_count": 0,
        "secret_flag_file_count": 0,
        "publication_authorized": False,
    }


def verify_pathmnist_standard_readiness() -> dict[str, object]:
    verification = json.loads(
        (RESULTS / "pathmnist_official_weights_clean_validation_v1" / "verification.json").read_text(
            encoding="utf-8"
        )
    )
    assert verification["status"] == "PASS"
    assert verification["failures"] == []
    assert verification["claim_scope"].startswith("Official clean-checkpoint provenance only")
    expected = {
        "resnet18_28_1.pth": "64774e5fdf8786c7f0182eb6a7300d162b12a7a93455805cb2987eb0c12258e0",
        "resnet18_28_2.pth": "9984d6c42189d79462e22c190bd4b0c669d98a8dac1d0e98f915b9c866ead63e",
    }
    for member in verification["members"]:
        assert member["clean_metrics"]["strict_load"] is True
        assert member["clean_metrics"]["trainable_parameter_count"] == 11173449
        path = Path(member["path"])
        assert path.exists() and path.stat().st_size == 44752263
        assert member["sha256"] == expected[member["name"]] == sha256(path)

    scratch_protocol = PROGRAM / "pathmnist_resnet18_victim_pilot_protocol_v1.json"
    warm_protocol = PROGRAM / "pathmnist_official_weight_warmstart_protocol_v1.json"
    assert sha256(scratch_protocol) == "1ddd71c2985404e6c0984159367ba0d2e9309ed48dfb0d4c8919bda0ef64807e"
    assert sha256(warm_protocol) == "8831e7bed5a2f6e30cf95cf40033e9cf8367784ea86817e44020acdc8b7dd77c"
    scratch_dry = json.loads(
        (RESULTS / "pathmnist_resnet18_victim_pilot_dry_run_official_compat" / "verification.json").read_text(
            encoding="utf-8"
        )
    )
    assert scratch_dry["status"] == "DRY_RUN_VALID_NONFORMAL"
    assert scratch_dry["formal_evidence"] is False
    assert scratch_dry["failures"] == []

    formal_root = RESULTS / "pathmnist_official_weight_warmstart_v1"
    formal_verification = json.loads(
        (formal_root / "verification.json").read_text(encoding="utf-8")
    )
    formal_manifest = json.loads(
        (formal_root / "manifest.json").read_text(encoding="utf-8")
    )
    formal_receipts = json.loads(
        (formal_root / "seed_receipts.json").read_text(encoding="utf-8")
    )
    formal_metrics = read_csv(formal_root / "victim_metrics.csv")
    assert formal_verification["status"] == "PASS"
    assert formal_verification["failures"] == []
    assert formal_verification["formal_evidence"] is True
    assert formal_verification["completed_seeds"] == [93, 94]
    assert formal_verification["protocol_sha256"] == sha256(warm_protocol)
    assert formal_verification["gate"]["authorize_followup"] is True
    assert formal_verification["gate"]["both_fixed_epoch5_victims_eligible"] is True
    assert formal_manifest["status"] == "FORMAL_COMPLETE_GATE_PASSED"
    assert formal_manifest["formal"] is True
    assert formal_manifest["runs_detectors"] is False
    assert formal_manifest["trains_clean_pair"] is False
    assert formal_manifest["test_data_used_during_training_or_selection"] is False
    assert formal_manifest["completed_seeds"] == [93, 94]
    assert len(formal_metrics) == len(formal_receipts) == 2
    metrics_by_seed = {int(row["seed"]): row for row in formal_metrics}
    assert sorted(metrics_by_seed) == [93, 94]
    for receipt in formal_receipts:
        seed = int(receipt["seed"])
        row = metrics_by_seed[seed]
        assert row["formal"] == "True"
        assert row["eligible"] == "True"
        assert int(row["checkpoint_epoch"]) == 5
        assert row["primary_checkpoint"] == "fixed_final_epoch_5"
        assert row["validation_selection_used"] == "False"
        assert float(row["test_balanced_accuracy"]) >= 0.75
        assert float(row["attack_success_rate"]) >= 0.80
        checkpoint = Path(receipt["primary_checkpoint"]["path"])
        assert checkpoint.exists()
        assert receipt["primary_checkpoint"]["epoch"] == 5
        assert receipt["primary_checkpoint"]["role"] == "primary_fixed_final_epoch_5"
        assert sha256(checkpoint) == receipt["primary_checkpoint"]["file_sha256"]
        assert receipt["training"]["initial_state_bit_identical_to_official"] is True
        assert receipt["training"]["test_data_used_during_training_or_selection"] is False
    return {
        "status": "FORMAL_VICTIM_FEASIBILITY_PASSED_DETECTOR_NOT_RUN",
        "official_weights_verified": len(verification["members"]),
        "strict_parameter_count": 11173449,
        "prepared_protocol_hashes_verified": 2,
        "from_scratch_dry_run": scratch_dry["status"],
        "formal_victim_runs_completed": 2,
        "formal_eligible_victims": 2,
        "victim_metrics": {
            str(seed): {
                "test_balanced_accuracy": float(row["test_balanced_accuracy"]),
                "test_macro_auc": float(row["test_macro_auc"]),
                "attack_success_rate": float(row["attack_success_rate"]),
            }
            for seed, row in sorted(metrics_by_seed.items())
        },
        "detector_evidence": False,
    }


def verify_pathmnist_detector_confirmation_readiness() -> dict[str, object]:
    protocol_path = PROGRAM / "pathmnist_official_weight_detector_confirmation_protocol_v1.json"
    addendum_path = PROGRAM / "PATHMNIST_OFFICIAL_WEIGHT_DETECTOR_CONFIRMATION_ADDENDUM.md"
    runner_path = ROOT / "src" / "run_pathmnist_official_weight_detector_confirmation.py"
    verifier_path = ROOT / "src" / "verify_pathmnist_official_weight_detector_confirmation.py"
    test_path = ROOT / "tests" / "test_pathmnist_official_weight_detector_confirmation.py"
    expected_hashes = {
        protocol_path: "52bb254a7ddd1929ba7687bd4be83f5ebf038005d25e046ad9dea679168adca3",
        addendum_path: "5214c7d669d5a216f7621135d69c48a09ee04caaaee3ee9bb2414d83d0b948de",
        runner_path: "a5045474d0ea25a84a1365339eef1ee2ef7bd448784d223bb9d6e583cdf54711",
        verifier_path: "d565bfab8d4e9f64ada8468adc01f33145c33a634cfd1a5c160d651a699c15c0",
        test_path: "5fc4d681cc1b177aec4959e02f36330492804a35ea78293e5fdd5cbf12456379",
    }
    for path, expected in expected_hashes.items():
        assert path.exists() and sha256(path) == expected, path
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["status"] == "protocol_and_code_readiness_only_no_formal_execution"
    assert protocol["seed_plan"]["formal_confirmation_seeds"] == [103, 104, 105, 106, 107]
    assert protocol["seed_plan"]["official_weight_mapping"] == {
        "103": "resnet18_28_1.pth",
        "104": "resnet18_28_2.pth",
        "105": "resnet18_28_1.pth",
        "106": "resnet18_28_2.pth",
        "107": "resnet18_28_1.pth",
    }
    assert "20271000" in protocol["dataset"]["validation_index_selection"]
    assert protocol["detectors"]["primary"]["name"].startswith("CF-CMD-U")
    assert protocol["detectors"]["primary"]["feature_transform"].startswith("identity")
    assert "both clean and suspect" in protocol["victim_eligibility"]["population_rule"]
    assert protocol["primary_efficacy_gate"]["pooled_point_thresholds"] == {
        "tpr_min": 0.9,
        "all_clean_fpr_max": 0.05,
    }
    assert protocol["execution_firewall"]["success_is_not_performance"] is True
    assert protocol["execution_environment"]["primary_device"] == "Apple MPS"
    assert protocol["execution_environment"]["deterministic_algorithms"] is True
    mps_receipt = json.loads(
        (PROGRAM / "pathmnist_mps_smoke_receipt.json").read_text(encoding="utf-8")
    )
    assert mps_receipt["status"] == "PASS_NONFORMAL_SYNTHETIC_ONLY"
    assert mps_receipt["formal_seed_opened"] is False
    assert mps_receipt["same_seed_repeatability"]["state_bit_identical"] is True
    assert close(
        mps_receipt["official_architecture_synthetic_benchmark"][
            "observed_step_speedup_cpu_over_mps"
        ],
        11.7896508250194,
    )
    smoke_root = RESULTS / "pathmnist_real_input_mps_smoke_v1"
    smoke_receipt_path = smoke_root / "smoke_receipt.json"
    smoke_verification_path = smoke_root / "smoke_verification.json"
    smoke = json.loads(smoke_receipt_path.read_text(encoding="utf-8"))
    smoke_verification = json.loads(smoke_verification_path.read_text(encoding="utf-8"))
    assert sha256(smoke_receipt_path) == "18e051931c3774bd70db7f0fce1714317e95755488a3e88c3acf1aa2f17abbf7"
    assert sha256(smoke_verification_path) == "016d4c72fc6dfd8931b452c6618de6f9276ed80eceaf84aba4d15671935ccd9c"
    assert smoke["status"] == "PASS_NONFORMAL_REAL_INPUT_MPS"
    assert smoke["formal_seed_opened"] is False
    assert smoke["confirmation_seed_files_opened"] is False
    assert smoke["nonformal_seed"] == 20261002
    assert smoke["bit_identical_resume"] is True
    assert smoke["feature_dimension"] == 512
    assert smoke["runner_sha256"] == sha256(runner_path)
    assert smoke["protocol_sha256"] == sha256(protocol_path)
    assert smoke_verification["status"] == "PASS_NONFORMAL_REAL_INPUT_MPS"
    assert smoke_verification["failures"] == []
    for key, filename in smoke["checkpoint_files"].items():
        artifact = smoke_root / filename
        assert artifact.is_file() and sha256(artifact) == smoke["checkpoint_sha256"][key]
    feature_artifact = smoke_root / smoke["feature_file"]
    assert feature_artifact.is_file() and sha256(feature_artifact) == smoke["feature_file_sha256"]
    from run_pathmnist_official_weight_detector_confirmation import synthetic_dry_run

    dry = synthetic_dry_run()
    assert dry["status"] == "synthetic_dry_run_passed_nonformal"
    assert dry["formal_data_or_weights_loaded"] is False
    assert dry["training_performed"] is False
    assert dry["confirmation_seed_files_opened"] is False
    assert dry["primary_receipt"]["whitening"] is False
    formal_root = RESULTS / "pathmnist_official_weight_detector_confirmation_v1"
    formal_status = "NOT_STARTED"
    formal_seed_opened = False
    detector_evidence = False
    eligible_seed_count = None
    all_pairs_eligible = None
    primary_point_gate_pass = None
    if formal_root.exists():
        formal_manifest = json.loads((formal_root / "manifest.json").read_text(encoding="utf-8"))
        assert formal_manifest["protocol_sha256"] == sha256(protocol_path)
        assert formal_manifest["device"] == "mps"
        formal_status = formal_manifest["status"]
        formal_seed_opened = formal_status in {"RUNNING", "COMPLETE_FORMAL"}
        if formal_status == "COMPLETE_FORMAL":
            from verify_pathmnist_official_weight_detector_confirmation import verify

            replay = verify(formal_root)
            assert replay["status"] == "PASS", replay
            assert replay["failures"] == [], replay
            assert replay["verified_seeds"] == [103, 104, 105, 106, 107], replay
            assert replay["verified_method_rows"] == 25, replay
            detector_evidence = replay["formal_evidence"] is True
            eligibility = formal_manifest.get("victim_eligibility_by_seed", {})
            assert set(eligibility) == {"103", "104", "105", "106", "107"}
            assert all(isinstance(value, bool) for value in eligibility.values())
            eligible_seed_count = sum(bool(value) for value in eligibility.values())
            all_pairs_eligible = formal_manifest.get("all_pairs_eligible")
            primary_point_gate_pass = formal_manifest.get("primary_point_gate_pass")
            assert all_pairs_eligible is (eligible_seed_count == 5)
            assert isinstance(primary_point_gate_pass, bool)
        else:
            assert formal_status == "RUNNING"
    return {
        "status": "FORMAL_RUNNING" if formal_status == "RUNNING" else (
            "FORMAL_VERIFIED" if formal_status == "COMPLETE_FORMAL" else "CODE_AND_REAL_INPUT_MPS_SMOKE_READY"
        ),
        "formal_execution_enabled": True,
        "formal_seed_opened": formal_seed_opened,
        "confirmation_seeds": [103, 104, 105, 106, 107],
        "focused_tests": "13/13 PASS",
        "serial_cpu_hour_planning_range": [70.6, 304.6],
        "primary_training_device": "Apple MPS",
        "synthetic_step_speedup_cpu_over_mps": 11.7896508250194,
        "real_input_mps_smoke": smoke_verification["status"],
        "formal_status": formal_status,
        "detector_evidence": detector_evidence,
        "eligible_seed_count": eligible_seed_count,
        "all_pairs_eligible": all_pairs_eligible,
        "primary_point_gate_pass": primary_point_gate_pass,
    }


def pathmnist_detector_limitation(summary: dict[str, object]) -> str:
    """Describe the evidence boundary without leaving a stale running claim."""

    status = summary["status"]
    if status == "FORMAL_RUNNING":
        return (
            "The PathMNIST detector-confirmation protocol fixes five paired seeds "
            "and passes synthetic plus real-input MPS exact-resume/feature checks. "
            "Its 103--107 formal block is running, so no detector endpoint is "
            "accepted before completion and independent replay."
        )
    if status == "FORMAL_VERIFIED":
        eligible = summary["eligible_seed_count"]
        if summary["all_pairs_eligible"] is not True:
            return (
                "The independently replayed PathMNIST 103--107 block has only "
                f"{eligible} of 5 attack-eligible victim pairs; detector outputs "
                "remain diagnostic and no external-domain efficacy endpoint is "
                "pooled or promoted."
            )
        if summary["primary_point_gate_pass"] is not True:
            return (
                "All five PathMNIST victim pairs independently replay, but the "
                "frozen primary detector point gate fails; this is a confirmed "
                "external-domain boundary rather than positive efficacy evidence."
            )
        return (
            "The independently replayed PathMNIST 103--107 block passes its frozen "
            "victim and primary point gates only for the official-weight ResNet-18 "
            "route; it does not establish dataset-, architecture- or attack-"
            "invariant detection."
        )
    return (
        "The PathMNIST detector confirmation is not formal evidence until its "
        "fixed 103--107 block completes and independently replays."
    )


def select_pathmnist_terminal_claim_route(summary: dict[str, object]) -> str:
    """Apply the frozen terminal-claim contract without outcome discretion."""

    if summary["status"] != "FORMAL_VERIFIED":
        return "PENDING_TERMINAL"
    eligible = summary["eligible_seed_count"]
    if not isinstance(eligible, int) or isinstance(eligible, bool):
        raise AssertionError("verified PathMNIST summary lacks an eligibility count")
    if summary["all_pairs_eligible"] is not True:
        assert 0 <= eligible < 5
        assert summary["primary_point_gate_pass"] is False
        return "R1_INSUFFICIENT_ELIGIBLE_VICTIMS"
    assert eligible == 5
    if summary["primary_point_gate_pass"] is not True:
        return "R2_ELIGIBLE_PRIMARY_GATE_FAIL"
    return "R3_ELIGIBLE_PRIMARY_GATE_PASS"


def expected_pathmnist_claim_decision(route: str) -> str:
    decisions = {
        "PENDING_TERMINAL": "PENDING_TERMINAL",
        "R0_INVALID_OR_INCOMPLETE": "INVALID_NO_FORMAL_CLAIM",
        "R1_INSUFFICIENT_ELIGIBLE_VICTIMS": "INCONCLUSIVE",
        "R2_ELIGIBLE_PRIMARY_GATE_FAIL": "REJECTED",
        "R3_ELIGIBLE_PRIMARY_GATE_PASS": "SUPPORTED_CONDITIONALLY",
    }
    if route not in decisions:
        raise AssertionError(f"unknown PathMNIST terminal claim route: {route}")
    return decisions[route]


def verify_pathmnist_terminal_claim_contract(
    selected_route: str = "PENDING_TERMINAL",
) -> dict[str, object]:
    path = PROGRAM / "pathmnist_terminal_claim_integration_contract_v1.json"
    amendment_path = PROGRAM / (
        "pathmnist_terminal_claim_integration_contract_v1_"
        "amendment_2026-10-02.json"
    )
    expected_contract_sha256 = (
        "71e09c3fb678fad1024d822b929aca8379d61372c24d5e1441cb10d5d85f79c0"
    )
    expected_amendment_sha256 = (
        "d98f677260fbf4226d503cf9b1e65c8651c32cdbdc1f68ebed7de9aed788815e"
    )
    assert sha256(path) == expected_contract_sha256
    assert sha256(amendment_path) == expected_amendment_sha256
    contract = json.loads(path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    assert contract["status"] == "FROZEN_BEFORE_TERMINAL_RESULT_INSPECTION"
    assert contract["formal_block"]["seeds"] == [103, 104, 105, 106, 107]
    assert contract["formal_block"]["state_at_freeze"] == "RUNNING"
    assert contract["formal_block"]["performance_endpoints_inspected"] is False
    bindings = contract["bindings"]
    assert bindings["protocol_sha256"] == sha256(
        PROGRAM / "pathmnist_official_weight_detector_confirmation_protocol_v1.json"
    )
    assert bindings["runner_sha256"] == sha256(
        ROOT / "src/run_pathmnist_official_weight_detector_confirmation.py"
    )
    assert bindings["standalone_verifier_sha256"] == sha256(
        ROOT / "src/verify_pathmnist_official_weight_detector_confirmation.py"
    )
    routes = contract["ordered_routes"]
    assert [route["route"] for route in routes] == [
        "R0_INVALID_OR_INCOMPLETE",
        "R1_INSUFFICIENT_ELIGIBLE_VICTIMS",
        "R2_ELIGIBLE_PRIMARY_GATE_FAIL",
        "R3_ELIGIBLE_PRIMARY_GATE_PASS",
    ]
    assert routes[0]["efficacy_pooling_allowed"] is False
    assert routes[1]["efficacy_pooling_allowed"] is False
    assert routes[2]["efficacy_pooling_allowed"] is True
    assert routes[3]["efficacy_pooling_allowed"] is True
    base_targets = contract["mandatory_integration_targets"]
    assert len(base_targets) == len(set(base_targets))
    targets = set(base_targets)
    assert amendment["status"] == "FROZEN_PRE_TERMINAL_LOGISTICS_ONLY"
    assert amendment["base_contract_sha256"] == expected_contract_sha256
    assert amendment["performance_endpoints_inspected"] is False
    assert amendment["scientific_decision_rules_changed"] is False
    assert amendment["route_conditions_changed"] is False
    assert amendment["claim_permissions_changed"] is False
    added_targets = amendment["add_mandatory_integration_targets"]
    assert added_targets == [
        "README.md",
        "manuscript_zh/CLAIMS_EVIDENCE_ZH.md",
    ]
    assert not (targets & set(added_targets))
    targets.update(added_targets)
    assert len(targets) == 16
    for relative in targets:
        target = ROOT / relative
        assert target.is_file() and target.stat().st_size > 0, target
    claim_rows = read_csv(PROGRAM / "FINAL_CLAIM_MATRIX.csv")
    claim_ids = [row["claim_id"] for row in claim_rows]
    assert len(claim_ids) == len(set(claim_ids))
    pending_claim = next(row for row in claim_rows if row["claim_id"] == "C27")
    expected_decision = expected_pathmnist_claim_decision(selected_route)
    assert pending_claim["decision"] == expected_decision
    if selected_route == "PENDING_TERMINAL":
        assert all(
            route in pending_claim["evidence"]
            for route in ("R0", "R1", "R2", "R3")
        )
        assert "partial endpoints" in pending_claim["required_qualification"]
    else:
        assert selected_route in pending_claim["evidence"]
        assert "still running" not in pending_claim["evidence"].lower()
    assert {"manuscript_zh/main_zh.tex", "manuscript_en/main_en.tex"} <= targets
    assert "results/top_journal_program_v1/final_bundle_receipt.json" in targets
    assert len(contract["forbidden_actions"]) >= 7
    return {
        "status": "FROZEN_PRE_TERMINAL",
        "contract_sha256": expected_contract_sha256,
        "logistics_amendment_sha256": expected_amendment_sha256,
        "routes": [route["route"] for route in routes],
        "mandatory_integration_targets": len(targets),
        "integration_target_paths_verified": True,
        "pending_claim_id": "C27",
        "selected_route": selected_route,
        "expected_claim_decision": expected_decision,
        "forbidden_actions": len(contract["forbidden_actions"]),
    }


def verify_p6c_noise_bridge() -> dict[str, object]:
    base = RESULTS / "p6c_qml_noise_bridge_v1"
    diagnostic = base / "diagnostic_2048"
    config = ROOT / "configs" / "p6c_qml_noise_bridge_v1.json"
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    diagnostic_manifest = json.loads((diagnostic / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and manifest["mode"] == "full"
    assert manifest["config_sha256"] == sha256(config)
    assert diagnostic_manifest["status"] == "diagnostic_complete"
    assert diagnostic_manifest["diagnostic_only"] is True
    assert diagnostic_manifest["primary_summary_sha256"] == sha256(base / "paired_summary.json")

    receipt = json.loads(
        (PROGRAM / "p6c_qml_noise_bridge_code_receipt.json").read_text(encoding="utf-8")
    )
    hashed_paths = {
        "config": config,
        "runner": ROOT / "src" / "run_p6c_qml_noise_bridge.py",
        "verifier": ROOT / "src" / "verify_p6c_qml_noise_bridge.py",
        "primary_manifest": base / "manifest.json",
        "primary_summary": base / "paired_summary.json",
        "primary_metrics": base / "per_seed_metrics.csv",
        "primary_scores": base / "per_sample_scores.csv",
        "diagnostic_manifest": diagnostic / "manifest.json",
        "diagnostic_metrics": diagnostic / "per_seed_metrics.csv",
        "diagnostic_scores": diagnostic / "per_sample_scores.csv",
    }
    for name, path in hashed_paths.items():
        assert receipt["sha256"][name] == sha256(path), (name, path)

    def replay(directory: Path, expected_metric_rows: int, expected_score_rows: int) -> list[dict[str, str]]:
        scores = read_csv(directory / "per_sample_scores.csv")
        metrics = read_csv(directory / "per_seed_metrics.csv")
        assert len(scores) == expected_score_rows
        assert len(metrics) == expected_metric_rows
        test_groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
        for row in scores:
            if row["split"] == "test":
                test_groups[(row["seed"], row["condition"], row["victim_kind"])].append(row)
        assert len(test_groups) == expected_metric_rows
        for metric in metrics:
            key = (metric["seed"], metric["condition"], metric["victim_kind"])
            rows = test_groups[key]
            labels = [int(row["label"]) for row in rows]
            flags = [float(row["score"]) >= float(row["threshold"]) for row in rows]
            tp = sum(label == 1 and flag for label, flag in zip(labels, flags))
            fn = sum(label == 1 and not flag for label, flag in zip(labels, flags))
            fp = sum(label == 0 and flag for label, flag in zip(labels, flags))
            tn = sum(label == 0 and not flag for label, flag in zip(labels, flags))
            assert close(tp / (tp + fn), float(metric["tpr"]))
            assert close(fp / (fp + tn), float(metric["fpr"]))
        return metrics

    primary_metrics = replay(base, 30, 2592)
    diagnostic_metrics = replay(diagnostic, 6, 864)
    primary_backdoor = [
        row for row in primary_metrics
        if row["victim_kind"] == "backdoor" and row["condition"] == "nominal_noise_512"
    ]
    diagnostic_backdoor = [row for row in diagnostic_metrics if row["victim_kind"] == "backdoor"]
    assert close(sum(float(row["auc"]) for row in primary_backdoor) / 3, 0.9889403292181069)
    assert close(sum(float(row["tpr"]) for row in primary_backdoor) / 3, 1.0)
    assert close(sum(float(row["fpr"]) for row in primary_backdoor) / 3, 0.7129629629629629)
    assert close(sum(float(row["auc"]) for row in diagnostic_backdoor) / 3, 0.9979423868312757)
    assert close(sum(float(row["tpr"]) for row in diagnostic_backdoor) / 3, 1.0)
    assert close(sum(float(row["fpr"]) for row in diagnostic_backdoor) / 3, 0.7129629629629629)
    return {
        "status": "PASS",
        "primary_score_rows": 2592,
        "primary_metric_rows": 30,
        "diagnostic_score_rows": 864,
        "diagnostic_metric_rows": 6,
        "nominal_512_mean_auc": 0.9889403292181069,
        "nominal_2048_diagnostic_mean_auc": 0.9979423868312757,
        "nominal_512_and_2048_pooled_fpr": 0.7129629629629629,
        "primary_operating_point_gate": "FAIL",
        "real_hardware_evidence": False,
    }


def verify_prior_receipts() -> dict[str, str]:
    receipt_names = [
        "phase1_receipt.json",
        "phase2_pilot_receipt.json",
        "phase2_confirmation_receipt.json",
        "cnn_pilot_receipt.json",
        "input_aware_pilot_receipt.json",
        "badnets_confirmation_receipt.json",
        "input_aware_confirmation_receipt.json",
        "phase3_adaptive_receipt.json",
        "official_backdoorbench_confirmation_receipt.json",
    ]
    statuses: dict[str, str] = {}
    for name in receipt_names:
        payload = json.loads((PROGRAM / name).read_text(encoding="utf-8"))
        status = str(payload.get("status", ""))
        assert status == "PASS", (name, status)
        statuses[name] = status
    return statuses


def verify_boundary() -> dict[str, int]:
    per_config = read_csv(RESULTS / "bloodmnist_boundary_study_v1" / "per_config_metrics.csv")
    pooled = read_csv(RESULTS / "bloodmnist_boundary_study_v1" / "pooled_boundary_grid.csv")
    eligible = [row for row in per_config if int(row["eligible_victim"]) == 1]
    key_fields = ("method", "shots", "poison_fraction", "trusted_contamination")
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        grouped[tuple(row[field] for field in key_fields)].append(row)
    assert len(grouped) == len(pooled) == 240
    for row in pooled:
        key = tuple(row[field] for field in key_fields)
        source = grouped[key]
        totals = {field: sum(int(item[field]) for item in source) for field in ("tp", "fn", "fp", "tn")}
        for field, value in totals.items():
            assert int(row[field]) == value, (key, field)
        tpr = totals["tp"] / (totals["tp"] + totals["fn"])
        fpr = totals["fp"] / (totals["fp"] + totals["tn"])
        assert close(tpr, float(row["pooled_tpr"])), key
        assert close(fpr, float(row["pooled_fpr"])), key
        assert int(row["gate_pass"]) == int(tpr >= 0.90 and fpr <= 0.05), key
    return {"per_config_rows": len(per_config), "pooled_rows": len(pooled), "eligible_rows": len(eligible)}


def verify_post_sanitization() -> dict[str, int]:
    rows = read_csv(RESULTS / "bloodmnist_post_sanitization_v1" / "per_seed_metrics.csv")
    summary = read_csv(RESULTS / "bloodmnist_post_sanitization_v1" / "summary.csv")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    mapping = {
        "mean_clean_removed": "clean_candidates_removed",
        "mean_poison_removed": "poisons_removed",
        "mean_qml_balanced_ca": "qml_balanced_ca",
        "mean_qml_asr": "qml_asr",
        "mean_classical_balanced_ca": "classical_balanced_ca",
        "mean_classical_asr": "classical_asr",
    }
    for item in summary:
        source = groups[item["condition"]]
        assert len(source) == int(item["seed_count"]) == 7
        for target, source_field in mapping.items():
            mean = sum(float(row[source_field]) for row in source) / len(source)
            assert close(mean, float(item[target])), (item["condition"], target)
    return {"per_seed_rows": len(rows), "summary_rows": len(summary)}


def verify_p6d_code_readiness() -> dict[str, object]:
    receipt = json.loads(
        (PROGRAM / "p6d_code_readiness_receipt.json").read_text(encoding="utf-8")
    )
    expected_v1 = {
        "protocol_sha256": PROGRAM / "p6d_final_observable_adaptive_protocol_v1.json",
        "runner_sha256": ROOT / "src" / "run_p6d_final_observable_adaptive.py",
        "verifier_sha256": ROOT / "src" / "verify_p6d_final_observable_adaptive.py",
        "focused_test_sha256": ROOT / "tests" / "test_p6d_final_observable_adaptive.py",
    }
    expected_v2 = {
        "protocol_sha256": PROGRAM / "p6d_final_observable_adaptive_protocol_v2.json",
        "runner_sha256": ROOT / "src" / "run_p6d_final_observable_adaptive_v2.py",
        "verifier_sha256": ROOT / "src" / "verify_p6d_final_observable_adaptive_v2.py",
        "focused_test_sha256": ROOT / "tests" / "test_p6d_final_observable_adaptive_v2.py",
    }
    expected_v3 = {
        "protocol_sha256": PROGRAM / "p6d_final_observable_adaptive_protocol_v3.json",
        "runner_sha256": ROOT / "src" / "run_p6d_final_observable_adaptive_v3.py",
        "verifier_sha256": ROOT / "src" / "verify_p6d_final_observable_adaptive_v3.py",
        "focused_test_sha256": ROOT / "tests" / "test_p6d_final_observable_adaptive_v3.py",
    }
    assert receipt["status"] == "V1_V2_PRETEST_FAILURES_V3_FORMAL_COMPLETE_4_OF_5_ELIGIBLE_NO_POOLED_EFFICACY"
    assert receipt["formal_seed_opened"] is True
    assert receipt["formal_experiment_completed"] is True
    assert receipt["primary_efficacy_pooled"] is False
    assert receipt["formal_cli_status"] == "V3_FORMAL_COMPLETE_POSTWRITE_VERIFIER_STAGE_CONSTANT_FAILURE_FORENSIC_REPLAY_PASS"
    assert receipt["focused_tests"] == "17/17 PASS"
    assert receipt["full_regression_tests"] == "130/130 PASS"
    assert receipt["development_launcher"] is True
    assert receipt["confirmation_launcher"] is True
    assert receipt["local_dataset_sha256_verified"] == (
        "062023e186f537e26b3c21ea3b2614ddfc475e8a14825dfd20663bbb7e37bddc"
    )
    for field, path in expected_v1.items():
        assert path.exists(), path
        assert receipt[field] == sha256(path), (field, path)
    incident = receipt["v1_incident"]
    assert incident["development_seeds_consumed"] == [95, 96, 97]
    assert incident["confirmation_seeds_permanently_closed"] == [98, 99, 100, 101, 102]
    assert incident["failure_before_test_access"] is True
    assert incident["detector_efficacy_claim_allowed"] is False
    incident_v2 = receipt["v2_incident"]
    assert incident_v2["formal_seed_opened"] is True
    assert incident_v2["development_seeds_consumed"] == [108, 109, 110]
    assert incident_v2["confirmation_seeds_permanently_closed"] == [111, 112, 113, 114, 115]
    assert incident_v2["reserved_for_pathmnist"] == [103, 104, 105, 106, 107]
    assert incident_v2["failure_before_test_access"] is True
    assert incident_v2["detector_efficacy_claim_allowed"] is False
    assert incident_v2["focused_tests"] == "4/4 PASS"
    for field, path in expected_v2.items():
        assert path.exists(), path
        assert incident_v2[field] == sha256(path), (field, path)
    from run_p6d_final_observable_adaptive_v2 import lossless_csv_roundtrip_self_check

    roundtrip = lossless_csv_roundtrip_self_check(10108)
    assert roundtrip["status"] == "PASS"
    assert roundtrip["rows"] == {"observable": 860, "scores": 660, "history": 1}
    recovery_v3 = receipt["v3_recovery"]
    assert recovery_v3["formal_seed_opened"] is True
    assert recovery_v3["development_seeds"] == [116, 117, 118]
    assert recovery_v3["confirmation_seeds"] == [119, 120, 121, 122, 123]
    for field, path in expected_v3.items():
        assert path.exists(), path
        assert recovery_v3[field] == sha256(path), (field, path)
    formal_paths = {
        "development_manifest_sha256": RESULTS / "p6d_final_observable_adaptive_v3/development/manifest.json",
        "confirmation_lock_sha256": RESULTS / "p6d_final_observable_adaptive_v3/confirmation_lock_receipt.json",
        "confirmation_manifest_sha256": RESULTS / "p6d_final_observable_adaptive_v3/confirmation/manifest.json",
        "forensic_verifier_sha256": ROOT / "src/verify_p6d_v3_confirmation_delivery.py",
        "forensic_test_sha256": ROOT / "tests/test_p6d_v3_confirmation_delivery.py",
    }
    for field, path in formal_paths.items():
        assert path.exists(), path
        assert recovery_v3[field] == sha256(path), (field, path)
    from run_p6d_final_observable_adaptive_v3 import full_path_synthetic_shadow_self_check

    shadow = full_path_synthetic_shadow_self_check(10116)
    assert shadow["status"] == "PASS"
    assert shadow["independent_replay"] == "valid"
    self_check = receipt["synthetic_self_check"]
    assert self_check["seed"] == 10020
    assert self_check["max_abs_score_error"] < 2e-12
    assert self_check["max_abs_threshold_error"] < 2e-12
    assert all(
        self_check[field]
        for field in (
            "trusted_gradient_nonzero",
            "clean_gradient_nonzero",
            "poison_gradient_nonzero",
        )
    )
    return {
        "status": receipt["status"],
        "formal_seed_opened": True,
        "formal_experiment_completed": True,
        "focused_tests": receipt["focused_tests"],
        "full_regression_tests": receipt["full_regression_tests"],
        "development_launcher": True,
        "confirmation_launcher": True,
        "v1_failure_before_test_access": True,
        "v2_formal_seed_opened": True,
        "v2_failure_before_test_access": True,
        "v2_roundtrip_was_insufficient": roundtrip["status"],
        "v3_formal_seed_opened": True,
        "v3_full_path_shadow": shadow["status"],
    }


def verify_p6d_v1_formal_failure() -> dict[str, object]:
    from verify_p6d_v1_formal_failure import verify_p6d_v1_formal_failure as verify

    result = verify(ROOT)
    assert result["status"] == "PASS"
    assert result["test_split_read"] is False
    assert result["victim_metrics_present"] is False
    assert result["confirmation_lock_present"] is False
    assert result["detector_efficacy_claim_allowed"] is False
    return result


def verify_p6d_v2_failure_and_v3_recovery() -> dict[str, object]:
    from verify_p6d_v2_formal_failure import verify_p6d_v2_formal_failure
    from run_p6d_final_observable_adaptive_v3 import full_path_synthetic_shadow_self_check

    incident = verify_p6d_v2_formal_failure(ROOT)
    assert incident["status"] == "PASS"
    assert incident["test_split_read"] is False
    assert incident["validation_training_groups"] == 9
    assert incident["detector_efficacy_claim_allowed"] is False
    protocol_path = PROGRAM / "p6d_final_observable_adaptive_protocol_v3.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    paths = {
        "v3_runner_sha256": ROOT / "src/run_p6d_final_observable_adaptive_v3.py",
        "v3_verifier_sha256": ROOT / "src/verify_p6d_final_observable_adaptive_v3.py",
    }
    for field, path in paths.items():
        assert protocol["source_audit"][field] == sha256(path), (field, path)
    seeds = protocol["seed_audit"]
    assert seeds["v2_consumed_seeds"] == [108, 109, 110]
    assert seeds["v2_confirmation_seeds_permanently_closed"] == [111, 112, 113, 114, 115]
    assert seeds["development_seeds"] == [116, 117, 118]
    assert seeds["confirmation_seeds"] == [119, 120, 121, 122, 123]
    shadow = full_path_synthetic_shadow_self_check(10116)
    assert shadow["status"] == "PASS"
    assert shadow["json_numpy_roundtrip"] is True
    assert shadow["test_firewall_denied_unselected"] is True
    assert shadow["independent_replay"] == "valid"
    return {"status": "PASS", "v2_incident": incident, "v3_shadow": shadow}


def verify_p6d_v3_formal_confirmation() -> dict[str, object]:
    from verify_p6d_v3_confirmation_delivery import verify as forensic_verify

    receipt_path = PROGRAM / "p6d_v3_formal_confirmation_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS_FORENSIC_REPLAY_PRIMARY_GATE_INSUFFICIENT"
    assert receipt["formal_bundle_unchanged_after_execution"] is True
    paths = {
        receipt["development"]["manifest_sha256"]: RESULTS / "p6d_final_observable_adaptive_v3/development/manifest.json",
        receipt["confirmation_lock"]["receipt_sha256"]: RESULTS / "p6d_final_observable_adaptive_v3/confirmation_lock_receipt.json",
        receipt["confirmation"]["manifest_sha256"]: RESULTS / "p6d_final_observable_adaptive_v3/confirmation/manifest.json",
        receipt["confirmation"]["victim_metrics_sha256"]: RESULTS / "p6d_final_observable_adaptive_v3/confirmation/victim_metrics.csv",
        receipt["confirmation"]["per_seed_metrics_sha256"]: RESULTS / "p6d_final_observable_adaptive_v3/confirmation/per_seed_metrics.csv",
        receipt["confirmation"]["confirmation_summary_sha256"]: RESULTS / "p6d_final_observable_adaptive_v3/confirmation/confirmation_summary.json",
        receipt["postwrite_verifier_incident"]["forensic_verifier_sha256"]: ROOT / "src/verify_p6d_v3_confirmation_delivery.py",
        receipt["postwrite_verifier_incident"]["forensic_test_sha256"]: ROOT / "tests/test_p6d_v3_confirmation_delivery.py",
    }
    for expected, path in paths.items():
        assert sha256(path) == expected, path
    replay = forensic_verify(RESULTS / "p6d_final_observable_adaptive_v3/confirmation")
    assert replay["status"] == "PASS_STAGE_AWARE_FORENSIC_REPLAY"
    assert replay["formal_bundle_unchanged"] is True
    numeric = replay["pinned_numerical_replay"]
    assert numeric["status"] == "valid"
    assert numeric["observable_score_groups"] == 5
    assert numeric["epoch_selections"] == 5
    assert numeric["victim_rows"] == 5
    assert numeric["max_abs_score_error"] == 0.0
    outcome = numeric["confirmation"]
    assert outcome == {
        "eligible_victim_count": 4,
        "decision": "INSUFFICIENT_ELIGIBLE_VICTIMS",
        "efficacy_pooled": False,
    }
    recorded = receipt["confirmation"]
    assert recorded["victims_total"] == 5
    assert recorded["eligible_victim_count"] == 4
    assert recorded["efficacy_pooled"] is False
    assert recorded["decision"] == "INSUFFICIENT_ELIGIBLE_VICTIMS"
    assert recorded["ineligible_seed"] == 119
    return {
        "status": receipt["status"],
        "development_eligible_victims": 3,
        "confirmation_eligible_victims": 4,
        "confirmation_total_victims": 5,
        "primary_efficacy_pooled": False,
        "decision": outcome["decision"],
        "postwrite_forensic_replay": replay["status"],
    }


def verify_bilingual_manuscript_receipt() -> dict[str, object]:
    receipt = json.loads(
        (PROGRAM / "bilingual_manuscript_receipt.json").read_text(encoding="utf-8")
    )
    zh_source = ROOT / "manuscript_zh" / "main_zh.tex"
    zh_appendix = ROOT / "manuscript_zh" / "theory_appendix_zh.tex"
    zh_pdf = ROOT / "manuscript_zh" / "output" / "pdf" / "main_zh.pdf"
    en_source = ROOT / "manuscript_en" / "main_en.tex"
    en_appendix = ROOT / "manuscript_en" / "theory_appendix.tex"
    en_pdf = ROOT / "manuscript_en" / "output" / "pdf" / "main_en.pdf"
    zh_bib = ROOT / "manuscript_zh" / "references_zh.bib"
    en_bib = ROOT / "manuscript_en" / "references_en.bib"
    assert receipt["status"] == "PASS"
    assert receipt["chinese"]["source_sha256"] == sha256(zh_source)
    assert receipt["chinese"]["theory_appendix_sha256"] == sha256(zh_appendix)
    assert receipt["chinese"]["pdf_sha256"] == sha256(zh_pdf)
    assert receipt["english"]["source_sha256"] == sha256(en_source)
    assert receipt["english"]["theory_appendix_sha256"] == sha256(en_appendix)
    assert receipt["english"]["pdf_sha256"] == sha256(en_pdf)
    assert zh_bib.read_bytes() == en_bib.read_bytes()
    assert receipt["shared_bibliography"]["byte_identical"] is True
    assert receipt["shared_bibliography"]["sha256"] == sha256(zh_bib)
    assert "\\input{theory_appendix_zh}" in zh_source.read_text(encoding="utf-8")
    assert "\\input{theory_appendix}" in en_source.read_text(encoding="utf-8")
    zh_appendix_text = zh_appendix.read_text(encoding="utf-8")
    en_appendix_text = en_appendix.read_text(encoding="utf-8")
    assert zh_appendix_text.count("\\paragraph{命题 A.") == 8
    assert en_appendix_text.count("\\paragraph{Proposition A.") == 8
    theory_text = (ROOT / "theory" / "CHECKPOINT_AUDITABILITY_THEORY.md").read_text(
        encoding="utf-8"
    )
    assert "standard-source placeholders" not in theory_text
    assert "Citations should be inserted" not in theory_text
    standard_keys = {
        "lehmann2005testing",
        "vershynin2018highdimensional",
        "anderson2003multivariate",
        "bhatia2007positive",
    }
    bibliography = zh_bib.read_text(encoding="utf-8")
    assert all(f"{{{key}," in bibliography for key in standard_keys)
    assert receipt["chinese"]["pages"] == 26
    assert receipt["english"]["pages"] == 29
    assert receipt["chinese"]["overfull_boxes"] == 0
    assert receipt["english"]["overfull_boxes"] == 0
    assert receipt["chinese"]["undefined_citations_or_references"] == 0
    assert receipt["english"]["undefined_citations_or_references"] == 0
    assert receipt["regression_tests"] == "171/171 CURRENT FULL SUITE PASS"
    return {
        "status": "PASS",
        "bibliographies_byte_identical": True,
        "chinese_pages": 26,
        "english_pages": 29,
        "formal_propositions_per_language": 8,
        "regression_tests": receipt["regression_tests"],
    }


def verify_structured_novelty_search() -> dict[str, object]:
    receipt = json.loads(
        (PROGRAM / "structured_novelty_patent_search_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == "PASS_STRUCTURED_SEARCH_NOT_SYSTEMATIC_OR_LEGAL"
    assert receipt["search_date"] == "2026-10-02"
    assert len(receipt["publication_queries"]) >= 3
    assert len(receipt["patent_queries"]) >= 5
    assert len(receipt["closest_publication_records_rechecked"]) >= 7
    assert len(receipt["included_patent_records"]) >= 5
    decision = receipt["decision"]
    assert decision["exact_isomorphic_record_identified"] is False
    assert decision["single_component_priority_claim_allowed"] is False
    assert decision["combined_first_claim_allowed"] is False
    assert decision["negative_patent_result_is_exhaustive"] is False
    for name, expected in receipt["document_sha256"].items():
        assert sha256(PROGRAM / name) == expected
    return {
        "status": "PASS_STRUCTURED_NOT_SYSTEMATIC",
        "publication_records_rechecked": len(
            receipt["closest_publication_records_rechecked"]
        ),
        "patent_records_included": len(receipt["included_patent_records"]),
        "priority_claim_allowed": False,
        "freedom_to_operate_opinion": False,
    }


def required_file_paths() -> list[Path]:
    files = [
        ROOT / "README.md",
        ROOT / "requirements-replay.txt",
        PROGRAM / "FINAL_EVIDENCE_REPORT.md",
        PROGRAM / "ARTIFACT_INDEX.md",
        PROGRAM / "REPRODUCE.md",
        PROGRAM / "PAPER_TABLES.md",
        PROGRAM / "FINAL_CLAIM_MATRIX.csv",
        PROGRAM / "STATUS.md",
        PROGRAM / "TIFS_PUBLISHED_PAPER_AND_REVIEW_EVIDENCE_AUDIT_2026-10-03.md",
        PROGRAM / "INFORMATION_BUDGET_AND_RESOURCE_AUDIT_2026-10-03.md",
        PROGRAM / "PROTOCOL_DEVIATIONS.md",
        PROGRAM / "OFFICIAL_BASELINE_ADDENDUM.md",
        PROGRAM / "NEXT_EXPERIMENTS.md",
        PROGRAM / "TOP_JOURNAL_MASTER_PLAN_V2.md",
        PROGRAM / "TOP_JOURNAL_READINESS_GATE_2026-10-01.md",
        PROGRAM / "AUDITABILITY_THEORY_MULTICLASS_ADDENDUM.md",
        PROGRAM / "NOVELTY_AUDIT_ADDENDUM_2026-09-29.md",
        PROGRAM / "NOVELTY_REFRESH_2026-09-30.md",
        PROGRAM / "STRUCTURED_NOVELTY_AND_PATENT_SEARCH_2026-10-02.md",
        PROGRAM / "structured_novelty_patent_search_receipt.json",
        PROGRAM / "reproducibility_environment_2026-10-02.json",
        PROGRAM / "mnist_source_provenance_receipt_2026-10-02.json",
        PROGRAM / "MNIST_HISTORICAL_ENVIRONMENT_RECOVERY_AUDIT_2026-10-03.md",
        PROGRAM / "mnist_historical_environment_recovery_receipt_2026-10-03.json",
        PROGRAM / "PUBLIC_CODE_PROFILE_ADDENDUM_2026-10-03.md",
        PROGRAM / "public_code_profile_manifest_preterminal_2026-10-03.csv",
        PROGRAM / "public_code_profile_preflight_preterminal_2026-10-03.json",
        PROGRAM / "RELEASE_ARCHIVE_PLAN.md",
        PROGRAM / "BLIND_REVIEW_SANITIZATION_PLAN.md",
        PROGRAM / "release_archive_manifest_preterminal_2026-10-02.csv",
        PROGRAM / "release_archive_preflight_preterminal_2026-10-02.json",
        PROGRAM / "SUBMISSION_STATEMENTS_DRAFT.md",
        PROGRAM / "THIRD_PARTY_NOTICES.md",
        PROGRAM / "ATTACK_SOURCE_AUDIT.md",
        PROGRAM / "P6D_FINAL_OBSERVABLE_ADAPTIVE_ADDENDUM.md",
        PROGRAM / "p6d_final_observable_adaptive_protocol_v1.json",
        PROGRAM / "p6d_v1_formal_failure_receipt.json",
        PROGRAM / "P6D_V1_FORMAL_FAILURE.md",
        PROGRAM / "P6D_CODE_READINESS.md",
        PROGRAM / "p6d_code_readiness_receipt.json",
        PROGRAM / "p6d_final_observable_adaptive_protocol_v2.json",
        PROGRAM / "p6d_v2_formal_failure_receipt.json",
        PROGRAM / "P6D_V2_FORMAL_FAILURE.md",
        PROGRAM / "p6d_final_observable_adaptive_protocol_v3.json",
        PROGRAM / "p6d_v3_formal_confirmation_receipt.json",
        PROGRAM / "P6D_V3_CONFIRMATION_INCIDENT_AND_RESULT.md",
        PROGRAM / "THEORY_REFEREE_AUDIT_2026-09-30.md",
        PROGRAM / "MODERN_AGPD_ADAPTER_ADDENDUM.md",
        PROGRAM / "FRESH_QUERY_CALIBRATION_ADDENDUM.md",
        PROGRAM / "PATHMNIST_EXTERNAL_VALIDATION_ADDENDUM.md",
        PROGRAM / "PATHMNIST_STANDARD_ROUTE_ADDENDUM.md",
        PROGRAM / "PATHMNIST_OFFICIAL_WEIGHT_DETECTOR_CONFIRMATION_ADDENDUM.md",
        PROGRAM / "pathmnist_official_weight_detector_confirmation_protocol_v1.json",
        PROGRAM / "pathmnist_terminal_claim_integration_contract_v1.json",
        PROGRAM / "pathmnist_terminal_claim_integration_contract_v1_amendment_2026-10-02.json",
        PROGRAM / "PATHMNIST_MPS_EXECUTION_ADDENDUM.md",
        PROGRAM / "pathmnist_mps_smoke_receipt.json",
        PROGRAM / "P6C_QML_NOISE_BRIDGE_ADDENDUM.md",
        PROGRAM / "p6c_qml_noise_bridge_code_receipt.json",
        PROGRAM / "bilingual_manuscript_receipt.json",
        PROGRAM / "official_backdoorbench_confirmation_receipt.json",
        PROGRAM / "checkpoint_auditability_theory_protocol_v1.json",
        PROGRAM / "checkpoint_auditability_theory_receipt.json",
        PROGRAM / "AUDITABILITY_GEOMETRY_DIAGNOSTIC_ADDENDUM.md",
        PROGRAM / "auditability_geometry_diagnostic_protocol_v1.json",
        PROGRAM / "auditability_geometry_diagnostic_receipt.json",
        PROGRAM / "SINGLE_SAMPLE_REFERENCE_AUDIT_ADDENDUM.md",
        PROGRAM / "single_sample_reference_audit_protocol_v1.json",
        PROGRAM / "single_sample_reference_audit_receipt.json",
        PROGRAM / "MANUSCRIPT_CLAIM_CONSISTENCY_AUDIT_2026-10-02.md",
        PROGRAM / "manuscript_claim_consistency_receipt_2026-10-02.json",
        PROGRAM / "PATHMNIST_WEIGHT_REDISTRIBUTION_AUDIT_2026-10-02.md",
        PROGRAM / "pathmnist_weight_redistribution_audit_receipt_2026-10-02.json",
        PROGRAM / "multiclass_backdoorbench_pilot_protocol.json",
        PROGRAM / "multiclass_backdoorbench_program_receipt.json",
        PROGRAM / "multiclass_agpd_adapter_protocol_v1.json",
        PROGRAM / "multiclass_agpd_adapter_verification.json",
        PROGRAM / "fresh_query_calibration_protocol_v1.json",
        PROGRAM / "fresh_query_calibration_verification.json",
        PROGRAM / "reference_calibration_budget_sensitivity_protocol_v1.json",
        PROGRAM / "reference_calibration_budget_sensitivity_verification.json",
        PROGRAM / "pathmnist_acquisition_protocol_v1.json",
        PROGRAM / "pathmnist_acquisition_receipt.json",
        PROGRAM / "pathmnist_external_validation_protocol_v1.json",
        PROGRAM / "pathmnist_external_validation_verification.json",
        PROGRAM / "pathmnist_external_validation_protocol_v2.json",
        PROGRAM / "pathmnist_external_validation_verification_v2.json",
        PROGRAM / "pathmnist_warmstart_protocol_v1.json",
        PROGRAM / "pathmnist_warmstart_verification.json",
        PROGRAM / "pathmnist_official_weights_acquisition_receipt.json",
        PROGRAM / "pathmnist_resnet18_victim_pilot_protocol_v1.json",
        PROGRAM / "pathmnist_official_weight_warmstart_protocol_v1.json",
        ROOT / "manuscript_zh" / "main_zh.tex",
        ROOT / "manuscript_zh" / "theory_appendix_zh.tex",
        ROOT / "manuscript_zh" / "references_zh.bib",
        ROOT / "manuscript_zh" / "CLAIMS_EVIDENCE_ZH.md",
        ROOT / "manuscript_zh" / "output" / "pdf" / "main_zh.pdf",
        ROOT / "manuscript_en" / "main_en.tex",
        ROOT / "manuscript_en" / "theory_appendix.tex",
        ROOT / "manuscript_en" / "references_en.bib",
        ROOT / "manuscript_en" / "output" / "pdf" / "main_en.pdf",
        ROOT / "configs" / "p6c_qml_noise_bridge_v1.json",
        ROOT / "theory" / "CHECKPOINT_AUDITABILITY_THEORY.md",
        ROOT / "src" / "verify_top_journal_final_bundle.py",
        ROOT / "src" / "build_release_archive_manifest.py",
        ROOT / "src" / "build_blind_review_package.py",
        ROOT / "src" / "verify_blind_review_package.py",
        ROOT / "src" / "build_public_code_profile.py",
        ROOT / "src" / "make_top_journal_figures.py",
        ROOT / "src" / "make_auditability_multiclass_figure.py",
        ROOT / "src" / "backdoorbench_feature_adapter.py",
        ROOT / "src" / "run_bloodmnist_official_backdoorbench_pilot.py",
        ROOT / "src" / "verify_official_backdoorbench_confirmation.py",
        ROOT / "src" / "analyze_checkpoint_auditability_theory_probe.py",
        ROOT / "src" / "verify_checkpoint_auditability_theory_probe.py",
        ROOT / "src" / "analyze_auditability_geometry_diagnostic.py",
        ROOT / "src" / "verify_auditability_geometry_diagnostic.py",
        ROOT / "src" / "analyze_single_sample_reference_audit.py",
        ROOT / "src" / "verify_single_sample_reference_audit.py",
        ROOT / "src" / "verify_manuscript_claim_consistency.py",
        ROOT / "src" / "run_bloodmnist_multiclass_backdoorbench_pilot.py",
        ROOT / "src" / "verify_multiclass_backdoorbench_program.py",
        ROOT / "src" / "agpd_feature_adapter.py",
        ROOT / "src" / "run_bloodmnist_multiclass_agpd_adapter.py",
        ROOT / "src" / "verify_bloodmnist_multiclass_agpd_adapter.py",
        ROOT / "src" / "run_bloodmnist_fresh_query_calibration.py",
        ROOT / "src" / "verify_bloodmnist_fresh_query_calibration.py",
        ROOT / "src" / "make_fresh_query_calibration_figure.py",
        ROOT / "src" / "run_reference_calibration_budget_sensitivity.py",
        ROOT / "src" / "verify_reference_calibration_budget_sensitivity.py",
        ROOT / "src" / "pathmnist_external_validation.py",
        ROOT / "src" / "verify_pathmnist_acquisition.py",
        ROOT / "src" / "run_pathmnist_external_validation.py",
        ROOT / "src" / "verify_pathmnist_external_validation.py",
        ROOT / "src" / "run_pathmnist_external_validation_v2.py",
        ROOT / "src" / "verify_pathmnist_external_validation_v2.py",
        ROOT / "src" / "run_pathmnist_warmstart_pilot.py",
        ROOT / "src" / "verify_pathmnist_warmstart_pilot.py",
        ROOT / "src" / "pathmnist_official_resnet18.py",
        ROOT / "src" / "verify_pathmnist_official_weights.py",
        ROOT / "src" / "run_pathmnist_resnet18_victim_pilot.py",
        ROOT / "src" / "verify_pathmnist_resnet18_victim_pilot.py",
        ROOT / "src" / "run_pathmnist_official_weight_warmstart.py",
        ROOT / "src" / "verify_pathmnist_official_weight_warmstart.py",
        ROOT / "src" / "run_pathmnist_official_weight_detector_confirmation.py",
        ROOT / "src" / "verify_pathmnist_official_weight_detector_confirmation.py",
        ROOT / "src" / "run_p6c_qml_noise_bridge.py",
        ROOT / "src" / "verify_p6c_qml_noise_bridge.py",
        ROOT / "src" / "make_p6c_qml_noise_figure.py",
        ROOT / "src" / "run_p6d_final_observable_adaptive.py",
        ROOT / "src" / "verify_p6d_final_observable_adaptive.py",
        ROOT / "src" / "verify_p6d_v1_formal_failure.py",
        ROOT / "src" / "run_p6d_final_observable_adaptive_v2.py",
        ROOT / "src" / "verify_p6d_final_observable_adaptive_v2.py",
        ROOT / "src" / "verify_p6d_v2_formal_failure.py",
        ROOT / "src" / "run_p6d_final_observable_adaptive_v3.py",
        ROOT / "src" / "verify_p6d_final_observable_adaptive_v3.py",
        ROOT / "src" / "verify_p6d_v3_confirmation_delivery.py",
        ROOT / "tests" / "test_pathmnist_resnet18_victim_pilot.py",
        ROOT / "tests" / "test_pathmnist_official_weight_warmstart.py",
        ROOT / "tests" / "test_pathmnist_official_weight_detector_confirmation.py",
        ROOT / "tests" / "test_p6c_qml_noise_bridge.py",
        ROOT / "tests" / "test_p6d_final_observable_adaptive.py",
        ROOT / "tests" / "test_p6d_v1_formal_failure.py",
        ROOT / "tests" / "test_p6d_final_observable_adaptive_v2.py",
        ROOT / "tests" / "test_p6d_v2_formal_failure.py",
        ROOT / "tests" / "test_p6d_final_observable_adaptive_v3.py",
        ROOT / "tests" / "test_p6d_v3_confirmation_delivery.py",
        ROOT / "tests" / "test_auditability_geometry_diagnostic.py",
        ROOT / "tests" / "test_single_sample_reference_audit.py",
        ROOT / "tests" / "test_manuscript_claim_consistency.py",
        ROOT / "tests" / "test_reference_calibration_budget_sensitivity.py",
        ROOT / "tests" / "test_top_journal_bundle_inventory.py",
        RESULTS / "auditability_geometry_diagnostic_v1" / "manifest.json",
        RESULTS / "auditability_geometry_diagnostic_v1" / "decision.json",
        RESULTS / "auditability_geometry_diagnostic_v1" / "EXPERIMENT_REPORT.md",
        RESULTS / "auditability_geometry_diagnostic_v1" / "phase_a_per_configuration.csv",
        RESULTS / "auditability_geometry_diagnostic_v1" / "phase_a_aggregate.csv",
        RESULTS / "auditability_geometry_diagnostic_v1" / "blood_per_victim.csv",
        RESULTS / "auditability_geometry_diagnostic_v1" / "blood_population_summary.csv",
        RESULTS / "auditability_geometry_diagnostic_v1" / "blood_primary_paired_contrasts.csv",
        RESULTS / "auditability_geometry_diagnostic_v1" / "source_reconstruction_receipts.json",
        RESULTS / "single_sample_reference_audit_v1" / "manifest.json",
        RESULTS / "single_sample_reference_audit_v1" / "decision.json",
        RESULTS / "single_sample_reference_audit_v1" / "EXPERIMENT_REPORT.md",
        RESULTS / "single_sample_reference_audit_v1" / "per_victim_metrics.csv",
        RESULTS / "single_sample_reference_audit_v1" / "per_sample_scores.csv",
        RESULTS / "single_sample_reference_audit_v1" / "population_summary.csv",
        RESULTS / "single_sample_reference_audit_v1" / "feature_receipts.json",
        RESULTS / "reference_calibration_budget_sensitivity_v1" / "manifest.json",
        RESULTS / "reference_calibration_budget_sensitivity_v1" / "EXPERIMENT_REPORT.md",
        RESULTS / "reference_calibration_budget_sensitivity_v1" / "per_seed_grid_metrics.csv",
        RESULTS / "reference_calibration_budget_sensitivity_v1" / "per_sample_grid_scores.csv",
        RESULTS / "reference_calibration_budget_sensitivity_v1" / "eligible_grid_summary.csv",
        RESULTS / "pathmnist_official_weights_clean_validation_v1" / "verification.json",
        RESULTS / "pathmnist_resnet18_victim_pilot_dry_run_official_compat" / "verification.json",
        RESULTS / "pathmnist_official_weight_warmstart_v1" / "manifest.json",
        RESULTS / "pathmnist_official_weight_warmstart_v1" / "verification.json",
        RESULTS / "pathmnist_official_weight_warmstart_v1" / "victim_metrics.csv",
        RESULTS / "pathmnist_official_weight_warmstart_v1" / "seed_receipts.json",
        RESULTS / "pathmnist_official_weight_warmstart_v1" / "models" / "seed_93_epoch5.pt",
        RESULTS / "pathmnist_official_weight_warmstart_v1" / "models" / "seed_94_epoch5.pt",
        RESULTS / "pathmnist_real_input_mps_smoke_v1" / "smoke_receipt.json",
        RESULTS / "pathmnist_real_input_mps_smoke_v1" / "smoke_verification.json",
        RESULTS / "pathmnist_real_input_mps_smoke_v1" / "feature_metric_replay.npz",
        RESULTS / "p6c_qml_noise_bridge_v1" / "manifest.json",
        RESULTS / "p6c_qml_noise_bridge_v1" / "paired_summary.json",
        RESULTS / "p6c_qml_noise_bridge_v1" / "per_seed_metrics.csv",
        RESULTS / "p6c_qml_noise_bridge_v1" / "per_sample_scores.csv",
        RESULTS / "p6c_qml_noise_bridge_v1" / "diagnostic_2048" / "manifest.json",
        RESULTS / "p6c_qml_noise_bridge_v1" / "diagnostic_2048" / "per_seed_metrics.csv",
        RESULTS / "p6c_qml_noise_bridge_v1" / "diagnostic_2048" / "per_sample_scores.csv",
        RESULTS / "p6d_final_observable_adaptive_v3" / "development" / "manifest.json",
        RESULTS / "p6d_final_observable_adaptive_v3" / "confirmation_lock_receipt.json",
        RESULTS / "p6d_final_observable_adaptive_v3" / "confirmation" / "manifest.json",
        RESULTS / "p6d_final_observable_adaptive_v3" / "confirmation" / "victim_metrics.csv",
        RESULTS / "p6d_final_observable_adaptive_v3" / "confirmation" / "per_seed_metrics.csv",
        RESULTS / "p6d_final_observable_adaptive_v3" / "confirmation" / "confirmation_summary.json",
    ]
    for index in range(1, 6):
        stems = {
            1: "fig01_strong_baseline_comparison",
            2: "fig02_adaptive_architecture_forest",
            3: "fig03_mechanism_dimension_ablations",
            4: "fig04_boundary_heatmaps",
            5: "fig05_post_sanitization",
        }
        for suffix in ("png", "pdf"):
            files.append(PROGRAM / "figures" / f"{stems[index]}.{suffix}")
    for suffix in ("png", "pdf"):
        files.append(PROGRAM / "figures" / f"fig06_auditability_multiclass.{suffix}")
        files.append(PROGRAM / "figures" / f"fig07_fresh_query_calibration.{suffix}")
        files.append(PROGRAM / "figures" / f"fig08_qml_noise_bridge.{suffix}")
    vendor_root = ROOT / "external" / "BackdoorBench"
    files.extend(
        sorted(
            (
                path
                for path in vendor_root.rglob("*")
                if path.is_file()
                and ".git" not in path.relative_to(vendor_root).parts
                and "__pycache__" not in path.relative_to(vendor_root).parts
            ),
            key=lambda path: str(path.relative_to(ROOT)),
        )
    )
    files.extend(
        RESULTS / "reference_calibration_budget_sensitivity_v1" / "feature_cache" / f"seed_{seed}_xz16_drift.npz"
        for seed in range(62, 72)
    )
    return files


def verify_required_files() -> list[str]:
    files = required_file_paths()
    assert len(files) == len(set(files)), "required bundle inventory contains duplicates"
    excluded = {
        relative
        for relative, detail in staged_ledger_index().items()
        if detail.get("profile_action") == "EXCLUDED_WEIGHT_BINARY"
    }
    missing = [
        str(path)
        for path in files
        if (not path.exists() and str(path.relative_to(ROOT)) not in excluded)
        or (
            path.exists()
            and path.stat().st_size == 0
            and not str(path.relative_to(ROOT)).startswith(
                "external/BackdoorBench/"
            )
        )
    ]
    assert not missing, missing
    return [str(path.relative_to(ROOT)) for path in files]


def verify_release_archive_manifest() -> dict[str, object]:
    from build_public_code_profile import project_code_files
    from build_release_archive_manifest import release_payload_paths

    outputs = {RELEASE_MANIFEST.resolve(), RELEASE_PREFLIGHT.resolve()}
    expected_paths = sorted({
        str(path.resolve().relative_to(ROOT.resolve()))
        for path in release_payload_paths()
        if path.resolve() not in outputs
    })
    rows = read_csv(RELEASE_MANIFEST)
    actual_paths = [row["path"] for row in rows]
    assert actual_paths == expected_paths
    assert len(actual_paths) == len(set(actual_paths))
    project_code_paths = {
        str(path.resolve().relative_to(ROOT.resolve()))
        for path in project_code_files()
    }
    assert project_code_paths <= set(actual_paths)
    ledger_index = staged_ledger_index()
    for row in rows:
        path = ROOT / row["path"]
        if path.is_file():
            assert int(row["size_bytes"]) == effective_size(path, ROOT)
            assert row["sha256"] == sha256(path)
        else:
            detail = ledger_index.get(row["path"])
            assert detail is not None
            assert detail["profile_action"] == "EXCLUDED_WEIGHT_BINARY"
            assert int(row["size_bytes"]) == int(detail["source_size_bytes"])
            assert row["sha256"] == detail["source_sha256"]
        assert row["license_status"]
        assert row["release_action"]
    preflight = json.loads(RELEASE_PREFLIGHT.read_text(encoding="utf-8"))
    assert preflight["schema"] == "paper14_release_archive_preflight_v1"
    assert preflight["status"] == "PRETERMINAL_NOT_RELEASE_READY"
    assert preflight["manifest"]["sha256"] == sha256(RELEASE_MANIFEST)
    assert preflight["manifest"]["payload_file_count"] == len(rows)
    assert preflight["manifest"]["payload_total_bytes"] == sum(
        int(row["size_bytes"]) for row in rows
    )
    assert preflight["raw_benchmark_archives_included"] is False
    assert preflight["backdoorbench_snapshot_fully_inventoried"] is True
    assert preflight["claim_c27_state"] == "PENDING_TERMINAL"
    assert preflight["active_blocker_count"] == 8
    assert preflight["closed_control_count"] == 1
    blocker_ids = {item["id"] for item in preflight["blockers"]}
    assert blocker_ids == {
        "B01_TERMINAL_EXTERNAL_VALIDITY",
        "B02_PROJECT_CODE_LICENSE",
        "B03_ARTIFACT_LICENSE",
        "B04_WEIGHT_REDISTRIBUTION",
        "B05_ANONYMITY_AND_PORTABILITY",
        "B06_PUBLIC_ACCESSION",
        "B07_AUTHOR_STATEMENTS",
        "B08_MNIST_HISTORICAL_BOUNDARY",
        "B09_CLEAN_ARCHIVE_REPLAY",
    }
    return {
        "status": preflight["status"],
        "payload_file_count": len(rows),
        "payload_total_bytes": preflight["manifest"]["payload_total_bytes"],
        "manifest_sha256": sha256(RELEASE_MANIFEST),
        "preflight_sha256": sha256(RELEASE_PREFLIGHT),
        "backdoorbench_file_count": sum(
            row["path"].startswith("external/BackdoorBench/") for row in rows
        ),
        "project_code_file_count": len(project_code_paths),
        "local_user_path_file_count": preflight["text_risk_counts"].get(
            "LOCAL_USER_PATH", 0
        ),
        "release_blocker_count": len(preflight["blockers"]),
        "active_release_blocker_count": preflight["active_blocker_count"],
        "closed_release_control_count": preflight["closed_control_count"],
    }


def main() -> None:
    pathmnist_detector = verify_pathmnist_detector_confirmation_readiness()
    pathmnist_detector["terminal_claim_route"] = (
        select_pathmnist_terminal_claim_route(pathmnist_detector)
    )
    receipt = {
        "status": "PASS",
        "experiment_program": "cf_cmd_top_journal_v1",
        "final_verifier": final_verifier_provenance(),
        "reproducibility_environment": verify_reproducibility_environment(),
        "mnist_source_provenance": verify_mnist_source_provenance(),
        "mnist_historical_environment_recovery": verify_mnist_historical_environment_recovery(),
        "public_code_profile": verify_public_code_profile(),
        "release_archive_preflight": verify_release_archive_manifest(),
        "prior_receipts": verify_prior_receipts(),
        "protocol_hashes_verified": verify_protocol_hashes(),
        "new_program_replays": verify_new_program_receipts(),
        "auditability_geometry_diagnostic": verify_auditability_geometry_diagnostic(),
        "single_sample_reference_audit": verify_single_sample_reference_audit(),
        "reference_calibration_budget_sensitivity": verify_reference_calibration_budget_sensitivity(),
        "manuscript_claim_consistency": verify_manuscript_claim_consistency(),
        "pathmnist_weight_redistribution_audit": verify_pathmnist_weight_redistribution_audit(),
        "pathmnist_standard_readiness": verify_pathmnist_standard_readiness(),
        "pathmnist_detector_confirmation_readiness": pathmnist_detector,
        "pathmnist_terminal_claim_contract": verify_pathmnist_terminal_claim_contract(
            pathmnist_detector["terminal_claim_route"]
        ),
        "p6c_qml_noise_bridge": verify_p6c_noise_bridge(),
        "p6d_code_readiness": verify_p6d_code_readiness(),
        "p6d_v1_formal_failure": verify_p6d_v1_formal_failure(),
        "p6d_v2_failure_and_v3_recovery": verify_p6d_v2_failure_and_v3_recovery(),
        "p6d_v3_formal_confirmation": verify_p6d_v3_formal_confirmation(),
        "bilingual_manuscripts": verify_bilingual_manuscript_receipt(),
        "structured_novelty_search": verify_structured_novelty_search(),
        "sample_replays": {
            "residual_reupload": replay_sample_table(
                RESULTS / "bloodmnist_residual_reupload_confirmation_v1",
                "per_sample_scores.csv",
                "per_config_metrics.csv",
                ("seed", "trigger", "representation"),
            ),
            "strong_baselines": replay_sample_table(
                RESULTS / "bloodmnist_strong_baselines_v1",
                "per_sample_scores_baselines.csv",
                "per_config_metrics.csv",
                ("scenario", "seed", "view", "method"),
                {"spectral_signatures", "spectre_que", "activation_clustering", "scan_style"},
            ),
            "mechanism_ablations": replay_sample_table(
                RESULTS / "bloodmnist_mechanism_ablations_v1",
                "per_sample_scores.csv",
                "per_config_metrics.csv",
                ("seed", "trigger", "representation"),
            ),
            "covariance_free_confirmation": replay_sample_table(
                RESULTS / "bloodmnist_covariance_free_confirmation_v1",
                "per_sample_scores.csv",
                "per_config_metrics.csv",
                ("seed", "representation", "scenario"),
            ),
        },
        "boundary_replay": verify_boundary(),
        "post_sanitization_replay": verify_post_sanitization(),
        "required_files": verify_required_files(),
        "limitations": [
            "The original SPECTRE/QUE and SCAn-style table is clean-room. The addendum directly calls pinned upstream feature primitives, but it is not a bit-identical end-to-end BackdoorBench pipeline.",
            "Official SPECTRE receives oracle target/budget information; SCAn's binary anomaly index is structurally degenerate, while the completed five-seed eight-class confirmation shows that suspect-only SCAn is strong in its applicable setting.",
            "Boundary prevalence changes the evaluated candidate pool under fixed victims; it is not a retrained poison-ratio experiment.",
            "The covariance-free adaptive confirmation has only two attack-eligible seeds and is inconclusive.",
            "The trusted-reference/calibration grid is a retrospective checkpoint-only sensitivity analysis. It shows a severe eight-reference-row failure and does not establish a universal 32-sample minimum or authorize post-hoc budget selection.",
            "The frozen checkpoint-auditability probe uses independently trained checkpoints, so it rejects a general drift advantage but does not test the warm-start coordinate-aligned proposition.",
            "Training-member candidate scores are not proven exchangeable with validation calibration scores; they are clean-calibrated rank p-values, not unqualified exact conformal p-values.",
            "The AGPD experiment is a feature-level two-layer CNN adapter with declared short-layer/window repairs; its low TPR is an applicability boundary, not a native-architecture end-to-end ranking.",
            "Fresh test queries do not uniformly improve rank calibration: the member-minus-fresh FPR gap reverses between suspect hidden64 and paired drift, and official validation/test exchangeability remains an assumption.",
            "The lightweight PathMNIST route is a victim-feasibility boundary, not a confirmed external-dataset detector result: only one of four from-scratch victims and none of two coordinate-aligned warm-start victims met the CA/ASR gate.",
            "PathMNIST detector metrics from ineligible lightweight victims are diagnostic only and are excluded from efficacy claims; the completed standard ResNet-18 victim gate authorizes, but does not replace, a fresh detector confirmation.",
            "The official-weight PathMNIST ResNet-18 formal route produced two eligible victims, but it trained neither a coordinate-aligned clean pair nor a detector; victim feasibility is not detector evidence.",
            pathmnist_detector_limitation(pathmnist_detector),
            "The QML noise bridge is simulator-only. Nominal-noise ranking survives, but the exact-calibrated FPR rises to 71.3%; 2048 shots do not repair the operating point, and no real-hardware claim is supported.",
            "P6-D v1 and v2 consumed 95--97 and 108--110, respectively, then failed before test access on distinct CSV-schema and nested-NumPy JSON delivery errors; 98--102 and 111--115 are closed. V3 completed 116--123 and passes a stage-aware forensic replay, but only four of five confirmation victims met the frozen attack gate, so the preregistered primary efficacy endpoint is inconclusive and no pooled adaptive-efficacy claim is supported.",
        ],
    }
    output = PROGRAM / "final_bundle_receipt.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
