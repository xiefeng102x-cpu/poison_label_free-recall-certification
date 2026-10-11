from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import verify_top_journal_final_bundle as verifier  # noqa: E402
import build_blind_review_package as blind_review  # noqa: E402
import verify_blind_review_package as blind_verify  # noqa: E402
from staged_provenance import raw_sha256  # noqa: E402


PROGRAM = PAPER14 / "results/top_journal_program_v1"
HISTORICAL_RELEASE_MANIFEST_SHA256 = (
    "031f54e7403a4939ba593373c077c9f63694752db7386b0f83f17f401c692e78"
)
HISTORICAL_RELEASE_PREFLIGHT_SHA256 = (
    "967f05f784d279e847b5f1b7a7a8e508b28f0f84573af97d6a91e8c168805f74"
)
HISTORICAL_PUBLIC_PROFILE_MANIFEST_SHA256 = (
    "8e6f64409ae6794029d94d71770bc0ad6fd0d50fb7bdb9f21214198b039c24b2"
)
HISTORICAL_PUBLIC_PROFILE_PREFLIGHT_SHA256 = (
    "01b5626ab2e1aebf4fdeeb2afc558f2566e2f31fff79d1f73c5600981096213e"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_v2_migration_is_active() -> dict[str, object]:
    receipt = json.loads(
        (PROGRAM / "final_bundle_v2_migration_readiness_preterminal_2026-10-04.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == "PASS_PRETERMINAL_MIGRATION"
    assert receipt["stale_v1_preserved_not_current"] is True
    assert receipt["submission_ready"] is False
    return receipt


def test_required_bundle_inventory_is_unique_and_keeps_public_entry_points() -> None:
    files = verifier.verify_required_files()
    assert len(files) == len(set(files))
    required = {
        "README.md",
        "requirements-replay.txt",
        "results/top_journal_program_v1/FINAL_EVIDENCE_REPORT.md",
        "results/top_journal_program_v1/ARTIFACT_INDEX.md",
        "results/top_journal_program_v1/REPRODUCE.md",
        "results/top_journal_program_v1/TIFS_PUBLISHED_PAPER_AND_REVIEW_EVIDENCE_AUDIT_2026-10-03.md",
        "results/top_journal_program_v1/INFORMATION_BUDGET_AND_RESOURCE_AUDIT_2026-10-03.md",
        "results/top_journal_program_v1/TOP_JOURNAL_READINESS_GATE_2026-10-01.md",
        "results/top_journal_program_v1/pathmnist_terminal_claim_integration_contract_v1.json",
        "results/top_journal_program_v1/pathmnist_terminal_claim_integration_contract_v1_amendment_2026-10-02.json",
        "results/top_journal_program_v1/reproducibility_environment_2026-10-02.json",
        "results/top_journal_program_v1/mnist_source_provenance_receipt_2026-10-02.json",
        "results/top_journal_program_v1/MNIST_HISTORICAL_ENVIRONMENT_RECOVERY_AUDIT_2026-10-03.md",
        "results/top_journal_program_v1/mnist_historical_environment_recovery_receipt_2026-10-03.json",
        "results/top_journal_program_v1/PUBLIC_CODE_PROFILE_ADDENDUM_2026-10-03.md",
        "results/top_journal_program_v1/public_code_profile_manifest_preterminal_2026-10-03.csv",
        "results/top_journal_program_v1/public_code_profile_preflight_preterminal_2026-10-03.json",
        "results/top_journal_program_v1/RELEASE_ARCHIVE_PLAN.md",
        "results/top_journal_program_v1/BLIND_REVIEW_SANITIZATION_PLAN.md",
        "results/top_journal_program_v1/PATHMNIST_WEIGHT_REDISTRIBUTION_AUDIT_2026-10-02.md",
        "results/top_journal_program_v1/pathmnist_weight_redistribution_audit_receipt_2026-10-02.json",
        "results/top_journal_program_v1/release_archive_manifest_preterminal_2026-10-02.csv",
        "results/top_journal_program_v1/release_archive_preflight_preterminal_2026-10-02.json",
        "results/top_journal_program_v1/SUBMISSION_STATEMENTS_DRAFT.md",
        "results/top_journal_program_v1/THIRD_PARTY_NOTICES.md",
        "results/top_journal_program_v1/ATTACK_SOURCE_AUDIT.md",
        "results/top_journal_program_v1/AUDITABILITY_GEOMETRY_DIAGNOSTIC_ADDENDUM.md",
        "results/top_journal_program_v1/auditability_geometry_diagnostic_protocol_v1.json",
        "results/top_journal_program_v1/auditability_geometry_diagnostic_receipt.json",
        "results/auditability_geometry_diagnostic_v1/manifest.json",
        "results/top_journal_program_v1/SINGLE_SAMPLE_REFERENCE_AUDIT_ADDENDUM.md",
        "results/top_journal_program_v1/single_sample_reference_audit_protocol_v1.json",
        "results/top_journal_program_v1/single_sample_reference_audit_receipt.json",
        "results/single_sample_reference_audit_v1/manifest.json",
        "results/top_journal_program_v1/MANUSCRIPT_CLAIM_CONSISTENCY_AUDIT_2026-10-02.md",
        "results/top_journal_program_v1/manuscript_claim_consistency_receipt_2026-10-02.json",
        "results/top_journal_program_v1/reference_calibration_budget_sensitivity_protocol_v1.json",
        "results/top_journal_program_v1/reference_calibration_budget_sensitivity_verification.json",
        "results/reference_calibration_budget_sensitivity_v1/manifest.json",
        "external/BackdoorBench/LICENSE",
        "manuscript_zh/theory_appendix_zh.tex",
        "manuscript_zh/output/pdf/main_zh.pdf",
        "manuscript_en/theory_appendix.tex",
        "manuscript_en/output/pdf/main_en.pdf",
        "src/verify_top_journal_final_bundle.py",
        "src/build_release_archive_manifest.py",
        "src/build_blind_review_package.py",
        "src/verify_blind_review_package.py",
        "src/build_public_code_profile.py",
        "src/analyze_auditability_geometry_diagnostic.py",
        "src/verify_auditability_geometry_diagnostic.py",
        "tests/test_auditability_geometry_diagnostic.py",
        "src/analyze_single_sample_reference_audit.py",
        "src/verify_single_sample_reference_audit.py",
        "tests/test_single_sample_reference_audit.py",
        "src/verify_manuscript_claim_consistency.py",
        "tests/test_manuscript_claim_consistency.py",
        "src/run_reference_calibration_budget_sensitivity.py",
        "src/verify_reference_calibration_budget_sensitivity.py",
        "tests/test_reference_calibration_budget_sensitivity.py",
        "tests/test_top_journal_bundle_inventory.py",
    }
    assert required <= set(files)
    provenance = verifier.final_verifier_provenance()
    verifier_path = PAPER14 / provenance["path"]
    assert provenance["path"] == "src/verify_top_journal_final_bundle.py"
    assert provenance["sha256"] == verifier.sha256(verifier_path)
    assert len(provenance["sha256"]) == 64


def test_bilingual_theory_appendix_has_eight_scoped_propositions() -> None:
    receipt = json.loads(
        (PROGRAM / "bilingual_manuscript_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "PASS"
    assert receipt["chinese"]["pages"] == 28
    assert receipt["english"]["pages"] == 30
    assert receipt["chinese"]["source_sha256"] == sha256(
        PAPER14 / "manuscript_zh/main_zh.tex"
    )
    assert receipt["chinese"]["pdf_sha256"] == sha256(
        PAPER14 / "manuscript_zh/output/pdf/main_zh.pdf"
    )
    assert receipt["english"]["source_sha256"] == sha256(
        PAPER14 / "manuscript_en/main_en.tex"
    )
    assert receipt["english"]["pdf_sha256"] == sha256(
        PAPER14 / "manuscript_en/output/pdf/main_en.pdf"
    )
    assert (
        PAPER14 / "manuscript_zh/theory_appendix_zh.tex"
    ).read_text(encoding="utf-8").count("\\paragraph{命题 A.") == 8
    assert (
        PAPER14 / "manuscript_en/theory_appendix.tex"
    ).read_text(encoding="utf-8").count("\\paragraph{Proposition A.") == 8
    assert_v2_migration_is_active()


def test_realized_geometry_diagnostic_is_hash_bound_and_negative() -> None:
    result = verifier.verify_auditability_geometry_diagnostic()
    assert result["status"] == "PASS"
    assert result["phase_a_configurations"] == 36
    assert result["bloodmnist_victims"] == 30
    assert result["paired_contrasts"] == 5
    assert result["decision"] == "NO_CLEAN_CANCELLATION_DIAGNOSIS"


def test_single_sample_reference_audit_is_hash_bound_and_negative() -> None:
    result = verifier.verify_single_sample_reference_audit()
    assert result["status"] == "PASS"
    assert result["victims"] == 30
    assert result["sample_scores"] == 25800
    assert result["population_summaries"] == 6
    assert result["decision"] == "NO_RELIABLE_SINGLE_SAMPLE_CHANNEL_IN_THIS_ARCHIVE"


def test_reference_calibration_budget_grid_is_hash_bound_and_replayed() -> None:
    result = verifier.verify_reference_calibration_budget_sensitivity()
    assert result["status"] == "PASS"
    assert result["budget_cells"] == 15
    assert result["seed_grid_instances"] == 150
    assert result["sample_rows"] == 99000
    assert result["original_cell_exact_replay"] is True


def test_bilingual_manuscript_claim_consistency_receipts_preserve_history_and_bind_terminal() -> None:
    historical = json.loads(
        (PROGRAM / "manuscript_claim_consistency_receipt_2026-10-02.json").read_text(
            encoding="utf-8"
        )
    )
    assert historical["status"] == "PASS"
    assert historical["errors"] == []
    assert historical["required_bilingual_tokens_checked"] == 74
    assert historical["pathmnist_partial_artifacts_read"] is False
    assert historical["verifier_sha256"] == (
        "9b7954b1e7cbc8eb4398c445f78db71508d37adb20fe67673b9f82b9b990acd7"
    )

    terminal = json.loads(
        (PROGRAM / "manuscript_claim_consistency_terminal_v2_2026-10-05.json").read_text(
            encoding="utf-8"
        )
    )
    assert terminal["status"] == "PASS"
    assert terminal["errors"] == []
    assert terminal["required_bilingual_tokens_checked"] == 76
    assert terminal["pathmnist_terminal_capsule_read"] is True
    assert terminal["pathmnist_raw_or_partial_performance_artifacts_read"] is False
    assert terminal["verifier_sha256"] == sha256(
        PAPER14 / "src/verify_manuscript_claim_consistency.py"
    )
    assert_v2_migration_is_active()


def test_pathmnist_limitation_tracks_terminal_gate_state() -> None:
    running_summary = {"status": "FORMAL_RUNNING"}
    running = verifier.pathmnist_detector_limitation(running_summary)
    assert "running" in running
    assert "no detector endpoint" in running
    assert (
        verifier.select_pathmnist_terminal_claim_route(running_summary)
        == "PENDING_TERMINAL"
    )

    ineligible_summary = {
        "status": "FORMAL_VERIFIED",
        "eligible_seed_count": 4,
        "all_pairs_eligible": False,
        "primary_point_gate_pass": False,
    }
    ineligible = verifier.pathmnist_detector_limitation(ineligible_summary)
    assert "4 of 5" in ineligible
    assert "no external-domain efficacy endpoint" in ineligible
    assert (
        verifier.select_pathmnist_terminal_claim_route(ineligible_summary)
        == "R1_INSUFFICIENT_ELIGIBLE_VICTIMS"
    )
    assert (
        verifier.expected_pathmnist_claim_decision(
            "R1_INSUFFICIENT_ELIGIBLE_VICTIMS"
        )
        == "INCONCLUSIVE"
    )

    point_fail_summary = {
        "status": "FORMAL_VERIFIED",
        "eligible_seed_count": 5,
        "all_pairs_eligible": True,
        "primary_point_gate_pass": False,
    }
    point_fail = verifier.pathmnist_detector_limitation(point_fail_summary)
    assert "primary detector point gate fails" in point_fail
    assert (
        verifier.select_pathmnist_terminal_claim_route(point_fail_summary)
        == "R2_ELIGIBLE_PRIMARY_GATE_FAIL"
    )
    assert (
        verifier.expected_pathmnist_claim_decision(
            "R2_ELIGIBLE_PRIMARY_GATE_FAIL"
        )
        == "REJECTED"
    )

    point_pass_summary = {
        "status": "FORMAL_VERIFIED",
        "eligible_seed_count": 5,
        "all_pairs_eligible": True,
        "primary_point_gate_pass": True,
    }
    point_pass = verifier.pathmnist_detector_limitation(point_pass_summary)
    assert "passes its frozen victim and primary point gates" in point_pass
    assert "does not establish" in point_pass
    assert (
        verifier.select_pathmnist_terminal_claim_route(point_pass_summary)
        == "R3_ELIGIBLE_PRIMARY_GATE_PASS"
    )
    assert (
        verifier.expected_pathmnist_claim_decision(
            "R3_ELIGIBLE_PRIMARY_GATE_PASS"
        )
        == "SUPPORTED_CONDITIONALLY"
    )


def test_pathmnist_terminal_claim_contract_is_frozen_and_hash_bound() -> None:
    result = verifier.verify_pathmnist_terminal_claim_contract(
        "R3_ELIGIBLE_PRIMARY_GATE_PASS"
    )
    assert result["status"] == "FROZEN_PRE_TERMINAL"
    assert result["routes"] == [
        "R0_INVALID_OR_INCOMPLETE",
        "R1_INSUFFICIENT_ELIGIBLE_VICTIMS",
        "R2_ELIGIBLE_PRIMARY_GATE_FAIL",
        "R3_ELIGIBLE_PRIMARY_GATE_PASS",
    ]
    assert result["mandatory_integration_targets"] == 16
    assert result["integration_target_paths_verified"] is True
    assert result["pending_claim_id"] == "C27"
    assert result["selected_route"] == "R3_ELIGIBLE_PRIMARY_GATE_PASS"
    assert result["expected_claim_decision"] == "SUPPORTED_CONDITIONALLY"
    assert result["forbidden_actions"] == 7
    assert len(result["contract_sha256"]) == 64
    assert result["logistics_amendment_sha256"] == (
        "d98f677260fbf4226d503cf9b1e65c8651c32cdbdc1f68ebed7de9aed788815e"
    )


def test_reproducibility_environment_is_hash_bound_and_matches_host() -> None:
    result = verifier.verify_reproducibility_environment()
    assert result["status"] == "MATCH_RECORDED_EVIDENCE_HOST"
    assert result["package_count"] == 12
    assert result["mismatches"] == {}
    assert result["cross_platform_bit_identity_claimed"] is False
    assert result["requirements_sha256"] == (
        "fe39b4c201286078b5e695c1ca425d47f4e2f05f97bc042ab66e05d04bf1f759"
    )
    assert len(result["snapshot_sha256"]) == 64


def test_submission_draft_keeps_unknown_author_facts_explicit() -> None:
    draft = (
        PAPER14
        / "results/top_journal_program_v1/SUBMISSION_STATEMENTS_DRAFT.md"
    ).read_text(encoding="utf-8")
    assert "PUBLIC REPOSITORY OR DOI REQUIRED BEFORE SUBMISSION" in draft
    assert "AUTHOR CONFIRMATION REQUIRED" in draft
    assert "AUTHOR NAMES AND CRediT ROLES REQUIRED" in draft
    assert "AUTHOR/RIGHTS-HOLDER LEGAL CHOICE REQUIRED" in draft
    assert "No AI system is an author" in draft
    assert "[需要作者确认" in draft


def test_mnist_source_provenance_separates_identity_from_open_terms() -> None:
    result = verifier.verify_mnist_source_provenance()
    expected = (
        "RECEIPT_ONLY"
        if (PAPER14 / "SANITIZATION_LEDGER.json").is_file()
        else "RAW_AND_DIRECT_INPUTS_VERIFIED"
    )
    assert result["status"] == expected
    assert result["raw_resource_count"] == 4
    assert result["official_md5_match"] is True
    assert result["direct_checkpoint_count"] == 6
    assert result["generation_environment_fully_recorded"] is False
    assert result["dataset_redistribution_terms_confirmed"] is False
    assert len(result["receipt_sha256"]) == 64


def test_mnist_historical_environment_is_closed_without_back_projection() -> None:
    result = verifier.verify_mnist_historical_environment_recovery()
    assert result["status"] == "CLOSED_DOCUMENTED_IRRECOVERABLE_RAW_DATA_EXCLUDED"
    assert result["historical_torchvision_version_recoverable"] is False
    assert result["raw_mnist_in_release_profiles"] is False
    assert result["scientific_input_chain_hash_bound"] is True


def test_third_party_notice_matches_vendored_backdoorbench_snapshot() -> None:
    notices = (
        PAPER14 / "results/top_journal_program_v1/THIRD_PARTY_NOTICES.md"
    ).read_text(encoding="utf-8")
    audit = (
        PAPER14 / "results/top_journal_program_v1/ATTACK_SOURCE_AUDIT.md"
    ).read_text(encoding="utf-8")
    license_path = PAPER14 / "external/BackdoorBench/LICENSE"
    assert verifier.sha256(license_path) == (
        "c67d7a90ddf78746757ebff6fcde6ef57cf57c6c9321faa5ffd275565f8f816e"
    )
    assert "f02e3534645f0ee63d6848653062cd6c0d6c400d" in notices
    assert "CC BY-NC 4.0" in notices
    assert "not merely a bibliography/reference snapshot" in audit
    assert "became false" in audit


def test_release_manifest_is_preserved_as_controlled_preterminal_history() -> None:
    manifest = PROGRAM / "release_archive_manifest_preterminal_2026-10-02.csv"
    preflight_path = PROGRAM / "release_archive_preflight_preterminal_2026-10-02.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert sha256(manifest) == HISTORICAL_RELEASE_MANIFEST_SHA256
    assert sha256(preflight_path) == HISTORICAL_RELEASE_PREFLIGHT_SHA256
    assert preflight["status"] == "PRETERMINAL_NOT_RELEASE_READY"
    assert preflight["claim_c27_state"] == "PENDING_TERMINAL"
    assert preflight["manifest"]["sha256"] == HISTORICAL_RELEASE_MANIFEST_SHA256
    assert any(
        item["id"] == "B09_CLEAN_ARCHIVE_REPLAY"
        and item["status"] == "PRETERMINAL_PASS_FINAL_REPLAY_REQUIRED"
        for item in preflight["blockers"]
    )
    assert_v2_migration_is_active()


def test_pathmnist_weight_redistribution_receipt_is_conservative_and_hash_bound() -> None:
    result = verifier.verify_pathmnist_weight_redistribution_audit()
    assert result["status"] == "SAFE_RELEASE_POLICY_FIXED_TERMS_UNCONFIRMED"
    assert result["terms_confirmed"] is False
    assert result["binary_count"] == 4
    assert result["public_binary_redistribution_authorized"] is False


def test_public_code_profile_is_preserved_as_controlled_preterminal_history() -> None:
    manifest = PROGRAM / "public_code_profile_manifest_preterminal_2026-10-03.csv"
    preflight_path = PROGRAM / "public_code_profile_preflight_preterminal_2026-10-03.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert sha256(manifest) == HISTORICAL_PUBLIC_PROFILE_MANIFEST_SHA256
    assert sha256(preflight_path) == HISTORICAL_PUBLIC_PROFILE_PREFLIGHT_SHA256
    assert preflight["status"] == "PRETERMINAL_PROJECT_LICENSE_REQUIRED"
    assert preflight["binary_artifact_count"] == 0
    assert preflight["secret_flag_file_count"] == 0
    assert preflight["publication_authorized"] is False
    assert preflight["manifest"]["sha256"] == HISTORICAL_PUBLIC_PROFILE_MANIFEST_SHA256
    assert_v2_migration_is_active()


def test_blind_review_path_substitution_is_narrow_and_deterministic() -> None:
    source = (
        "/" + "Users/example/work/file.json "
        "C:" + "\\\\Users\\\\reviewer\\\\work\\\\file.json "
        "/private" + "/tmp/data/file.npz relative/path.csv"
    )
    sanitized, counts = blind_review.sanitize_text(source)
    assert sanitized == (
        "<USER_HOME>/work/file.json "
        "<USER_HOME>\\\\work\\\\file.json "
        "<TMP>/data/file.npz relative/path.csv"
    )
    assert counts == {
        "POSIX_USER_HOME": 1,
        "PRIVATE_TMP_PREFIX": 1,
        "WINDOWS_USER_HOME": 1,
    }
    assert blind_review.remaining_path_risks(sanitized) == ()


def test_blind_review_control_plane_pairs_preterminal_and_terminal_profiles(
    tmp_path: Path,
) -> None:
    for phase in ("preterminal_2026-10-02", "terminal_2026-10-04"):
        manifest = tmp_path / f"release_archive_manifest_{phase}.csv"
        preflight = tmp_path / f"release_archive_preflight_{phase}.json"
        manifest.write_text("path,size_bytes,sha256\n", encoding="utf-8")
        preflight.write_text("{}\n", encoding="utf-8")
        assert blind_review.control_plane_sources(manifest) == [
            manifest.resolve(),
            preflight.resolve(),
        ]


def test_blind_review_audit_rejects_the_controlled_stale_preterminal_manifest() -> None:
    if (PAPER14 / "SANITIZATION_LEDGER.json").is_file():
        pytest.skip("source-workspace packaging audit is not rerun from a staged copy")
    assert_v2_migration_is_active()
    with pytest.raises(AssertionError, match=r"manifest (size|hash) drift"):
        blind_review.audit_payload()


def test_blind_review_weight_filter_is_narrow() -> None:
    assert blind_review.is_excluded_weight_binary(
        "data/official_pathmnist_weights/resnet18_28_1.pth"
    )
    assert blind_review.is_excluded_weight_binary(
        "results/pathmnist_official_weight_warmstart_v1/models/seed_93_epoch5.pt"
    )
    assert blind_review.is_excluded_weight_binary(
        "results/pathmnist_official_weight_detector_confirmation_v1/checkpoints/seed_103.pt"
    )
    assert not blind_review.is_excluded_weight_binary(
        "results/pathmnist_official_weight_warmstart_v1/victim_metrics.csv"
    )
    assert not blind_review.is_excluded_weight_binary("src/model.py")


def test_blind_review_audit_receipt_is_deterministic(tmp_path: Path) -> None:
    if (PAPER14 / "SANITIZATION_LEDGER.json").is_file():
        pytest.skip("source-workspace packaging audit is not rerun from a staged copy")
    report = {
        "schema": "paper14_blind_review_sanitization_audit_v1",
        "status": "CONTROLLED_PRETERMINAL_SNAPSHOT_STALE",
        "package_built": False,
        "source_evidence_modified": False,
    }
    receipt = tmp_path / "audit.json"
    first = blind_review.write_audit_receipt(receipt, report)
    second = blind_review.write_audit_receipt(receipt, report)
    assert first == second
    assert len(first["sha256"]) == 64
    stored = json.loads(receipt.read_text(encoding="utf-8"))
    assert stored["status"] == "CONTROLLED_PRETERMINAL_SNAPSHOT_STALE"
    assert stored["package_built"] is False


def test_blind_review_staging_writes_verified_copy_and_ledger(tmp_path: Path) -> None:
    source = PAPER14 / "README.md"
    source_hash = raw_sha256(source)
    _, source_transformations = blind_review.sanitize_text(
        source.read_text(encoding="utf-8")
    )
    manifest = tmp_path / "mini_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "size_bytes",
                "sha256",
                "component",
                "license_status",
                "release_action",
                "risk_flags",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "path": "README.md",
                "size_bytes": source.stat().st_size,
                "sha256": source_hash,
                "component": "project_generated_evidence",
                "license_status": "TEST_ONLY",
                "release_action": "TEST_ONLY",
                "risk_flags": "LOCAL_USER_PATH",
            }
        )
    destination = tmp_path / "blind-copy"
    result = blind_review.stage_package(
        destination,
        manifest,
        blind_review.ACKNOWLEDGEMENT,
    )
    assert result["status"] == "STAGED_COPY_VERIFIED"
    assert result["payload_file_count"] == 1
    assert result["transformed_file_count"] == int(bool(source_transformations))
    staged_text = (destination / "README.md").read_text(encoding="utf-8")
    assert blind_review.remaining_path_risks(staged_text) == ()
    ledger = json.loads(
        (destination / "SANITIZATION_LEDGER.json").read_text(encoding="utf-8")
    )
    assert ledger["source_evidence_modified"] is False
    assert ledger["files"][0]["source_sha256"] == source_hash
    assert raw_sha256(source) == source_hash
    verification = blind_verify.verify_package(destination)
    assert verification["status"] == "PASS"
    assert verification["payload_file_count"] == 1
    assert verification["unexpected_file_count"] == 0


def test_blind_review_preserves_frozen_hash_bound_control_bytes(
    tmp_path: Path,
) -> None:
    relative = next(iter(blind_review.HASH_BOUND_OPAQUE_PATHS))
    source = PAPER14 / relative
    manifest = tmp_path / "opaque_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "size_bytes",
                "sha256",
                "component",
                "license_status",
                "release_action",
                "risk_flags",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "path": relative,
                "size_bytes": source.stat().st_size,
                "sha256": raw_sha256(source),
                "component": "project_generated_evidence",
                "license_status": "TEST_ONLY",
                "release_action": "TEST_ONLY",
                "risk_flags": "EPHEMERAL_ABSOLUTE_PATH",
            }
        )
    destination = tmp_path / "opaque-copy"
    result = blind_review.stage_package(
        destination, manifest, blind_review.ACKNOWLEDGEMENT
    )
    staged = destination / relative
    assert staged.read_bytes() == source.read_bytes()
    ledger = json.loads(
        (destination / "SANITIZATION_LEDGER.json").read_text(encoding="utf-8")
    )
    assert ledger["opaque_hash_bound_files"] == [relative]
    verification = blind_verify.verify_package(destination)
    assert verification["status"] == "PASS"
    assert verification["opaque_hash_bound_file_count"] == 1


def test_blind_review_independent_verifier_rejects_post_stage_tampering(
    tmp_path: Path,
) -> None:
    source = PAPER14 / "README.md"
    manifest = tmp_path / "mini_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "size_bytes",
                "sha256",
                "component",
                "license_status",
                "release_action",
                "risk_flags",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "path": "README.md",
                "size_bytes": source.stat().st_size,
                "sha256": raw_sha256(source),
                "component": "project_generated_evidence",
                "license_status": "TEST_ONLY",
                "release_action": "TEST_ONLY",
                "risk_flags": "LOCAL_USER_PATH",
            }
        )
    destination = tmp_path / "blind-copy"
    blind_review.stage_package(
        destination,
        manifest,
        blind_review.ACKNOWLEDGEMENT,
    )
    (destination / "README.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="staged size mismatch"):
        blind_verify.verify_package(destination)
