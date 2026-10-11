from __future__ import annotations

from pathlib import Path
import sys


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import build_terminal_release_profile_v2 as profile  # noqa: E402


def test_preterminal_self_check_is_outcome_blind() -> None:
    result = profile.preterminal_self_check()
    assert result["status"] == "PASS_PRETERMINAL_RELEASE_PROFILE_PREP"
    assert result["terminal_receipts_opened"] is False
    assert result["pathmnist_performance_artifacts_parsed"] is False
    assert result["submission_ready"] is False
    assert result["author_blockers_preserved"] == list(profile.AUTHOR_BLOCKERS)


def test_blocker_ledger_never_infers_author_decisions() -> None:
    regression = {"pytest": {"passed": 220, "tests": 220}}
    prepared = profile.blocker_ledger(
        route="R3_ELIGIBLE_PRIMARY_GATE_PASS",
        c27_decision="SUPPORTED_CONDITIONALLY",
        regression=regression,
        reviewer=None,
    )
    statuses = {item["id"]: item["status"] for item in prepared}
    assert statuses["B01_TERMINAL_EXTERNAL_VALIDITY"].startswith("CLOSED_")
    assert statuses["B05_ANONYMITY_AND_PORTABILITY"] == "REVIEWER_REPLAY_REQUIRED"
    assert statuses["B02_PROJECT_CODE_LICENSE"] == "AUTHOR_DECISION_REQUIRED"
    assert statuses["B06_PUBLIC_ACCESSION"] == "AUTHOR_ACTION_REQUIRED"

    reviewer = {"status": "PASS_TERMINAL_REVIEWER_PACKAGE"}
    finalized = profile.blocker_ledger(
        route="R2_ELIGIBLE_PRIMARY_GATE_FAIL",
        c27_decision="REJECTED",
        regression=regression,
        reviewer=reviewer,
    )
    final_statuses = {item["id"]: item["status"] for item in finalized}
    assert final_statuses["B05_ANONYMITY_AND_PORTABILITY"].startswith("CLOSED_")
    assert final_statuses["B09_CLEAN_ARCHIVE_REPLAY"].startswith("CLOSED_")
    assert final_statuses["B03_ARTIFACT_LICENSE"] == "AUTHOR_DECISION_REQUIRED"
    assert final_statuses["B07_AUTHOR_STATEMENTS"] == "AUTHOR_CONFIRMATION_REQUIRED"


def test_terminal_payload_excludes_circular_receipts(tmp_path: Path) -> None:
    contract = {
        "mutable_integration_targets": [],
        "protected_history": [],
        "required_terminal_inputs": [],
        "required_new_terminal_outputs": [
            str(profile.REVIEWER_RECEIPT.relative_to(PAPER14)),
            str(profile.FINAL_RECEIPT.relative_to(PAPER14)),
            "results/top_journal_program_v1/final_bundle_v2_receipt_2026-10-04.json",
        ],
    }
    # The helper also includes the legacy runnable closure.  Its critical
    # property here is that none of the circular/post-profile outputs enter.
    paths = set(profile.terminal_payload_paths(contract))
    assert not (paths & profile.CIRCULAR_OR_POSTPROFILE_OUTPUTS)
    program_files = {
        path.resolve()
        for path in profile.PROGRAM.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert program_files - profile.CIRCULAR_OR_POSTPROFILE_OUTPUTS <= paths
    assert {path.resolve() for path in profile.TIFS_REVIEWER_RUNTIME_FILES} <= paths


def test_release_rows_are_hash_bound_and_classified() -> None:
    path = PAPER14 / "requirements-replay.txt"
    rows, summary = profile.build_release_rows([path])
    assert rows[0]["path"] == "requirements-replay.txt"
    assert rows[0]["sha256"] == profile.sha256(path)
    assert summary["payload_file_count"] == 1
    assert summary["payload_total_bytes"] == path.stat().st_size
