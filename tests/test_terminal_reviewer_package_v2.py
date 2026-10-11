from __future__ import annotations

from pathlib import Path
import sys


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import run_terminal_reviewer_package_v2 as reviewer  # noqa: E402


def test_preterminal_self_check_is_outcome_blind() -> None:
    result = reviewer.preterminal_self_check()
    assert result["status"] == "PASS_PRETERMINAL_REVIEWER_PACKAGE_PREP"
    assert result["terminal_receipts_opened"] is False
    assert result["pathmnist_performance_artifacts_parsed"] is False
    assert result["package_built"] is False
    assert result["expected_policy_skips"] == 5


def test_policy_skip_validator_accepts_only_frozen_five() -> None:
    skips = [
        {
            "classname": "tests.test_pathmnist_official_weight_warmstart",
            "name": "test_protocol_locks_all_sources_receipt_weights_and_mapping",
            "message": "official PathMNIST weights are intentionally excluded from the blind-review profile pending explicit redistribution terms",
        },
        {
            "classname": "tests.test_pathmnist_official_weight_warmstart",
            "name": "test_official_initial_state_strict_loads_bit_identically[93]",
            "message": "official PathMNIST weights are intentionally excluded from the blind-review profile pending explicit redistribution terms",
        },
        {
            "classname": "tests.test_pathmnist_official_weight_warmstart",
            "name": "test_official_initial_state_strict_loads_bit_identically[94]",
            "message": "official PathMNIST weights are intentionally excluded from the blind-review profile pending explicit redistribution terms",
        },
        {
            "classname": "tests.test_top_journal_bundle_inventory",
            "name": "test_blind_review_audit_rejects_the_controlled_stale_preterminal_manifest",
            "message": "source-workspace packaging audit is not rerun from a staged copy",
        },
        {
            "classname": "tests.test_top_journal_bundle_inventory",
            "name": "test_blind_review_audit_receipt_is_deterministic",
            "message": "source-workspace packaging audit is not rerun from a staged copy",
        },
    ]
    assert reviewer.validate_policy_skips(skips) == []
    assert reviewer.validate_policy_skips(skips[:-1])
    unexpected = [*skips, {"classname": "x", "name": "test_other", "message": "x"}]
    assert any("unexpected skipped test" in error for error in reviewer.validate_policy_skips(unexpected))


def test_skipped_tests_reads_junit_inventory(tmp_path: Path) -> None:
    junit = tmp_path / "pytest.xml"
    junit.write_text(
        """<?xml version='1.0' encoding='utf-8'?>
<testsuites tests='2' failures='0' errors='0' skipped='1'>
  <testsuite name='pytest' tests='2' failures='0' errors='0' skipped='1'>
    <testcase classname='a' name='test_ok'/>
    <testcase classname='b' name='test_skip'>
      <skipped message='policy reason'/>
    </testcase>
  </testsuite>
</testsuites>
""",
        encoding="utf-8",
    )
    assert reviewer.skipped_tests(junit) == [
        {"classname": "b", "name": "test_skip", "message": "policy reason"}
    ]
