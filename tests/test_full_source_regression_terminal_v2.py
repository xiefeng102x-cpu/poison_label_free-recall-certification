from __future__ import annotations

import json
from pathlib import Path
import sys


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import run_full_source_regression_terminal_v2 as runner  # noqa: E402


def test_parse_junit_sums_nested_suites(tmp_path: Path) -> None:
    junit = tmp_path / "pytest.xml"
    junit.write_text(
        """<?xml version='1.0' encoding='utf-8'?>
<testsuites>
  <testsuite name='a' tests='3' failures='0' errors='0' skipped='0'/>
  <testsuite name='b' tests='2' failures='1' errors='0' skipped='1'/>
</testsuites>
""",
        encoding="utf-8",
    )
    assert runner.parse_junit(junit) == {
        "tests": 5,
        "failures": 1,
        "errors": 0,
        "skipped": 1,
        "passed": 3,
    }


def test_source_inventory_is_deterministic_and_content_bound(tmp_path: Path) -> None:
    for relative in runner.INVENTORY_ROOTS:
        (tmp_path / relative).mkdir()
    source = tmp_path / "src/example.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "tests/test_example.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    first = runner.source_inventory(tmp_path)
    second = runner.source_inventory(tmp_path)
    assert first == second
    assert first["file_count"] == 2
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert runner.source_inventory(tmp_path)["path_size_sha256_ledger_sha256"] != (
        first["path_size_sha256_ledger_sha256"]
    )


def preconditions() -> dict[str, object]:
    return {
        "terminal_control": {
            "selected_route": "R3_ELIGIBLE_PRIMARY_GATE_PASS",
            "replay_receipt_sha256": "a" * 64,
            "route_receipt_sha256": "b" * 64,
        },
        "capsule_verification_sha256": "c" * 64,
        "tifs_terminal_receipt_sha256": "d" * 64,
    }


def test_build_result_requires_every_collected_test_to_pass() -> None:
    passed = runner.build_result(
        returncode=0,
        totals={"tests": 216, "passed": 216, "failures": 0, "errors": 0, "skipped": 0},
        inventory={"file_count": 1},
        preconditions=preconditions(),
        stdout="216 passed",
        stderr="",
    )
    assert passed["status"] == "PASS_TERMINAL_SOURCE_REGRESSION"
    assert passed["errors"] == []
    skipped = runner.build_result(
        returncode=0,
        totals={"tests": 216, "passed": 215, "failures": 0, "errors": 0, "skipped": 1},
        inventory={"file_count": 1},
        preconditions=preconditions(),
        stdout="215 passed, 1 skipped",
        stderr="",
    )
    assert skipped["status"] == "FAIL"
    assert "pytest skipped=1" in skipped["errors"]


def test_preterminal_self_check_does_not_open_terminal_receipts() -> None:
    result = runner.preterminal_self_check()
    assert result["status"] == "PASS_PRETERMINAL_TERMINAL_REGRESSION_PREP"
    assert result["terminal_receipts_opened"] is False
    assert result["pathmnist_performance_artifacts_parsed"] is False
    assert result["submission_ready"] is False
    contract = json.loads(runner.EXECUTION_CONTRACT.read_text(encoding="utf-8"))
    assert str(runner.DEFAULT_RECEIPT.relative_to(PAPER14)) in contract[
        "required_new_terminal_outputs"
    ]
