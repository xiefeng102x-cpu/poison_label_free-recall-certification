#!/usr/bin/env python3
"""Run and receipt the complete post-integration paper14 source regression.

Terminal mode is deliberately gated by the valid V2 PathMNIST replay/route,
independently verified claim capsule, and terminal TIFS receipt.  The resulting
JSON is one of the required V2 final-bundle inputs.  Preterminal self-check
mode never opens terminal receipts or performance artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from typing import Any, Iterable

import select_pathmnist_terminal_route_v2 as selector
import verify_top_journal_final_bundle_v2 as final_v2


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "results/top_journal_program_v1"
EXECUTION_CONTRACT = PROGRAM / "pathmnist_terminal_integration_execution_contract_v2.json"
CAPSULE_VERIFICATION = PROGRAM / (
    "pathmnist_terminal_claim_capsule_verification_v2_2026-10-04.json"
)
TIFS_TERMINAL_RECEIPT = PROGRAM / "tifs_main_terminal_receipt_2026-10-04.json"
DEFAULT_RECEIPT = PROGRAM / "full_source_regression_terminal_2026-10-04.json"
FORMAL_OUTPUT = ROOT / "results/pathmnist_official_weight_detector_confirmation_v1"
INVENTORY_ROOTS = ("src", "tests", "configs", "theory")
INVENTORY_SUFFIXES = {".py", ".json", ".md"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_inventory(root: Path = ROOT) -> dict[str, Any]:
    """Hash the deterministic source/configuration closure used by pytest."""

    paths = sorted(
        (
            path
            for relative in INVENTORY_ROOTS
            for path in (root / relative).rglob("*")
            if path.is_file()
            and path.suffix in INVENTORY_SUFFIXES
            and "__pycache__" not in path.parts
        ),
        key=lambda path: str(path.relative_to(root)),
    )
    ledger = hashlib.sha256()
    for path in paths:
        relative = str(path.relative_to(root))
        ledger.update(relative.encode("utf-8"))
        ledger.update(b"\0")
        ledger.update(str(path.stat().st_size).encode("ascii"))
        ledger.update(b"\0")
        ledger.update(sha256(path).encode("ascii"))
        ledger.update(b"\n")
    return {
        "roots": list(INVENTORY_ROOTS),
        "included_suffixes": sorted(INVENTORY_SUFFIXES),
        "file_count": len(paths),
        "path_size_sha256_ledger_sha256": ledger.hexdigest(),
    }


def _suite_totals(suite: ET.Element) -> dict[str, int]:
    return {
        key: int(float(suite.attrib.get(key, "0")))
        for key in ("tests", "failures", "errors", "skipped")
    }


def parse_junit(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    if root.tag == "testsuite":
        totals = _suite_totals(root)
    elif root.tag == "testsuites":
        direct = _suite_totals(root)
        if direct["tests"]:
            totals = direct
        else:
            totals = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
            for suite in root.findall("testsuite"):
                observed = _suite_totals(suite)
                for key, value in observed.items():
                    totals[key] += value
    else:
        raise RuntimeError(f"unexpected JUnit root: {root.tag}")
    totals["passed"] = (
        totals["tests"]
        - totals["failures"]
        - totals["errors"]
        - totals["skipped"]
    )
    if totals["tests"] <= 0 or totals["passed"] < 0:
        raise RuntimeError("invalid JUnit totals")
    return totals


def preterminal_self_check() -> dict[str, Any]:
    contract = json.loads(EXECUTION_CONTRACT.read_text(encoding="utf-8"))
    expected = str(DEFAULT_RECEIPT.relative_to(ROOT))
    errors: list[str] = []
    if expected not in contract.get("required_new_terminal_outputs", []):
        errors.append("terminal source-regression receipt is absent from V2 contract")
    if not (ROOT / "tests").is_dir() or not any((ROOT / "tests").glob("test_*.py")):
        errors.append("pytest source suite is absent")
    result = {
        "status": "PASS_PRETERMINAL_TERMINAL_REGRESSION_PREP" if not errors else "FAIL",
        "errors": errors,
        "receipt_target": expected,
        "source_inventory": source_inventory(),
        "runner_sha256": sha256(Path(__file__)),
        "execution_contract_sha256": sha256(EXECUTION_CONTRACT),
        "terminal_receipts_opened": False,
        "pathmnist_performance_artifacts_parsed": False,
        "submission_ready": False,
    }
    if errors:
        raise RuntimeError("; ".join(errors))
    return result


def terminal_preconditions(
    *,
    replay_receipt: Path,
    route_receipt: Path,
    formal_output: Path,
) -> dict[str, Any]:
    control = final_v2.verify_terminal_control(
        replay_receipt, route_receipt, formal_output
    )
    if control["status"] != "PASS_TERMINAL_CONTROL":
        raise RuntimeError("; ".join(control["errors"]))
    capsule = json.loads(CAPSULE_VERIFICATION.read_text(encoding="utf-8"))
    if capsule.get("status") != "PASS":
        raise RuntimeError("independent terminal claim-capsule verification is not PASS")
    tifs = json.loads(TIFS_TERMINAL_RECEIPT.read_text(encoding="utf-8"))
    if tifs.get("status") != "PASS_TERMINAL":
        raise RuntimeError("terminal TIFS verification is not PASS_TERMINAL")
    if capsule.get("selected_route") != control.get("selected_route"):
        raise RuntimeError("capsule verification route differs from terminal control")
    if tifs.get("selected_route") != control.get("selected_route"):
        raise RuntimeError("TIFS receipt route differs from terminal control")
    return {
        "terminal_control": control,
        "capsule_verification_sha256": sha256(CAPSULE_VERIFICATION),
        "tifs_terminal_receipt_sha256": sha256(TIFS_TERMINAL_RECEIPT),
    }


def build_result(
    *,
    returncode: int,
    totals: dict[str, int],
    inventory: dict[str, Any],
    preconditions: dict[str, Any],
    stdout: str,
    stderr: str,
) -> dict[str, Any]:
    errors: list[str] = []
    if returncode != 0:
        errors.append(f"pytest returned {returncode}")
    for key in ("failures", "errors", "skipped"):
        if totals[key] != 0:
            errors.append(f"pytest {key}={totals[key]}")
    if totals["passed"] != totals["tests"]:
        errors.append("not every collected source test passed")
    control = preconditions["terminal_control"]
    return {
        "schema": "paper14_full_source_regression_terminal_v2",
        "status": "PASS_TERMINAL_SOURCE_REGRESSION" if not errors else "FAIL",
        "errors": errors,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, "-m", "pytest", "-q"],
        "pytest": totals,
        "returncode": returncode,
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
        "source_inventory": inventory,
        "selected_route": control["selected_route"],
        "replay_receipt_sha256": control["replay_receipt_sha256"],
        "route_receipt_sha256": control["route_receipt_sha256"],
        "capsule_verification_sha256": preconditions[
            "capsule_verification_sha256"
        ],
        "tifs_terminal_receipt_sha256": preconditions[
            "tifs_terminal_receipt_sha256"
        ],
        "runner_sha256": sha256(Path(__file__)),
        "terminal_control_verified_before_pytest": True,
        "direct_pathmnist_performance_endpoint_access": False,
        "submission_ready": False,
    }


def terminal_run(
    *,
    replay_receipt: Path = selector.DEFAULT_REPLAY_RECEIPT,
    route_receipt: Path = selector.DEFAULT_SELECTION_RECEIPT,
    formal_output: Path = FORMAL_OUTPUT,
    receipt_path: Path = DEFAULT_RECEIPT,
    extra_pytest_args: Iterable[str] = (),
) -> dict[str, Any]:
    preconditions = terminal_preconditions(
        replay_receipt=replay_receipt,
        route_receipt=route_receipt,
        formal_output=formal_output,
    )
    inventory = source_inventory()
    with tempfile.TemporaryDirectory(prefix="paper14-terminal-regression-") as temporary:
        junit = Path(temporary) / "pytest.xml"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            f"--junitxml={junit}",
            *extra_pytest_args,
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if not junit.is_file():
            raise RuntimeError("pytest did not produce the terminal JUnit receipt")
        totals = parse_junit(junit)
    result = build_result(
        returncode=completed.returncode,
        totals=totals,
        inventory=inventory,
        preconditions=preconditions,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    final_v2.atomic_json(receipt_path, result)
    if result["errors"]:
        raise RuntimeError("; ".join(result["errors"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("preterminal-self-check", "terminal"), required=True
    )
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    if args.mode == "preterminal-self-check":
        result = preterminal_self_check()
    else:
        result = terminal_run(receipt_path=args.receipt)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
