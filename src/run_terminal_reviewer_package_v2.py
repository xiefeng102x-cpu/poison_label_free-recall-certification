#!/usr/bin/env python3
"""Build, independently verify, and replay the terminal blind-review package.

Only the terminal release manifest may be used.  The package builder removes
local paths and conservatively excludes official/official-initialized
PathMNIST weight binaries.  Pytest runs inside the staged copy with bytecode
and cache writes disabled.  Exactly five policy-scoped skips are permitted:
three weight-dependent checks and two source-workspace packaging audits.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from typing import Any

import build_blind_review_package as builder
import build_terminal_release_profile_v2 as release_profile
import run_full_source_regression_terminal_v2 as regression
import verify_blind_review_package as package_verifier
import verify_top_journal_final_bundle_v2 as final_v2


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "results/top_journal_program_v1"
EXECUTION_CONTRACT = PROGRAM / "pathmnist_terminal_integration_execution_contract_v2.json"
RELEASE_MANIFEST = release_profile.RELEASE_MANIFEST
RELEASE_PREFLIGHT = release_profile.RELEASE_PREFLIGHT
DEFAULT_RECEIPT = PROGRAM / "reviewer_package_terminal_receipt_2026-10-04.json"
ALLOWED_SKIP_PREFIXES = (
    "test_protocol_locks_all_sources_receipt_weights_and_mapping",
    "test_official_initial_state_strict_loads_bit_identically",
    "test_blind_review_audit_rejects_the_controlled_stale_preterminal_manifest",
    "test_blind_review_audit_receipt_is_deterministic",
)
EXPECTED_POLICY_SKIPS = 5


def sha256(path: Path) -> str:
    return final_v2.sha256(path)


def preterminal_self_check() -> dict[str, Any]:
    contract = json.loads(EXECUTION_CONTRACT.read_text(encoding="utf-8"))
    expected = str(DEFAULT_RECEIPT.relative_to(ROOT))
    errors: list[str] = []
    if expected not in contract.get("required_new_terminal_outputs", []):
        errors.append("terminal reviewer-package receipt is absent from V2 contract")
    for path in (
        ROOT / "src/build_blind_review_package.py",
        ROOT / "src/verify_blind_review_package.py",
        ROOT / "src/build_terminal_release_profile_v2.py",
    ):
        if not path.is_file():
            errors.append(f"terminal reviewer dependency missing: {path.name}")
    result = {
        "status": "PASS_PRETERMINAL_REVIEWER_PACKAGE_PREP" if not errors else "FAIL",
        "errors": errors,
        "terminal_receipt_target": expected,
        "terminal_manifest_target": str(RELEASE_MANIFEST.relative_to(ROOT)),
        "expected_policy_skips": EXPECTED_POLICY_SKIPS,
        "allowed_skip_prefixes": list(ALLOWED_SKIP_PREFIXES),
        "runner_sha256": sha256(Path(__file__)),
        "execution_contract_sha256": sha256(EXECUTION_CONTRACT),
        "terminal_receipts_opened": False,
        "pathmnist_performance_artifacts_parsed": False,
        "package_built": False,
        "submission_ready": False,
    }
    if errors:
        raise RuntimeError("; ".join(errors))
    return result


def skipped_tests(junit: Path) -> list[dict[str, str]]:
    root = ET.parse(junit).getroot()
    skipped: list[dict[str, str]] = []
    for case in root.iter("testcase"):
        marker = case.find("skipped")
        if marker is None:
            continue
        skipped.append(
            {
                "classname": case.attrib.get("classname", ""),
                "name": case.attrib.get("name", ""),
                "message": marker.attrib.get("message", ""),
            }
        )
    return skipped


def validate_policy_skips(skips: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    if len(skips) != EXPECTED_POLICY_SKIPS:
        errors.append(
            f"policy skip count {len(skips)} differs from {EXPECTED_POLICY_SKIPS}"
        )
    for item in skips:
        if not any(item["name"].startswith(prefix) for prefix in ALLOWED_SKIP_PREFIXES):
            errors.append(f"unexpected skipped test: {item['classname']}::{item['name']}")
    weight_skips = [
        item for item in skips if "intentionally excluded" in item["message"]
    ]
    workspace_skips = [
        item for item in skips if "source-workspace packaging audit" in item["message"]
    ]
    if len(weight_skips) != 3:
        errors.append(f"weight-policy skip count is {len(weight_skips)}, expected 3")
    if len(workspace_skips) != 2:
        errors.append(
            f"source-workspace policy skip count is {len(workspace_skips)}, expected 2"
        )
    return errors


def terminal_preconditions() -> dict[str, Any]:
    scientific = release_profile.terminal_preconditions()
    preflight = release_profile.load_pass(
        RELEASE_PREFLIGHT,
        {"PASS_TERMINAL_MANIFEST_PREP_REVIEWER_REPLAY_REQUIRED"},
    )
    if preflight["manifest"]["sha256"] != sha256(RELEASE_MANIFEST):
        raise RuntimeError("terminal release preflight does not bind the manifest")
    if preflight.get("selected_route") != scientific["control"]["selected_route"]:
        raise RuntimeError("terminal release preflight route mismatch")
    return {"scientific": scientific, "release_preflight": preflight}


def run(
    *, destination: Path, receipt_path: Path = DEFAULT_RECEIPT
) -> dict[str, Any]:
    preconditions = terminal_preconditions()
    source_manifest_hash = sha256(RELEASE_MANIFEST)
    stage = builder.stage_package(
        destination,
        RELEASE_MANIFEST,
        builder.ACKNOWLEDGEMENT,
    )
    before = package_verifier.verify_package(destination)
    with tempfile.TemporaryDirectory(prefix="paper14-reviewer-junit-") as temporary:
        junit = Path(temporary) / "pytest.xml"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={junit}",
        ]
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            command,
            cwd=destination,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if not junit.is_file():
            raise RuntimeError("staged pytest did not produce JUnit output")
        totals = regression.parse_junit(junit)
        skips = skipped_tests(junit)
    after = package_verifier.verify_package(destination)

    errors = validate_policy_skips(skips)
    if completed.returncode != 0:
        errors.append(f"staged pytest returned {completed.returncode}")
    if totals["failures"] or totals["errors"]:
        errors.append("staged pytest contains failure/error outcomes")
    if totals["skipped"] != len(skips):
        errors.append("JUnit skipped total differs from skipped testcase inventory")
    if before != after:
        errors.append("staged package changed during its pytest replay")
    if source_manifest_hash != sha256(RELEASE_MANIFEST):
        errors.append("source release manifest changed during package replay")

    control = preconditions["scientific"]["control"]
    receipt = {
        "schema": "paper14_terminal_reviewer_package_v2",
        "status": "PASS_TERMINAL_REVIEWER_PACKAGE" if not errors else "FAIL",
        "errors": errors,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_route": control["selected_route"],
        "source_manifest_sha256": source_manifest_hash,
        "release_preflight_sha256": sha256(RELEASE_PREFLIGHT),
        "sanitization_ledger_sha256": stage["ledger_sha256"],
        "payload_file_count": before["payload_file_count"],
        "package_file_count": before["package_file_count"],
        "control_plane_file_count": before["control_plane_file_count"],
        "excluded_weight_binary_count": before["excluded_weight_binary_count"],
        "transformed_file_count": before["transformed_file_count"],
        "residual_path_file_count": before["residual_path_file_count"],
        "unexpected_file_count": before["unexpected_file_count"],
        "pytest": totals,
        "policy_skips": skips,
        "stdout_sha256": __import__("hashlib").sha256(
            completed.stdout.encode("utf-8")
        ).hexdigest(),
        "stderr_sha256": __import__("hashlib").sha256(
            completed.stderr.encode("utf-8")
        ).hexdigest(),
        "source_evidence_modified": False,
        "package_unchanged_by_replay": before == after,
        "staging_directory": "<USER_SELECTED_STAGING_DIRECTORY>",
        "runner_sha256": sha256(Path(__file__)),
        "submission_ready": False,
    }
    final_v2.atomic_json(receipt_path, receipt)
    if errors:
        raise RuntimeError("; ".join(errors))
    return {**receipt, "runtime_staging_directory": str(destination.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("preterminal-self-check", "terminal"), required=True
    )
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    if args.mode == "preterminal-self-check":
        result = preterminal_self_check()
    else:
        if args.destination is None:
            raise SystemExit("--destination is required in terminal mode")
        result = run(destination=args.destination, receipt_path=args.receipt)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
