#!/usr/bin/env python3
"""Independently verify a staged blind-review package from its ledger."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import build_blind_review_package as builder
import verify_top_journal_final_bundle as bundle


LEDGER_NAME = "SANITIZATION_LEDGER.json"


def verify_package(root: Path) -> dict[str, object]:
    root = root.resolve()
    ledger_path = root / LEDGER_NAME
    if not root.is_dir() or not ledger_path.is_file():
        raise AssertionError("staged package and SANITIZATION_LEDGER.json are required")

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("schema") != "paper14_blind_review_sanitization_audit_v1":
        raise AssertionError("unexpected sanitization-ledger schema")
    if ledger.get("status") != "STAGED_COPY_VERIFIED":
        raise AssertionError("ledger does not describe a verified staged copy")
    if ledger.get("destination") != "<USER_SELECTED_STAGING_DIRECTORY>":
        raise AssertionError("ledger must not retain the local staging path")
    if ledger.get("source_evidence_modified") is not False:
        raise AssertionError("ledger must preserve source evidence")
    if ledger.get("package_built") is not True:
        raise AssertionError("ledger does not record a built package")

    details = ledger.get("files")
    if not isinstance(details, list) or not details:
        raise AssertionError("ledger file inventory is missing")
    paths = [item.get("path") for item in details]
    if any(not isinstance(path, str) or not path for path in paths):
        raise AssertionError("ledger contains an invalid relative path")
    if len(paths) != len(set(paths)):
        raise AssertionError("ledger contains duplicate paths")

    included = {
        item["path"]: item
        for item in details
        if item.get("profile_action") == "INCLUDE"
    }
    controls = {
        item["path"]: item
        for item in details
        if item.get("profile_action") == "CONTROL_PLANE_INCLUDE"
    }
    excluded = {
        item["path"]: item
        for item in details
        if item.get("profile_action") == "EXCLUDED_WEIGHT_BINARY"
    }
    if len(included) + len(controls) + len(excluded) != len(details):
        raise AssertionError("ledger contains an unknown profile action")
    if len(included) != int(ledger.get("payload_file_count", -1)):
        raise AssertionError("payload file count does not match ledger details")
    if len(excluded) != int(ledger.get("excluded_weight_binary_count", -1)):
        raise AssertionError("excluded-weight count does not match ledger details")
    if len(controls) != int(ledger.get("control_plane_file_count", -1)):
        raise AssertionError("control-plane count does not match ledger details")
    if len(included) + len(controls) != int(ledger.get("package_file_count", -1)):
        raise AssertionError("package file count does not match ledger details")
    if sorted(excluded) != sorted(ledger.get("excluded_weight_binary_files", [])):
        raise AssertionError("excluded-weight paths do not match ledger summary")
    if any(not builder.is_excluded_weight_binary(path) for path in excluded):
        raise AssertionError("ledger excludes a file outside the narrow weight policy")

    opaque = {
        path: item
        for path, item in included.items()
        if item.get("preservation_mode") == "HASH_BOUND_CONTROL_EVIDENCE"
    }
    if sorted(opaque) != sorted(ledger.get("opaque_hash_bound_files", [])):
        raise AssertionError("opaque hash-bound file inventory mismatch")
    if len(opaque) != int(ledger.get("opaque_hash_bound_file_count", -1)):
        raise AssertionError("opaque hash-bound file count mismatch")
    if any(path not in builder.HASH_BOUND_OPAQUE_PATHS for path in opaque):
        raise AssertionError("ledger marks a file outside the opaque control policy")
    for relative, detail in opaque.items():
        if detail.get("transformation_counts") != {}:
            raise AssertionError(f"opaque control file was transformed: {relative}")
        if detail.get("source_sha256") != detail.get("staged_sha256"):
            raise AssertionError(f"opaque control file changed bytes: {relative}")

    expected_files = set(included) | set(controls) | {LEDGER_NAME}
    actual_files = {
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise AssertionError(
            f"staged package file closure mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )

    aggregate: Counter[str] = Counter()
    transformed = 0
    text_files_checked = 0
    for relative, detail in {**included, **controls}.items():
        target = root / relative
        if target.stat().st_size != int(detail["staged_size_bytes"]):
            raise AssertionError(f"staged size mismatch: {relative}")
        if bundle.sha256(target) != detail["staged_sha256"]:
            raise AssertionError(f"staged hash mismatch: {relative}")
        rules = detail.get("transformation_counts", {})
        if rules:
            transformed += 1
            aggregate.update({key: int(value) for key, value in rules.items()})
        if (
            target.suffix.lower() in builder.TEXT_SUFFIXES
            and relative != builder.SELF_PATH
            and relative not in opaque
        ):
            try:
                text = target.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            text_files_checked += 1
            risks = builder.remaining_path_risks(text)
            if risks:
                raise AssertionError(f"residual path risk in {relative}: {risks}")

    if transformed != int(ledger.get("transformed_file_count", -1)):
        raise AssertionError("transformed-file count does not match ledger summary")
    if dict(sorted(aggregate.items())) != ledger.get("transformation_counts"):
        raise AssertionError("transformation counts do not match ledger summary")
    if int(ledger.get("residual_path_file_count", -1)) != 0:
        raise AssertionError("ledger records residual path risks")
    if ledger.get("residual_path_files") != []:
        raise AssertionError("ledger residual-path list must be empty")
    if any((root / relative).exists() for relative in excluded):
        raise AssertionError("an excluded weight binary is present in the package")

    return {
        "schema": "paper14_blind_review_staged_package_verification_v1",
        "status": "PASS",
        "staging_directory": "<VERIFIED_STAGING_DIRECTORY>",
        "ledger_sha256": bundle.sha256(ledger_path),
        "payload_file_count": len(included),
        "package_file_count": len(included) + len(controls),
        "control_plane_file_count": len(controls),
        "excluded_weight_binary_count": len(excluded),
        "transformed_file_count": transformed,
        "transformation_counts": dict(sorted(aggregate.items())),
        "text_files_checked_for_residual_paths": text_files_checked,
        "residual_path_file_count": 0,
        "opaque_hash_bound_file_count": len(opaque),
        "unexpected_file_count": 0,
        "source_evidence_modified": False,
    }


def write_receipt(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify_package(args.package)
    if args.receipt is not None:
        write_receipt(args.receipt, report)
        report = {
            **report,
            "receipt": str(args.receipt),
            "receipt_sha256": bundle.sha256(args.receipt),
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
