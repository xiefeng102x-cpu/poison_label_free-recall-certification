#!/usr/bin/env python3
"""Audit or build a path-sanitized blind-review copy of the release payload.

The default mode is read-only.  Staging requires an explicit acknowledgement,
refuses an existing destination, never edits source evidence, and records a
per-file transformation ledger.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import verify_top_journal_final_bundle as verifier


DEFAULT_MANIFEST = (
    verifier.PROGRAM / "release_archive_manifest_preterminal_2026-10-02.csv"
)
DEFAULT_AUDIT_RECEIPT = (
    verifier.PROGRAM / "blind_review_sanitization_audit_preterminal_2026-10-02.json"
)
ACKNOWLEDGEMENT = "BUILD_PATH_SANITIZED_BLIND_REVIEW_COPY"
TEXT_SUFFIXES = {
    ".bib",
    ".cfg",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}
SELF_PATH = "src/build_blind_review_package.py"
HASH_BOUND_OPAQUE_PATHS = {
    "results/top_journal_program_v1/"
    "pathmnist_terminal_replay_dataset_lifetime_amendment_2026-10-04.json",
}
WEIGHT_BINARY_PREFIXES = (
    "data/official_pathmnist_weights/",
    "results/pathmnist_official_weight_warmstart_v1/models/",
    "results/pathmnist_official_weight_detector_confirmation_v1/",
)
WEIGHT_BINARY_SUFFIXES = (".pt", ".pth")
USER_HOME_RE = re.compile(r"/Users/[A-Za-z0-9._-]+")
WINDOWS_HOME_RE = re.compile(r"[A-Za-z]:\\\\Users\\\\[A-Za-z0-9._-]+")
TMP_PREFIX = "/private" + "/tmp/"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len(rows) != len({row["path"] for row in rows}):
        raise AssertionError("release manifest is empty or contains duplicate paths")
    return rows


def sanitize_text(text: str) -> tuple[str, dict[str, int]]:
    counts: Counter[str] = Counter()

    def replace_posix(match: re.Match[str]) -> str:
        counts["POSIX_USER_HOME"] += 1
        return "<USER_HOME>"

    def replace_windows(match: re.Match[str]) -> str:
        counts["WINDOWS_USER_HOME"] += 1
        return "<USER_HOME>"

    sanitized = USER_HOME_RE.sub(replace_posix, text)
    sanitized = WINDOWS_HOME_RE.sub(replace_windows, sanitized)
    tmp_count = sanitized.count(TMP_PREFIX)
    if tmp_count:
        counts["PRIVATE_TMP_PREFIX"] += tmp_count
        sanitized = sanitized.replace(TMP_PREFIX, "<TMP>/")
    return sanitized, dict(sorted(counts.items()))


def remaining_path_risks(text: str) -> tuple[str, ...]:
    risks: list[str] = []
    if USER_HOME_RE.search(text) or WINDOWS_HOME_RE.search(text):
        risks.append("LOCAL_USER_PATH")
    if TMP_PREFIX in text:
        risks.append("EPHEMERAL_ABSOLUTE_PATH")
    return tuple(risks)


def is_excluded_weight_binary(relative: str) -> bool:
    """Apply the conservative PathMNIST weight-release policy."""
    return relative.endswith(WEIGHT_BINARY_SUFFIXES) and relative.startswith(
        WEIGHT_BINARY_PREFIXES
    )


def audit_payload(
    manifest: Path = DEFAULT_MANIFEST,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = load_manifest(manifest)
    details: list[dict[str, object]] = []
    transformed_files = 0
    transformation_counts: Counter[str] = Counter()
    residual_files: list[str] = []
    excluded_weight_files: list[str] = []
    opaque_hash_bound_files: list[str] = []
    for row in rows:
        source = verifier.ROOT / row["path"]
        if not source.is_file():
            raise AssertionError(f"manifest source missing: {source}")
        source_bytes = source.read_bytes()
        if len(source_bytes) != int(row["size_bytes"]):
            raise AssertionError(f"manifest size drift: {row['path']}")
        if sha256_bytes(source_bytes) != row["sha256"]:
            raise AssertionError(f"manifest hash drift: {row['path']}")
        excluded_weight = is_excluded_weight_binary(row["path"])
        if excluded_weight:
            excluded_weight_files.append(row["path"])
        rules: dict[str, int] = {}
        staged_bytes = source_bytes
        risks: tuple[str, ...] = ()
        opaque_hash_bound = row["path"] in HASH_BOUND_OPAQUE_PATHS
        if opaque_hash_bound:
            opaque_hash_bound_files.append(row["path"])
        if (
            source.suffix.lower() in TEXT_SUFFIXES
            and row["path"] != SELF_PATH
            and not opaque_hash_bound
        ):
            try:
                source_text = source_bytes.decode("utf-8")
            except UnicodeDecodeError:
                source_text = ""
            if source_text:
                staged_text, rules = sanitize_text(source_text)
                staged_bytes = staged_text.encode("utf-8")
                risks = remaining_path_risks(staged_text)
        if rules:
            transformed_files += 1
            transformation_counts.update(rules)
        if risks:
            residual_files.append(row["path"])
        details.append(
            {
                "path": row["path"],
                "source_size_bytes": len(source_bytes),
                "source_sha256": row["sha256"],
                "staged_size_bytes": len(staged_bytes),
                "staged_sha256": sha256_bytes(staged_bytes),
                "transformation_counts": rules,
                "remaining_path_risks": list(risks),
                "preservation_mode": (
                    "HASH_BOUND_CONTROL_EVIDENCE" if opaque_hash_bound else "STANDARD"
                ),
                "profile_action": (
                    "EXCLUDED_WEIGHT_BINARY" if excluded_weight else "INCLUDE"
                ),
            }
        )
    try:
        manifest_label = str(manifest.resolve().relative_to(verifier.ROOT.resolve()))
    except ValueError:
        manifest_label = "<EXTERNAL_VALIDATION_MANIFEST>"
    report = {
        "schema": "paper14_blind_review_sanitization_audit_v1",
        "status": "AUDIT_PASS" if not residual_files else "AUDIT_FAIL_RESIDUAL_PATHS",
        "source_manifest": manifest_label,
        "source_manifest_sha256": verifier.sha256(manifest),
        "source_manifest_file_count": len(rows),
        "payload_file_count": len(rows) - len(excluded_weight_files),
        "excluded_weight_binary_count": len(excluded_weight_files),
        "excluded_weight_binary_files": sorted(excluded_weight_files),
        "transformed_file_count": transformed_files,
        "transformation_counts": dict(sorted(transformation_counts.items())),
        "residual_path_file_count": len(residual_files),
        "residual_path_files": sorted(residual_files),
        "opaque_hash_bound_file_count": len(opaque_hash_bound_files),
        "opaque_hash_bound_files": sorted(opaque_hash_bound_files),
        "source_evidence_modified": False,
        "package_built": False,
        "control_plane_file_count": 0,
        "package_file_count": len(rows) - len(excluded_weight_files),
    }
    return report, details


def control_plane_sources(manifest: Path) -> list[Path]:
    """Return self-excluded release-control files that belong in a staged copy."""
    sources = [manifest.resolve()]
    if "_manifest_" in manifest.name:
        preflight = manifest.with_name(
            manifest.name.replace("_manifest_", "_preflight_").removesuffix(".csv")
            + ".json"
        ).resolve()
        if preflight.is_file():
            sources.append(preflight)
    return sources


def write_audit_receipt(path: Path, report: dict[str, object]) -> dict[str, object]:
    """Persist the read-only audit summary outside the staged review payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(path),
        "sha256": verifier.sha256(path),
    }


def ensure_safe_destination(destination: Path) -> Path:
    resolved = destination.resolve()
    root = verifier.ROOT.resolve()
    if resolved == root or resolved in root.parents or root in resolved.parents:
        raise ValueError("destination must be outside the source paper14 tree")
    if resolved.exists():
        raise FileExistsError(f"destination already exists: {resolved}")
    return resolved


def stage_package(destination: Path, manifest: Path, acknowledgement: str) -> dict[str, object]:
    if acknowledgement != ACKNOWLEDGEMENT:
        raise PermissionError(f"staging requires --ack {ACKNOWLEDGEMENT}")
    destination = ensure_safe_destination(destination)
    report, details = audit_payload(manifest)
    if report["status"] != "AUDIT_PASS":
        raise AssertionError("sanitization audit has residual path risks")
    destination.mkdir(parents=True, exist_ok=False)
    detail_by_path = {item["path"]: item for item in details}
    for row in load_manifest(manifest):
        relative = row["path"]
        detail = detail_by_path[relative]
        if detail["profile_action"] == "EXCLUDED_WEIGHT_BINARY":
            continue
        source = verifier.ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if detail["transformation_counts"]:
            text = source.read_text(encoding="utf-8")
            sanitized, _ = sanitize_text(text)
            target.write_text(sanitized, encoding="utf-8")
        else:
            shutil.copy2(source, target)
        if verifier.sha256(target) != detail["staged_sha256"]:
            raise AssertionError(f"staged hash mismatch: {relative}")
    control_details: list[dict[str, object]] = []
    for source in control_plane_sources(manifest):
        try:
            relative = str(source.relative_to(verifier.ROOT.resolve()))
        except ValueError:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = verifier.sha256(target)
        control_details.append(
            {
                "path": relative,
                "source_size_bytes": source.stat().st_size,
                "source_sha256": verifier.sha256(source),
                "staged_size_bytes": target.stat().st_size,
                "staged_sha256": digest,
                "transformation_counts": {},
                "remaining_path_risks": [],
                "profile_action": "CONTROL_PLANE_INCLUDE",
            }
        )
    report["control_plane_file_count"] = len(control_details)
    report["package_file_count"] = int(report["payload_file_count"]) + len(
        control_details
    )
    ledger = {
        **report,
        "status": "STAGED_COPY_VERIFIED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "destination": "<USER_SELECTED_STAGING_DIRECTORY>",
        "package_built": True,
        "files": details + control_details,
    }
    ledger_path = destination / "SANITIZATION_LEDGER.json"
    ledger_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "status": ledger["status"],
        "destination": str(destination),
        "payload_file_count": report["payload_file_count"],
        "package_file_count": report["package_file_count"],
        "control_plane_file_count": report["control_plane_file_count"],
        "transformed_file_count": report["transformed_file_count"],
        "ledger": str(ledger_path),
        "ledger_sha256": verifier.sha256(ledger_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--ack", default="")
    parser.add_argument(
        "--audit-receipt",
        type=Path,
        help="optional path for a deterministic read-only audit receipt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage:
        if args.destination is None:
            raise SystemExit("--destination is required with --stage")
        result = stage_package(args.destination, args.manifest, args.ack)
    else:
        result, _ = audit_payload(args.manifest)
        if args.audit_receipt is not None:
            result["audit_receipt"] = write_audit_receipt(args.audit_receipt, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
