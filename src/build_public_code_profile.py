#!/usr/bin/env python3
"""Build a non-destructive, preterminal public-code profile manifest.

This profile is deliberately separate from the verifier/evidence payload.  It
inventory-hashes the complete project code closure, frozen machine-readable
protocols and the vendored BackdoorBench snapshot without copying files.  The
default run never stages or publishes an archive.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from build_blind_review_package import remaining_path_risks, sanitize_text


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "results" / "top_journal_program_v1"
MANIFEST = PROGRAM / "public_code_profile_manifest_preterminal_2026-10-03.csv"
PREFLIGHT = PROGRAM / "public_code_profile_preflight_preterminal_2026-10-03.json"
OUTPUTS = {MANIFEST.resolve(), PREFLIGHT.resolve()}
PROJECT_ROOTS = ("src", "tests", "configs", "theory")
ROOT_ENTRYPOINTS = ("README.md", "requirements-replay.txt")
PROGRAM_DOCUMENTS = (
    "REPRODUCE.md",
    "THIRD_PARTY_NOTICES.md",
    "ATTACK_SOURCE_AUDIT.md",
    "RELEASE_ARCHIVE_PLAN.md",
    "BLIND_REVIEW_SANITIZATION_PLAN.md",
    "PATHMNIST_WEIGHT_REDISTRIBUTION_AUDIT_2026-10-02.md",
    "MNIST_HISTORICAL_ENVIRONMENT_RECOVERY_AUDIT_2026-10-03.md",
    "PUBLIC_CODE_PROFILE_ADDENDUM_2026-10-03.md",
    "submission_author_actions_template_v1.json",
    "TARGET_JOURNAL_SELECTION_DOSSIER_2026-10-03.md",
)
TEXT_SUFFIXES = {
    ".bib",
    ".cfg",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}
BINARY_ARTIFACT_SUFFIXES = {".ckpt", ".npy", ".npz", ".pickle", ".pkl", ".pt", ".pth"}
SECRET_PATTERNS = {
    "OPENAI_STYLE_KEY": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "ANTHROPIC_STYLE_KEY": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    "GITHUB_TOKEN": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "AWS_ACCESS_KEY": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "PRIVATE_KEY_BLOCK": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eligible_file(path: Path) -> bool:
    if not path.is_file():
        return False
    parts = path.relative_to(ROOT).parts
    if any(part in {".git", "__pycache__"} for part in parts):
        return False
    if path.suffix == ".pyc" or path.name == ".DS_Store":
        return False
    return path.resolve() not in OUTPUTS


def project_code_files() -> set[Path]:
    files: set[Path] = set()
    for name in PROJECT_ROOTS:
        base = ROOT / name
        if base.is_dir():
            files.update(path.resolve() for path in base.rglob("*") if eligible_file(path))
    return files


def protocol_files() -> set[Path]:
    patterns = ("*protocol*.json", "*contract*.json")
    files: set[Path] = set()
    for pattern in patterns:
        files.update(
            path.resolve() for path in PROGRAM.glob(pattern) if eligible_file(path)
        )
    return files


def vendor_files() -> set[Path]:
    base = ROOT / "external" / "BackdoorBench"
    return {path.resolve() for path in base.rglob("*") if eligible_file(path)}


def selected_files() -> list[Path]:
    files = project_code_files() | protocol_files() | vendor_files()
    files.update((ROOT / name).resolve() for name in ROOT_ENTRYPOINTS)
    files.update((PROGRAM / name).resolve() for name in PROGRAM_DOCUMENTS)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise AssertionError(f"public-code profile source missing: {missing}")
    return sorted(files, key=lambda path: str(path.relative_to(ROOT)))


def classify(relative: str) -> tuple[str, str, str]:
    if relative.startswith("external/BackdoorBench/"):
        return (
            "third_party_backdoorbench",
            "CC_BY_NC_4_0_WITH_LISTED_EXCEPTIONS",
            "INCLUDE_WITH_UPSTREAM_LICENSE_AND_NONCOMMERCIAL_BOUNDARY",
        )
    if relative.startswith(PROJECT_ROOTS):
        return (
            "project_owned_code",
            "PROJECT_CODE_LICENSE_PENDING",
            "HOLD_PUBLICATION_UNTIL_RIGHTS_HOLDER_SELECTS_LICENSE",
        )
    return (
        "reproducibility_metadata",
        "MIXED_PROJECT_DOCUMENTATION_AND_PROTOCOL_METADATA",
        "INCLUDE_AFTER_LICENSE_AND_ANONYMITY_REVIEW",
    )


def inspect_text(path: Path) -> tuple[dict[str, int], tuple[str, ...], tuple[str, ...]]:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return {}, (), ()
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {}, ("TEXT_DECODE_UNCHECKED",), ()
    sanitized, substitutions = sanitize_text(text)
    residual = remaining_path_risks(sanitized)
    secrets = tuple(
        sorted(name for name, pattern in SECRET_PATTERNS.items() if pattern.search(text))
    )
    return substitutions, residual, secrets


def build_profile() -> tuple[list[dict[str, object]], dict[str, object]]:
    files = selected_files()
    project_expected = project_code_files()
    selected_set = set(files)
    missing_project_code = sorted(
        str(path.relative_to(ROOT)) for path in project_expected - selected_set
    )
    if missing_project_code:
        raise AssertionError(f"project code closure incomplete: {missing_project_code}")

    rows: list[dict[str, object]] = []
    component_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    substitution_counts: Counter[str] = Counter()
    residual_files: list[str] = []
    secret_files: dict[str, list[str]] = {}
    binary_artifacts: list[str] = []
    total_bytes = 0
    for path in files:
        relative = str(path.relative_to(ROOT))
        component, license_status, action = classify(relative)
        substitutions, residual, secrets = inspect_text(path)
        if path.suffix.lower() in BINARY_ARTIFACT_SUFFIXES:
            binary_artifacts.append(relative)
        if residual:
            residual_files.append(relative)
        for flag in secrets:
            secret_files.setdefault(flag, []).append(relative)
        substitution_counts.update(substitutions)
        component_counts[component] += 1
        action_counts[action] += 1
        size = path.stat().st_size
        total_bytes += size
        rows.append(
            {
                "path": relative,
                "size_bytes": size,
                "sha256": sha256(path),
                "component": component,
                "license_status": license_status,
                "release_action": action,
                "path_substitution_counts": json.dumps(
                    substitutions, sort_keys=True, separators=(",", ":")
                ),
                "residual_path_risks": ";".join(residual),
                "secret_flags": ";".join(secrets),
            }
        )
    if binary_artifacts:
        raise AssertionError(f"binary artifacts entered public-code profile: {binary_artifacts}")
    status = (
        "PRETERMINAL_PROJECT_LICENSE_REQUIRED"
        if not residual_files and not secret_files
        else "AUDIT_FAILURE"
    )
    preflight = {
        "schema": "paper14_public_code_profile_preflight_v1",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": "public_code",
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "complete_project_code_closure": not missing_project_code,
        "project_code_file_count": len(project_expected),
        "protocol_file_count": len(protocol_files()),
        "backdoorbench_file_count": len(vendor_files()),
        "component_counts": dict(sorted(component_counts.items())),
        "release_action_counts": dict(sorted(action_counts.items())),
        "path_substitution_counts": dict(sorted(substitution_counts.items())),
        "residual_path_file_count": len(residual_files),
        "residual_path_files": residual_files,
        "secret_flag_file_count": sum(len(paths) for paths in secret_files.values()),
        "secret_flag_files": {
            key: sorted(value) for key, value in sorted(secret_files.items())
        },
        "binary_artifact_count": len(binary_artifacts),
        "binary_artifacts": binary_artifacts,
        "project_code_license_selected": False,
        "publication_authorized": False,
        "excluded_scopes": [
            "datasets and model/checkpoint binaries",
            "manuscript PDFs and sources",
            "result score/metric tables and figures except selected reproducibility metadata",
            "Claude bridge outputs, prompts and temporary workspaces",
        ],
        "interpretation": "A complete code/profile inventory and safety audit, not a publication authorization or legal opinion.",
    }
    return rows, preflight


def write_outputs(rows: list[dict[str, object]], preflight: dict[str, object]) -> None:
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    preflight = {
        **preflight,
        "manifest": {
            "path": str(MANIFEST.relative_to(ROOT)),
            "sha256": sha256(MANIFEST),
            "self_exclusion": [
                str(MANIFEST.relative_to(ROOT)),
                str(PREFLIGHT.relative_to(ROOT)),
            ],
        },
    }
    PREFLIGHT.write_text(
        json.dumps(preflight, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    rows, preflight = build_profile()
    write_outputs(rows, preflight)
    print(
        json.dumps(
            {
                "status": preflight["status"],
                "file_count": preflight["file_count"],
                "project_code_file_count": preflight["project_code_file_count"],
                "protocol_file_count": preflight["protocol_file_count"],
                "backdoorbench_file_count": preflight["backdoorbench_file_count"],
                "residual_path_file_count": preflight["residual_path_file_count"],
                "secret_flag_file_count": preflight["secret_flag_file_count"],
                "binary_artifact_count": preflight["binary_artifact_count"],
                "manifest_sha256": sha256(MANIFEST),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
