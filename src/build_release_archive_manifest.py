#!/usr/bin/env python3
"""Build a non-destructive, preterminal release payload manifest.

The script hashes the final verifier's required payload without copying,
renaming, sanitizing, or packaging evidence.  Its outputs are a preflight
snapshot, not authorization to publish the bundle.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import verify_top_journal_final_bundle as verifier
from build_public_code_profile import project_code_files, selected_files


PROGRAM = verifier.PROGRAM
MANIFEST = PROGRAM / "release_archive_manifest_preterminal_2026-10-02.csv"
PREFLIGHT = PROGRAM / "release_archive_preflight_preterminal_2026-10-02.json"
STAGED_REPLAY_RECEIPT = (
    PROGRAM / "blind_review_staged_replay_receipt_preterminal_2026-10-03.json"
)
OUTPUTS = {MANIFEST.resolve(), PREFLIGHT.resolve()}
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

REVIEWER_RUNTIME_EVIDENCE = (
    verifier.ROOT / "data" / "official_pathmnist_weights" / "resnet18_28_1.pth",
    verifier.ROOT / "data" / "official_pathmnist_weights" / "resnet18_28_2.pth",
    verifier.ROOT / "results" / "bloodmnist_whitebox_adaptive_confirmation_v1" / "eligible_summary.csv",
    verifier.ROOT / "results" / "p6d_final_observable_adaptive_v1" / "development",
    verifier.ROOT / "results" / "p6d_final_observable_adaptive_v2" / "development",
    verifier.ROOT / "results" / "p6d_final_observable_adaptive_v3" / "development",
    verifier.ROOT / "results" / "p6d_final_observable_adaptive_v3" / "confirmation",
    verifier.ROOT / "results" / "phase_a_mnist5q_real_v1",
    verifier.PROGRAM / "final_bundle_receipt.json",
)


def reviewer_runtime_evidence_files() -> set[Path]:
    """Files needed by the shipped verifier/tests beyond their code closure."""
    files: set[Path] = set()
    excluded = {
        relative
        for relative, detail in verifier.staged_ledger_index().items()
        if detail.get("profile_action") == "EXCLUDED_WEIGHT_BINARY"
    }
    for entry in REVIEWER_RUNTIME_EVIDENCE:
        if entry.is_file():
            files.add(entry.resolve())
        elif entry.is_dir():
            files.update(
                path.resolve()
                for path in entry.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
        elif str(entry.relative_to(verifier.ROOT)) in excluded:
            files.add(entry.resolve())
        else:
            raise AssertionError(f"reviewer runtime evidence missing: {entry}")
    return files


def release_payload_paths() -> set[Path]:
    """Union scientific evidence with the complete runnable public-code profile."""
    return (
        set(verifier.required_file_paths())
        | set(selected_files())
        | reviewer_runtime_evidence_files()
    )


def classify(relative: str) -> tuple[str, str, str]:
    if relative.startswith("external/BackdoorBench/"):
        return (
            "third_party_backdoorbench",
            "CC_BY_NC_4_0_WITH_LISTED_EXCEPTIONS",
            "INCLUDE_ONLY_WITH_UPSTREAM_LICENSE_ATTRIBUTION_AND_NONCOMMERCIAL_BOUNDARY",
        )
    if relative.startswith("data/official_pathmnist_weights/"):
        return (
            "upstream_model_weights",
            "UPSTREAM_WEIGHT_TERMS_NOT_YET_CONFIRMED",
            "HOLD_PUBLIC_REDISTRIBUTION_PENDING_TERMS",
        )
    if (
        relative.startswith("results/pathmnist_official_weight_warmstart_v1/models/")
        and relative.endswith(".pt")
    ):
        return (
            "derived_model_weights",
            "PROJECT_AND_UPSTREAM_WEIGHT_TERMS_PENDING",
            "HOLD_PUBLIC_REDISTRIBUTION_PENDING_TERMS",
        )
    if relative.startswith("manuscript_"):
        return (
            "submission_manuscript",
            "AUTHOR_COPYRIGHT_SELECTION_PENDING",
            "INCLUDE_IN_REVIEW_PACKAGE; PUBLIC_RELEASE_AFTER_AUTHOR_CONFIRMATION",
        )
    if relative.startswith(("src/", "tests/", "configs/", "theory/", "tools/")):
        return (
            "project_owned_code",
            "PROJECT_CODE_LICENSE_PENDING",
            "HOLD_PUBLIC_RELEASE_UNTIL_LICENSE_SELECTED",
        )
    return (
        "project_generated_evidence",
        "ARTIFACT_LICENSE_PENDING",
        "INCLUDE_IN_REVIEW_PACKAGE; PUBLIC_RELEASE_AFTER_LICENSE_AND_PRIVACY_REVIEW",
    )


def text_risk_flags(path: Path) -> tuple[str, ...]:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return ()
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ("TEXT_DECODE_UNCHECKED",)
    flags: list[str] = []
    if re.search(r"/Users/[^/\s\"']+/", text) or re.search(
        r"[A-Za-z]:\\\\Users\\\\[^\\\s\"']+\\\\", text
    ):
        flags.append("LOCAL_USER_PATH")
    if "/private/tmp/" in text or "/tmp/" in text:
        flags.append("EPHEMERAL_ABSOLUTE_PATH")
    if re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text):
        flags.append("EMAIL_LIKE_STRING")
    if re.search(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b", text):
        flags.append("ORCID_LIKE_STRING")
    return tuple(flags)


def claim_c27_state() -> str:
    with (PROGRAM / "FINAL_CLAIM_MATRIX.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row["claim_id"] == "C27"]
    if len(matches) != 1:
        raise AssertionError("claim C27 must appear exactly once")
    return matches[0]["decision"]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> None:
    required = release_payload_paths()
    payload_paths = sorted(
        {path.resolve() for path in required if path.resolve() not in OUTPUTS},
        key=lambda path: str(path.relative_to(verifier.ROOT)),
    )
    rows: list[dict[str, object]] = []
    risk_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    component_counts: Counter[str] = Counter()
    risk_files: dict[str, list[str]] = {}
    total_bytes = 0
    for path in payload_paths:
        relative = str(path.relative_to(verifier.ROOT))
        zero_byte_vendor_marker = (
            path.exists()
            and path.stat().st_size == 0
            and relative.startswith("external/BackdoorBench/")
        )
        if not path.is_file() or (
            path.stat().st_size == 0 and not zero_byte_vendor_marker
        ):
            raise AssertionError(f"missing or empty release payload file: {path}")
        component, license_status, release_action = classify(relative)
        flags = text_risk_flags(path)
        if relative in {
            "src/build_release_archive_manifest.py",
            "src/build_blind_review_package.py",
        }:
            flags = tuple(
                flag
                for flag in flags
                if flag not in {"LOCAL_USER_PATH", "EPHEMERAL_ABSOLUTE_PATH"}
            )
        for flag in flags:
            risk_counts[flag] += 1
            risk_files.setdefault(flag, []).append(relative)
        action_counts[release_action] += 1
        component_counts[component] += 1
        size = path.stat().st_size
        total_bytes += size
        rows.append(
            {
                "path": relative,
                "size_bytes": size,
                "sha256": verifier.sha256(path),
                "component": component,
                "license_status": license_status,
                "release_action": release_action,
                "risk_flags": ";".join(flags),
            }
        )
    write_csv(MANIFEST, rows)

    c27 = claim_c27_state()
    staged_replay = json.loads(STAGED_REPLAY_RECEIPT.read_text(encoding="utf-8"))
    if staged_replay["status"] != (
        "PRETERMINAL_STAGED_REPLAY_PASS_FINAL_REBUILD_REQUIRED"
    ):
        raise AssertionError("preterminal staged replay receipt is not a pass")
    blockers = [
        {
            "id": "B01_TERMINAL_EXTERNAL_VALIDITY",
            "status": "OPEN" if c27 == "PENDING_TERMINAL" else "RECHECK",
            "evidence": f"FINAL_CLAIM_MATRIX.csv:C27={c27}",
            "resolution": "Integrate exactly one frozen terminal route, then rebuild this manifest.",
        },
        {
            "id": "B02_PROJECT_CODE_LICENSE",
            "status": "AUTHOR_DECISION_REQUIRED",
            "evidence": f"The public-code profile now proves a complete {len(project_code_files())}-file project-code closure with no binary artifacts, high-confidence secrets or residual sanitized paths, but publication_authorized remains false.",
            "resolution": "Rights holder selects a license that does not relicense third-party code.",
        },
        {
            "id": "B03_ARTIFACT_LICENSE",
            "status": "AUTHOR_DECISION_REQUIRED",
            "evidence": "No project-owned data/artifact license is present.",
            "resolution": "Select terms for generated tables, scores, figures and model artifacts.",
        },
        {
            "id": "B04_WEIGHT_REDISTRIBUTION",
            "status": "PRETERMINAL_FOUR_BINARY_FILTER_STAGED_VERIFIED_FINAL_REBUILD_REQUIRED",
            "evidence": f"The Zenodo weight record displays no license value. The staged replay excluded and ledger-verified {staged_replay['excluded_weight_binary_count']} official/derived binaries without reclassifying their terms.",
            "resolution": "Rebuild the same exclusion profile after terminal integration unless explicit authorization is archived; retain DOI, acquisition/member hashes, runners and regeneration instructions.",
        },
        {
            "id": "B05_ANONYMITY_AND_PORTABILITY",
            "status": "PRETERMINAL_SANITIZED_COPY_VERIFIED_FINAL_REBUILD_REQUIRED",
            "evidence": f"A disposable staged copy independently verified {staged_replay['package_file_count']} package files with {staged_replay['transformed_file_count']} transformed files and zero unexpected/residual path files; the source inventory still contains {risk_counts['LOCAL_USER_PATH']} local-path-bearing files by design.",
            "resolution": "Rebuild and independently verify the derived reviewer copy after terminal integration; preserve original evidence hashes.",
        },
        {
            "id": "B06_PUBLIC_ACCESSION",
            "status": "AUTHOR_ACTION_REQUIRED",
            "evidence": "No stable public repository URL or archival DOI is recorded.",
            "resolution": "Insert reviewer-access URL and final versioned DOI before submission.",
        },
        {
            "id": "B07_AUTHOR_STATEMENTS",
            "status": "AUTHOR_CONFIRMATION_REQUIRED",
            "evidence": "Submission statement placeholders remain explicit.",
            "resolution": "Confirm conflicts, funding, ethics, CRediT roles and AI disclosure.",
        },
        {
            "id": "B08_MNIST_HISTORICAL_BOUNDARY",
            "status": "CLOSED_BY_EXCLUSION_DOCUMENTED_IRRECOVERABLE",
            "evidence": "A repository-wide recovery audit found no historical package lock; the current replay pin is not back-projected, direct analysis inputs remain hash-bound, and raw MNIST files are excluded from release profiles.",
            "resolution": "Reopen only if a genuine historical environment record or explicit dataset terms become available.",
        },
        {
            "id": "B09_CLEAN_ARCHIVE_REPLAY",
            "status": "PRETERMINAL_PASS_FINAL_REPLAY_REQUIRED",
            "evidence": f"The disposable staged copy collected {staged_replay['pytest']['collected']} tests: {staged_replay['pytest']['passed']} passed and {staged_replay['pytest']['policy_skipped']} policy-scoped tests skipped; no unexpected skip was accepted.",
            "resolution": "After terminal integration, rebuild the selected release package and repeat the full staged verifier/tests before submission.",
        },
    ]
    preflight = {
        "schema": "paper14_release_archive_preflight_v1",
        "status": "PRETERMINAL_NOT_RELEASE_READY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "path": str(MANIFEST.relative_to(verifier.ROOT)),
            "sha256": verifier.sha256(MANIFEST),
            "payload_file_count": len(rows),
            "payload_total_bytes": total_bytes,
            "self_exclusion": [
                str(MANIFEST.relative_to(verifier.ROOT)),
                str(PREFLIGHT.relative_to(verifier.ROOT)),
            ],
        },
        "component_counts": dict(sorted(component_counts.items())),
        "release_action_counts": dict(sorted(action_counts.items())),
        "text_risk_counts": dict(sorted(risk_counts.items())),
        "text_risk_files": {key: sorted(value) for key, value in sorted(risk_files.items())},
        "raw_benchmark_archives_included": False,
        "backdoorbench_snapshot_fully_inventoried": True,
        "claim_c27_state": c27,
        "blockers": blockers,
        "active_blocker_count": sum(
            not item["status"].startswith("CLOSED_") for item in blockers
        ),
        "closed_control_count": sum(
            item["status"].startswith("CLOSED_") for item in blockers
        ),
        "interpretation": "A cryptographic payload inventory and release-risk audit, not a publication-ready archive or legal opinion.",
    }
    write_json(PREFLIGHT, preflight)
    print(json.dumps({
        "status": preflight["status"],
        "payload_file_count": len(rows),
        "payload_total_bytes": total_bytes,
        "manifest_sha256": preflight["manifest"]["sha256"],
        "open_blockers": preflight["active_blocker_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
