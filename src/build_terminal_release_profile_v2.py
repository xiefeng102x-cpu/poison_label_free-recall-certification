#!/usr/bin/env python3
"""Build the post-integration public-code and release profiles for paper14.

The terminal workflow is intentionally two phase. ``prepare-manifests`` writes
hash inventories only after the PathMNIST terminal route, claim capsule, TIFS
receipt, and complete source regression pass.  A separate reviewer-package
replay then consumes the release manifest. ``finalize`` verifies that replay
and emits the required terminal release-profile receipt.  Author-controlled
licences, accessions, identities, and declarations remain explicit blockers.

``preterminal-self-check`` is outcome blind and opens no terminal receipt.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable

import build_public_code_profile as public
import build_release_archive_manifest as legacy_release
import select_pathmnist_terminal_route_v2 as selector
import verify_top_journal_final_bundle_v2 as final_v2


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "results/top_journal_program_v1"
EXECUTION_CONTRACT = PROGRAM / "pathmnist_terminal_integration_execution_contract_v2.json"
CAPSULE = PROGRAM / "pathmnist_terminal_claim_capsule_v2_2026-10-04.json"
CAPSULE_VERIFICATION = PROGRAM / (
    "pathmnist_terminal_claim_capsule_verification_v2_2026-10-04.json"
)
TIFS_TERMINAL_RECEIPT = PROGRAM / "tifs_main_terminal_receipt_2026-10-04.json"
SOURCE_REGRESSION = PROGRAM / "full_source_regression_terminal_2026-10-04.json"
REVIEWER_RECEIPT = PROGRAM / "reviewer_package_terminal_receipt_2026-10-04.json"
FINAL_RECEIPT = PROGRAM / "release_profile_terminal_receipt_2026-10-04.json"
PUBLIC_MANIFEST = PROGRAM / "public_code_profile_manifest_terminal_2026-10-04.csv"
PUBLIC_PREFLIGHT = PROGRAM / "public_code_profile_preflight_terminal_2026-10-04.json"
RELEASE_MANIFEST = PROGRAM / "release_archive_manifest_terminal_2026-10-04.csv"
RELEASE_PREFLIGHT = PROGRAM / "release_archive_preflight_terminal_2026-10-04.json"
FORMAL_OUTPUT = ROOT / "results/pathmnist_official_weight_detector_confirmation_v1"

TIFS_REVIEWER_RUNTIME_FILES = (
    ROOT / "manuscript_tifs_en/output/main_tifs.log",
    ROOT / "manuscript_tifs_en/output/main_tifs.bbl",
    ROOT / "manuscript_tifs_en/output/supplement_tifs.log",
    ROOT / "manuscript_tifs_en/output/supplement_tifs.bbl",
    ROOT / "manuscript_tifs_en/references_en.bib",
    ROOT / "manuscript_tifs_en/references_tifs.bib",
)

CIRCULAR_OR_POSTPROFILE_OUTPUTS = {
    REVIEWER_RECEIPT.resolve(),
    FINAL_RECEIPT.resolve(),
    (PROGRAM / "final_bundle_v2_receipt_2026-10-04.json").resolve(),
    RELEASE_MANIFEST.resolve(),
    RELEASE_PREFLIGHT.resolve(),
}
AUTHOR_BLOCKERS = (
    "B02_PROJECT_CODE_LICENSE",
    "B03_ARTIFACT_LICENSE",
    "B06_PUBLIC_ACCESSION",
    "B07_AUTHOR_STATEMENTS",
)


def sha256(path: Path) -> str:
    return final_v2.sha256(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("cannot write an empty terminal manifest")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_pass(path: Path, statuses: set[str]) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required terminal receipt is absent: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") not in statuses:
        raise RuntimeError(f"terminal receipt is not in {sorted(statuses)}: {path.name}")
    return payload


def preterminal_self_check() -> dict[str, Any]:
    contract = json.loads(EXECUTION_CONTRACT.read_text(encoding="utf-8"))
    errors: list[str] = []
    expected = str(FINAL_RECEIPT.relative_to(ROOT))
    if expected not in contract.get("required_new_terminal_outputs", []):
        errors.append("terminal release-profile receipt is absent from V2 contract")
    for path in (
        ROOT / "src/build_public_code_profile.py",
        ROOT / "src/build_release_archive_manifest.py",
        ROOT / "src/build_blind_review_package.py",
        ROOT / "src/verify_blind_review_package.py",
    ):
        if not path.is_file():
            errors.append(f"release-profile dependency missing: {path.name}")
    result = {
        "status": "PASS_PRETERMINAL_RELEASE_PROFILE_PREP" if not errors else "FAIL",
        "errors": errors,
        "terminal_receipt_target": expected,
        "terminal_manifest_targets": [
            str(path.relative_to(ROOT))
            for path in (
                PUBLIC_MANIFEST,
                PUBLIC_PREFLIGHT,
                RELEASE_MANIFEST,
                RELEASE_PREFLIGHT,
            )
        ],
        "author_blockers_preserved": list(AUTHOR_BLOCKERS),
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
    replay_receipt: Path = selector.DEFAULT_REPLAY_RECEIPT,
    route_receipt: Path = selector.DEFAULT_SELECTION_RECEIPT,
    formal_output: Path = FORMAL_OUTPUT,
) -> dict[str, Any]:
    control = final_v2.verify_terminal_control(
        replay_receipt, route_receipt, formal_output
    )
    if control["status"] != "PASS_TERMINAL_CONTROL":
        raise RuntimeError("; ".join(control["errors"]))
    capsule_verification = load_pass(CAPSULE_VERIFICATION, {"PASS"})
    tifs = load_pass(TIFS_TERMINAL_RECEIPT, {"PASS_TERMINAL"})
    regression = load_pass(
        SOURCE_REGRESSION, {"PASS_TERMINAL_SOURCE_REGRESSION"}
    )
    route = control["selected_route"]
    for label, payload in (
        ("capsule", capsule_verification),
        ("TIFS", tifs),
        ("source regression", regression),
    ):
        if payload.get("selected_route") != route:
            raise RuntimeError(f"{label} receipt route mismatch")

    contract = json.loads(EXECUTION_CONTRACT.read_text(encoding="utf-8"))
    marker = final_v2.verify_route_markers(contract, route)
    capsule_binding = final_v2.verify_capsule_surface_binding(
        contract, route, CAPSULE
    )
    route_record = json.loads(route_receipt.read_text(encoding="utf-8"))
    c27 = final_v2.verify_c27_decision(route_record)
    errors = marker["errors"] + capsule_binding["errors"] + c27["errors"]
    if errors:
        raise RuntimeError("; ".join(errors))
    return {
        "control": control,
        "contract": contract,
        "capsule_verification": capsule_verification,
        "tifs": tifs,
        "regression": regression,
        "c27_decision": c27["c27_decision"],
    }


def write_public_profile(preconditions: dict[str, Any]) -> dict[str, Any]:
    rows, preflight = public.build_profile()
    if preflight["residual_path_file_count"] or preflight["secret_flag_file_count"]:
        raise RuntimeError("public-code profile contains residual paths or secrets")
    if preflight["binary_artifact_count"]:
        raise RuntimeError("public-code profile contains binary artifacts")
    atomic_csv(PUBLIC_MANIFEST, rows)
    control = preconditions["control"]
    terminal = {
        **preflight,
        "schema": "paper14_public_code_profile_terminal_v2",
        "status": "PASS_TERMINAL_CODE_PROFILE_AUTHOR_LICENSE_REQUIRED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_route": control["selected_route"],
        "replay_receipt_sha256": control["replay_receipt_sha256"],
        "route_receipt_sha256": control["route_receipt_sha256"],
        "manifest": {
            "path": str(PUBLIC_MANIFEST.relative_to(ROOT)),
            "sha256": sha256(PUBLIC_MANIFEST),
            "self_exclusion": [
                str(PUBLIC_MANIFEST.relative_to(ROOT)),
                str(PUBLIC_PREFLIGHT.relative_to(ROOT)),
            ],
        },
        "scientific_bundle_closed": True,
        "project_code_license_selected": False,
        "publication_authorized": False,
        "submission_ready": False,
    }
    final_v2.atomic_json(PUBLIC_PREFLIGHT, terminal)
    return terminal


def terminal_payload_paths(
    contract: dict[str, Any], extra: Iterable[Path] = ()
) -> list[Path]:
    paths = {path.resolve() for path in legacy_release.release_payload_paths()}
    # The reviewer package runs the complete test suite, so it must carry the
    # full terminal control/evidence directory rather than only the legacy
    # verifier inventory. Circular outputs are removed below.
    paths.update(
        path.resolve()
        for path in PROGRAM.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    for relative in (
        contract["mutable_integration_targets"]
        + contract["protected_history"]
        + contract["required_terminal_inputs"]
        + contract["required_new_terminal_outputs"]
    ):
        path = (ROOT / relative).resolve()
        if path.is_file() and path not in CIRCULAR_OR_POSTPROFILE_OUTPUTS:
            paths.add(path)
    for relative in (
        "manuscript_tifs_en/main_tifs.tex",
        "manuscript_tifs_en/output/main_tifs.pdf",
        "manuscript_tifs_en/supplement_tifs.tex",
        "manuscript_tifs_en/output/supplement_tifs.pdf",
        "results/top_journal_program_v1/pathmnist_terminal_replay_v2_supervision_incident_2026-10-04.json",
        "results/top_journal_program_v1/pathmnist_terminal_replay_v2_supervision_receipt_2026-10-04.json",
        "results/top_journal_program_v1/full_source_regression_preterminal_v2_2026-10-04.json",
    ):
        path = (ROOT / relative).resolve()
        if path.is_file():
            paths.add(path)
    paths.update(path.resolve() for path in extra if path.is_file())
    paths.update(path.resolve() for path in TIFS_REVIEWER_RUNTIME_FILES if path.is_file())
    paths.difference_update(CIRCULAR_OR_POSTPROFILE_OUTPUTS)
    return sorted(paths, key=lambda path: str(path.relative_to(ROOT)))


def build_release_rows(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    risk_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    component_counts: Counter[str] = Counter()
    risk_files: dict[str, list[str]] = {}
    total_bytes = 0
    for path in paths:
        relative = str(path.relative_to(ROOT))
        zero_byte_vendor_marker = (
            path.stat().st_size == 0 and relative.startswith("external/BackdoorBench/")
        )
        if not path.is_file() or (path.stat().st_size == 0 and not zero_byte_vendor_marker):
            raise RuntimeError(f"missing or empty terminal release payload: {relative}")
        component, licence, action = legacy_release.classify(relative)
        flags = legacy_release.text_risk_flags(path)
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
                "license_status": licence,
                "release_action": action,
                "risk_flags": ";".join(flags),
            }
        )
    return rows, {
        "payload_file_count": len(rows),
        "payload_total_bytes": total_bytes,
        "component_counts": dict(sorted(component_counts.items())),
        "release_action_counts": dict(sorted(action_counts.items())),
        "text_risk_counts": dict(sorted(risk_counts.items())),
        "text_risk_files": {
            key: sorted(values) for key, values in sorted(risk_files.items())
        },
    }


def blocker_ledger(
    *, route: str, c27_decision: str, regression: dict[str, Any], reviewer: dict[str, Any] | None
) -> list[dict[str, str]]:
    reviewer_pass = reviewer is not None and reviewer.get("status") == (
        "PASS_TERMINAL_REVIEWER_PACKAGE"
    )
    return [
        {
            "id": "B01_TERMINAL_EXTERNAL_VALIDITY",
            "status": "CLOSED_BY_FROZEN_ROUTE_INTEGRATION",
            "evidence": f"{route}; FINAL_CLAIM_MATRIX.csv:C27={c27_decision}",
        },
        {
            "id": "B02_PROJECT_CODE_LICENSE",
            "status": "AUTHOR_DECISION_REQUIRED",
            "evidence": "Project-code inventory passes, but no rights-holder licence choice is inferred.",
        },
        {
            "id": "B03_ARTIFACT_LICENSE",
            "status": "AUTHOR_DECISION_REQUIRED",
            "evidence": "Generated evidence and model-artifact terms remain an author/rights-holder choice.",
        },
        {
            "id": "B04_WEIGHT_REDISTRIBUTION",
            "status": "CLOSED_BY_VERIFIED_EXCLUSION" if reviewer_pass else "REVIEWER_REPLAY_REQUIRED",
            "evidence": "Official and official-initialized PathMNIST binaries remain excluded unless explicit terms are archived.",
        },
        {
            "id": "B05_ANONYMITY_AND_PORTABILITY",
            "status": "CLOSED_BY_TERMINAL_REVIEWER_REPLAY" if reviewer_pass else "REVIEWER_REPLAY_REQUIRED",
            "evidence": "Terminal reviewer package must independently verify its sanitized ledger and file closure.",
        },
        {
            "id": "B06_PUBLIC_ACCESSION",
            "status": "AUTHOR_ACTION_REQUIRED",
            "evidence": "No stable public repository URL or archival DOI is inferred.",
        },
        {
            "id": "B07_AUTHOR_STATEMENTS",
            "status": "AUTHOR_CONFIRMATION_REQUIRED",
            "evidence": "Identities, conflicts, funding, CRediT, ethics and AI disclosure remain author-confirmed.",
        },
        {
            "id": "B08_MNIST_HISTORICAL_BOUNDARY",
            "status": "CLOSED_BY_EXCLUSION_DOCUMENTED_IRRECOVERABLE",
            "evidence": "The historical environment is not back-projected and raw MNIST remains excluded.",
        },
        {
            "id": "B09_CLEAN_ARCHIVE_REPLAY",
            "status": "CLOSED_BY_TERMINAL_REVIEWER_REPLAY" if reviewer_pass else "REVIEWER_REPLAY_REQUIRED",
            "evidence": (
                f"Source regression passed {regression['pytest']['passed']}/"
                f"{regression['pytest']['tests']}; reviewer replay remains independently gated."
            ),
        },
    ]


def prepare_manifests() -> dict[str, Any]:
    preconditions = terminal_preconditions()
    public_preflight = write_public_profile(preconditions)
    paths = terminal_payload_paths(
        preconditions["contract"], extra=(PUBLIC_MANIFEST, PUBLIC_PREFLIGHT)
    )
    rows, summary = build_release_rows(paths)
    atomic_csv(RELEASE_MANIFEST, rows)
    control = preconditions["control"]
    blockers = blocker_ledger(
        route=control["selected_route"],
        c27_decision=preconditions["c27_decision"],
        regression=preconditions["regression"],
        reviewer=None,
    )
    preflight = {
        "schema": "paper14_release_archive_preflight_terminal_v2",
        "status": "PASS_TERMINAL_MANIFEST_PREP_REVIEWER_REPLAY_REQUIRED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_route": control["selected_route"],
        "claim_c27_state": preconditions["c27_decision"],
        "manifest": {
            "path": str(RELEASE_MANIFEST.relative_to(ROOT)),
            "sha256": sha256(RELEASE_MANIFEST),
            "self_exclusion": [
                str(RELEASE_MANIFEST.relative_to(ROOT)),
                str(RELEASE_PREFLIGHT.relative_to(ROOT)),
            ],
            **summary,
        },
        "public_code_profile_sha256": sha256(PUBLIC_PREFLIGHT),
        "blockers": blockers,
        "active_blocker_count": sum(
            not item["status"].startswith("CLOSED_") for item in blockers
        ),
        "submission_ready": False,
        "publication_authorized": False,
        "interpretation": "Terminal scientific/profile inventory prepared; reviewer replay and author-controlled release decisions remain.",
    }
    final_v2.atomic_json(RELEASE_PREFLIGHT, preflight)
    return {
        "status": preflight["status"],
        "selected_route": control["selected_route"],
        "public_manifest_sha256": sha256(PUBLIC_MANIFEST),
        "public_preflight_sha256": sha256(PUBLIC_PREFLIGHT),
        "release_manifest_sha256": sha256(RELEASE_MANIFEST),
        "release_preflight_sha256": sha256(RELEASE_PREFLIGHT),
        **summary,
        "submission_ready": False,
    }


def finalize() -> dict[str, Any]:
    preconditions = terminal_preconditions()
    reviewer = load_pass(REVIEWER_RECEIPT, {"PASS_TERMINAL_REVIEWER_PACKAGE"})
    if reviewer.get("source_manifest_sha256") != sha256(RELEASE_MANIFEST):
        raise RuntimeError("reviewer package is not bound to the terminal release manifest")
    public_preflight = load_pass(
        PUBLIC_PREFLIGHT, {"PASS_TERMINAL_CODE_PROFILE_AUTHOR_LICENSE_REQUIRED"}
    )
    prepared = load_pass(
        RELEASE_PREFLIGHT,
        {
            "PASS_TERMINAL_MANIFEST_PREP_REVIEWER_REPLAY_REQUIRED",
            "PASS_TERMINAL_RELEASE_PROFILE_AUTHOR_ACTIONS_REMAIN",
        },
    )
    control = preconditions["control"]
    blockers = blocker_ledger(
        route=control["selected_route"],
        c27_decision=preconditions["c27_decision"],
        regression=preconditions["regression"],
        reviewer=reviewer,
    )
    final_preflight = {
        **prepared,
        "status": "PASS_TERMINAL_RELEASE_PROFILE_AUTHOR_ACTIONS_REMAIN",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "blockers": blockers,
        "active_blocker_count": sum(
            not item["status"].startswith("CLOSED_") for item in blockers
        ),
        "reviewer_package_receipt_sha256": sha256(REVIEWER_RECEIPT),
        "submission_ready": False,
        "publication_authorized": False,
    }
    final_v2.atomic_json(RELEASE_PREFLIGHT, final_preflight)
    receipt = {
        "schema": "paper14_release_profile_terminal_v2",
        "status": "PASS_TERMINAL_RELEASE_PROFILE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_route": control["selected_route"],
        "claim_c27_state": preconditions["c27_decision"],
        "public_manifest_sha256": sha256(PUBLIC_MANIFEST),
        "public_preflight_sha256": sha256(PUBLIC_PREFLIGHT),
        "release_manifest_sha256": sha256(RELEASE_MANIFEST),
        "release_preflight_sha256": sha256(RELEASE_PREFLIGHT),
        "reviewer_package_receipt_sha256": sha256(REVIEWER_RECEIPT),
        "source_regression_receipt_sha256": sha256(SOURCE_REGRESSION),
        "project_code_file_count": public_preflight["project_code_file_count"],
        "payload_file_count": final_preflight["manifest"]["payload_file_count"],
        "author_blockers": list(AUTHOR_BLOCKERS),
        "active_blocker_count": final_preflight["active_blocker_count"],
        "scientific_route_closed": True,
        "reviewer_package_replayed": True,
        "submission_ready": False,
        "publication_authorized": False,
        "interpretation": "The terminal release profile is reproducible and sanitized; submission/publication still requires the listed author-controlled decisions.",
    }
    final_v2.atomic_json(FINAL_RECEIPT, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("preterminal-self-check", "prepare-manifests", "finalize"),
        required=True,
    )
    args = parser.parse_args()
    if args.mode == "preterminal-self-check":
        result = preterminal_self_check()
    elif args.mode == "prepare-manifests":
        result = prepare_manifests()
    else:
        result = finalize()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
