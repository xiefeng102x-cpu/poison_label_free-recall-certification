#!/usr/bin/env python3
"""Replay the frozen PathMNIST verifier without a post-run dataset dependency.

Version 1 correctly repaired the recorded feature paths with a symlink-only
view, but reopened the temporary dataset while writing its final receipt.  A
host cleanup removed that file after the long frozen verifier had returned and
before the receipt was persisted.  This logistics-only wrapper binds and hashes
the dataset before replay, calls the same frozen verifier unchanged, and writes
the preverified digest without reopening the dataset afterwards.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import run_pathmnist_terminal_staged_replay as predecessor
import verify_pathmnist_official_weight_detector_confirmation as frozen


PAPER14 = Path(__file__).resolve().parents[1]
PROGRAM = PAPER14 / "results/top_journal_program_v1"
DEFAULT_OUTPUT = predecessor.DEFAULT_OUTPUT
DEFAULT_DATASET = predecessor.DEFAULT_DATASET
DEFAULT_AMENDMENT = PROGRAM / (
    "pathmnist_terminal_replay_dataset_lifetime_amendment_2026-10-04.json"
)
DEFAULT_RECEIPT = PROGRAM / (
    "pathmnist_terminal_staged_replay_v2_receipt_2026-10-04.json"
)
ACQUISITION_RECEIPT = PROGRAM / "pathmnist_acquisition_receipt.json"


def sha256_file(path: Path) -> str:
    return predecessor.sha256_file(path)


def validate_amendment(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN_BEFORE_DATASET_LIFETIME_REPLAY":
        raise RuntimeError("dataset-lifetime amendment is not frozen")
    if payload.get("performance_endpoints_inspected") is not False:
        raise RuntimeError("amendment does not preserve endpoint sealing")
    if payload.get("scientific_decision_rules_changed") is not False:
        raise RuntimeError("amendment changes scientific decision rules")
    expected = {
        "frozen_verifier_sha256": sha256_file(Path(frozen.__file__).resolve()),
        "predecessor_wrapper_sha256": sha256_file(
            Path(predecessor.__file__).resolve()
        ),
        "staging_wrapper_v2_sha256": sha256_file(Path(__file__).resolve()),
    }
    for key, observed in expected.items():
        if payload.get(key) != observed:
            raise RuntimeError(f"{key} does not match the frozen amendment")
    return payload


def bind_dataset(dataset: Path, output: Path) -> dict[str, Any]:
    """Hash the dataset once and bind it to both acquisition and run receipts."""
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    digest = sha256_file(dataset)
    size = dataset.stat().st_size
    acquisition = json.loads(ACQUISITION_RECEIPT.read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "acquisition_sha256": acquisition.get("sha256"),
        "manifest_sha256": manifest.get("frozen_hashes", {}).get("dataset_sha256"),
    }
    if any(value != digest for value in expected.values()):
        raise RuntimeError(f"dataset hash binding failed: {expected}")
    if int(acquisition.get("byte_count", -1)) != size:
        raise RuntimeError("dataset byte count differs from acquisition receipt")
    return {
        "sha256": digest,
        "byte_count": size,
        "acquisition_receipt_sha256": sha256_file(ACQUISITION_RECEIPT),
    }


def replay(
    output: Path,
    dataset: Path,
    amendment: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    amendment_payload = validate_amendment(amendment)
    dataset_binding = bind_dataset(dataset, output)
    evidence_hashes = {
        name: sha256_file(output / name) for name in predecessor.TOP_LEVEL_FILES
    }
    original_cwd = Path.cwd()
    try:
        with predecessor.staging_view(output) as stage:
            os.chdir(predecessor.REPOSITORY_ROOT)
            result = frozen.verify(stage, dataset, allow_nonformal=False)
    finally:
        os.chdir(original_cwd)

    # Deliberately do not reopen ``dataset`` here.  The frozen verifier already
    # checked the same path and digest at entry and replayed the dataset-backed
    # contracts.  A temporary-file cleanup after return must not erase result.
    receipt: dict[str, Any] = {
        "status": result["status"],
        "formal_evidence": result["formal_evidence"],
        "failures": result["failures"],
        "verified_seeds": result["verified_seeds"],
        "verified_method_rows": result["verified_method_rows"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "path_only_staging": True,
        "dataset_bound_before_replay": True,
        "dataset_reopened_after_replay": False,
        "dataset_present_after_replay": dataset.is_file(),
        "evidence_files_modified": False,
        "performance_endpoints_inspected_before_replay": False,
        "formal_output": str(output.resolve()),
        "dataset_sha256": dataset_binding["sha256"],
        "dataset_byte_count": dataset_binding["byte_count"],
        "acquisition_receipt_sha256": dataset_binding[
            "acquisition_receipt_sha256"
        ],
        "evidence_hashes": evidence_hashes,
        "frozen_verifier_sha256": sha256_file(Path(frozen.__file__).resolve()),
        "predecessor_wrapper_sha256": sha256_file(
            Path(predecessor.__file__).resolve()
        ),
        "staging_wrapper_v2_sha256": sha256_file(Path(__file__).resolve()),
        "amendment_sha256": sha256_file(amendment),
        "base_contract_sha256": amendment_payload["base_contract_sha256"],
        "predecessor_amendment_sha256": amendment_payload[
            "predecessor_amendment_sha256"
        ],
    }
    predecessor.atomic_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    result = replay(args.output, args.dataset, args.amendment, args.receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
