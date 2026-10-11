#!/usr/bin/env python3
"""Run the hash-frozen PathMNIST verifier through a path-only staging view.

The formal runner recorded repository-root-relative artifact paths.  The
frozen verifier resolves feature paths relative to its ``output`` argument,
which duplicates that prefix.  This wrapper does not alter receipts, evidence,
metrics, or verifier logic.  It presents the original files through temporary
symlinks and calls the frozen verifier unchanged from the repository root.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterator

import verify_pathmnist_official_weight_detector_confirmation as frozen


PAPER14 = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PAPER14.parents[1]
DEFAULT_OUTPUT = PAPER14 / "results/pathmnist_official_weight_detector_confirmation_v1"
DEFAULT_DATASET = Path("/private/tmp/paper14-data/pathmnist.npz")
DEFAULT_AMENDMENT = PAPER14 / (
    "results/top_journal_program_v1/"
    "pathmnist_terminal_replay_path_resolution_amendment_2026-10-03.json"
)
DEFAULT_RECEIPT = PAPER14 / (
    "results/top_journal_program_v1/"
    "pathmnist_terminal_staged_replay_receipt_2026-10-03.json"
)
TOP_LEVEL_FILES = (
    "manifest.json",
    "seed_receipts.json",
    "detector_metrics.csv",
    "scores.csv",
)
RECORDED_OUTPUT_PREFIX = Path(
    "Blackdoor_1/paper14/results/pathmnist_official_weight_detector_confirmation_v1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_amendment(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN_BEFORE_CORRECTED_TERMINAL_REPLAY":
        raise RuntimeError("path-resolution amendment is not frozen")
    if payload.get("performance_endpoints_inspected") is not False:
        raise RuntimeError("amendment does not preserve endpoint sealing")
    if payload.get("scientific_decision_rules_changed") is not False:
        raise RuntimeError("amendment changes scientific decision rules")
    expected_verifier = payload.get("frozen_verifier_sha256")
    if expected_verifier != sha256_file(Path(frozen.__file__).resolve()):
        raise RuntimeError("frozen verifier hash does not match amendment")
    expected_wrapper = payload.get("staging_wrapper_sha256")
    if expected_wrapper != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("staging wrapper hash does not match amendment")
    return payload


@contextmanager
def staging_view(output: Path) -> Iterator[Path]:
    """Expose original evidence at the exact paths expected by the verifier."""

    output = output.resolve()
    for name in TOP_LEVEL_FILES:
        if not (output / name).is_file():
            raise FileNotFoundError(output / name)
    if not (output / "features").is_dir():
        raise FileNotFoundError(output / "features")

    with tempfile.TemporaryDirectory(prefix="paper14-path-terminal-replay-") as raw:
        stage = Path(raw)
        for name in TOP_LEVEL_FILES:
            (stage / name).symlink_to(output / name)
        nested = stage / RECORDED_OUTPUT_PREFIX
        nested.mkdir(parents=True)
        (nested / "features").symlink_to(output / "features", target_is_directory=True)
        yield stage


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def replay(
    output: Path,
    dataset: Path,
    amendment: Path,
    receipt_path: Path,
) -> dict[str, object]:
    amendment_payload = validate_amendment(amendment)
    evidence_hashes = {name: sha256_file(output / name) for name in TOP_LEVEL_FILES}
    original_cwd = Path.cwd()
    try:
        with staging_view(output) as stage:
            os.chdir(REPOSITORY_ROOT)
            result = frozen.verify(stage, dataset, allow_nonformal=False)
    finally:
        os.chdir(original_cwd)

    receipt: dict[str, object] = {
        "status": result["status"],
        "formal_evidence": result["formal_evidence"],
        "failures": result["failures"],
        "verified_seeds": result["verified_seeds"],
        "verified_method_rows": result["verified_method_rows"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "path_only_staging": True,
        "evidence_files_modified": False,
        "performance_endpoints_inspected_before_replay": False,
        "formal_output": str(output.resolve()),
        "dataset_sha256": sha256_file(dataset),
        "evidence_hashes": evidence_hashes,
        "frozen_verifier_sha256": sha256_file(Path(frozen.__file__).resolve()),
        "staging_wrapper_sha256": sha256_file(Path(__file__).resolve()),
        "amendment_sha256": sha256_file(amendment),
        "base_contract_sha256": amendment_payload["base_contract_sha256"],
    }
    atomic_json(receipt_path, receipt)
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
