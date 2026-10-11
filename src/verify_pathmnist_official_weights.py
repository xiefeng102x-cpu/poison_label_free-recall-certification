#!/usr/bin/env python3
"""Verify selectively extracted official PathMNIST ResNet-18 checkpoints.

This is a provenance and clean-baseline check only.  It does not train an
attack and it does not turn the official clean checkpoints into detector
evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

import pathmnist_external_validation as core
import pathmnist_official_resnet18 as architecture
import run_pathmnist_resnet18_victim_pilot as victim_runner


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path("/private/tmp/paper14-data/pathmnist.npz")
DEFAULT_RECEIPT = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/pathmnist_official_weights_acquisition_receipt.json"
)
DEFAULT_OUTPUT = (
    PAPER14_ROOT
    / "results/pathmnist_official_weights_clean_validation_v1/verification.json"
)


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def crc32_hex(path: Path) -> str:
    checksum = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum = zlib.crc32(block, checksum)
    return f"{checksum & 0xFFFFFFFF:08x}"


def evaluate_checkpoint(
    path: Path, data: dict[str, dict[str, np.ndarray]]
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if sorted(checkpoint) != ["net"]:
        raise ValueError(f"unexpected top-level checkpoint keys: {sorted(checkpoint)}")
    model = architecture.PathMNISTResNet18()
    model.load_state_dict(checkpoint["net"], strict=True)
    result: dict[str, Any] = {
        "strict_load": True,
        "trainable_parameter_count": sum(p.numel() for p in model.parameters()),
    }
    for split_key, split_name in (("val", "validation"), ("test", "test")):
        labels = data[split_key]["labels"]
        probabilities = victim_runner.predict_probabilities(
            model, data[split_key]["images"], batch_size=512
        )
        predictions = probabilities.argmax(axis=1)
        result[split_name] = {
            "macro_ovr_auc": victim_runner.macro_auc(labels, probabilities),
            "overall_accuracy": float(np.mean(predictions == labels)),
            "balanced_accuracy": core.balanced_accuracy(labels, predictions),
            "sample_count": int(len(labels)),
        }
    return result


def verify(dataset: Path, receipt_path: Path) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if core.sha256_file(dataset) != (
        "81823f52dc622e69db2db4c72f8e8e617938dd6864d3c1f23d4e49724a28ea72"
    ):
        failures.append("PathMNIST dataset SHA-256 mismatch")
    data = core.load_pathmnist(dataset)
    members: list[dict[str, Any]] = []
    for expected in receipt["extraction"]["members"]:
        path = PAPER14_ROOT.parent / expected["local_path"]
        observed = {
            "name": expected["name"],
            "path": str(path),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "md5": digest(path, "md5") if path.exists() else None,
            "sha256": digest(path, "sha256") if path.exists() else None,
            "crc32_hex": crc32_hex(path) if path.exists() else None,
        }
        for field, expected_field in (
            ("size_bytes", "size_bytes"),
            ("md5", "observed_md5"),
            ("sha256", "observed_sha256"),
            ("crc32_hex", "zip_crc32_hex"),
        ):
            if observed[field] != expected[expected_field]:
                failures.append(f"{expected['name']}: {field} mismatch")
        if path.exists() and not any(
            failure.startswith(expected["name"] + ":") for failure in failures
        ):
            try:
                observed["clean_metrics"] = evaluate_checkpoint(path, data)
            except Exception as error:  # pragma: no cover - surfaced in receipt
                failures.append(f"{expected['name']}: evaluation failed: {error}")
        members.append(observed)
    return {
        "status": "PASS" if not failures else "FAIL",
        "claim_scope": receipt["claim_scope"],
        "dataset": str(dataset),
        "dataset_sha256": core.sha256_file(dataset),
        "acquisition_receipt": str(receipt_path),
        "acquisition_receipt_sha256": core.sha256_file(receipt_path),
        "members": members,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = verify(args.dataset, args.receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
