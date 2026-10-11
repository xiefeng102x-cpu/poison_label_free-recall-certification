#!/usr/bin/env python3
"""Verify the frozen PathMNIST acquisition contract without training models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PAPER14_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/pathmnist_acquisition_protocol_v1.json"
)
DEFAULT_DATASET = Path("/private/tmp/paper14-data/pathmnist.npz")
DEFAULT_RECEIPT = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/pathmnist_acquisition_receipt.json"
)


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    return file_digest(path, "sha256")


def read_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def verify(dataset: Path) -> dict[str, Any]:
    protocol = read_protocol()
    source = protocol["official_source"]
    contract = protocol["dataset_contract"]
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    byte_count = dataset.stat().st_size
    md5 = file_digest(dataset, "md5")
    sha256 = sha256_file(dataset)
    failures: list[str] = []
    if not 0 < byte_count <= int(source["maximum_accepted_bytes"]):
        failures.append("byte_count_out_of_contract")
    if md5 != source["expected_md5"]:
        failures.append("md5_mismatch")

    expected_keys = set(contract["expected_keys"])
    split_receipts: dict[str, Any] = {}
    with np.load(dataset, allow_pickle=False) as payload:
        actual_keys = set(payload.files)
        if actual_keys != expected_keys:
            failures.append("npz_key_mismatch")
        for split, count in contract["expected_split_counts"].items():
            images = payload[f"{split}_images"]
            labels = payload[f"{split}_labels"]
            flattened = labels.reshape(-1)
            expected_image_shape = (int(count), 28, 28, 3)
            expected_label_shape = (int(count), 1)
            if images.shape != expected_image_shape:
                failures.append(f"{split}_image_shape_mismatch")
            if labels.shape != expected_label_shape:
                failures.append(f"{split}_label_shape_mismatch")
            if str(images.dtype) != contract["expected_image_dtype"]:
                failures.append(f"{split}_image_dtype_mismatch")
            if not np.issubdtype(labels.dtype, np.integer):
                failures.append(f"{split}_label_dtype_not_integer")
            observed_labels = sorted(int(value) for value in np.unique(flattened))
            if observed_labels != contract["expected_label_set"]:
                failures.append(f"{split}_label_set_mismatch")
            split_receipts[split] = {
                "image_shape": list(images.shape),
                "image_dtype": str(images.dtype),
                "label_shape": list(labels.shape),
                "label_dtype": str(labels.dtype),
                "label_set": observed_labels,
                "class_histogram": {
                    str(label): int(np.sum(flattened == label))
                    for label in contract["expected_label_set"]
                },
                "pixel_min": int(images.min()),
                "pixel_max": int(images.max()),
            }

    return {
        "status": "PASS" if not failures else "FAIL",
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "dataset": str(dataset),
        "byte_count": byte_count,
        "md5": md5,
        "sha256": sha256,
        "split_receipts": split_receipts,
        "failures": failures,
        "training_authorized": not failures,
    }


def main(args: argparse.Namespace) -> None:
    receipt = verify(args.dataset)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    main(parser.parse_args())
