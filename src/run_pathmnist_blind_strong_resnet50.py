#!/usr/bin/env python3
"""Frozen strong-architecture wrapper for the blind PathMNIST certificate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import pathmnist_official_resnet50 as architecture
import run_pathmnist_blind_e2e_multidetector as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pathmnist_blind_strong_resnet50_certificate_v1.json"
PREREG = ROOT / "auditability_certificate_v1/PREREG_BLIND_STRONG_RESNET50_V1.md"
FREEZE = ROOT / "auditability_certificate_v1/BLIND_STRONG_RESNET50_FREEZE_V1.json"
CERT_RUNNER = ROOT / "auditability_certificate_v1/analysis/run_pathmnist_blind_extension_certificate.py"
VERIFIER = ROOT / "src/verify_pathmnist_blind_extension.py"
WEIGHT_RECEIPT = ROOT / "results/top_journal_program_v1/pathmnist_official_resnet50_weights_receipt.json"
DEFAULT_OUTPUT = ROOT / "results/pathmnist_blind_strong_resnet50_certificate_v1"
SEEDS = (2201, 2202, 2203, 2204, 2205)
WEIGHT_MAP = {
    2201: "resnet50_28_1.pth",
    2202: "resnet50_28_3.pth",
    2203: "resnet50_28_1.pth",
    2204: "resnet50_28_3.pth",
    2205: "resnet50_28_1.pth",
}
ACK = "OPEN_BLIND_STRONG_RESNET50_SEEDS_2201_2205"


def _hidden(
    model: torch.nn.Module,
    images: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            values = base.engine._tensor_batch(images[start : start + batch_size], device)
            hidden = model.hidden(values)
            if hidden.ndim != 2 or hidden.shape[1] != architecture.HIDDEN_DIMENSION:
                raise RuntimeError("official ResNet-50 hidden layer must be 2048-D")
            pieces.append(hidden.cpu().numpy().astype(np.float32))
    return np.concatenate(pieces, axis=0)


_BASE_WRITE_SEED_SCORES = base.write_seed_scores


def _write_seed_scores(*args: Any, **kwargs: Any):
    """Reuse the frozen score implementation while giving victims honest IDs."""
    score_path, truth_path, receipts = _BASE_WRITE_SEED_SCORES(*args, **kwargs)
    old = "pathmnist_badnets_resnet18_seed"
    new = "pathmnist_badnets_resnet50_seed"
    for path in (score_path, truth_path):
        text = path.read_text(encoding="utf-8")
        if old not in text:
            raise RuntimeError(f"expected base victim identifier missing from {path}")
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text.replace(old, new), encoding="utf-8")
        os.replace(tmp, path)
    return score_path, truth_path, receipts


def verify_freeze() -> dict[str, Any]:
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    expected = {
        "config_sha256": base.sha256(CONFIG),
        "prereg_sha256": base.sha256(PREREG),
        "runner_sha256": base.sha256(Path(__file__)),
        "base_runner_sha256": base.sha256(Path(base.__file__)),
        "certificate_runner_sha256": base.sha256(CERT_RUNNER),
        "verifier_sha256": base.sha256(VERIFIER),
        "certificate_library_sha256": base.sha256(
            ROOT / "auditability_certificate_v1/cert/certificate.py"
        ),
        "architecture_sha256": base.sha256(Path(architecture.__file__)),
        "weight_receipt_sha256": base.sha256(WEIGHT_RECEIPT),
    }
    if frozen.get("status") != "FROZEN_BEFORE_UNBLINDING":
        raise RuntimeError("strong-architecture freeze status is invalid")
    if frozen.get("hashes") != expected:
        raise RuntimeError("strong-architecture frozen file hashes do not match")
    return frozen


def configure_base() -> None:
    base.CONFIG = CONFIG
    base.PREREG = PREREG
    base.FREEZE = FREEZE
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.SEEDS = SEEDS
    base.WEIGHT_MAP = WEIGHT_MAP
    base.ACK = ACK
    base.architecture = architecture
    base.verify_freeze = verify_freeze
    base.write_seed_scores = _write_seed_scores
    base.engine.architecture = architecture
    base.engine.PROTOCOL_PATH = CONFIG
    base.engine.CONFIRMATION_SEEDS = SEEDS
    base.engine.WEIGHT_MAP = WEIGHT_MAP
    base.engine.WEIGHT_RECEIPT = WEIGHT_RECEIPT
    base.engine._hidden = _hidden


def run_formal(dataset: Path, output: Path, *, resume: bool, ack: str | None) -> dict[str, Any]:
    configure_base()
    manifest = base.run_formal(dataset, output, resume=resume, ack=ack)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frozen_hashes"]["extension_runner_sha256"] = base.sha256(Path(__file__))
    manifest["frozen_hashes"]["base_orchestrator_sha256"] = base.sha256(Path(base.__file__))
    manifest["architecture"] = "official PathMNIST 28x28 ResNet-50"
    manifest["hidden_dimension"] = architecture.HIDDEN_DIMENSION
    base.atomic_json(manifest_path, manifest)
    return manifest


def selftest() -> None:
    configure_base()
    receipt = json.loads(WEIGHT_RECEIPT.read_text(encoding="utf-8"))
    for seed in SEEDS:
        path = base.engine.resolve_weight(receipt, seed)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = architecture.PathMNISTResNet50()
        model.load_state_dict(checkpoint["net"], strict=True)
    base.selftest()
    print("strong ResNet-50 wrapper selftest PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--dataset", type=Path, default=base.DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ack")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.formal:
        run_formal(args.dataset, args.output, resume=args.resume, ack=args.ack)
    else:
        parser.error("choose --selftest or --formal")


if __name__ == "__main__":
    main()
