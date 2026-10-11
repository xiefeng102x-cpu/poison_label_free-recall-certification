#!/usr/bin/env python3
"""Frozen certificate-aware multi-score adaptive PathMNIST blind block."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import pathmnist_official_resnet18 as architecture
import run_pathmnist_blind_e2e_multidetector as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pathmnist_blind_adaptive_multiscore_certificate_v1.json"
PREREG = ROOT / "auditability_certificate_v1/PREREG_BLIND_ADAPTIVE_MULTISCORE_V1.md"
FREEZE = ROOT / "auditability_certificate_v1/BLIND_ADAPTIVE_MULTISCORE_FREEZE_V1.json"
CERT_RUNNER = ROOT / "auditability_certificate_v1/analysis/run_pathmnist_blind_extension_certificate.py"
VERIFIER = ROOT / "src/verify_pathmnist_blind_extension.py"
WEIGHT_RECEIPT = ROOT / "results/top_journal_program_v1/pathmnist_official_weights_acquisition_receipt.json"
DEFAULT_OUTPUT = ROOT / "results/pathmnist_blind_adaptive_multiscore_certificate_v1"
SEEDS = (3201, 3202, 3203, 3204, 3205)
WEIGHT_MAP = {
    3201: "resnet18_28_1.pth",
    3202: "resnet18_28_2.pth",
    3203: "resnet18_28_1.pth",
    3204: "resnet18_28_2.pth",
    3205: "resnet18_28_1.pth",
}
ACK = "OPEN_BLIND_ADAPTIVE_MULTISCORE_SEEDS_3201_3205"
ALIGNMENT_WEIGHT = 1.0
LOG_NORM_WEIGHT = 0.1
GRADIENT_CLIP = 5.0
ATTACK_REFERENCE_COUNT = 512

_CURRENT_POISON_INDICES: np.ndarray | None = None
_BASE_POOL = base.engine.build_suspect_candidate_pool
_BASE_WRITE_SEED_SCORES = base.write_seed_scores


def _adaptive_pool(*args: Any, **kwargs: Any) -> dict[str, Any]:
    global _CURRENT_POISON_INDICES
    result = _BASE_POOL(*args, **kwargs)
    _CURRENT_POISON_INDICES = np.asarray(result["selected_indices"], dtype=np.int64).copy()
    return result


def _contract(
    *,
    seed: int,
    initial: dict[str, torch.Tensor],
    order_seed: int,
    device: torch.device,
    dataset_sha256: str,
    weight_sha256: str,
    training_data_sha256: str,
    poison_indices: np.ndarray,
    target_reference_indices: np.ndarray,
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "condition": "certificate_aware_multiscore_alignment",
        "data_order_seed": int(order_seed),
        "initial_state_sha256": base.core.state_fingerprint(initial),
        "epochs": int(base.engine.EPOCHS),
        "device": str(device),
        "protocol_sha256": base.sha256(CONFIG),
        "dataset_sha256": dataset_sha256,
        "weight_sha256": weight_sha256,
        "training_data_sha256": training_data_sha256,
        "runner_sha256": base.sha256(Path(__file__)),
        "base_engine_sha256": base.sha256(Path(base.engine.__file__)),
        "core_sha256": base.sha256(Path(base.core.__file__)),
        "architecture_sha256": base.sha256(Path(architecture.__file__)),
        "poison_indices_sha256": base.array_hash(poison_indices),
        "target_reference_indices_sha256": base.array_hash(target_reference_indices),
        "alignment_weight": ALIGNMENT_WEIGHT,
        "log_norm_weight": LOG_NORM_WEIGHT,
        "gradient_clip": GRADIENT_CLIP,
        "attacker_secret_member_access": False,
    }


def _reference_features(
    model: nn.Module,
    reference_images: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(reference_images), 256):
            values = base.engine._tensor_batch(reference_images[start : start + 256], device)
            pieces.append(model.hidden(values).detach())
    model.train()
    result = torch.cat(pieces, dim=0)
    if result.shape != (ATTACK_REFERENCE_COUNT, architecture.HIDDEN_DIMENSION):
        raise RuntimeError("adaptive reference feature shape mismatch")
    return result


def _adaptive_train_member(
    images: np.ndarray,
    labels: np.ndarray,
    initial: dict[str, torch.Tensor],
    *,
    seed: int,
    condition: str,
    device: torch.device,
    order_seed: int,
    output: Path,
    checkpoint_path: Path,
    resume: bool,
    dataset_sha256: str,
    weight_sha256: str,
    training_data_sha256: str,
):
    del condition, output
    if _CURRENT_POISON_INDICES is None:
        raise RuntimeError("adaptive poison membership was not established")
    poison_indices = np.asarray(_CURRENT_POISON_INDICES, dtype=np.int64)
    if len(poison_indices) != 4500 or np.any(labels[poison_indices] != base.TARGET):
        raise RuntimeError("adaptive poison membership does not match the frozen attack")
    poison_mask = np.zeros(len(images), dtype=bool)
    poison_mask[poison_indices] = True
    clean_target = np.flatnonzero((np.asarray(labels) == base.TARGET) & ~poison_mask)
    if len(clean_target) < ATTACK_REFERENCE_COUNT:
        raise RuntimeError("insufficient clean target training members for adaptive surrogate")
    target_reference_indices = np.sort(
        np.random.default_rng(32305000 + int(seed)).choice(
            clean_target, size=ATTACK_REFERENCE_COUNT, replace=False
        )
    ).astype(np.int64)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device) == "mps":
        torch.mps.manual_seed(seed)
    model = architecture.PathMNISTResNet18()
    model.load_state_dict(initial, strict=True)
    if not all(torch.equal(model.state_dict()[key].cpu(), value.cpu()) for key, value in initial.items()):
        raise RuntimeError("CPU initializer differs from the official checkpoint")
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=base.engine.LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    generator = torch.Generator(device="cpu").manual_seed(order_seed)
    dataset = torch.utils.data.TensorDataset(
        base.core._tensor_images(images),
        torch.as_tensor(labels, dtype=torch.long),
        torch.as_tensor(poison_mask, dtype=torch.bool),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=base.engine.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    contract = _contract(
        seed=seed,
        initial=initial,
        order_seed=order_seed,
        device=device,
        dataset_sha256=dataset_sha256,
        weight_sha256=weight_sha256,
        training_data_sha256=training_data_sha256,
        poison_indices=poison_indices,
        target_reference_indices=target_reference_indices,
    )
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if checkpoint_path.exists():
        if not resume:
            raise RuntimeError(f"checkpoint exists; pass --resume: {checkpoint_path}")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("contract") != contract:
            raise RuntimeError("adaptive resume checkpoint contract mismatch")
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["sampler_generator"])
        random.setstate(saved["rng_state"]["python"])
        np.random.set_state(saved["rng_state"]["numpy"])
        torch.set_rng_state(saved["rng_state"]["torch_cpu"])
        if str(device) == "mps" and saved["rng_state"].get("torch_mps") is not None:
            torch.mps.set_rng_state(saved["rng_state"]["torch_mps"])
        history = list(saved["history"])
        start_epoch = int(saved["epoch"]) + 1
        if not 1 <= start_epoch <= base.engine.EPOCHS + 1 or len(history) != start_epoch - 1:
            raise RuntimeError("adaptive resume epoch/history mismatch")
    elif resume:
        print(f"adaptive seed={seed}: no checkpoint; starting at epoch 1", flush=True)

    reference_images = images[target_reference_indices]
    for epoch in range(start_epoch, base.engine.EPOCHS + 1):
        target_features = _reference_features(model, reference_images, device)
        pairing_rng = np.random.default_rng(32306000 + 1009 * int(seed) + epoch)
        model.train()
        total_sum = ce_sum = alignment_sum = 0.0
        seen = poison_seen = 0
        for xb, yb, poison_batch in loader:
            xb = xb.to(device=device, dtype=torch.float32).div_(255.0)
            yb = yb.to(device)
            poison_batch = poison_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            hidden = model.hidden(xb)
            logits = model.linear(hidden)
            ce = F.cross_entropy(logits, yb)
            count = int(poison_batch.sum().item())
            if count:
                prototype_indices = pairing_rng.integers(
                    0, ATTACK_REFERENCE_COUNT, size=count, endpoint=False
                )
                prototypes = target_features[
                    torch.as_tensor(prototype_indices, device=device, dtype=torch.long)
                ]
                poison_hidden = hidden[poison_batch]
                cosine = 1.0 - torch.sum(
                    F.normalize(poison_hidden, dim=1) * F.normalize(prototypes, dim=1), dim=1
                )
                log_norm = (
                    torch.log(torch.linalg.vector_norm(poison_hidden, dim=1) + 1e-6)
                    - torch.log(torch.linalg.vector_norm(prototypes, dim=1) + 1e-6)
                ).square()
                alignment = cosine.mean() + LOG_NORM_WEIGHT * log_norm.mean()
            else:
                alignment = hidden.sum() * 0.0
            loss = ce + ALIGNMENT_WEIGHT * alignment
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("non-finite adaptive gradient")
            optimizer.step()
            batch_size = len(xb)
            total_sum += float(loss.detach().cpu()) * batch_size
            ce_sum += float(ce.detach().cpu()) * batch_size
            alignment_sum += float(alignment.detach().cpu()) * count
            seen += batch_size
            poison_seen += count
        if seen != len(images) or poison_seen != len(poison_indices):
            raise RuntimeError("adaptive epoch did not process the frozen population exactly once")
        history.append(
            {
                "epoch": epoch,
                "loss": total_sum / seen,
                "cross_entropy": ce_sum / seen,
                "poison_alignment": alignment_sum / poison_seen,
                "rows": seen,
                "poison_rows": poison_seen,
            }
        )
        payload = {
            "contract": contract,
            "seed": seed,
            "condition": contract["condition"],
            "epoch": epoch,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "sampler_generator": generator.get_state(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_mps": torch.mps.get_rng_state() if str(device) == "mps" else None,
            },
            "data_order_seed": order_seed,
            "history": history,
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, checkpoint_path)
        print(
            f"adaptive seed={seed} epoch={epoch} loss={history[-1]['loss']:.4f} "
            f"align={history[-1]['poison_alignment']:.4f}",
            flush=True,
        )
    return model, {
        "condition": contract["condition"],
        "history": history,
        "checkpoint_sha256": base.sha256(checkpoint_path),
        "state_sha256": base.core.state_fingerprint(model.state_dict()),
        "checkpoint_path": str(checkpoint_path),
        "adaptive_contract": contract,
    }


def _write_seed_scores(*args: Any, **kwargs: Any):
    score_path, truth_path, receipts = _BASE_WRITE_SEED_SCORES(*args, **kwargs)
    old = "pathmnist_badnets_resnet18_seed"
    new = "pathmnist_adaptive_multiscore_resnet18_seed"
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
        raise RuntimeError("adaptive freeze status is invalid")
    if frozen.get("hashes") != expected:
        raise RuntimeError("adaptive frozen file hashes do not match")
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
    base.engine.build_suspect_candidate_pool = _adaptive_pool
    base.engine._train_member = _adaptive_train_member


def run_formal(dataset: Path, output: Path, *, resume: bool, ack: str | None) -> dict[str, Any]:
    configure_base()
    manifest = base.run_formal(dataset, output, resume=resume, ack=ack)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frozen_hashes"]["extension_runner_sha256"] = base.sha256(Path(__file__))
    manifest["frozen_hashes"]["base_orchestrator_sha256"] = base.sha256(Path(base.__file__))
    manifest["adaptive_threat_model"] = {
        "certificate_aware": True,
        "secret_trusted_member_identities_known": False,
        "private_truth_known": False,
    }
    base.atomic_json(manifest_path, manifest)
    return manifest


def selftest() -> None:
    configure_base()
    rng = np.random.default_rng(32010001)
    poison = torch.as_tensor(rng.normal(size=(8, 512)), dtype=torch.float32)
    target = torch.as_tensor(rng.normal(size=(8, 512)), dtype=torch.float32)
    cosine = 1.0 - torch.sum(F.normalize(poison, dim=1) * F.normalize(target, dim=1), dim=1)
    log_norm = (
        torch.log(torch.linalg.vector_norm(poison, dim=1) + 1e-6)
        - torch.log(torch.linalg.vector_norm(target, dim=1) + 1e-6)
    ).square()
    loss = cosine.mean() + LOG_NORM_WEIGHT * log_norm.mean()
    if not bool(torch.isfinite(loss)) or float(loss) <= 0:
        raise RuntimeError("adaptive surrogate selftest failed")
    base.selftest()
    print("adaptive multi-score wrapper selftest PASS")


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
