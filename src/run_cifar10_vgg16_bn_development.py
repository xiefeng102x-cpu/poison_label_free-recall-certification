#!/usr/bin/env python3
"""Nonformal victim-only development for the prospective CIFAR/VGG blind block.

This script is deliberately unable to compute detector or certificate outcomes.
Its only purpose is to freeze a training horizon that produces a viable victim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torchvision.datasets import CIFAR10

from cifar10_vgg16_bn import CIFAR10VGG16BN, parameter_count


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cifar10_vgg16_bn_development_v1.json"
DEFAULT_OUTPUT = ROOT / "results/cifar10_vgg16_bn_development_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def resolve_dataset(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    spec = config["dataset"]
    root = (ROOT / spec["root_relative_to_paper14"]).resolve()
    archive = root / spec["archive"]
    if hashlib.md5(archive.read_bytes()).hexdigest() != spec["archive_md5"] or sha256(archive) != spec["archive_sha256"]:
        raise RuntimeError("CIFAR-10 archive hash mismatch")
    training = CIFAR10(root=str(root), train=True, download=False)
    testing = CIFAR10(root=str(root), train=False, download=False)
    train_images = np.asarray(training.data, dtype=np.uint8)
    train_labels = np.asarray(training.targets, dtype=np.int64)
    test_images = np.asarray(testing.data, dtype=np.uint8)
    test_labels = np.asarray(testing.targets, dtype=np.int64)
    if train_images.shape != (50_000, 32, 32, 3) or test_images.shape != (10_000, 32, 32, 3):
        raise RuntimeError("unexpected CIFAR-10 shape")
    return train_images, train_labels, test_images, test_labels, {
        "root": str(root), "archive_sha256": sha256(archive), "train_rows": 50_000, "test_rows": 10_000,
    }


def fixed_split(labels: np.ndarray, *, rows_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    validation: list[np.ndarray] = []
    training: list[np.ndarray] = []
    for label in range(10):
        order = rng.permutation(np.flatnonzero(labels == label))
        validation.append(order[:rows_per_class])
        training.append(order[rows_per_class:])
    return np.sort(np.concatenate(training)).astype(np.int64), np.sort(np.concatenate(validation)).astype(np.int64)


def poison_plan(labels: np.ndarray, train_indices: np.ndarray, *, target: int, count: int, seed: int) -> np.ndarray:
    eligible = train_indices[labels[train_indices] != target]
    selected = np.random.default_rng(4_242_100 + seed).choice(eligible, size=count, replace=False)
    return np.sort(selected).astype(np.int64)


def patch(values: torch.Tensor, size: int = 4) -> torch.Tensor:
    result = values.clone()
    result[:, :, -size:, -size:] = 1.0
    return result


def augment(values: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    padded = F.pad(values, (4, 4, 4, 4), mode="reflect")
    n = len(values)
    top = torch.randint(0, 9, (n,), generator=generator)
    left = torch.randint(0, 9, (n,), generator=generator)
    # Per-image crops keep the canonical CIFAR augmentation while avoiding a
    # Python/PIL transform pipeline in the training hot path.
    cropped = torch.stack([padded[i, :, int(top[i]):int(top[i]) + 32, int(left[i]):int(left[i]) + 32] for i in range(n)])
    flip = torch.rand(n, generator=generator) < 0.5
    cropped[flip] = torch.flip(cropped[flip], dims=(3,))
    return cropped


def uint8_batch(images: np.ndarray, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    values = torch.from_numpy(np.ascontiguousarray(images[indices])).permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
    return values.to(device)


def evaluate(model: CIFAR10VGG16BN, images: np.ndarray, labels: np.ndarray, device: torch.device, target: int, batch_size: int) -> dict[str, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            values = uint8_batch(images, np.arange(start, min(start + batch_size, len(images))), device)
            predictions.append(model(values).argmax(1).cpu().numpy())
    predicted = np.concatenate(predictions)
    accuracy = float(np.mean(predicted == labels))
    bca = float(balanced_accuracy_score(labels, predicted))
    non_target = np.flatnonzero(labels != target)
    triggered: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(non_target), batch_size):
            indices = non_target[start:start + batch_size]
            values = patch(uint8_batch(images, indices, device))
            triggered.append(model(values).argmax(1).cpu().numpy())
    asr = float(np.mean(np.concatenate(triggered) == target))
    return {"clean_accuracy": accuracy, "balanced_accuracy": bca, "attack_success_rate": asr}


def train(output: Path, device: torch.device) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    seed = int(config["development_seed"])
    victim = config["victim"]
    attack = config["attack"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    train_images, train_labels, test_images, test_labels, dataset_receipt = resolve_dataset(config)
    train_indices, validation_indices = fixed_split(
        train_labels,
        rows_per_class=int(config["dataset"]["validation_rows_per_class"]),
        seed=int(config["dataset"]["validation_split_seed"]),
    )
    poisons = poison_plan(
        train_labels, train_indices, target=int(attack["target_class"]),
        count=int(attack["poison_count"]), seed=seed,
    )
    is_poison = np.isin(train_indices, poisons, assume_unique=True)
    assigned = train_labels[train_indices].copy()
    assigned[is_poison] = int(attack["target_class"])
    model = CIFAR10VGG16BN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(victim["learning_rate"]), weight_decay=float(victim["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(victim["epochs"]), eta_min=0.0)
    batch_size = int(victim["batch_size"])
    evaluation_epochs = set(int(x) for x in victim["evaluation_epochs"])
    metrics: list[dict[str, Any]] = []
    started = time.time()
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(victim["epochs"]) + 1):
        model.train()
        order_rng = np.random.default_rng(4_242_200 + 1009 * seed + epoch)
        order = order_rng.permutation(len(train_indices))
        augmentation_rng = torch.Generator().manual_seed(4_242_300 + 1009 * seed + epoch)
        loss_sum = 0.0
        correct = 0
        seen = 0
        for start in range(0, len(order), batch_size):
            positions = order[start:start + batch_size]
            base_indices = train_indices[positions]
            values = augment(uint8_batch(train_images, base_indices, torch.device("cpu")), augmentation_rng)
            poison_mask = torch.from_numpy(is_poison[positions])
            if bool(poison_mask.any()):
                values[poison_mask] = patch(values[poison_mask])
            values = values.to(device)
            labels = torch.from_numpy(assigned[positions]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(positions)
            correct += int((logits.argmax(1) == labels).sum().detach())
            seen += len(positions)
        scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch, "train_loss": loss_sum / seen, "train_accuracy": correct / seen,
            "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.time() - started,
        }
        if epoch in evaluation_epochs:
            row.update(evaluate(model, test_images, test_labels, device, int(attack["target_class"]), batch_size))
            checkpoint = output / f"development_seed{seed}_epoch{epoch}.pt"
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "seed": seed, "metrics": row}, checkpoint)
            row["checkpoint"] = str(checkpoint)
            row["checkpoint_sha256"] = sha256(checkpoint)
        metrics.append(row)
        atomic_json(output / "metrics.partial.json", metrics)
        print(json.dumps(row, sort_keys=True), flush=True)
    gate = config["development_gate"]
    eligible = [
        row for row in metrics
        if "balanced_accuracy" in row
        and row["balanced_accuracy"] >= float(gate["balanced_accuracy_min"])
        and row["attack_success_rate"] >= float(gate["attack_success_rate_min"])
    ]
    selected = int(eligible[0]["epoch"]) if eligible else None
    report = {
        "status": "PASS_VICTIM_ONLY_DEVELOPMENT" if selected is not None else "FAIL_VICTIM_ONLY_DEVELOPMENT",
        "formal_evidence": False,
        "detector_outcomes_opened": False,
        "certificate_outcomes_opened": False,
        "selected_training_epochs_for_freeze": selected,
        "development_seed": seed,
        "parameter_count": parameter_count(),
        "device": str(device),
        "dataset": dataset_receipt,
        "train_indices_sha256": hashlib.sha256(train_indices.tobytes()).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(validation_indices.tobytes()).hexdigest(),
        "poison_indices_sha256": hashlib.sha256(poisons.tobytes()).hexdigest(),
        "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(Path(__file__)),
        "architecture_sha256": sha256(ROOT / "src/cifar10_vgg16_bn.py"),
        "metrics": metrics,
    }
    atomic_json(output / "development_report.json", report)
    print(json.dumps({key: report[key] for key in ("status", "selected_training_epochs_for_freeze", "parameter_count")}, sort_keys=True))
    return report


def selftest() -> None:
    model = CIFAR10VGG16BN()
    logits, features = model.forward_with_features(torch.rand(2, 3, 32, 32))
    if logits.shape != (2, 10) or features.shape != (2, 512) or parameter_count() < 14_000_000:
        raise RuntimeError("architecture selftest failed")
    labels = np.repeat(np.arange(10), 10)
    train, validation = fixed_split(labels, rows_per_class=2, seed=1)
    if len(train) != 80 or len(validation) != 20 or len(np.intersect1d(train, validation)):
        raise RuntimeError("split selftest failed")
    print("selftest PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--develop", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.develop:
        train(args.output, torch.device(args.device))
    else:
        parser.error("choose --selftest or --develop")


if __name__ == "__main__":
    main()
