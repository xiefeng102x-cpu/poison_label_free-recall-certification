#!/usr/bin/env python3
"""Train and audit native 43-class GTSRB patch-backdoor victims.

Every seed is committed independently (checkpoint, metrics, receipt and
per-sample scores) so an interrupted ten-seed run can resume safely.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import resource
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import GTSRB


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_tifs_native_cifar10 as common  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "configs/tifs_native_gtsrb_development_v1.json"
DEFAULT_OUTPUT = ROOT / "results/tifs_native_cifar_gtsrb_v1/gtsrb_development"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class RunLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")

    def __call__(self, message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class GTSRBTinyCNN(nn.Module):
    def __init__(self, classes: int = 43) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 48, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 96, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(96)
        self.conv4 = nn.Conv2d(96, 128, 3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.head = nn.Linear(64, classes)

    def forward_with_rep(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = F.relu(self.bn1(self.conv1(values)))
        values = F.relu(self.bn2(self.conv2(values)))
        values = F.relu(self.bn3(self.conv3(values)))
        values = F.relu(self.bn4(self.conv4(values)))
        values = F.adaptive_avg_pool2d(values, 1).flatten(1)
        values = F.relu(self.fc1(values))
        rep = torch.tanh(self.fc2(values))
        return self.head(rep), rep

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.forward_with_rep(values)[0]


@dataclass
class SeedData:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    poison_base_indices: np.ndarray
    poison_sources: np.ndarray
    train_images: np.ndarray
    train_labels: np.ndarray
    validation_images: np.ndarray
    validation_labels: np.ndarray
    candidate_images: np.ndarray
    candidate_base_indices: np.ndarray
    candidate_original_labels: np.ndarray
    candidate_truth: np.ndarray
    candidate_positions: np.ndarray
    candidate_record_ids: list[str]


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset(protocol: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    spec = protocol["dataset"]
    root = (ROOT / spec["root_relative_to_paper14"]).resolve()
    archive_rows = []
    for row in spec["archives"]:
        path = root / row["file"]
        actual_md5 = md5_file(path)
        actual_sha256 = common.sha256_file(path)
        if actual_md5 != row["md5"] or actual_sha256 != row["sha256"]:
            raise RuntimeError(f"GTSRB archive hash mismatch: {path}")
        archive_rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "md5": actual_md5,
                "sha256": actual_sha256,
            }
        )
    train = GTSRB(root, split="train", download=False)
    test = GTSRB(root, split="test", download=False)
    if len(train) != int(spec["train_rows"]) or len(test) != int(spec["test_rows"]):
        raise RuntimeError("GTSRB row-count mismatch")
    train_classes = {int(label) for _, label in train._samples}
    test_classes = {int(label) for _, label in test._samples}
    if train_classes != set(range(43)) or test_classes != set(range(43)):
        raise RuntimeError("GTSRB class coverage mismatch")
    return root, {
        "root": str(root),
        "train_rows": len(train),
        "test_rows": len(test),
        "class_count": 43,
        "archives": archive_rows,
    }


def build_resized_cache(root: Path, log: RunLog) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = root / "gtsrb_32x32_uint8_v1.npz"
    if not path.exists():
        log("Building deterministic 32x32 uint8 cache from official PPM/JPEG files")
        arrays: dict[str, np.ndarray] = {}
        for split in ("train", "test"):
            dataset = GTSRB(root, split=split, download=False)
            images = np.empty((len(dataset), 32, 32, 3), dtype=np.uint8)
            labels = np.empty(len(dataset), dtype=np.int64)
            for index, (image_path, label) in enumerate(dataset._samples):
                with Image.open(image_path) as image:
                    images[index] = np.asarray(
                        image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR),
                        dtype=np.uint8,
                    )
                labels[index] = int(label)
            arrays[f"{split}_images"] = images
            arrays[f"{split}_labels"] = labels
        np.savez_compressed(path, **arrays)
    loaded = np.load(path)
    arrays = {key: loaded[key] for key in loaded.files}
    expected = {
        "train_images": (26640, 32, 32, 3),
        "train_labels": (26640,),
        "test_images": (12630, 32, 32, 3),
        "test_labels": (12630,),
    }
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise RuntimeError(f"derived cache shape mismatch: {key}")
    return arrays, {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
        "resize": "PIL bilinear 32x32 RGB",
    }


def patch_uint8(images: np.ndarray, protocol: dict[str, Any]) -> np.ndarray:
    attack = protocol["attack"]
    result = np.asarray(images, dtype=np.uint8).copy()
    size = int(attack["patch_size"])
    alpha = float(attack["patch_alpha"])
    patch = np.asarray(attack["patch_rgb_uint8"], dtype=np.float32).reshape(1, 1, 1, 3)
    region = result[:, -size:, -size:, :].astype(np.float32)
    result[:, -size:, -size:, :] = np.rint((1.0 - alpha) * region + alpha * patch).astype(
        np.uint8
    )
    return result


def build_seed_data(
    images: np.ndarray, labels: np.ndarray, protocol: dict[str, Any], seed: int
) -> SeedData:
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    fraction = float(protocol["split"]["train_fraction_per_class"])
    for label in range(43):
        indices = np.flatnonzero(labels == label)
        order = np.random.default_rng(2026100600 + seed + label * 1009).permutation(indices)
        count = int(math_floor(fraction * len(order)))
        train_parts.append(order[:count])
        validation_parts.append(order[count:])
    train_indices = np.concatenate(train_parts)
    validation_indices = np.concatenate(validation_parts)
    target = int(protocol["attack"]["target_class"])
    poison_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    poison_fraction = float(protocol["attack"]["poison_fraction_per_non_target_training_class"])
    for label in range(43):
        if label == target:
            continue
        candidates = train_indices[labels[train_indices] == label]
        count = max(1, int(round(poison_fraction * len(candidates))))
        order = np.random.default_rng(2026101600 + seed + label * 131).permutation(candidates)
        poison_parts.append(order[:count])
        source_parts.append(np.full(count, label, dtype=np.int64))
    poison_base = np.concatenate(poison_parts)
    poison_sources = np.concatenate(source_parts)
    poison_images = patch_uint8(images[poison_base], protocol)
    clean_train_images = images[train_indices]
    clean_train_labels = labels[train_indices]
    training_images = np.concatenate([clean_train_images, poison_images])
    training_labels = np.concatenate(
        [clean_train_labels, np.full(len(poison_images), target, dtype=np.int64)]
    )
    target_clean = train_indices[labels[train_indices] == target]
    candidate_images = np.concatenate([images[target_clean], poison_images])
    candidate_base = np.concatenate([target_clean, poison_base])
    candidate_original = np.concatenate(
        [np.full(len(target_clean), target, dtype=np.int64), poison_sources]
    )
    truth = np.concatenate(
        [np.zeros(len(target_clean), dtype=bool), np.ones(len(poison_base), dtype=bool)]
    )
    candidate_positions = np.concatenate(
        [
            np.flatnonzero(clean_train_labels == target),
            np.arange(len(clean_train_labels), len(training_labels)),
        ]
    )
    record_ids = [
        *[f"train_clean_{int(index)}" for index in target_clean],
        *[
            f"train_poison_source{int(source)}_base{int(index)}"
            for source, index in zip(poison_sources, poison_base)
        ],
    ]
    return SeedData(
        train_indices=train_indices,
        validation_indices=validation_indices,
        poison_base_indices=poison_base,
        poison_sources=poison_sources,
        train_images=training_images,
        train_labels=training_labels,
        validation_images=images[validation_indices],
        validation_labels=labels[validation_indices],
        candidate_images=candidate_images,
        candidate_base_indices=candidate_base,
        candidate_original_labels=candidate_original,
        candidate_truth=truth,
        candidate_positions=candidate_positions,
        candidate_record_ids=record_ids,
    )


def validate_seed_data(data: SeedData, labels: np.ndarray, protocol: dict[str, Any]) -> None:
    """Fail closed if one selected source record is not one poison record."""
    target = int(protocol["attack"]["target_class"])
    expected_poison = 0
    fraction = float(protocol["attack"]["poison_fraction_per_non_target_training_class"])
    for label in range(43):
        if label == target:
            continue
        source_count = int(np.sum(labels[data.train_indices] == label))
        expected_poison += max(1, int(round(fraction * source_count)))
    aligned_lengths = {
        len(data.candidate_images),
        len(data.candidate_base_indices),
        len(data.candidate_original_labels),
        len(data.candidate_truth),
        len(data.candidate_positions),
        len(data.candidate_record_ids),
    }
    if len(aligned_lengths) != 1:
        raise RuntimeError(f"candidate metadata length mismatch: {sorted(aligned_lengths)}")
    if len(data.poison_base_indices) != expected_poison:
        raise RuntimeError(
            f"poison multiplicity mismatch: actual={len(data.poison_base_indices)} "
            f"expected={expected_poison}"
        )
    if len(np.unique(data.poison_base_indices)) != len(data.poison_base_indices):
        raise RuntimeError("a selected source record contributed more than one poison record")
    if len(data.poison_sources) != len(data.poison_base_indices):
        raise RuntimeError("poison source metadata length mismatch")


def math_floor(value: float) -> int:
    return int(np.floor(value))


def to_tensor(images: np.ndarray) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(images))
        .float()
        .permute(0, 3, 1, 2)
        .contiguous()
        / 255.0
    )


def predict_and_features(
    model: GTSRBTinyCNN,
    images: np.ndarray,
    batch_size: int,
    device: torch.device,
    need_features: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    predictions: list[np.ndarray] = []
    features: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            values = to_tensor(images[start : start + batch_size]).to(device)
            logits, rep = model.forward_with_rep(values)
            predictions.append(logits.argmax(1).cpu().numpy())
            if need_features:
                features.append(rep.cpu().numpy().astype(np.float32))
    return np.concatenate(predictions), np.concatenate(features) if need_features else None


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, list[float]]:
    recalls = [float(np.mean(predictions[labels == label] == label)) for label in range(43)]
    return float(np.mean(recalls)), recalls


def train_model(
    data: SeedData,
    protocol: dict[str, Any],
    seed: int,
    threads: int,
    device: torch.device,
    log: RunLog,
) -> tuple[GTSRBTinyCNN, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(threads)
    model = GTSRBTinyCNN().to(device)
    config = protocol["victim"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    clean_counts = np.bincount(
        data.train_labels[: len(data.train_indices)], minlength=43
    ).astype(np.float64)
    weights = 1.0 / np.sqrt(np.maximum(clean_counts, 1.0))
    weights /= weights.mean()
    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)
    epochs = int(config["epochs"])
    batch_size = int(config["batch_size"])
    best_bca = -1.0
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.default_rng(seed * 100000 + epoch).permutation(len(data.train_images))
        loss_total = 0.0
        count_total = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            values = to_tensor(data.train_images[indices]).to(device)
            targets = torch.from_numpy(data.train_labels[indices]).long().to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(values), targets, weight=class_weights)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(indices)
            count_total += len(indices)
        validation_pred, _ = predict_and_features(
            model, data.validation_images, batch_size, device, need_features=False
        )
        validation_bca, _ = balanced_accuracy(data.validation_labels, validation_pred)
        row = {
            "epoch": epoch,
            "train_loss": loss_total / count_total,
            "validation_balanced_accuracy": validation_bca,
        }
        history.append(row)
        log(
            f"seed={seed} epoch={epoch}/{epochs} loss={row['train_loss']:.6f} "
            f"val_BCA={validation_bca:.6f}"
        )
        if validation_bca > best_bca:
            best_bca = validation_bca
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "best_validation_balanced_accuracy": best_bca,
        "history": history,
        "state_sha256": common.state_hash(best_state),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def victim_metrics(
    model: GTSRBTinyCNN,
    cache: dict[str, np.ndarray],
    protocol: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    labels = cache["test_labels"]
    predicted, _ = predict_and_features(model, cache["test_images"], batch_size, device)
    bca, recalls = balanced_accuracy(labels, predicted)
    target = int(protocol["attack"]["target_class"])
    non_target = labels != target
    attack_images = patch_uint8(cache["test_images"][non_target], protocol)
    attack_predicted, _ = predict_and_features(model, attack_images, batch_size, device)
    asr = float(np.mean(attack_predicted == target))
    if protocol["stage"] == "development_smoke":
        gate = protocol["smoke_gate"]
    else:
        gate = protocol["victim_eligibility"]
    bca_min = float(gate.get("test_balanced_clean_accuracy_min", gate.get("clean_balanced_accuracy_min")))
    asr_min = float(gate["attack_success_rate_min"])
    return {
        "clean_accuracy": float(np.mean(predicted == labels)),
        "balanced_clean_accuracy": bca,
        "attack_success_rate": asr,
        "eligible": int(bca >= bca_min and asr >= asr_min),
        "per_class_recall_json": json.dumps(recalls, separators=(",", ":")),
        "clean_test_count": len(labels),
        "asr_test_count": int(non_target.sum()),
    }


def trusted_features(
    model: GTSRBTinyCNN,
    cache: dict[str, np.ndarray],
    protocol: dict[str, Any],
    seed: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    target = int(protocol["attack"]["target_class"])
    indices = np.flatnonzero(cache["test_labels"] == target)
    order = np.random.default_rng(2026102600 + seed).permutation(indices)
    if len(order) != 750:
        raise RuntimeError("target trusted pool count mismatch")
    _, features = predict_and_features(model, cache["test_images"][order], batch_size, device, True)
    assert features is not None
    return features[:250], features[250:500], features[500:], {
        "reference_indices": order[:250].tolist(),
        "calibration_indices": order[250:500].tolist(),
        "reserved_indices": order[500:].tolist(),
        "reference_indices_sha256": hashlib.sha256(order[:250].astype(np.int64).tobytes()).hexdigest(),
        "calibration_indices_sha256": hashlib.sha256(order[250:500].astype(np.int64).tobytes()).hexdigest(),
    }


def detector_sample_rows(
    seed: int,
    method: str,
    data: SeedData,
    result: dict[str, Any],
    target: int,
) -> list[dict[str, Any]]:
    return [
        {
            "seed": seed,
            "dataset": "GTSRB",
            "method": method,
            "candidate_position": int(data.candidate_positions[index]),
            "record_id": data.candidate_record_ids[index],
            "base_train_index": int(data.candidate_base_indices[index]),
            "original_label": int(data.candidate_original_labels[index]),
            "assigned_label": target,
            "poison_truth": int(data.candidate_truth[index]),
            "defined": 1,
            "score": float(result["scores"][index]),
            "p_value": float(result["pvalues"][index]),
            "flag": int(result["flags"][index]),
        }
        for index in range(len(data.candidate_truth))
    ]


def synthetic_self_test() -> None:
    model = GTSRBTinyCNN()
    logits, features = model.forward_with_rep(torch.zeros(3, 3, 32, 32))
    if logits.shape != (3, 43) or features.shape != (3, 64):
        raise RuntimeError("GTSRB model shape self-test failed")
    rng = np.random.default_rng(1)
    reference = rng.normal(size=(250, 64))
    calibration = rng.normal(size=(250, 64))
    candidates = rng.normal(size=(400, 64))
    candidates[-40:, 0] += 4.0
    for builder in (
        lambda: common.cf_cmd_u(reference, calibration, candidates, 1, 0.05),
        lambda: common.centroid_distance(reference, calibration, candidates, 0.05),
        lambda: common.crossfit_spectral(reference, calibration, candidates, 1, 0.05),
    ):
        result = builder()
        if not np.isfinite(result["scores"]).all():
            raise RuntimeError("detector self-test failed")
    print("GTSRB_STATIC_SELF_TEST_PASS")


def resolve_device(requested: str, protocol: dict[str, Any] | None = None) -> torch.device:
    execution = (protocol or {}).get("execution", {})
    formal = execution.get("formal_device")
    if formal is not None and formal not in {"cpu", "mps"}:
        raise RuntimeError(f"unsupported formal_device in protocol: {formal}")
    if formal is not None and requested not in {"auto", formal}:
        raise RuntimeError(
            f"requested device {requested!r} conflicts with frozen formal_device {formal!r}"
        )
    effective = formal if requested == "auto" and formal is not None else requested
    if effective == "auto":
        effective = "mps" if torch.backends.mps.is_available() else "cpu"
    if effective == "mps" and not torch.backends.mps.is_available():
        fallback = execution.get("fallback", "fail")
        if fallback == "fail" or formal == "mps":
            raise RuntimeError("MPS was required but torch.backends.mps.is_available() is false")
        raise RuntimeError(f"unsupported device fallback policy: {fallback}")
    return torch.device(effective)


def synchronize_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def checkpoint_inference_smoke(
    checkpoint: Path, device: torch.device, receipt_path: Path | None
) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    model = GTSRBTinyCNN().to(device)
    model.load_state_dict(state)
    model.eval()
    values = torch.zeros(4, 3, 32, 32, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits, features = model.forward_with_rep(values)
    synchronize_device(device)
    passed = logits.shape == (4, 43) and features.shape == (4, 64)
    passed = bool(passed and torch.isfinite(logits).all().item() and torch.isfinite(features).all().item())
    if not passed:
        raise RuntimeError("checkpoint inference smoke failed")
    receipt = {
        "checked_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": common.sha256_file(checkpoint),
        "checkpoint_seed": int(payload["seed"]),
        "state_sha256": common.state_hash(state),
        "requested_device": str(device),
        "actual_device": logits.device.type,
        "logits_shape": list(logits.shape),
        "features_shape": list(features.shape),
        "finite": True,
        "status": "pass",
    }
    if receipt_path is not None:
        write_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True))


def mps_train_self_test(receipt_path: Path | None) -> None:
    device = resolve_device("mps")
    torch.manual_seed(20261006)
    model = GTSRBTinyCNN().to(device)
    images = np.random.default_rng(20261006).integers(
        0, 256, size=(4, 32, 32, 3), dtype=np.uint8
    )
    values = to_tensor(images).to(device)
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
    class_weights = torch.ones(43, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    optimizer.zero_grad()
    loss = F.cross_entropy(model(values), targets, weight=class_weights)
    loss.backward()
    optimizer.step()
    synchronize_device(device)
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )
    passed = bool(values.is_contiguous() and torch.isfinite(loss).item() and gradients_finite)
    if not passed:
        raise RuntimeError("MPS forward/backward training self-test failed")
    receipt = {
        "checked_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": common.sha256_file(Path(__file__).resolve()),
        "actual_device": device.type,
        "input_is_contiguous": bool(values.is_contiguous()),
        "input_shape": list(values.shape),
        "targets_device": targets.device.type,
        "class_weights_device": class_weights.device.type,
        "loss_finite": bool(torch.isfinite(loss).item()),
        "gradients_finite": gradients_finite,
        "optimizer_step_completed": True,
        "status": "pass",
    }
    if receipt_path is not None:
        write_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True))


def aggregate_summary(
    protocol: dict[str, Any], victims: list[dict[str, Any]], detectors: list[dict[str, Any]]
) -> dict[str, Any]:
    eligible = {int(row["seed"]) for row in victims if int(row["eligible"]) == 1}
    methods: dict[str, Any] = {}
    for method in ("cf_cmd_u", "target_centroid_distance", "crossfit_spectral"):
        rows = [row for row in detectors if row["method"] == method and int(row["seed"]) in eligible]
        tp = sum(int(row["tp"]) for row in rows)
        fn = sum(int(row["fn"]) for row in rows)
        fp = sum(int(row["fp"]) for row in rows)
        tn = sum(int(row["tn_target"]) for row in rows)
        methods[method] = {
            "eligible_seed_count": len(rows),
            "pooled_tpr": tp / (tp + fn) if tp + fn else None,
            "pooled_target_clean_fpr": fp / (fp + tn) if fp + tn else None,
            "mean_seed_auc": float(np.mean([float(row["auc_target_assigned"]) for row in rows]))
            if rows
            else None,
        }
    return {
        "stage": protocol["stage"],
        "eligible_seed_count": len(eligible),
        "eligible_seeds": sorted(eligible),
        "methods": methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="auto")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--checkpoint-inference-smoke", type=Path)
    parser.add_argument("--mps-train-self-test", action="store_true")
    parser.add_argument("--smoke-receipt", type=Path)
    parser.add_argument(
        "--victim-only",
        action="store_true",
        help="train/evaluate/archive victims without computing real-sample detector outcomes",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        synthetic_self_test()
        return
    if args.checkpoint_inference_smoke is not None:
        checkpoint_inference_smoke(
            args.checkpoint_inference_smoke.resolve(),
            resolve_device(args.device),
            args.smoke_receipt.resolve() if args.smoke_receipt is not None else None,
        )
        return
    if args.mps_train_self_test:
        mps_train_self_test(
            args.smoke_receipt.resolve() if args.smoke_receipt is not None else None
        )
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = RunLog(output / "run.log")
    started = time.time()
    try:
        protocol = read_json(args.protocol)
        if protocol.get("protocol_id") not in {
            "tifs_native_gtsrb_development_v1",
            "tifs_native_gtsrb_confirmation_v1",
            "tifs_native_gtsrb_confirmation_v1_1",
        }:
            raise RuntimeError("unexpected GTSRB protocol")
        device = resolve_device(args.device, protocol)
        log(f"device requested={args.device} actual={device.type}")
        data_root, dataset_receipt = validate_dataset(protocol)
        cache, cache_receipt = build_resized_cache(data_root, log)
        shutil.copy2(args.protocol, output / "frozen_protocol.json")
        victim_path = output / "victim_metrics.csv"
        detector_path = output / "detector_metrics.csv"
        victims: list[dict[str, Any]] = list(read_csv(victim_path))
        detectors: list[dict[str, Any]] = list(read_csv(detector_path))
        completed = {int(row["seed"]) for row in victims}
        for seed_value in protocol["seeds"]:
            seed = int(seed_value)
            if seed in completed and not args.force:
                log(f"seed={seed} skip existing complete receipt")
                continue
            if args.force:
                victims = [row for row in victims if int(row["seed"]) != seed]
                detectors = [row for row in detectors if int(row["seed"]) != seed]
            seed_started = time.time()
            log(f"seed={seed} start stage={protocol['stage']}")
            seed_data = build_seed_data(cache["train_images"], cache["train_labels"], protocol, seed)
            validate_seed_data(seed_data, cache["train_labels"], protocol)
            model, training = train_model(seed_data, protocol, seed, args.threads, device, log)
            metrics = victim_metrics(model, cache, protocol, args.eval_batch_size, device)
            synchronize_device(device)
            checkpoint_payload = {
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": common.sha256_file(args.protocol),
                "seed": seed,
                "model": "GTSRBTinyCNN-43",
                "state_sha256": training["state_sha256"],
                "training": training,
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            }
            checkpoint = output / "models" / f"seed_{seed}_suspect.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_payload, checkpoint)
            checkpoint_sha256 = common.sha256_file(checkpoint)
            victim_row = {
                "seed": seed,
                **metrics,
                "best_epoch": training["best_epoch"],
                "best_validation_balanced_accuracy": training[
                    "best_validation_balanced_accuracy"
                ],
                "parameter_count": training["parameter_count"],
                "clean_training_count": len(seed_data.train_indices),
                "poison_count": len(seed_data.poison_base_indices),
                "candidate_count": len(seed_data.train_images),
                "defined_target_candidate_count": len(seed_data.candidate_truth),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "state_sha256": training["state_sha256"],
                "actual_device": device.type,
                "elapsed_seconds": time.time() - seed_started,
            }
            if args.victim_only:
                victims.append(victim_row)
                receipt = {
                    "seed": seed,
                    "victim_only": True,
                    "protocol_sha256": common.sha256_file(args.protocol),
                    "checkpoint_sha256": checkpoint_sha256,
                    "state_sha256": training["state_sha256"],
                    "actual_device": device.type,
                    "candidate_base_indices_sha256": hashlib.sha256(
                        seed_data.candidate_base_indices.astype(np.int64).tobytes()
                    ).hexdigest(),
                    "candidate_truth_sha256": hashlib.sha256(
                        seed_data.candidate_truth.astype(np.uint8).tobytes()
                    ).hexdigest(),
                    "training_history": training["history"],
                }
                write_json(output / "receipts" / f"seed_{seed}.json", receipt)
                write_csv(victim_path, victims)
                write_csv(detector_path, detectors)
                write_json(output / "summary.json", aggregate_summary(protocol, victims, detectors))
                completed.add(seed)
                log(
                    f"seed={seed} victim-only committed CA={metrics['clean_accuracy']:.6f} "
                    f"BCA={metrics['balanced_clean_accuracy']:.6f} ASR={metrics['attack_success_rate']:.6f} "
                    f"eligible={metrics['eligible']} elapsed={time.time() - seed_started:.1f}s"
                )
                continue
            _, candidate_features = predict_and_features(
                model, seed_data.candidate_images, args.eval_batch_size, device, True
            )
            assert candidate_features is not None
            reference, calibration, _, trusted_receipt = trusted_features(
                model, cache, protocol, seed, args.eval_batch_size, device
            )
            alpha = float(protocol["detectors"]["primary"]["alpha"])
            method_results = {
                "cf_cmd_u": common.cf_cmd_u(reference, calibration, candidate_features, seed, alpha),
                "target_centroid_distance": common.centroid_distance(
                    reference, calibration, candidate_features, alpha
                ),
                "crossfit_spectral": common.crossfit_spectral(
                    reference, calibration, candidate_features, seed, alpha
                ),
            }
            seed_detector_rows: list[dict[str, Any]] = []
            sample_rows: list[dict[str, Any]] = []
            detector_receipts: dict[str, Any] = {}
            for method, result in method_results.items():
                row = {
                    "seed": seed,
                    "method": method,
                    **common.detector_metrics(
                        result, seed_data.candidate_truth, len(seed_data.train_indices)
                    ),
                    "victim_eligible": metrics["eligible"],
                }
                seed_detector_rows.append(row)
                sample_rows.extend(
                    detector_sample_rows(
                        seed,
                        method,
                        seed_data,
                        result,
                        int(protocol["attack"]["target_class"]),
                    )
                )
                detector_receipts[method] = result["receipt"]
                log(
                    f"seed={seed} method={method} TPR={row['tpr']:.6f} "
                    f"target_FPR={row['target_clean_fpr']:.6f} AUC={row['auc_target_assigned']:.6f}"
                )
            victims.append(victim_row)
            detectors.extend(seed_detector_rows)
            per_sample_path = output / "per_sample" / f"seed_{seed}_scores.csv"
            write_csv(per_sample_path, sample_rows)
            receipt = {
                "seed": seed,
                "protocol_sha256": common.sha256_file(args.protocol),
                "checkpoint_sha256": checkpoint_sha256,
                "state_sha256": training["state_sha256"],
                "actual_device": device.type,
                "candidate_base_indices_sha256": hashlib.sha256(
                    seed_data.candidate_base_indices.astype(np.int64).tobytes()
                ).hexdigest(),
                "candidate_truth_sha256": hashlib.sha256(
                    seed_data.candidate_truth.astype(np.uint8).tobytes()
                ).hexdigest(),
                "trusted": trusted_receipt,
                "detectors": detector_receipts,
                "per_sample_path": str(per_sample_path),
                "per_sample_sha256": common.sha256_file(per_sample_path),
                "training_history": training["history"],
            }
            write_json(output / "receipts" / f"seed_{seed}.json", receipt)
            write_csv(victim_path, victims)
            write_csv(detector_path, detectors)
            summary = aggregate_summary(protocol, victims, detectors)
            write_json(output / "summary.json", summary)
            completed.add(seed)
            log(
                f"seed={seed} committed CA={metrics['clean_accuracy']:.6f} "
                f"BCA={metrics['balanced_clean_accuracy']:.6f} ASR={metrics['attack_success_rate']:.6f} "
                f"eligible={metrics['eligible']} elapsed={time.time() - seed_started:.1f}s"
            )
        summary = aggregate_summary(protocol, victims, detectors)
        manifest = {
            "experiment": protocol["protocol_id"],
            "status": "complete",
            "stage": protocol["stage"],
            "protocol_path": str(args.protocol.resolve()),
            "protocol_sha256": common.sha256_file(args.protocol),
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": common.sha256_file(Path(__file__).resolve()),
            "dataset": dataset_receipt,
            "derived_cache": cache_receipt,
            "completed_seeds": sorted(completed),
            "elapsed_seconds_this_invocation": time.time() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "requested_device": args.device,
            "actual_device_by_seed": {
                str(row["seed"]): row.get("actual_device", "unrecorded") for row in victims
            },
            "victim_only": bool(args.victim_only),
            "summary": summary,
        }
        write_json(output / "manifest.json", manifest)
        log(f"stage complete seeds={sorted(completed)}")
    finally:
        log.close()


if __name__ == "__main__":
    main()
