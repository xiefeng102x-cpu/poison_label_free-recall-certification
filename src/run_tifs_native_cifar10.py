#!/usr/bin/env python3
"""Checkpoint-only native CIFAR-10 poison-screening confirmation.

The frozen protocol is ``configs/tifs_native_cifar10_v1.json``.  The runner
reconstructs the unique poisoned training records used by the archived audited
victims, replays each checkpoint, and applies CF-CMD-U plus two matched
calibration baselines.  It never downloads or substitutes data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/tifs_native_cifar10_v1.json"
DEFAULT_OUTPUT = ROOT / "results/tifs_native_cifar_gtsrb_v1/cifar10"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


class RunLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8")

    def __call__(self, message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
            if stride != 1 or in_planes != planes
            else nn.Identity()
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        result = F.relu(self.bn1(self.conv1(values)))
        result = self.bn2(self.conv2(result))
        return F.relu(result + self.shortcut(values))


class CifarResNet18Multi(nn.Module):
    def __init__(
        self,
        width: int = 64,
        rep_hidden: int = 256,
        rep_penultimate: int = 128,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.in_planes = width
        self.conv1 = nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.layer1 = self._make_layer(width, 2, stride=1)
        self.layer2 = self._make_layer(width * 2, 2, stride=2)
        self.layer3 = self._make_layer(width * 4, 2, stride=2)
        self.layer4 = self._make_layer(width * 8, 2, stride=2)
        self.fc1 = nn.Linear(width * 8, rep_hidden)
        self.fc2 = nn.Linear(rep_hidden, rep_penultimate)
        self.head = nn.Linear(rep_penultimate, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for item_stride in [stride] + [1] * (blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, item_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward_with_reps(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z0 = F.relu(self.bn1(self.conv1(values)))
        z1 = self.layer1(z0)
        z2 = self.layer2(z1)
        z3 = self.layer3(z2)
        z4 = self.layer4(z3)
        pooled = F.adaptive_avg_pool2d(z4, 1).flatten(1)
        hidden1 = torch.tanh(self.fc1(pooled))
        penultimate = torch.tanh(self.fc2(hidden1))
        return self.head(penultimate), {
            "hidden1": hidden1,
            "penultimate": penultimate,
            "multi": torch.cat([hidden1, penultimate], dim=1),
        }

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.forward_with_reps(values)[0]


def resolve(root: Path, relative: str) -> Path:
    return (root / relative).resolve()


def validate_dataset(protocol: dict[str, Any]) -> tuple[CIFAR10, CIFAR10, dict[str, Any]]:
    spec = protocol["dataset"]
    data_root = resolve(ROOT, spec["root_relative_to_paper14"])
    archive = data_root / spec["archive"]
    if hashlib.md5(archive.read_bytes()).hexdigest() != spec["archive_md5"]:
        raise RuntimeError("CIFAR-10 archive MD5 mismatch")
    if sha256_file(archive) != spec["archive_sha256"]:
        raise RuntimeError("CIFAR-10 archive SHA256 mismatch")
    train = CIFAR10(root=str(data_root), train=True, download=False)
    test = CIFAR10(root=str(data_root), train=False, download=False)
    if len(train) != spec["expected_train_rows"] or len(test) != spec["expected_test_rows"]:
        raise RuntimeError("CIFAR-10 row-count mismatch")
    if list(train.data.shape[1:]) != spec["expected_shape"]:
        raise RuntimeError("CIFAR-10 image-shape mismatch")
    counts_train = np.bincount(np.asarray(train.targets), minlength=10)
    counts_test = np.bincount(np.asarray(test.targets), minlength=10)
    if not np.all(counts_train == 5000) or not np.all(counts_test == 1000):
        raise RuntimeError("CIFAR-10 class-count mismatch")
    return train, test, {
        "root": str(data_root),
        "archive": str(archive),
        "archive_md5": spec["archive_md5"],
        "archive_sha256": spec["archive_sha256"],
        "train_rows": len(train),
        "test_rows": len(test),
        "train_class_counts": counts_train.tolist(),
        "test_class_counts": counts_test.tolist(),
        "torchvision_integrity_check": "passed_by_constructor",
    }


def load_model(
    protocol: dict[str, Any], entry: dict[str, Any]
) -> tuple[CifarResNet18Multi, dict[str, Any], Path]:
    source_root = resolve(
        ROOT, protocol["archived_victim_source"]["root_relative_to_paper14"]
    )
    path = source_root / entry["file"]
    if sha256_file(path) != entry["file_sha256"]:
        raise RuntimeError(f"checkpoint file hash mismatch for seed {entry['seed']}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_model = {"name": "cifar_resnet18", "width": 64, "num_classes": 10}
    if payload.get("seed") != entry["seed"] or payload.get("model") != expected_model:
        raise RuntimeError(f"checkpoint metadata mismatch for seed {entry['seed']}")
    if payload.get("implementation_version") != "audited_v1_20260715":
        raise RuntimeError(f"checkpoint implementation mismatch for seed {entry['seed']}")
    config = payload.get("config", {})
    expected = {
        "target": 8,
        "train_per_class": 5000,
        "test_per_class": 500,
        "poison_frac": 0.15,
        "poison_repeat": 3,
        "patch_size": 4,
        "patch_alpha": 0.45,
        "patch_rgb": (1.0, 0.85, 0.05),
    }
    for key, value in expected.items():
        actual = tuple(config[key]) if key == "patch_rgb" else config[key]
        if actual != value:
            raise RuntimeError(f"checkpoint config mismatch for seed {entry['seed']}: {key}")
    actual_state_hash = state_hash(payload["state_dict"])
    if actual_state_hash != entry["state_sha256"] or actual_state_hash != payload.get(
        "final_state_sha256"
    ):
        raise RuntimeError(f"checkpoint state hash mismatch for seed {entry['seed']}")
    model = CifarResNet18Multi(width=64, num_classes=10)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload, path


def validate_archived_source(protocol: dict[str, Any]) -> dict[str, Any]:
    spec = protocol["archived_victim_source"]
    source_root = resolve(ROOT, spec["root_relative_to_paper14"])
    paper5_root = source_root.parents[1]
    expected = {
        source_root / "run_manifest.json": spec["source_manifest_sha256"],
        source_root / "per_seed_metrics.csv": spec["source_metrics_sha256"],
        paper5_root / "work/cifar10_multiclass_resnet18_alignment.py": spec[
            "base_runner_sha256"
        ],
        paper5_root / "work/cifar10_multiclass_resnet18_alignment_audited.py": spec[
            "audited_runner_sha256"
        ],
    }
    for path, digest in expected.items():
        if not path.exists() or sha256_file(path) != digest:
            raise RuntimeError(f"archived source hash mismatch: {path}")
    return {
        "source_root": str(source_root),
        "validated_files": {
            str(path): digest for path, digest in expected.items()
        },
    }


def class_order(targets: np.ndarray, label: int, seed: int) -> np.ndarray:
    indices = np.flatnonzero(targets == int(label))
    generator = torch.Generator().manual_seed(int(seed) + int(label) * 1009)
    order = torch.randperm(len(indices), generator=generator).numpy()
    return indices[order]


def candidate_index(protocol: dict[str, Any], train: CIFAR10, seed: int) -> dict[str, Any]:
    target = int(protocol["scope"]["target_class"])
    targets = np.asarray(train.targets, dtype=np.int64)
    target_clean = class_order(targets, target, seed)
    poison_indices: list[np.ndarray] = []
    poison_sources: list[np.ndarray] = []
    for source in protocol["candidate_construction"]["poison_source_classes"]:
        ordered = class_order(targets, int(source), seed)
        generator = torch.Generator().manual_seed(
            int(seed) + 2200 + int(source) * 131 + target * 17
        )
        local = torch.randperm(len(ordered), generator=generator)[:750].numpy()
        poison_indices.append(ordered[local])
        poison_sources.append(np.full(750, int(source), dtype=np.int64))
    poisons = np.concatenate(poison_indices)
    sources = np.concatenate(poison_sources)
    if len(target_clean) != 5000 or len(poisons) != 6750:
        raise RuntimeError("candidate target/poison count mismatch")
    if len(np.unique(poisons)) != len(poisons):
        raise RuntimeError("poison source records are not unique")
    return {
        "base_indices": np.concatenate([target_clean, poisons]),
        "original_labels": np.concatenate(
            [np.full(5000, target, dtype=np.int64), sources]
        ),
        "assigned_labels": np.full(11750, target, dtype=np.int64),
        "poison_truth": np.concatenate(
            [np.zeros(5000, dtype=bool), np.ones(6750, dtype=bool)]
        ),
        "candidate_positions": np.concatenate(
            [np.arange(40000, 45000), np.arange(50000, 56750)]
        ),
        "record_ids": [
            *[f"train_clean_{int(index)}" for index in target_clean],
            *[
                f"train_poison_source{int(source)}_base{int(index)}"
                for source, index in zip(sources, poisons)
            ],
        ],
    }


def images_to_tensor(images: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(images)
    return torch.from_numpy(array).to(torch.float32).permute(0, 3, 1, 2) / 255.0


def apply_patch(values: torch.Tensor, protocol: dict[str, Any]) -> torch.Tensor:
    spec = protocol["candidate_construction"]
    result = values.clone()
    size = int(spec["patch_size"])
    alpha = float(spec["patch_alpha"])
    patch = torch.tensor(spec["patch_rgb"], dtype=result.dtype).view(1, 3, 1, 1)
    region = result[:, :, -size:, -size:]
    result[:, :, -size:, -size:] = (1.0 - alpha) * region + alpha * patch
    return result.clamp(0.0, 1.0)


def extract_features(
    model: CifarResNet18Multi,
    images: np.ndarray,
    indices: np.ndarray,
    poison: np.ndarray,
    protocol: dict[str, Any],
    batch_size: int,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            stop = min(start + batch_size, len(indices))
            values = images_to_tensor(images[indices[start:stop]])
            mask = poison[start:stop]
            if np.any(mask):
                local = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long)
                values[local] = apply_patch(values[local], protocol)
            features = model.forward_with_reps(values)[1]["multi"]
            pieces.append(features.detach().cpu().numpy().astype(np.float32))
    result = np.concatenate(pieces, axis=0)
    if result.shape != (len(indices), 384) or not np.isfinite(result).all():
        raise RuntimeError("invalid extracted feature matrix")
    return result


def evaluate_victim(
    model: CifarResNet18Multi,
    test: CIFAR10,
    protocol: dict[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    labels = np.asarray(test.targets, dtype=np.int64)
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(test), batch_size):
            values = images_to_tensor(test.data[start : start + batch_size])
            predictions.append(model(values).argmax(1).cpu().numpy())
    predicted = np.concatenate(predictions)
    recall = [float(np.mean(predicted[labels == label] == label)) for label in range(10)]
    non_target_indices = np.flatnonzero(labels != int(protocol["scope"]["target_class"]))
    attack_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(non_target_indices), batch_size):
            indices = non_target_indices[start : start + batch_size]
            values = apply_patch(images_to_tensor(test.data[indices]), protocol)
            attack_predictions.append(model(values).argmax(1).cpu().numpy())
    attack_predicted = np.concatenate(attack_predictions)
    clean_accuracy = float(np.mean(predicted == labels))
    balanced_accuracy = float(np.mean(recall))
    asr = float(np.mean(attack_predicted == int(protocol["scope"]["target_class"])))
    gate = protocol["victim_eligibility"]
    eligible = balanced_accuracy >= float(gate["clean_balanced_accuracy_min"]) and asr >= float(
        gate["attack_success_rate_min"]
    )
    return {
        "clean_accuracy": clean_accuracy,
        "balanced_clean_accuracy": balanced_accuracy,
        "attack_success_rate": asr,
        "eligible": int(eligible),
        "per_class_recall_json": json.dumps(recall, separators=(",", ":")),
        "clean_test_count": int(len(labels)),
        "asr_test_count": int(len(non_target_indices)),
    }


def rank_pvalues(calibration: np.ndarray, query: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(calibration, dtype=np.float64).reshape(-1))
    values = np.asarray(query, dtype=np.float64).reshape(-1)
    counts = len(ordered) - np.searchsorted(ordered, values, side="left")
    return (1.0 + counts) / (len(ordered) + 1.0)


def cf_cmd_u(
    reference: np.ndarray,
    calibration: np.ndarray,
    candidates: np.ndarray,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    rng_seed = 20271001 + int(seed)
    rng = np.random.default_rng(rng_seed)
    halves = np.array_split(rng.permutation(len(candidates)), 2)
    ref_mean = np.asarray(reference, dtype=np.float64).mean(axis=0)
    cal_centered = np.asarray(calibration, dtype=np.float64) - ref_mean
    values = np.asarray(candidates, dtype=np.float64)
    scores = np.empty(len(values), dtype=np.float64)
    pvalues = np.ones(len(values), dtype=np.float64)
    norms: list[float] = []
    for scoring, fitting in ((halves[0], halves[1]), (halves[1], halves[0])):
        direction = values[fitting].mean(axis=0) - ref_mean
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            raise RuntimeError("CF-CMD-U direction collapsed")
        direction /= norm
        cal_scores = cal_centered @ direction
        query_scores = (values[scoring] - ref_mean) @ direction
        scores[scoring] = query_scores
        pvalues[scoring] = rank_pvalues(cal_scores, query_scores)
        norms.append(norm)
    return {
        "scores": scores,
        "pvalues": pvalues,
        "flags": pvalues <= alpha,
        "receipt": {
            "rng_seed": rng_seed,
            "direction_norms": norms,
            "crossfit_half_sizes": [len(part) for part in halves],
        },
    }


def centroid_distance(
    reference: np.ndarray,
    calibration: np.ndarray,
    candidates: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    center = np.asarray(reference, dtype=np.float64).mean(axis=0)
    cal_scores = np.linalg.norm(np.asarray(calibration, dtype=np.float64) - center, axis=1)
    scores = np.linalg.norm(np.asarray(candidates, dtype=np.float64) - center, axis=1)
    pvalues = rank_pvalues(cal_scores, scores)
    return {
        "scores": scores,
        "pvalues": pvalues,
        "flags": pvalues <= alpha,
        "receipt": {"reference_center_norm": float(np.linalg.norm(center))},
    }


def crossfit_spectral(
    reference: np.ndarray,
    calibration: np.ndarray,
    candidates: np.ndarray,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    rng_seed = 20271001 + int(seed)
    rng = np.random.default_rng(rng_seed)
    halves = np.array_split(rng.permutation(len(candidates)), 2)
    ref_mean = np.asarray(reference, dtype=np.float64).mean(axis=0)
    cal_centered = np.asarray(calibration, dtype=np.float64) - ref_mean
    values = np.asarray(candidates, dtype=np.float64)
    scores = np.empty(len(values), dtype=np.float64)
    pvalues = np.ones(len(values), dtype=np.float64)
    eigenvalues: list[float] = []
    for scoring, fitting in ((halves[0], halves[1]), (halves[1], halves[0])):
        fitted = values[fitting] - ref_mean
        covariance = fitted.T @ fitted / max(len(fitted) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(covariance)
        direction = eigvecs[:, -1]
        eigenvalues.append(float(eigvals[-1]))
        cal_scores = np.square(cal_centered @ direction)
        query_scores = np.square((values[scoring] - ref_mean) @ direction)
        scores[scoring] = query_scores
        pvalues[scoring] = rank_pvalues(cal_scores, query_scores)
    return {
        "scores": scores,
        "pvalues": pvalues,
        "flags": pvalues <= alpha,
        "receipt": {
            "rng_seed": rng_seed,
            "top_eigenvalues": eigenvalues,
            "crossfit_half_sizes": [len(part) for part in halves],
        },
    }


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    positives = int(truth.sum())
    negatives = int((~truth).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("AUC requires both classes")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_ranks = np.empty(len(values), dtype=np.float64)
    boundaries = np.concatenate(
        [[0], np.flatnonzero(np.diff(sorted_values) != 0) + 1, [len(values)]]
    )
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        sorted_ranks[start:stop] = (start + 1 + stop) / 2.0
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = sorted_ranks
    u = ranks[truth].sum() - positives * (positives + 1) / 2.0
    return float(u / (positives * negatives))


def detector_metrics(
    result: dict[str, Any], truth: np.ndarray, total_clean: int
) -> dict[str, Any]:
    flags = np.asarray(result["flags"], dtype=bool)
    poison = np.asarray(truth, dtype=bool)
    tp = int(np.sum(flags & poison))
    fn = int(np.sum(~flags & poison))
    fp = int(np.sum(flags & ~poison))
    tn_target = int(np.sum(~flags & ~poison))
    tpr = tp / (tp + fn)
    target_fpr = fp / (fp + tn_target)
    all_clean_fpr = fp / total_clean
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn_target": tn_target,
        "tn_all_clean": total_clean - fp,
        "tpr": tpr,
        "target_clean_fpr": target_fpr,
        "all_clean_fpr": all_clean_fpr,
        "auc_target_assigned": binary_auc(poison, result["scores"]),
        "flagged_count": int(flags.sum()),
    }


def trusted_indices(test: CIFAR10, seed: int, target: int = 8) -> dict[str, np.ndarray]:
    target_indices = np.flatnonzero(np.asarray(test.targets) == target)
    order = np.random.default_rng(2026100600 + int(seed)).permutation(target_indices)
    return {"reference": order[:400], "calibration": order[400:800], "reserved": order[800:]}


def per_sample_rows(
    seed: int,
    method: str,
    candidate: dict[str, Any],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "seed": seed,
            "dataset": "CIFAR-10",
            "method": method,
            "candidate_position": int(candidate["candidate_positions"][index]),
            "record_id": candidate["record_ids"][index],
            "base_train_index": int(candidate["base_indices"][index]),
            "original_label": int(candidate["original_labels"][index]),
            "assigned_label": int(candidate["assigned_labels"][index]),
            "poison_truth": int(candidate["poison_truth"][index]),
            "defined": 1,
            "score": float(result["scores"][index]),
            "p_value": float(result["pvalues"][index]),
            "flag": int(result["flags"][index]),
        }
        for index in range(len(candidate["base_indices"]))
    ]


def copy_or_link_model(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) == sha256_file(source):
            return
        raise RuntimeError(f"existing model differs: {destination}")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def bootstrap_seed_mean(values: list[float], seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(10000, len(array)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def synthetic_smoke(output: Path, protocol: dict[str, Any], log: RunLog) -> None:
    rng = np.random.default_rng(20261006)
    reference = rng.normal(0, 1, size=(400, 16))
    calibration = rng.normal(0, 1, size=(400, 16))
    clean = rng.normal(0, 1, size=(100, 16))
    poison = rng.normal(0, 1, size=(20, 16))
    poison[:, 0] += 5.0
    candidates = np.concatenate([clean, poison])
    truth = np.concatenate([np.zeros(100, dtype=bool), np.ones(20, dtype=bool)])
    methods = {
        "cf_cmd_u": cf_cmd_u(reference, calibration, candidates, 999, 0.05),
        "target_centroid_distance": centroid_distance(
            reference, calibration, candidates, 0.05
        ),
        "crossfit_spectral": crossfit_spectral(
            reference, calibration, candidates, 999, 0.05
        ),
    }
    rows = []
    for method, result in methods.items():
        row = {"method": method, **detector_metrics(result, truth, 100)}
        if not all(math.isfinite(float(row[key])) for key in ("tpr", "target_clean_fpr", "auc_target_assigned")):
            raise RuntimeError("synthetic smoke produced non-finite metrics")
        rows.append(row)
    write_csv(output / "smoke_metrics.csv", rows)
    write_json(
        output / "smoke_receipt.json",
        {
            "status": "PASS",
            "synthetic_only": True,
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "metrics": rows,
        },
    )
    log("Synthetic detector smoke PASS")


def summarize(
    protocol: dict[str, Any], victims: list[dict[str, Any]], methods: list[dict[str, Any]]
) -> dict[str, Any]:
    eligible_seeds = {int(row["seed"]) for row in victims if int(row["eligible"]) == 1}
    summaries: dict[str, Any] = {}
    for method in ("cf_cmd_u", "target_centroid_distance", "crossfit_spectral"):
        rows = [
            row
            for row in methods
            if row["method"] == method and int(row["seed"]) in eligible_seeds
        ]
        tp = sum(int(row["tp"]) for row in rows)
        fn = sum(int(row["fn"]) for row in rows)
        fp = sum(int(row["fp"]) for row in rows)
        tn_target = sum(int(row["tn_target"]) for row in rows)
        total_clean = len(rows) * 50000
        seed_tpr = [float(row["tpr"]) for row in rows]
        seed_target_fpr = [float(row["target_clean_fpr"]) for row in rows]
        seed_auc = [float(row["auc_target_assigned"]) for row in rows]
        summaries[method] = {
            "eligible_seed_count": len(rows),
            "eligible_seeds": [int(row["seed"]) for row in rows],
            "pooled_tpr": tp / (tp + fn) if tp + fn else None,
            "pooled_target_clean_fpr": fp / (fp + tn_target) if fp + tn_target else None,
            "pooled_all_clean_fpr": fp / total_clean if total_clean else None,
            "mean_seed_tpr": float(np.mean(seed_tpr)) if rows else None,
            "mean_seed_target_clean_fpr": float(np.mean(seed_target_fpr)) if rows else None,
            "mean_seed_auc": float(np.mean(seed_auc)) if rows else None,
            "seed_bootstrap_tpr_ci95": bootstrap_seed_mean(seed_tpr, 202610061) if rows else None,
            "seed_bootstrap_target_clean_fpr_ci95": bootstrap_seed_mean(
                seed_target_fpr, 202610062
            )
            if rows
            else None,
            "seed_bootstrap_auc_ci95": bootstrap_seed_mean(seed_auc, 202610063)
            if rows
            else None,
        }
    primary = summaries["cf_cmd_u"]
    min_eligible = int(protocol["scope"]["minimum_eligible_victims"])
    enough = len(eligible_seeds) >= min_eligible
    passes = bool(
        enough
        and primary["pooled_tpr"] is not None
        and primary["pooled_tpr"] >= 0.90
        and primary["pooled_target_clean_fpr"] <= 0.05
    )
    if not enough:
        decision = "INCONCLUSIVE_INSUFFICIENT_ELIGIBLE_VICTIMS"
    elif passes:
        decision = "PRIMARY_GATE_PASS"
    else:
        decision = "PRIMARY_GATE_FAIL"
    return {
        "eligible_victim_count": len(eligible_seeds),
        "required_eligible_victims": min_eligible,
        "eligible_seeds": sorted(eligible_seeds),
        "decision": decision,
        "detector_summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = RunLog(output / "run.log")
    started = time.time()
    try:
        protocol = read_json(args.protocol)
        if protocol.get("protocol_id") != "tifs_native_cifar10_v1":
            raise RuntimeError("unexpected protocol")
        torch.set_num_threads(args.threads)
        train, test, dataset_receipt = validate_dataset(protocol)
        source_receipt = validate_archived_source(protocol)
        log(
            f"Official CIFAR-10 integrity PASS: train={len(train)} test={len(test)} "
            f"archive_sha256={dataset_receipt['archive_sha256']}"
        )
        first_entry = protocol["archived_victim_source"]["checkpoints"][0]
        first_model, _, _ = load_model(protocol, first_entry)
        with torch.no_grad():
            check = first_model(images_to_tensor(test.data[:2]))
        if check.shape != (2, 10) or not torch.isfinite(check).all():
            raise RuntimeError("checkpoint forward-pass smoke failed")
        log("Checkpoint integrity/load/forward smoke PASS")
        if args.smoke:
            synthetic_smoke(output, protocol, log)
            return

        shutil.copy2(args.protocol, output / "frozen_protocol.json")
        victims: list[dict[str, Any]] = []
        method_rows: list[dict[str, Any]] = []
        sample_rows: list[dict[str, Any]] = []
        seed_receipts: list[dict[str, Any]] = []
        alpha = float(protocol["detectors"]["primary"]["alpha"])
        for entry in protocol["archived_victim_source"]["checkpoints"]:
            seed_started = time.time()
            seed = int(entry["seed"])
            log(f"seed={seed} start")
            model, payload, checkpoint_path = load_model(protocol, entry)
            candidate = candidate_index(protocol, train, seed)
            target_features = extract_features(
                model,
                train.data,
                candidate["base_indices"],
                candidate["poison_truth"],
                protocol,
                args.batch_size,
            )
            split = trusted_indices(test, seed)
            trusted_all = np.concatenate(
                [split["reference"], split["calibration"], split["reserved"]]
            )
            trusted_features = extract_features(
                model,
                test.data,
                trusted_all,
                np.zeros(len(trusted_all), dtype=bool),
                protocol,
                args.batch_size,
            )
            reference = trusted_features[:400]
            calibration = trusted_features[400:800]
            victim = {
                "seed": seed,
                **evaluate_victim(model, test, protocol, args.batch_size),
                "checkpoint_file_sha256": entry["file_sha256"],
                "checkpoint_state_sha256": entry["state_sha256"],
                "candidate_count": 56750,
                "defined_target_candidate_count": len(target_features),
                "poison_count": int(candidate["poison_truth"].sum()),
                "clean_target_count": int((~candidate["poison_truth"]).sum()),
            }
            victims.append(victim)
            methods: dict[str, Callable[[], dict[str, Any]]] = {
                "cf_cmd_u": lambda: cf_cmd_u(
                    reference, calibration, target_features, seed, alpha
                ),
                "target_centroid_distance": lambda: centroid_distance(
                    reference, calibration, target_features, alpha
                ),
                "crossfit_spectral": lambda: crossfit_spectral(
                    reference, calibration, target_features, seed, alpha
                ),
            }
            detector_receipts: dict[str, Any] = {}
            for method, builder in methods.items():
                result = builder()
                metrics = detector_metrics(result, candidate["poison_truth"], 50000)
                gate_pass = int(
                    metrics["tpr"] >= 0.90 and metrics["target_clean_fpr"] <= 0.05
                )
                method_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        **metrics,
                        "victim_eligible": victim["eligible"],
                        "detector_gate_pass": gate_pass,
                    }
                )
                sample_rows.extend(per_sample_rows(seed, method, candidate, result))
                detector_receipts[method] = result["receipt"]
                log(
                    f"seed={seed} method={method} TPR={metrics['tpr']:.6f} "
                    f"target_FPR={metrics['target_clean_fpr']:.6f} "
                    f"all_FPR={metrics['all_clean_fpr']:.6f} "
                    f"AUC={metrics['auc_target_assigned']:.6f}"
                )
            output_model = output / "models" / f"seed_{seed}_suspect.pt"
            copy_or_link_model(checkpoint_path, output_model)
            seed_receipts.append(
                {
                    "seed": seed,
                    "reference_indices": split["reference"].tolist(),
                    "calibration_indices": split["calibration"].tolist(),
                    "reserved_indices": split["reserved"].tolist(),
                    "reference_indices_sha256": hashlib.sha256(
                        split["reference"].astype(np.int64).tobytes()
                    ).hexdigest(),
                    "calibration_indices_sha256": hashlib.sha256(
                        split["calibration"].astype(np.int64).tobytes()
                    ).hexdigest(),
                    "candidate_base_indices_sha256": hashlib.sha256(
                        candidate["base_indices"].astype(np.int64).tobytes()
                    ).hexdigest(),
                    "candidate_truth_sha256": hashlib.sha256(
                        candidate["poison_truth"].astype(np.uint8).tobytes()
                    ).hexdigest(),
                    "checkpoint_source": str(checkpoint_path),
                    "checkpoint_output": str(output_model),
                    "checkpoint_file_sha256": sha256_file(output_model),
                    "checkpoint_state_sha256": state_hash(payload["state_dict"]),
                    "detector_receipts": detector_receipts,
                    "elapsed_seconds": time.time() - seed_started,
                }
            )
            log(
                f"seed={seed} victim CA={victim['clean_accuracy']:.6f} "
                f"BCA={victim['balanced_clean_accuracy']:.6f} "
                f"ASR={victim['attack_success_rate']:.6f} eligible={victim['eligible']} "
                f"elapsed={time.time() - seed_started:.1f}s"
            )

        summary = summarize(protocol, victims, method_rows)
        write_csv(output / "victim_metrics.csv", victims)
        write_csv(output / "detector_metrics.csv", method_rows)
        write_csv(output / "per_sample_scores.csv", sample_rows)
        write_json(output / "seed_receipts.json", seed_receipts)
        write_json(output / "summary.json", summary)
        log(f"complete decision={summary['decision']} elapsed={time.time() - started:.1f}s")
        runtime = {
            "experiment": "tifs_native_cifar10_v1",
            "status": "complete",
            "protocol_path": str(args.protocol.resolve()),
            "protocol_sha256": sha256_file(args.protocol),
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "dataset_receipt": dataset_receipt,
            "archived_source_receipt": source_receipt,
            "seeds": protocol["scope"]["confirmation_seeds"],
            "device": "cpu",
            "torch_version": torch.__version__,
            "torchvision_version": __import__("torchvision").__version__,
            "numpy_version": np.__version__,
            "python_version": sys.version,
            "platform": platform.platform(),
            "threads": args.threads,
            "batch_size": args.batch_size,
            "elapsed_seconds": time.time() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "output_files": {},
            "summary": summary,
        }
        for name in (
            "frozen_protocol.json",
            "victim_metrics.csv",
            "detector_metrics.csv",
            "per_sample_scores.csv",
            "seed_receipts.json",
            "summary.json",
            "run.log",
        ):
            path = output / name
            runtime["output_files"][name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        write_json(output / "manifest.json", runtime)
    finally:
        log.close()


if __name__ == "__main__":
    main()
