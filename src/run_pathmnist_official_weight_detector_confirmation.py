#!/usr/bin/env python3
"""MPS executor for the frozen PathMNIST detector confirmation.

The formal path is implemented but locked behind a real-input MPS smoke
receipt and explicit seed-consumption ACK. Synthetic readiness never touches
PathMNIST, official weights, checkpoints, or confirmation seed files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import pathmnist_external_validation as core
import pathmnist_official_resnet18 as architecture


PAPER14_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/"
    "pathmnist_official_weight_detector_confirmation_protocol_v1.json"
)
CONFIRMATION_SEEDS = (103, 104, 105, 106, 107)
WEIGHT_MAP = {
    103: "resnet18_28_1.pth",
    104: "resnet18_28_2.pth",
    105: "resnet18_28_1.pth",
    106: "resnet18_28_2.pth",
    107: "resnet18_28_1.pth",
}
TARGET_CLASS = 8
REFERENCE_COUNT = 512
CALIBRATION_COUNT = 512
ALPHA = 0.05
DEFAULT_DATASET = Path("/private/tmp/paper14-data/pathmnist.npz")
DEFAULT_OUTPUT = PAPER14_ROOT / "results/pathmnist_official_weight_detector_confirmation_v1"
DEFAULT_SMOKE_OUTPUT = PAPER14_ROOT / "results/pathmnist_real_input_mps_smoke_v1"
WEIGHT_RECEIPT = PAPER14_ROOT / "results/top_journal_program_v1/pathmnist_official_weights_acquisition_receipt.json"
BATCH_SIZE = 128
EPOCHS = 5
LR = 3e-4
FORMAL_ACK = "CONSUME_PATHMNIST_FORMAL_SEEDS_103_107"
NONFORMAL_SMOKE_SEED = 20261002
CANDIDATE_FEATURE_SOURCE = "same_realized_suspect_poison_pool_for_clean_and_suspect_models"


def validate_seed_map(seeds: tuple[int, ...], mapping: dict[int, str]) -> None:
    """Guard the future seed plan without opening files for any seed."""

    if seeds != CONFIRMATION_SEEDS:
        raise ValueError(f"formal seed plan is fixed to {CONFIRMATION_SEEDS}")
    if mapping != WEIGHT_MAP:
        raise ValueError("official-weight map differs from the frozen protocol")


def _matrix(name: str, values: Any, *, dimension: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or not result.shape[0] or not result.shape[1]:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if dimension is not None and result.shape[1] != dimension:
        raise ValueError(f"{name} feature dimensions do not match")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def _labels(name: str, labels: Any, length: int) -> np.ndarray:
    result = np.asarray(labels)
    if result.ndim != 1 or len(result) != length:
        raise ValueError(f"{name} must be one-dimensional and align with features")
    if not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{name} must have integer dtype")
    return result.astype(np.int64, copy=False)


def _rank_pvalues(calibration_scores: np.ndarray, query_scores: np.ndarray) -> np.ndarray:
    calibration = np.asarray(calibration_scores, dtype=np.float64).reshape(-1)
    query = np.asarray(query_scores, dtype=np.float64).reshape(-1)
    if not len(calibration):
        raise ValueError("calibration scores must be nonempty")
    # Finite-sample conservative upper-tail rank; ties count against a flag.
    return (1.0 + (calibration[:, None] >= query[None, :]).sum(axis=0)) / (
        len(calibration) + 1.0
    )


def cf_cmd_u_unwhitened(
    reference_features: Any,
    calibration_features: Any,
    candidate_features: Any,
    candidate_labels: Any,
    *,
    seed: int,
    target_class: int = TARGET_CLASS,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Two-fold target-candidate cross-fit mean direction in raw coordinates.

    There is no covariance estimate or learned feature transform here. The
    validation reference mean centers the calibration/query projections; each
    query half is scored with a unit direction formed from the opposite half.
    """

    reference = _matrix("reference_features", reference_features)
    calibration = _matrix(
        "calibration_features", calibration_features, dimension=reference.shape[1]
    )
    candidates = _matrix(
        "candidate_features", candidate_features, dimension=reference.shape[1]
    )
    labels = _labels("candidate_labels", candidate_labels, len(candidates))
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if len(reference) != REFERENCE_COUNT or len(calibration) != CALIBRATION_COUNT:
        raise ValueError("CF-CMD-U requires distinct 512-row reference and calibration sets")
    target_indices = np.flatnonzero(labels == int(target_class))
    if len(target_indices) < 4:
        raise ValueError("at least four target-assigned candidates are required")

    rng_seed = 20271001 + int(seed)
    rng = np.random.default_rng(rng_seed)
    shuffled = rng.permutation(target_indices)
    halves = np.array_split(shuffled, 2)
    ref_mean = reference.mean(axis=0)
    calibration_centered = calibration - ref_mean
    scores = np.full(len(candidates), np.nan, dtype=np.float64)
    p_values = np.ones(len(candidates), dtype=np.float64)
    flags = np.zeros(len(candidates), dtype=bool)
    direction_norms: list[float] = []

    for scoring, fitting in ((halves[0], halves[1]), (halves[1], halves[0])):
        direction = candidates[fitting].mean(axis=0) - ref_mean
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            raise RuntimeError("unwhitened CF-CMD-U direction collapsed")
        direction = direction / norm
        calibration_scores = calibration_centered @ direction
        query_scores = (candidates[scoring] - ref_mean) @ direction
        query_p = _rank_pvalues(calibration_scores, query_scores)
        scores[scoring] = query_scores
        p_values[scoring] = query_p
        flags[scoring] = query_p <= float(alpha)
        direction_norms.append(norm)

    return {
        "scores": scores,
        "p_values": p_values,
        "flags": flags,
        "defined": labels == int(target_class),
        "receipt": {
            "method": "CF-CMD-U",
            "whitening": False,
            "covariance_estimation": False,
            "projection": "identity/raw hidden-feature coordinates",
            "reference_count": len(reference),
            "calibration_count": len(calibration),
            "target_class": int(target_class),
            "candidate_target_count": len(target_indices),
            "alpha": float(alpha),
            "rng_seed": rng_seed,
            "crossfit": "two disjoint candidate halves; each half scored by the opposite-half raw mean direction",
            "direction_norms_before_normalization": direction_norms,
            "p_value_rule": "(1 + count(calibration_score >= query_score)) / (n_calibration + 1)",
        },
    }


def target_centroid_distance(
    reference_features: Any,
    calibration_features: Any,
    candidate_features: Any,
    candidate_labels: Any,
    *,
    target_class: int = TARGET_CLASS,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Simple raw-space Euclidean centroid distance with rank calibration."""

    reference = _matrix("reference_features", reference_features)
    calibration = _matrix(
        "calibration_features", calibration_features, dimension=reference.shape[1]
    )
    candidates = _matrix(
        "candidate_features", candidate_features, dimension=reference.shape[1]
    )
    labels = _labels("candidate_labels", candidate_labels, len(candidates))
    ref_center = reference.mean(axis=0)
    calibration_scores = np.linalg.norm(calibration - ref_center, axis=1)
    scores = np.linalg.norm(candidates - ref_center, axis=1)
    p_values = np.ones(len(candidates), dtype=np.float64)
    defined = labels == int(target_class)
    p_values[defined] = _rank_pvalues(calibration_scores, scores[defined])
    flags = defined & (p_values <= float(alpha))
    return {
        "scores": scores,
        "p_values": p_values,
        "flags": flags,
        "defined": defined,
        "receipt": {
            "method": "target_reference_centroid_distance",
            "metric": "Euclidean distance in raw feature coordinates",
            "target_class": int(target_class),
            "alpha": float(alpha),
            "reference_count": len(reference),
            "calibration_count": len(calibration),
        },
    }


def paired_drift_view(
    clean_validation_features: Any,
    suspect_validation_features: Any,
    clean_candidate_features: Any,
    suspect_candidate_features: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the frozen secondary view from same-input paired representations."""

    clean_validation = _matrix("clean_validation_features", clean_validation_features)
    suspect_validation = _matrix(
        "suspect_validation_features",
        suspect_validation_features,
        dimension=clean_validation.shape[1],
    )
    clean_candidates = _matrix(
        "clean_candidate_features",
        clean_candidate_features,
        dimension=clean_validation.shape[1],
    )
    suspect_candidates = _matrix(
        "suspect_candidate_features",
        suspect_candidate_features,
        dimension=clean_validation.shape[1],
    )
    if clean_validation.shape[0] != suspect_validation.shape[0]:
        raise ValueError("paired validation features must have matching rows")
    if clean_candidates.shape[0] != suspect_candidates.shape[0]:
        raise ValueError("paired candidate features must have matching rows")
    return suspect_validation - clean_validation, suspect_candidates - clean_candidates


def run_scan_baseline(
    validation_features: Any,
    validation_labels: Any,
    candidate_features: Any,
    candidate_labels: Any,
    *,
    seed: int,
    n_classes: int = 9,
) -> dict[str, Any]:
    """Call the existing pinned BackdoorBench SCAn feature adapter."""

    import backdoorbench_feature_adapter as adapter

    result = adapter.scan(
        validation_features,
        validation_labels,
        candidate_features,
        candidate_labels,
        n_classes=n_classes,
        seed=seed,
    )
    return {"scores": result.scores, "flags": result.suspicious_mask, "receipt": result.receipt}


def run_spectral_signatures_baseline(
    candidate_features: Any,
    candidate_labels: Any,
    *,
    percentile: float = 85.0,
) -> dict[str, Any]:
    """Call the existing pinned BackdoorBench Spectral Signatures adapter."""

    import backdoorbench_feature_adapter as adapter

    result = adapter.spectral_signatures(
        candidate_features, candidate_labels, percentile=percentile
    )
    return {"scores": result.scores, "flags": result.suspicious_mask, "receipt": result.receipt}


def confusion_counts(
    flags: Any,
    poison_truth: Any,
    assigned_labels: Any,
    *,
    target_class: int = TARGET_CLASS,
) -> dict[str, int | float]:
    """Join poison truth only after the detector has emitted fixed flags."""

    flagged = np.asarray(flags, dtype=bool)
    poison = np.asarray(poison_truth, dtype=bool)
    assigned = _labels("assigned_labels", assigned_labels, len(flagged))
    if poison.shape != flagged.shape:
        raise ValueError("poison truth must align with detector flags")
    clean = ~poison
    target_clean = clean & (assigned == int(target_class))
    tp = int(np.sum(flagged & poison))
    n_poison = int(poison.sum())
    fp = int(np.sum(flagged & clean))
    n_clean = int(clean.sum())
    target_fp = int(np.sum(flagged & target_clean))
    n_target_clean = int(target_clean.sum())
    return {
        "tp": tp,
        "n_poison": n_poison,
        "fp_all_clean": fp,
        "n_all_clean": n_clean,
        "fp_target_clean": target_fp,
        "n_target_clean": n_target_clean,
        "tpr": tp / n_poison if n_poison else float("nan"),
        "all_clean_fpr": fp / n_clean if n_clean else float("nan"),
        "target_clean_fpr": target_fp / n_target_clean if n_target_clean else float("nan"),
    }


def summarize_seed_blocks(
    rows: list[dict[str, Any]],
    *,
    bootstrap_replicates: int = 20_000,
    bootstrap_seed: int = 20261001,
) -> dict[str, Any]:
    """Micro-pool fixed seed counts and resample whole seed blocks for CIs."""

    if len(rows) != len(CONFIRMATION_SEEDS):
        raise ValueError("a five-seed summary requires exactly five seed blocks")
    seeds = tuple(int(row["seed"]) for row in rows)
    if seeds != CONFIRMATION_SEEDS:
        raise ValueError("seed blocks must appear in frozen confirmation order")
    if any(not bool(row["eligible"]) for row in rows):
        raise ValueError("five-seed confirmation summary requires five eligible victims")
    count_keys = (
        "tp",
        "n_poison",
        "fp_all_clean",
        "n_all_clean",
        "fp_target_clean",
        "n_target_clean",
    )
    counts = {key: int(sum(int(row[key]) for row in rows)) for key in count_keys}
    point = {
        "pooled_tpr": counts["tp"] / counts["n_poison"],
        "pooled_all_clean_fpr": counts["fp_all_clean"] / counts["n_all_clean"],
        "pooled_target_clean_fpr": counts["fp_target_clean"] / counts["n_target_clean"],
    }
    rng = np.random.default_rng(int(bootstrap_seed))
    values = np.empty((int(bootstrap_replicates), 3), dtype=np.float64)
    for replicate in range(int(bootstrap_replicates)):
        sampled = rng.integers(0, len(rows), size=len(rows))
        sample_counts = {
            key: sum(int(rows[index][key]) for index in sampled) for key in count_keys
        }
        values[replicate] = (
            sample_counts["tp"] / sample_counts["n_poison"],
            sample_counts["fp_all_clean"] / sample_counts["n_all_clean"],
            sample_counts["fp_target_clean"] / sample_counts["n_target_clean"],
        )
    names = ("pooled_tpr", "pooled_all_clean_fpr", "pooled_target_clean_fpr")
    intervals = {
        name: [float(x) for x in np.quantile(values[:, index], [0.025, 0.975])]
        for index, name in enumerate(names)
    }
    gate = bool(point["pooled_tpr"] >= 0.90 and point["pooled_all_clean_fpr"] <= 0.05)
    return {
        "seeds": list(seeds),
        "eligible_seed_count": len(rows),
        "counts": counts,
        **point,
        "seed_block_bootstrap_95pct_ci": intervals,
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(bootstrap_seed),
        "primary_point_gate_pass": gate,
        "primary_confirmation_pass": gate and len(rows) == 5,
        "uncertainty_note": "Only five independent seed blocks are available; intervals show seed-level uncertainty and do not establish broad population generality.",
    }


def synthetic_dry_run() -> dict[str, Any]:
    """Exercise raw CF-CMD-U, distance, metrics and stats with synthetic data."""

    rng = np.random.default_rng(94103)  # synthetic-data RNG; not an experiment seed
    dimension = 12
    reference = rng.normal(size=(REFERENCE_COUNT, dimension))
    calibration = rng.normal(size=(CALIBRATION_COUNT, dimension))
    labels = np.full(96, TARGET_CLASS, dtype=np.int64)
    labels[:16] = 3
    poison = np.zeros(96, dtype=bool)
    poison[48:64] = True
    candidates = rng.normal(size=(96, dimension))
    candidates[poison] += 1.25
    primary = cf_cmd_u_unwhitened(
        reference, calibration, candidates, labels, seed=999_001
    )
    distance = target_centroid_distance(
        reference, calibration, candidates, labels
    )
    primary_metrics = confusion_counts(primary["flags"], poison, labels)
    distance_metrics = confusion_counts(distance["flags"], poison, labels)
    assert np.isfinite(primary["scores"][primary["defined"]]).all()
    assert np.isfinite(primary["p_values"]).all()
    assert not primary["flags"][~primary["defined"]].any()
    assert primary["receipt"]["whitening"] is False
    assert primary_metrics["n_poison"] == 16
    assert 0.0 <= primary_metrics["all_clean_fpr"] <= 1.0
    return {
        "status": "synthetic_dry_run_passed_nonformal",
        "formal_data_or_weights_loaded": False,
        "training_performed": False,
        "confirmation_seed_files_opened": False,
        "synthetic_rows": len(candidates),
        "feature_dimension": dimension,
        "primary_receipt": primary["receipt"],
        "primary_metrics": primary_metrics,
        "distance_metrics": distance_metrics,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def resolve_weight(receipt: dict[str, Any], seed: int) -> Path:
    filename = WEIGHT_MAP[seed]
    matches = [x for x in receipt.get("extraction", {}).get("members", []) if x.get("name") == filename]
    if len(matches) != 1:
        raise RuntimeError(f"official-weight receipt does not uniquely identify {filename}")
    return PAPER14_ROOT.parent / matches[0]["local_path"]


def _frozen_hashes(dataset: Path, protocol_sha: str) -> dict[str, Any]:
    if not dataset.is_file() or not WEIGHT_RECEIPT.is_file():
        raise FileNotFoundError("dataset or official-weight acquisition receipt is missing")
    weights_receipt = json.loads(WEIGHT_RECEIPT.read_text(encoding="utf-8"))
    weights = {str(s): sha256_file(resolve_weight(weights_receipt, s)) for s in CONFIRMATION_SEEDS}
    return {
        "protocol_sha256": protocol_sha,
        "dataset_sha256": sha256_file(dataset),
        "weight_receipt_sha256": sha256_file(WEIGHT_RECEIPT),
        "official_weight_sha256_by_seed": weights,
        "runner_sha256": sha256_file(Path(__file__)),
        "core_sha256": sha256_file(Path(core.__file__)),
        "architecture_sha256": sha256_file(Path(architecture.__file__)),
        "feature_adapter_sha256": sha256_file(PAPER14_ROOT / "src/backdoorbench_feature_adapter.py"),
    }


def _mps_device() -> torch.device:
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("frozen primary device Apple MPS is unavailable; no CPU fallback")
    if torch.__version__.split("+")[0] != "2.7.1":
        raise RuntimeError(f"protocol requires torch 2.7.1, observed {torch.__version__}")
    torch.use_deterministic_algorithms(True)
    return torch.device("mps")


def _load_train_validation(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Read train/validation arrays only; test keys remain unopened until evaluation."""
    with np.load(path, allow_pickle=False) as payload:
        result = {}
        for split, count in (("train", 89996), ("val", 10004)):
            images = payload[f"{split}_images"]
            labels = payload[f"{split}_labels"].reshape(-1)
            if images.shape != (count, 28, 28, 3) or images.dtype != np.uint8:
                raise ValueError(f"unexpected {split} image array")
            if labels.shape != (count,) or set(np.unique(labels)) != set(range(9)):
                raise ValueError(f"unexpected {split} labels")
            result[split] = {"images": images, "labels": labels.astype(np.int64, copy=False)}
    return result


def _load_test(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        images = payload["test_images"]
        labels = payload["test_labels"].reshape(-1)
    if images.shape != (7180, 28, 28, 3) or images.dtype != np.uint8 or labels.shape != (7180,):
        raise ValueError("unexpected official test split")
    return {"images": images, "labels": labels.astype(np.int64, copy=False)}


def _select_validation_indices(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = np.flatnonzero(np.asarray(labels, dtype=np.int64) == TARGET_CLASS)
    if len(target) < 1024:
        raise ValueError("validation split has fewer than 1,024 target-class rows")
    rng = np.random.default_rng(20271000)
    order = rng.permutation(np.sort(target))
    return order[:512].astype(np.int64), order[512:1024].astype(np.int64)


def build_suspect_candidate_pool(images: np.ndarray, labels: np.ndarray, *, seed: int,
                                poison_count: int = 4500) -> dict[str, Any]:
    """Build BadNets suspect training/candidate pool while preserving source rows."""
    source = np.asarray(images)
    original = np.asarray(labels, dtype=np.int64)
    if source.ndim != 4 or source.shape[1:] != (28, 28, 3) or source.dtype != np.uint8 or original.shape != (len(source),):
        raise ValueError("candidate pool requires aligned uint8 PathMNIST images and labels")
    eligible = np.flatnonzero(original != TARGET_CLASS)
    if poison_count < 1 or poison_count >= len(eligible):
        raise ValueError("invalid poison count")
    selected = np.sort(np.random.default_rng(20260930 + int(seed)).choice(eligible, size=poison_count, replace=False))
    poisoned_images = source.copy()
    poisoned_images[selected, -4:, -4:, :] = 255
    assigned = original.copy(); assigned[selected] = TARGET_CLASS
    return {"images": poisoned_images, "original_labels": original.copy(),
            "assigned_labels": assigned, "selected_indices": selected}


def candidate_pool_image_hash(images: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(images).tobytes()).hexdigest()


def shared_candidate_feature_inputs(poisoned_pool_images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact same poisoned candidate input array for each pair member."""
    values = np.asarray(poisoned_pool_images)
    if values.dtype != np.uint8 or values.ndim != 4 or values.shape[1:] != (28, 28, 3):
        raise ValueError("candidate feature source must be a uint8 PathMNIST image pool")
    return values, values


def _tensor_batch(images: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2))).to(device=device, dtype=torch.float32).div_(255.0)


def _hidden(model: nn.Module, images: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            value = _tensor_batch(images[start:start + batch_size], device)
            hidden = model.hidden(value)
            if hidden.ndim != 2 or hidden.shape[1] != 512:
                raise RuntimeError("official model hidden layer must be 512-D")
            pieces.append(hidden.cpu().numpy().astype(np.float32))
    return np.concatenate(pieces, axis=0)


def _train_member(images: np.ndarray, labels: np.ndarray, initial: dict[str, torch.Tensor], *, seed: int,
                  condition: str, device: torch.device, order_seed: int, output: Path,
                  checkpoint_path: Path, resume: bool, dataset_sha256: str,
                  weight_sha256: str, training_data_sha256: str) -> tuple[architecture.PathMNISTResNet18, dict[str, Any]]:
    # Each pair member gets a fresh model loaded on CPU from the same immutable official state.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device) == "mps":
        torch.mps.manual_seed(seed)
    model = architecture.PathMNISTResNet18()
    model.load_state_dict(initial, strict=True)
    if not all(torch.equal(model.state_dict()[k].cpu(), v.cpu()) for k, v in initial.items()):
        raise RuntimeError("CPU initializer differs from the official checkpoint")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    gen = torch.Generator(device="cpu").manual_seed(order_seed)
    ds = torch.utils.data.TensorDataset(core._tensor_images(images), torch.as_tensor(labels, dtype=torch.long))
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=gen)
    history = []
    start_epoch = 1
    if checkpoint_path.exists():
        if not resume:
            raise RuntimeError(f"checkpoint exists; pass --resume: {checkpoint_path}")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected = {"seed": seed, "condition": condition, "data_order_seed": order_seed,
                    "initial_state_sha256": core.state_fingerprint(initial), "epochs": EPOCHS,
                    "device": str(device), "protocol_sha256": sha256_file(PROTOCOL_PATH),
                    "dataset_sha256": dataset_sha256, "weight_sha256": weight_sha256,
                    "training_data_sha256": training_data_sha256,
                    "runner_sha256": sha256_file(Path(__file__)), "core_sha256": sha256_file(Path(core.__file__)),
                    "architecture_sha256": sha256_file(Path(architecture.__file__))}
        if saved.get("contract") != expected:
            raise RuntimeError("resume checkpoint contract mismatch")
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        gen.set_state(saved["sampler_generator"])
        random.setstate(saved["rng_state"]["python"])
        np.random.set_state(saved["rng_state"]["numpy"])
        torch.set_rng_state(saved["rng_state"]["torch_cpu"])
        if str(device) == "mps" and saved["rng_state"].get("torch_mps") is not None:
            torch.mps.set_rng_state(saved["rng_state"]["torch_mps"])
        history = list(saved["history"])
        start_epoch = int(saved["epoch"]) + 1
        if not 1 <= start_epoch <= EPOCHS + 1 or len(history) != start_epoch - 1:
            raise RuntimeError("resume checkpoint epoch/history mismatch")
    elif resume:
        print(f"{condition} seed={seed}: no checkpoint; starting at epoch 1", flush=True)
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32).div_(255.0)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(xb)
            seen += len(xb)
        if seen != len(images):
            raise RuntimeError("epoch did not process every training row")
        history.append({"epoch": epoch, "loss": loss_sum / seen, "rows": seen})
        payload = {"contract": {"seed": seed, "condition": condition, "data_order_seed": order_seed,
                                "initial_state_sha256": core.state_fingerprint(initial), "epochs": EPOCHS,
                                "device": str(device), "protocol_sha256": sha256_file(PROTOCOL_PATH),
                                "dataset_sha256": dataset_sha256, "weight_sha256": weight_sha256,
                                "training_data_sha256": training_data_sha256,
                                "runner_sha256": sha256_file(Path(__file__)), "core_sha256": sha256_file(Path(core.__file__)),
                                "architecture_sha256": sha256_file(Path(architecture.__file__))},
                   "seed": seed, "condition": condition, "epoch": epoch,
                   "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                   "optimizer": optimizer.state_dict(), "sampler_generator": gen.get_state(),
                   "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
                                 "torch_mps": torch.mps.get_rng_state() if str(device) == "mps" else None},
                   "data_order_seed": order_seed, "history": history}
        tmp = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, tmp)
        os.replace(tmp, checkpoint_path)
    return model, {"condition": condition, "history": history, "checkpoint_sha256": sha256_file(checkpoint_path),
                   "state_sha256": core.state_fingerprint(model.state_dict()), "checkpoint_path": str(checkpoint_path)}


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean([np.mean(predictions[labels == cls] == cls) for cls in np.unique(labels)]))


def _test_metrics(model: nn.Module, images: np.ndarray, labels: np.ndarray, device: torch.device) -> dict[str, float]:
    probs = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(images), 512):
            probs.append(torch.softmax(model(_tensor_batch(images[s:s+512], device)), dim=1).cpu().numpy())
    probabilities = np.concatenate(probs)
    pred = probabilities.argmax(axis=1)
    triggered = core.apply_patch(images[labels != TARGET_CLASS], patch_size=4)
    attack = []
    with torch.no_grad():
        for s in range(0, len(triggered), 512):
            attack.append(model(_tensor_batch(triggered[s:s+512], device)).argmax(1).cpu().numpy())
    return {"balanced_accuracy": _balanced_accuracy(labels, pred),
            "asr": float(np.mean(np.concatenate(attack) == TARGET_CLASS))}


def _load_nonformal_weight() -> tuple[dict[str, torch.Tensor], Path, str, str]:
    receipt = json.loads(WEIGHT_RECEIPT.read_text(encoding="utf-8"))
    entries = [m for m in receipt.get("extraction", {}).get("members", []) if m.get("name") == "resnet18_28_1.pth"]
    if len(entries) != 1:
        raise RuntimeError("nonformal smoke requires exactly one acquired resnet18_28_1.pth member")
    path = PAPER14_ROOT.parent / entries[0]["local_path"]
    receipt_hash, weight_hash = sha256_file(WEIGHT_RECEIPT), sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if sorted(payload) != ["net"]:
        raise RuntimeError("unexpected nonformal smoke initializer structure")
    model = architecture.PathMNISTResNet18()
    model.load_state_dict(payload["net"], strict=True)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return initial, path, receipt_hash, weight_hash


def _smoke_fit(images: np.ndarray, labels: np.ndarray, initial: dict[str, torch.Tensor], *,
               device: torch.device, checkpoint: Path, resume_from: Path | None = None,
               stop_epoch: int = 5, protocol_sha: str, data_hash: str, weight_hash: str) -> dict[str, torch.Tensor]:
    seed = NONFORMAL_SMOKE_SEED
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.mps.manual_seed(seed)
    model = architecture.PathMNISTResNet18(); model.load_state_dict(initial, strict=True); model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(.9, .999), eps=1e-8, weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    start = 1
    if resume_from is not None:
        state = torch.load(resume_from, map_location="cpu", weights_only=False)
        expected_contract = {"seed": seed, "protocol_sha256": protocol_sha, "dataset_sha256": data_hash,
                             "training_data_sha256": data_hash, "weight_sha256": weight_hash,
                             "initial_state_sha256": core.state_fingerprint(initial), "device": "mps", "epochs": 5}
        if state.get("contract") != expected_contract or int(state.get("epoch", -1)) != 2:
            raise RuntimeError("nonformal smoke resume checkpoint contract mismatch")
        model.load_state_dict(state["state_dict"], strict=True); optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["sampler_generator"])
        random.setstate(state["rng_state"]["python"]); np.random.set_state(state["rng_state"]["numpy"])
        torch.set_rng_state(state["rng_state"]["torch_cpu"]); torch.mps.set_rng_state(state["rng_state"]["torch_mps"])
        start = 3
    dataset = torch.utils.data.TensorDataset(core._tensor_images(images), torch.as_tensor(labels, dtype=torch.long))
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, generator=generator)
    contract = {"seed": seed, "protocol_sha256": protocol_sha, "dataset_sha256": data_hash,
                "training_data_sha256": data_hash, "weight_sha256": weight_hash,
                "initial_state_sha256": core.state_fingerprint(initial), "device": "mps", "epochs": 5}
    history = []
    if resume_from is not None:
        history = list(state["history"])
    for epoch in range(start, stop_epoch + 1):
        model.train(); loss_sum = 0.0; seen = 0
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32).div_(255.0); yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True); loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward(); optimizer.step(); loss_sum += float(loss.detach().cpu()) * len(xb); seen += len(xb)
        if seen != len(images):
            raise RuntimeError("nonformal smoke epoch omitted fixed training rows")
        history.append({"epoch": epoch, "loss": loss_sum / seen, "rows": seen})
        payload = {"contract": contract, "epoch": epoch,
                   "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                   "optimizer": optimizer.state_dict(), "sampler_generator": generator.get_state(), "history": history,
                   "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(), "torch_mps": torch.mps.get_rng_state()}}
        tmp = checkpoint.with_name(checkpoint.name + ".tmp"); checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, tmp); os.replace(tmp, checkpoint)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def execute_real_input_smoke(dataset: Path, output: Path) -> dict[str, Any]:
    """Run a small real-input MPS, nonconfirmation resume/reload smoke test."""
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("smoke output directory must be empty or new")
    device = _mps_device()
    protocol_sha = sha256_file(PROTOCOL_PATH)
    data_hash = sha256_file(dataset)
    data = _load_train_validation(dataset)
    # Fixed nonformal subset; neither sample selection nor initialization uses 103-107.
    train_idx = np.arange(64, dtype=np.int64)
    val_idx = np.arange(64, dtype=np.int64)
    train_x, train_y = data["train"]["images"][train_idx], data["train"]["labels"][train_idx]
    val_x, val_y = data["val"]["images"][val_idx], data["val"]["labels"][val_idx]
    train_data_hash = hashlib.sha256(train_x.tobytes() + train_y.tobytes()).hexdigest()
    initial, weight_path, receipt_hash, weight_hash = _load_nonformal_weight()
    output.mkdir(parents=True, exist_ok=True)
    continuous_path = output / "continuous_epoch5.pt"
    interrupted_path = output / "interrupted_epoch2.pt"
    resumed_path = output / "resumed_epoch5.pt"
    continuous = _smoke_fit(train_x, train_y, initial, device=device, checkpoint=continuous_path,
                            stop_epoch=5, protocol_sha=protocol_sha, data_hash=train_data_hash, weight_hash=weight_hash)
    # Epoch 2 checkpoint, then instantiate from CPU initializer and continue exactly through epoch 5.
    _smoke_fit(train_x, train_y, initial, device=device, checkpoint=interrupted_path, stop_epoch=2,
               protocol_sha=protocol_sha, data_hash=train_data_hash, weight_hash=weight_hash)
    resumed = _smoke_fit(train_x, train_y, initial, device=device, checkpoint=resumed_path,
                         resume_from=interrupted_path, stop_epoch=5, protocol_sha=protocol_sha,
                         data_hash=train_data_hash, weight_hash=weight_hash)
    bit_equal = set(continuous) == set(resumed) and all(torch.equal(continuous[k], resumed[k]) for k in continuous)
    if not bit_equal:
        raise RuntimeError("continuous and epoch-2-resumed smoke states differ")
    # Reload the resumed state and compare same-input feature and metric replay on MPS.
    reloaded = architecture.PathMNISTResNet18(); reloaded.load_state_dict(resumed, strict=True); reloaded.to(device); reloaded.eval()
    val_features = _hidden(reloaded, val_x, device)
    with torch.no_grad():
        logits = torch.cat([reloaded(_tensor_batch(val_x[s:s+32], device)).cpu() for s in range(0, len(val_x), 32)]).numpy()
    accuracy = float(np.mean(logits.argmax(1) == val_y))
    feature_path = output / "feature_metric_replay.npz"
    tmp_feature = feature_path.with_name(feature_path.name + ".tmp.npz")
    np.savez_compressed(tmp_feature, validation_images=val_x, validation_labels=val_y,
                        features=val_features, logits=logits)
    os.replace(tmp_feature, feature_path)
    state_paths = {"continuous": continuous_path, "epoch2": interrupted_path, "resumed": resumed_path}
    hashes = {key: sha256_file(path) for key, path in state_paths.items()}
    weights_receipt_hash = sha256_file(WEIGHT_RECEIPT)
    receipt = {
        "receipt_id": "pathmnist_real_input_mps_smoke_2026-10-01_v1",
        "status": "PENDING_INDEPENDENT_VERIFICATION", "formal_seed_opened": False,
        "confirmation_seed_files_opened": False,
        "nonformal_seed": NONFORMAL_SMOKE_SEED, "device": str(device), "torch": torch.__version__,
        "dtype": "float32", "feature_dimension": 512, "epochs": 5, "resume_epoch": 2,
        "dataset_sha256": data_hash, "protocol_sha256": protocol_sha,
        "runner_sha256": sha256_file(Path(__file__)), "core_sha256": sha256_file(Path(core.__file__)),
        "architecture_sha256": sha256_file(Path(architecture.__file__)),
        "weight_receipt_sha256": weights_receipt_hash, "weight_sha256": weight_hash,
        "initial_state_sha256": core.state_fingerprint(initial),
        "weight_path": str(weight_path), "train_index_sha256": hashlib.sha256(train_idx.tobytes()).hexdigest(),
        "validation_index_sha256": hashlib.sha256(val_idx.tobytes()).hexdigest(),
        "training_data_sha256": train_data_hash, "continuous_resumed_state_sha256": core.state_fingerprint(resumed),
        "continuous_state_sha256": core.state_fingerprint(continuous), "bit_identical_resume": bit_equal,
        "checkpoint_files": {key: value.name for key, value in state_paths.items()},
        "checkpoint_sha256": hashes, "feature_file": feature_path.name, "feature_file_sha256": sha256_file(feature_path),
        "validation_accuracy": accuracy, "validation_row_count": len(val_y),
    }
    atomic_json(output / "smoke_receipt.json", receipt)
    from verify_pathmnist_official_weight_detector_confirmation import verify_smoke_receipt
    verification = verify_smoke_receipt(output / "smoke_receipt.json", dataset)
    atomic_json(output / "smoke_verification.json", verification)
    if verification.get("status") != "PASS_NONFORMAL_REAL_INPUT_MPS":
        raise RuntimeError(f"independent MPS smoke verification failed: {verification.get('failures')}")
    receipt["status"] = "PASS_NONFORMAL_REAL_INPUT_MPS"
    atomic_json(output / "smoke_receipt.json", receipt)
    return verification


def execute_formal(dataset: Path, output: Path, smoke_receipt: Path, *, resume: bool = False,
                   ack: str | None = None) -> dict[str, Any]:
    """Formal executor. Never call from readiness checks; gated on an MPS real-input smoke receipt."""
    if ack != FORMAL_ACK:
        raise RuntimeError(f"formal execution requires explicit --ack {FORMAL_ACK}")
    if output.exists() and any(output.iterdir()) and not resume:
        raise RuntimeError(f"refusing nonempty output directory: {output}; pass --resume for exact checkpoint resume")
    if output.exists() and resume and (output / "manifest.json").is_file():
        existing_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if existing_manifest.get("status") == "COMPLETE_FORMAL":
            raise RuntimeError("completed formal output cannot be resumed")
    from verify_pathmnist_official_weight_detector_confirmation import verify_smoke_receipt
    verification = verify_smoke_receipt(smoke_receipt, dataset)
    recorded_verification = smoke_receipt.parent / "smoke_verification.json"
    if verification.get("status") != "PASS_NONFORMAL_REAL_INPUT_MPS" or not recorded_verification.is_file():
        raise RuntimeError("required real-input MPS smoke receipt lacks an independent passing verifier result")
    saved_verification = json.loads(recorded_verification.read_text(encoding="utf-8"))
    if saved_verification != verification:
        raise RuntimeError("required real-input MPS resume/reload/feature/metric smoke receipt did not pass")
    device = _mps_device()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_seed_map(tuple(protocol["seed_plan"]["formal_confirmation_seeds"]),
                      {int(k): v for k, v in protocol["seed_plan"]["official_weight_mapping"].items()})
    protocol_sha = sha256_file(PROTOCOL_PATH)
    frozen = _frozen_hashes(dataset, protocol_sha)
    acquisition = json.loads(WEIGHT_RECEIPT.read_text(encoding="utf-8"))
    data = _load_train_validation(dataset)
    if len(data["train"]["labels"]) != 89996 or len(data["val"]["labels"]) != 10004:
        raise RuntimeError("official PathMNIST split sizes mismatch")
    ref_idx, cal_idx = _select_validation_indices(data["val"]["labels"])
    index_hashes = {"reference": hashlib.sha256(ref_idx.tobytes()).hexdigest(), "calibration": hashlib.sha256(cal_idx.tobytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=True)
    rows, seed_receipts = [], []
    manifest = {"experiment": "pathmnist_official_weight_detector_confirmation_v1", "formal": True,
                "status": "RUNNING", "protocol_sha256": protocol_sha, "frozen_hashes": frozen,
                "device": str(device), "torch": torch.__version__, "dtype": "float32", "epochs": EPOCHS,
                "validation_index_hashes": index_hashes, "created_unix": time.time(),
                "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                "torch_threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
                "command_line": sys.argv}
    atomic_json(output / "manifest.json", manifest)
    for seed in CONFIRMATION_SEEDS:
        per_sample: list[dict[str, Any]] = []
        start = time.time()
        weight_path = resolve_weight(acquisition, seed)
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
        if sorted(checkpoint) != ["net"]:
            raise RuntimeError("official checkpoint must contain only net")
        initial = {k: v.detach().cpu().clone() for k, v in checkpoint["net"].items()}
        cpu_model = architecture.PathMNISTResNet18()
        cpu_model.load_state_dict(initial, strict=True)
        initial = {k: v.detach().cpu().clone() for k, v in cpu_model.state_dict().items()}
        initial_hash = core.state_fingerprint(initial)
        candidate_pool = build_suspect_candidate_pool(data["train"]["images"], data["train"]["labels"], seed=seed)
        selected = candidate_pool["selected_indices"]
        suspect_x, suspect_y = candidate_pool["images"], candidate_pool["assigned_labels"]
        clean_x, clean_y = data["train"]["images"], data["train"]["labels"]
        if not all(torch.equal(initial[k], cpu_model.state_dict()[k].cpu()) for k in initial):
            raise RuntimeError("paired CPU initial states are not bit-identical")
        order_seed = 20261001 + seed
        weight_sha = frozen["official_weight_sha256_by_seed"][str(seed)]
        clean_data_sha = hashlib.sha256(clean_x.tobytes() + clean_y.tobytes()).hexdigest()
        suspect_data_sha = hashlib.sha256(suspect_x.tobytes() + suspect_y.tobytes()).hexdigest()
        clean_model, clean_train = _train_member(clean_x, clean_y, initial, seed=seed, condition="clean", device=device, order_seed=order_seed,
            output=output, checkpoint_path=output / "resume" / f"seed_{seed}_clean_latest.pt", resume=resume,
            dataset_sha256=frozen["dataset_sha256"], weight_sha256=weight_sha, training_data_sha256=clean_data_sha)
        suspect_model, suspect_train = _train_member(suspect_x, suspect_y, initial, seed=seed, condition="suspect", device=device, order_seed=order_seed,
            output=output, checkpoint_path=output / "resume" / f"seed_{seed}_suspect_latest.pt", resume=resume,
            dataset_sha256=frozen["dataset_sha256"], weight_sha256=weight_sha, training_data_sha256=suspect_data_sha)
        # Evaluation/test is first opened after both fixed epoch-5 members finish.
        test = _load_test(dataset)
        clean_eval = _test_metrics(clean_model, test["images"], test["labels"], device)
        suspect_eval = _test_metrics(suspect_model, test["images"], test["labels"], device)
        # Both models represent the same realized poisoned candidate pool.
        clean_candidate_input, suspect_candidate_input = shared_candidate_feature_inputs(suspect_x)
        train_features_clean = _hidden(clean_model, clean_candidate_input, device)
        train_features_suspect = _hidden(suspect_model, suspect_candidate_input, device)
        ref_images, cal_images = data["val"]["images"][ref_idx], data["val"]["images"][cal_idx]
        clean_ref = _hidden(clean_model, ref_images, device); suspect_ref = _hidden(suspect_model, ref_images, device)
        clean_cal = _hidden(clean_model, cal_images, device); suspect_cal = _hidden(suspect_model, cal_images, device)
        # Primary and secondary use the same raw-space helper. Truth is deliberately joined below.
        assigned = suspect_y
        methods = {}
        for name, ref, cal, feats in (("primary_cf_cmd_u", suspect_ref, suspect_cal, train_features_suspect),
                                      ("paired_drift_cf_cmd_u", suspect_ref-clean_ref, suspect_cal-clean_cal, train_features_suspect-train_features_clean)):
            score = cf_cmd_u_unwhitened(ref, cal, feats, assigned, seed=seed)
            methods[name] = score
        cent = target_centroid_distance(suspect_ref, suspect_cal, train_features_suspect, assigned)
        methods["centroid"] = cent
        import backdoorbench_feature_adapter as adapter
        # Baselines receive only features and assigned labels. SCAn uses full clean validation / labels.
        all_val_clean = _hidden(clean_model, data["val"]["images"], device)
        scan = adapter.scan(all_val_clean, data["val"]["labels"], train_features_suspect, assigned, n_classes=9, seed=seed)
        spectral = adapter.spectral_signatures(train_features_suspect, assigned, percentile=85.0)
        methods["scan"] = {"scores": scan.scores, "flags": scan.suspicious_mask, "receipt": scan.receipt}
        methods["spectral"] = {"scores": spectral.scores, "flags": spectral.suspicious_mask, "receipt": spectral.receipt}
        feature_path = output / "features" / f"seed_{seed}.npz"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_features = feature_path.with_name(feature_path.name + ".tmp.npz")
        candidate_image_sha = candidate_pool_image_hash(suspect_x)
        np.savez_compressed(tmp_features, clean_candidate=train_features_clean, suspect_candidate=train_features_suspect,
                            clean_reference=clean_ref, suspect_reference=suspect_ref, clean_calibration=clean_cal,
                            suspect_calibration=suspect_cal, all_clean_validation=all_val_clean,
                            validation_labels=data["val"]["labels"], assigned_labels=assigned, original_labels=data["train"]["labels"],
                            candidate_image_sha256=np.asarray(candidate_image_sha))
        os.replace(tmp_features, feature_path)
        # Poison truth is joined only after every detector has fixed its scores and flags.
        truth = np.zeros(89996, dtype=bool); truth[selected] = True
        for method, result in methods.items():
            flags = np.asarray(result["flags"], dtype=bool)
            counts = confusion_counts(flags, truth, assigned)
            rows.append({"seed": seed, "method": method, **counts})
            per_sample.extend({"seed": seed, "method": method, "view": "suspect_hidden" if method != "paired_drift_cf_cmd_u" else "paired_drift",
                "original_index": i, "original_label": int(data["train"]["labels"][i]), "assigned_label": int(assigned[i]),
                "score": float(result["scores"][i]) if np.isfinite(result["scores"][i]) else "",
                "p_value": float(result["p_values"][i]) if "p_values" in result else "",
                "flag": bool(flags[i]), "is_poison": bool(truth[i])} for i in range(89996))
        seed_receipts.append({"seed": seed, "weight": str(weight_path), "weight_sha256": frozen["official_weight_sha256_by_seed"][str(seed)],
            "initial_state_sha256": initial_hash, "clean_training": clean_train, "suspect_training": suspect_train,
            "clean_test": clean_eval, "suspect_test": suspect_eval, "eligible": clean_eval["balanced_accuracy"] >= .75 and suspect_eval["balanced_accuracy"] >= .75 and suspect_eval["asr"] >= .8,
            "poison_indices_sha256": hashlib.sha256(selected.tobytes()).hexdigest(), "feature_file": str(feature_path),
            "candidate_image_sha256": candidate_image_sha,
            "candidate_feature_source": CANDIDATE_FEATURE_SOURCE,
            "feature_file_sha256": sha256_file(feature_path),
            "detectors": {k: v["receipt"] for k,v in methods.items()}, "elapsed_seconds": time.time()-start})
        for label, model in (("clean", clean_model), ("suspect", suspect_model)):
            model_path = output / "models" / f"seed_{seed}_{label}_epoch5.pt"
            tmp_model = model_path.with_name(model_path.name + ".tmp")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, tmp_model)
            os.replace(tmp_model, model_path)
            seed_receipts[-1][f"{label}_model_file"] = str(model_path)
            seed_receipts[-1][f"{label}_model_file_sha256"] = sha256_file(model_path)
        atomic_csv(output / f"scores_seed_{seed}.csv", per_sample)
        atomic_csv(output / "detector_metrics.partial.csv", rows)
        atomic_json(output / "seed_receipts.partial.json", seed_receipts)
    manifest["status"] = "COMPLETE_FORMAL"
    manifest["completed_seeds"] = list(CONFIRMATION_SEEDS)
    eligible = {int(row["seed"]): bool(row["eligible"]) for row in seed_receipts}
    manifest["victim_eligibility_by_seed"] = {str(k): v for k, v in eligible.items()}
    manifest["all_pairs_eligible"] = all(eligible.get(seed, False) for seed in CONFIRMATION_SEEDS)
    manifest["detector_summaries"] = {}
    for method in ("primary_cf_cmd_u", "paired_drift_cf_cmd_u", "centroid", "scan", "spectral"):
        block_rows = [row for row in rows if row["method"] == method]
        counts = {key: int(sum(int(row[key]) for row in block_rows)) for key in
                  ("tp", "n_poison", "fp_all_clean", "n_all_clean", "fp_target_clean", "n_target_clean")}
        summary = {"per_seed": block_rows, "micro_pooled_counts": counts,
                   "pooled_tpr": counts["tp"] / counts["n_poison"],
                   "pooled_all_clean_fpr": counts["fp_all_clean"] / counts["n_all_clean"],
                   "pooled_target_clean_fpr": counts["fp_target_clean"] / counts["n_target_clean"]}
        if all(eligible.get(seed, False) for seed in CONFIRMATION_SEEDS):
            boot_rng = np.random.default_rng(20261001)
            replicates = np.empty((20_000, 3), dtype=np.float64)
            for b in range(len(replicates)):
                sample = [block_rows[i] for i in boot_rng.integers(0, 5, size=5)]
                d = {key: sum(int(r[key]) for r in sample) for key in counts}
                replicates[b] = (d["tp"] / d["n_poison"], d["fp_all_clean"] / d["n_all_clean"],
                                 d["fp_target_clean"] / d["n_target_clean"])
            summary["seed_block_bootstrap_95pct_ci"] = {
                name: [float(v) for v in np.quantile(replicates[:, i], [.025, .975])]
                for i, name in enumerate(("pooled_tpr", "pooled_all_clean_fpr", "pooled_target_clean_fpr"))}
            summary["bootstrap_replicates"] = 20_000
            summary["bootstrap_seed"] = 20261001
        manifest["detector_summaries"][method] = summary
    primary = manifest["detector_summaries"]["primary_cf_cmd_u"]
    manifest["primary_point_gate_pass"] = bool(manifest["all_pairs_eligible"] and
        primary["pooled_tpr"] >= .90 and primary["pooled_all_clean_fpr"] <= .05)
    manifest["completed_unix"] = time.time()
    merged_tmp = output / "scores.csv.tmp"
    with merged_tmp.open("wb") as merged:
        for index, seed in enumerate(CONFIRMATION_SEEDS):
            source = output / f"scores_seed_{seed}.csv"
            with source.open("rb") as handle:
                if index:
                    handle.readline()  # one header per final CSV
                shutil.copyfileobj(handle, merged)
    os.replace(merged_tmp, output / "scores.csv")
    atomic_csv(output / "detector_metrics.csv", rows)
    atomic_json(output / "seed_receipts.json", seed_receipts)
    atomic_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-dry-run",
        action="store_true",
        help="run only synthetic detector/readiness checks; no real data or weights are opened",
    )
    parser.add_argument(
        "--formal",
        action="store_true",
        help="run the fixed MPS formal protocol (requires the real-input smoke receipt)",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-receipt", type=Path, default=DEFAULT_SMOKE_OUTPUT / "smoke_receipt.json")
    parser.add_argument("--resume", action="store_true", help="resume only from a contract-matching atomic MPS checkpoint")
    parser.add_argument("--ack", type=str, default=None, help=f"required to open formal seeds: {FORMAL_ACK}")
    parser.add_argument("--real-input-mps-smoke", action="store_true", help="run small nonformal real-input MPS resume smoke")
    args = parser.parse_args()
    if args.real_input_mps_smoke:
        smoke_output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_SMOKE_OUTPUT
        print(json.dumps(execute_real_input_smoke(args.dataset, smoke_output), indent=2, sort_keys=True))
        return
    if args.formal:
        print(json.dumps(execute_formal(args.dataset, args.output, args.smoke_receipt, resume=args.resume, ack=args.ack), indent=2, sort_keys=True))
        return
    if not args.synthetic_dry_run:
        parser.error("select --synthetic-dry-run; formal execution is not enabled")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_seed_map(
        tuple(protocol["seed_plan"]["formal_confirmation_seeds"]),
        {int(seed): name for seed, name in protocol["seed_plan"]["official_weight_mapping"].items()},
    )
    print(json.dumps(synthetic_dry_run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
