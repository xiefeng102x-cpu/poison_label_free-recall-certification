"""Core components for the frozen PathMNIST external-validation program."""

from __future__ import annotations

import copy
import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F


N_CLASSES = 9
TARGET_CLASS = 8
TRAIN_PER_CLASS = 3000
POISON_COUNT = 1350
PATCH_SIZE = 4
REFERENCE_COUNT = 512
CALIBRATION_COUNT = 512
ALPHA = 0.05


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        values = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def load_pathmnist(path: Path) -> dict[str, dict[str, np.ndarray]]:
    expected_counts = {"train": 89996, "val": 10004, "test": 7180}
    output: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as payload:
        for split, count in expected_counts.items():
            images = payload[f"{split}_images"]
            labels = payload[f"{split}_labels"].reshape(-1)
            if images.shape != (count, 28, 28, 3) or images.dtype != np.uint8:
                raise ValueError(f"unexpected {split} images: {images.shape} {images.dtype}")
            if labels.shape != (count,) or not np.issubdtype(labels.dtype, np.integer):
                raise ValueError(f"unexpected {split} labels: {labels.shape} {labels.dtype}")
            labels = labels.astype(np.int64, copy=False)
            if set(np.unique(labels)) != set(range(N_CLASSES)):
                raise ValueError(f"{split} does not contain exactly nine classes")
            output[split] = {"images": images, "labels": labels}
    return output


def channel_stats(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(images)
    if values.ndim != 4 or values.shape[1:] != (28, 28, 3):
        raise ValueError("expected PathMNIST NHWC images")
    total = np.zeros(3, dtype=np.float64)
    total_squared = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for start in range(0, len(values), 4096):
        batch = values[start : start + 4096].astype(np.float64) / 255.0
        total += batch.sum(axis=(0, 1, 2))
        total_squared += np.square(batch).sum(axis=(0, 1, 2))
        pixel_count += int(np.prod(batch.shape[:3]))
    mean = total / pixel_count
    second = total_squared / pixel_count
    variance = np.maximum(second - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_patch(images: np.ndarray, patch_size: int = PATCH_SIZE) -> np.ndarray:
    values = np.asarray(images)
    if values.ndim != 4 or values.shape[1:] != (28, 28, 3) or values.dtype != np.uint8:
        raise ValueError("patch input must be uint8 PathMNIST NHWC images")
    if patch_size < 1 or patch_size > 14:
        raise ValueError("patch_size must lie in [1, 14]")
    output = values.copy()
    output[:, -patch_size:, -patch_size:, :] = 255
    return output


def select_balanced_training_subset(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    per_class: int = TRAIN_PER_CLASS,
    seed: int = 20260930,
) -> dict[str, Any]:
    values = np.asarray(images)
    truth = np.asarray(labels, dtype=np.int64)
    if values.shape != (len(truth), 28, 28, 3) or values.dtype != np.uint8:
        raise ValueError("images and labels must be aligned PathMNIST arrays")
    if per_class < 1:
        raise ValueError("per_class must be positive")
    rng = np.random.default_rng(seed)
    parts = []
    for label in range(N_CLASSES):
        available = np.flatnonzero(truth == label)
        if len(available) < per_class:
            raise ValueError(f"class {label} has fewer than {per_class} rows")
        parts.append(np.sort(rng.choice(available, size=per_class, replace=False)))
    selected = np.sort(np.concatenate(parts)).astype(np.int64, copy=False)
    return {
        "images": values[selected],
        "labels": truth[selected],
        "original_indices": selected,
        "receipt": {
            "seed": int(seed),
            "per_class": int(per_class),
            "count": int(len(selected)),
            "selected_indices_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
        },
    }


def build_poisoned_training(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    poison_count: int = POISON_COUNT,
    target_class: int = TARGET_CLASS,
    patch_size: int = PATCH_SIZE,
    original_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    values = np.asarray(images)
    original = np.asarray(labels, dtype=np.int64)
    if values.shape != (len(original), 28, 28, 3) or values.dtype != np.uint8:
        raise ValueError("images and labels must be aligned PathMNIST arrays")
    eligible = np.flatnonzero(original != target_class)
    if poison_count <= 0 or poison_count >= len(eligible):
        raise ValueError("invalid poison_count")
    selection_seed = 20260930 + int(seed)
    rng = np.random.default_rng(selection_seed)
    selected = np.sort(rng.choice(eligible, size=poison_count, replace=False))
    poisoned_images = values.copy()
    poisoned_images[selected] = apply_patch(values[selected], patch_size)
    assigned = original.copy()
    assigned[selected] = target_class
    truth = np.zeros(len(original), dtype=bool)
    truth[selected] = True
    provenance = (
        np.arange(len(original), dtype=np.int64)
        if original_indices is None
        else np.asarray(original_indices, dtype=np.int64)
    )
    if provenance.shape != (len(original),) or len(np.unique(provenance)) != len(provenance):
        raise ValueError("original_indices must be a unique aligned vector")
    return {
        "images": poisoned_images,
        "assigned_labels": assigned,
        "original_labels": original.copy(),
        "is_poison": truth,
        "original_index": provenance.copy(),
        "selected_indices": selected,
        "receipt": {
            "selection_seed": selection_seed,
            "poison_count": int(poison_count),
            "target_class": int(target_class),
            "patch_size": int(patch_size),
            "patch_value_uint8": 255,
            "selected_indices_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
            "selected_original_class_counts": {
                str(label): int(np.sum(original[selected] == label))
                for label in range(N_CLASSES)
            },
            "construction": "in_place_replacement",
        },
    }


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        if stride != 1 or input_channels != output_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.bn1(self.conv1(values)), inplace=False)
        hidden = self.bn2(self.conv2(hidden))
        return F.relu(hidden + self.skip(values), inplace=False)


class PathResidualCNN(nn.Module):
    """A compute-bounded image-native residual network with a 64-D audit layer."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("channel_mean", torch.as_tensor(mean).reshape(1, 3, 1, 1))
        self.register_buffer("channel_std", torch.as_tensor(std).reshape(1, 3, 1, 1))
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.stage1 = ResidualBlock(16, 16)
        self.stage2 = ResidualBlock(16, 32, 2)
        self.stage3 = ResidualBlock(32, 64, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, N_CLASSES)

    def hidden(self, values: torch.Tensor) -> torch.Tensor:
        values = (values - self.channel_mean) / self.channel_std
        values = self.stem(values)
        values = self.stage1(values)
        values = self.stage2(values)
        values = self.stage3(values)
        return self.pool(values).flatten(1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(values))


def make_model(seed: int, mean: np.ndarray, std: np.ndarray) -> PathResidualCNN:
    torch.manual_seed(int(seed))
    return PathResidualCNN(mean, std)


def _tensor_images(images: np.ndarray) -> torch.Tensor:
    values = np.asarray(images)
    if values.ndim != 4 or values.shape[1:] != (28, 28, 3) or values.dtype != np.uint8:
        raise ValueError("expected uint8 PathMNIST NHWC images")
    return torch.from_numpy(values.transpose(0, 3, 1, 2))


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(predictions, dtype=np.int64)
    if truth.shape != pred.shape or truth.ndim != 1:
        raise ValueError("labels and predictions must be aligned vectors")
    return float(
        np.mean([np.mean(pred[truth == label] == label) for label in np.unique(truth)])
    )


def per_class_recall(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    truth = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(predictions, dtype=np.int64)
    return {
        str(label): float(np.mean(pred[truth == label] == label))
        for label in range(N_CLASSES)
    }


def predict(model: PathResidualCNN, images: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    tensor = _tensor_images(images)
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size].float().div_(255.0)
            rows.append(model(batch).argmax(1).cpu().numpy())
    return np.concatenate(rows).astype(np.int64, copy=False)


def hidden_features(
    model: PathResidualCNN, images: np.ndarray, batch_size: int = 512
) -> np.ndarray:
    model.eval()
    tensor = _tensor_images(images)
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size].float().div_(255.0)
            rows.append(model.hidden(batch).cpu().numpy().astype(np.float32))
    return np.concatenate(rows)


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    seed: int,
    mean: np.ndarray,
    std: np.ndarray,
    maximum_epochs: int = 8,
    batch_size: int = 512,
    patience: int = 2,
    minimum_delta: float = 1e-4,
) -> tuple[PathResidualCNN, dict[str, Any]]:
    model = make_model(seed, mean, std)
    initial_hash = state_fingerprint(model.state_dict())
    dataset = torch.utils.data.TensorDataset(
        _tensor_images(train_x), torch.as_tensor(train_y, dtype=torch.long)
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(20260930 + int(seed)),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    augmentation_generator = torch.Generator().manual_seed(20270930 + int(seed))
    best_score = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    started = time.time()
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb.float().div_(255.0)
            horizontal = torch.rand(len(xb), generator=augmentation_generator) < 0.5
            vertical = torch.rand(len(xb), generator=augmentation_generator) < 0.5
            xb[horizontal] = torch.flip(xb[horizontal], dims=(3,))
            xb[vertical] = torch.flip(xb[vertical], dims=(2,))
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(xb)
            count += len(xb)
        validation_prediction = predict(model, val_x)
        validation_balanced = balanced_accuracy(val_y, validation_prediction)
        history.append(
            {
                "epoch": epoch,
                "train_cross_entropy": loss_sum / count,
                "validation_balanced_accuracy": validation_balanced,
            }
        )
        if validation_balanced > best_score + minimum_delta:
            best_score = validation_balanced
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "initial_state_sha256": initial_hash,
        "best_state_sha256": state_fingerprint(best_state),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_balanced_accuracy": best_score,
        "elapsed_seconds": time.time() - started,
        "history": history,
    }


def _rank_pvalues(calibration: np.ndarray, query: np.ndarray) -> np.ndarray:
    calibration = np.asarray(calibration, dtype=np.float64).reshape(-1)
    query = np.asarray(query, dtype=np.float64).reshape(-1)
    if len(calibration) == 0 or not np.isfinite(calibration).all():
        raise ValueError("calibration scores must be finite and nonempty")
    ordered = np.sort(calibration)
    counts_greater_equal = len(ordered) - np.searchsorted(ordered, query, side="left")
    return (1.0 + counts_greater_equal) / (len(ordered) + 1.0)


def _whitening(reference: np.ndarray, ridge_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(reference, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("reference must be a nonempty matrix")
    mean = values.mean(axis=0)
    centered = values - mean
    covariance = centered.T @ centered / (len(values) - 1)
    mean_variance = max(float(np.trace(covariance) / values.shape[1]), 1e-12)
    ridge = ridge_fraction * mean_variance
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    transform = (eigenvectors / np.sqrt(np.maximum(eigenvalues, 0.0) + ridge)) @ eigenvectors.T
    return mean, transform, ridge


def cf_cmd_u(
    reference_features: np.ndarray,
    reference_labels: np.ndarray,
    candidate_features: np.ndarray,
    candidate_labels: np.ndarray,
    *,
    seed: int,
    target_class: int = TARGET_CLASS,
    alpha: float = ALPHA,
    reference_count: int = REFERENCE_COUNT,
    calibration_count: int = CALIBRATION_COUNT,
) -> dict[str, Any]:
    reference = np.asarray(reference_features, dtype=np.float64)
    candidates = np.asarray(candidate_features, dtype=np.float64)
    ref_y = np.asarray(reference_labels, dtype=np.int64)
    cand_y = np.asarray(candidate_labels, dtype=np.int64)
    if reference.ndim != 2 or candidates.ndim != 2 or reference.shape[1] != candidates.shape[1]:
        raise ValueError("reference and candidate features must be compatible matrices")
    if len(ref_y) != len(reference) or len(cand_y) != len(candidates):
        raise ValueError("labels must align with features")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    trusted_indices = np.flatnonzero(ref_y == target_class)
    if len(trusted_indices) < reference_count + calibration_count:
        raise ValueError("insufficient target-class trusted samples")
    rng_seed = 20271001 + int(seed)
    rng = np.random.default_rng(rng_seed)
    trusted_order = rng.permutation(trusted_indices)
    selected_reference = trusted_order[:reference_count]
    selected_calibration = trusted_order[reference_count : reference_count + calibration_count]
    mean, transform, ridge = _whitening(reference[selected_reference])
    whiten = lambda values: (np.asarray(values, dtype=np.float64) - mean) @ transform
    reference_w = whiten(reference[selected_reference])
    calibration_w = whiten(reference[selected_calibration])
    target_indices = np.flatnonzero(cand_y == target_class)
    if len(target_indices) < 4:
        raise ValueError("insufficient target-class candidates")
    shuffled = rng.permutation(target_indices)
    halves = np.array_split(shuffled, 2)
    scores = np.full(len(candidates), np.nan, dtype=np.float64)
    p_values = np.ones(len(candidates), dtype=np.float64)
    flags = np.zeros(len(candidates), dtype=bool)
    direction_norms: list[float] = []
    direction_cosine = np.nan
    directions: list[np.ndarray] = []
    reference_mean = reference_w.mean(axis=0)
    for scoring, fitting in ((halves[0], halves[1]), (halves[1], halves[0])):
        direction = whiten(candidates[fitting]).mean(axis=0) - reference_mean
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            raise RuntimeError("CF-CMD-U direction collapsed")
        direction /= norm
        directions.append(direction)
        direction_norms.append(norm)
        calibration_scores = calibration_w @ direction
        query_scores = whiten(candidates[scoring]) @ direction
        query_p = _rank_pvalues(calibration_scores, query_scores)
        scores[scoring] = query_scores
        p_values[scoring] = query_p
        flags[scoring] = query_p <= alpha
    if len(directions) == 2:
        direction_cosine = float(np.dot(directions[0], directions[1]))
    return {
        "suspicious_mask": flags,
        "scores": scores,
        "p_values": p_values,
        "receipt": {
            "method": "cf_cmd_u",
            "target_class": int(target_class),
            "alpha": float(alpha),
            "rng_seed": rng_seed,
            "reference_count": int(reference_count),
            "calibration_count": int(calibration_count),
            "candidate_target_count": int(len(target_indices)),
            "reference_indices_sha256": hashlib.sha256(
                np.asarray(selected_reference, dtype=np.int64).tobytes()
            ).hexdigest(),
            "calibration_indices_sha256": hashlib.sha256(
                np.asarray(selected_calibration, dtype=np.int64).tobytes()
            ).hexdigest(),
            "ridge_fraction": 0.1,
            "ridge": ridge,
            "direction_norms": direction_norms,
            "direction_cosine": direction_cosine,
            "crossfit": "two disjoint target-candidate halves; each half scored by the opposite-half direction",
            "calibration_scope": "clean validation target-class rank calibration; training-member exchangeability is not claimed",
        },
    }


def evaluate_flags(
    flags: np.ndarray,
    scores: np.ndarray,
    truth: np.ndarray,
    assigned_labels: np.ndarray,
    original_labels: np.ndarray,
) -> dict[str, Any]:
    flagged = np.asarray(flags, dtype=bool)
    poison = np.asarray(truth, dtype=bool)
    assigned = np.asarray(assigned_labels, dtype=np.int64)
    original = np.asarray(original_labels, dtype=np.int64)
    if not (flagged.shape == poison.shape == assigned.shape == original.shape):
        raise ValueError("flag, truth and label arrays must align")
    clean = ~poison
    tp = int(np.sum(flagged & poison))
    fn = int(np.sum(~flagged & poison))
    fp = int(np.sum(flagged & clean))
    tn = int(np.sum(~flagged & clean))
    target_clean = clean & (original == TARGET_CLASS)
    finite = np.isfinite(scores)
    auc_mask = finite & (assigned == TARGET_CLASS)
    auc = None
    if len(np.unique(poison[auc_mask])) == 2:
        auc = float(roc_auc_score(poison[auc_mask].astype(np.int64), scores[auc_mask]))
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tpr": tp / (tp + fn),
        "fpr_all_clean": fp / (fp + tn),
        "target_clean_fpr": float(np.mean(flagged[target_clean])),
        "flagged_count": int(flagged.sum()),
        "target_class_score_auc": auc,
    }
