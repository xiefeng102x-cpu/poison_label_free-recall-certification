#!/usr/bin/env python3
"""P6-A multiclass BloodMNIST applicability pilot for BackdoorBench detectors.

The protocol is frozen in ``multiclass_backdoorbench_pilot_protocol.json``.
Training uses the official eight-class splits.  Selected non-target training
rows are replaced in place by patched, target-labelled rows so every detector
can be evaluated against an auditable original-index mapping.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import backdoorbench_feature_adapter as official
import bloodmnist_attack_families as attack_ops


PAPER14_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/multiclass_backdoorbench_pilot_protocol.json"
)
DEFAULT_OUTPUT = PAPER14_ROOT / "results/bloodmnist_multiclass_backdoorbench_pilot_v1"
DEFAULT_CONFIRMATION_OUTPUT = (
    PAPER14_ROOT / "results/bloodmnist_multiclass_backdoorbench_confirmation_v1"
)
DEVELOPMENT_SEEDS = (72, 73)
CONFIRMATION_SEEDS = (74, 75, 76, 77, 78)
TARGET_CLASS = 6
N_CLASSES = 8
POISON_COUNT = 600
METHODS = (
    "backdoorbench_spectral",
    "backdoorbench_scan",
    "backdoorbench_spectre_oracle_target",
    "backdoorbench_spectre_uniform_5pct",
)
VIEWS = ("suspect_hidden64", "clean_to_suspect_hidden64_drift")


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


def load_bloodmnist(dataset_root: Path) -> dict[str, dict[str, np.ndarray]]:
    path = dataset_root / "bloodmnist.npz"
    with np.load(path, allow_pickle=False) as payload:
        output = {}
        for split in ("train", "val", "test"):
            images = payload[f"{split}_images"].astype(np.float32) / 255.0
            labels = payload[f"{split}_labels"].reshape(-1).astype(np.int64)
            if images.shape != (len(labels), 28, 28, 3):
                raise ValueError(f"unexpected {split} image shape {images.shape}")
            if set(np.unique(labels)) != set(range(N_CLASSES)):
                raise ValueError(f"{split} does not contain all eight classes")
            output[split] = {"images": images, "labels": labels}
    return output


def build_poisoned_training(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    poison_count: int = POISON_COUNT,
    target_class: int = TARGET_CLASS,
) -> dict[str, Any]:
    """Replace selected non-target rows in place and retain index provenance."""

    x = np.asarray(images, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 4 or x.shape[1:] != (28, 28, 3) or len(x) != len(y):
        raise ValueError("images/labels must be aligned BloodMNIST arrays")
    eligible = np.flatnonzero(y != target_class)
    if poison_count <= 0 or poison_count >= len(eligible):
        raise ValueError("poison_count must lie within the non-target population")
    rng = np.random.default_rng(20260929 + int(seed))
    selected = np.sort(rng.choice(eligible, size=poison_count, replace=False))
    poisoned_x = x.copy()
    poisoned_y = y.copy()
    patched, trigger_receipt = attack_ops.apply_badnets_patch(
        poisoned_x[selected], patch_size=3, value=1.0
    )
    poisoned_x[selected] = patched
    original_labels = y[selected].copy()
    poisoned_y[selected] = target_class
    truth = np.zeros(len(y), dtype=bool)
    truth[selected] = True
    kind = np.full(len(y), "clean", dtype="<U6")
    kind[selected] = "poison"
    return {
        "images": poisoned_x,
        "assigned_labels": poisoned_y,
        "original_labels": y.copy(),
        "is_poison": truth,
        "sample_kind": kind,
        "original_index": np.arange(len(y), dtype=np.int64),
        "selected_indices": selected,
        "selected_original_labels": original_labels,
        "receipt": {
            "selection_seed": 20260929 + int(seed),
            "poison_count": int(poison_count),
            "target_class": int(target_class),
            "selected_indices_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
            "selected_original_class_counts": {
                str(label): int(np.sum(original_labels == label))
                for label in np.unique(original_labels)
            },
            "trigger": trigger_receipt.to_dict(),
            "construction": "in_place_replacement",
        },
    }


def nchw(images: np.ndarray) -> np.ndarray:
    values = np.asarray(images, dtype=np.float32)
    if values.ndim != 4 or values.shape[1:] != (28, 28, 3):
        raise ValueError("expected BloodMNIST NHWC images")
    return values.transpose(0, 3, 1, 2).copy()


def channel_stats(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = nchw(images)
    mean = values.mean(axis=(0, 2, 3)).astype(np.float32)
    std = values.std(axis=(0, 2, 3)).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


class MulticlassBloodCNN(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("channel_mean", torch.as_tensor(mean.reshape(1, 3, 1, 1)))
        self.register_buffer("channel_std", torch.as_tensor(std.reshape(1, 3, 1, 1)))
        self.features_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.projector = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU()
        )
        self.head = nn.Linear(64, N_CLASSES)

    def hidden(self, values: torch.Tensor) -> torch.Tensor:
        normalized = (values - self.channel_mean) / self.channel_std
        return self.projector(self.features_net(normalized))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(values))


def make_model(seed: int, mean: np.ndarray, std: np.ndarray) -> MulticlassBloodCNN:
    torch.manual_seed(int(seed))
    return MulticlassBloodCNN(mean, std)


def initial_state_fingerprint(seed: int, mean: np.ndarray, std: np.ndarray) -> str:
    return state_fingerprint(make_model(seed, mean, std).state_dict())


def predict(
    model: MulticlassBloodCNN, images: np.ndarray, batch_size: int = 256
) -> np.ndarray:
    model.eval()
    values = nchw(images)
    rows = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(values[start : start + batch_size])
            rows.append(model(batch).argmax(1).cpu().numpy())
    return np.concatenate(rows).astype(np.int64, copy=False)


def hidden_features(
    model: MulticlassBloodCNN, images: np.ndarray, batch_size: int = 256
) -> np.ndarray:
    model.eval()
    values = nchw(images)
    rows = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(values[start : start + batch_size])
            rows.append(model.hidden(batch).cpu().numpy().astype(np.float64))
    return np.concatenate(rows)


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(predictions, dtype=np.int64)
    if truth.shape != pred.shape or truth.ndim != 1:
        raise ValueError("labels and predictions must be aligned vectors")
    recalls = [float(np.mean(pred[truth == label] == label)) for label in np.unique(truth)]
    return float(np.mean(recalls))


def per_class_recall(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    truth = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(predictions, dtype=np.int64)
    return {
        str(label): float(np.mean(pred[truth == label] == label))
        for label in range(N_CLASSES)
    }


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    seed: int,
    mean: np.ndarray,
    std: np.ndarray,
    maximum_epochs: int = 10,
    batch_size: int = 64,
    patience: int = 2,
    minimum_delta: float = 1e-4,
) -> tuple[MulticlassBloodCNN, dict[str, Any]]:
    model = make_model(seed, mean, std)
    initial_hash = state_fingerprint(model.state_dict())
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(nchw(train_x), dtype=torch.float32),
        torch.as_tensor(train_y, dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(20260929 + int(seed)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_score = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    history = []
    started = time.time()
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        loss_sum = 0.0
        observations = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(xb)
            observations += len(xb)
        val_prediction = predict(model, val_x)
        val_balanced = balanced_accuracy(val_y, val_prediction)
        history.append(
            {
                "epoch": epoch,
                "train_cross_entropy": loss_sum / observations,
                "validation_balanced_accuracy": val_balanced,
            }
        )
        if val_balanced > best_score + minimum_delta:
            best_score = val_balanced
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
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


def save_checkpoint(
    path: Path,
    model: MulticlassBloodCNN,
    training: dict[str, Any],
    *,
    seed: int,
    role: str,
    mean: np.ndarray,
    std: np.ndarray,
    protocol_sha256: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": "multiclass_blood_cnn_hidden64",
            "seed": int(seed),
            "role": role,
            "mean": mean,
            "std": std,
            "protocol_sha256": protocol_sha256,
            "training": training,
        },
        path,
    )
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "state_sha256": state_fingerprint(model.state_dict()),
    }


def load_matching_checkpoint(
    path: Path,
    *,
    seed: int,
    role: str,
    mean: np.ndarray,
    std: np.ndarray,
    protocol_sha256: str,
) -> tuple[MulticlassBloodCNN, dict[str, Any]] | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = (int(seed), role, protocol_sha256)
    actual = (payload.get("seed"), payload.get("role"), payload.get("protocol_sha256"))
    if actual != expected:
        return None
    if not np.array_equal(payload.get("mean"), mean) or not np.array_equal(
        payload.get("std"), std
    ):
        return None
    model = MulticlassBloodCNN(mean, std)
    model.load_state_dict(payload["state_dict"])
    return model, payload["training"]


def get_or_train(
    checkpoint: Path,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    seed: int,
    role: str,
    mean: np.ndarray,
    std: np.ndarray,
    protocol_sha256: str,
) -> tuple[MulticlassBloodCNN, dict[str, Any], dict[str, Any], bool]:
    loaded = load_matching_checkpoint(
        checkpoint,
        seed=seed,
        role=role,
        mean=mean,
        std=std,
        protocol_sha256=protocol_sha256,
    )
    reused = loaded is not None
    if loaded is None:
        model, training = train_model(
            train_x, train_y, val_x, val_y, seed=seed, mean=mean, std=std
        )
        artifact = save_checkpoint(
            checkpoint,
            model,
            training,
            seed=seed,
            role=role,
            mean=mean,
            std=std,
            protocol_sha256=protocol_sha256,
        )
    else:
        model, training = loaded
        artifact = {
            "path": str(checkpoint),
            "file_sha256": sha256_file(checkpoint),
            "state_sha256": state_fingerprint(model.state_dict()),
        }
    return model, training, artifact, reused


def detector_budgets(candidate_labels: np.ndarray) -> dict[str, dict[int, int]]:
    labels = np.asarray(candidate_labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=N_CLASSES)
    return {
        "oracle_target": {
            label: POISON_COUNT if label == TARGET_CLASS else 0
            for label in range(N_CLASSES)
        },
        "uniform_5pct": {
            label: int(np.floor(0.05 * counts[label])) for label in range(N_CLASSES)
        },
    }


def run_methods(
    reference: np.ndarray,
    reference_y: np.ndarray,
    candidates: np.ndarray,
    candidate_y: np.ndarray,
    *,
    scan_seed: int,
) -> dict[str, official.FeatureAdapterResult]:
    budgets = detector_budgets(candidate_y)
    return {
        "backdoorbench_spectral": official.spectral_signatures(
            candidates, candidate_y, percentile=85.0
        ),
        "backdoorbench_scan": official.scan(
            reference,
            reference_y,
            candidates,
            candidate_y,
            n_classes=N_CLASSES,
            seed=scan_seed,
        ),
        "backdoorbench_spectre_oracle_target": official.spectre(
            candidates, candidate_y, budgets["oracle_target"], pca_dimension=16
        ),
        "backdoorbench_spectre_uniform_5pct": official.spectre(
            candidates, candidate_y, budgets["uniform_5pct"], pca_dimension=16
        ),
    }


def evaluate_flags(
    flags: np.ndarray,
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
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tpr": tp / (tp + fn),
        "fpr_all_clean": fp / (fp + tn),
        "target_clean_fpr": float(np.mean(flagged[target_clean])),
        "flagged_count": int(np.sum(flagged)),
        "assigned_class_clean_fpr_json": json.dumps(
            {
                str(label): (
                    float(np.mean(flagged[clean & (assigned == label)]))
                    if np.any(clean & (assigned == label))
                    else None
                )
                for label in range(N_CLASSES)
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def validate_stage_seeds(stage: str, seeds: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    actual = tuple(int(seed) for seed in seeds)
    expected = DEVELOPMENT_SEEDS if stage == "development" else CONFIRMATION_SEEDS
    if actual != expected:
        raise ValueError(f"{stage} is frozen to seeds {expected}")
    return actual


def confirmation_is_authorized(development_root: Path) -> bool:
    manifest = development_root / "manifest.json"
    if not manifest.exists():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    gate = payload.get("gate", {})
    required_gate_fields = (
        "all_development_victims_eligible",
        "all_classes_have_required_samples",
        "scan_finite_and_nondegenerate_both_views",
        "sample_mappings_and_confusions_replay",
        "at_least_one_seed_within_runtime_budget",
        "expand_to_confirmation",
    )
    return bool(
        payload.get("experiment")
        == "bloodmnist_multiclass_backdoorbench_pilot_v1"
        and payload.get("stage") == "development"
        and payload.get("protocol_sha256") == sha256_file(PROTOCOL)
        and payload.get("seeds") == list(DEVELOPMENT_SEEDS)
        and isinstance(gate, dict)
        and all(gate.get(field) is True for field in required_gate_fields)
    )


def replay_sample_mappings_and_confusions(
    sample_rows: list[dict[str, Any]],
    config_rows: list[dict[str, Any]],
    candidate_count: int,
) -> bool:
    """Independently replay mappings and confusion counts from sample rows."""

    expected = {
        (int(row["seed"]), str(row["view"]), str(row["method"])): row
        for row in config_rows
    }
    if len(expected) != len(config_rows):
        return False
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for row in sample_rows:
        key = (int(row["seed"]), str(row["view"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)
    if set(grouped) != set(expected):
        return False

    mapping_by_seed: dict[int, tuple[np.ndarray, ...]] = {}
    for key, rows in grouped.items():
        if len(rows) != candidate_count:
            return False
        ordered = sorted(rows, key=lambda row: int(row["sample_index"]))
        sample_index = np.asarray(
            [int(row["sample_index"]) for row in ordered], dtype=np.int64
        )
        original_index = np.asarray(
            [int(row["original_index"]) for row in ordered], dtype=np.int64
        )
        original_label = np.asarray(
            [int(row["original_label"]) for row in ordered], dtype=np.int64
        )
        assigned_label = np.asarray(
            [int(row["assigned_label"]) for row in ordered], dtype=np.int64
        )
        poison = np.asarray([bool(int(row["is_poison"])) for row in ordered])
        flagged = np.asarray([bool(int(row["flagged"])) for row in ordered])
        identity = np.arange(candidate_count, dtype=np.int64)
        if not (
            np.array_equal(sample_index, identity)
            and np.array_equal(original_index, identity)
            and int(np.sum(poison)) == POISON_COUNT
            and np.all(original_label[poison] != TARGET_CLASS)
            and np.all(assigned_label[poison] == TARGET_CLASS)
            and np.array_equal(assigned_label[~poison], original_label[~poison])
        ):
            return False

        mapping = (original_index, original_label, assigned_label, poison)
        prior = mapping_by_seed.setdefault(key[0], mapping)
        if any(not np.array_equal(left, right) for left, right in zip(prior, mapping)):
            return False

        clean = ~poison
        replayed = {
            "tp": int(np.sum(flagged & poison)),
            "fn": int(np.sum(~flagged & poison)),
            "fp": int(np.sum(flagged & clean)),
            "tn": int(np.sum(~flagged & clean)),
        }
        reported = expected[key]
        if any(replayed[name] != int(reported[name]) for name in replayed):
            return False
        if replayed["tp"] + replayed["fn"] != POISON_COUNT:
            return False
        if replayed["fp"] + replayed["tn"] != candidate_count - POISON_COUNT:
            return False
    return True


def run(args: argparse.Namespace) -> dict[str, Any]:
    seeds = validate_stage_seeds(args.stage, args.seeds)
    if args.stage == "confirmation" and not confirmation_is_authorized(
        args.development_root
    ):
        raise RuntimeError("development expansion gate has not passed")
    if (
        args.stage == "confirmation"
        and args.output.resolve() == args.development_root.resolve()
    ):
        raise RuntimeError("confirmation output must not overwrite development artifacts")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_hash = sha256_file(PROTOCOL)
    data = load_bloodmnist(args.dataset_root)
    mean, std = channel_stats(data["train"]["images"])
    all_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    victim_rows: list[dict[str, Any]] = []
    seed_receipts: list[dict[str, Any]] = []
    scan_checks: list[dict[str, Any]] = []
    started_all = time.time()

    for seed in seeds:
        seed_started = time.time()
        print(f"seed={seed}: constructing in-place BadNets pool", flush=True)
        pool = build_poisoned_training(
            data["train"]["images"], data["train"]["labels"], seed=seed
        )
        models: dict[str, MulticlassBloodCNN] = {}
        training_receipts = {}
        checkpoint_receipts = {}
        reused = {}
        for role, train_x, train_y in (
            ("clean", data["train"]["images"], data["train"]["labels"]),
            ("suspect", pool["images"], pool["assigned_labels"]),
        ):
            print(f"seed={seed}: {role} CNN", flush=True)
            checkpoint = args.output / "models" / f"seed_{seed}_{role}.pt"
            model, training, artifact, was_reused = get_or_train(
                checkpoint,
                train_x,
                train_y,
                data["val"]["images"],
                data["val"]["labels"],
                seed=seed,
                role=role,
                mean=mean,
                std=std,
                protocol_sha256=protocol_hash,
            )
            models[role] = model
            training_receipts[role] = training
            checkpoint_receipts[role] = artifact
            reused[role] = was_reused

        if (
            training_receipts["clean"]["initial_state_sha256"]
            != training_receipts["suspect"]["initial_state_sha256"]
        ):
            raise RuntimeError("paired clean/suspect initialization mismatch")

        clean_prediction = predict(models["suspect"], data["test"]["images"])
        clean_balanced = balanced_accuracy(data["test"]["labels"], clean_prediction)
        non_target_test = data["test"]["labels"] != TARGET_CLASS
        triggered_test, _ = attack_ops.apply_badnets_patch(
            data["test"]["images"][non_target_test], patch_size=3, value=1.0
        )
        attack_prediction = predict(models["suspect"], triggered_test)
        asr = float(np.mean(attack_prediction == TARGET_CLASS))
        eligible = clean_balanced >= 0.80 and asr >= 0.80
        victim_rows.append(
            {
                "stage": args.stage,
                "seed": seed,
                "balanced_clean_accuracy": clean_balanced,
                "attack_success_rate": asr,
                "eligible": int(eligible),
                "per_class_recall_json": json.dumps(
                    per_class_recall(data["test"]["labels"], clean_prediction),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "clean_best_epoch": training_receipts["clean"]["best_epoch"],
                "suspect_best_epoch": training_receipts["suspect"]["best_epoch"],
            }
        )
        print(
            f"seed={seed}: balanced CA={clean_balanced:.3f}, ASR={asr:.3f}, "
            f"eligible={int(eligible)}",
            flush=True,
        )

        val_suspect = hidden_features(models["suspect"], data["val"]["images"])
        val_clean = hidden_features(models["clean"], data["val"]["images"])
        candidate_suspect = hidden_features(models["suspect"], pool["images"])
        candidate_clean = hidden_features(models["clean"], pool["images"])
        view_values = {
            "suspect_hidden64": (val_suspect, candidate_suspect),
            "clean_to_suspect_hidden64_drift": (
                val_suspect - val_clean,
                candidate_suspect - candidate_clean,
            ),
        }
        for view_index, view in enumerate(VIEWS):
            print(f"seed={seed}: detectors on {view}", flush=True)
            reference, candidates = view_values[view]
            outputs = run_methods(
                reference,
                data["val"]["labels"],
                candidates,
                pool["assigned_labels"],
                scan_seed=seed + 1000 * view_index,
            )
            scan_scores = outputs["backdoorbench_scan"].scores
            score_range = float(np.max(scan_scores) - np.min(scan_scores))
            scan_checks.append(
                {
                    "seed": seed,
                    "view": view,
                    "finite": bool(np.isfinite(scan_scores).all()),
                    "class_score_range": score_range,
                    "nondegenerate": bool(score_range > 1e-8),
                    "unique_class_score_count": int(len(np.unique(scan_scores))),
                }
            )
            for method, result in outputs.items():
                metrics = evaluate_flags(
                    result.suspicious_mask,
                    pool["is_poison"],
                    pool["assigned_labels"],
                    pool["original_labels"],
                )
                all_rows.append(
                    {
                        "stage": args.stage,
                        "seed": seed,
                        "view": view,
                        "method": method,
                        "eligible_victim": int(eligible),
                        "dimension": candidates.shape[1],
                        **metrics,
                        "receipt_json": json.dumps(
                            result.receipt, sort_keys=True, separators=(",", ":")
                        ),
                    }
                )
                for index in range(len(candidates)):
                    score = result.scores[index]
                    sample_rows.append(
                        {
                            "seed": seed,
                            "view": view,
                            "method": method,
                            "sample_index": index,
                            "original_index": int(pool["original_index"][index]),
                            "original_label": int(pool["original_labels"][index]),
                            "assigned_label": int(pool["assigned_labels"][index]),
                            "is_poison": int(pool["is_poison"][index]),
                            "flagged": int(result.suspicious_mask[index]),
                            "score": "" if not np.isfinite(score) else float(score),
                        }
                    )

        elapsed_seed = time.time() - seed_started
        seed_receipts.append(
            {
                "seed": seed,
                "elapsed_seconds": elapsed_seed,
                "within_runtime_budget": bool(elapsed_seed <= 600.0),
                "pool": pool["receipt"],
                "paired_initial_state_sha256": training_receipts["clean"][
                    "initial_state_sha256"
                ],
                "training": training_receipts,
                "checkpoints": checkpoint_receipts,
                "reused_checkpoints": reused,
                "candidate_count": int(len(pool["images"])),
                "reference_count": int(len(data["val"]["images"])),
                "assigned_class_counts": {
                    str(label): int(np.sum(pool["assigned_labels"] == label))
                    for label in range(N_CLASSES)
                },
            }
        )

    class_coverage_ok = all(
        np.all(np.bincount(data["val"]["labels"], minlength=N_CLASSES) >= 2)
        and np.all(
            np.bincount(
                build_poisoned_training(
                    data["train"]["images"], data["train"]["labels"], seed=seed
                )["assigned_labels"],
                minlength=N_CLASSES,
            )
            > 0
        )
        for seed in seeds
    )
    sample_mapping_and_confusions_replay = (
        all(
            receipt["candidate_count"] == len(data["train"]["images"])
            and receipt["pool"]["poison_count"] == POISON_COUNT
            for receipt in seed_receipts
        )
        and replay_sample_mappings_and_confusions(
            sample_rows, all_rows, len(data["train"]["images"])
        )
    )
    gate = {
        "all_development_victims_eligible": all(row["eligible"] for row in victim_rows),
        "all_classes_have_required_samples": bool(class_coverage_ok),
        "scan_finite_and_nondegenerate_both_views": all(
            row["finite"] and row["nondegenerate"] for row in scan_checks
        ),
        "sample_mappings_and_confusions_replay": bool(
            sample_mapping_and_confusions_replay
        ),
        "at_least_one_seed_within_runtime_budget": any(
            row["within_runtime_budget"] for row in seed_receipts
        ),
    }
    if args.stage == "development":
        gate["expand_to_confirmation"] = all(gate.values())
    else:
        gate["confirmation_complete"] = all(
            value for key, value in gate.items() if key != "all_development_victims_eligible"
        )

    write_csv(args.output / "victim_metrics.csv", victim_rows)
    write_csv(args.output / "per_config_metrics.csv", all_rows)
    write_csv(args.output / "per_sample_flags.csv", sample_rows)
    write_csv(args.output / "scan_applicability.csv", scan_checks)
    write_json(args.output / "seed_receipts.json", seed_receipts)
    manifest = {
        "experiment": "bloodmnist_multiclass_backdoorbench_pilot_v1",
        "stage": args.stage,
        "protocol": str(PROTOCOL),
        "protocol_sha256": protocol_hash,
        "runner_sha256": sha256_file(Path(__file__)),
        "dataset_npz_sha256": sha256_file(args.dataset_root / "bloodmnist.npz"),
        "upstream_commit": official.UPSTREAM_COMMIT,
        "seeds": list(seeds),
        "target_class": TARGET_CLASS,
        "poison_count": POISON_COUNT,
        "views": list(VIEWS),
        "view_access_assumptions": {
            "suspect_hidden64": "suspect-checkpoint features only",
            "clean_to_suspect_hidden64_drift": (
                "paired-clean-checkpoint oracle; counterfactual comparison view, "
                "not assumed available in ordinary deployment"
            ),
        },
        "methods": list(METHODS),
        "gate": gate,
        "elapsed_seconds": time.time() - started_all,
    }
    write_json(args.output / "manifest.json", manifest)
    lines = [
        f"# BloodMNIST eight-class BackdoorBench applicability ({args.stage})",
        "",
        f"Gate: **{'PASS' if all(gate.values()) else 'STOP'}**",
        "",
        "| seed | balanced CA | ASR | eligible |",
        "|---:|---:|---:|---:|",
    ]
    for row in victim_rows:
        lines.append(
            f"| {row['seed']} | {row['balanced_clean_accuracy']:.3f} | "
            f"{row['attack_success_rate']:.3f} | {row['eligible']} |"
        )
    lines.extend(
        [
            "",
            "| seed | view | method | TPR | all-clean FPR | target-clean FPR |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in all_rows:
        lines.append(
            f"| {row['seed']} | {row['view']} | {row['method']} | "
            f"{row['tpr']:.3f} | {row['fpr_all_clean']:.3f} | "
            f"{row['target_clean_fpr']:.3f} |"
        )
    lines.extend(
        [
            "",
            "SPECTRE oracle-target uses the true target class and true poison count; "
            "the uniform-5% variant does not use poison labels or the target class.",
            "The clean-to-suspect drift view is a paired-clean-checkpoint oracle "
            "counterfactual, not a generally deployable detector input.",
            "Detection performance was not used as an expansion gate.",
            "",
        ]
    )
    (args.output / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data")
    )
    parser.add_argument(
        "--stage", choices=("development", "confirmation"), default="development"
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--development-root", type=Path, default=DEFAULT_OUTPUT
    )
    args = parser.parse_args()
    if args.seeds is None:
        args.seeds = list(
            DEVELOPMENT_SEEDS if args.stage == "development" else CONFIRMATION_SEEDS
        )
    if args.output is None:
        args.output = (
            DEFAULT_OUTPUT
            if args.stage == "development"
            else DEFAULT_CONFIRMATION_OUTPUT
        )
    run(args)


if __name__ == "__main__":
    main()
