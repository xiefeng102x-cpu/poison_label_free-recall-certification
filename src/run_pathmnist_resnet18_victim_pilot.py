#!/usr/bin/env python3
"""Full-data PathMNIST official-style ResNet-18 victim-feasibility pilot.

This runner trains suspect victims only.  It does not train a clean pair and
does not run any detector.  Formal execution is frozen to seeds 93/94 and 100
epochs.  ``--dry-run`` and ``--max-epochs`` produce explicitly non-formal
artifacts that can never satisfy the expansion gate.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

import pathmnist_external_validation as core
import pathmnist_official_resnet18 as architecture


PAPER14_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/pathmnist_resnet18_victim_pilot_protocol_v1.json"
)
DEFAULT_DATASET = Path("/private/tmp/paper14-data/pathmnist.npz")
DEFAULT_OUTPUT = PAPER14_ROOT / "results/pathmnist_resnet18_victim_pilot_v1"
DEFAULT_DRY_OUTPUT = PAPER14_ROOT / "results/pathmnist_resnet18_victim_pilot_dry_run"
FORMAL_SEEDS = (93, 94)
N_CLASSES = 9
TARGET_CLASS = 8
PATCH_SIZE = 4
POISON_FRACTION = 0.05
BATCH_SIZE = 128
FORMAL_EPOCHS = 100
MILESTONES = (50, 75)
LEARNING_RATE = 1e-3
SCHEDULER_GAMMA = 0.1
SELECTION_METRIC = "clean_validation_macro_ovr_auc"


@dataclass(frozen=True)
class ExecutionMode:
    formal: bool
    target_epochs: int
    max_batches_per_epoch: int | None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def cpu_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def poison_count_for_training(training_count: int) -> int:
    if training_count <= 0:
        raise ValueError("training_count must be positive")
    return int(round(POISON_FRACTION * training_count))


def validate_execution_mode(
    *,
    dry_run: bool,
    max_epochs: int | None,
    dry_run_batches: int,
    seeds: tuple[int, ...],
) -> ExecutionMode:
    if dry_run:
        target_epochs = 1 if max_epochs is None else int(max_epochs)
        if not 1 <= target_epochs <= 3:
            raise ValueError("dry-run max_epochs must lie in [1, 3]")
        if not 1 <= dry_run_batches <= 5:
            raise ValueError("dry_run_batches must lie in [1, 5]")
        if not seeds or any(seed not in FORMAL_SEEDS for seed in seeds):
            raise ValueError(f"dry-run seeds must be drawn from {FORMAL_SEEDS}")
        return ExecutionMode(False, target_epochs, dry_run_batches)
    if seeds != FORMAL_SEEDS:
        raise ValueError(f"formal execution is frozen to seeds {FORMAL_SEEDS}")
    if max_epochs not in (None, FORMAL_EPOCHS):
        raise ValueError("--max-epochs may shorten execution only with --dry-run")
    return ExecutionMode(True, FORMAL_EPOCHS, None)


def observed_source_hashes(dataset: Path) -> dict[str, str]:
    return {
        "dataset_sha256": core.sha256_file(dataset),
        "core_source_sha256": core.sha256_file(Path(core.__file__)),
        "model_source_sha256": core.sha256_file(Path(architecture.__file__)),
        "runner_source_sha256": core.sha256_file(Path(__file__)),
    }


def runtime_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "torch_threads": torch.get_num_threads(),
        "mps_available": bool(torch.backends.mps.is_available()),
    }


def validate_frozen_inputs(dataset: Path) -> tuple[dict[str, Any], dict[str, str]]:
    protocol = read_json(PROTOCOL)
    expected = protocol["frozen_inputs"]
    observed = observed_source_hashes(dataset)
    mismatches = {
        key: {"expected": expected.get(key), "actual": value}
        for key, value in observed.items()
        if expected.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"frozen input mismatch: {mismatches}")
    return protocol, observed


def balanced_dry_indices(labels: np.ndarray, per_class: int = 8) -> np.ndarray:
    truth = np.asarray(labels, dtype=np.int64)
    parts = []
    for label in range(N_CLASSES):
        indices = np.flatnonzero(truth == label)
        if len(indices) < per_class:
            raise ValueError("dry-run evaluation lacks a required class")
        parts.append(indices[:per_class])
    return np.sort(np.concatenate(parts)).astype(np.int64, copy=False)


def predict_probabilities(
    model: nn.Module, images: np.ndarray, batch_size: int = 512
) -> np.ndarray:
    model.eval()
    tensor = core._tensor_images(images)
    rows = []
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size].float().div_(255.0)
            rows.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(rows).astype(np.float64, copy=False)


def macro_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.shape != (len(truth), N_CLASSES):
        raise ValueError("probabilities must align with labels and all nine classes")
    if set(np.unique(truth)) != set(range(N_CLASSES)):
        raise ValueError("macro AUC requires all nine classes")
    return float(
        roc_auc_score(
            truth,
            scores,
            labels=np.arange(N_CLASSES),
            multi_class="ovr",
            average="macro",
        )
    )


def validation_metrics(
    model: nn.Module, images: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    probabilities = predict_probabilities(model, images)
    predictions = probabilities.argmax(axis=1)
    return {
        "validation_macro_auc": macro_auc(labels, probabilities),
        "validation_balanced_accuracy": core.balanced_accuracy(labels, predictions),
    }


def capture_rng_state(shuffle_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "shuffle_generator": shuffle_generator.get_state(),
    }


def restore_rng_state(payload: dict[str, Any], shuffle_generator: torch.Generator) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    shuffle_generator.set_state(payload["shuffle_generator"])


def checkpoint_contract(
    *,
    protocol_sha256: str,
    source_hashes: dict[str, str],
    seed: int,
    mode: ExecutionMode,
) -> dict[str, Any]:
    return {
        "protocol_sha256": protocol_sha256,
        **source_hashes,
        "seed": int(seed),
        "formal": mode.formal,
        "target_epochs": mode.target_epochs,
        "max_batches_per_epoch": mode.max_batches_per_epoch,
        "selection_metric": SELECTION_METRIC,
        "runtime_environment": runtime_environment(),
    }


def load_resume_checkpoint(
    path: Path,
    *,
    expected_contract: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    shuffle_generator: torch.Generator,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actual_contract = payload.get("contract")
    if actual_contract != expected_contract:
        raise RuntimeError(
            f"resume checkpoint contract mismatch: expected {expected_contract}, "
            f"observed {actual_contract}"
        )
    required = (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "epoch",
        "rng_state",
        "best_validation_macro_auc",
        "best_epoch",
        "best_model_state",
        "history",
        "initial_state_sha256",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"resume checkpoint is missing fields: {missing}")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    restore_rng_state(payload["rng_state"], shuffle_generator)
    return payload


def train_seed(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    validation_images: np.ndarray,
    validation_labels: np.ndarray,
    *,
    seed: int,
    mode: ExecutionMode,
    output: Path,
    protocol_sha256: str,
    source_hashes: dict[str, str],
    resume: bool,
) -> tuple[architecture.PathMNISTResNet18, dict[str, Any], dict[str, torch.Tensor]]:
    """Train using train/validation only; test data are intentionally absent."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = architecture.make_model(seed)
    initial_state_sha256 = core.state_fingerprint(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(MILESTONES), gamma=SCHEDULER_GAMMA
    )
    shuffle_generator = torch.Generator().manual_seed(20261001 + seed)
    contract = checkpoint_contract(
        protocol_sha256=protocol_sha256,
        source_hashes=source_hashes,
        seed=seed,
        mode=mode,
    )
    latest = output / "resume" / f"seed_{seed}_latest.pt"
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_auc = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    if latest.exists():
        if not resume:
            raise RuntimeError(f"resume checkpoint exists; pass --resume: {latest}")
        restored = load_resume_checkpoint(
            latest,
            expected_contract=contract,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            shuffle_generator=shuffle_generator,
        )
        start_epoch = int(restored["epoch"]) + 1
        history = list(restored["history"])
        best_auc = float(restored["best_validation_macro_auc"])
        best_epoch = int(restored["best_epoch"])
        best_state = cpu_state(restored["best_model_state"])
        initial_state_sha256 = str(restored["initial_state_sha256"])
    elif resume:
        print(f"seed={seed}: no checkpoint found; starting from epoch 1", flush=True)

    dataset = torch.utils.data.TensorDataset(
        core._tensor_images(train_images),
        torch.as_tensor(train_labels, dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=shuffle_generator,
    )
    started = time.time()
    for epoch in range(start_epoch, mode.target_epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        learning_rate = float(optimizer.param_groups[0]["lr"])
        for batch_index, (xb, yb) in enumerate(loader):
            if (
                mode.max_batches_per_epoch is not None
                and batch_index >= mode.max_batches_per_epoch
            ):
                break
            xb = xb.float().div_(255.0)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(xb)
            count += len(xb)
        if count == 0:
            raise RuntimeError("training epoch processed no samples")
        metrics = validation_metrics(model, validation_images, validation_labels)
        candidate_auc = metrics["validation_macro_auc"]
        if candidate_auc > best_auc:
            best_auc = candidate_auc
            best_epoch = epoch
            best_state = cpu_state(model.state_dict())
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_cross_entropy": loss_sum / count,
                "training_observations": count,
                **metrics,
            }
        )
        if best_state is None:
            raise RuntimeError("best validation state was not captured")
        payload = {
            "contract": contract,
            "model_state": cpu_state(model.state_dict()),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "rng_state": capture_rng_state(shuffle_generator),
            "best_validation_macro_auc": best_auc,
            "best_epoch": best_epoch,
            "best_model_state": best_state,
            "history": history,
            "initial_state_sha256": initial_state_sha256,
        }
        atomic_torch_save(latest, payload)
        print(
            f"seed={seed} epoch={epoch}/{mode.target_epochs} "
            f"val_auc={candidate_auc:.4f} best={best_auc:.4f}",
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("training produced no best-validation checkpoint")
    return model, {
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": core.state_fingerprint(model.state_dict()),
        "best_state_sha256": core.state_fingerprint(best_state),
        "best_epoch": best_epoch,
        "best_validation_macro_auc": best_auc,
        "epochs_completed": len(history),
        "elapsed_seconds_this_invocation": time.time() - started,
        "selection_metric": SELECTION_METRIC,
        "selection_data_split": "clean_validation_only",
        "test_data_used_during_training_or_selection": False,
        "history": history,
    }, best_state


def test_metrics(
    model: nn.Module, test_images: np.ndarray, test_labels: np.ndarray
) -> dict[str, Any]:
    probabilities = predict_probabilities(model, test_images)
    predictions = probabilities.argmax(axis=1)
    non_target = test_labels != TARGET_CLASS
    triggered = core.apply_patch(test_images[non_target], patch_size=PATCH_SIZE)
    attack_prediction = predict_probabilities(model, triggered).argmax(axis=1)
    balanced = core.balanced_accuracy(test_labels, predictions)
    attack_success = float(np.mean(attack_prediction == TARGET_CLASS))
    return {
        "test_balanced_accuracy": balanced,
        "test_macro_auc": macro_auc(test_labels, probabilities),
        "attack_success_rate": attack_success,
        "eligible": bool(balanced >= 0.75 and attack_success >= 0.80),
        "per_class_recall": core.per_class_recall(test_labels, predictions),
    }


def save_model_artifact(
    path: Path,
    state: dict[str, torch.Tensor],
    *,
    seed: int,
    role: str,
    protocol_sha256: str,
    source_hashes: dict[str, str],
    formal: bool,
    epoch: int,
    selection_metric: str | None,
) -> dict[str, Any]:
    payload = {
        "architecture": "pathmnist_official_style_resnet18_cifar_stem_hidden512",
        "state_dict": cpu_state(state),
        "seed": seed,
        "role": role,
        "formal": formal,
        "epoch": epoch,
        "selection_metric": selection_metric,
        "protocol_sha256": protocol_sha256,
        **source_hashes,
    }
    atomic_torch_save(path, payload)
    return {
        "path": str(path),
        "file_sha256": core.sha256_file(path),
        "state_sha256": core.state_fingerprint(state),
        "epoch": epoch,
        "selection_metric": selection_metric,
    }


def compute_gate(rows: list[dict[str, Any]], *, formal: bool) -> dict[str, bool]:
    seeds_complete = (
        len(rows) == len(FORMAL_SEEDS)
        and {int(row["seed"]) for row in rows} == set(FORMAL_SEEDS)
    )
    both_eligible = seeds_complete and all(bool(row["eligible"]) for row in rows)
    return {
        "formal_execution": formal,
        "both_frozen_seeds_complete": seeds_complete,
        "both_best_validation_victims_eligible": both_eligible,
        "authorize_followup": bool(formal and seeds_complete and both_eligible),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    seeds = tuple(int(seed) for seed in args.seeds)
    mode = validate_execution_mode(
        dry_run=args.dry_run,
        max_epochs=args.max_epochs,
        dry_run_batches=args.dry_run_batches,
        seeds=seeds,
    )
    protocol, source_hashes = validate_frozen_inputs(args.dataset)
    protocol_sha256 = core.sha256_file(PROTOCOL)
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise RuntimeError(
            f"output is nonempty; use a new directory or pass --resume: {args.output}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    data = core.load_pathmnist(args.dataset)
    poison_count = poison_count_for_training(len(data["train"]["labels"]))
    if poison_count != int(protocol["attack"]["poison_count"]):
        raise RuntimeError("protocol poison count does not match the full training set")

    if mode.formal:
        validation_images = data["val"]["images"]
        validation_labels = data["val"]["labels"]
        test_images = data["test"]["images"]
        test_labels = data["test"]["labels"]
    else:
        val_indices = balanced_dry_indices(data["val"]["labels"])
        test_indices = balanced_dry_indices(data["test"]["labels"])
        validation_images = data["val"]["images"][val_indices]
        validation_labels = data["val"]["labels"][val_indices]
        test_images = data["test"]["images"][test_indices]
        test_labels = data["test"]["labels"][test_indices]

    rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for seed in seeds:
        seed_started = time.time()
        pool = core.build_poisoned_training(
            data["train"]["images"],
            data["train"]["labels"],
            seed=seed,
            poison_count=poison_count,
            target_class=TARGET_CLASS,
            patch_size=PATCH_SIZE,
        )
        model, training, best_state = train_seed(
            pool["images"],
            pool["assigned_labels"],
            validation_images,
            validation_labels,
            seed=seed,
            mode=mode,
            output=args.output,
            protocol_sha256=protocol_sha256,
            source_hashes=source_hashes,
            resume=args.resume,
        )
        best_model = architecture.make_model(seed)
        best_model.load_state_dict(best_state)
        best_metrics = test_metrics(best_model, test_images, test_labels)
        final_metrics = test_metrics(model, test_images, test_labels)
        best_path = args.output / "models" / f"seed_{seed}_best_validation_auc.pt"
        final_label = "epoch100" if mode.formal else f"dry_epoch{mode.target_epochs}"
        final_path = args.output / "models" / f"seed_{seed}_{final_label}.pt"
        best_artifact = save_model_artifact(
            best_path,
            best_state,
            seed=seed,
            role="primary_best_clean_validation_macro_auc",
            protocol_sha256=protocol_sha256,
            source_hashes=source_hashes,
            formal=mode.formal,
            epoch=int(training["best_epoch"]),
            selection_metric=SELECTION_METRIC,
        )
        final_artifact = save_model_artifact(
            final_path,
            model.state_dict(),
            seed=seed,
            role="secondary_fixed_final_epoch",
            protocol_sha256=protocol_sha256,
            source_hashes=source_hashes,
            formal=mode.formal,
            epoch=mode.target_epochs,
            selection_metric=None,
        )
        rows.append(
            {
                "seed": seed,
                "formal": mode.formal,
                "primary_checkpoint": "best_clean_validation_macro_auc",
                "best_epoch": training["best_epoch"],
                "best_validation_macro_auc": training["best_validation_macro_auc"],
                **best_metrics,
                "secondary_epoch": mode.target_epochs,
                "secondary_test_balanced_accuracy": final_metrics[
                    "test_balanced_accuracy"
                ],
                "secondary_attack_success_rate": final_metrics[
                    "attack_success_rate"
                ],
            }
        )
        receipts.append(
            {
                "seed": seed,
                "formal": mode.formal,
                "elapsed_seconds": time.time() - seed_started,
                "pool": pool["receipt"],
                "training": training,
                "primary_checkpoint": best_artifact,
                "secondary_checkpoint": final_artifact,
            }
        )
        write_csv(args.output / "victim_metrics.partial.csv", rows)
        atomic_write_json(args.output / "seed_receipts.partial.json", receipts)

    gate = compute_gate(rows, formal=mode.formal)
    manifest = {
        "experiment": "pathmnist_full_data_official_resnet18_victim_pilot_v1",
        "status": "FORMAL_COMPLETE" if mode.formal else "DRY_RUN_NONFORMAL",
        "formal": mode.formal,
        "dry_run": not mode.formal,
        "seeds": list(seeds),
        "target_epochs": mode.target_epochs,
        "max_batches_per_epoch": mode.max_batches_per_epoch,
        "dataset": str(args.dataset),
        "protocol": str(PROTOCOL),
        "protocol_sha256": protocol_sha256,
        "source_hashes": source_hashes,
        "runtime_environment": runtime_environment(),
        "training_count": int(len(data["train"]["labels"])),
        "validation_count_used": int(len(validation_labels)),
        "test_count_used": int(len(test_labels)),
        "poison_count": poison_count,
        "poison_fraction": POISON_FRACTION,
        "target_class": TARGET_CLASS,
        "patch_size": PATCH_SIZE,
        "selection_metric": SELECTION_METRIC,
        "test_data_used_during_training_or_selection": False,
        "gate": gate,
    }
    write_csv(args.output / "victim_metrics.csv", rows)
    atomic_write_json(args.output / "seed_receipts.json", receipts)
    atomic_write_json(args.output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--dry-run-batches", type=int, default=2)
    args = parser.parse_args()
    if args.seeds is None:
        args.seeds = list(FORMAL_SEEDS if not args.dry_run else FORMAL_SEEDS[:1])
    if args.output is None:
        args.output = DEFAULT_OUTPUT if not args.dry_run else DEFAULT_DRY_OUTPUT
    manifest = run(args)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
