#!/usr/bin/env python3
"""Official-weight warm-start PathMNIST victim-feasibility runner.

This package fine-tunes one suspect victim per frozen seed from a prespecified
official MedMNIST ResNet-18 checkpoint.  It never trains a clean pair and never
runs a detector.  Formal execution is fixed to five epochs and seeds 93 then
94.  Seed 94 is omitted only when the completed seed-93 victim fails the frozen
CA/ASR gate, because the two-seed conjunction is then already impossible.

``--dry-run`` artifacts are permanently non-formal and cannot authorize any
follow-up experiment.
"""

from __future__ import annotations

import argparse
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
    / "results/top_journal_program_v1/"
    "pathmnist_official_weight_warmstart_protocol_v1.json"
)
RECEIPT = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/"
    "pathmnist_official_weights_acquisition_receipt.json"
)
DEFAULT_DATASET = Path("/private/tmp/paper14-data/pathmnist.npz")
DEFAULT_OUTPUT = PAPER14_ROOT / "results/pathmnist_official_weight_warmstart_v1"
DEFAULT_DRY_OUTPUT = (
    PAPER14_ROOT / "results/pathmnist_official_weight_warmstart_dry_run"
)
FORMAL_SEEDS = (93, 94)
WEIGHT_FILENAMES = {
    93: "resnet18_28_1.pth",
    94: "resnet18_28_2.pth",
}
N_CLASSES = 9
TARGET_CLASS = 8
PATCH_SIZE = 4
POISON_FRACTION = 0.05
POISON_COUNT = 4_500
BATCH_SIZE = 128
FORMAL_EPOCHS = 5
LEARNING_RATE = 3e-4
PRIMARY_CHECKPOINT = "fixed_final_epoch_5"
VALIDATION_ROLE = "diagnostic_only_no_model_selection"


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


def states_bit_identical(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    if set(left) != set(right):
        return False
    return all(
        left[name].dtype == right[name].dtype
        and left[name].shape == right[name].shape
        and torch.equal(left[name].detach().cpu(), right[name].detach().cpu())
        for name in left
    )


def validate_execution_mode(
    *,
    dry_run: bool,
    max_epochs: int | None,
    dry_run_batches: int,
    seeds: tuple[int, ...],
) -> ExecutionMode:
    if dry_run:
        target_epochs = 1 if max_epochs is None else int(max_epochs)
        if not 1 <= target_epochs <= FORMAL_EPOCHS:
            raise ValueError(f"dry-run max_epochs must lie in [1, {FORMAL_EPOCHS}]")
        if not 1 <= dry_run_batches <= 5:
            raise ValueError("dry_run_batches must lie in [1, 5]")
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("dry-run seeds must be a nonempty unique sequence")
        if any(seed not in FORMAL_SEEDS for seed in seeds):
            raise ValueError(f"dry-run seeds must be drawn from {FORMAL_SEEDS}")
        if tuple(sorted(seeds, key=FORMAL_SEEDS.index)) != seeds:
            raise ValueError("seeds must preserve the frozen order 93 then 94")
        return ExecutionMode(False, target_epochs, dry_run_batches)
    if seeds != FORMAL_SEEDS:
        raise ValueError(f"formal execution is frozen to ordered seeds {FORMAL_SEEDS}")
    if max_epochs not in (None, FORMAL_EPOCHS):
        raise ValueError("--max-epochs may shorten execution only with --dry-run")
    return ExecutionMode(True, FORMAL_EPOCHS, None)


def poison_count_for_training(training_count: int) -> int:
    if training_count != 89_996:
        raise ValueError("official warm-start protocol requires all 89,996 train rows")
    count = int(round(POISON_FRACTION * training_count))
    if count != POISON_COUNT:
        raise RuntimeError("frozen poison-count calculation changed")
    return count


def weight_path_from_receipt(receipt: dict[str, Any], seed: int) -> Path:
    expected_name = WEIGHT_FILENAMES[int(seed)]
    matches = [
        member
        for member in receipt["extraction"]["members"]
        if member["name"] == expected_name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"receipt does not uniquely identify {expected_name}")
    return PAPER14_ROOT.parent / matches[0]["local_path"]


def observed_frozen_hashes(dataset: Path) -> dict[str, Any]:
    receipt = read_json(RECEIPT)
    weights = {
        str(seed): core.sha256_file(weight_path_from_receipt(receipt, seed))
        for seed in FORMAL_SEEDS
    }
    return {
        "dataset_sha256": core.sha256_file(dataset),
        "core_source_sha256": core.sha256_file(Path(core.__file__)),
        "model_source_sha256": core.sha256_file(Path(architecture.__file__)),
        "runner_source_sha256": core.sha256_file(Path(__file__)),
        "receipt_sha256": core.sha256_file(RECEIPT),
        "official_weight_sha256_by_seed": weights,
    }


def validate_frozen_inputs(
    dataset: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = read_json(PROTOCOL)
    receipt = read_json(RECEIPT)
    observed = observed_frozen_hashes(dataset)
    expected = protocol["frozen_inputs"]
    mismatches = {
        key: {"expected": expected.get(key), "actual": value}
        for key, value in observed.items()
        if expected.get(key) != value
    }
    expected_mapping = {str(k): v for k, v in WEIGHT_FILENAMES.items()}
    if receipt.get("selection", {}).get("intended_attack_seed_mapping") != expected_mapping:
        mismatches["receipt_seed_mapping"] = {
            "expected": expected_mapping,
            "actual": receipt.get("selection", {}).get("intended_attack_seed_mapping"),
        }
    if mismatches:
        raise RuntimeError(f"frozen input mismatch: {mismatches}")
    return protocol, receipt, observed


def runtime_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
    }


def load_official_initial_state(
    seed: int, receipt: dict[str, Any]
) -> tuple[architecture.PathMNISTResNet18, dict[str, torch.Tensor], Path]:
    path = weight_path_from_receipt(receipt, seed)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if sorted(checkpoint) != ["net"]:
        raise RuntimeError(
            f"official checkpoint {path.name} has unexpected keys {sorted(checkpoint)}"
        )
    official_state = cpu_state(checkpoint["net"])
    model = architecture.PathMNISTResNet18()
    model.load_state_dict(official_state, strict=True)
    if not states_bit_identical(model.state_dict(), official_state):
        raise RuntimeError("initial model state is not bit-identical to official state")
    return model, official_state, path


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


def validation_diagnostics(
    model: nn.Module, images: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    probabilities = predict_probabilities(model, images)
    predictions = probabilities.argmax(axis=1)
    return {
        "validation_macro_auc": macro_auc(labels, probabilities),
        "validation_balanced_accuracy": core.balanced_accuracy(labels, predictions),
    }


def test_metrics(
    model: nn.Module, test_images: np.ndarray, test_labels: np.ndarray
) -> dict[str, Any]:
    probabilities = predict_probabilities(model, test_images)
    predictions = probabilities.argmax(axis=1)
    non_target = test_labels != TARGET_CLASS
    triggered = core.apply_patch(test_images[non_target], patch_size=PATCH_SIZE)
    attack_predictions = predict_probabilities(model, triggered).argmax(axis=1)
    balanced = core.balanced_accuracy(test_labels, predictions)
    attack_success = float(np.mean(attack_predictions == TARGET_CLASS))
    return {
        "test_balanced_accuracy": balanced,
        "test_macro_auc": macro_auc(test_labels, probabilities),
        "attack_success_rate": attack_success,
        "eligible": bool(balanced >= 0.75 and attack_success >= 0.80),
        "per_class_recall": core.per_class_recall(test_labels, predictions),
    }


def capture_rng_state(shuffle_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "shuffle_generator": shuffle_generator.get_state(),
    }


def restore_rng_state(payload: dict[str, Any], shuffle_generator: torch.Generator) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    shuffle_generator.set_state(payload["shuffle_generator"])


def checkpoint_contract(
    *,
    protocol_sha256: str,
    frozen_hashes: dict[str, Any],
    seed: int,
    official_state_sha256: str,
    mode: ExecutionMode,
) -> dict[str, Any]:
    return {
        "protocol_sha256": protocol_sha256,
        "frozen_hashes": frozen_hashes,
        "seed": int(seed),
        "official_weight_filename": WEIGHT_FILENAMES[int(seed)],
        "official_state_sha256": official_state_sha256,
        "formal": mode.formal,
        "target_epochs": mode.target_epochs,
        "max_batches_per_epoch": mode.max_batches_per_epoch,
        "primary_checkpoint": PRIMARY_CHECKPOINT,
        "validation_role": VALIDATION_ROLE,
        "runtime_environment": runtime_environment(),
    }


def load_resume_checkpoint(
    path: Path,
    *,
    expected_contract: dict[str, Any],
    expected_official_state: dict[str, torch.Tensor],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    shuffle_generator: torch.Generator,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("contract") != expected_contract:
        raise RuntimeError(
            f"resume checkpoint contract mismatch: expected {expected_contract}, "
            f"observed {payload.get('contract')}"
        )
    required = (
        "model_state",
        "optimizer_state",
        "epoch",
        "rng_state",
        "history",
        "initial_official_state",
        "initial_official_state_sha256",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"resume checkpoint is missing fields: {missing}")
    saved_initial = payload["initial_official_state"]
    expected_hash = core.state_fingerprint(expected_official_state)
    if payload["initial_official_state_sha256"] != expected_hash:
        raise RuntimeError("resume initial-official-state hash mismatch")
    if not states_bit_identical(saved_initial, expected_official_state):
        raise RuntimeError("resume initial official state is not bit-identical")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
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
    frozen_hashes: dict[str, Any],
    receipt: dict[str, Any],
    resume: bool,
) -> tuple[architecture.PathMNISTResNet18, dict[str, Any]]:
    """Fine-tune from an official state using train/validation only."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model, official_state, weight_path = load_official_initial_state(seed, receipt)
    official_state_sha256 = core.state_fingerprint(official_state)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    shuffle_generator = torch.Generator().manual_seed(20261001 + seed)
    contract = checkpoint_contract(
        protocol_sha256=protocol_sha256,
        frozen_hashes=frozen_hashes,
        seed=seed,
        official_state_sha256=official_state_sha256,
        mode=mode,
    )
    latest = output / "resume" / f"seed_{seed}_latest.pt"
    start_epoch = 1
    history: list[dict[str, Any]] = []
    if latest.exists():
        if not resume:
            raise RuntimeError(f"resume checkpoint exists; pass --resume: {latest}")
        restored = load_resume_checkpoint(
            latest,
            expected_contract=contract,
            expected_official_state=official_state,
            model=model,
            optimizer=optimizer,
            shuffle_generator=shuffle_generator,
        )
        start_epoch = int(restored["epoch"]) + 1
        history = list(restored["history"])
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
        diagnostics = validation_diagnostics(
            model, validation_images, validation_labels
        )
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_cross_entropy": loss_sum / count,
                "training_observations": count,
                "validation_role": VALIDATION_ROLE,
                **diagnostics,
            }
        )
        payload = {
            "contract": contract,
            "model_state": cpu_state(model.state_dict()),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "rng_state": capture_rng_state(shuffle_generator),
            "history": history,
            "initial_official_state": official_state,
            "initial_official_state_sha256": official_state_sha256,
        }
        atomic_torch_save(latest, payload)
        print(
            f"seed={seed} epoch={epoch}/{mode.target_epochs} "
            f"val_auc={diagnostics['validation_macro_auc']:.4f} "
            "(diagnostic only)",
            flush=True,
        )
    if len(history) != mode.target_epochs:
        raise RuntimeError("training history does not cover the fixed target epochs")
    return model, {
        "official_weight_path": str(weight_path),
        "official_weight_sha256": frozen_hashes["official_weight_sha256_by_seed"][
            str(seed)
        ],
        "initial_official_state_sha256": official_state_sha256,
        "initial_state_bit_identical_to_official": True,
        "final_state_sha256": core.state_fingerprint(model.state_dict()),
        "epochs_completed": len(history),
        "elapsed_seconds_this_invocation": time.time() - started,
        "primary_checkpoint": PRIMARY_CHECKPOINT,
        "validation_role": VALIDATION_ROLE,
        "test_data_used_during_training_or_selection": False,
        "history": history,
    }


def save_model_artifact(
    path: Path,
    state: dict[str, torch.Tensor],
    *,
    seed: int,
    protocol_sha256: str,
    frozen_hashes: dict[str, Any],
    formal: bool,
    epoch: int,
    initial_official_state_sha256: str,
) -> dict[str, Any]:
    role = "primary_fixed_final_epoch_5" if formal else "dry_run_final_epoch_nonformal"
    payload = {
        "architecture": "pathmnist_official_resnet18_cifar_stem_hidden512",
        "state_dict": cpu_state(state),
        "seed": seed,
        "role": role,
        "formal": formal,
        "epoch": epoch,
        "validation_selection": False,
        "test_selection": False,
        "initial_official_state_sha256": initial_official_state_sha256,
        "protocol_sha256": protocol_sha256,
        "frozen_hashes": frozen_hashes,
    }
    atomic_torch_save(path, payload)
    return {
        "path": str(path),
        "file_sha256": core.sha256_file(path),
        "state_sha256": core.state_fingerprint(state),
        "epoch": epoch,
        "role": role,
    }


def should_stop_after_seed93(row: dict[str, Any], *, formal: bool) -> bool:
    return bool(formal and int(row["seed"]) == 93 and not bool(row["eligible"]))


def compute_gate(rows: list[dict[str, Any]], *, formal: bool) -> dict[str, bool]:
    unique = len({int(row["seed"]) for row in rows}) == len(rows)
    by_seed = {int(row["seed"]): bool(row["eligible"]) for row in rows}
    seed93_complete = unique and 93 in by_seed
    seed93_failed = seed93_complete and not by_seed[93]
    both_complete = unique and set(by_seed) == set(FORMAL_SEEDS)
    both_eligible = both_complete and all(by_seed[seed] for seed in FORMAL_SEEDS)
    valid_early_stop = bool(formal and seed93_failed and set(by_seed) == {93})
    formal_conclusion_complete = bool(formal and (valid_early_stop or both_complete))
    definitive_gate_failure = bool(
        formal and (valid_early_stop or (both_complete and not both_eligible))
    )
    return {
        "formal_execution": formal,
        "seed93_complete": seed93_complete,
        "seed93_failed_eligibility": seed93_failed,
        "seed94_required": bool(formal and seed93_complete and by_seed[93]),
        "both_frozen_seeds_complete": both_complete,
        "both_fixed_epoch5_victims_eligible": both_eligible,
        "valid_prespecified_early_stop_after_seed93": valid_early_stop,
        "formal_conclusion_complete": formal_conclusion_complete,
        "definitive_gate_failure": definitive_gate_failure,
        "authorize_followup": bool(formal and both_eligible),
    }


def status_from_gate(gate: dict[str, bool], *, formal: bool) -> str:
    if not formal:
        return "DRY_RUN_NONFORMAL"
    if gate["valid_prespecified_early_stop_after_seed93"]:
        return "FORMAL_STOPPED_AFTER_SEED93_GATE_FAILED"
    if gate["authorize_followup"]:
        return "FORMAL_COMPLETE_GATE_PASSED"
    if gate["formal_conclusion_complete"]:
        return "FORMAL_COMPLETE_GATE_FAILED"
    return "FORMAL_INCOMPLETE"


def run(args: argparse.Namespace) -> dict[str, Any]:
    seeds = tuple(int(seed) for seed in args.seeds)
    mode = validate_execution_mode(
        dry_run=args.dry_run,
        max_epochs=args.max_epochs,
        dry_run_batches=args.dry_run_batches,
        seeds=seeds,
    )
    protocol, receipt, frozen_hashes = validate_frozen_inputs(args.dataset)
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
        model, training = train_seed(
            pool["images"],
            pool["assigned_labels"],
            validation_images,
            validation_labels,
            seed=seed,
            mode=mode,
            output=args.output,
            protocol_sha256=protocol_sha256,
            frozen_hashes=frozen_hashes,
            receipt=receipt,
            resume=args.resume,
        )
        # Test is reached only after the fixed target epoch has completed.
        metrics = test_metrics(model, test_images, test_labels)
        epoch_label = "epoch5" if mode.formal else f"dry_epoch{mode.target_epochs}"
        model_path = args.output / "models" / f"seed_{seed}_{epoch_label}.pt"
        artifact = save_model_artifact(
            model_path,
            model.state_dict(),
            seed=seed,
            protocol_sha256=protocol_sha256,
            frozen_hashes=frozen_hashes,
            formal=mode.formal,
            epoch=mode.target_epochs,
            initial_official_state_sha256=training[
                "initial_official_state_sha256"
            ],
        )
        row = {
            "seed": seed,
            "formal": mode.formal,
            "official_weight": WEIGHT_FILENAMES[seed],
            "primary_checkpoint": (
                PRIMARY_CHECKPOINT if mode.formal else "dry_run_final_epoch_nonformal"
            ),
            "checkpoint_epoch": mode.target_epochs,
            "validation_selection_used": False,
            **metrics,
        }
        rows.append(row)
        receipts.append(
            {
                "seed": seed,
                "formal": mode.formal,
                "elapsed_seconds": time.time() - seed_started,
                "pool": pool["receipt"],
                "training": training,
                "primary_checkpoint": artifact,
            }
        )
        write_csv(args.output / "victim_metrics.partial.csv", rows)
        atomic_write_json(args.output / "seed_receipts.partial.json", receipts)
        if should_stop_after_seed93(row, formal=mode.formal):
            print(
                "seed=93 failed fixed epoch-5 victim eligibility; "
                "prespecified stop prevents seed 94",
                flush=True,
            )
            break

    gate = compute_gate(rows, formal=mode.formal)
    manifest = {
        "experiment": "pathmnist_official_weight_warmstart_victim_feasibility_v1",
        "status": status_from_gate(gate, formal=mode.formal),
        "formal": mode.formal,
        "dry_run": not mode.formal,
        "requested_seeds": list(seeds),
        "completed_seeds": [int(row["seed"]) for row in rows],
        "formal_seed_order": list(FORMAL_SEEDS),
        "target_epochs": mode.target_epochs,
        "max_batches_per_epoch": mode.max_batches_per_epoch,
        "dataset": str(args.dataset),
        "protocol": str(PROTOCOL),
        "protocol_sha256": protocol_sha256,
        "receipt": str(RECEIPT),
        "frozen_hashes": frozen_hashes,
        "runtime_environment": runtime_environment(),
        "training_count": int(len(data["train"]["labels"])),
        "validation_count_used": int(len(validation_labels)),
        "test_count_used": int(len(test_labels)),
        "poison_count": poison_count,
        "poison_fraction": POISON_FRACTION,
        "target_class": TARGET_CLASS,
        "patch_size": PATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "scheduler": None,
        "primary_checkpoint": PRIMARY_CHECKPOINT,
        "produced_checkpoint_role": (
            PRIMARY_CHECKPOINT if mode.formal else "dry_run_final_epoch_nonformal"
        ),
        "validation_role": VALIDATION_ROLE,
        "test_timing": "after_fixed_target_epoch_only",
        "test_data_used_during_training_or_selection": False,
        "trains_clean_pair": False,
        "runs_detectors": False,
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
    parser.add_argument("--dry-run-batches", type=int, default=1)
    args = parser.parse_args()
    if args.seeds is None:
        args.seeds = list(FORMAL_SEEDS if not args.dry_run else FORMAL_SEEDS[:1])
    if args.output is None:
        args.output = DEFAULT_OUTPUT if not args.dry_run else DEFAULT_DRY_OUTPUT
    manifest = run(args)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
