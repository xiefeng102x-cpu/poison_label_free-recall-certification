#!/usr/bin/env python3
"""Adapt the replacement BloodMNIST clean encoder while preserving clean CA.

The frozen starting point is the validation-selected supervised local-angle
encoder.  Only its PCA256-to-8 projector is adapted; the 8-to-32-to-2
classifier remains frozen.  Candidate selection uses train/validation data and
held-out triggered validation samples.  Test evaluation is opt-in and occurs
once after validation checkpoint selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

import evaluate_bloodmnist_checkpoints as heldout
import train_bloodmnist_clean_encoder as clean_encoder


PROJECT_ROOT = heldout.PROJECT_ROOT
SOURCE_ROOT = heldout.SOURCE_ROOT
CLEAN_ROOT = PROJECT_ROOT / "paper14/results/bloodmnist_clean_encoder_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "paper14/results/bloodmnist_new_encoder_adaptive_v1"


def clean_checkpoint_path(seed: int) -> Path:
    split = "development" if seed == 42 else "confirmation"
    return CLEAN_ROOT / split / f"seed_{seed}_local_l0.pt"


def transform(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((x - mean) / scale).astype(np.float32, copy=False)


def load_data(seed: int, dataset_root: Path, checkpoint: dict) -> tuple[dict, dict]:
    class_data = heldout.collect_classes(dataset_root)
    reconstruction = heldout.reconstruct_split(class_data, seed)
    errors = heldout.reconstruction_error(reconstruction["processed"]["train"], seed)
    if max(errors["target_max_abs_error"], errors["non_target_max_abs_error"]) > 5e-3:
        raise RuntimeError(f"Seed {seed} reconstruction mismatch: {errors}")

    mean = np.asarray(checkpoint["input_scaler_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["input_scaler_scale"], dtype=np.float32)
    if np.any(scale <= 0):
        raise RuntimeError("Checkpoint scaler contains non-positive scale")

    base = SOURCE_ROOT / f"seed_{seed}/layer_8_grid/t6_vs_t0/pr_0.1"
    meta = torch.load(
        base / f"poisoned_samples_meta_seed_{seed}_pr_0.1.pt",
        map_location="cpu",
        weights_only=False,
    )
    trigger = meta["trigger_tensor"].numpy().astype(np.float32)
    archived_poison_pca = meta["poisoned_non_target_data"].numpy().astype(np.float32)
    poison_indices = meta["poison_indices_in_non_target"].numpy().astype(int)
    rebuilt_poison_pca = np.clip(
        reconstruction["processed"]["train"][0][poison_indices] + trigger,
        -3,
        3,
    )
    poison_reconstruction_error = float(
        np.max(np.abs(rebuilt_poison_pca - archived_poison_pca))
    )
    if poison_reconstruction_error > 5e-3:
        raise RuntimeError(
            f"Seed {seed} triggered reconstruction mismatch: "
            f"{poison_reconstruction_error}"
        )

    splits = {}
    for split in ("train", "val", "test"):
        target_pca = reconstruction["processed"][split][6]
        non_target_pca = reconstruction["processed"][split][0]
        triggered_pca = np.clip(non_target_pca + trigger, -3, 3)
        target = transform(target_pca, mean, scale)
        non_target = transform(non_target_pca, mean, scale)
        triggered = transform(triggered_pca, mean, scale)
        splits[split] = {
            "target": target,
            "non_target": non_target,
            "triggered_non_target": triggered,
            "clean_x": np.concatenate([target, non_target], axis=0),
            "clean_y": np.concatenate(
                [
                    np.ones(len(target), dtype=np.int64),
                    np.zeros(len(non_target), dtype=np.int64),
                ]
            ),
        }
    splits["archived_poison"] = transform(archived_poison_pca, mean, scale)
    receipt = {
        "seed": seed,
        "reconstruction_error": errors,
        "triggered_reconstruction_max_abs_error": poison_reconstruction_error,
        "trigger_l2_norm_in_historical_pca_space": float(np.linalg.norm(trigger)),
        "trigger_clip_range": [-3.0, 3.0],
        "split_counts": {
            split: {
                "target": len(splits[split]["target"]),
                "non_target": len(splits[split]["non_target"]),
                "triggered_non_target": len(splits[split]["triggered_non_target"]),
            }
            for split in ("train", "val", "test")
        },
        "archived_poison_count": len(splits["archived_poison"]),
    }
    return splits, receipt


def xyz_features(model: clean_encoder.CleanQNN, x: torch.Tensor) -> torch.Tensor:
    latent = model.encoder.projected_latent(x)
    alpha = (math.pi / 2) * latent
    return torch.cat(
        [torch.cos(alpha), torch.zeros_like(alpha), torch.sin(alpha)], dim=1
    )


def numpy_xyz(model: clean_encoder.CleanQNN, x: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32)
            rows.append(xyz_features(model, batch).cpu().numpy())
    return np.concatenate(rows, axis=0)


def predict(
    model: clean_encoder.CleanQNN, x: np.ndarray, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32)
            logits.append(model(batch).cpu())
    all_logits = torch.cat(logits, dim=0)
    return (
        all_logits.argmax(dim=1).numpy(),
        F.softmax(all_logits, dim=1)[:, 1].numpy(),
    )


def clean_metrics(
    model: clean_encoder.CleanQNN, split: dict, batch_size: int
) -> dict:
    prediction, probability = predict(model, split["clean_x"], batch_size)
    y = split["clean_y"]
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "target_recall": float(tp / (tp + fn)),
        "non_target_recall": float(tn / (tn + fp)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def detector_metrics(
    reference_features: np.ndarray,
    clean_target_features: np.ndarray,
    triggered_features: np.ndarray,
) -> dict:
    estimator = LedoitWolf().fit(reference_features)
    precision = estimator.get_precision()
    location = estimator.location_

    def score(features: np.ndarray) -> np.ndarray:
        delta = features - location
        return np.einsum("ni,ij,nj->n", delta, precision, delta)

    clean_score = score(clean_target_features)
    triggered_score = score(triggered_features)
    auc = float(
        roc_auc_score(
            np.concatenate(
                [np.zeros(len(clean_score)), np.ones(len(triggered_score))]
            ),
            np.concatenate([clean_score, triggered_score]),
        )
    )
    return {
        "auc_xyz_independent_reference": auc,
        "separability_xyz_independent_reference": max(auc, 1.0 - auc),
        "clean_score_mean": float(np.mean(clean_score)),
        "triggered_score_mean": float(np.mean(triggered_score)),
        "reference_count": len(reference_features),
        "clean_evaluation_count": len(clean_target_features),
        "triggered_evaluation_count": len(triggered_features),
    }


def evaluate_split(
    model: clean_encoder.CleanQNN,
    split: dict,
    detector_reference: np.ndarray,
    batch_size: int,
) -> dict:
    trigger_prediction, trigger_probability = predict(
        model, split["triggered_non_target"], batch_size
    )
    clean_target_xyz = numpy_xyz(model, split["target"], batch_size)
    triggered_xyz = numpy_xyz(model, split["triggered_non_target"], batch_size)
    return {
        "clean": clean_metrics(model, split, batch_size),
        "attack": {
            "asr": float(np.mean(trigger_prediction == 1)),
            "mean_target_probability": float(np.mean(trigger_probability)),
            "triggered_count": len(trigger_prediction),
        },
        "detector": detector_metrics(
            detector_reference, clean_target_xyz, triggered_xyz
        ),
    }


def archived_poison_metrics(
    model: clean_encoder.CleanQNN, x: np.ndarray, batch_size: int
) -> dict:
    prediction, probability = predict(model, x, batch_size)
    return {
        "asr": float(np.mean(prediction == 1)),
        "mean_target_probability": float(np.mean(probability)),
        "count": len(prediction),
    }


def selection_key(point: dict, initial_val_ca: float) -> tuple:
    ca = float(point["validation"]["clean"]["accuracy"])
    asr = float(point["validation"]["attack"]["asr"])
    separation = float(
        point["validation"]["detector"][
            "separability_xyz_independent_reference"
        ]
    )
    ca_floor = max(0.90, initial_val_ca - 0.05)
    feasible = ca >= ca_floor and asr >= 0.80 and separation <= 0.65
    if feasible:
        return (1, ca, -separation, asr, -int(point["epoch"]))
    # When no epoch passes all gates, prioritize gate count, then the weakest
    # normalized margin so one excellent metric cannot hide a failed gate.
    gate_count = int(ca >= ca_floor) + int(asr >= 0.80) + int(separation <= 0.65)
    weakest_margin = min(ca - ca_floor, asr - 0.80, 0.65 - separation)
    return (0, gate_count, weakest_margin, ca, asr, -separation)


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    checkpoint_path = (
        args.clean_checkpoint.resolve()
        if args.clean_checkpoint is not None
        else clean_checkpoint_path(args.seed).resolve()
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["encoder"] != "local" or int(checkpoint["layers"]) != 0:
        raise RuntimeError("Adaptive experiment requires the selected local:0 encoder")
    if int(checkpoint["seed"]) != args.seed:
        raise RuntimeError("Clean checkpoint seed does not match requested seed")
    model = clean_encoder.CleanQNN("local", 0)
    model.load_state_dict(checkpoint["state_dict"])
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(False)

    splits, data_receipt = load_data(args.seed, args.dataset_root, checkpoint)
    x_clean = splits["train"]["clean_x"]
    y_clean = splits["train"]["clean_y"]
    x_poison = splits["archived_poison"]
    clean_tensor = torch.as_tensor(x_clean, dtype=torch.float32)
    with torch.no_grad():
        teacher_logits = model(clean_tensor).detach()
        clean_reference_xyz = xyz_features(
            model,
            torch.as_tensor(splits["train"]["target"], dtype=torch.float32),
        ).numpy()
    stealth_estimator = LedoitWolf().fit(clean_reference_xyz)
    stealth_mu = torch.as_tensor(stealth_estimator.location_, dtype=torch.float32)
    stealth_precision = torch.as_tensor(
        stealth_estimator.get_precision(), dtype=torch.float32
    )
    initial_projector = copy.deepcopy(model.encoder.projector.state_dict())
    initial_parameters = [
        parameter.detach().clone() for parameter in model.encoder.projector.parameters()
    ]
    adapted_parameters = list(model.encoder.projector.parameters())
    rng = np.random.default_rng(args.seed + 7301)
    history = []
    states = []
    start = time.time()

    def record(epoch: int, objective_summary: dict | None) -> None:
        detector_reference = numpy_xyz(
            model, splits["train"]["target"], args.batch_size
        )
        history.append(
            {
                "epoch": epoch,
                "objective": objective_summary,
                "train": evaluate_split(
                    model, splits["train"], detector_reference, args.batch_size
                ),
                "validation": evaluate_split(
                    model, splits["val"], detector_reference, args.batch_size
                ),
                "archived_poison": archived_poison_metrics(
                    model, x_poison, args.batch_size
                ),
                "elapsed_seconds": time.time() - start,
            }
        )
        states.append(copy.deepcopy(model.encoder.projector.state_dict()))

    record(0, None)
    for epoch in range(1, args.epochs + 1):
        clean_order = rng.permutation(len(x_clean))
        poison_order = rng.permutation(len(x_poison))
        poison_cursor = 0
        objective_sums = {
            "clean_ce": 0.0,
            "backdoor_ce": 0.0,
            "stealth": 0.0,
            "distill": 0.0,
            "anchor": 0.0,
            "total": 0.0,
            "gradient_norm_before_clip_max": 0.0,
            "updates": 0,
        }
        for batch_start in range(0, len(x_clean), args.batch_size):
            clean_indices = clean_order[batch_start : batch_start + args.batch_size]
            size = len(clean_indices)
            if poison_cursor + size > len(poison_order):
                poison_order = rng.permutation(len(x_poison))
                poison_cursor = 0
            poison_indices = poison_order[poison_cursor : poison_cursor + size]
            poison_cursor += size
            if len(poison_indices) < size:
                extra = rng.choice(len(x_poison), size - len(poison_indices), replace=True)
                poison_indices = np.concatenate([poison_indices, extra])

            clean_batch = torch.as_tensor(x_clean[clean_indices], dtype=torch.float32)
            clean_labels = torch.as_tensor(y_clean[clean_indices], dtype=torch.long)
            poison_batch = torch.as_tensor(x_poison[poison_indices], dtype=torch.float32)
            clean_logits = model(clean_batch)
            poison_logits = model(poison_batch)
            clean_loss = F.cross_entropy(clean_logits, clean_labels)
            backdoor_loss = F.cross_entropy(
                poison_logits, torch.ones(len(poison_batch), dtype=torch.long)
            )

            poison_xyz = xyz_features(model, poison_batch)
            delta = poison_xyz - stealth_mu
            stealth_loss = ((delta @ stealth_precision) * delta).sum(dim=1).mean()
            temperature = 2.0
            distill_loss = F.kl_div(
                F.log_softmax(clean_logits / temperature, dim=1),
                F.softmax(teacher_logits[clean_indices] / temperature, dim=1),
                reduction="batchmean",
            ) * temperature**2
            anchor_loss = torch.stack(
                [
                    torch.mean((parameter - initial) ** 2)
                    for parameter, initial in zip(
                        adapted_parameters, initial_parameters
                    )
                ]
            ).mean()
            objective = (
                clean_loss
                + args.backdoor_weight * backdoor_loss
                + args.stealth_weight * stealth_loss
                + args.distill_weight * distill_loss
                + args.anchor_weight * anchor_loss
            )
            gradients = torch.autograd.grad(objective, adapted_parameters)
            gradient_norm = torch.sqrt(
                sum(torch.sum(gradient**2) for gradient in gradients)
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Non-finite gradient encountered")
            scale = 1.0
            if args.gradient_clip > 0 and gradient_norm > args.gradient_clip:
                scale = args.gradient_clip / float(gradient_norm)
            with torch.no_grad():
                for parameter, gradient in zip(
                    adapted_parameters, gradients
                ):
                    parameter -= args.learning_rate * scale * gradient

            for name, value in (
                ("clean_ce", clean_loss),
                ("backdoor_ce", backdoor_loss),
                ("stealth", stealth_loss),
                ("distill", distill_loss),
                ("anchor", anchor_loss),
                ("total", objective),
            ):
                objective_sums[name] += float(value.detach())
            objective_sums["gradient_norm_before_clip_max"] = max(
                objective_sums["gradient_norm_before_clip_max"], float(gradient_norm)
            )
            objective_sums["updates"] += 1

        updates = objective_sums["updates"]
        objective_summary = {
            name: value / updates
            for name, value in objective_sums.items()
            if name not in {"updates", "gradient_norm_before_clip_max"}
        }
        objective_summary["gradient_norm_before_clip_max"] = objective_sums[
            "gradient_norm_before_clip_max"
        ]
        objective_summary["updates"] = updates
        record(epoch, objective_summary)
        point = history[-1]
        print(
            f"seed={args.seed} epoch={epoch} "
            f"val_ca={point['validation']['clean']['accuracy']:.4f} "
            f"val_asr={point['validation']['attack']['asr']:.4f} "
            "val_sep="
            f"{point['validation']['detector']['separability_xyz_independent_reference']:.4f}",
            flush=True,
        )

    initial_val_ca = float(history[0]["validation"]["clean"]["accuracy"])
    selected_index = max(
        range(len(history)), key=lambda index: selection_key(history[index], initial_val_ca)
    )
    model.encoder.projector.load_state_dict(states[selected_index])
    selected = copy.deepcopy(history[selected_index])
    selected["selection_key"] = list(selection_key(selected, initial_val_ca))
    selected["clean_ca_floor"] = max(0.90, initial_val_ca - 0.05)
    selected["passes_validation_gates"] = bool(selected["selection_key"][0] == 1)
    selected["test_evaluated"] = bool(args.evaluate_test)
    if args.evaluate_test:
        detector_reference = numpy_xyz(
            model, splits["train"]["target"], args.batch_size
        )
        selected["test"] = evaluate_split(
            model, splits["test"], detector_reference, args.batch_size
        )

    initial_flat = torch.cat([value.reshape(-1) for value in initial_parameters])
    selected_flat = torch.cat(
        [
            states[selected_index]["weight"].reshape(-1),
            states[selected_index]["bias"].reshape(-1),
        ]
    )
    return {
        "seed": args.seed,
        "clean_checkpoint": str(checkpoint_path),
        "configuration": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "backdoor_weight": args.backdoor_weight,
            "stealth_weight": args.stealth_weight,
            "distill_weight": args.distill_weight,
            "anchor_weight": args.anchor_weight,
            "gradient_clip": args.gradient_clip,
            "batch_size": args.batch_size,
            "adapted_parameters": sum(
                parameter.numel() for parameter in adapted_parameters
            ),
            "frozen_classifier_parameters": sum(
                parameter.numel() for parameter in model.classifier.parameters()
            ),
        },
        "data_receipt": data_receipt,
        "history": history,
        "selected": selected,
        "projector_l2_drift": float(
            torch.linalg.vector_norm(selected_flat - initial_flat)
        ),
        "initial_projector_state": initial_projector,
        "selected_projector_state": {
            name: value.detach().cpu() for name, value in states[selected_index].items()
        },
    }


def json_ready(result: dict) -> dict:
    output = copy.deepcopy(result)
    for state_name in ("initial_projector_state", "selected_projector_state"):
        output[state_name] = {
            name: value.numpy().tolist() for name, value in output[state_name].items()
        }
    return output


def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    result = run(args)
    tag = (
        f"seed={args.seed}_lr={args.learning_rate}_b={args.backdoor_weight}_"
        f"s={args.stealth_weight}_d={args.distill_weight}_a={args.anchor_weight}"
    )
    artifact = args.output / f"{tag}.json"
    artifact.write_text(
        json.dumps(json_ready(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    checkpoint = torch.load(
        result["clean_checkpoint"], map_location="cpu", weights_only=False
    )
    checkpoint["state_dict"]["encoder.projector.weight"] = result[
        "selected_projector_state"
    ]["weight"]
    checkpoint["state_dict"]["encoder.projector.bias"] = result[
        "selected_projector_state"
    ]["bias"]
    checkpoint["adaptive_configuration"] = result["configuration"]
    checkpoint["adaptive_selected_epoch"] = result["selected"]["epoch"]
    checkpoint["adaptive_source_artifact"] = str(artifact)
    torch.save(checkpoint, args.output / f"{tag}.pt")
    compact = {
        "artifact": str(artifact),
        "seed": args.seed,
        "selected_epoch": result["selected"]["epoch"],
        "passes_validation_gates": result["selected"]["passes_validation_gates"],
        "validation": result["selected"]["validation"],
        "test": result["selected"].get("test"),
    }
    print(json.dumps(compact, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "development")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--backdoor-weight", type=float, default=10.0)
    parser.add_argument("--stealth-weight", type=float, default=0.0)
    parser.add_argument("--distill-weight", type=float, default=20.0)
    parser.add_argument("--anchor-weight", type=float, default=0.001)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--evaluate-test", action="store_true")
    parsed = parser.parse_args()
    parsed.dataset_root = parsed.dataset_root.resolve()
    parsed.output = parsed.output.resolve()
    if parsed.clean_checkpoint is not None:
        parsed.clean_checkpoint = parsed.clean_checkpoint.resolve()
    main(parsed)
