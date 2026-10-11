#!/usr/bin/env python3
"""CA-preserving adaptive QNN intervention for BloodMNIST.

This keeps the archived clean circuit and classifier architecture but replaces
the invalid scalar-loss parameter shift with exact reverse-mode gradients.  It
separates clean and backdoor losses and adds optional clean-logit distillation,
weight anchoring, gradient clipping, and a properly small stealth coefficient.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf

import evaluate_bloodmnist_checkpoints as heldout
from run_bloodmnist_corrected_qnn import (
    circuit_xyz,
    classifier_logits,
    evaluate,
    features_numpy,
    simulator_check,
)


PROJECT_ROOT = heldout.PROJECT_ROOT
SOURCE_ROOT = heldout.SOURCE_ROOT


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(
    seed: int,
    dataset_root: Path,
    epochs: int,
    learning_rate: float,
    backdoor_weight: float,
    stealth_weight: float,
    distill_weight: float,
    anchor_weight: float,
    gradient_clip: float,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    base = SOURCE_ROOT / f"seed_{seed}/layer_8_grid/t6_vs_t0/pr_0.1"
    state = torch.load(
        base / "models/clean_model_seed_200.pt", map_location="cpu", weights_only=True
    )
    meta = torch.load(
        base / f"poisoned_samples_meta_seed_{seed}_pr_0.1.pt",
        map_location="cpu",
        weights_only=False,
    )
    x_target = meta["target_clean_data"].numpy().astype(np.float64)
    x_nontarget = meta["non_target_clean_data"].numpy().astype(np.float64)
    x_poison = meta["poisoned_non_target_data"].numpy().astype(np.float64)
    x_clean = np.concatenate([x_target, x_nontarget], axis=0)
    y_clean = np.concatenate(
        [
            np.ones(len(x_target), dtype=np.int64),
            np.zeros(len(x_nontarget), dtype=np.int64),
        ]
    )

    initial_np = state["weights"].numpy().astype(np.float64)
    simulator_error = simulator_check(
        np.concatenate([x_target[:2], x_nontarget[:2]], axis=0), initial_np
    )
    if simulator_error > 1e-7:
        raise RuntimeError(f"Simulator mismatch: {simulator_error}")
    initial = torch.tensor(initial_np, dtype=torch.float64)
    weights = initial.clone().requires_grad_(True)

    xyz_reference = features_numpy(x_target, initial)
    estimator = LedoitWolf().fit(xyz_reference)
    mu = estimator.location_
    precision = estimator.get_precision()
    mu_t = torch.tensor(mu, dtype=torch.float64)
    precision_t = torch.tensor(precision, dtype=torch.float64)

    # Teacher logits are fixed clean-checkpoint predictions and are computed
    # once, so distillation does not add another circuit pass per update.
    clean_t = torch.tensor(x_clean, dtype=torch.float64)
    with torch.no_grad():
        teacher_xyz = circuit_xyz(clean_t, initial)
        teacher_logits = classifier_logits(state, teacher_xyz[:, 16:])

    class_data = heldout.collect_classes(dataset_root)
    reconstruction = heldout.reconstruct_split(class_data, seed)
    errors = heldout.reconstruction_error(reconstruction["processed"]["train"], seed)
    if max(errors["target_max_abs_error"], errors["non_target_max_abs_error"]) > 5e-3:
        raise RuntimeError(f"Seed {seed} reconstruction mismatch: {errors}")
    validation_target = reconstruction["processed"]["val"][6].astype(np.float64)
    validation_nontarget = reconstruction["processed"]["val"][0].astype(np.float64)
    test_target = reconstruction["processed"]["test"][6].astype(np.float64)
    test_nontarget = reconstruction["processed"]["test"][0].astype(np.float64)

    rng = np.random.default_rng(seed + 7301)
    history: list[dict] = []
    start = time.time()

    def record(epoch: int) -> None:
        history.append(
            {
                "epoch": epoch,
                "archived_clean_pool": evaluate(
                    x_target, x_nontarget, x_poison, state, weights, mu, precision
                ),
                "validation": evaluate(
                    validation_target,
                    validation_nontarget,
                    x_poison,
                    state,
                    weights,
                    mu,
                    precision,
                ),
                "untouched_test": evaluate(
                    test_target,
                    test_nontarget,
                    x_poison,
                    state,
                    weights,
                    mu,
                    precision,
                ),
                "elapsed_seconds": time.time() - start,
            }
        )

    record(0)
    for epoch in range(1, epochs + 1):
        clean_order = rng.permutation(len(x_clean))
        poison_order = rng.permutation(len(x_poison))
        poison_cursor = 0
        for batch_start in range(0, len(x_clean), 32):
            clean_indices = clean_order[batch_start : batch_start + 32]
            batch_size = len(clean_indices)
            if poison_cursor + batch_size > len(poison_order):
                poison_order = rng.permutation(len(x_poison))
                poison_cursor = 0
            poison_indices = poison_order[poison_cursor : poison_cursor + batch_size]
            poison_cursor += batch_size
            if len(poison_indices) < batch_size:
                extra = rng.choice(len(x_poison), batch_size - len(poison_indices), replace=True)
                poison_indices = np.concatenate([poison_indices, extra])

            clean_batch = torch.tensor(x_clean[clean_indices], dtype=torch.float64)
            clean_labels = torch.tensor(y_clean[clean_indices], dtype=torch.long)
            poison_batch = torch.tensor(x_poison[poison_indices], dtype=torch.float64)

            clean_xyz = circuit_xyz(clean_batch, weights)
            clean_logits = classifier_logits(state, clean_xyz[:, 16:])
            clean_loss = F.cross_entropy(clean_logits, clean_labels)

            poison_xyz = circuit_xyz(poison_batch, weights)
            poison_logits = classifier_logits(state, poison_xyz[:, 16:])
            poison_labels = torch.ones(len(poison_batch), dtype=torch.long)
            backdoor_loss = F.cross_entropy(poison_logits, poison_labels)

            delta = poison_xyz - mu_t
            stealth_loss = ((delta @ precision_t) * delta).sum(dim=1).mean()

            temperature = 2.0
            teacher_batch = teacher_logits[clean_indices]
            distill_loss = F.kl_div(
                F.log_softmax(clean_logits / temperature, dim=1),
                F.softmax(teacher_batch / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature**2)
            anchor_loss = torch.mean((weights - initial) ** 2)
            objective = (
                clean_loss
                + backdoor_weight * backdoor_loss
                + stealth_weight * stealth_loss
                + distill_weight * distill_loss
                + anchor_weight * anchor_loss
            )
            gradient = torch.autograd.grad(objective, weights)[0]
            gradient_norm = torch.linalg.vector_norm(gradient)
            if gradient_clip > 0 and gradient_norm > gradient_clip:
                gradient = gradient * (gradient_clip / gradient_norm)
            with torch.no_grad():
                weights -= learning_rate * gradient
        record(epoch)

    return {
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "backdoor_weight": backdoor_weight,
        "stealth_weight": stealth_weight,
        "distill_weight": distill_weight,
        "anchor_weight": anchor_weight,
        "gradient_clip": gradient_clip,
        "simulator_max_abs_z_error": simulator_error,
        "reconstruction_error": errors,
        "history": history,
        "final_weights": weights.detach().numpy().tolist(),
    }


def main(args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    result = run(
        seed=args.seed,
        dataset_root=args.dataset_root,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        backdoor_weight=args.backdoor_weight,
        stealth_weight=args.stealth_weight,
        distill_weight=args.distill_weight,
        anchor_weight=args.anchor_weight,
        gradient_clip=args.gradient_clip,
    )
    tag = (
        f"seed={args.seed}_a={args.backdoor_weight}_s={args.stealth_weight}_"
        f"d={args.distill_weight}_g={args.anchor_weight}_lr={args.learning_rate}"
    )
    artifact = args.output / f"{tag}.json"
    artifact.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows: list[dict] = []
    for point in result["history"]:
        for split in ("archived_clean_pool", "validation", "untouched_test"):
            rows.append(
                {
                    "seed": args.seed,
                    "epoch": point["epoch"],
                    "split": split,
                    **point[split],
                    "elapsed_seconds": point["elapsed_seconds"],
                }
            )
    write_csv(args.output / f"{tag}.csv", rows)
    print(json.dumps({"artifact": str(artifact), "final": result["history"][-1]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "paper14/results/bloodmnist_ca_improvement_v1/phase2_adaptive",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--backdoor-weight", type=float, default=1.0)
    parser.add_argument("--stealth-weight", type=float, default=0.0)
    parser.add_argument("--distill-weight", type=float, default=5.0)
    parser.add_argument("--anchor-weight", type=float, default=0.001)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parsed = parser.parse_args()
    parsed.dataset_root = parsed.dataset_root.resolve()
    parsed.output = parsed.output.resolve()
    main(parsed)
