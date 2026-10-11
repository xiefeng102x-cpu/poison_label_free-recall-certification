#!/usr/bin/env python3
"""Correct-gradient intervention for the adaptive BloodMNIST QNN.

The archived implementation applies the parameter-shift identity directly to
nonlinear scalar losses.  Here a differentiable state-vector simulator uses
reverse-mode automatic differentiation, preserving the archived circuit,
fixed classifier, minibatches, objective, learning rate and plain-SGD update.
This is an implementation-correction experiment, not a new architecture.
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
from sklearn.metrics import roc_auc_score

import evaluate_bloodmnist_checkpoints as heldout


PROJECT_ROOT = heldout.PROJECT_ROOT
SOURCE_ROOT = heldout.SOURCE_ROOT


def apply_one_qubit(psi: torch.Tensor, gate: torch.Tensor, qubit: int) -> torch.Tensor:
    dim = psi.shape[1]
    step = dim >> (qubit + 1)
    groups = dim // (2 * step)
    shaped = psi.reshape(-1, groups, 2, step)
    return torch.einsum("ij,bkjl->bkil", gate, shaped).reshape(-1, dim)


def cnot_source(n_qubits: int, control: int, target: int, device) -> torch.Tensor:
    dim = 2**n_qubits
    states = torch.arange(dim, dtype=torch.long, device=device)
    control_bit = n_qubits - 1 - control
    target_bit = n_qubits - 1 - target
    control_on = (states >> control_bit) & 1
    return torch.where(control_on.bool(), states ^ (1 << target_bit), states)


def circuit_xyz(
    x: torch.Tensor,
    weights: torch.Tensor,
    n_layers: int = 8,
    n_qubits: int = 8,
) -> torch.Tensor:
    real_dtype = weights.dtype
    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
    psi = (x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)).to(complex_dtype)
    reshaped = weights.reshape(n_layers, n_qubits, 2)
    for layer in range(n_layers):
        for qubit in range(n_qubits):
            theta, phi = reshaped[layer, qubit]
            c, s = torch.cos(theta / 2), torch.sin(theta / 2)
            ry = torch.stack((c, -s, s, c)).reshape(2, 2).to(complex_dtype)
            psi = apply_one_qubit(psi, ry, qubit)
            phase_minus = torch.exp(torch.complex(torch.zeros_like(phi), -phi / 2))
            phase_plus = torch.exp(torch.complex(torch.zeros_like(phi), phi / 2))
            zero = torch.zeros((), dtype=complex_dtype, device=weights.device)
            rz = torch.stack((phase_minus, zero, zero, phase_plus)).reshape(2, 2)
            psi = apply_one_qubit(psi, rz, qubit)
        for qubit in range(n_qubits):
            source = cnot_source(n_qubits, qubit, (qubit + 1) % n_qubits, psi.device)
            psi = psi[:, source]

    x_values, y_values, z_values = [], [], []
    for qubit in range(n_qubits):
        dim = psi.shape[1]
        step = dim >> (qubit + 1)
        groups = dim // (2 * step)
        pair = psi.reshape(-1, groups, 2, step)
        amp0, amp1 = pair[:, :, 0, :], pair[:, :, 1, :]
        cross = (amp0.conj() * amp1).sum(dim=(1, 2))
        x_values.append(2 * cross.real)
        y_values.append(2 * cross.imag)
        z_values.append(
            ((amp0.conj() * amp0).real - (amp1.conj() * amp1).real).sum(dim=(1, 2))
        )
    return torch.stack(x_values + y_values + z_values, dim=1)


def classifier_logits(state: dict, z: torch.Tensor) -> torch.Tensor:
    dtype = z.dtype
    w1 = state["classifier.0.weight"].to(dtype=dtype)
    b1 = state["classifier.0.bias"].to(dtype=dtype)
    w2 = state["classifier.3.weight"].to(dtype=dtype)
    b2 = state["classifier.3.bias"].to(dtype=dtype)
    return F.linear(F.relu(F.linear(z, w1, b1)), w2, b2)


def features_numpy(x: np.ndarray, weights: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return circuit_xyz(torch.as_tensor(x, dtype=weights.dtype), weights).cpu().numpy()


def evaluate(
    x_target: np.ndarray,
    x_nontarget: np.ndarray,
    x_poison: np.ndarray,
    state: dict,
    weights: torch.Tensor,
    reference_mu: np.ndarray,
    reference_precision: np.ndarray,
) -> dict:
    xyz_target = features_numpy(x_target, weights)
    xyz_nontarget = features_numpy(x_nontarget, weights)
    xyz_poison = features_numpy(x_poison, weights)
    with torch.no_grad():
        prediction_target = classifier_logits(
            state, torch.as_tensor(xyz_target[:, 16:], dtype=weights.dtype)
        ).argmax(dim=1).numpy()
        prediction_nontarget = classifier_logits(
            state, torch.as_tensor(xyz_nontarget[:, 16:], dtype=weights.dtype)
        ).argmax(dim=1).numpy()
        prediction_poison = classifier_logits(
            state, torch.as_tensor(xyz_poison[:, 16:], dtype=weights.dtype)
        ).argmax(dim=1).numpy()

    clean_accuracy = (
        int(np.sum(prediction_target == 1)) + int(np.sum(prediction_nontarget == 0))
    ) / (len(x_target) + len(x_nontarget))

    def fixed_reference_score(features: np.ndarray) -> np.ndarray:
        delta = features - reference_mu
        return np.einsum("ni,ij,nj->n", delta, reference_precision, delta)

    clean_score = fixed_reference_score(xyz_target)
    poison_score = fixed_reference_score(xyz_poison)
    fixed_reference_auc = roc_auc_score(
        np.concatenate([np.zeros(len(clean_score)), np.ones(len(poison_score))]),
        np.concatenate([clean_score, poison_score]),
    )
    # Match the archived reported AUC: deterministic subsampling followed by a
    # freshly calibrated Ledoit-Wolf model on the current clean features.
    rng = np.random.default_rng(0)
    clean_count = min(len(xyz_target), 300)
    poison_count = min(len(xyz_poison), 300)
    clean_subset = xyz_target[rng.choice(len(xyz_target), clean_count, replace=False)]
    poison_subset = xyz_poison[
        rng.choice(len(xyz_poison), poison_count, replace=False)
    ]
    matched_count = min(clean_count, poison_count)
    clean_subset = clean_subset[:matched_count]
    poison_subset = poison_subset[:matched_count]
    calibrated = LedoitWolf().fit(clean_subset)
    combined = np.concatenate([clean_subset, poison_subset], axis=0)
    delta = combined - calibrated.location_
    score = np.einsum("ni,ij,nj->n", delta, calibrated.get_precision(), delta)
    recalibrated_auc = roc_auc_score(
        np.concatenate([np.zeros(matched_count), np.ones(matched_count)]), score
    )
    return {
        "ca": float(clean_accuracy),
        "asr": float(np.mean(prediction_poison == 1)),
        "auc_xyz": float(recalibrated_auc),
        "auc_xyz_fixed_reference": float(fixed_reference_auc),
        "mahalanobis_squared_mean": float(np.mean(poison_score)),
    }


def simulator_check(x: np.ndarray, weights: np.ndarray) -> float:
    torch_weights = torch.tensor(weights, dtype=torch.float64)
    torch_z = features_numpy(x, torch_weights)[:, 16:]
    numpy_z = heldout.RUN_CIRCUIT_BATCH(x, weights, n_layers=8, n_qubits=8)[:, :8]
    return float(np.max(np.abs(torch_z - numpy_z)))


def run_seed(
    seed: int,
    epochs: int,
    learning_rate: float,
    lambda_r: float,
    eval_every: int,
    dataset_root: Path | None,
) -> dict:
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
    initial = state["weights"].numpy().astype(np.float64)
    check_error = simulator_check(
        np.concatenate([x_target[:2], x_nontarget[:2]], axis=0), initial
    )
    if check_error > 1e-5:
        raise RuntimeError(f"Differentiable simulator mismatch: {check_error}")

    weights = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    xyz_reference = features_numpy(x_target, weights)
    estimator = LedoitWolf().fit(xyz_reference)
    mu = estimator.location_
    precision = estimator.get_precision()
    mu_t = torch.tensor(mu, dtype=torch.float64)
    precision_t = torch.tensor(precision, dtype=torch.float64)
    poison_t = torch.tensor(x_poison, dtype=torch.float64)

    x_train = np.concatenate([x_target, x_nontarget, x_poison], axis=0)
    y_train = np.concatenate(
        [
            np.ones(len(x_target), dtype=np.int64),
            np.zeros(len(x_nontarget), dtype=np.int64),
            np.ones(len(x_poison), dtype=np.int64),
        ]
    )
    rng = np.random.default_rng(42)
    history: list[dict] = []
    start = time.time()

    test_target = test_nontarget = None
    if dataset_root is not None:
        class_data = heldout.collect_classes(dataset_root)
        reconstruction = heldout.reconstruct_split(class_data, seed)
        errors = heldout.reconstruction_error(reconstruction["processed"]["train"], seed)
        if max(errors["target_max_abs_error"], errors["non_target_max_abs_error"]) > 5e-3:
            raise RuntimeError(f"Seed {seed} reconstruction mismatch: {errors}")
        test_target = reconstruction["processed"]["test"][6].astype(np.float64)
        test_nontarget = reconstruction["processed"]["test"][0].astype(np.float64)

    def record(epoch: int) -> None:
        train_metrics = evaluate(
            x_target, x_nontarget, x_poison, state, weights, mu, precision
        )
        test_metrics = None
        if test_target is not None and test_nontarget is not None:
            test_metrics = evaluate(
                test_target, test_nontarget, x_poison, state, weights, mu, precision
            )
        history.append(
            {
                "epoch": epoch,
                "archived_clean_pool": train_metrics,
                "untouched_test": test_metrics,
                "elapsed_seconds": time.time() - start,
            }
        )

    record(0)
    for epoch in range(1, epochs + 1):
        indices = rng.permutation(len(x_train))
        for batch_start in range(0, len(x_train), 32):
            batch_indices = indices[batch_start : batch_start + 32]
            x_batch = torch.tensor(x_train[batch_indices], dtype=torch.float64)
            y_batch = torch.tensor(y_train[batch_indices], dtype=torch.long)
            xyz_batch = circuit_xyz(x_batch, weights)
            logits = classifier_logits(state, xyz_batch[:, 16:])
            cross_entropy = F.cross_entropy(logits, y_batch)
            xyz_poison = circuit_xyz(poison_t, weights)
            delta = xyz_poison - mu_t
            representation = ((delta @ precision_t) * delta).sum(dim=1).mean()
            objective = cross_entropy + lambda_r * representation
            gradient = torch.autograd.grad(objective, weights)[0]
            with torch.no_grad():
                weights -= learning_rate * gradient
        if epoch % eval_every == 0 or epoch == epochs:
            record(epoch)

    return {
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "lambda_r": lambda_r,
        "simulator_max_abs_z_error": check_error,
        "gradient_method": "reverse-mode exact differentiation of the nonlinear objective",
        "history": history,
        "final_weights": weights.detach().numpy().tolist(),
    }


def main(
    output: Path,
    seeds: list[int],
    epochs: int,
    learning_rate: float,
    lambda_r: float,
    eval_every: int,
    dataset_root: Path | None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in seeds:
        result = run_seed(
            seed, epochs, learning_rate, lambda_r, eval_every, dataset_root
        )
        runs.append(result)
        (output / f"corrected_seed={seed}_lambda={lambda_r}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    rows: list[dict] = []
    for result in runs:
        for point in result["history"]:
            for split in ("archived_clean_pool", "untouched_test"):
                metrics = point[split]
                if metrics is not None:
                    rows.append(
                        {
                            "seed": result["seed"],
                            "lambda_r": lambda_r,
                            "epoch": point["epoch"],
                            "split": split,
                            **metrics,
                            "elapsed_seconds": point["elapsed_seconds"],
                        }
                    )
    write_csv(output / "corrected_qnn_trajectory.csv", rows)
    print(json.dumps(rows, indent=2, sort_keys=True, allow_nan=False))


def reevaluate_artifacts(
    artifact_paths: list[Path], dataset_root: Path, output: Path
) -> None:
    """Recompute comparable AUC/CA metrics from saved final-weight artifacts."""
    output.mkdir(parents=True, exist_ok=True)
    class_data = heldout.collect_classes(dataset_root)
    rows: list[dict] = []
    for artifact_path in artifact_paths:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        seed = int(artifact["seed"])
        base = SOURCE_ROOT / f"seed_{seed}/layer_8_grid/t6_vs_t0/pr_0.1"
        state = torch.load(
            base / "models/clean_model_seed_200.pt",
            map_location="cpu",
            weights_only=True,
        )
        meta = torch.load(
            base / f"poisoned_samples_meta_seed_{seed}_pr_0.1.pt",
            map_location="cpu",
            weights_only=False,
        )
        x_target = meta["target_clean_data"].numpy().astype(np.float64)
        x_nontarget = meta["non_target_clean_data"].numpy().astype(np.float64)
        x_poison = meta["poisoned_non_target_data"].numpy().astype(np.float64)
        initial_weights = torch.tensor(state["weights"].numpy(), dtype=torch.float64)
        xyz_reference = features_numpy(x_target, initial_weights)
        estimator = LedoitWolf().fit(xyz_reference)

        reconstruction = heldout.reconstruct_split(class_data, seed)
        test_target = reconstruction["processed"]["test"][6].astype(np.float64)
        test_nontarget = reconstruction["processed"]["test"][0].astype(np.float64)
        final_weights = torch.tensor(artifact["final_weights"], dtype=torch.float64)
        for endpoint, weights in (("initial", initial_weights), ("final", final_weights)):
            for split, target, nontarget in (
                ("archived_clean_pool", x_target, x_nontarget),
                ("untouched_test", test_target, test_nontarget),
            ):
                rows.append(
                    {
                        "artifact": str(artifact_path),
                        "seed": seed,
                        "lambda_r": artifact["lambda_r"],
                        "learning_rate": artifact["learning_rate"],
                        "epochs": artifact["epochs"],
                        "endpoint": endpoint,
                        "split": split,
                        **evaluate(
                            target,
                            nontarget,
                            x_poison,
                            state,
                            weights,
                            estimator.location_,
                            estimator.get_precision(),
                        ),
                    }
                )
    write_csv(output / "corrected_qnn_recalibrated_summary.csv", rows)
    (output / "corrected_qnn_recalibrated_summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2, sort_keys=True, allow_nan=False))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "paper14/results/bloodmnist_ca_diagnosis_v1/corrected_qnn",
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--lambda-r", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument(
        "--reevaluate-jsons",
        nargs="+",
        type=Path,
        help="Recompute comparable metrics from existing corrected-run JSON artifacts",
    )
    args = parser.parse_args()
    if args.reevaluate_jsons:
        if args.dataset_root is None:
            parser.error("--dataset-root is required with --reevaluate-jsons")
        reevaluate_artifacts(
            [path.resolve() for path in args.reevaluate_jsons],
            args.dataset_root.resolve(),
            args.output.resolve(),
        )
        raise SystemExit(0)
    main(
        args.output.resolve(),
        args.seeds,
        args.epochs,
        args.learning_rate,
        args.lambda_r,
        args.eval_every,
        args.dataset_root.resolve() if args.dataset_root is not None else None,
    )
