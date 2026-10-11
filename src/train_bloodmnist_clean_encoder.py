#!/usr/bin/env python3
"""Train a replacement clean encoder for the BloodMNIST QNN experiment.

The historical PCA-256 train/validation/test split is reconstructed exactly.
This intervention replaces amplitude encoding with a supervised 256->8
projection followed by angle encoding while keeping an eight-observable Z
readout and the historical 8->32->2 classifier size.

Development runs select an encoder using train/validation data only.  Test
evaluation is opt-in so candidate selection cannot accidentally inspect the
historical held-out split.
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import evaluate_bloodmnist_checkpoints as heldout


PROJECT_ROOT = heldout.PROJECT_ROOT
DEFAULT_OUTPUT = PROJECT_ROOT / "paper14/results/bloodmnist_clean_encoder_v1"


def apply_one_qubit(psi: torch.Tensor, gate: torch.Tensor, qubit: int) -> torch.Tensor:
    dim = psi.shape[1]
    step = dim >> (qubit + 1)
    groups = dim // (2 * step)
    shaped = psi.reshape(-1, groups, 2, step)
    return torch.einsum("ij,bkjl->bkil", gate, shaped).reshape(-1, dim)


def apply_batch_ry(
    psi: torch.Tensor, angles: torch.Tensor, qubit: int
) -> torch.Tensor:
    """Apply a different RY angle to each item in a batch."""
    dim = psi.shape[1]
    step = dim >> (qubit + 1)
    groups = dim // (2 * step)
    shaped = psi.reshape(-1, groups, 2, step)
    amp0, amp1 = shaped[:, :, 0, :], shaped[:, :, 1, :]
    cosine = torch.cos(angles / 2)[:, None, None]
    sine = torch.sin(angles / 2)[:, None, None]
    output0 = cosine * amp0 - sine * amp1
    output1 = sine * amp0 + cosine * amp1
    return torch.stack((output0, output1), dim=2).reshape(-1, dim)


def cnot_source(n_qubits: int, control: int, target: int, device) -> torch.Tensor:
    dim = 2**n_qubits
    states = torch.arange(dim, dtype=torch.long, device=device)
    control_bit = n_qubits - 1 - control
    target_bit = n_qubits - 1 - target
    control_on = (states >> control_bit) & 1
    return torch.where(control_on.bool(), states ^ (1 << target_bit), states)


def z_expectations(psi: torch.Tensor, n_qubits: int) -> torch.Tensor:
    values = []
    dim = psi.shape[1]
    for qubit in range(n_qubits):
        step = dim >> (qubit + 1)
        groups = dim // (2 * step)
        pair = psi.reshape(-1, groups, 2, step)
        amp0, amp1 = pair[:, :, 0, :], pair[:, :, 1, :]
        values.append(
            ((amp0.conj() * amp0).real - (amp1.conj() * amp1).real).sum(
                dim=(1, 2)
            )
        )
    return torch.stack(values, dim=1)


class SupervisedAngleEncoder(nn.Module):
    """Supervised projection plus exact eight-qubit angle encoding."""

    def __init__(
        self,
        input_dim: int = 256,
        n_qubits: int = 8,
        encoder: str = "local",
        n_layers: int = 0,
    ) -> None:
        super().__init__()
        if encoder not in {"local", "single", "reupload"}:
            raise ValueError(f"Unknown encoder: {encoder}")
        if encoder == "local" and n_layers != 0:
            raise ValueError("The local encoder must use zero variational layers")
        if encoder != "local" and n_layers < 1:
            raise ValueError("Entangled encoders require at least one layer")
        self.n_qubits = n_qubits
        self.encoder = encoder
        self.n_layers = n_layers
        self.projector = nn.Linear(input_dim, n_qubits)
        nn.init.xavier_uniform_(self.projector.weight, gain=0.25)
        nn.init.zeros_(self.projector.bias)
        if n_layers:
            self.qweights = nn.Parameter(torch.zeros(n_layers, n_qubits, 2))
        else:
            self.register_parameter("qweights", None)

    def projected_latent(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.projector(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.projected_latent(x)
        # RY(pi/2 - alpha) gives <Z> = sin(alpha), retaining the sign of the
        # supervised projection.  alpha is bounded to avoid periodic aliasing.
        alpha = (math.pi / 2) * latent
        if self.encoder == "local":
            return torch.sin(alpha)

        angles = (math.pi / 2) - alpha
        complex_dtype = torch.complex64 if x.dtype == torch.float32 else torch.complex128
        psi = torch.zeros(
            (len(x), 2**self.n_qubits), dtype=complex_dtype, device=x.device
        )
        psi[:, 0] = 1.0
        if self.encoder == "single":
            for qubit in range(self.n_qubits):
                psi = apply_batch_ry(psi, angles[:, qubit], qubit)

        assert self.qweights is not None
        for layer in range(self.n_layers):
            if self.encoder == "reupload":
                for qubit in range(self.n_qubits):
                    psi = apply_batch_ry(psi, angles[:, qubit], qubit)
            for qubit in range(self.n_qubits):
                theta, phi = self.qweights[layer, qubit]
                cosine, sine = torch.cos(theta / 2), torch.sin(theta / 2)
                ry = torch.stack((cosine, -sine, sine, cosine)).reshape(2, 2)
                psi = apply_one_qubit(psi, ry.to(complex_dtype), qubit)
                phase_minus = torch.exp(
                    torch.complex(torch.zeros_like(phi), -phi / 2)
                )
                phase_plus = torch.exp(
                    torch.complex(torch.zeros_like(phi), phi / 2)
                )
                zero = torch.zeros((), dtype=complex_dtype, device=x.device)
                rz = torch.stack((phase_minus, zero, zero, phase_plus)).reshape(2, 2)
                psi = apply_one_qubit(psi, rz, qubit)
            for qubit in range(self.n_qubits):
                source = cnot_source(
                    self.n_qubits, qubit, (qubit + 1) % self.n_qubits, x.device
                )
                psi = psi[:, source]
        return z_expectations(psi, self.n_qubits)


class CleanQNN(nn.Module):
    def __init__(self, encoder: str, n_layers: int) -> None:
        super().__init__()
        self.encoder = SupervisedAngleEncoder(encoder=encoder, n_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


def metric_dict(y: np.ndarray, prediction: np.ndarray, score: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "roc_auc": float(roc_auc_score(y, score)),
        "target_recall": float(tp / (tp + fn)),
        "non_target_recall": float(tn / (tn + fp)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int) -> dict:
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32)
            logits.append(model(batch).cpu())
    all_logits = torch.cat(logits, dim=0)
    probabilities = F.softmax(all_logits, dim=1)[:, 1].numpy()
    prediction = all_logits.argmax(dim=1).numpy()
    return metric_dict(y, prediction, probabilities)


def state_norm_error(model: CleanQNN, x: np.ndarray) -> float:
    if model.encoder.encoder == "local":
        return 0.0
    model.eval()
    encoder = model.encoder
    with torch.no_grad():
        batch = torch.as_tensor(x[: min(8, len(x))], dtype=torch.float32)
        latent = encoder.projected_latent(batch)
        angles = (math.pi / 2) - (math.pi / 2) * latent
        psi = torch.zeros(
            (len(batch), 2**encoder.n_qubits), dtype=torch.complex64
        )
        psi[:, 0] = 1.0
        if encoder.encoder == "single":
            for qubit in range(encoder.n_qubits):
                psi = apply_batch_ry(psi, angles[:, qubit], qubit)
        assert encoder.qweights is not None
        for layer in range(encoder.n_layers):
            if encoder.encoder == "reupload":
                for qubit in range(encoder.n_qubits):
                    psi = apply_batch_ry(psi, angles[:, qubit], qubit)
            for qubit in range(encoder.n_qubits):
                theta, phi = encoder.qweights[layer, qubit]
                cosine, sine = torch.cos(theta / 2), torch.sin(theta / 2)
                ry = torch.stack((cosine, -sine, sine, cosine)).reshape(2, 2)
                psi = apply_one_qubit(psi, ry.to(torch.complex64), qubit)
                phase_minus = torch.exp(torch.complex(torch.zeros_like(phi), -phi / 2))
                phase_plus = torch.exp(torch.complex(torch.zeros_like(phi), phi / 2))
                zero = torch.zeros((), dtype=torch.complex64)
                rz = torch.stack((phase_minus, zero, zero, phase_plus)).reshape(2, 2)
                psi = apply_one_qubit(psi, rz, qubit)
            for qubit in range(encoder.n_qubits):
                source = cnot_source(
                    encoder.n_qubits, qubit, (qubit + 1) % encoder.n_qubits, batch.device
                )
                psi = psi[:, source]
        norms = (psi.conj() * psi).real.sum(dim=1)
    return float(torch.max(torch.abs(norms - 1)).item())


def load_seed_data(
    class_data: dict[int, np.ndarray], seed: int
) -> tuple[dict, dict, dict]:
    reconstruction = heldout.reconstruct_split(class_data, seed)
    errors = heldout.reconstruction_error(reconstruction["processed"]["train"], seed)
    if max(errors["target_max_abs_error"], errors["non_target_max_abs_error"]) > 5e-3:
        raise RuntimeError(f"Seed {seed} reconstruction mismatch: {errors}")

    x_train, y_train = heldout.arrays_for_split(reconstruction["processed"], "train")
    x_val, y_val = heldout.arrays_for_split(reconstruction["processed"], "val")
    x_test, y_test = heldout.arrays_for_split(reconstruction["processed"], "test")
    scaler = StandardScaler().fit(x_train)
    arrays = {
        "train": (scaler.transform(x_train).astype(np.float32), y_train),
        "val": (scaler.transform(x_val).astype(np.float32), y_val),
        "test": (scaler.transform(x_test).astype(np.float32), y_test),
    }
    receipt = {
        "seed": seed,
        "pca_explained_variance": reconstruction["explained_variance"],
        "reconstruction_error": errors,
        "split_counts": {name: len(value[1]) for name, value in arrays.items()},
        "post_pca_scaler_fit_scope": "train only",
    }
    scaler_state = {
        "mean": scaler.mean_.astype(np.float32),
        "scale": scaler.scale_.astype(np.float32),
    }
    return arrays, receipt, scaler_state


def train_one(
    arrays: dict,
    seed: int,
    encoder: str,
    layers: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    patience: int,
) -> tuple[CleanQNN, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CleanQNN(encoder, layers)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    x_train, y_train = arrays["train"]
    train_data = TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(seed + 22103)
    loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, generator=generator
    )

    best_state = None
    best_epoch = 0
    best_key = (-math.inf, -math.inf)
    stale_epochs = 0
    history = []
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        gradient_norm_max = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x_batch), y_batch)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Non-finite gradient encountered")
            gradient_norm_max = max(gradient_norm_max, float(gradient_norm))
            optimizer.step()
            epoch_loss += float(loss) * len(y_batch)

        train_metrics = evaluate(model, *arrays["train"], batch_size=batch_size)
        val_metrics = evaluate(model, *arrays["val"], batch_size=batch_size)
        key = (val_metrics["accuracy"], val_metrics["roc_auc"])
        improved = key > best_key
        if improved:
            best_key = key
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss / len(train_data),
                "gradient_norm_max": gradient_norm_max,
                "train": train_metrics,
                "val": val_metrics,
                "selected_so_far": improved,
                "elapsed_seconds": time.time() - start_time,
            }
        )
        print(
            f"seed={seed} encoder={encoder} layers={layers} epoch={epoch} "
            f"loss={epoch_loss / len(train_data):.4f} "
            f"train={train_metrics['accuracy']:.4f} val={val_metrics['accuracy']:.4f} "
            f"auc={val_metrics['roc_auc']:.4f}",
            flush=True,
        )
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    final_train = evaluate(model, *arrays["train"], batch_size=batch_size)
    final_val = evaluate(model, *arrays["val"], batch_size=batch_size)
    record = {
        "seed": seed,
        "encoder": encoder,
        "layers": layers,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "early_stopping_patience": patience,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "projector_parameter_count": sum(
            parameter.numel() for parameter in model.encoder.projector.parameters()
        ),
        "quantum_parameter_count": 0
        if model.encoder.qweights is None
        else model.encoder.qweights.numel(),
        "classifier_parameter_count": sum(
            parameter.numel() for parameter in model.classifier.parameters()
        ),
        "train": final_train,
        "val": final_val,
        "train_val_accuracy_gap": final_train["accuracy"] - final_val["accuracy"],
        "state_norm_max_abs_error": state_norm_error(model, arrays["val"][0]),
        "history": history,
        "elapsed_seconds": time.time() - start_time,
    }
    return model, record


def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    class_data = heldout.collect_classes(args.dataset_root)
    all_records = []
    for seed in args.seeds:
        arrays, receipt, scaler_state = load_seed_data(class_data, seed)
        for encoder, layers in args.variant:
            model, record = train_one(
                arrays=arrays,
                seed=seed,
                encoder=encoder,
                layers=layers,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                patience=args.patience,
            )
            record["data_receipt"] = receipt
            record["test_evaluated"] = bool(args.evaluate_test)
            if args.evaluate_test:
                record["test"] = evaluate(
                    model, *arrays["test"], batch_size=args.batch_size
                )

            run_name = f"seed_{seed}_{encoder}_l{layers}"
            checkpoint_path = args.output / f"{run_name}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "encoder": encoder,
                    "layers": layers,
                    "seed": seed,
                    "input_scaler_mean": scaler_state["mean"],
                    "input_scaler_scale": scaler_state["scale"],
                    "note": "Scaler is fit on the reconstructed historical training split only.",
                },
                checkpoint_path,
            )
            record["checkpoint"] = str(checkpoint_path)
            result_path = args.output / f"{run_name}.json"
            result_path.write_text(
                json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            all_records.append(record)

    summary_path = args.output / "run_summary.json"
    summary_path.write_text(
        json.dumps(all_records, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    compact = [
        {
            "seed": row["seed"],
            "encoder": row["encoder"],
            "layers": row["layers"],
            "best_epoch": row["best_epoch"],
            "parameters": row["parameter_count"],
            "train_accuracy": row["train"]["accuracy"],
            "val_accuracy": row["val"]["accuracy"],
            "val_auc": row["val"]["roc_auc"],
            "test_accuracy": row.get("test", {}).get("accuracy"),
        }
        for row in all_records
    ]
    print(json.dumps(compact, indent=2, sort_keys=True, allow_nan=False))


def parse_variant(value: str) -> tuple[str, int]:
    try:
        encoder, layer_text = value.split(":", maxsplit=1)
        layers = int(layer_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Variant must be ENCODER:LAYERS") from exc
    if encoder not in {"local", "single", "reupload"}:
        raise argparse.ArgumentTypeError("Encoder must be local, single, or reupload")
    if layers < 0:
        raise argparse.ArgumentTypeError("Layers must be non-negative")
    return encoder, layers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "development")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--variant",
        nargs="+",
        type=parse_variant,
        default=[("local", 0), ("single", 1), ("reupload", 2)],
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--evaluate-test", action="store_true")
    parsed = parser.parse_args()
    parsed.dataset_root = parsed.dataset_root.resolve()
    parsed.output = parsed.output.resolve()
    main(parsed)
