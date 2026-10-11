#!/usr/bin/env python3
"""Evaluate archived adaptive BloodMNIST QNN weights on untouched test data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_bloodmnist_checkpoints as heldout


PROJECT_ROOT = heldout.PROJECT_ROOT
SOURCE_ROOT = heldout.SOURCE_ROOT
ADAPTIVE_ROOT = (
    PROJECT_ROOT / "qml_adaptive_nc_project/paper_nc_adaptive/results/training_time_bloodmnist"
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def predict(state: dict, weights: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = heldout.RUN_CIRCUIT_BATCH(x, weights, n_layers=8, n_qubits=8)[:, :8]
    w1 = state["classifier.0.weight"].detach().cpu().numpy()
    b1 = state["classifier.0.bias"].detach().cpu().numpy()
    w2 = state["classifier.3.weight"].detach().cpu().numpy()
    b2 = state["classifier.3.bias"].detach().cpu().numpy()
    hidden = np.maximum(z @ w1.T + b1, 0.0)
    logits = hidden @ w2.T + b2
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= np.sum(probability, axis=1, keepdims=True)
    return np.argmax(logits, axis=1), probability[:, 1]


def main(dataset_root: Path, output: Path, seeds: list[int], lambdas: list[float]) -> None:
    class_data = heldout.collect_classes(dataset_root)
    rows: list[dict] = []
    archive_differences: list[float] = []
    for seed in seeds:
        reconstruction = heldout.reconstruct_split(class_data, seed)
        errors = heldout.reconstruction_error(reconstruction["processed"]["train"], seed)
        if max(errors["target_max_abs_error"], errors["non_target_max_abs_error"]) > 5e-3:
            raise RuntimeError(f"Seed {seed} reconstruction mismatch: {errors}")
        x_test, y_test = heldout.arrays_for_split(reconstruction["processed"], "test")

        source_base = SOURCE_ROOT / f"seed_{seed}/layer_8_grid/t6_vs_t0/pr_0.1"
        state = torch.load(
            source_base / "models/clean_model_seed_200.pt",
            map_location="cpu",
            weights_only=True,
        )
        meta = torch.load(
            source_base / f"poisoned_samples_meta_seed_{seed}_pr_0.1.pt",
            map_location="cpu",
            weights_only=False,
        )
        x_archive = np.concatenate(
            [meta["target_clean_data"].numpy(), meta["non_target_clean_data"].numpy()],
            axis=0,
        )
        y_archive = np.concatenate(
            [
                np.ones(len(meta["target_clean_data"]), dtype=np.int64),
                np.zeros(len(meta["non_target_clean_data"]), dtype=np.int64),
            ]
        )

        for lambda_r in lambdas:
            tag = f"{lambda_r:.1f}"
            weights = np.load(
                ADAPTIVE_ROOT / f"weights_seed={seed}_L8_lr{tag}.npy"
            ).astype(np.float64)
            result = json.loads(
                (ADAPTIVE_ROOT / f"result_seed={seed}_L8_lr{tag}.json").read_text(
                    encoding="utf-8"
                )
            )
            for split, x, y in (
                ("archived_clean_pool", x_archive, y_archive),
                ("untouched_test", x_test, y_test),
            ):
                prediction, score = predict(state, weights, x)
                metrics = heldout.metric_row(y, prediction, score)
                rows.append(
                    {
                        "seed": seed,
                        "lambda_r": lambda_r,
                        "split": split,
                        "n": len(y),
                        **metrics,
                    }
                )
                if split == "archived_clean_pool":
                    archive_differences.append(
                        abs(metrics["accuracy"] - float(result["final"]["ca"]))
                    )

    write_csv(output / "adaptive_qnn_heldout_ca.csv", rows)
    summaries: list[dict] = []
    for lambda_r in lambdas:
        for split in ("archived_clean_pool", "untouched_test"):
            selected = [
                row
                for row in rows
                if row["lambda_r"] == lambda_r and row["split"] == split
            ]
            summaries.append(
                {
                    "lambda_r": lambda_r,
                    "split": split,
                    "n_seeds": len(selected),
                    "accuracy_mean": float(np.mean([row["accuracy"] for row in selected])),
                    "accuracy_population_sd": float(
                        np.std([row["accuracy"] for row in selected], ddof=0)
                    ),
                    "balanced_accuracy_mean": float(
                        np.mean([row["balanced_accuracy"] for row in selected])
                    ),
                    "roc_auc_mean": float(np.mean([row["roc_auc"] for row in selected])),
                }
            )
    write_csv(output / "adaptive_qnn_heldout_ca_summary.csv", summaries)
    receipt = {
        "scope": "Archived adaptive weights with frozen historical classifier",
        "seeds": seeds,
        "lambdas": lambdas,
        "max_abs_difference_from_archived_clean_pool_ca": max(archive_differences),
        "results": summaries,
    }
    (output / "adaptive_qnn_heldout_ca_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "paper14/results/bloodmnist_ca_diagnosis_v1/heldout_checkpoint_audit"
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    args = parser.parse_args()
    main(args.dataset_root.resolve(), args.output.resolve(), args.seeds, args.lambdas)
