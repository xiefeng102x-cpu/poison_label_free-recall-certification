#!/usr/bin/env python3
"""Separate BloodMNIST QNN feature loss from classifier-head limitations.

For each saved clean-QNN checkpoint, extract the eight local-Z observables used
by the historical classifier and the 28 additional ZZ correlators already
available from the same circuit state.  Refit small post-hoc classifiers on the
historical clean training split and evaluate on its untouched test split.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import evaluate_bloodmnist_checkpoints as heldout


PROJECT_ROOT = heldout.PROJECT_ROOT
SOURCE_ROOT = heldout.SOURCE_ROOT


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def saved_head_prediction(state: dict, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def evaluate_readout(
    name: str,
    model,
    train_features: np.ndarray,
    test_features: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    model.fit(train_features, y_train)
    prediction = model.predict(test_features)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(test_features)[:, 1]
    else:
        score = model.decision_function(test_features)
    return {"readout": name, **heldout.metric_row(y_test, prediction, score)}


def run_seed(class_data: dict[int, np.ndarray], seed: int) -> list[dict]:
    reconstruction = heldout.reconstruct_split(class_data, seed)
    errors = heldout.reconstruction_error(reconstruction["processed"]["train"], seed)
    if max(errors["target_max_abs_error"], errors["non_target_max_abs_error"]) > 5e-3:
        raise RuntimeError(f"Seed {seed} reconstruction mismatch: {errors}")
    x_train, y_train = heldout.arrays_for_split(reconstruction["processed"], "train")
    x_test, y_test = heldout.arrays_for_split(reconstruction["processed"], "test")
    joined = np.concatenate([x_train, x_test], axis=0)
    n_train = len(x_train)

    rows: list[dict] = []
    model_root = SOURCE_ROOT / f"seed_{seed}/layer_8_grid/t6_vs_t0/pr_0.1/models"
    for epoch in (50, 100, 150, 200):
        state = torch.load(
            model_root / f"clean_model_seed_{epoch}.pt",
            map_location="cpu",
            weights_only=True,
        )
        weights = state["weights"].detach().cpu().numpy().astype(np.float64)
        features = heldout.RUN_CIRCUIT_BATCH(
            joined, weights, n_layers=8, n_qubits=8
        )
        z_train, z_test = features[:n_train, :8], features[n_train:, :8]
        zzz_train, zzz_test = features[:n_train], features[n_train:]

        prediction, score = saved_head_prediction(state, z_test)
        results = [
            {
                "readout": "saved_head_z8",
                **heldout.metric_row(y_test, prediction, score),
            },
            evaluate_readout(
                "refit_logreg_z8",
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                z_train,
                z_test,
                y_train,
                y_test,
            ),
            evaluate_readout(
                "refit_rbf_z8",
                SVC(C=10.0, kernel="rbf", gamma="scale", probability=True, random_state=seed),
                z_train,
                z_test,
                y_train,
                y_test,
            ),
            evaluate_readout(
                "refit_logreg_z_plus_zz36",
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                zzz_train,
                zzz_test,
                y_train,
                y_test,
            ),
            evaluate_readout(
                "refit_rbf_z_plus_zz36",
                SVC(C=10.0, kernel="rbf", gamma="scale", probability=True, random_state=seed),
                zzz_train,
                zzz_test,
                y_train,
                y_test,
            ),
        ]
        rows.extend(
            {
                "seed": seed,
                "epoch": epoch,
                "n_train": len(y_train),
                "n_test": len(y_test),
                **result,
            }
            for result in results
        )
    return rows


def main(dataset_root: Path, output: Path, seeds: list[int]) -> None:
    class_data = heldout.collect_classes(dataset_root)
    rows: list[dict] = []
    for seed in seeds:
        rows.extend(run_seed(class_data, seed))
    write_csv(output / "qnn_feature_readouts.csv", rows)

    summaries: list[dict] = []
    for epoch in (50, 100, 150, 200):
        for readout in sorted({row["readout"] for row in rows}):
            selected = [
                row for row in rows if row["epoch"] == epoch and row["readout"] == readout
            ]
            summaries.append(
                {
                    "epoch": epoch,
                    "readout": readout,
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
    write_csv(output / "qnn_feature_readouts_summary.csv", summaries)
    receipt = {
        "scope": "Post-hoc readouts on historical clean train and untouched test splits",
        "seeds": seeds,
        "results": summaries,
    }
    (output / "qnn_feature_readouts_receipt.json").write_text(
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
    args = parser.parse_args()
    main(args.dataset_root.resolve(), args.output.resolve(), args.seeds)
