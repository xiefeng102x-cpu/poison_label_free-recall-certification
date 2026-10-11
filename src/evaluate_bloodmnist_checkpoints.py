#!/usr/bin/env python3
"""Reconstruct the historical BloodMNIST split and audit held-out CA.

The source experiment did not save its fitted scaler/PCA objects or its
validation/test tensors.  This script reconstructs them from the official
BloodMNIST train split using the exact historical sampling, split, scaler and
PCA procedure.  Before reporting any checkpoint result, it validates the
reconstruction against the saved source-training tensors.

It then evaluates the saved QNN checkpoints at epochs 50/100/150/200 and a
small set of classical feature controls on the untouched historical test
split.  No QNN retraining is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
try:
    from medmnist import BloodMNIST
except ImportError:  # The released MedMNIST npz is sufficient for replay.
    BloodMNIST = None
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fast_circuit import run_circuit_batch


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_ROOT / "Blackdoor_1"
SOURCE_ROOT = PROJECT_ROOT / "data/MedMnist/single_detect_20260411/epsilon_0.8"
RUN_CIRCUIT_BATCH = run_circuit_batch


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_classes(dataset_root: Path) -> dict[int, np.ndarray]:
    if BloodMNIST is None:
        payload = np.load(dataset_root / "bloodmnist.npz")
        images = payload["train_images"]
        labels = payload["train_labels"].reshape(-1)
        return {
            class_id: (
                images[labels == class_id][:1000]
                .reshape(-1, 28 * 28 * 3)
                .astype(np.float32)
                / 255.0
            )
            for class_id in (6, 0)
        }

    dataset = BloodMNIST(split="train", download=False, root=str(dataset_root))
    selected: dict[int, list[np.ndarray]] = {6: [], 0: []}
    for image, label in dataset:
        class_id = int(label[0])
        if class_id in selected and len(selected[class_id]) < 1000:
            pixels = np.asarray(image).reshape(-1).astype(np.float32) / 255.0
            selected[class_id].append(pixels)
        if all(len(rows) >= 1000 for rows in selected.values()):
            break
    return {
        class_id: np.stack(rows).astype(np.float32, copy=False)
        for class_id, rows in selected.items()
    }


def reconstruct_split(class_data: dict[int, np.ndarray], seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    raw: dict[str, dict[int, np.ndarray]] = {
        "train": {},
        "val": {},
        "test": {},
    }
    for class_id in (6, 0):
        samples = class_data[class_id]
        n_samples = len(samples)
        n_train = int(n_samples * 0.6)
        n_val = int(n_samples * 0.2)
        indices = torch.randperm(
            n_samples, generator=torch.Generator().manual_seed(seed)
        ).numpy()
        raw["train"][class_id] = samples[indices[:n_train]]
        raw["val"][class_id] = samples[indices[n_train : n_train + n_val]]
        raw["test"][class_id] = samples[indices[n_train + n_val :]]

    train_combined = np.concatenate([raw["train"][6], raw["train"][0]], axis=0)
    scaler = StandardScaler().fit(train_combined)
    pca = PCA(n_components=256).fit(scaler.transform(train_combined))
    processed: dict[str, dict[int, np.ndarray]] = {
        split: {
            class_id: pca.transform(scaler.transform(raw[split][class_id])).astype(
                np.float32, copy=False
            )
            for class_id in (6, 0)
        }
        for split in ("train", "val", "test")
    }
    return {
        "raw": raw,
        "processed": processed,
        "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
    }


def arrays_for_split(splits: dict[str, dict[int, np.ndarray]], split: str):
    x = np.concatenate([splits[split][6], splits[split][0]], axis=0)
    y = np.concatenate(
        [
            np.ones(len(splits[split][6]), dtype=np.int64),
            np.zeros(len(splits[split][0]), dtype=np.int64),
        ]
    )
    return x, y


def reconstruction_error(processed_train: dict[int, np.ndarray], seed: int) -> dict:
    base = SOURCE_ROOT / f"seed_{seed}/layer_8_grid/t6_vs_t0/pr_0.1"
    meta = torch.load(
        base / f"poisoned_samples_meta_seed_{seed}_pr_0.1.pt",
        map_location="cpu",
        weights_only=False,
    )
    source_non_target = meta["non_target_clean_data"].numpy()
    clean_indices = torch.randperm(
        len(processed_train[6]),
        generator=torch.Generator().manual_seed(seed + 10007),
    ).numpy()[int(len(processed_train[6]) * 0.1) :]
    source_target = meta["target_clean_data"].numpy()
    rebuilt_target = processed_train[6][clean_indices]
    rebuilt_non_target = processed_train[0]
    return {
        "target_max_abs_error": float(np.max(np.abs(rebuilt_target - source_target))),
        "target_mean_abs_error": float(np.mean(np.abs(rebuilt_target - source_target))),
        "non_target_max_abs_error": float(
            np.max(np.abs(rebuilt_non_target - source_non_target))
        ),
        "non_target_mean_abs_error": float(
            np.mean(np.abs(rebuilt_non_target - source_non_target))
        ),
    }


def checkpoint_scores(state: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = state["weights"].detach().cpu().numpy().astype(np.float64)
    quantum_features = RUN_CIRCUIT_BATCH(x, weights, n_layers=8, n_qubits=8)[:, :8]
    w1 = state["classifier.0.weight"].detach().cpu().numpy()
    b1 = state["classifier.0.bias"].detach().cpu().numpy()
    w2 = state["classifier.3.weight"].detach().cpu().numpy()
    b2 = state["classifier.3.bias"].detach().cpu().numpy()
    hidden = np.maximum(quantum_features @ w1.T + b1, 0.0)
    logits = hidden @ w2.T + b2
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= np.sum(probability, axis=1, keepdims=True)
    return np.argmax(logits, axis=1), probability[:, 1]


def metric_row(y: np.ndarray, prediction: np.ndarray, score: np.ndarray) -> dict:
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


def evaluate_checkpoints(seed: int, processed: dict) -> list[dict]:
    base = SOURCE_ROOT / f"seed_{seed}/layer_8_grid/t6_vs_t0/pr_0.1/models"
    split_arrays = {
        split: arrays_for_split(processed, split) for split in ("train", "val", "test")
    }
    rows: list[dict] = []
    for epoch in (50, 100, 150, 200):
        state = torch.load(
            base / f"clean_model_seed_{epoch}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for split, (x, y) in split_arrays.items():
            prediction, score = checkpoint_scores(state, x)
            rows.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "split": split,
                    "n": len(y),
                    **metric_row(y, prediction, score),
                }
            )
    return rows


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def heldout_baselines(seed: int, reconstruction: dict) -> list[dict]:
    raw_train, y_train = arrays_for_split(reconstruction["raw"], "train")
    raw_test, y_test = arrays_for_split(reconstruction["raw"], "test")
    pca_train, _ = arrays_for_split(reconstruction["processed"], "train")
    pca_test, _ = arrays_for_split(reconstruction["processed"], "test")
    l2_train, l2_test = l2_normalize(pca_train), l2_normalize(pca_test)
    norm_train = np.log(np.linalg.norm(pca_train, axis=1, keepdims=True) + 1e-12)
    norm_test = np.log(np.linalg.norm(pca_test, axis=1, keepdims=True) + 1e-12)

    candidates = [
        (
            "raw2352_logreg",
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
            raw_train,
            raw_test,
        ),
        (
            "pca256_logreg",
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
            pca_train,
            pca_test,
        ),
        (
            "norm_only_logreg",
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
            norm_train,
            norm_test,
        ),
        (
            "l2norm256_logreg",
            LogisticRegression(max_iter=5000, C=1.0),
            l2_train,
            l2_test,
        ),
        (
            "l2norm256_rbf_svm",
            SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=seed),
            l2_train,
            l2_test,
        ),
        (
            "l2norm_pca8_logreg",
            make_pipeline(
                PCA(n_components=8, whiten=True, random_state=seed),
                LogisticRegression(max_iter=5000, C=1.0),
            ),
            l2_train,
            l2_test,
        ),
    ]
    rows: list[dict] = []
    for name, model, x_train, x_test in candidates:
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(x_test)[:, 1]
        else:
            score = model.decision_function(x_test)
        rows.append(
            {
                "seed": seed,
                "model": name,
                "n_train": len(y_train),
                "n_test": len(y_test),
                **metric_row(y_test, prediction, score),
            }
        )
    return rows


def main(dataset_root: Path, output: Path, seeds: list[int]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    class_data = collect_classes(dataset_root)
    checkpoint_rows: list[dict] = []
    baseline_rows: list[dict] = []
    receipts: list[dict] = []
    for seed in seeds:
        reconstruction = reconstruct_split(class_data, seed)
        errors = reconstruction_error(reconstruction["processed"]["train"], seed)
        max_error = max(errors["target_max_abs_error"], errors["non_target_max_abs_error"])
        # PCA uses threaded BLAS/SVD and the historical run was produced on a
        # different host.  Few-milliscale drift is expected even when the
        # sample order and PCA basis are the same; a sign/permutation mismatch
        # would instead produce order-one errors.
        if max_error > 5e-3:
            raise RuntimeError(
                f"Seed {seed} reconstruction does not match saved tensors: {errors}"
            )
        checkpoint_rows.extend(evaluate_checkpoints(seed, reconstruction["processed"]))
        baseline_rows.extend(heldout_baselines(seed, reconstruction))
        receipts.append(
            {
                "seed": seed,
                "class_counts": {str(k): len(v) for k, v in class_data.items()},
                "split_counts": {
                    split: {
                        str(k): len(v)
                        for k, v in reconstruction["raw"][split].items()
                    }
                    for split in ("train", "val", "test")
                },
                "pca_explained_variance": reconstruction["explained_variance"],
                "saved_training_tensor_match": errors,
            }
        )

    write_csv(output / "checkpoint_ca.csv", checkpoint_rows)
    write_csv(output / "heldout_feature_baselines.csv", baseline_rows)
    receipt = {
        "dataset_root": str(dataset_root),
        "source_scope": "Official BloodMNIST train split, historical 60/20/20 split",
        "reconstruction_acceptance_threshold_max_abs_error": 5e-3,
        "receipts": receipts,
    }
    (output / "reconstruction_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    test_rows = [row for row in checkpoint_rows if row["split"] == "test"]
    print(
        json.dumps(
            {"reconstruction": receipt, "checkpoint_test": test_rows, "baselines": baseline_rows},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


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
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()
    main(args.dataset_root.resolve(), args.output.resolve(), args.seeds)
