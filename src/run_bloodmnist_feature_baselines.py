#!/usr/bin/env python3
"""Low-cost feature controls for the BloodMNIST QNN CA gap.

The saved BloodMNIST tensors are the 256-dimensional PCA features fed to the
amplitude encoder.  This script asks three focused questions on a nested
stratified holdout of those clean tensors:

* Does the pre-normalisation vector norm carry the class signal?
* Is class information still available after the exact L2 normalisation used
  by amplitude embedding?
* How much accuracy is lost by an explicit eight-dimensional bottleneck?

The nested holdout is independent of the classifiers fitted here, although the
upstream PCA transform was fitted on the full source-training split.  Results
therefore diagnose representation information; they are not a replacement for
the original untouched BloodMNIST test split.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_ROOT / "Blackdoor_1"
SOURCE_ROOT = PROJECT_ROOT / "data/MedMnist/single_detect_20260411/epsilon_0.8"


def load_clean(seed: int) -> tuple[np.ndarray, np.ndarray]:
    base = SOURCE_ROOT / f"seed_{seed}/layer_8_grid/t6_vs_t0/pr_0.1"
    meta = torch.load(
        base / f"poisoned_samples_meta_seed_{seed}_pr_0.1.pt",
        map_location="cpu",
        weights_only=False,
    )
    x_target = meta["target_clean_data"].numpy().astype(np.float64)
    x_nontarget = meta["non_target_clean_data"].numpy().astype(np.float64)
    x = np.vstack([x_target, x_nontarget])
    y = np.concatenate(
        [np.ones(len(x_target), dtype=int), np.zeros(len(x_nontarget), dtype=int)]
    )
    return x, y


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def evaluate_model(name: str, model, x_train, x_test, y_train, y_test) -> dict:
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(x_test)[:, 1]
    else:
        score = model.decision_function(x_test)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "roc_auc": float(roc_auc_score(y_test, score)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(output: Path, seeds: list[int]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for seed in seeds:
        x, y = load_clean(seed)
        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=seed + 2026,
            stratify=y,
        )
        x_train_raw, x_test_raw = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        x_train_l2, x_test_l2 = l2_normalize(x_train_raw), l2_normalize(x_test_raw)
        x_train_norm = np.log(np.linalg.norm(x_train_raw, axis=1, keepdims=True) + 1e-12)
        x_test_norm = np.log(np.linalg.norm(x_test_raw, axis=1, keepdims=True) + 1e-12)

        models = [
            (
                "norm_only_logreg",
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                x_train_norm,
                x_test_norm,
            ),
            (
                "raw256_logreg",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=5000, C=1.0),
                ),
                x_train_raw,
                x_test_raw,
            ),
            (
                "l2norm256_logreg",
                LogisticRegression(max_iter=5000, C=1.0),
                x_train_l2,
                x_test_l2,
            ),
            (
                "l2norm256_rbf_svm",
                SVC(kernel="rbf", C=10.0, gamma="scale", probability=True),
                x_train_l2,
                x_test_l2,
            ),
            (
                "l2norm_pca8_logreg",
                make_pipeline(
                    PCA(n_components=8, whiten=True, random_state=seed),
                    LogisticRegression(max_iter=5000, C=1.0),
                ),
                x_train_l2,
                x_test_l2,
            ),
        ]
        for name, model, x_train, x_test in models:
            result = evaluate_model(name, model, x_train, x_test, y_train, y_test)
            rows.append(
                {
                    "seed": seed,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    **result,
                }
            )

    write_csv(output / "feature_baselines_per_seed.csv", rows)
    summaries: list[dict] = []
    for model in sorted({row["model"] for row in rows}):
        selected = [row for row in rows if row["model"] == model]
        summaries.append(
            {
                "model": model,
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
    write_csv(output / "feature_baselines_summary.csv", summaries)
    receipt = {
        "scope": "Nested clean holdout within the saved BloodMNIST source-training PCA tensors",
        "upstream_preprocessing_caveat": (
            "The original StandardScaler and PCA were fitted before this nested split; "
            "the control measures retained representation information and is not an untouched-test estimate."
        ),
        "seeds": seeds,
        "results": summaries,
    }
    (output / "feature_baselines_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "paper14/results/bloodmnist_ca_diagnosis_v1",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    main(args.output.resolve(), args.seeds)
