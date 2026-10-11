#!/usr/bin/env python3
"""Low-cost held-out clean-CA intervention using X/Y/Z QNN observables.

The circuit weights stay frozen at the archived clean epoch-200 checkpoint.
For each seed, classifier hyperparameters are selected only on the historical
validation split, then evaluated once on the untouched test split.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import evaluate_bloodmnist_checkpoints as heldout
from run_bloodmnist_corrected_qnn import features_numpy


PROJECT_ROOT = heldout.PROJECT_ROOT
SOURCE_ROOT = heldout.SOURCE_ROOT


FEATURE_SLICES = {
    "Z": np.r_[16:24],
    "X": np.r_[0:8],
    "Y": np.r_[8:16],
    "XZ": np.r_[0:8, 16:24],
    "XY": np.r_[0:16],
    "YZ": np.r_[8:24],
    "XYZ": np.r_[0:24],
}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_model(model, x: np.ndarray, y: np.ndarray) -> dict:
    prediction = model.predict(x)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(x)[:, 1]
    else:
        score = model.decision_function(x)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "roc_auc": float(roc_auc_score(y, score)),
    }


def candidate_model(family: str, c_value: float, seed: int):
    if family == "logreg":
        estimator = LogisticRegression(C=c_value, max_iter=5000)
    elif family == "rbf_svm":
        estimator = SVC(
            C=c_value,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown family: {family}")
    return make_pipeline(StandardScaler(), estimator)


def main(dataset_root: Path, output: Path, seeds: list[int]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    class_data = heldout.collect_classes(dataset_root)
    rows: list[dict] = []
    selection_rows: list[dict] = []
    reconstruction_receipts: list[dict] = []

    for seed in seeds:
        reconstruction = heldout.reconstruct_split(class_data, seed)
        errors = heldout.reconstruction_error(reconstruction["processed"]["train"], seed)
        if max(errors["target_max_abs_error"], errors["non_target_max_abs_error"]) > 5e-3:
            raise RuntimeError(f"Seed {seed} reconstruction mismatch: {errors}")
        reconstruction_receipts.append({"seed": seed, **errors})

        split_arrays = {
            split: heldout.arrays_for_split(reconstruction["processed"], split)
            for split in ("train", "val", "test")
        }
        lengths = {split: len(split_arrays[split][1]) for split in split_arrays}
        joined = np.concatenate([split_arrays[s][0] for s in ("train", "val", "test")])
        state = torch.load(
            SOURCE_ROOT
            / f"seed_{seed}/layer_8_grid/t6_vs_t0/pr_0.1/models/clean_model_seed_200.pt",
            map_location="cpu",
            weights_only=True,
        )
        weights = torch.tensor(state["weights"].numpy(), dtype=torch.float64)
        all_xyz = features_numpy(joined.astype(np.float64), weights)
        n_train, n_val = lengths["train"], lengths["val"]
        xyz = {
            "train": all_xyz[:n_train],
            "val": all_xyz[n_train : n_train + n_val],
            "test": all_xyz[n_train + n_val :],
        }

        for feature_name, columns in FEATURE_SLICES.items():
            for family, c_grid in (
                ("logreg", [0.01, 0.1, 1.0, 10.0, 100.0]),
                ("rbf_svm", [0.1, 1.0, 10.0, 100.0]),
            ):
                candidates = []
                for c_value in c_grid:
                    model = candidate_model(family, c_value, seed)
                    model.fit(xyz["train"][:, columns], split_arrays["train"][1])
                    validation = score_model(
                        model, xyz["val"][:, columns], split_arrays["val"][1]
                    )
                    candidates.append((validation["accuracy"], validation["roc_auc"], c_value, model))
                    selection_rows.append(
                        {
                            "seed": seed,
                            "feature_set": feature_name,
                            "family": family,
                            "c_value": c_value,
                            **{f"validation_{key}": value for key, value in validation.items()},
                        }
                    )
                _, _, selected_c, selected_model = max(
                    candidates, key=lambda item: (item[0], item[1], -item[2])
                )
                validation = score_model(
                    selected_model, xyz["val"][:, columns], split_arrays["val"][1]
                )
                test = score_model(
                    selected_model, xyz["test"][:, columns], split_arrays["test"][1]
                )
                rows.append(
                    {
                        "seed": seed,
                        "checkpoint_epoch": 200,
                        "feature_set": feature_name,
                        "n_features": len(columns),
                        "family": family,
                        "selected_c": selected_c,
                        **{f"validation_{key}": value for key, value in validation.items()},
                        **{f"test_{key}": value for key, value in test.items()},
                    }
                )

    write_csv(output / "xyz_readout_results.csv", rows)
    write_csv(output / "xyz_readout_validation_grid.csv", selection_rows)
    summary: list[dict] = []
    for feature_name in FEATURE_SLICES:
        for family in ("logreg", "rbf_svm"):
            selected = [
                row
                for row in rows
                if row["feature_set"] == feature_name and row["family"] == family
            ]
            summary.append(
                {
                    "feature_set": feature_name,
                    "family": family,
                    "n_seeds": len(selected),
                    "test_accuracy_mean": float(
                        np.mean([row["test_accuracy"] for row in selected])
                    ),
                    "test_accuracy_population_sd": float(
                        np.std([row["test_accuracy"] for row in selected], ddof=0)
                    ),
                    "test_balanced_accuracy_mean": float(
                        np.mean([row["test_balanced_accuracy"] for row in selected])
                    ),
                    "test_roc_auc_mean": float(
                        np.mean([row["test_roc_auc"] for row in selected])
                    ),
                }
            )
    write_csv(output / "xyz_readout_summary.csv", summary)
    receipt = {
        "scope": "Frozen epoch-200 clean QNN; validation-selected readout; untouched test",
        "seeds": seeds,
        "reconstruction": reconstruction_receipts,
        "results": summary,
    }
    (output / "xyz_readout_receipt.json").write_text(
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
        default=PROJECT_ROOT / "paper14/results/bloodmnist_ca_improvement_v1/phase1_xyz",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    main(args.dataset_root.resolve(), args.output.resolve(), args.seeds)
