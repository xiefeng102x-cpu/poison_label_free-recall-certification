#!/usr/bin/env python3
"""Pauli-weight and compression audit for the adaptive BloodMNIST victim.

The selected victim is an eight-qubit product state with local Bloch vectors
(X, Y, Z) = (cos(alpha), 0, sin(alpha)).  Every Pauli string containing Y is
therefore identically zero.  The 6,560 non-zero X/Z strings are an exact
effective representation of the 65,535-dimensional non-identity Pauli space.

Paired clean/triggered examples are split by the historical train/validation/
test partitions.  PCA and supervised feature ranking are fit on train only;
probe regularization is selected on validation only; test is scored afterward.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import run_bloodmnist_new_encoder_adaptive as adaptive
import train_bloodmnist_clean_encoder as clean_encoder


PROJECT_ROOT = adaptive.PROJECT_ROOT
ADAPTIVE_ROOT = (
    PROJECT_ROOT / "paper14/results/bloodmnist_new_encoder_adaptive_v1"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "paper14/results/bloodmnist_pauli_compression_v1"


def adaptive_checkpoint_path(seed: int) -> Path:
    folder = "delayed_test" if seed == 42 else "confirmation"
    matches = sorted((ADAPTIVE_ROOT / folder).glob(f"seed={seed}_*.pt"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one adaptive checkpoint for seed {seed}: {matches}")
    return matches[0]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def local_xz(model: clean_encoder.CleanQNN, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        batch = torch.as_tensor(x, dtype=torch.float32)
        latent = model.encoder.projected_latent(batch)
        alpha = (math.pi / 2) * latent
        local = torch.stack((torch.cos(alpha), torch.sin(alpha)), dim=2)
    return local.cpu().numpy().astype(np.float32, copy=False)


def pauli_xz_by_weight(
    local: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, list[str]]]:
    n_samples, n_qubits, axes = local.shape
    if axes != 2:
        raise ValueError("Expected local X/Z coordinates")
    features: dict[int, np.ndarray] = {}
    labels: dict[int, list[str]] = {}
    for weight in range(1, n_qubits + 1):
        columns = []
        names = []
        for qubits in itertools.combinations(range(n_qubits), weight):
            for operators in itertools.product(range(2), repeat=weight):
                value = np.ones(n_samples, dtype=np.float32)
                name = ["I"] * n_qubits
                for qubit, operator in zip(qubits, operators):
                    value *= local[:, qubit, operator]
                    name[qubit] = "X" if operator == 0 else "Z"
                columns.append(value)
                names.append("".join(name))
        features[weight] = np.stack(columns, axis=1)
        labels[weight] = names
    return features, labels


def make_probe(detector: str, c_value: float, random_seed: int):
    if detector == "logistic":
        classifier = LogisticRegression(
            C=c_value,
            penalty="l2",
            solver="liblinear",
            max_iter=5000,
            random_state=random_seed,
        )
    elif detector == "rbf_svm":
        classifier = SVC(
            C=c_value,
            kernel="rbf",
            gamma="scale",
            probability=False,
            random_state=random_seed,
        )
    else:
        raise ValueError(detector)
    return Pipeline(
        [
            ("center", StandardScaler(with_mean=True, with_std=False)),
            ("model", classifier),
        ]
    )


def scores(estimator, x: np.ndarray) -> np.ndarray:
    return np.asarray(estimator.decision_function(x), dtype=np.float64)


def tpr_at_fpr(y: np.ndarray, score: np.ndarray, limit: float) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    eligible = tpr[fpr <= limit + 1e-15]
    return float(np.max(eligible)) if len(eligible) else 0.0


def metrics(y: np.ndarray, score: np.ndarray) -> dict:
    return {
        "auc": float(roc_auc_score(y, score)),
        "aucpr": float(average_precision_score(y, score)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, (score >= 0).astype(int))
        ),
        "tpr_at_fpr_01": tpr_at_fpr(y, score, 0.01),
        "tpr_at_fpr_05": tpr_at_fpr(y, score, 0.05),
    }


def select_c(
    detector: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    c_grid: list[float],
    seed: int,
) -> tuple[float, dict]:
    candidates = []
    for c_value in c_grid:
        estimator = make_probe(detector, c_value, seed)
        estimator.fit(x_train, y_train)
        candidate_metrics = metrics(y_val, scores(estimator, x_val))
        candidates.append((candidate_metrics["auc"], c_value, candidate_metrics))
    candidates.sort(key=lambda row: (-row[0], row[1]))
    return float(candidates[0][1]), candidates[0][2]


def bootstrap_auc(
    y: np.ndarray, score: np.ndarray, n_pairs: int, repetitions: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        selected = rng.integers(0, n_pairs, size=n_pairs)
        rows = np.concatenate([selected, selected + n_pairs])
        values[repetition] = roc_auc_score(y[rows], score[rows])
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def evaluate_probe(
    detector: str,
    feature_sets: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    c_grid: list[float],
    seed: int,
    bootstrap_repetitions: int,
) -> dict:
    selected_c, validation = select_c(
        detector,
        feature_sets["train"],
        labels["train"],
        feature_sets["val"],
        labels["val"],
        c_grid,
        seed,
    )
    train_val_x = np.concatenate(
        [feature_sets["train"], feature_sets["val"]], axis=0
    )
    train_val_y = np.concatenate([labels["train"], labels["val"]], axis=0)
    estimator = make_probe(detector, selected_c, seed)
    estimator.fit(train_val_x, train_val_y)
    test_score = scores(estimator, feature_sets["test"])
    test = metrics(labels["test"], test_score)
    n_pairs = len(labels["test"]) // 2
    ci_low, ci_high = bootstrap_auc(
        labels["test"],
        test_score,
        n_pairs=n_pairs,
        repetitions=bootstrap_repetitions,
        seed=seed + 91001,
    )
    test["auc_ci_low"] = ci_low
    test["auc_ci_high"] = ci_high
    return {"selected_c": selected_c, "validation": validation, "test": test}


def split_rows(clean: np.ndarray, attack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([clean, attack], axis=0)
    y = np.concatenate(
        [np.zeros(len(clean), dtype=int), np.ones(len(attack), dtype=int)]
    )
    return x, y


def feature_slices(counts: dict[str, int]) -> dict[str, slice]:
    output = {}
    cursor = 0
    for split in ("train", "val", "test"):
        count = counts[split]
        output[split] = slice(cursor, cursor + 2 * count)
        cursor += 2 * count
    return output


def row_from_result(
    seed: int,
    role: str,
    path: str,
    representation: str,
    dimension: int,
    detector: str,
    result: dict,
    **extra,
) -> dict:
    return {
        "seed": seed,
        "role": role,
        "path": path,
        "representation": representation,
        "dimension": dimension,
        "detector": detector,
        "selected_c": result["selected_c"],
        **{f"validation_{key}": value for key, value in result["validation"].items()},
        **{f"test_{key}": value for key, value in result["test"].items()},
        **extra,
    }


def run_seed(args: argparse.Namespace, seed: int) -> tuple[list[dict], list[dict], dict]:
    checkpoint_path = adaptive_checkpoint_path(seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = clean_encoder.CleanQNN("local", 0)
    model.load_state_dict(checkpoint["state_dict"])
    splits, data_receipt = adaptive.load_data(seed, args.dataset_root, checkpoint)
    role = "development" if seed == 42 else "confirmation"
    counts = {split: len(splits[split]["non_target"]) for split in ("train", "val", "test")}
    all_local = []
    labels_by_split = {}
    for split in ("train", "val", "test"):
        clean_local = local_xz(model, splits[split]["non_target"])
        attack_local = local_xz(model, splits[split]["triggered_non_target"])
        all_local.append(np.concatenate([clean_local, attack_local], axis=0))
        labels_by_split[split] = np.concatenate(
            [np.zeros(counts[split], dtype=int), np.ones(counts[split], dtype=int)]
        )
    combined_local = np.concatenate(all_local, axis=0)
    slices = feature_slices(counts)
    exact_by_weight, labels_by_weight = pauli_xz_by_weight(combined_local)
    expected_counts = {
        weight: math.comb(8, weight) * 2**weight for weight in range(1, 9)
    }
    if any(exact_by_weight[w].shape[1] != expected_counts[w] for w in range(1, 9)):
        raise RuntimeError("Pauli weight feature count mismatch")
    full = np.concatenate([exact_by_weight[w] for w in range(1, 9)], axis=1)
    if full.shape[1] != 3**8 - 1:
        raise RuntimeError("Effective full Pauli dimension mismatch")

    c_grid = [float(value) for value in args.c_grid]
    rows: list[dict] = []
    l1_rows: list[dict] = []

    def feature_dict(matrix: np.ndarray) -> dict[str, np.ndarray]:
        return {split: matrix[slices[split]] for split in ("train", "val", "test")}

    def add_probe(
        path: str,
        representation: str,
        matrix: np.ndarray,
        detector: str = "logistic",
        **extra,
    ) -> None:
        result = evaluate_probe(
            detector,
            feature_dict(matrix),
            labels_by_split,
            c_grid,
            seed,
            args.bootstrap_repetitions,
        )
        rows.append(
            row_from_result(
                seed,
                role,
                path,
                representation,
                matrix.shape[1],
                detector,
                result,
                **extra,
            )
        )

    z_only = combined_local[:, :, 1]
    local_xz_matrix = combined_local.reshape(len(combined_local), -1)
    add_probe("local", "Z_only", z_only)
    add_probe("local", "local_XZ", local_xz_matrix)
    cumulative = []
    for weight in range(1, 9):
        exact = exact_by_weight[weight]
        cumulative.append(exact)
        add_probe(
            "exact_weight",
            f"weight_{weight}",
            exact,
            pauli_weight=weight,
            physical_full_dimension=4**8 - 1,
            effective_full_dimension=3**8 - 1,
        )
        add_probe(
            "cumulative_weight",
            f"weight_le_{weight}",
            np.concatenate(cumulative, axis=1),
            pauli_weight=weight,
            physical_full_dimension=4**8 - 1,
            effective_full_dimension=3**8 - 1,
        )

    # Nonlinear controls at the two local boundaries and effective full space.
    add_probe("local", "Z_only", z_only, detector="rbf_svm")
    add_probe("local", "local_XZ", local_xz_matrix, detector="rbf_svm")
    add_probe("full", "full_effective_XZ", full, detector="rbf_svm")

    train_slice = slices["train"]
    pca_components = min(args.pca_max_components, full[train_slice].shape[0] - 1)
    pca = PCA(
        n_components=pca_components,
        svd_solver="randomized",
        random_state=seed + 4301,
    )
    pca.fit(full[train_slice])
    pca_all = pca.transform(full).astype(np.float32, copy=False)
    explained = np.cumsum(pca.explained_variance_ratio_)
    compression_dims = [
        dimension
        for dimension in args.compression_dims
        if dimension <= pca_components
    ]
    for dimension in compression_dims:
        add_probe(
            "pca",
            f"pca_{dimension}",
            pca_all[:, :dimension],
            explained_variance=float(explained[dimension - 1]),
        )

    train_full = full[train_slice]
    train_labels = labels_by_split["train"]
    clean_train = train_full[train_labels == 0]
    attack_train = train_full[train_labels == 1]
    fisher = (attack_train.mean(axis=0) - clean_train.mean(axis=0)) ** 2 / (
        attack_train.var(axis=0) + clean_train.var(axis=0) + 1e-12
    )
    fisher_order = np.argsort(-fisher, kind="stable")
    all_labels = sum((labels_by_weight[w] for w in range(1, 9)), [])
    for dimension in args.compression_dims:
        selected = fisher_order[:dimension]
        add_probe(
            "fisher",
            f"fisher_{dimension}",
            full[:, selected],
            top_feature=all_labels[int(selected[0])],
            top_feature_score=float(fisher[int(selected[0])]),
        )

    rng = np.random.default_rng(seed + 4703)
    for dimension in args.random_dims:
        if dimension > full.shape[1]:
            continue
        aucs = []
        for repetition in range(args.random_repeats):
            selected = rng.choice(full.shape[1], dimension, replace=False)
            result = evaluate_probe(
                "logistic",
                feature_dict(full[:, selected]),
                labels_by_split,
                c_grid,
                seed + repetition,
                args.bootstrap_repetitions,
            )
            aucs.append(float(result["test"]["auc"]))
        rows.append(
            {
                "seed": seed,
                "role": role,
                "path": "random",
                "representation": f"random_{dimension}",
                "dimension": dimension,
                "detector": "logistic",
                "selected_c": "per_repeat_validation_selected",
                "validation_auc": "",
                "validation_aucpr": "",
                "validation_balanced_accuracy": "",
                "validation_tpr_at_fpr_01": "",
                "validation_tpr_at_fpr_05": "",
                "test_auc": float(np.mean(aucs)),
                "test_aucpr": "",
                "test_balanced_accuracy": "",
                "test_tpr_at_fpr_01": "",
                "test_tpr_at_fpr_05": "",
                "test_auc_ci_low": float(np.quantile(aucs, 0.025)),
                "test_auc_ci_high": float(np.quantile(aucs, 0.975)),
                "random_repeats": args.random_repeats,
            }
        )

    # Sparse path.  C is intentionally swept rather than validation-selected;
    # each point describes a different sparsity operating point.
    feature_sets = feature_dict(full)
    train_val_x = np.concatenate(
        [feature_sets["train"], feature_sets["val"]], axis=0
    )
    train_val_y = np.concatenate(
        [labels_by_split["train"], labels_by_split["val"]], axis=0
    )
    for c_value in args.l1_c_grid:
        train_model = Pipeline(
            [
                ("center", StandardScaler(with_mean=True, with_std=False)),
                (
                    "model",
                    LogisticRegression(
                        C=c_value,
                        penalty="l1",
                        solver="liblinear",
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            ]
        )
        train_model.fit(feature_sets["train"], labels_by_split["train"])
        validation = metrics(
            labels_by_split["val"], scores(train_model, feature_sets["val"])
        )
        model = Pipeline(
            [
                ("center", StandardScaler(with_mean=True, with_std=False)),
                (
                    "model",
                    LogisticRegression(
                        C=c_value,
                        penalty="l1",
                        solver="liblinear",
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            ]
        )
        model.fit(train_val_x, train_val_y)
        test_score = scores(model, feature_sets["test"])
        test = metrics(labels_by_split["test"], test_score)
        coefficients = model.named_steps["model"].coef_.reshape(-1)
        nonzero = np.flatnonzero(np.abs(coefficients) > 1e-10)
        l1_rows.append(
            {
                "seed": seed,
                "role": role,
                "c": c_value,
                "nonzero_features": len(nonzero),
                "validation_auc": validation["auc"],
                "test_auc": test["auc"],
                "test_aucpr": test["aucpr"],
                "test_balanced_accuracy": test["balanced_accuracy"],
                "top_nonzero_feature": ""
                if not len(nonzero)
                else all_labels[int(nonzero[np.argmax(np.abs(coefficients[nonzero]))])],
            }
        )

    receipt = {
        "seed": seed,
        "role": role,
        "adaptive_checkpoint": str(checkpoint_path),
        "data_receipt": data_receipt,
        "pair_counts": counts,
        "physical_full_pauli_dimension": 4**8 - 1,
        "effective_nonzero_xz_dimension": 3**8 - 1,
        "identically_zero_y_dimension": (4**8 - 1) - (3**8 - 1),
        "feature_counts_by_exact_weight": expected_counts,
        "local_bloch_norm_max_abs_error": float(
            np.max(np.abs(np.sum(combined_local**2, axis=2) - 1.0))
        ),
        "full_pauli_purity_max_abs_error": float(
            np.max(np.abs(np.sum(full.astype(np.float64) ** 2, axis=1) - (2**8 - 1)))
        ),
    }
    return rows, l1_rows, receipt


def choose_development_representation(rows: list[dict]) -> dict:
    candidates = [
        row
        for row in rows
        if row["seed"] == 42
        and row["detector"] == "logistic"
        and row["path"] in {"cumulative_weight", "pca", "fisher"}
        and row["validation_auc"] != ""
    ]
    full = next(
        row
        for row in candidates
        if row["path"] == "cumulative_weight" and row["representation"] == "weight_le_8"
    )
    threshold = float(full["validation_auc"]) - 0.02
    eligible = [row for row in candidates if float(row["validation_auc"]) >= threshold]
    eligible.sort(
        key=lambda row: (
            int(row["dimension"]),
            -float(row["validation_auc"]),
            {"cumulative_weight": 0, "pca": 1, "fisher": 2}[row["path"]],
        )
    )
    selected = eligible[0]
    return {
        "full_validation_auc": float(full["validation_auc"]),
        "epsilon": 0.02,
        "threshold": threshold,
        "path": selected["path"],
        "representation": selected["representation"],
        "dimension": int(selected["dimension"]),
        "validation_auc": float(selected["validation_auc"]),
        "test_auc_not_used_for_selection": float(selected["test_auc"]),
        "compression_ratio_physical": int(selected["dimension"]) / (4**8 - 1),
        "compression_ratio_effective": int(selected["dimension"]) / (3**8 - 1),
    }


def summarize_confirmation(rows: list[dict], selected: dict) -> dict:
    confirmation = [
        row
        for row in rows
        if row["role"] == "confirmation"
        and row["path"] == selected["path"]
        and row["representation"] == selected["representation"]
        and row["detector"] == "logistic"
    ]
    full = [
        row
        for row in rows
        if row["role"] == "confirmation"
        and row["path"] == "cumulative_weight"
        and row["representation"] == "weight_le_8"
        and row["detector"] == "logistic"
    ]
    if not confirmation:
        return {
            "seeds": [],
            "status": "not_run",
            "note": "No confirmation seeds were requested.",
        }
    selected_auc = np.asarray([float(row["test_auc"]) for row in confirmation])
    full_auc = np.asarray([float(row["test_auc"]) for row in full])
    return {
        "seeds": [int(row["seed"]) for row in confirmation],
        "selected_auc_mean": float(np.mean(selected_auc)),
        "selected_auc_population_std": float(np.std(selected_auc, ddof=0)),
        "selected_auc_values": selected_auc.tolist(),
        "full_auc_mean": float(np.mean(full_auc)),
        "full_auc_population_std": float(np.std(full_auc, ddof=0)),
        "full_auc_values": full_auc.tolist(),
        "paired_auc_gap_selected_minus_full_mean": float(np.mean(selected_auc - full_auc)),
        "all_selected_within_epsilon_of_full": bool(
            np.all(selected_auc >= full_auc - float(selected["epsilon"]))
        ),
    }


def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_l1_rows = []
    receipts = []
    for seed in args.seeds:
        rows, l1_rows, receipt = run_seed(args, seed)
        all_rows.extend(rows)
        all_l1_rows.extend(l1_rows)
        receipts.append(receipt)
        print(
            f"seed={seed} completed representations={len(rows)} "
            f"full_dim={receipt['effective_nonzero_xz_dimension']}",
            flush=True,
        )
    selected = choose_development_representation(all_rows)
    confirmation = summarize_confirmation(all_rows, selected)
    write_csv(args.output / "probe_metrics.csv", all_rows)
    write_csv(args.output / "l1_sparse_metrics.csv", all_l1_rows)
    summary = {
        "status": "PASS",
        "protocol": {
            "development_seed": 42,
            "confirmation_seeds": [seed for seed in args.seeds if seed != 42],
            "paired_split": "historical train/validation/test; clean and triggered rows remain paired",
            "probe_c_selection": "validation only; refit on train+validation before test",
            "pca_fit_scope": "train only",
            "fisher_fit_scope": "train only",
            "primary_metric": "paired held-out ROC AUC",
        },
        "selected_minimum_representation": selected,
        "confirmation": confirmation,
        "receipts": receipts,
    }
    (args.output / "pauli_compression_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected": selected, "confirmation": confirmation}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--c-grid", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    parser.add_argument(
        "--compression-dims",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64],
    )
    parser.add_argument("--pca-max-components", type=int, default=64)
    parser.add_argument("--random-dims", nargs="+", type=int, default=[8, 32, 64])
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument(
        "--l1-c-grid",
        nargs="+",
        type=float,
        default=[0.001, 0.003, 0.01, 0.03, 0.1],
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    parsed = parser.parse_args()
    parsed.dataset_root = parsed.dataset_root.resolve()
    parsed.output = parsed.output.resolve()
    main(parsed)
