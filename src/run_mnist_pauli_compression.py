#!/usr/bin/env python3
"""Compression audit of the cached 5-qubit MNIST Phase-A Pauli features.

This script deliberately reuses the frozen matched pairs, pair-grouped splits,
and exact 1023-dimensional Pauli features from ``phase_a_mnist5q_real_v1``.
PCA is fitted on training rows only.  Probe regularisation is selected on the
validation split, the probe is refitted on train+validation, and test is read
only afterward.  Seed 42 is the development seed; seeds 43 and 44 are frozen
confirmation seeds.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PAPER14_ROOT / "results/phase_a_mnist5q_real_v1"
DEFAULT_OUTPUT = PAPER14_ROOT / "results/mnist_pauli_compression_v1"
SEEDS = (42, 43, 44)
C_GRID = (0.01, 0.1, 1.0, 10.0)
PCA_DIMS = (1, 2, 4, 8, 16, 32, 64)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pauli_labels(n_qubits: int) -> list[str]:
    return [
        "".join(chars)
        for chars in itertools.product("IXYZ", repeat=n_qubits)
        if any(char != "I" for char in chars)
    ]


def estimator(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("center", StandardScaler(with_mean=True, with_std=False)),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def scores(model: Pipeline, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.decision_function(x), dtype=np.float64)


def tpr_at_fpr(y: np.ndarray, score: np.ndarray, maximum: float) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    eligible = tpr[fpr <= maximum + 1e-15]
    return float(np.max(eligible)) if len(eligible) else 0.0


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y, score)),
        "aucpr": float(average_precision_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, score >= 0.0)),
        "tpr_at_fpr_01": tpr_at_fpr(y, score, 0.01),
        "tpr_at_fpr_05": tpr_at_fpr(y, score, 0.05),
    }


def paired_bootstrap_auc(
    y: np.ndarray, score: np.ndarray, pairs: int, repetitions: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        chosen = rng.integers(0, pairs, size=pairs)
        rows = np.concatenate([chosen, chosen + pairs])
        values[repetition] = roc_auc_score(y[rows], score[rows])
    return tuple(float(x) for x in np.quantile(values, (0.025, 0.975)))


def evaluate(
    features: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    seed: int,
    bootstrap_repetitions: int,
) -> dict:
    candidates = []
    for c_value in C_GRID:
        model = estimator(c_value, seed)
        model.fit(features["train"], labels["train"])
        auc = roc_auc_score(labels["validation"], scores(model, features["validation"]))
        candidates.append((float(auc), c_value))
    validation_auc, selected_c = sorted(candidates, key=lambda x: (-x[0], x[1]))[0]
    model = estimator(selected_c, seed)
    model.fit(
        np.concatenate([features["train"], features["validation"]]),
        np.concatenate([labels["train"], labels["validation"]]),
    )
    test_score = scores(model, features["test"])
    test = metrics(labels["test"], test_score)
    pairs = len(labels["test"]) // 2
    low, high = paired_bootstrap_auc(
        labels["test"], test_score, pairs, bootstrap_repetitions, 91001 + seed
    )
    return {
        "selected_c": selected_c,
        "validation_auc": validation_auc,
        **{f"test_{key}": value for key, value in test.items()},
        "test_auc_ci_low": low,
        "test_auc_ci_high": high,
    }


def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    split_rows = list(csv.DictReader((args.input / "splits.csv").open()))
    split_map = {
        (int(row["seed"]), row["base_sample_id"]): row["split"] for row in split_rows
    }
    labels_full = pauli_labels(5)
    weights = np.asarray([sum(char != "I" for char in label) for label in labels_full])
    z_indices = np.asarray(
        [i for i, label in enumerate(labels_full) if weights[i] == 1 and "Z" in label]
    )
    xyz_indices = np.flatnonzero(weights == 1)
    rows: list[dict] = []
    receipts: list[dict] = []

    for seed in SEEDS:
        payload = np.load(args.input / "cache" / f"seed{seed}_backdoor_output_pauli.npz")
        clean = payload["clean_full"]
        attack = payload["attack_full"]
        pair_ids = payload["pair_ids"].astype(str)
        if clean.shape != (180, 1023) or attack.shape != clean.shape:
            raise AssertionError(f"unexpected cache shape for seed {seed}: {clean.shape}")

        full: dict[str, np.ndarray] = {}
        target: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        for split in ("train", "validation", "test"):
            mask = np.asarray([split_map[(seed, pair_id)] == split for pair_id in pair_ids])
            counts[split] = int(mask.sum())
            full[split] = np.concatenate([clean[mask], attack[mask]])
            target[split] = np.concatenate(
                [np.zeros(mask.sum(), dtype=int), np.ones(mask.sum(), dtype=int)]
            )
        if counts != {"train": 108, "validation": 36, "test": 36}:
            raise AssertionError(f"unexpected pair counts for seed {seed}: {counts}")

        def add(path: str, name: str, matrices: dict[str, np.ndarray], **extra) -> None:
            result = evaluate(matrices, target, seed, args.bootstrap_repetitions)
            rows.append(
                {
                    "seed": seed,
                    "role": "development" if seed == 42 else "confirmation",
                    "path": path,
                    "representation": name,
                    "dimension": matrices["train"].shape[1],
                    **result,
                    **extra,
                }
            )

        add("local", "Z", {split: values[:, z_indices] for split, values in full.items()})
        add("local", "XYZ", {split: values[:, xyz_indices] for split, values in full.items()})
        add("full", "Full", full)
        for weight in range(1, 6):
            exact = np.flatnonzero(weights == weight)
            cumulative = np.flatnonzero(weights <= weight)
            add(
                "exact_weight",
                f"weight_{weight}",
                {split: values[:, exact] for split, values in full.items()},
                pauli_weight=weight,
            )
            add(
                "cumulative_weight",
                f"weight_le_{weight}",
                {split: values[:, cumulative] for split, values in full.items()},
                pauli_weight=weight,
            )

        pca = PCA(
            n_components=max(PCA_DIMS),
            svd_solver="randomized",
            random_state=seed + 4301,
        )
        pca.fit(full["train"])
        transformed = {split: pca.transform(values) for split, values in full.items()}
        explained = np.cumsum(pca.explained_variance_ratio_)
        for dimension in PCA_DIMS:
            add(
                "pca",
                f"pca_{dimension}",
                {split: values[:, :dimension] for split, values in transformed.items()},
                explained_variance=float(explained[dimension - 1]),
            )

        receipts.append(
            {
                "seed": seed,
                "source_cache": str(
                    args.input / "cache" / f"seed{seed}_backdoor_output_pauli.npz"
                ),
                "pair_counts": counts,
                "pauli_dimension": clean.shape[1],
                "state_norm_max_abs_error": float(payload["state_norm_max_abs_error"]),
                "pauli_purity_max_abs_error": float(payload["pauli_purity_max_abs_error"]),
            }
        )
        print(f"seed={seed} complete", flush=True)

    development_pca = [
        row for row in rows if row["seed"] == 42 and row["path"] == "pca"
    ]
    development_full = next(
        row for row in rows if row["seed"] == 42 and row["representation"] == "Full"
    )
    epsilon = 0.02
    threshold = float(development_full["validation_auc"]) - epsilon
    eligible = [row for row in development_pca if float(row["validation_auc"]) >= threshold]
    selected = min(eligible, key=lambda row: int(row["dimension"]))
    confirmation = [
        row
        for row in rows
        if row["role"] == "confirmation"
        and row["representation"] == selected["representation"]
    ]
    pca1 = [row for row in rows if row["representation"] == "pca_1"]
    summary = {
        "status": "PASS",
        "question": "Does MNIST compress to one unsupervised PCA direction like BloodMNIST?",
        "answer": "low_dimensional_but_not_robustly_one_dimensional",
        "protocol": {
            "source": "cached exact output-checkpoint Pauli features from phase_a_mnist5q_real_v1",
            "development_seed": 42,
            "confirmation_seeds": [43, 44],
            "pca_fit_scope": "train only",
            "probe_c_selection": "validation only; refit on train+validation before test",
            "pair_grouping": "clean and triggered member of each base sample remain in one split",
            "selection_epsilon_auc": epsilon,
        },
        "selected_pca": {
            "representation": selected["representation"],
            "dimension": int(selected["dimension"]),
            "development_validation_auc": float(selected["validation_auc"]),
            "development_test_auc_not_used_for_selection": float(selected["test_auc"]),
            "development_explained_variance": float(selected["explained_variance"]),
            "validation_threshold": threshold,
            "confirmation_auc_values": [float(row["test_auc"]) for row in confirmation],
            "confirmation_auc_mean": float(np.mean([row["test_auc"] for row in confirmation])),
        },
        "pca_1": {
            "test_auc_by_seed": {str(row["seed"]): float(row["test_auc"]) for row in pca1},
            "test_auc_mean": float(np.mean([row["test_auc"] for row in pca1])),
            "explained_variance_by_seed": {
                str(row["seed"]): float(row["explained_variance"]) for row in pca1
            },
        },
        "receipts": receipts,
    }
    write_csv(args.output / "probe_metrics.csv", rows)
    (args.output / "pauli_compression_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["selected_pca"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    parsed = parser.parse_args()
    parsed.input = parsed.input.resolve()
    parsed.output = parsed.output.resolve()
    main(parsed)
