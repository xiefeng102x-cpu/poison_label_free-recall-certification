#!/usr/bin/env python3
"""Evaluate single-sample attack screening on the frozen 5-qubit MNIST data.

The primary detector is deliberately small and transferable: a logistic probe
on the five single-qubit Z expectations at the shared encoding checkpoint.  It
is trained and thresholded with seed 42 only, then evaluated without any
recalibration on held-out samples from seeds 42, 43, and 44.  Every test sample
is scored independently; matched pairs are used only to construct leakage-free
train/validation/test partitions and to balance the evaluation.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PAPER14_ROOT / "results/phase_a_mnist5q_real_v1"
DEFAULT_OUTPUT = PAPER14_ROOT / "results/mnist_single_sample_screening_v1"
SEEDS = (42, 43, 44)
C_GRID = (0.01, 0.1, 1.0, 10.0)


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


def make_model(c_value: float, seed: int) -> Pipeline:
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


def fit_and_select(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    seed: int,
) -> tuple[Pipeline, float, float, np.ndarray]:
    candidates = []
    for c_value in C_GRID:
        model = make_model(c_value, seed)
        model.fit(train_x, train_y)
        validation_score = model.decision_function(validation_x)
        candidates.append(
            (
                float(roc_auc_score(validation_y, validation_score)),
                c_value,
                model,
                np.asarray(validation_score, dtype=np.float64),
            )
        )
    validation_auc, selected_c, model, validation_score = sorted(
        candidates, key=lambda row: (-row[0], row[1])
    )[0]
    return model, selected_c, validation_auc, validation_score


def select_threshold(validation_y: np.ndarray, validation_score: np.ndarray) -> dict:
    fpr, tpr, thresholds = roc_curve(validation_y, validation_score)
    youden = tpr - fpr
    best = np.flatnonzero(np.isclose(youden, np.max(youden)))
    # Among equally good validation operating points, prefer lower FPR and then
    # the higher (more conservative) threshold.
    index = sorted(best, key=lambda i: (fpr[i], -thresholds[i]))[0]
    return {
        "threshold": float(thresholds[index]),
        "validation_fpr": float(fpr[index]),
        "validation_tpr": float(tpr[index]),
        "validation_youden_j": float(youden[index]),
    }


def clopper_pearson(successes: int, total: int) -> tuple[float, float]:
    low = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, total - successes + 1))
    high = 1.0 if successes == total else float(beta.ppf(0.975, successes + 1, total - successes))
    return low, high


def classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    predicted = score >= threshold
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(y, predicted).ravel())
    positives = tp + fn
    negatives = tn + fp
    tpr = tp / positives
    fpr = fp / negatives
    tpr_low, tpr_high = clopper_pearson(tp, positives)
    fpr_low, fpr_high = clopper_pearson(fp, negatives)
    return {
        "n_clean": negatives,
        "n_attack": positives,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "auc": float(roc_auc_score(y, score)),
        "tpr": tpr,
        "tpr_ci_low": tpr_low,
        "tpr_ci_high": tpr_high,
        "fpr": fpr,
        "fpr_ci_low": fpr_low,
        "fpr_ci_high": fpr_high,
        "specificity": tn / negatives,
        "balanced_accuracy": 0.5 * (tpr + tn / negatives),
    }


def pooled_count_metrics(rows: list[dict]) -> dict:
    tn = sum(int(row["tn"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    tp = sum(int(row["tp"]) for row in rows)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    tpr_low, tpr_high = clopper_pearson(tp, tp + fn)
    fpr_low, fpr_high = clopper_pearson(fp, fp + tn)
    return {
        "n_clean": tn + fp,
        "n_attack": tp + fn,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tpr": tpr,
        "tpr_ci_low": tpr_low,
        "tpr_ci_high": tpr_high,
        "fpr": fpr,
        "fpr_ci_low": fpr_low,
        "fpr_ci_high": fpr_high,
        "specificity": tn / (tn + fp),
        "balanced_accuracy": 0.5 * (tpr + tn / (tn + fp)),
    }


def load_seed(
    input_root: Path,
    split_map: dict[tuple[int, str], str],
    seed: int,
    cache_kind: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    payload = np.load(input_root / "cache" / f"seed{seed}_{cache_kind}_pauli.npz")
    pair_ids = payload["pair_ids"].astype(str)
    clean = payload["clean_full"]
    attack = payload["attack_full"]
    features: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    ids: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "test"):
        mask = np.asarray([split_map[(seed, pair_id)] == split for pair_id in pair_ids])
        features[split] = np.concatenate([clean[mask], attack[mask]])
        labels[split] = np.concatenate(
            [np.zeros(mask.sum(), dtype=int), np.ones(mask.sum(), dtype=int)]
        )
        ids[split] = np.concatenate([pair_ids[mask], pair_ids[mask]])
    return features, labels, ids


def evaluate_configuration(
    name: str,
    source_checkpoint: str,
    representation: str,
    calibration_scope: str,
    training_seed: int,
    model: Pipeline,
    selected_c: float,
    validation_auc: float,
    threshold_info: dict,
    evaluation: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    sample_rows: list[dict],
    metric_rows: list[dict],
) -> None:
    threshold = float(threshold_info["threshold"])
    for seed, (test_x, test_y, test_ids) in evaluation.items():
        score = np.asarray(model.decision_function(test_x), dtype=np.float64)
        result = classification_metrics(test_y, score, threshold)
        metric_rows.append(
            {
                "configuration": name,
                "source_checkpoint": source_checkpoint,
                "representation": representation,
                "calibration_scope": calibration_scope,
                "training_seed": training_seed,
                "evaluation_seed": seed,
                "selected_c": selected_c,
                "validation_auc": validation_auc,
                **threshold_info,
                **result,
            }
        )
        for pair_id, label, value in zip(test_ids, test_y, score):
            sample_rows.append(
                {
                    "configuration": name,
                    "evaluation_seed": seed,
                    "base_sample_id": str(pair_id),
                    "label": int(label),
                    "score": float(value),
                    "threshold": threshold,
                    "predicted_attack": int(value >= threshold),
                    "margin": float(value - threshold),
                }
            )


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
    z_labels = [labels_full[index] for index in z_indices]

    encoding = {
        seed: load_seed(args.input, split_map, seed, "shared_encoding") for seed in SEEDS
    }
    output = {
        seed: load_seed(args.input, split_map, seed, "backdoor_output") for seed in SEEDS
    }
    sample_rows: list[dict] = []
    metric_rows: list[dict] = []

    # Primary universal detector: one five-Z model trained/calibrated on seed 42.
    encoding_z = {
        seed: {
            split: encoding[seed][0][split][:, z_indices]
            for split in ("train", "validation", "test")
        }
        for seed in SEEDS
    }
    primary_model, primary_c, primary_val_auc, primary_val_score = fit_and_select(
        encoding_z[42]["train"],
        encoding[42][1]["train"],
        encoding_z[42]["validation"],
        encoding[42][1]["validation"],
        42,
    )
    primary_threshold = select_threshold(encoding[42][1]["validation"], primary_val_score)
    evaluate_configuration(
        "universal_encoding_z",
        "shared_encoding",
        "five_single_qubit_Z",
        "seed42_only",
        42,
        primary_model,
        primary_c,
        primary_val_auc,
        primary_threshold,
        {
            seed: (
                encoding_z[seed]["test"],
                encoding[seed][1]["test"],
                encoding[seed][2]["test"],
            )
            for seed in SEEDS
        },
        sample_rows,
        metric_rows,
    )

    # Universal PCA-2 control: same seed-42-only transfer protocol.
    pca = PCA(n_components=2, svd_solver="randomized", random_state=4343)
    pca.fit(encoding[42][0]["train"])
    encoding_pca = {
        seed: {
            split: pca.transform(encoding[seed][0][split])
            for split in ("train", "validation", "test")
        }
        for seed in SEEDS
    }
    pca_model, pca_c, pca_val_auc, pca_val_score = fit_and_select(
        encoding_pca[42]["train"],
        encoding[42][1]["train"],
        encoding_pca[42]["validation"],
        encoding[42][1]["validation"],
        42,
    )
    pca_threshold = select_threshold(encoding[42][1]["validation"], pca_val_score)
    evaluate_configuration(
        "universal_encoding_pca2",
        "shared_encoding",
        "PCA_2",
        "seed42_only",
        42,
        pca_model,
        pca_c,
        pca_val_auc,
        pca_threshold,
        {
            seed: (
                encoding_pca[seed]["test"],
                encoding[seed][1]["test"],
                encoding[seed][2]["test"],
            )
            for seed in SEEDS
        },
        sample_rows,
        metric_rows,
    )

    # Victim-specific controls: each output-state detector gets its own train
    # and validation calibration; only its untouched test split is evaluated.
    for representation in ("z", "pca2"):
        for seed in SEEDS:
            if representation == "z":
                transformed = {
                    split: output[seed][0][split][:, z_indices]
                    for split in ("train", "validation", "test")
                }
                display = "five_single_qubit_Z"
            else:
                local_pca = PCA(n_components=2, svd_solver="randomized", random_state=seed + 4301)
                local_pca.fit(output[seed][0]["train"])
                transformed = {
                    split: local_pca.transform(output[seed][0][split])
                    for split in ("train", "validation", "test")
                }
                display = "PCA_2"
            model, selected_c, validation_auc, validation_score = fit_and_select(
                transformed["train"],
                output[seed][1]["train"],
                transformed["validation"],
                output[seed][1]["validation"],
                seed,
            )
            threshold_info = select_threshold(output[seed][1]["validation"], validation_score)
            evaluate_configuration(
                f"victim_specific_output_{representation}",
                "backdoor_output",
                display,
                "per_victim",
                seed,
                model,
                selected_c,
                validation_auc,
                threshold_info,
                {
                    seed: (
                        transformed["test"],
                        output[seed][1]["test"],
                        output[seed][2]["test"],
                    )
                },
                sample_rows,
                metric_rows,
            )

    pooled = {}
    for configuration in sorted({row["configuration"] for row in metric_rows}):
        selected = [row for row in metric_rows if row["configuration"] == configuration]
        pooled[configuration] = pooled_count_metrics(selected)

    center = primary_model.named_steps["center"]
    classifier = primary_model.named_steps["model"]
    parameters = {
        "feature_order": z_labels,
        "center": center.mean_.tolist(),
        "coefficient": classifier.coef_.reshape(-1).tolist(),
        "intercept": float(classifier.intercept_[0]),
        "threshold": float(primary_threshold["threshold"]),
        "decision_rule": "flag attack when dot(coefficient, Z - center) + intercept >= threshold",
        "measurement_checkpoint": "shared encoding, before learned QNN layers",
        "training_seed": 42,
        "selected_c": primary_c,
    }
    summary = {
        "status": "PASS",
        "primary_configuration": "universal_encoding_z",
        "primary_result": pooled["universal_encoding_z"],
        "per_seed_primary_auc": {
            str(row["evaluation_seed"]): float(row["auc"])
            for row in metric_rows
            if row["configuration"] == "universal_encoding_z"
        },
        "comparison_results": pooled,
        "scope": {
            "trigger_family": "t7_vs_t1, poison_ratio_0.3",
            "inference_unit": "one sample; no paired clean sample is required",
            "training_and_thresholding": "seed 42 only for the primary detector",
            "confirmation": "untouched test samples from seeds 42, 43, and 44",
            "measurement": "exact simulated Pauli expectations; finite-shot hardware noise not evaluated",
        },
    }
    write_csv(args.output / "sample_scores.csv", sample_rows)
    write_csv(args.output / "screening_metrics.csv", metric_rows)
    (args.output / "screener_parameters.json").write_text(
        json.dumps(parameters, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "screening_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["primary_result"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parsed = parser.parse_args()
    parsed.input = parsed.input.resolve()
    parsed.output = parsed.output.resolve()
    main(parsed)
