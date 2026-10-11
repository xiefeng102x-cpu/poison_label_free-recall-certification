#!/usr/bin/env python3
"""Stage-1 unified one-class poison screening for classical and QML views.

The detector never sees attack samples or attack labels during fitting.  It
fits a clean target-class PCA model, fuses whitened in-subspace distance and
orthogonal residual, and obtains its threshold from a disjoint clean
calibration split.  Exactly the same detector is used for raw input, a
backdoored classical MLP, and low-dimensional Pauli views.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn as nn
from scipy.stats import beta
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER14_ROOT = PROJECT_ROOT / "paper14"
DATA_ROOT = PROJECT_ROOT / "data/Mnist/detail_single_samle"
DEFAULT_OUTPUT = PAPER14_ROOT / "results/unified_oneclass_poison_screening_v1"
SEEDS = (42, 43, 44)
RANDOM_SEED = 20260928


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def clopper_pearson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    low = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, total - successes + 1))
    high = 1.0 if successes == total else float(beta.ppf(0.975, successes + 1, total - successes))
    return low, high


def source_dir(seed: int) -> Path:
    return DATA_ROOT / f"seed_{seed}/layer_12_grid/t7_vs_t1/pr_0.3"


def load_seed(seed: int) -> dict[str, np.ndarray | dict]:
    root = source_dir(seed)
    meta = torch.load(
        root / f"poisoned_samples_meta_seed_{seed}_pr_0.3.pt",
        map_location="cpu",
        weights_only=False,
    )
    eligibility = json.loads(
        (root / "enhanced_scb_comprehensive_results_fixed.json").read_text(encoding="utf-8")
    )["aggregated_results"]["0.3"]
    return {
        "target": meta["target_clean_data"].numpy().astype(np.float32),
        "source": meta["non_target_clean_data"].numpy().astype(np.float32),
        "poison_source": meta["poisoned_non_target_data"].numpy().astype(np.float32),
        "trigger_target": meta["poisoned_target_data"].numpy().astype(np.float32),
        "poison_source_indices": meta["poison_indices_in_non_target"].numpy().astype(int),
        "qml_eligibility": {
            "ca": float(eligibility["clean_accuracy"]["mean"]) / 100.0,
            "asr": float(eligibility["attack_success_rate"]["mean"]) / 100.0,
        },
    }


def make_splits(data: dict, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED + seed)
    target_order = rng.permutation(len(data["target"]))
    if len(target_order) < 420:
        raise ValueError("Stage-1 protocol requires at least 420 clean target samples")
    poison_order = rng.permutation(len(data["poison_source"]))
    source_poison_indices = np.asarray(data["poison_source_indices"], dtype=int)
    source_remaining = np.setdiff1d(
        np.arange(len(data["source"])), source_poison_indices, assume_unique=False
    )
    rng.shuffle(source_remaining)
    trigger_target_order = rng.permutation(len(data["trigger_target"]))
    return {
        "target_classifier_train": target_order[:210],
        "target_reference": target_order[210:315],
        "target_calibration": target_order[315:378],
        "target_test": target_order[378:420],
        "source_classifier_train": source_remaining[:-84],
        "source_clean_test": source_remaining[-84:],
        "poison_train": poison_order[:108],
        "poison_validation": poison_order[108:144],
        "poison_test": poison_order[144:180],
        "trigger_target_test": trigger_target_order[:36],
    }


class BackdoorMLP(nn.Module):
    def __init__(self, input_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.head = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


@dataclass
class ClassicalVictim:
    model: BackdoorMLP
    mean: np.ndarray
    scale: np.ndarray
    metrics: dict

    def representations(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = (np.asarray(x, dtype=np.float32) - self.mean) / self.scale
        tensor = torch.tensor(z, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            hidden = self.model.encoder(tensor).numpy().astype(np.float64)
            logits = self.model.head(torch.tensor(hidden, dtype=torch.float32)).numpy()
        margin = (logits[:, 1] - logits[:, 0]).reshape(-1, 1).astype(np.float64)
        return hidden, margin


def train_classical_victim(data: dict, split: dict, seed: int, epochs: int) -> ClassicalVictim:
    x_target = data["target"][split["target_classifier_train"]]
    x_source = data["source"][split["source_classifier_train"]]
    x_poison = data["poison_source"][split["poison_train"]]
    clean_train = np.concatenate([x_target, x_source])
    mean = clean_train.mean(axis=0, keepdims=True)
    scale = clean_train.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    x_train = np.concatenate([x_target, x_source, x_poison])
    y_train = np.concatenate(
        [
            np.ones(len(x_target), dtype=np.int64),
            np.zeros(len(x_source), dtype=np.int64),
            np.ones(len(x_poison), dtype=np.int64),
        ]
    )
    z_train = (x_train - mean) / scale
    generator = torch.Generator().manual_seed(seed + RANDOM_SEED)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(z_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=64,
        shuffle=True,
        generator=generator,
    )
    torch.manual_seed(seed)
    model = BackdoorMLP(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

    def predict(x: np.ndarray) -> np.ndarray:
        z = (x - mean) / scale
        model.eval()
        with torch.no_grad():
            return model(torch.tensor(z, dtype=torch.float32)).argmax(1).numpy()

    target_eval = data["target"][np.concatenate([
        split["target_reference"], split["target_calibration"], split["target_test"]
    ])]
    source_eval = data["source"][split["source_clean_test"]]
    poison_eval = data["poison_source"][split["poison_test"]]
    target_acc = float(np.mean(predict(target_eval) == 1))
    source_acc = float(np.mean(predict(source_eval) == 0))
    metrics = {
        "target_clean_accuracy": target_acc,
        "source_clean_accuracy": source_acc,
        "balanced_clean_accuracy": 0.5 * (target_acc + source_acc),
        "attack_success_rate": float(np.mean(predict(poison_eval) == 1)),
        "n_classifier_train_target": int(len(x_target)),
        "n_classifier_train_source": int(len(x_source)),
        "n_classifier_train_poison": int(len(x_poison)),
        "epochs": int(epochs),
    }
    return ClassicalVictim(model=model, mean=mean, scale=scale, metrics=metrics)


def local_pauli(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(samples, dtype=np.float64).reshape(len(samples), -1)
    if states.shape[1] < 32:
        states = np.pad(states, ((0, 0), (0, 32 - states.shape[1])))
    elif states.shape[1] > 32:
        states = states[:, :32]
    states = states / (np.linalg.norm(states, axis=1, keepdims=True) + 1e-12)
    basis = np.arange(32, dtype=np.int64)
    columns: dict[tuple[int, str], np.ndarray] = {}
    for qubit in range(5):
        bit_position = 4 - qubit
        bit = (basis >> bit_position) & 1
        permutation = basis ^ (1 << bit_position)
        columns[(qubit, "X")] = np.sum(states[:, permutation] * states, axis=1)
        columns[(qubit, "Z")] = np.sum(states * states * (1 - 2 * bit), axis=1)
    z = np.stack([columns[(q, "Z")] for q in range(5)], axis=1)
    xz = np.stack(
        [columns[(q, axis)] for q in range(5) for axis in ("X", "Z")], axis=1
    )
    return z.astype(np.float64), xz.astype(np.float64)


@dataclass
class DualSubspaceDetector:
    mean: np.ndarray
    scale: np.ndarray
    basis: np.ndarray
    eigenvalues: np.ndarray
    k: int
    component_center: np.ndarray
    component_scale: np.ndarray
    residual_scale: float
    threshold: float
    calibration_scores: np.ndarray
    alpha: float

    def score_components(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        projections = z @ self.basis.T
        within = np.sum(
            projections[:, : self.k] ** 2 / (self.eigenvalues[: self.k] + 1e-8),
            axis=1,
        )
        captured = np.sum(projections[:, : self.k] ** 2, axis=1)
        residual = np.maximum(np.sum(z * z, axis=1) - captured, 0.0) / self.residual_scale
        return within, residual

    def score(self, x: np.ndarray) -> np.ndarray:
        within, residual = self.score_components(x)
        components = np.stack([np.log1p(within), np.log1p(residual)], axis=1)
        standardized = (components - self.component_center) / self.component_scale
        return np.max(standardized, axis=1)

    def p_value(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                (1 + np.sum(self.calibration_scores >= value))
                / (len(self.calibration_scores) + 1)
                for value in scores
            ],
            dtype=np.float64,
        )


def robust_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center), axis=0) * 1.4826
    standard = np.std(values, axis=0, ddof=1)
    scale = np.where(mad > 1e-8, mad, np.where(standard > 1e-8, standard, 1.0))
    return center, scale


def fit_detector(reference: np.ndarray, calibration: np.ndarray, alpha: float) -> DualSubspaceDetector:
    reference = np.asarray(reference, dtype=np.float64)
    mean = reference.mean(axis=0, keepdims=True)
    scale = reference.std(axis=0, ddof=1, keepdims=True)
    scale[scale < 1e-8] = 1.0
    z_ref = (reference - mean) / scale
    _, singular, vt = np.linalg.svd(z_ref, full_matrices=False)
    eigenvalues = singular**2 / max(len(reference) - 1, 1)
    total = float(np.sum(eigenvalues))
    if total <= 1e-12:
        k = 1
    else:
        k = int(np.searchsorted(np.cumsum(eigenvalues) / total, 0.90) + 1)
    k = min(max(k, 1), vt.shape[0])
    projections = z_ref @ vt.T
    captured = np.sum(projections[:, :k] ** 2, axis=1)
    raw_residual = np.maximum(np.sum(z_ref * z_ref, axis=1) - captured, 0.0)
    residual_scale = max(float(np.mean(raw_residual)), 1e-8)
    within = np.sum(projections[:, :k] ** 2 / (eigenvalues[:k] + 1e-8), axis=1)
    residual = raw_residual / residual_scale
    component_center, component_scale = robust_center_scale(
        np.stack([np.log1p(within), np.log1p(residual)], axis=1)
    )
    detector = DualSubspaceDetector(
        mean=mean,
        scale=scale,
        basis=vt,
        eigenvalues=eigenvalues,
        k=k,
        component_center=component_center,
        component_scale=component_scale,
        residual_scale=residual_scale,
        threshold=0.0,
        calibration_scores=np.empty(0),
        alpha=alpha,
    )
    calibration_scores = detector.score(calibration)
    rank = int(math.ceil((len(calibration_scores) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(calibration_scores))
    detector.calibration_scores = calibration_scores
    detector.threshold = float(np.sort(calibration_scores)[rank - 1])
    return detector


def representation_sets(data: dict, split: dict, victim: ClassicalVictim) -> dict[str, dict[str, np.ndarray]]:
    arrays = {
        "reference": data["target"][split["target_reference"]],
        "calibration": data["target"][split["target_calibration"]],
        "clean_target": data["target"][split["target_test"]],
        "poison_source": data["poison_source"][split["poison_test"]],
        "clean_source": data["source"][data["poison_source_indices"][split["poison_test"]]],
        "trigger_target": data["trigger_target"][split["trigger_target_test"]],
    }
    output: dict[str, dict[str, np.ndarray]] = {
        "input32": {key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()}
    }
    hidden: dict[str, np.ndarray] = {}
    margin: dict[str, np.ndarray] = {}
    z_features: dict[str, np.ndarray] = {}
    xz_features: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        hidden[key], margin[key] = victim.representations(value)
        z_features[key], xz_features[key] = local_pauli(value)
    output["classical_hidden16"] = hidden
    output["classical_margin1"] = margin
    output["qml_z5"] = z_features
    output["qml_xz10"] = xz_features
    return output


def evaluate_representation(
    seed: int,
    name: str,
    sets: dict[str, np.ndarray],
    alpha: float,
) -> tuple[dict, list[dict]]:
    detector = fit_detector(sets["reference"], sets["calibration"], alpha)
    scored: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    sample_rows: list[dict] = []
    for group in ("calibration", "clean_target", "poison_source", "clean_source", "trigger_target"):
        scores = detector.score(sets[group])
        p_values = detector.p_value(scores)
        flags = p_values <= alpha
        scored[group] = scores, p_values, flags
        for index, (score, p_value, flag) in enumerate(zip(scores, p_values, flags)):
            sample_rows.append(
                {
                    "seed": seed,
                    "representation": name,
                    "group": group,
                    "sample_position": index,
                    "score": float(score),
                    "p_value": float(p_value),
                    "flagged": int(flag),
                    "alpha": alpha,
                    "threshold": detector.threshold,
                }
            )
    clean_score = scored["clean_target"][0]
    poison_score = scored["poison_source"][0]
    clean_flags = scored["clean_target"][2]
    poison_flags = scored["poison_source"][2]
    tp, fn = int(poison_flags.sum()), int((~poison_flags).sum())
    fp, tn = int(clean_flags.sum()), int((~clean_flags).sum())
    tpr_low, tpr_high = clopper_pearson(tp, tp + fn)
    fpr_low, fpr_high = clopper_pearson(fp, fp + tn)
    row = {
        "seed": seed,
        "representation": name,
        "input_dimension": int(sets["reference"].shape[1]),
        "selected_clean_pca_dimension": int(detector.k),
        "alpha": alpha,
        "n_reference": int(len(sets["reference"])),
        "n_calibration": int(len(sets["calibration"])),
        "n_clean_test": int(tn + fp),
        "n_poison_test": int(tp + fn),
        "calibration_fpr": float(scored["calibration"][2].mean()),
        "threshold": detector.threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "auc": float(
            roc_auc_score(
                np.concatenate([np.zeros(len(clean_score)), np.ones(len(poison_score))]),
                np.concatenate([clean_score, poison_score]),
            )
        ),
        "tpr": tp / (tp + fn),
        "tpr_ci_low": tpr_low,
        "tpr_ci_high": tpr_high,
        "fpr": fp / (fp + tn),
        "fpr_ci_low": fpr_low,
        "fpr_ci_high": fpr_high,
        "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp)),
        "clean_source_flag_rate": float(scored["clean_source"][2].mean()),
        "trigger_target_flag_rate": float(scored["trigger_target"][2].mean()),
        "poison_minus_clean_source_rate": float(
            scored["poison_source"][2].mean() - scored["clean_source"][2].mean()
        ),
        "trigger_target_minus_clean_fpr": float(
            scored["trigger_target"][2].mean() - scored["clean_target"][2].mean()
        ),
    }
    return row, sample_rows


def evaluate_multiview(
    seed: int,
    name: str,
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
    alpha: float,
) -> tuple[dict, list[dict]]:
    """Fuse two clean-only anomaly views and recalibrate their maximum score."""
    detectors = [
        fit_detector(first["reference"], first["calibration"], alpha),
        fit_detector(second["reference"], second["calibration"], alpha),
    ]

    def combined(group: str) -> np.ndarray:
        return np.maximum(detectors[0].score(first[group]), detectors[1].score(second[group]))

    calibration_scores = combined("calibration")
    rank = int(math.ceil((len(calibration_scores) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(calibration_scores))
    threshold = float(np.sort(calibration_scores)[rank - 1])
    scored: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    sample_rows: list[dict] = []
    for group in ("calibration", "clean_target", "poison_source", "clean_source", "trigger_target"):
        scores = combined(group)
        p_values = np.asarray(
            [
                (1 + np.sum(calibration_scores >= value)) / (len(calibration_scores) + 1)
                for value in scores
            ],
            dtype=np.float64,
        )
        flags = p_values <= alpha
        scored[group] = scores, p_values, flags
        for index, (score, p_value, flag) in enumerate(zip(scores, p_values, flags)):
            sample_rows.append(
                {
                    "seed": seed,
                    "representation": name,
                    "group": group,
                    "sample_position": index,
                    "score": float(score),
                    "p_value": float(p_value),
                    "flagged": int(flag),
                    "alpha": alpha,
                    "threshold": threshold,
                }
            )
    clean_score = scored["clean_target"][0]
    poison_score = scored["poison_source"][0]
    clean_flags = scored["clean_target"][2]
    poison_flags = scored["poison_source"][2]
    tp, fn = int(poison_flags.sum()), int((~poison_flags).sum())
    fp, tn = int(clean_flags.sum()), int((~clean_flags).sum())
    tpr_low, tpr_high = clopper_pearson(tp, tp + fn)
    fpr_low, fpr_high = clopper_pearson(fp, fp + tn)
    row = {
        "seed": seed,
        "representation": name,
        "input_dimension": f"{first['reference'].shape[1]}+{second['reference'].shape[1]}",
        "selected_clean_pca_dimension": f"{detectors[0].k}+{detectors[1].k}",
        "alpha": alpha,
        "n_reference": int(len(first["reference"])),
        "n_calibration": int(len(first["calibration"])),
        "n_clean_test": int(tn + fp),
        "n_poison_test": int(tp + fn),
        "calibration_fpr": float(scored["calibration"][2].mean()),
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "auc": float(
            roc_auc_score(
                np.concatenate([np.zeros(len(clean_score)), np.ones(len(poison_score))]),
                np.concatenate([clean_score, poison_score]),
            )
        ),
        "tpr": tp / (tp + fn),
        "tpr_ci_low": tpr_low,
        "tpr_ci_high": tpr_high,
        "fpr": fp / (fp + tn),
        "fpr_ci_low": fpr_low,
        "fpr_ci_high": fpr_high,
        "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp)),
        "clean_source_flag_rate": float(scored["clean_source"][2].mean()),
        "trigger_target_flag_rate": float(scored["trigger_target"][2].mean()),
        "poison_minus_clean_source_rate": float(
            scored["poison_source"][2].mean() - scored["clean_source"][2].mean()
        ),
        "trigger_target_minus_clean_fpr": float(
            scored["trigger_target"][2].mean() - scored["clean_target"][2].mean()
        ),
    }
    return row, sample_rows


def pooled_summary(rows: list[dict]) -> list[dict]:
    summaries = []
    for representation in sorted({row["representation"] for row in rows}):
        selected = [row for row in rows if row["representation"] == representation]
        tp = sum(int(row["tp"]) for row in selected)
        fn = sum(int(row["fn"]) for row in selected)
        fp = sum(int(row["fp"]) for row in selected)
        tn = sum(int(row["tn"]) for row in selected)
        tpr_low, tpr_high = clopper_pearson(tp, tp + fn)
        fpr_low, fpr_high = clopper_pearson(fp, fp + tn)
        summaries.append(
            {
                "representation": representation,
                "n_seeds": len(selected),
                "pooled_tpr": tp / (tp + fn),
                "pooled_tpr_ci_low": tpr_low,
                "pooled_tpr_ci_high": tpr_high,
                "pooled_fpr": fp / (fp + tn),
                "pooled_fpr_ci_low": fpr_low,
                "pooled_fpr_ci_high": fpr_high,
                "mean_auc": float(np.mean([row["auc"] for row in selected])),
                "mean_selected_clean_pca_dimension": (
                    float(np.mean([row["selected_clean_pca_dimension"] for row in selected]))
                    if all(isinstance(row["selected_clean_pca_dimension"], int) for row in selected)
                    else ";".join(str(row["selected_clean_pca_dimension"]) for row in selected)
                ),
                "mean_clean_source_flag_rate": float(
                    np.mean([row["clean_source_flag_rate"] for row in selected])
                ),
                "mean_trigger_target_flag_rate": float(
                    np.mean([row["trigger_target_flag_rate"] for row in selected])
                ),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
            }
        )
    return summaries


def render_report(
    summaries: list[dict],
    victim_rows: list[dict],
    stage1: dict,
    elapsed: float,
) -> str:
    by_name = {row["representation"]: row for row in summaries}
    lines = [
        "# Unified one-class poison screening: Stage 1",
        "",
        "## Outcome",
        "",
        stage1["conclusion"],
        "",
        "The detector was fitted and calibrated with clean target-class samples only; attack labels were used only for final evaluation.",
        "",
        "## Pooled held-out results",
        "",
        "| representation | clean PCA k | TPR | clean FPR | mean AUC | clean-source flagged | triggered-target flagged |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "input32",
        "classical_margin1",
        "classical_hidden16",
        "classical_input_hidden_mv",
        "qml_z5",
        "qml_xz10",
        "qml_input_z5_mv",
    ):
        row = by_name[name]
        dimension = row["mean_selected_clean_pca_dimension"]
        dimension_text = f"{dimension:.1f}" if isinstance(dimension, float) else str(dimension)
        lines.append(
            f"| {name} | {dimension_text} | "
            f"{row['pooled_tpr']:.3f} | {row['pooled_fpr']:.3f} | {row['mean_auc']:.3f} | "
            f"{row['mean_clean_source_flag_rate']:.3f} | {row['mean_trigger_target_flag_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "TPR/FPR are pooled over seeds 42--44. The two control columns separate label-manifold mismatch from a class-independent trigger signature.",
            "",
            "## Victim eligibility",
            "",
            "| seed | classical balanced CA | classical ASR | QML CA | QML ASR |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in victim_rows:
        lines.append(
            f"| {row['seed']} | {row['classical_balanced_clean_accuracy']:.3f} | "
            f"{row['classical_attack_success_rate']:.3f} | {row['qml_ca']:.3f} | {row['qml_asr']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- A high clean-source flag rate means the detector is exploiting label/semantic inconsistency, which is useful for poison screening but is not trigger-specific.",
            "- A triggered-target flag rate above clean-target FPR is evidence of a trigger-specific shift.",
            "- This pilot establishes algorithmic portability of the rule, not transfer of one fitted detector across classical and quantum representations.",
            "- Next expansion should run only if both primary tracks pass; otherwise diagnose the failed representation before scanning more triggers.",
            "",
            f"Runtime: {elapsed:.1f} seconds.",
            "",
        ]
    )
    return "\n".join(lines)


def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    metric_rows: list[dict] = []
    sample_rows: list[dict] = []
    victim_rows: list[dict] = []
    split_receipts: list[dict] = []
    for seed in args.seeds:
        data = load_seed(seed)
        split = make_splits(data, seed)
        victim = train_classical_victim(data, split, seed, args.epochs)
        representations = representation_sets(data, split, victim)
        qml = data["qml_eligibility"]
        victim_rows.append(
            {
                "seed": seed,
                "classical_balanced_clean_accuracy": victim.metrics["balanced_clean_accuracy"],
                "classical_target_clean_accuracy": victim.metrics["target_clean_accuracy"],
                "classical_source_clean_accuracy": victim.metrics["source_clean_accuracy"],
                "classical_attack_success_rate": victim.metrics["attack_success_rate"],
                "qml_ca": qml["ca"],
                "qml_asr": qml["asr"],
                "classical_eligible": int(
                    victim.metrics["balanced_clean_accuracy"] >= 0.90
                    and victim.metrics["attack_success_rate"] >= 0.80
                ),
                "qml_eligible": int(qml["ca"] >= 0.90 and qml["asr"] >= 0.80),
            }
        )
        split_receipts.append(
            {
                "seed": seed,
                **{key: [int(value) for value in values] for key, values in split.items()},
            }
        )
        for name, sets in representations.items():
            row, samples = evaluate_representation(seed, name, sets, args.alpha)
            metric_rows.append(row)
            sample_rows.extend(samples)
        for name, left, right in (
            (
                "classical_input_hidden_mv",
                representations["input32"],
                representations["classical_hidden16"],
            ),
            ("qml_input_z5_mv", representations["input32"], representations["qml_z5"]),
        ):
            row, samples = evaluate_multiview(seed, name, left, right, args.alpha)
            metric_rows.append(row)
            sample_rows.extend(samples)

    summaries = pooled_summary(metric_rows)
    summary_by_name = {row["representation"]: row for row in summaries}
    victim_gate = all(row["classical_eligible"] and row["qml_eligible"] for row in victim_rows)
    primary_gate = all(
        summary_by_name[name]["pooled_tpr"] >= 0.80
        and summary_by_name[name]["pooled_fpr"] <= 0.10
        for name in ("classical_hidden16", "qml_z5")
    )
    passed = bool(victim_gate and primary_gate)
    candidate_gate = all(
        summary_by_name[name]["pooled_tpr"] >= 0.80
        and summary_by_name[name]["pooled_fpr"] <= 0.10
        for name in ("classical_input_hidden_mv", "qml_input_z5_mv")
    )
    conclusion = (
        "Stage 1 PASS: the same clean-only detector rule meets the prespecified operating gate on both the classical hidden representation and quantum Z5 representation."
        if passed
        else "Stage 1 does not pass all prespecified gates; inspect victim eligibility and the failed primary representation before expanding the trigger family."
    )
    stage1 = {
        "passed": passed,
        "victim_gate_passed": bool(victim_gate),
        "primary_detector_gate_passed": bool(primary_gate),
        "post_stage1_multiview_candidate_passed": bool(victim_gate and candidate_gate),
        "conclusion": conclusion,
        "criteria": {
            "classical_and_qml_victim_ca_min": 0.90,
            "classical_and_qml_victim_asr_min": 0.80,
            "primary_pooled_tpr_min": 0.80,
            "primary_pooled_fpr_max": 0.10,
        },
    }
    elapsed = time.time() - started
    write_csv(args.output / "per_seed_metrics.csv", metric_rows)
    write_csv(args.output / "per_sample_scores.csv", sample_rows)
    write_csv(args.output / "classical_and_qml_victim_metrics.csv", victim_rows)
    write_csv(args.output / "pooled_summary.csv", summaries)
    json_dump(args.output / "split_receipts.json", split_receipts)
    json_dump(args.output / "stage1_decision.json", stage1)
    json_dump(
        args.output / "manifest.json",
        {
            "experiment": "unified_oneclass_poison_screening_v1",
            "seeds": args.seeds,
            "alpha": args.alpha,
            "classical_epochs": args.epochs,
            "pair": "t7_vs_t1",
            "poison_ratio": 0.3,
            "detector_training_labels": "clean target only; no attack samples",
            "threshold_source": "disjoint clean target calibration split",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "elapsed_seconds": elapsed,
        },
    )
    (args.output / "EXPERIMENT_REPORT.md").write_text(
        render_report(summaries, victim_rows, stage1, elapsed), encoding="utf-8"
    )
    print(conclusion)
    for row in summaries:
        dimension = row["mean_selected_clean_pca_dimension"]
        dimension_text = f"{dimension:.1f}" if isinstance(dimension, float) else str(dimension)
        print(
            f"{row['representation']:>20s}: TPR={row['pooled_tpr']:.3f} "
            f"FPR={row['pooled_fpr']:.3f} AUC={row['mean_auc']:.3f} "
            f"k={dimension_text}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=lambda value: [int(x) for x in value.split(",")], default=list(SEEDS))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=100)
    main(parser.parse_args())
