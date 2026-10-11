#!/usr/bin/env python3
"""Audit the BloodMNIST clean-accuracy gap with existing frozen artifacts.

This script is deliberately read-only with respect to the source experiments. It
reproduces three pieces of evidence:

1. the clean-QNN accuracy distribution over the existing 300-config pr=0.1 grid;
2. the CA/ASR trajectory of the training-time adaptive QNN and the classical
   Bloch control; and
3. a numerical audit of the manual parameter-shift gradient used by the
   adaptive QNN script.

The gradient audit compares the historical implementation, which applies the
parameter-shift identity directly to a nonlinear scalar loss, against both a
chain-rule parameter-shift gradient and a central finite-difference reference.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from sklearn.covariance import LedoitWolf


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_ROOT / "Blackdoor_1"
SOURCE_ROOT = (
    PROJECT_ROOT
    / "data/MedMnist/single_detect_20260411/epsilon_0.8"
)
ADAPTIVE_ROOT = (
    PROJECT_ROOT
    / "qml_adaptive_nc_project/paper_nc_adaptive/results/training_time_bloodmnist"
)
CLASSICAL_CSV = (
    PROJECT_ROOT
    / "qml_adaptive_nc_project/qml_adaptive_reviewer_package"
    / "reviewer_data/classical_controls/training_time_summary_classical.csv"
)
ADAPTIVE_CODE = (
    PROJECT_ROOT
    / "qml_adaptive_nc_project/paper_nc_adaptive/code/train_adaptive_qnn_fast.py"
)


def load_adaptive_module():
    # The archived training script assumes a server-side directory layout when
    # it inserts its own fast_circuit path.  In this repository the matching
    # module is stored next to the script, so expose that directory explicitly
    # before importing the historical implementation.
    sys.path.insert(0, str(ADAPTIVE_CODE.parent))
    spec = importlib.util.spec_from_file_location("blood_adaptive_impl", ADAPTIVE_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {ADAPTIVE_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values))


def population_sd(values: Iterable[float]) -> float:
    values = list(values)
    avg = mean(values)
    return float(math.sqrt(sum((value - avg) ** 2 for value in values) / len(values)))


def summarize(values: Iterable[float]) -> dict:
    values = sorted(float(value) for value in values)
    return {
        "n": len(values),
        "mean": mean(values),
        "population_sd": population_sd(values),
        "min": values[0],
        "median": values[len(values) // 2],
        "max": values[-1],
    }


def source_grid_audit(output: Path) -> dict:
    rows: list[dict] = []
    pattern = "seed_*/layer_*_grid/t6_vs_t*/pr_0.1/enhanced_scb_comprehensive_results_fixed.json"
    for path in sorted(SOURCE_ROOT.glob(pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["config"]
        result = payload["aggregated_results"]["0.1"]
        seed = int(re.search(r"seed_(\d+)", str(path)).group(1))
        rows.append(
            {
                "seed": seed,
                "layer": int(config["n_layers"]),
                "pair": f"t{config['target_class']}_vs_t{config['non_target_class']}",
                "clean_accuracy_pct": float(result["clean_accuracy"]["mean"]),
                "attack_success_rate_pct": float(result["attack_success_rate"]["mean"]),
                "source": str(path.relative_to(REPO_ROOT)),
            }
        )
    if len(rows) != 300:
        raise RuntimeError(f"Expected 300 pr=0.1 source configurations, found {len(rows)}")
    write_csv(output / "source_qnn_ca_rows.csv", rows)

    grouped: list[dict] = []
    for layer in sorted({row["layer"] for row in rows}):
        layer_rows = [row for row in rows if row["layer"] == layer]
        stats = summarize(row["clean_accuracy_pct"] for row in layer_rows)
        grouped.append({"group": "layer", "value": layer, **stats})
    for pair in sorted({row["pair"] for row in rows}):
        pair_rows = [row for row in rows if row["pair"] == pair]
        stats = summarize(row["clean_accuracy_pct"] for row in pair_rows)
        grouped.append({"group": "pair", "value": pair, **stats})
    write_csv(output / "source_qnn_ca_summary.csv", grouped)

    log_path = (
        SOURCE_ROOT
        / "seed_42/layer_8_grid/t6_vs_t0/pr_0.1/run.log"
    )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    train_match = re.search(r"Epoch 200/200 \| Loss: [^|]+\| Acc: ([0-9.]+)%", log_text)
    test_match = re.search(r"Clean model accuracy: ([0-9.]+)%", log_text)
    pca_match = re.search(
        r"Fitted PCA on training data: (\d+) -> (\d+).*?PCA explained variance: ([0-9.]+)",
        log_text,
        flags=re.DOTALL,
    )
    if not (train_match and test_match and pca_match):
        raise RuntimeError(f"Could not parse reference log {log_path}")

    return {
        "grid_clean_accuracy_pct": summarize(row["clean_accuracy_pct"] for row in rows),
        "reference_seed42_layer8_t6_vs_t0": {
            "train_accuracy_pct_epoch200": float(train_match.group(1)),
            "heldout_test_accuracy_pct": float(test_match.group(1)),
            "generalization_gap_percentage_points": float(train_match.group(1))
            - float(test_match.group(1)),
            "pca_input_dim": int(pca_match.group(1)),
            "pca_output_dim": int(pca_match.group(2)),
            "pca_explained_variance": float(pca_match.group(3)),
            "source": str(log_path.relative_to(REPO_ROOT)),
        },
    }


def adaptive_trajectory_audit(output: Path) -> dict:
    rows: list[dict] = []
    run_endpoints: list[dict] = []
    for path in sorted(ADAPTIVE_ROOT.glob("result_seed=*_L8_lr*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload["seed"])
        lambda_r = float(payload["lambda_r"])
        for point in payload["history"]:
            rows.append(
                {
                    "system": "qml",
                    "seed": seed,
                    "lambda_r": lambda_r,
                    "epoch": int(point["epoch"]),
                    "ca": float(point["ca"]),
                    "asr": float(point["asr"]),
                    "auc_xyz": float(point["auc_xyz"]),
                }
            )
        for label, point in (("initial", payload["history"][0]), ("final", payload["final"])):
            run_endpoints.append(
                {
                    "system": "qml",
                    "seed": seed,
                    "lambda_r": lambda_r,
                    "endpoint": label,
                    "epoch": int(point["epoch"]),
                    "ca": float(point["ca"]),
                    "asr": float(point["asr"]),
                    "auc_xyz": float(point["auc_xyz"]),
                }
            )
    if not rows:
        raise RuntimeError(f"No adaptive QML result JSONs found under {ADAPTIVE_ROOT}")
    write_csv(output / "adaptive_qml_trajectory.csv", rows)

    endpoint_rows: list[dict] = []
    for lambda_r in sorted({row["lambda_r"] for row in rows}):
        for label in ("initial", "final"):
            selected = [
                row
                for row in run_endpoints
                if row["lambda_r"] == lambda_r and row["endpoint"] == label
            ]
            endpoint_rows.append(
                {
                    "system": "qml",
                    "lambda_r": lambda_r,
                    "endpoint": label,
                    "epoch": max(row["epoch"] for row in selected),
                    "n": len(selected),
                    "ca_mean": mean(row["ca"] for row in selected),
                    "ca_population_sd": population_sd(row["ca"] for row in selected),
                    "asr_mean": mean(row["asr"] for row in selected),
                    "auc_xyz_mean": mean(row["auc_xyz"] for row in selected),
                }
            )

    classical_rows: list[dict] = []
    with CLASSICAL_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["source"] != "bloch_blood":
                continue
            classical_rows.append(
                {
                    "system": "classical_bloch",
                    "seed": int(row["seed"]),
                    "lambda_r": float(row["lambda_r"]),
                    "ca": float(row["ca"]),
                    "asr": float(row["asr"]),
                    "auc_xyz": float(row["auc_xyz"]),
                }
            )
    for lambda_r in sorted({row["lambda_r"] for row in classical_rows}):
        selected = [row for row in classical_rows if row["lambda_r"] == lambda_r]
        endpoint_rows.append(
            {
                "system": "classical_bloch",
                "lambda_r": lambda_r,
                "endpoint": "final",
                "epoch": 200,
                "n": len(selected),
                "ca_mean": mean(row["ca"] for row in selected),
                "ca_population_sd": population_sd(row["ca"] for row in selected),
                "asr_mean": mean(row["asr"] for row in selected),
                "auc_xyz_mean": mean(row["auc_xyz"] for row in selected),
            }
        )
    write_csv(output / "adaptive_endpoint_summary.csv", endpoint_rows)

    qml_lam1_initial = next(
        row
        for row in endpoint_rows
        if row["system"] == "qml" and row["lambda_r"] == 1.0 and row["endpoint"] == "initial"
    )
    qml_lam1_final = next(
        row
        for row in endpoint_rows
        if row["system"] == "qml" and row["lambda_r"] == 1.0 and row["endpoint"] == "final"
    )
    classical_lam1 = next(
        row
        for row in endpoint_rows
        if row["system"] == "classical_bloch"
        and row["lambda_r"] == 1.0
        and row["endpoint"] == "final"
    )
    return {
        "qml_lambda1_initial": qml_lam1_initial,
        "qml_lambda1_final": qml_lam1_final,
        "qml_lambda1_ca_drop_percentage_points": 100
        * (qml_lam1_initial["ca_mean"] - qml_lam1_final["ca_mean"]),
        "classical_bloch_lambda1_final": classical_lam1,
        "classical_minus_qml_final_ca_percentage_points": 100
        * (classical_lam1["ca_mean"] - qml_lam1_final["ca_mean"]),
        "protocol_warning": (
            "Classical Bloch jointly pretrains its encoder and classifier on x_ct/x_cnt/x_p, "
            "then reports CA on x_ct/x_cnt. QML loads a clean checkpoint, freezes the classifier, "
            "and adapts only circuit weights on the same pool. The final CA values are therefore "
            "not a matched held-out model-family benchmark."
        ),
    }


def load_gradient_fixture(adaptive):
    base = SOURCE_ROOT / "seed_42/layer_8_grid/t6_vs_t0/pr_0.1"
    state = torch.load(
        base / "models/clean_model_seed_200.pt",
        map_location="cpu",
        weights_only=False,
    )
    meta = torch.load(
        base / "poisoned_samples_meta_seed_42_pr_0.1.pt",
        map_location="cpu",
        weights_only=False,
    )
    weights = state["weights"].detach().numpy().astype(np.float64)
    classifier = adaptive.FixedClassifier(
        w0=state["classifier.0.weight"].numpy().astype(np.float64),
        b0=state["classifier.0.bias"].numpy().astype(np.float64),
        w3=state["classifier.3.weight"].numpy().astype(np.float64),
        b3=state["classifier.3.bias"].numpy().astype(np.float64),
    )
    x_target = meta["target_clean_data"].numpy().astype(np.float64)
    x_nontarget = meta["non_target_clean_data"].numpy().astype(np.float64)
    x_poison = meta["poisoned_non_target_data"].numpy().astype(np.float64)

    x_batch = np.vstack([x_target[:4], x_nontarget[:4], x_poison[:4]])
    y_batch = np.concatenate(
        [np.ones(4, dtype=int), np.zeros(4, dtype=int), np.ones(4, dtype=int)]
    )
    poison_batch = x_poison[:8]
    xyz_ref = adaptive.xyz_feats(x_target, weights, n_layers=8, n_qubits=8)
    lw = LedoitWolf().fit(xyz_ref)
    return (
        weights,
        classifier,
        x_batch,
        y_batch,
        poison_batch,
        lw.location_.astype(np.float64),
        lw.get_precision().astype(np.float64),
    )


def ce_value_and_z_gradient(classifier, z: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    preactivation = z @ classifier.w0.T + classifier.b0
    hidden = np.maximum(0.0, preactivation)
    logits = hidden @ classifier.w3.T + classifier.b3
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    loss = float(-np.log(probabilities[np.arange(len(labels)), labels]).mean())

    dlogits = probabilities
    dlogits[np.arange(len(labels)), labels] -= 1.0
    dlogits /= len(labels)
    dhidden = dlogits @ classifier.w3
    dpreactivation = dhidden * (preactivation > 0.0)
    dz = dpreactivation @ classifier.w0
    return loss, dz


def mahal_value_and_gradient(xyz: np.ndarray, mu: np.ndarray, precision: np.ndarray) -> tuple[float, np.ndarray]:
    diff = xyz - mu
    symmetric_precision = 0.5 * (precision + precision.T)
    value = float(np.einsum("ni,ij,nj->n", diff, symmetric_precision, diff).mean())
    gradient = 2.0 * diff @ symmetric_precision / len(xyz)
    return value, gradient


def objective(
    adaptive,
    weights: np.ndarray,
    classifier,
    x_batch: np.ndarray,
    labels: np.ndarray,
    poison_batch: np.ndarray,
    mu: np.ndarray,
    precision: np.ndarray,
    lambda_r: float,
) -> float:
    z = adaptive.z_feats(x_batch, weights, n_layers=8, n_qubits=8)
    ce = classifier.ce_loss(z, labels)
    if lambda_r == 0.0:
        return ce
    xyz = adaptive.xyz_feats(poison_batch, weights, n_layers=8, n_qubits=8)
    return ce + lambda_r * adaptive.mahal_loss(xyz, mu, precision)


def selected_parameter_indices() -> list[int]:
    indices: list[int] = []
    for layer in range(8):
        for qubit in (0, 4):
            base = (layer * 8 + qubit) * 2
            indices.extend([base, base + 1])
    return indices


def numerical_gradient(
    fn: Callable[[np.ndarray], float],
    weights: np.ndarray,
    indices: list[int],
    step: float,
) -> np.ndarray:
    gradient = np.zeros(len(indices), dtype=np.float64)
    for output_index, parameter_index in enumerate(indices):
        plus = weights.copy()
        minus = weights.copy()
        plus[parameter_index] += step
        minus[parameter_index] -= step
        gradient[output_index] = (fn(plus) - fn(minus)) / (2.0 * step)
    return gradient


def gradient_comparison(candidate: np.ndarray, reference: np.ndarray) -> dict:
    denominator = float(np.linalg.norm(candidate) * np.linalg.norm(reference))
    cosine = float(np.dot(candidate, reference) / denominator) if denominator else float("nan")
    reference_norm = float(np.linalg.norm(reference))
    return {
        "cosine_similarity": cosine,
        "relative_l2_error": float(np.linalg.norm(candidate - reference) / reference_norm),
        "sign_agreement": float(np.mean(np.sign(candidate) == np.sign(reference))),
        "candidate_norm": float(np.linalg.norm(candidate)),
        "reference_norm": reference_norm,
        "directional_derivative_for_negative_step": float(-np.dot(reference, candidate)),
    }


def gradient_audit(output: Path) -> dict:
    adaptive = load_adaptive_module()
    (
        weights,
        classifier,
        x_batch,
        labels,
        poison_batch,
        mu,
        precision,
    ) = load_gradient_fixture(adaptive)
    indices = selected_parameter_indices()
    rows: list[dict] = []
    summaries: dict[str, dict] = {}

    # First verify that parameter shift itself is correct for a raw expectation.
    feature_fn = lambda w: float(adaptive.z_feats(x_batch[:1], w, 8, 8)[0, 0])
    # numerical_gradient divides by 2*step; the exact two-eigenvalue
    # parameter-shift coefficient is 1/2, so rescale the pi/2 central quotient.
    feature_ps = (
        numerical_gradient(feature_fn, weights, indices, math.pi / 2) * math.pi / 2
    )
    feature_fd_1e4 = numerical_gradient(feature_fn, weights, indices, 1e-4)
    feature_fd_1e5 = numerical_gradient(feature_fn, weights, indices, 1e-5)
    summaries["raw_z_expectation_parameter_shift_vs_finite_difference"] = {
        **gradient_comparison(feature_ps, feature_fd_1e5),
        "finite_difference_crosscheck": gradient_comparison(feature_fd_1e4, feature_fd_1e5),
    }

    for lambda_r in (0.0, 1.0):
        base_z = adaptive.z_feats(x_batch, weights, 8, 8)
        _, dloss_dz = ce_value_and_z_gradient(classifier, base_z, labels)
        if lambda_r:
            base_xyz = adaptive.xyz_feats(poison_batch, weights, 8, 8)
            _, dloss_dxyz = mahal_value_and_gradient(base_xyz, mu, precision)
        else:
            dloss_dxyz = None

        historical = np.zeros(len(indices), dtype=np.float64)
        chain_rule = np.zeros(len(indices), dtype=np.float64)
        for output_index, parameter_index in enumerate(indices):
            plus = weights.copy()
            minus = weights.copy()
            plus[parameter_index] += math.pi / 2
            minus[parameter_index] -= math.pi / 2

            z_plus = adaptive.z_feats(x_batch, plus, 8, 8)
            z_minus = adaptive.z_feats(x_batch, minus, 8, 8)
            historical_ce = 0.5 * (
                classifier.ce_loss(z_plus, labels) - classifier.ce_loss(z_minus, labels)
            )
            dz_dtheta = 0.5 * (z_plus - z_minus)
            chain_ce = float(np.sum(dloss_dz * dz_dtheta))

            historical_repr = 0.0
            chain_repr = 0.0
            if lambda_r:
                xyz_plus = adaptive.xyz_feats(poison_batch, plus, 8, 8)
                xyz_minus = adaptive.xyz_feats(poison_batch, minus, 8, 8)
                historical_repr = 0.5 * (
                    adaptive.mahal_loss(xyz_plus, mu, precision)
                    - adaptive.mahal_loss(xyz_minus, mu, precision)
                )
                dxyz_dtheta = 0.5 * (xyz_plus - xyz_minus)
                chain_repr = float(np.sum(dloss_dxyz * dxyz_dtheta))

            historical[output_index] = historical_ce + lambda_r * historical_repr
            chain_rule[output_index] = chain_ce + lambda_r * chain_repr

        objective_fn = lambda w: objective(
            adaptive,
            w,
            classifier,
            x_batch,
            labels,
            poison_batch,
            mu,
            precision,
            lambda_r,
        )
        finite_1e4 = numerical_gradient(objective_fn, weights, indices, 1e-4)
        finite_1e5 = numerical_gradient(objective_fn, weights, indices, 1e-5)

        historical_full = np.zeros_like(weights)
        chain_rule_full = np.zeros_like(weights)
        finite_full = np.zeros_like(weights)
        historical_full[indices] = historical
        chain_rule_full[indices] = chain_rule
        finite_full[indices] = finite_1e5

        key = f"lambda_{lambda_r:g}"
        summaries[key] = {
            "historical_loss_shift_vs_finite_difference": gradient_comparison(
                historical, finite_1e5
            ),
            "chain_rule_parameter_shift_vs_finite_difference": gradient_comparison(
                chain_rule, finite_1e5
            ),
            "finite_difference_crosscheck": gradient_comparison(finite_1e4, finite_1e5),
            "objective_at_start": objective_fn(weights),
            "objective_after_historical_step_lr_0.01": objective_fn(
                weights - 0.01 * historical_full
            ),
            "objective_after_chain_rule_step_lr_0.01": objective_fn(
                weights - 0.01 * chain_rule_full
            ),
            "objective_after_fd_step_lr_0.01": objective_fn(weights - 0.01 * finite_full),
        }
        for output_index, parameter_index in enumerate(indices):
            rows.append(
                {
                    "lambda_r": lambda_r,
                    "parameter_index": parameter_index,
                    "historical_loss_shift": historical[output_index],
                    "chain_rule_parameter_shift": chain_rule[output_index],
                    "finite_difference_1e4": finite_1e4[output_index],
                    "finite_difference_1e5": finite_1e5[output_index],
                }
            )
    write_csv(output / "gradient_audit.csv", rows)
    return {
        "seed": 42,
        "layer": 8,
        "pair": "t6_vs_t0",
        "poison_ratio": 0.1,
        "parameters_checked": len(indices),
        "parameter_indices": indices,
        "batch_composition": "4 clean target + 4 clean non-target + 4 poisoned; 8 poisoned for representation loss",
        "summaries": summaries,
    }


def main(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "scope": "BloodMNIST CA diagnostic using frozen local artifacts",
        "source_grid": source_grid_audit(output),
        "adaptive_trajectory": adaptive_trajectory_audit(output),
        "gradient_audit": gradient_audit(output),
    }
    summary_path = output / "diagnostic_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "paper14/results/bloodmnist_ca_diagnosis_v1",
    )
    args = parser.parse_args()
    main(args.output.resolve())
