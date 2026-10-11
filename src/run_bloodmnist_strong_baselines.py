#!/usr/bin/env python3
"""Strong unsupervised baselines under the matched CF-CMD evaluation protocol."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_cross_attack_pilot as cross
import run_bloodmnist_group_coherent_10seeds as baseline
import run_bloodmnist_residual_reupload_pilot as residual
import run_bloodmnist_whitebox_adaptive_pilot as whitebox
import run_mnist_group_coherent_drift as coherent
import run_unified_oneclass_poison_screening as unified
import train_bloodmnist_clean_encoder as clean_encoder


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PAPER14_ROOT / "results/bloodmnist_strong_baselines_v1"
PROTOCOL = PAPER14_ROOT / "results/top_journal_program_v1/phase4_strong_baseline_protocol.json"
LOCAL_ROOT = PAPER14_ROOT / "results/bloodmnist_xz16_fresh_confirmation_v1"
ADAPTIVE_ROOT = PAPER14_ROOT / "results/bloodmnist_whitebox_adaptive_confirmation_v1"
RESIDUAL_ROOT = PAPER14_ROOT / "results/bloodmnist_residual_reupload_confirmation_v1"
SEEDS = tuple(range(52, 62))
SCENARIOS = (
    "fixed_attack_local_angle",
    "whitebox_representation_matching_local_angle",
    "fixed_attack_residual_reupload",
)
METHODS = ("spectral_signatures", "spectre_que", "activation_clustering", "scan_style")
VIEWS = ("checkpoint_drift_xz16", "joint_input256_xz16")


def ridge_whiten(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return coherent.whitening(np.asarray(reference, dtype=np.float64))


def standardized(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    mean = reference.mean(axis=0, keepdims=True)
    scale = reference.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return (values - mean) / scale


def fit_scores(
    method: str,
    reference: np.ndarray,
    calibration: np.ndarray,
    candidates: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    reference = np.asarray(reference, dtype=np.float64)
    calibration = np.asarray(calibration, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    if method == "spectral_signatures":
        reference_z = standardized(reference, reference)
        calibration_z = standardized(reference, calibration)
        candidate_z = standardized(reference, candidates)
        center = candidate_z.mean(axis=0, keepdims=True)
        _, _, right = np.linalg.svd(candidate_z - center, full_matrices=False)
        direction = right[0]
        candidate_scores = ((candidate_z - center) @ direction) ** 2
        calibration_scores = ((calibration_z - center) @ direction) ** 2
        receipt = {"rank": 1, "coordinate_scaling": "trusted_reference_std"}
    else:
        mean, transform = ridge_whiten(reference)
        reference_z = (reference - mean) @ transform
        calibration_z = (calibration - mean) @ transform
        candidate_z = (candidates - mean) @ transform
        if method == "spectre_que":
            covariance = candidate_z.T @ candidate_z / max(len(candidate_z) - 1, 1)
            values, vectors = np.linalg.eigh(covariance)
            span = max(float(values[-1] - values[0]), 1e-8)
            weights = np.exp(4.0 * (values - values[-1]) / span)
            weights /= weights.sum()
            candidate_scores = np.sum((candidate_z @ vectors) ** 2 * weights, axis=1)
            calibration_scores = np.sum((calibration_z @ vectors) ** 2 * weights, axis=1)
            receipt = {"que_temperature": 4.0, "whitening": "trusted_ridge"}
        else:
            dimension = min(16, candidate_z.shape[1], len(reference_z) - 1)
            reducer = PCA(n_components=dimension, random_state=seed).fit(
                np.concatenate([reference_z, candidate_z], axis=0)
            )
            reference_low = reducer.transform(reference_z)
            calibration_low = reducer.transform(calibration_z)
            candidate_low = reducer.transform(candidate_z)
            kmeans = KMeans(n_clusters=2, random_state=seed, n_init=20).fit(candidate_low)
            centers = kmeans.cluster_centers_
            reference_center = reference_low.mean(axis=0)
            poison_component = int(
                np.argmax(np.linalg.norm(centers - reference_center[None, :], axis=1))
            )
            clean_component = 1 - poison_component
            if method == "activation_clustering":
                def score(values: np.ndarray) -> np.ndarray:
                    poison_distance = np.sum((values - centers[poison_component]) ** 2, axis=1)
                    clean_distance = np.sum((values - centers[clean_component]) ** 2, axis=1)
                    return clean_distance - poison_distance
                candidate_scores = score(candidate_low)
                calibration_scores = score(calibration_low)
                receipt = {"pca_dimension": dimension, "n_clusters": 2}
            elif method == "scan_style":
                assignments = kmeans.labels_
                priors = np.asarray([(assignments == component).mean() for component in range(2)])
                centered_rows = []
                for component in range(2):
                    selected = candidate_low[assignments == component]
                    centered_rows.append(selected - centers[component])
                centered_all = np.concatenate(centered_rows, axis=0)
                covariance = centered_all.T @ centered_all / max(len(centered_all) - 2, 1)
                ridge = max(float(np.trace(covariance) / dimension) * 0.1, 1e-6)
                precision = np.linalg.inv(covariance + ridge * np.eye(dimension))
                def log_likelihood(values: np.ndarray, component: int) -> np.ndarray:
                    delta = values - centers[component]
                    return -0.5 * np.einsum("ni,ij,nj->n", delta, precision, delta) + np.log(max(priors[component], 1e-8))
                candidate_scores = log_likelihood(candidate_low, poison_component) - log_likelihood(candidate_low, clean_component)
                calibration_scores = log_likelihood(calibration_low, poison_component) - log_likelihood(calibration_low, clean_component)
                receipt = {"pca_dimension": dimension, "shared_covariance": True, "components": 2}
            else:
                raise ValueError(method)
    return candidate_scores, calibration_scores, receipt


def evaluate_method(
    scenario: str,
    view: str,
    method: str,
    seed: int,
    trusted: np.ndarray,
    clean: np.ndarray,
    poison: np.ndarray,
    alpha: float,
) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(20260928 + 1009 * seed)
    order = rng.permutation(len(trusted))
    reference, calibration = trusted[order[:120]], trusted[order[120:]]
    candidates = np.concatenate([clean, poison], axis=0)
    labels = np.concatenate([np.zeros(len(clean), dtype=int), np.ones(len(poison), dtype=int)])
    scores, calibration_scores, receipt = fit_scores(
        method, reference, calibration, candidates, seed
    )
    p_values = coherent.one_sided_p_values(calibration_scores, scores)
    flags = p_values <= alpha
    clean_flags, poison_flags = flags[labels == 0], flags[labels == 1]
    fp, tn = int(clean_flags.sum()), int((~clean_flags).sum())
    tp, fn = int(poison_flags.sum()), int((~poison_flags).sum())
    row = {
        "scenario": scenario,
        "seed": seed,
        "view": view,
        "method": method,
        "dimension": candidates.shape[1],
        "n_clean": len(clean),
        "n_poison": len(poison),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tpr": tp / (tp + fn),
        "fpr": fp / (fp + tn),
        "auc": float(roc_auc_score(labels, scores)),
        "config_pass": int(tp / (tp + fn) >= 0.80 and fp / (fp + tn) <= 0.10),
        "receipt_json": json.dumps(receipt, sort_keys=True, separators=(",", ":")),
    }
    sample_rows = [
        {
            "scenario": scenario,
            "seed": seed,
            "view": view,
            "method": method,
            "sample_index": index,
            "label": int(label),
            "score": float(score),
            "p_value": float(p_value),
            "flagged": int(flag),
        }
        for index, (label, score, p_value, flag) in enumerate(
            zip(labels, scores, p_values, flags)
        )
    ]
    return row, sample_rows


def load_scenario(
    scenario: str, seed: int, splits: dict
) -> tuple[object, object, float, float]:
    clean_local, _ = cross.load_reference_clean_model(seed, LOCAL_ROOT)
    if scenario == "fixed_attack_local_angle":
        attacked = clean_encoder.CleanQNN("local", 0)
        attacked.load_state_dict(
            torch.load(
                LOCAL_ROOT / "models" / f"seed_{seed}_adaptive.pt",
                map_location="cpu",
                weights_only=False,
            )["state_dict"]
        )
        clean, attacked_model = clean_local, attacked
    elif scenario == "whitebox_representation_matching_local_angle":
        checkpoint = (
            ADAPTIVE_ROOT
            / "models/checkpoint_drift_representation_matching"
            / f"seed_{seed}.pt"
        )
        attacked = clean_encoder.CleanQNN("local", 0)
        attacked.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"])
        clean, attacked_model = clean_local, attacked
    elif scenario == "fixed_attack_residual_reupload":
        clean = residual.ResidualReuploadQNN()
        clean.load_state_dict(
            torch.load(RESIDUAL_ROOT / "models" / f"seed_{seed}_clean.pt", map_location="cpu", weights_only=False)["state_dict"]
        )
        attacked_model = residual.ResidualReuploadQNN()
        attacked_model.load_state_dict(
            torch.load(RESIDUAL_ROOT / "models" / f"seed_{seed}_adaptive.pt", map_location="cpu", weights_only=False)["state_dict"]
        )
    else:
        raise ValueError(scenario)
    ca, asr = whitebox.balanced_ca_and_asr(
        attacked_model,
        splits["test"]["target"],
        splits["test"]["non_target"],
        splits["test"]["triggered_non_target"],
    )
    return clean, attacked_model, ca, asr


def views_for_scenario(
    scenario: str,
    clean_model: object,
    attacked_model: object,
    splits: dict,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict], list[dict]]:
    inputs = {
        "trusted": splits["val"]["target"].astype(np.float64),
        "clean": splits["train"]["target"].astype(np.float64),
        "poison": splits["archived_poison"].astype(np.float64),
    }
    if scenario == "fixed_attack_residual_reupload":
        xz = {
            name: residual.numpy_xz(attacked_model, values.astype(np.float32))
            - residual.numpy_xz(clean_model, values.astype(np.float32))
            for name, values in inputs.items()
        }
    else:
        xz = {
            name: whitebox.checkpoint_drift(clean_model, attacked_model, values.astype(np.float32))
            for name, values in inputs.items()
        }
    joint = {name: np.concatenate([inputs[name], xz[name]], axis=1) for name in inputs}
    config = {"seed": 0, "trigger": scenario, "archive_ca": 0.0, "archive_asr": 0.0}
    # Seed is filled by the caller before evaluating CF-CMD.
    return {"checkpoint_drift_xz16": xz, "joint_input256_xz16": joint}, inputs, xz


def aggregate(rows: list[dict], victims: list[dict], draws: int) -> list[dict]:
    eligible = {(row["scenario"], int(row["seed"])) for row in victims if row["eligible"]}
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if (row["scenario"], int(row["seed"])) in eligible:
            grouped[(row["scenario"], row["view"], row["method"])].append(row)
    rng = np.random.default_rng(20260928)
    output = []
    for key, selected in sorted(grouped.items()):
        tp, fn = sum(row["tp"] for row in selected), sum(row["fn"] for row in selected)
        fp, tn = sum(row["fp"] for row in selected), sum(row["tn"] for row in selected)
        tpr = np.asarray([row["tpr"] for row in selected])
        fpr = np.asarray([row["fpr"] for row in selected])
        indices = rng.integers(0, len(selected), size=(draws, len(selected)))
        output.append(
            {
                "scenario": key[0],
                "view": key[1],
                "method": key[2],
                "eligible_seed_count": len(selected),
                "pooled_tpr": tp / (tp + fn),
                "pooled_fpr": fp / (fp + tn),
                "tpr_ci_low": float(np.quantile(tpr[indices].mean(1), 0.025)),
                "tpr_ci_high": float(np.quantile(tpr[indices].mean(1), 0.975)),
                "fpr_ci_low": float(np.quantile(fpr[indices].mean(1), 0.025)),
                "fpr_ci_high": float(np.quantile(fpr[indices].mean(1), 0.975)),
            }
        )
    return output


def main(args: argparse.Namespace) -> None:
    if tuple(args.seeds) != SEEDS:
        raise ValueError("Strong-baseline comparison is frozen to seeds 52--61")
    args.output.mkdir(parents=True, exist_ok=True)
    class_data = heldout.collect_classes(args.dataset_root)
    rgb_trigger, _ = baseline.fixed_rgb_trigger(class_data)
    rows, samples, victims = [], [], []
    for seed in args.seeds:
        print(f"seed={seed}: shared features for four strong baselines", flush=True)
        prepared = baseline.preprocess_seed(class_data, seed)
        splits, _ = baseline.seed_splits(prepared, rgb_trigger, seed)
        for scenario in SCENARIOS:
            clean_model, attacked_model, ca, asr = load_scenario(scenario, seed, splits)
            eligible = int(ca >= 0.90 and asr >= 0.80)
            victims.append({"scenario": scenario, "seed": seed, "balanced_ca": ca, "asr": asr, "eligible": eligible})
            views, inputs, xz = views_for_scenario(scenario, clean_model, attacked_model, splits)
            for view in VIEWS:
                groups = views[view]
                for method in METHODS:
                    row, sample_rows = evaluate_method(
                        scenario,
                        view,
                        method,
                        seed,
                        groups["trusted"],
                        groups["clean"],
                        groups["poison"],
                        args.alpha,
                    )
                    rows.append(row)
                    samples.extend(sample_rows)

            config = {"seed": seed, "trigger": scenario, "archive_ca": ca, "archive_asr": asr}
            cf_input = cross.evaluate_external_reference(
                config, "input256", inputs["trusted"], inputs["clean"], inputs["poison"], args.alpha
            )
            cf_xz = cross.evaluate_external_reference(
                config, "checkpoint_drift_xz16", xz["trusted"], xz["clean"], xz["poison"], args.alpha
            )
            cf_primary = coherent.combine_margin_and_correlator(
                config,
                cf_input[0],
                cf_input[1],
                cf_xz[1],
                args.alpha,
                combined_name="joint_multiview",
                second_dimension=16,
            )
            for view, result in (("checkpoint_drift_xz16", cf_xz), ("joint_multiview", cf_primary)):
                base_row = result[0]
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "view": view,
                        "method": "cf_cmd",
                        "dimension": base_row["dimension"],
                        "n_clean": base_row["n_clean"],
                        "n_poison": base_row["n_poison"],
                        "tp": base_row["tp"],
                        "fn": base_row["fn"],
                        "fp": base_row["fp"],
                        "tn": base_row["tn"],
                        "tpr": base_row["tpr"],
                        "fpr": base_row["fpr"],
                        "auc": base_row["auc_conformal_rank"],
                        "config_pass": base_row["config_pass"],
                        "receipt_json": base_row["fold_receipts_json"],
                    }
                )
    summaries = aggregate(rows, victims, args.bootstrap_draws)
    unified.write_csv(args.output / "per_config_metrics.csv", rows)
    unified.write_csv(args.output / "per_sample_scores_baselines.csv", samples)
    unified.write_csv(args.output / "victim_metrics.csv", victims)
    unified.write_csv(args.output / "eligible_summary.csv", summaries)
    unified.json_dump(
        args.output / "manifest.json",
        {
            "experiment": "bloodmnist_strong_baselines_v1",
            "protocol": str(PROTOCOL),
            "protocol_sha256": whitebox.sha256_file(PROTOCOL),
            "dataset_npz_sha256": whitebox.sha256_file(args.dataset_root / "bloodmnist.npz"),
            "seeds": list(args.seeds),
            "scenarios": list(SCENARIOS),
            "methods": list(METHODS),
            "views": list(VIEWS),
            "alpha": args.alpha,
            "bootstrap_draws": args.bootstrap_draws,
            "fidelity_note": "SPECTRE/QUE and SCAn-style are mechanism-faithful clean-room baselines, not bit-identical full-pipeline reproductions.",
        },
    )
    lines = [
        "# BloodMNIST matched strong-baseline comparison",
        "",
        "All methods use identical victims, candidate pools, external reference/calibration samples and a clean-calibrated 5% decision level.",
        "",
        "| scenario | view | method | eligible seeds | pooled TPR | pooled FPR |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['scenario']} | {row['view']} | {row['method']} | {row['eligible_seed_count']} | {row['pooled_tpr']:.3f} | {row['pooled_fpr']:.3f} |"
        )
    lines.extend(
        [
            "",
            "SPECTRE/QUE and SCAn-style are clean-room mechanism baselines under the shared external-reference protocol; they are not claimed to be bit-identical reproductions of the authors' full pipelines.",
            "",
        ]
    )
    (args.output / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    main(parser.parse_args())
