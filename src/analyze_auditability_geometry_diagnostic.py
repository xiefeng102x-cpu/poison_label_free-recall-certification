#!/usr/bin/env python3
"""Run the frozen cache-only auditability geometry diagnostic.

This script never trains a victim.  It reconstructs fixed feature channels from
archived Phase-A caches and BloodMNIST checkpoints, verifies their fingerprints,
and computes the label-informed diagnostics frozen in the protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import analyze_checkpoint_auditability_theory_probe as phase_probe
import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_group_coherent_10seeds as baseline
import run_bloodmnist_whitebox_adaptive_pilot as pilot
import train_bloodmnist_clean_encoder as clean_encoder
import verify_top_journal_phase3_adaptive as p3verify


PAPER14_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/auditability_geometry_diagnostic_protocol_v1.json"
)
DEFAULT_OUTPUT = PAPER14_ROOT / "results/auditability_geometry_diagnostic_v1"
EPSILON = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_frozen_sources(protocol: dict[str, Any]) -> tuple[Path, Path]:
    phase_root = PAPER14_ROOT / protocol["sources"]["phase_a"]["root"]
    blood_root = PAPER14_ROOT / protocol["sources"]["bloodmnist_whitebox"]["root"]
    expected = protocol["sources"]
    checks = (
        (phase_root / "manifest.json", expected["phase_a"]["manifest_sha256"]),
        (phase_root / "splits.csv", expected["phase_a"]["splits_sha256"]),
        (blood_root / "manifest.json", expected["bloodmnist_whitebox"]["manifest_sha256"]),
        (blood_root / "victim_metrics.csv", expected["bloodmnist_whitebox"]["victim_metrics_sha256"]),
    )
    for path, expected_hash in checks:
        actual = sha256_file(path)
        if actual != expected_hash:
            raise RuntimeError(f"frozen source hash mismatch: {path}: {actual}")
    manifest = json.loads((blood_root / "manifest.json").read_text(encoding="utf-8"))
    dataset = Path(manifest["dataset_root"]) / "bloodmnist.npz"
    if sha256_file(dataset) != expected["bloodmnist_whitebox"]["dataset_sha256"]:
        raise RuntimeError("BloodMNIST dataset hash mismatch")
    return phase_root, blood_root


def root_mean_square_radius(values: np.ndarray, center: np.ndarray) -> float:
    radius_squared = np.sum((np.asarray(values) - center) ** 2, axis=1)
    return float(np.sqrt(np.mean(radius_squared)))


def phase_a_diagnostic(
    source: Path, protocol: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    config = protocol["sources"]["phase_a"]
    fixed = protocol["fixed_methods"]
    split_map = phase_probe.load_split_map(source / "splits.csv")
    indices = phase_probe.pauli_indices(5)
    rows: list[dict[str, Any]] = []
    cache_receipts: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        for checkpoint in config["checkpoints"]:
            cache = phase_probe.load_pair_cache(source, seed, checkpoint)
            masks = phase_probe.split_masks(seed, cache["pair_ids"], split_map)
            for role in ("suspect_path", "clean_path"):
                path = Path(str(cache[role]))
                cache_receipts.append(
                    {
                        "seed": seed,
                        "checkpoint": checkpoint,
                        "role": role,
                        "path": str(path.relative_to(PAPER14_ROOT)),
                        "sha256": sha256_file(path),
                    }
                )
            for representation in config["representations"]:
                columns = indices[representation]
                views = {
                    "suspect_absolute": (
                        cache["suspect_clean"][:, columns],
                        cache["suspect_attack"][:, columns],
                    ),
                    "paired_checkpoint_drift": (
                        cache["suspect_clean"][:, columns]
                        - cache["clean_clean"][:, columns],
                        cache["suspect_attack"][:, columns]
                        - cache["clean_attack"][:, columns],
                    ),
                }
                for view in config["views"]:
                    clean, attack = views[view]
                    train_clean = clean[masks["train"]]
                    train_attack = attack[masks["train"]]
                    test_clean = clean[masks["test"]]
                    test_attack = attack[masks["test"]]
                    fit = phase_probe.ridge_discriminant(
                        train_clean,
                        train_attack,
                        float(fixed["ridge_fraction_of_mean_clean_variance"]),
                    )
                    train_delta = np.asarray(fit["delta"], dtype=np.float64)
                    test_delta = test_attack.mean(axis=0) - test_clean.mean(axis=0)
                    labels = np.concatenate(
                        [np.zeros(len(test_clean), dtype=int), np.ones(len(test_attack), dtype=int)]
                    )
                    scores = np.concatenate(
                        [phase_probe.score(test_clean, fit), phase_probe.score(test_attack, fit)]
                    )
                    denominator = root_mean_square_radius(test_clean, test_clean.mean(axis=0))
                    rows.append(
                        {
                            "seed": seed,
                            "checkpoint": checkpoint,
                            "representation": representation,
                            "dimension": int(len(columns)),
                            "view": view,
                            "train_pairs": int(masks["train"].sum()),
                            "test_pairs": int(masks["test"].sum()),
                            "regularized_train_coherence": float(fit["ridge_effect_size"]),
                            "test_mean_shift_ratio": float(np.linalg.norm(test_delta) / max(denominator, EPSILON)),
                            "train_test_direction_cosine": phase_probe.safe_cosine(train_delta, test_delta),
                            "train_test_direction_instability_ratio": float(
                                np.linalg.norm(train_delta - test_delta)
                                / max(np.linalg.norm(train_delta), EPSILON)
                            ),
                            "test_linear_auc": float(roc_auc_score(labels, scores)),
                        }
                    )
    aggregates: list[dict[str, Any]] = []
    metrics = protocol["estimands"]["phase_a"]
    for checkpoint in config["checkpoints"]:
        for representation in config["representations"]:
            for view in config["views"]:
                selected = [
                    row
                    for row in rows
                    if row["checkpoint"] == checkpoint
                    and row["representation"] == representation
                    and row["view"] == view
                ]
                aggregate: dict[str, Any] = {
                    "checkpoint": checkpoint,
                    "representation": representation,
                    "dimension": selected[0]["dimension"],
                    "view": view,
                    "seed_count": len(selected),
                }
                for metric in metrics:
                    values = np.asarray([row[metric] for row in selected], dtype=float)
                    aggregate[f"mean_{metric}"] = float(values.mean())
                    aggregate[f"std_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    aggregate[f"min_{metric}"] = float(values.min())
                    aggregate[f"max_{metric}"] = float(values.max())
                aggregates.append(aggregate)
    return rows, aggregates, cache_receipts


def covariance_ridge_direction(
    clean: np.ndarray, poison: np.ndarray, ridge_fraction: float
) -> tuple[np.ndarray, float, float, np.ndarray]:
    clean = np.asarray(clean, dtype=np.float64)
    poison = np.asarray(poison, dtype=np.float64)
    clean_mean = clean.mean(axis=0)
    delta = poison.mean(axis=0) - clean_mean
    covariance = np.cov(clean, rowvar=False, ddof=1)
    mean_variance = float(np.trace(covariance) / clean.shape[1])
    ridge = max(ridge_fraction * mean_variance, 1e-10)
    direction = np.linalg.solve(covariance + ridge * np.eye(clean.shape[1]), delta)
    coherence = float(np.sqrt(max(float(delta @ direction), 0.0)))
    return direction, ridge, coherence, clean_mean


def median_gamma(values: np.ndarray) -> float:
    distances = pdist(np.asarray(values, dtype=np.float64), metric="sqeuclidean")
    positive = distances[distances > 0.0]
    if len(positive) == 0:
        return 1.0
    return float(1.0 / (2.0 * np.median(positive)))


def energy_distance_v(clean: np.ndarray, poison: np.ndarray) -> float:
    cross = float(cdist(clean, poison, metric="euclidean").mean())
    within_clean = float(squareform(pdist(clean, metric="euclidean")).mean())
    within_poison = float(squareform(pdist(poison, metric="euclidean")).mean())
    return max(2.0 * cross - within_clean - within_poison, 0.0)


def rbf_mmd2_biased(clean: np.ndarray, poison: np.ndarray) -> tuple[float, float]:
    scaler = StandardScaler().fit(np.concatenate([clean, poison], axis=0))
    clean_z = scaler.transform(clean)
    poison_z = scaler.transform(poison)
    pooled = np.concatenate([clean_z, poison_z], axis=0)
    gamma = median_gamma(pooled)
    kcc = np.exp(-gamma * cdist(clean_z, clean_z, metric="sqeuclidean"))
    kpp = np.exp(-gamma * cdist(poison_z, poison_z, metric="sqeuclidean"))
    kcp = np.exp(-gamma * cdist(clean_z, poison_z, metric="sqeuclidean"))
    return max(float(kcc.mean() + kpp.mean() - 2.0 * kcp.mean()), 0.0), gamma


def crossfit_aucs(
    clean: np.ndarray,
    poison: np.ndarray,
    *,
    folds: int,
    random_seed: int,
    ridge_fraction: float,
    svc_c: float,
) -> tuple[float, float]:
    values = np.concatenate([clean, poison], axis=0).astype(np.float64)
    labels = np.concatenate([np.zeros(len(clean), dtype=int), np.ones(len(poison), dtype=int)])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    fisher_scores = np.empty(len(values), dtype=np.float64)
    rbf_scores = np.empty(len(values), dtype=np.float64)
    for train, test in splitter.split(values, labels):
        train_clean = values[train][labels[train] == 0]
        train_poison = values[train][labels[train] == 1]
        direction, _, _, clean_mean = covariance_ridge_direction(
            train_clean, train_poison, ridge_fraction
        )
        fisher_scores[test] = (values[test] - clean_mean) @ direction

        scaler = StandardScaler().fit(values[train])
        train_z = scaler.transform(values[train])
        test_z = scaler.transform(values[test])
        gamma = median_gamma(train_z)
        classifier = SVC(
            C=svc_c,
            kernel="rbf",
            gamma=gamma,
            class_weight="balanced",
        )
        classifier.fit(train_z, labels[train])
        rbf_scores[test] = classifier.decision_function(test_z)
    return float(roc_auc_score(labels, fisher_scores)), float(roc_auc_score(labels, rbf_scores))


def geometry_metrics(
    clean: np.ndarray, poison: np.ndarray, fixed: dict[str, Any]
) -> dict[str, float]:
    clean = np.asarray(clean, dtype=np.float64)
    poison = np.asarray(poison, dtype=np.float64)
    _, ridge, coherence, clean_mean = covariance_ridge_direction(
        clean, poison, float(fixed["ridge_fraction_of_mean_clean_variance"])
    )
    delta = poison.mean(axis=0) - clean_mean
    clean_radius = np.linalg.norm(clean - clean_mean, axis=1)
    poison_vectors = poison - clean_mean
    poison_radius = np.linalg.norm(poison_vectors, axis=1)
    unit = np.divide(
        poison_vectors,
        poison_radius[:, None],
        out=np.zeros_like(poison_vectors),
        where=poison_radius[:, None] > EPSILON,
    )
    mmd2, mmd_gamma = rbf_mmd2_biased(clean, poison)
    fisher_auc, rbf_auc = crossfit_aucs(
        clean,
        poison,
        folds=int(fixed["cross_validation_folds"]),
        random_seed=int(fixed["cross_validation_seed"]),
        ridge_fraction=float(fixed["ridge_fraction_of_mean_clean_variance"]),
        svc_c=float(fixed["rbf_svc_c"]),
    )
    return {
        "ridge_coherence": coherence,
        "ridge_value": ridge,
        "mean_shift_ratio": float(np.linalg.norm(delta) / max(root_mean_square_radius(clean, clean_mean), EPSILON)),
        "mean_resultant_length": float(np.linalg.norm(unit.mean(axis=0))),
        "poison_to_clean_radial_ratio": float(np.median(poison_radius) / max(np.median(clean_radius), EPSILON)),
        "energy_distance_v_statistic": energy_distance_v(clean, poison),
        "rbf_mmd2_biased": mmd2,
        "rbf_mmd_gamma": mmd_gamma,
        "crossfit_fisher_auc": fisher_auc,
        "crossfit_rbf_auc": rbf_auc,
    }


def bootstrap_mean_ci(
    values: np.ndarray, draws: int, seed: int, confidence: float
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        raise ValueError("bootstrap sample cannot be empty")
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[chosen].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, (tail, 1.0 - tail))
    return float(values.mean()), float(low), float(high)


def blood_diagnostic(
    source: Path, protocol: dict[str, Any]
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    config = protocol["sources"]["bloodmnist_whitebox"]
    fixed = protocol["fixed_methods"]
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    victims = {
        (row["mode"], int(row["seed"])): row
        for row in read_csv(source / "victim_metrics.csv")
    }
    receipts = {
        (row["mode"], int(row["seed"])): row for row in manifest["receipts"]
    }
    class_data = heldout.collect_classes(Path(manifest["dataset_root"]))
    rgb_trigger, _ = baseline.fixed_rgb_trigger(class_data)
    rows: list[dict[str, Any]] = []
    reconstruction: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        print(f"BloodMNIST seed {seed}: reconstructing fixed XZ16 views", flush=True)
        seed_data = pilot.prepare_seed(
            class_data,
            rgb_trigger,
            seed,
            clean_root=Path(manifest["clean_qml_root"]),
        )
        clean = seed_data["splits"]["train"]["target"]
        for mode in config["modes"]:
            key = (mode, seed)
            victim = victims[key]
            receipt = receipts[key]
            checkpoint_path = source / victim["checkpoint"]
            checkpoint_hash = sha256_file(checkpoint_path)
            if checkpoint_hash != victim["checkpoint_sha256"] or checkpoint_hash != receipt["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint fingerprint mismatch: {mode} seed {seed}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            attacked = clean_encoder.CleanQNN("local", 0)
            attacked.load_state_dict(checkpoint["state_dict"])
            attacked.eval()
            if mode == "joint_input_internal_detector_evasion":
                poison, _ = p3verify.reconstructed_joint(seed_data, checkpoint, "train")
            else:
                poison = seed_data["splits"]["archived_poison"]
            poison_hash = pilot.sha256_array(poison)
            if poison_hash != receipt["poison_sha256_float32"]:
                raise RuntimeError(f"poison fingerprint mismatch: {mode} seed {seed}")
            clean_xz = pilot.checkpoint_drift(seed_data["clean_model"], attacked, clean)
            poison_xz = pilot.checkpoint_drift(seed_data["clean_model"], attacked, poison)
            poison_xz_hash = pilot.sha256_array(poison_xz)
            if poison_xz_hash != receipt["xz_poison_sha256_float32"]:
                raise RuntimeError(f"XZ fingerprint mismatch: {mode} seed {seed}")
            metrics = geometry_metrics(clean_xz, poison_xz, fixed)
            rows.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "test_eligible": int(victim["test_eligible"]),
                    "test_balanced_ca": float(victim["test_balanced_ca"]),
                    "test_asr": float(victim["test_asr"]),
                    "clean_count": len(clean_xz),
                    "poison_count": len(poison_xz),
                    "dimension": clean_xz.shape[1],
                    **metrics,
                }
            )
            reconstruction.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "checkpoint": str(checkpoint_path.relative_to(PAPER14_ROOT)),
                    "checkpoint_sha256": checkpoint_hash,
                    "poison_sha256_float32": poison_hash,
                    "xz_clean_sha256_float32": pilot.sha256_array(clean_xz),
                    "xz_poison_sha256_float32": poison_xz_hash,
                }
            )

    metric_names = protocol["estimands"]["bloodmnist_whitebox"]
    summaries: list[dict[str, Any]] = []
    primary_by_mode: dict[str, dict[str, dict[str, float]]] = {}
    for mode_index, mode in enumerate(config["modes"]):
        primary_by_mode[mode] = {}
        for population in ("stored_test_eligible_victims", "all_archived_victims"):
            selected = [
                row
                for row in rows
                if row["mode"] == mode
                and (population == "all_archived_victims" or row["test_eligible"] == 1)
            ]
            if not selected:
                continue
            for metric_index, metric in enumerate(metric_names):
                values = np.asarray([row[metric] for row in selected], dtype=float)
                mean, low, high = bootstrap_mean_ci(
                    values,
                    int(fixed["bootstrap_draws"]),
                    int(fixed["bootstrap_seed"]) + 100 * mode_index + metric_index,
                    float(fixed["confidence_level"]),
                )
                summary = {
                    "mode": mode,
                    "population": population,
                    "metric": metric,
                    "seed_count": len(selected),
                    "mean": mean,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
                summaries.append(summary)
                if population == "stored_test_eligible_victims":
                    primary_by_mode[mode][metric] = summary

    left = protocol["primary_contrast"]["left"]
    right = protocol["primary_contrast"]["right"]
    common = sorted(
        set(row["seed"] for row in rows if row["mode"] == left and row["test_eligible"] == 1)
        & set(row["seed"] for row in rows if row["mode"] == right and row["test_eligible"] == 1)
    )
    indexed = {(row["mode"], row["seed"]): row for row in rows}
    contrasts: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(protocol["primary_contrast"]["metrics"]):
        differences = np.asarray(
            [indexed[(left, seed)][metric] - indexed[(right, seed)][metric] for seed in common],
            dtype=float,
        )
        mean, low, high = bootstrap_mean_ci(
            differences,
            int(fixed["bootstrap_draws"]),
            int(fixed["bootstrap_seed"]) + 1000 + metric_index,
            float(fixed["confidence_level"]),
        )
        contrasts.append(
            {
                "left_mode": left,
                "right_mode": right,
                "metric": metric,
                "common_eligible_seed_count": len(common),
                "common_eligible_seeds": ";".join(map(str, common)),
                "mean_paired_delta": mean,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "minimum_paired_delta": float(differences.min()),
                "maximum_paired_delta": float(differences.max()),
            }
        )

    rules = protocol["decision_rule"]
    multi_mean_shift = primary_by_mode[left]["mean_shift_ratio"]["mean"]
    rep_mean_shift = primary_by_mode[right]["mean_shift_ratio"]["mean"]
    relative_mean_shift = float(multi_mean_shift / max(rep_mean_shift, EPSILON))
    resultant_delta = next(
        row["mean_paired_delta"] for row in contrasts if row["metric"] == "mean_resultant_length"
    )
    residual_auc_low = primary_by_mode[left]["crossfit_rbf_auc"]["bootstrap_ci_low"]
    cancellation = bool(
        relative_mean_shift <= float(rules["mean_shift_relative_max"])
        and resultant_delta <= float(rules["resultant_length_paired_delta_max"])
    )
    residual = bool(residual_auc_low > float(rules["residual_nonlinear_auc_ci_low_min"]))
    if cancellation and residual:
        label = rules["labels"]["cancellation_with_residual_signal"]
    elif cancellation:
        label = rules["labels"]["broad_suppression"]
    else:
        label = rules["labels"]["mixed"]
    decision = {
        "label": label,
        "common_eligible_seeds": common,
        "multi_to_representation_mean_shift_ratio": relative_mean_shift,
        "multi_minus_representation_mean_resultant_length": resultant_delta,
        "multi_direction_crossfit_rbf_auc_ci_low": residual_auc_low,
        "first_order_cancellation_gate": cancellation,
        "residual_nonlinear_information_gate": residual,
        "scope": "retrospective_label_informed_xz16_mechanism_diagnostic",
    }
    return rows, summaries, contrasts, reconstruction, decision


def report_text(
    phase_aggregate: list[dict[str, Any]],
    blood_summary: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    decision: dict[str, Any],
) -> str:
    phase_selected = [
        row
        for row in phase_aggregate
        if row["view"] == "paired_checkpoint_drift"
    ]
    rbf_rows = [
        row
        for row in blood_summary
        if row["population"] == "stored_test_eligible_victims"
        and row["metric"] == "crossfit_rbf_auc"
    ]
    lines = [
        "# Auditability geometry diagnostic",
        "",
        "This is a retrospective, label-informed mechanism diagnostic over fixed archived artifacts. It is not a deployable detector evaluation and it does not add a fresh confirmatory attack result.",
        "",
        f"Frozen decision label: **{decision['label']}**",
        f"Common eligible seeds in the primary paired contrast: {decision['common_eligible_seeds']}",
        f"Multi-direction / representation-matching mean-shift ratio: {decision['multi_to_representation_mean_shift_ratio']:.4f}",
        f"Paired mean-resultant-length delta: {decision['multi_minus_representation_mean_resultant_length']:.4f}",
        f"Multi-direction nonlinear AUC bootstrap lower bound: {decision['multi_direction_crossfit_rbf_auc_ci_low']:.4f}",
        "",
        "## BloodMNIST primary-population nonlinear information",
        "",
        "| attack mode | eligible seeds | mean cross-fit RBF AUC | 95% seed-bootstrap CI |",
        "|---|---:|---:|---:|",
    ]
    for row in rbf_rows:
        lines.append(
            f"| {row['mode']} | {row['seed_count']} | {row['mean']:.4f} | "
            f"[{row['bootstrap_ci_low']:.4f}, {row['bootstrap_ci_high']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Phase-A paired-drift geometry",
            "",
            "| checkpoint | representation | dimension | mean train coherence | mean test shift ratio | mean test linear AUC |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in phase_selected:
        lines.append(
            f"| {row['checkpoint']} | {row['representation']} | {row['dimension']} | "
            f"{row['mean_regularized_train_coherence']:.4f} | "
            f"{row['mean_test_mean_shift_ratio']:.4f} | {row['mean_test_linear_auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Mandatory boundary",
            "",
            "Eligibility is inherited from the already stored CA/ASR gate. Labels are used to fit the diagnostic classifiers. MMD, energy distance and cross-fit AUC characterize discrepancy inside the fixed XZ16 channel; they neither prove causality nor establish unsupervised screening performance. The primary paired contrast contains only the common eligible seeds listed above.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    phase_root, blood_root = verify_frozen_sources(protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    phase_rows, phase_aggregate, phase_receipts = phase_a_diagnostic(phase_root, protocol)
    blood_rows, blood_summary, contrasts, reconstruction, decision = blood_diagnostic(
        blood_root, protocol
    )
    output_paths = {
        "phase_a_per_configuration": args.output / "phase_a_per_configuration.csv",
        "phase_a_aggregate": args.output / "phase_a_aggregate.csv",
        "blood_per_victim": args.output / "blood_per_victim.csv",
        "blood_population_summary": args.output / "blood_population_summary.csv",
        "blood_primary_paired_contrasts": args.output / "blood_primary_paired_contrasts.csv",
        "source_reconstruction_receipts": args.output / "source_reconstruction_receipts.json",
        "decision": args.output / "decision.json",
        "report": args.output / "EXPERIMENT_REPORT.md",
    }
    write_csv(output_paths["phase_a_per_configuration"], phase_rows)
    write_csv(output_paths["phase_a_aggregate"], phase_aggregate)
    write_csv(output_paths["blood_per_victim"], blood_rows)
    write_csv(output_paths["blood_population_summary"], blood_summary)
    write_csv(output_paths["blood_primary_paired_contrasts"], contrasts)
    write_json(
        output_paths["source_reconstruction_receipts"],
        {"phase_a_cache_receipts": phase_receipts, "bloodmnist_reconstruction_receipts": reconstruction},
    )
    write_json(output_paths["decision"], decision)
    output_paths["report"].write_text(
        report_text(phase_aggregate, blood_summary, contrasts, decision), encoding="utf-8"
    )
    output_hashes = {
        name: {"path": str(path.relative_to(PAPER14_ROOT)), "sha256": sha256_file(path)}
        for name, path in output_paths.items()
    }
    manifest = {
        "experiment": "auditability_geometry_diagnostic_v1",
        "role": "retrospective_label_informed_mechanism_diagnostic",
        "protocol": str(PROTOCOL.relative_to(PAPER14_ROOT)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner": str(Path(__file__).relative_to(PAPER14_ROOT)),
        "runner_sha256": sha256_file(Path(__file__)),
        "phase_a_source": str(phase_root.relative_to(PAPER14_ROOT)),
        "bloodmnist_source": str(blood_root.relative_to(PAPER14_ROOT)),
        "phase_a_configuration_count": len(phase_rows),
        "bloodmnist_victim_count": len(blood_rows),
        "decision": decision,
        "output_hashes": output_hashes,
        "claim_boundaries": protocol["claim_boundaries"],
    }
    write_json(args.output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
