#!/usr/bin/env python3
"""Independent raw-artifact replay for the auditability geometry diagnostic."""

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
DEFAULT_RESULT = PAPER14_ROOT / "results/auditability_geometry_diagnostic_v1"
DEFAULT_PROTOCOL = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/auditability_geometry_diagnostic_protocol_v1.json"
)
DEFAULT_RECEIPT = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/auditability_geometry_diagnostic_receipt.json"
)
EPSILON = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def close(left: Any, right: Any, tolerance: float = 1e-8) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def independent_phase_fit(
    clean: np.ndarray, attack: np.ndarray, ridge_fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    clean = np.asarray(clean, dtype=np.float64)
    attack = np.asarray(attack, dtype=np.float64)
    mean = clean.mean(axis=0)
    delta = attack.mean(axis=0) - mean
    centered = clean - mean
    mean_variance = float(np.sum(centered**2) / ((len(clean) - 1) * clean.shape[1]))
    ridge = max(ridge_fraction * mean_variance, 1e-10)
    system = (len(clean) - 1) * np.eye(len(clean)) + centered @ centered.T / ridge
    direction = delta / ridge - centered.T @ np.linalg.solve(
        system, centered @ delta
    ) / (ridge * ridge)
    coherence = float(np.sqrt(max(float(delta @ direction), 0.0)))
    return mean, delta, direction, coherence


def independent_covariance_fit(
    clean: np.ndarray, poison: np.ndarray, ridge_fraction: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    clean = np.asarray(clean, dtype=np.float64)
    poison = np.asarray(poison, dtype=np.float64)
    mean = clean.mean(axis=0)
    delta = poison.mean(axis=0) - mean
    covariance = np.cov(clean, rowvar=False, ddof=1)
    ridge = max(ridge_fraction * float(np.trace(covariance) / clean.shape[1]), 1e-10)
    direction = np.linalg.solve(covariance + ridge * np.eye(clean.shape[1]), delta)
    coherence = float(np.sqrt(max(float(delta @ direction), 0.0)))
    return mean, direction, ridge, coherence


def independent_gamma(values: np.ndarray) -> float:
    distances = pdist(values, metric="sqeuclidean")
    distances = distances[distances > 0.0]
    return float(1.0 / (2.0 * np.median(distances))) if len(distances) else 1.0


def independent_crossfit(
    clean: np.ndarray, poison: np.ndarray, fixed: dict[str, Any]
) -> tuple[float, float]:
    values = np.concatenate([clean, poison], axis=0).astype(np.float64)
    labels = np.concatenate([np.zeros(len(clean), dtype=int), np.ones(len(poison), dtype=int)])
    splitter = StratifiedKFold(
        n_splits=int(fixed["cross_validation_folds"]),
        shuffle=True,
        random_state=int(fixed["cross_validation_seed"]),
    )
    fisher = np.empty(len(values), dtype=float)
    nonlinear = np.empty(len(values), dtype=float)
    for train, test in splitter.split(values, labels):
        train_clean = values[train][labels[train] == 0]
        train_poison = values[train][labels[train] == 1]
        center, direction, _, _ = independent_covariance_fit(
            train_clean,
            train_poison,
            float(fixed["ridge_fraction_of_mean_clean_variance"]),
        )
        fisher[test] = (values[test] - center) @ direction
        scaler = StandardScaler().fit(values[train])
        training = scaler.transform(values[train])
        classifier = SVC(
            C=float(fixed["rbf_svc_c"]),
            kernel="rbf",
            gamma=independent_gamma(training),
            class_weight="balanced",
        )
        classifier.fit(training, labels[train])
        nonlinear[test] = classifier.decision_function(scaler.transform(values[test]))
    return float(roc_auc_score(labels, fisher)), float(roc_auc_score(labels, nonlinear))


def independent_geometry(
    clean: np.ndarray, poison: np.ndarray, fixed: dict[str, Any]
) -> dict[str, float]:
    clean = np.asarray(clean, dtype=np.float64)
    poison = np.asarray(poison, dtype=np.float64)
    center, _, ridge, coherence = independent_covariance_fit(
        clean, poison, float(fixed["ridge_fraction_of_mean_clean_variance"])
    )
    delta = poison.mean(axis=0) - center
    clean_radius = np.linalg.norm(clean - center, axis=1)
    poison_vectors = poison - center
    poison_radius = np.linalg.norm(poison_vectors, axis=1)
    unit = np.zeros_like(poison_vectors)
    nonzero = poison_radius > EPSILON
    unit[nonzero] = poison_vectors[nonzero] / poison_radius[nonzero, None]
    rms = float(np.sqrt(np.mean(clean_radius**2)))
    energy = max(
        2.0 * float(cdist(clean, poison).mean())
        - float(squareform(pdist(clean)).mean())
        - float(squareform(pdist(poison)).mean()),
        0.0,
    )
    scaler = StandardScaler().fit(np.concatenate([clean, poison], axis=0))
    clean_z, poison_z = scaler.transform(clean), scaler.transform(poison)
    gamma = independent_gamma(np.concatenate([clean_z, poison_z], axis=0))
    mmd = max(
        float(np.exp(-gamma * cdist(clean_z, clean_z, "sqeuclidean")).mean())
        + float(np.exp(-gamma * cdist(poison_z, poison_z, "sqeuclidean")).mean())
        - 2.0 * float(np.exp(-gamma * cdist(clean_z, poison_z, "sqeuclidean")).mean()),
        0.0,
    )
    fisher_auc, rbf_auc = independent_crossfit(clean, poison, fixed)
    return {
        "ridge_coherence": coherence,
        "ridge_value": ridge,
        "mean_shift_ratio": float(np.linalg.norm(delta) / max(rms, EPSILON)),
        "mean_resultant_length": float(np.linalg.norm(unit.mean(axis=0))),
        "poison_to_clean_radial_ratio": float(
            np.median(poison_radius) / max(np.median(clean_radius), EPSILON)
        ),
        "energy_distance_v_statistic": energy,
        "rbf_mmd2_biased": mmd,
        "rbf_mmd_gamma": gamma,
        "crossfit_fisher_auc": fisher_auc,
        "crossfit_rbf_auc": rbf_auc,
    }


def independent_bootstrap(
    values: np.ndarray, draws: int, seed: int, confidence: float
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(samples, (tail, 1.0 - tail))
    return float(values.mean()), float(low), float(high)


def verify_phase(
    source: Path,
    stored_rows: list[dict[str, str]],
    protocol: dict[str, Any],
    errors: list[str],
) -> int:
    indexed = {
        (int(row["seed"]), row["checkpoint"], row["representation"], row["view"]): row
        for row in stored_rows
    }
    config = protocol["sources"]["phase_a"]
    split_map = phase_probe.load_split_map(source / "splits.csv")
    indices = phase_probe.pauli_indices(5)
    verified = 0
    for seed in config["seeds"]:
        for checkpoint in config["checkpoints"]:
            cache = phase_probe.load_pair_cache(source, seed, checkpoint)
            masks = phase_probe.split_masks(seed, cache["pair_ids"], split_map)
            for representation in config["representations"]:
                columns = indices[representation]
                for view in config["views"]:
                    if view == "suspect_absolute":
                        clean = cache["suspect_clean"][:, columns]
                        attack = cache["suspect_attack"][:, columns]
                    else:
                        clean = cache["suspect_clean"][:, columns] - cache["clean_clean"][:, columns]
                        attack = cache["suspect_attack"][:, columns] - cache["clean_attack"][:, columns]
                    train_clean, train_attack = clean[masks["train"]], attack[masks["train"]]
                    test_clean, test_attack = clean[masks["test"]], attack[masks["test"]]
                    center, train_delta, direction, coherence = independent_phase_fit(
                        train_clean,
                        train_attack,
                        float(protocol["fixed_methods"]["ridge_fraction_of_mean_clean_variance"]),
                    )
                    test_delta = test_attack.mean(axis=0) - test_clean.mean(axis=0)
                    train_cosine = np.asarray(train_delta, dtype=np.float64)
                    test_cosine = np.asarray(test_delta, dtype=np.float64)
                    denominator = float(
                        np.sqrt(np.mean(np.sum((test_clean - test_clean.mean(axis=0)) ** 2, axis=1)))
                    )
                    labels = np.concatenate([np.zeros(len(test_clean)), np.ones(len(test_attack))])
                    scores = np.concatenate(
                        [(test_clean - center) @ direction, (test_attack - center) @ direction]
                    )
                    cosine_denominator = float(
                        np.linalg.norm(train_cosine) * np.linalg.norm(test_cosine)
                    )
                    expected = {
                        "regularized_train_coherence": coherence,
                        "test_mean_shift_ratio": float(np.linalg.norm(test_delta) / max(denominator, EPSILON)),
                        "train_test_direction_cosine": float(np.dot(train_cosine, test_cosine) / cosine_denominator) if cosine_denominator > 1e-15 else 0.0,
                        "train_test_direction_instability_ratio": float(np.linalg.norm(train_delta - test_delta) / max(np.linalg.norm(train_delta), EPSILON)),
                        "test_linear_auc": float(roc_auc_score(labels, scores)),
                    }
                    stored = indexed[(seed, checkpoint, representation, view)]
                    for field, value in expected.items():
                        if not close(stored[field], value):
                            errors.append(f"Phase-A {field} mismatch: {seed} {checkpoint} {representation} {view}")
                    verified += 1
    return verified


def verify_blood(
    source: Path,
    stored_rows: list[dict[str, str]],
    protocol: dict[str, Any],
    errors: list[str],
) -> int:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    victims = {(row["mode"], int(row["seed"])): row for row in read_csv(source / "victim_metrics.csv")}
    receipts = {(row["mode"], int(row["seed"])): row for row in manifest["receipts"]}
    stored = {(row["mode"], int(row["seed"])): row for row in stored_rows}
    config = protocol["sources"]["bloodmnist_whitebox"]
    class_data = heldout.collect_classes(Path(manifest["dataset_root"]))
    rgb_trigger, _ = baseline.fixed_rgb_trigger(class_data)
    verified = 0
    for seed in config["seeds"]:
        print(f"verify BloodMNIST seed {seed}", flush=True)
        seed_data = pilot.prepare_seed(
            class_data, rgb_trigger, seed, clean_root=Path(manifest["clean_qml_root"])
        )
        clean = seed_data["splits"]["train"]["target"]
        for mode in config["modes"]:
            victim = victims[(mode, seed)]
            receipt = receipts[(mode, seed)]
            checkpoint_path = source / victim["checkpoint"]
            if sha256_file(checkpoint_path) != receipt["checkpoint_sha256"]:
                errors.append(f"checkpoint hash mismatch: {mode} {seed}")
                continue
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            attacked = clean_encoder.CleanQNN("local", 0)
            attacked.load_state_dict(checkpoint["state_dict"])
            attacked.eval()
            poison = (
                p3verify.reconstructed_joint(seed_data, checkpoint, "train")[0]
                if mode == "joint_input_internal_detector_evasion"
                else seed_data["splits"]["archived_poison"]
            )
            clean_xz = pilot.checkpoint_drift(seed_data["clean_model"], attacked, clean)
            poison_xz = pilot.checkpoint_drift(seed_data["clean_model"], attacked, poison)
            if pilot.sha256_array(poison) != receipt["poison_sha256_float32"]:
                errors.append(f"poison hash mismatch: {mode} {seed}")
            if pilot.sha256_array(poison_xz) != receipt["xz_poison_sha256_float32"]:
                errors.append(f"XZ hash mismatch: {mode} {seed}")
            expected = independent_geometry(clean_xz, poison_xz, protocol["fixed_methods"])
            for field, value in expected.items():
                if not close(stored[(mode, seed)][field], value):
                    errors.append(f"BloodMNIST {field} mismatch: {mode} {seed}")
            if int(stored[(mode, seed)]["test_eligible"]) != int(victim["test_eligible"]):
                errors.append(f"eligibility mismatch: {mode} {seed}")
            verified += 1
    return verified


def verify_aggregates(
    result: Path, protocol: dict[str, Any], errors: list[str]
) -> tuple[int, int, dict[str, Any]]:
    rows = read_csv(result / "blood_per_victim.csv")
    summaries = read_csv(result / "blood_population_summary.csv")
    summary_index = {(row["mode"], row["population"], row["metric"]): row for row in summaries}
    config = protocol["sources"]["bloodmnist_whitebox"]
    fixed = protocol["fixed_methods"]
    primary: dict[str, dict[str, dict[str, str]]] = {mode: {} for mode in config["modes"]}
    checked_summary = 0
    for mode_index, mode in enumerate(config["modes"]):
        for population in ("stored_test_eligible_victims", "all_archived_victims"):
            selected = [
                row for row in rows if row["mode"] == mode and (population == "all_archived_victims" or int(row["test_eligible"]) == 1)
            ]
            for metric_index, metric in enumerate(protocol["estimands"]["bloodmnist_whitebox"]):
                values = np.asarray([float(row[metric]) for row in selected])
                expected = independent_bootstrap(
                    values,
                    int(fixed["bootstrap_draws"]),
                    int(fixed["bootstrap_seed"]) + 100 * mode_index + metric_index,
                    float(fixed["confidence_level"]),
                )
                stored = summary_index[(mode, population, metric)]
                for field, value in zip(("mean", "bootstrap_ci_low", "bootstrap_ci_high"), expected):
                    if not close(stored[field], value):
                        errors.append(f"summary {field} mismatch: {mode} {population} {metric}")
                if int(stored["seed_count"]) != len(selected):
                    errors.append(f"summary seed count mismatch: {mode} {population} {metric}")
                if population == "stored_test_eligible_victims":
                    primary[mode][metric] = stored
                checked_summary += 1

    contrast_rows = read_csv(result / "blood_primary_paired_contrasts.csv")
    contrast_index = {row["metric"]: row for row in contrast_rows}
    left, right = protocol["primary_contrast"]["left"], protocol["primary_contrast"]["right"]
    indexed = {(row["mode"], int(row["seed"])): row for row in rows}
    common = sorted(
        set(int(row["seed"]) for row in rows if row["mode"] == left and int(row["test_eligible"]) == 1)
        & set(int(row["seed"]) for row in rows if row["mode"] == right and int(row["test_eligible"]) == 1)
    )
    for metric_index, metric in enumerate(protocol["primary_contrast"]["metrics"]):
        differences = np.asarray(
            [float(indexed[(left, seed)][metric]) - float(indexed[(right, seed)][metric]) for seed in common]
        )
        expected = independent_bootstrap(
            differences,
            int(fixed["bootstrap_draws"]),
            int(fixed["bootstrap_seed"]) + 1000 + metric_index,
            float(fixed["confidence_level"]),
        )
        stored = contrast_index[metric]
        for field, value in zip(("mean_paired_delta", "bootstrap_ci_low", "bootstrap_ci_high"), expected):
            if not close(stored[field], value):
                errors.append(f"contrast {field} mismatch: {metric}")

    rules = protocol["decision_rule"]
    relative = float(primary[left]["mean_shift_ratio"]["mean"]) / max(
        float(primary[right]["mean_shift_ratio"]["mean"]), EPSILON
    )
    resultant = float(contrast_index["mean_resultant_length"]["mean_paired_delta"])
    auc_low = float(primary[left]["crossfit_rbf_auc"]["bootstrap_ci_low"])
    cancellation = relative <= float(rules["mean_shift_relative_max"]) and resultant <= float(rules["resultant_length_paired_delta_max"])
    residual = auc_low > float(rules["residual_nonlinear_auc_ci_low_min"])
    label = (
        rules["labels"]["cancellation_with_residual_signal"]
        if cancellation and residual
        else rules["labels"]["broad_suppression"]
        if cancellation
        else rules["labels"]["mixed"]
    )
    decision = json.loads((result / "decision.json").read_text(encoding="utf-8"))
    expected_decision = {
        "label": label,
        "common_eligible_seeds": common,
        "multi_to_representation_mean_shift_ratio": relative,
        "multi_minus_representation_mean_resultant_length": resultant,
        "multi_direction_crossfit_rbf_auc_ci_low": auc_low,
        "first_order_cancellation_gate": bool(cancellation),
        "residual_nonlinear_information_gate": bool(residual),
        "scope": "retrospective_label_informed_xz16_mechanism_diagnostic",
    }
    if decision != expected_decision:
        errors.append("decision replay mismatch")
    return checked_summary, len(contrast_rows), expected_decision


def verify(args: argparse.Namespace) -> dict[str, Any]:
    result, protocol_path = args.result.resolve(), args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    if sha256_file(protocol_path) != manifest["protocol_sha256"]:
        errors.append("protocol hash mismatch")
    runner = PAPER14_ROOT / manifest["runner"]
    if sha256_file(runner) != manifest["runner_sha256"]:
        errors.append("runner hash mismatch")
    for item in manifest["output_hashes"].values():
        path = PAPER14_ROOT / item["path"]
        if sha256_file(path) != item["sha256"]:
            errors.append(f"output hash mismatch: {item['path']}")
    phase_root = PAPER14_ROOT / protocol["sources"]["phase_a"]["root"]
    blood_root = PAPER14_ROOT / protocol["sources"]["bloodmnist_whitebox"]["root"]
    source_checks = (
        (phase_root / "manifest.json", protocol["sources"]["phase_a"]["manifest_sha256"]),
        (phase_root / "splits.csv", protocol["sources"]["phase_a"]["splits_sha256"]),
        (blood_root / "manifest.json", protocol["sources"]["bloodmnist_whitebox"]["manifest_sha256"]),
        (blood_root / "victim_metrics.csv", protocol["sources"]["bloodmnist_whitebox"]["victim_metrics_sha256"]),
    )
    for path, expected in source_checks:
        if sha256_file(path) != expected:
            errors.append(f"source hash mismatch: {path}")

    phase_rows = read_csv(result / "phase_a_per_configuration.csv")
    blood_rows = read_csv(result / "blood_per_victim.csv")
    if len(phase_rows) != 36:
        errors.append(f"expected 36 Phase-A rows, found {len(phase_rows)}")
    if len(blood_rows) != 30:
        errors.append(f"expected 30 BloodMNIST rows, found {len(blood_rows)}")
    phase_verified = verify_phase(phase_root, phase_rows, protocol, errors)
    blood_verified = verify_blood(blood_root, blood_rows, protocol, errors)
    summary_verified, contrasts_verified, decision = verify_aggregates(result, protocol, errors)
    receipt = {
        "status": "PASS" if not errors else "FAIL",
        "experiment": manifest["experiment"],
        "protocol_sha256": sha256_file(protocol_path),
        "analysis_manifest_sha256": sha256_file(result / "manifest.json"),
        "verifier": str(Path(__file__).relative_to(PAPER14_ROOT)),
        "verifier_sha256": sha256_file(Path(__file__)),
        "phase_a_configurations_replayed": phase_verified,
        "bloodmnist_victims_replayed": blood_verified,
        "population_summaries_replayed": summary_verified,
        "paired_contrasts_replayed": contrasts_verified,
        "decision": decision,
        "errors": errors,
    }
    write_json(args.receipt, receipt)
    if errors:
        raise RuntimeError("; ".join(errors))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    receipt = verify(parser.parse_args())
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
