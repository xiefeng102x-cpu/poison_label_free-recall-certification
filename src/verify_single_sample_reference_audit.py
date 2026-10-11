#!/usr/bin/env python3
"""Independent raw-artifact replay for the single-sample reference audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import roc_auc_score

import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_group_coherent_10seeds as baseline
import run_bloodmnist_whitebox_adaptive_pilot as pilot
import train_bloodmnist_clean_encoder as clean_encoder
import verify_top_journal_phase3_adaptive as p3verify


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = PAPER14_ROOT / "results/single_sample_reference_audit_v1"
DEFAULT_PROTOCOL = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/single_sample_reference_audit_protocol_v1.json"
)
DEFAULT_RECEIPT = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/single_sample_reference_audit_receipt.json"
)
SCORERS = ("ridge_mahalanobis", "rbf_reference_witness", "knn5_reference_distance")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_int_array(values: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
    return hashlib.sha256(values.tobytes()).hexdigest()


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
    return bool(
        np.isclose(float(left), float(right), rtol=tolerance, atol=tolerance)
    )


def independent_fit(
    reference: np.ndarray, ridge_fraction: float, neighbors: int
) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    mean = np.sum(reference, axis=0) / len(reference)
    centered = reference - mean
    covariance = centered.T @ centered / (len(reference) - 1)
    ridge = max(
        ridge_fraction * float(np.trace(covariance) / reference.shape[1]), 1e-10
    )
    precision = np.linalg.solve(
        covariance + ridge * np.eye(reference.shape[1]),
        np.eye(reference.shape[1]),
    )
    scale = np.maximum(
        np.sqrt(np.sum(centered**2, axis=0) / len(reference)), 1e-8
    )
    reference_z = centered / scale
    pairwise = pdist(reference_z, metric="sqeuclidean")
    positive = pairwise[pairwise > 0.0]
    gamma = float(1.0 / (2.0 * np.median(positive))) if len(positive) else 1.0
    return {
        "mean": mean,
        "precision": precision,
        "ridge": ridge,
        "scale": scale,
        "reference_z": reference_z,
        "gamma": gamma,
        "neighbors": neighbors,
    }


def independent_scores(state: dict[str, Any], query: np.ndarray) -> dict[str, np.ndarray]:
    query = np.asarray(query, dtype=np.float64)
    centered = query - state["mean"]
    mahalanobis = np.sum((centered @ state["precision"]) * centered, axis=1)
    squared = cdist(
        centered / state["scale"], state["reference_z"], metric="sqeuclidean"
    )
    rbf = 1.0 - np.mean(np.exp(-state["gamma"] * squared), axis=1)
    distances = np.sqrt(np.maximum(squared, 0.0))
    neighbor_index = int(state["neighbors"]) - 1
    knn = np.sort(distances, axis=1)[:, neighbor_index]
    return {
        "ridge_mahalanobis": mahalanobis,
        "rbf_reference_witness": rbf,
        "knn5_reference_distance": knn,
    }


def independent_pvalues(calibration: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            (1.0 + np.count_nonzero(calibration >= value))
            / (len(calibration) + 1.0)
            for value in query
        ],
        dtype=np.float64,
    )


def independent_calibrate(
    reference: np.ndarray,
    calibration: np.ndarray,
    populations: dict[str, np.ndarray],
    protocol: dict[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    fixed = protocol["fixed_scorers"]
    state = independent_fit(
        reference,
        float(
            fixed["ridge_mahalanobis"]["ridge_fraction_of_mean_reference_variance"]
        ),
        int(fixed["knn5_reference_distance"]["neighbors"]),
    )
    calibration_scores = independent_scores(state, calibration)
    outputs: dict[str, dict[str, np.ndarray]] = {}
    for population, values in populations.items():
        scores = independent_scores(state, values)
        pvalues = {
            scorer: independent_pvalues(calibration_scores[scorer], scores[scorer])
            for scorer in SCORERS
        }
        fused = np.minimum(
            1.0,
            len(SCORERS)
            * np.min(np.stack([pvalues[scorer] for scorer in SCORERS]), axis=0),
        )
        outputs[population] = {
            **{f"score_{name}": value for name, value in scores.items()},
            **{f"p_{name}": value for name, value in pvalues.items()},
            "fused_p_value": fused,
        }
    return outputs, state


def independent_auc(clean: np.ndarray, poison: np.ndarray) -> float:
    labels = np.concatenate(
        [np.zeros(len(clean), dtype=int), np.ones(len(poison), dtype=int)]
    )
    return float(roc_auc_score(labels, np.concatenate([clean, poison])))


def independent_metrics(
    outputs: dict[str, dict[str, np.ndarray]], alpha: float
) -> dict[str, Any]:
    member, fresh, poison = (
        outputs["member_clean"],
        outputs["fresh_clean"],
        outputs["poison"],
    )
    member_flags = member["fused_p_value"] <= alpha
    fresh_flags = fresh["fused_p_value"] <= alpha
    poison_flags = poison["fused_p_value"] <= alpha
    result: dict[str, Any] = {
        "poison_tp": int(np.count_nonzero(poison_flags)),
        "poison_fn": int(np.count_nonzero(~poison_flags)),
        "member_clean_fp": int(np.count_nonzero(member_flags)),
        "member_clean_tn": int(np.count_nonzero(~member_flags)),
        "fresh_clean_fp": int(np.count_nonzero(fresh_flags)),
        "fresh_clean_tn": int(np.count_nonzero(~fresh_flags)),
        "poison_tpr": float(np.mean(poison_flags)),
        "member_clean_fpr": float(np.mean(member_flags)),
        "fresh_clean_fpr": float(np.mean(fresh_flags)),
        "poison_auc_against_member_clean": independent_auc(
            -member["fused_p_value"], -poison["fused_p_value"]
        ),
        "poison_auc_against_fresh_clean": independent_auc(
            -fresh["fused_p_value"], -poison["fused_p_value"]
        ),
        "median_poison_fused_p_value": float(
            np.median(poison["fused_p_value"])
        ),
        "minimum_poison_fused_p_value": float(np.min(poison["fused_p_value"])),
    }
    for scorer in SCORERS:
        result[f"{scorer}_auc_member"] = independent_auc(
            member[f"score_{scorer}"], poison[f"score_{scorer}"]
        )
        result[f"{scorer}_auc_fresh"] = independent_auc(
            fresh[f"score_{scorer}"], poison[f"score_{scorer}"]
        )
    return result


def verify_one_victim(
    *,
    mode: str,
    seed: int,
    seed_data: dict[str, Any],
    source: Path,
    victim: dict[str, str],
    receipt: dict[str, Any],
    protocol: dict[str, Any],
    stored_victim: dict[str, str],
    stored_samples: dict[tuple[str, int, str, int], dict[str, str]],
    stored_feature: dict[str, Any],
    errors: list[str],
) -> tuple[dict[str, Any], int]:
    checkpoint_path = source / victim["checkpoint"]
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != receipt["checkpoint_sha256"]:
        errors.append(f"checkpoint hash mismatch: {mode} {seed}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    attacked = clean_encoder.CleanQNN("local", 0)
    attacked.load_state_dict(checkpoint["state_dict"])
    attacked.eval()
    poison_input = (
        p3verify.reconstructed_joint(seed_data, checkpoint, "train")[0]
        if mode == "joint_input_internal_detector_evasion"
        else seed_data["splits"]["archived_poison"]
    )
    if pilot.sha256_array(poison_input) != receipt["poison_sha256_float32"]:
        errors.append(f"poison hash mismatch: {mode} {seed}")
    inputs = {
        "trusted": seed_data["splits"]["val"]["target"],
        "member_clean": seed_data["splits"]["train"]["target"],
        "fresh_clean": seed_data["splits"]["test"]["target"],
        "poison": poison_input,
    }
    drifts = {
        name: pilot.checkpoint_drift(seed_data["clean_model"], attacked, values)
        for name, values in inputs.items()
    }
    if pilot.sha256_array(drifts["poison"]) != receipt["xz_poison_sha256_float32"]:
        errors.append(f"poison XZ hash mismatch: {mode} {seed}")
    rng = np.random.default_rng(20261003 + seed)
    order = rng.permutation(len(drifts["trusted"]))
    split = int(np.floor(len(order) * float(protocol["trusted_split"]["reference_fraction"])))
    reference_indices, calibration_indices = order[:split], order[split:]
    outputs, state = independent_calibrate(
        drifts["trusted"][reference_indices],
        drifts["trusted"][calibration_indices],
        {name: drifts[name] for name in ("member_clean", "fresh_clean", "poison")},
        protocol,
    )
    alpha = float(protocol["calibration_and_fusion"]["alpha"])
    metrics = independent_metrics(outputs, alpha)
    integer_fields = (
        "poison_tp",
        "poison_fn",
        "member_clean_fp",
        "member_clean_tn",
        "fresh_clean_fp",
        "fresh_clean_tn",
    )
    for field, value in metrics.items():
        matched = (
            int(stored_victim[field]) == int(value)
            if field in integer_fields
            else close(stored_victim[field], value)
        )
        if not matched:
            errors.append(f"victim {field} mismatch: {mode} {seed}")
    count_expectations = {
        "reference_count": len(reference_indices),
        "calibration_count": len(calibration_indices),
        "member_clean_count": len(drifts["member_clean"]),
        "fresh_clean_count": len(drifts["fresh_clean"]),
        "poison_count": len(drifts["poison"]),
        "test_eligible": int(victim["test_eligible"]),
    }
    for field, value in count_expectations.items():
        if int(stored_victim[field]) != value:
            errors.append(f"victim {field} mismatch: {mode} {seed}")
    for field in ("test_balanced_ca", "test_asr"):
        if not close(stored_victim[field], victim[field]):
            errors.append(f"victim {field} mismatch: {mode} {seed}")

    expected_feature = {
        "checkpoint_sha256": checkpoint_hash,
        "reference_indices_sha256_int64": sha256_int_array(reference_indices),
        "calibration_indices_sha256_int64": sha256_int_array(calibration_indices),
        "trusted_xz_sha256_float32": pilot.sha256_array(drifts["trusted"]),
        "member_clean_xz_sha256_float32": pilot.sha256_array(drifts["member_clean"]),
        "fresh_clean_xz_sha256_float32": pilot.sha256_array(drifts["fresh_clean"]),
        "poison_xz_sha256_float32": pilot.sha256_array(drifts["poison"]),
        "reference_count": len(reference_indices),
        "calibration_count": len(calibration_indices),
        "reference_mean_sha256_float32": pilot.sha256_array(
            np.asarray(state["mean"], dtype=np.float32)
        ),
        "reference_scale_sha256_float32": pilot.sha256_array(
            np.asarray(state["scale"], dtype=np.float32)
        ),
    }
    for field, value in expected_feature.items():
        if stored_feature[field] != value:
            errors.append(f"feature receipt {field} mismatch: {mode} {seed}")
    for field, value in (("ridge_value", state["ridge"]), ("rbf_gamma", state["gamma"])):
        if not close(stored_feature[field], value):
            errors.append(f"feature receipt {field} mismatch: {mode} {seed}")

    sample_count = 0
    for population in ("member_clean", "fresh_clean", "poison"):
        values = outputs[population]
        for index in range(len(values["fused_p_value"])):
            stored = stored_samples[(mode, seed, population, index)]
            expected_label = int(population == "poison")
            expected_flag = int(values["fused_p_value"][index] <= alpha)
            if int(stored["label"]) != expected_label or int(stored["flagged"]) != expected_flag:
                errors.append(f"sample decision mismatch: {mode} {seed} {population} {index}")
            for field, array in values.items():
                if not close(stored[field], array[index]):
                    errors.append(f"sample {field} mismatch: {mode} {seed} {population} {index}")
            sample_count += 1
    return {
        "mode": mode,
        "seed": seed,
        "test_eligible": int(victim["test_eligible"]),
        **metrics,
    }, sample_count


def independent_bootstrap(
    values: np.ndarray, draws: int, seed: int, confidence: float
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[selected].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, (tail, 1.0 - tail))
    return float(values.mean()), float(low), float(high)


def verify_aggregates(
    replayed: list[dict[str, Any]],
    stored_rows: list[dict[str, str]],
    protocol: dict[str, Any],
    errors: list[str],
) -> tuple[int, dict[str, Any]]:
    stored = {(row["mode"], row["population"]): row for row in stored_rows}
    reported = protocol["reported_endpoints"]
    primary: dict[str, dict[str, Any]] = {}
    checked = 0
    for mode_index, mode in enumerate(protocol["source"]["modes"]):
        for population in (
            "stored_test_eligible_victims_per_attack_mode",
            "all_archived_victims",
        ):
            selected = [
                row
                for row in replayed
                if row["mode"] == mode
                and (
                    population == "all_archived_victims"
                    or row["test_eligible"] == 1
                )
            ]
            row = stored[(mode, population)]
            confusion = {
                field: sum(item[field] for item in selected)
                for field in (
                    "poison_tp",
                    "poison_fn",
                    "member_clean_fp",
                    "member_clean_tn",
                    "fresh_clean_fp",
                    "fresh_clean_tn",
                )
            }
            for field, value in confusion.items():
                if int(row[field]) != value:
                    errors.append(f"aggregate {field} mismatch: {mode} {population}")
            rates = {
                "pooled_poison_tpr": confusion["poison_tp"]
                / max(confusion["poison_tp"] + confusion["poison_fn"], 1),
                "pooled_member_clean_fpr": confusion["member_clean_fp"]
                / max(confusion["member_clean_fp"] + confusion["member_clean_tn"], 1),
                "pooled_fresh_clean_fpr": confusion["fresh_clean_fp"]
                / max(confusion["fresh_clean_fp"] + confusion["fresh_clean_tn"], 1),
            }
            if int(row["victim_count"]) != len(selected):
                errors.append(f"aggregate victim_count mismatch: {mode} {population}")
            if row["seeds"] != ";".join(str(item["seed"]) for item in selected):
                errors.append(f"aggregate seeds mismatch: {mode} {population}")
            for field, value in rates.items():
                if not close(row[field], value):
                    errors.append(f"aggregate {field} mismatch: {mode} {population}")
            for metric_index, metric in enumerate(
                (
                    "poison_tpr",
                    "member_clean_fpr",
                    "fresh_clean_fpr",
                    "poison_auc_against_member_clean",
                    "poison_auc_against_fresh_clean",
                )
            ):
                expected = independent_bootstrap(
                    np.asarray([item[metric] for item in selected]),
                    int(reported["bootstrap_draws"]),
                    int(reported["bootstrap_seed"]) + 100 * mode_index + metric_index,
                    float(reported["confidence_level"]),
                )
                for field, value in zip(
                    (
                        f"mean_{metric}",
                        f"{metric}_ci_low",
                        f"{metric}_ci_high",
                    ),
                    expected,
                ):
                    if not close(row[field], value):
                        errors.append(f"aggregate {field} mismatch: {mode} {population}")
            for scorer in SCORERS:
                for suffix, source_field in (
                    ("member", f"{scorer}_auc_member"),
                    ("fresh", f"{scorer}_auc_fresh"),
                ):
                    value = float(np.mean([item[source_field] for item in selected]))
                    if not close(row[f"mean_{scorer}_auc_{suffix}"], value):
                        errors.append(f"aggregate scorer AUC mismatch: {mode} {population} {scorer} {suffix}")
            if population == "stored_test_eligible_victims_per_attack_mode":
                primary[mode] = {
                    "eligible_victims": len(selected),
                    **rates,
                }
            checked += 1

    rules = protocol["decision_rule"]

    def passes(mode: str, gate: dict[str, Any]) -> bool:
        row = primary[mode]
        return bool(
            row["eligible_victims"] >= int(rules["minimum_eligible_victims_per_mode"])
            and row["pooled_poison_tpr"] >= float(gate["pooled_poison_tpr_min"])
            and row["pooled_member_clean_fpr"] <= float(gate["pooled_member_clean_fpr_max"])
            and row["pooled_fresh_clean_fpr"] <= float(gate["pooled_fresh_clean_fpr_max"])
        )

    strong = [
        mode
        for mode in protocol["source"]["modes"]
        if passes(mode, rules["strong_gate"])
    ]
    partial = [
        mode
        for mode in protocol["source"]["modes"]
        if passes(mode, rules["partial_gate"])
    ]
    if len(strong) >= int(rules["strong_gate"]["required_modes"]):
        label = rules["strong_gate"]["label"]
    elif len(partial) >= int(rules["partial_gate"]["required_modes"]):
        label = rules["partial_gate"]["label"]
    else:
        label = rules["otherwise"]
    decision = {
        "label": label,
        "strong_gate_modes": strong,
        "partial_gate_modes": partial,
        "mode_endpoints": primary,
        "scope": "retrospective_development_only_single_query_reference_audit",
        "fresh_confirmation_authorized": bool(
            label == rules["strong_gate"]["label"]
        ),
    }
    return checked, decision


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
    source = PAPER14_ROOT / protocol["source"]["root"]
    source_checks = (
        (source / "manifest.json", protocol["source"]["manifest_sha256"]),
        (source / "victim_metrics.csv", protocol["source"]["victim_metrics_sha256"]),
    )
    for path, expected in source_checks:
        if sha256_file(path) != expected:
            errors.append(f"source hash mismatch: {path}")
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(Path(source_manifest["dataset_root"]) / "bloodmnist.npz") != protocol["source"]["dataset_sha256"]:
        errors.append("dataset hash mismatch")

    victim_rows = read_csv(result / "per_victim_metrics.csv")
    sample_rows = read_csv(result / "per_sample_scores.csv")
    summary_rows = read_csv(result / "population_summary.csv")
    feature_rows = json.loads((result / "feature_receipts.json").read_text(encoding="utf-8"))
    if len(victim_rows) != 30:
        errors.append(f"expected 30 victim rows, found {len(victim_rows)}")
    if len(sample_rows) != int(manifest["sample_score_rows"]):
        errors.append("sample row count does not match manifest")
    stored_victims = {(row["mode"], int(row["seed"])): row for row in victim_rows}
    stored_samples = {
        (row["mode"], int(row["seed"]), row["population"], int(row["sample_index"])): row
        for row in sample_rows
    }
    stored_features = {(row["mode"], int(row["seed"])): row for row in feature_rows}
    source_victims = {
        (row["mode"], int(row["seed"])): row
        for row in read_csv(source / "victim_metrics.csv")
    }
    source_receipts = {
        (row["mode"], int(row["seed"])): row
        for row in source_manifest["receipts"]
    }
    class_data = heldout.collect_classes(Path(source_manifest["dataset_root"]))
    rgb_trigger, _ = baseline.fixed_rgb_trigger(class_data)
    replayed: list[dict[str, Any]] = []
    replayed_samples = 0
    for seed in protocol["source"]["seeds"]:
        print(f"verify single-sample seed {seed}", flush=True)
        seed_data = pilot.prepare_seed(
            class_data,
            rgb_trigger,
            seed,
            clean_root=Path(source_manifest["clean_qml_root"]),
        )
        for mode in protocol["source"]["modes"]:
            key = (mode, seed)
            row, count = verify_one_victim(
                mode=mode,
                seed=seed,
                seed_data=seed_data,
                source=source,
                victim=source_victims[key],
                receipt=source_receipts[key],
                protocol=protocol,
                stored_victim=stored_victims[key],
                stored_samples=stored_samples,
                stored_feature=stored_features[key],
                errors=errors,
            )
            replayed.append(row)
            replayed_samples += count
    if replayed_samples != len(sample_rows):
        errors.append(
            f"sample replay count mismatch: replayed {replayed_samples}, stored {len(sample_rows)}"
        )
    summaries, decision = verify_aggregates(
        replayed, summary_rows, protocol, errors
    )
    stored_decision = json.loads((result / "decision.json").read_text(encoding="utf-8"))
    if stored_decision != decision:
        errors.append("decision replay mismatch")
    if manifest["decision"] != decision:
        errors.append("manifest decision mismatch")
    receipt = {
        "status": "PASS" if not errors else "FAIL",
        "experiment": manifest["experiment"],
        "protocol_sha256": sha256_file(protocol_path),
        "analysis_manifest_sha256": sha256_file(result / "manifest.json"),
        "verifier": str(Path(__file__).relative_to(PAPER14_ROOT)),
        "verifier_sha256": sha256_file(Path(__file__)),
        "victims_replayed": len(replayed),
        "sample_scores_replayed": replayed_samples,
        "population_summaries_replayed": summaries,
        "decision": decision,
        "errors": errors,
    }
    write_json(args.receipt, receipt)
    if errors:
        raise RuntimeError("; ".join(errors[:20]))
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
