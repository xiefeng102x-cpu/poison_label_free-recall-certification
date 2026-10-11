#!/usr/bin/env python3
"""Retrospective single-query XZ16 reference audit on archived checkpoints.

Every query is scored only against a fixed trusted reference/calibration state;
no candidate label or other candidate query enters the score.  The analysis is
development-only and never trains or modifies a victim.
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
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import roc_auc_score

import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_group_coherent_10seeds as baseline
import run_bloodmnist_whitebox_adaptive_pilot as pilot
import train_bloodmnist_clean_encoder as clean_encoder
import verify_top_journal_phase3_adaptive as p3verify


PAPER14_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/single_sample_reference_audit_protocol_v1.json"
)
DEFAULT_OUTPUT = PAPER14_ROOT / "results/single_sample_reference_audit_v1"
EPSILON = 1e-12
SCORERS = ("ridge_mahalanobis", "rbf_reference_witness", "knn5_reference_distance")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_int_array(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


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


def rank_pvalues(calibration: np.ndarray, query: np.ndarray) -> np.ndarray:
    calibration = np.asarray(calibration, dtype=np.float64).reshape(-1)
    query = np.asarray(query, dtype=np.float64).reshape(-1)
    if len(calibration) == 0:
        raise ValueError("calibration scores cannot be empty")
    return (1.0 + np.sum(calibration[None, :] >= query[:, None], axis=1)) / (
        len(calibration) + 1.0
    )


def fit_reference_state(
    reference: np.ndarray, *, ridge_fraction: float, neighbors: int
) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 2 or len(reference) <= neighbors:
        raise ValueError("reference must be a sufficiently large 2-D matrix")
    mean = reference.mean(axis=0)
    covariance = np.cov(reference, rowvar=False, ddof=1)
    mean_variance = float(np.trace(covariance) / reference.shape[1])
    ridge = max(ridge_fraction * mean_variance, 1e-10)
    precision = np.linalg.inv(covariance + ridge * np.eye(reference.shape[1]))
    scale = reference.std(axis=0, ddof=0)
    scale = np.maximum(scale, 1e-8)
    reference_z = (reference - mean) / scale
    distances = pdist(reference_z, metric="sqeuclidean")
    positive = distances[distances > 0.0]
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


def score_queries(state: dict[str, Any], query: np.ndarray) -> dict[str, np.ndarray]:
    query = np.asarray(query, dtype=np.float64)
    centered = query - state["mean"]
    mahalanobis = np.einsum("bi,ij,bj->b", centered, state["precision"], centered)
    query_z = centered / state["scale"]
    squared = cdist(query_z, state["reference_z"], metric="sqeuclidean")
    rbf = 1.0 - np.exp(-state["gamma"] * squared).mean(axis=1)
    distances = np.sqrt(np.maximum(squared, 0.0))
    neighbor_index = int(state["neighbors"]) - 1
    knn = np.partition(distances, neighbor_index, axis=1)[:, neighbor_index]
    return {
        "ridge_mahalanobis": mahalanobis,
        "rbf_reference_witness": rbf,
        "knn5_reference_distance": knn,
    }


def calibrate_populations(
    reference: np.ndarray,
    calibration: np.ndarray,
    populations: dict[str, np.ndarray],
    protocol: dict[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    fixed = protocol["fixed_scorers"]
    state = fit_reference_state(
        reference,
        ridge_fraction=float(
            fixed["ridge_mahalanobis"]["ridge_fraction_of_mean_reference_variance"]
        ),
        neighbors=int(fixed["knn5_reference_distance"]["neighbors"]),
    )
    calibration_scores = score_queries(state, calibration)
    multiplier = len(SCORERS)
    outputs: dict[str, dict[str, np.ndarray]] = {}
    for name, values in populations.items():
        scores = score_queries(state, values)
        pvalues = {
            scorer: rank_pvalues(calibration_scores[scorer], scores[scorer])
            for scorer in SCORERS
        }
        fused = np.minimum(
            1.0,
            multiplier * np.min(np.stack([pvalues[name] for name in SCORERS]), axis=0),
        )
        outputs[name] = {
            **{f"score_{key}": value for key, value in scores.items()},
            **{f"p_{key}": value for key, value in pvalues.items()},
            "fused_p_value": fused,
        }
    receipt = {
        "reference_count": len(reference),
        "calibration_count": len(calibration),
        "ridge_value": float(state["ridge"]),
        "rbf_gamma": float(state["gamma"]),
        "reference_mean_sha256_float32": pilot.sha256_array(
            np.asarray(state["mean"], dtype=np.float32)
        ),
        "reference_scale_sha256_float32": pilot.sha256_array(
            np.asarray(state["scale"], dtype=np.float32)
        ),
    }
    return outputs, receipt


def auc(clean_score: np.ndarray, poison_score: np.ndarray) -> float:
    labels = np.concatenate([np.zeros(len(clean_score), dtype=int), np.ones(len(poison_score), dtype=int)])
    scores = np.concatenate([clean_score, poison_score])
    return float(roc_auc_score(labels, scores))


def victim_metrics(
    outputs: dict[str, dict[str, np.ndarray]], alpha: float
) -> dict[str, Any]:
    member = outputs["member_clean"]
    fresh = outputs["fresh_clean"]
    poison = outputs["poison"]
    member_flags = member["fused_p_value"] <= alpha
    fresh_flags = fresh["fused_p_value"] <= alpha
    poison_flags = poison["fused_p_value"] <= alpha
    metrics: dict[str, Any] = {
        "poison_tp": int(poison_flags.sum()),
        "poison_fn": int((~poison_flags).sum()),
        "member_clean_fp": int(member_flags.sum()),
        "member_clean_tn": int((~member_flags).sum()),
        "fresh_clean_fp": int(fresh_flags.sum()),
        "fresh_clean_tn": int((~fresh_flags).sum()),
        "poison_tpr": float(poison_flags.mean()),
        "member_clean_fpr": float(member_flags.mean()),
        "fresh_clean_fpr": float(fresh_flags.mean()),
        "poison_auc_against_member_clean": auc(
            -member["fused_p_value"], -poison["fused_p_value"]
        ),
        "poison_auc_against_fresh_clean": auc(
            -fresh["fused_p_value"], -poison["fused_p_value"]
        ),
        "median_poison_fused_p_value": float(np.median(poison["fused_p_value"])),
        "minimum_poison_fused_p_value": float(np.min(poison["fused_p_value"])),
    }
    for scorer in SCORERS:
        metrics[f"{scorer}_auc_member"] = auc(
            member[f"score_{scorer}"], poison[f"score_{scorer}"]
        )
        metrics[f"{scorer}_auc_fresh"] = auc(
            fresh[f"score_{scorer}"], poison[f"score_{scorer}"]
        )
    return metrics


def sample_rows(
    *, mode: str, seed: int, outputs: dict[str, dict[str, np.ndarray]], alpha: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for population in ("member_clean", "fresh_clean", "poison"):
        values = outputs[population]
        for index in range(len(values["fused_p_value"])):
            rows.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "population": population,
                    "sample_index": index,
                    "label": int(population == "poison"),
                    **{
                        field: float(values[field][index])
                        for field in values
                    },
                    "flagged": int(values["fused_p_value"][index] <= alpha),
                }
            )
    return rows


def bootstrap_mean_ci(
    values: np.ndarray, *, draws: int, seed: int, confidence: float
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[selected].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, (tail, 1.0 - tail))
    return float(values.mean()), float(low), float(high)


def aggregate_rows(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    config = protocol["source"]
    reported = protocol["reported_endpoints"]
    output: list[dict[str, Any]] = []
    for mode_index, mode in enumerate(config["modes"]):
        for population in ("stored_test_eligible_victims_per_attack_mode", "all_archived_victims"):
            selected = [
                row
                for row in rows
                if row["mode"] == mode
                and (
                    population == "all_archived_victims"
                    or row["test_eligible"] == 1
                )
            ]
            if not selected:
                continue
            poison_tp = sum(row["poison_tp"] for row in selected)
            poison_fn = sum(row["poison_fn"] for row in selected)
            member_fp = sum(row["member_clean_fp"] for row in selected)
            member_tn = sum(row["member_clean_tn"] for row in selected)
            fresh_fp = sum(row["fresh_clean_fp"] for row in selected)
            fresh_tn = sum(row["fresh_clean_tn"] for row in selected)
            aggregate: dict[str, Any] = {
                "mode": mode,
                "population": population,
                "victim_count": len(selected),
                "seeds": ";".join(str(row["seed"]) for row in selected),
                "poison_tp": poison_tp,
                "poison_fn": poison_fn,
                "member_clean_fp": member_fp,
                "member_clean_tn": member_tn,
                "fresh_clean_fp": fresh_fp,
                "fresh_clean_tn": fresh_tn,
                "pooled_poison_tpr": poison_tp / max(poison_tp + poison_fn, 1),
                "pooled_member_clean_fpr": member_fp / max(member_fp + member_tn, 1),
                "pooled_fresh_clean_fpr": fresh_fp / max(fresh_fp + fresh_tn, 1),
            }
            for metric_index, metric in enumerate(
                (
                    "poison_tpr",
                    "member_clean_fpr",
                    "fresh_clean_fpr",
                    "poison_auc_against_member_clean",
                    "poison_auc_against_fresh_clean",
                )
            ):
                mean, low, high = bootstrap_mean_ci(
                    np.asarray([row[metric] for row in selected]),
                    draws=int(reported["bootstrap_draws"]),
                    seed=int(reported["bootstrap_seed"])
                    + 100 * mode_index
                    + metric_index,
                    confidence=float(reported["confidence_level"]),
                )
                aggregate[f"mean_{metric}"] = mean
                aggregate[f"{metric}_ci_low"] = low
                aggregate[f"{metric}_ci_high"] = high
            for scorer in SCORERS:
                aggregate[f"mean_{scorer}_auc_member"] = float(
                    np.mean([row[f"{scorer}_auc_member"] for row in selected])
                )
                aggregate[f"mean_{scorer}_auc_fresh"] = float(
                    np.mean([row[f"{scorer}_auc_fresh"] for row in selected])
                )
            output.append(aggregate)
    return output


def decide(summary_rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    rules = protocol["decision_rule"]
    primary = {
        row["mode"]: row
        for row in summary_rows
        if row["population"] == "stored_test_eligible_victims_per_attack_mode"
    }

    def passes(mode: str, gate: dict[str, Any]) -> bool:
        row = primary[mode]
        return bool(
            row["victim_count"] >= int(rules["minimum_eligible_victims_per_mode"])
            and row["pooled_poison_tpr"] >= float(gate["pooled_poison_tpr_min"])
            and row["pooled_member_clean_fpr"] <= float(gate["pooled_member_clean_fpr_max"])
            and row["pooled_fresh_clean_fpr"] <= float(gate["pooled_fresh_clean_fpr_max"])
        )

    strong_modes = [mode for mode in protocol["source"]["modes"] if passes(mode, rules["strong_gate"])]
    partial_modes = [mode for mode in protocol["source"]["modes"] if passes(mode, rules["partial_gate"])]
    if len(strong_modes) >= int(rules["strong_gate"]["required_modes"]):
        label = rules["strong_gate"]["label"]
    elif len(partial_modes) >= int(rules["partial_gate"]["required_modes"]):
        label = rules["partial_gate"]["label"]
    else:
        label = rules["otherwise"]
    return {
        "label": label,
        "strong_gate_modes": strong_modes,
        "partial_gate_modes": partial_modes,
        "mode_endpoints": {
            mode: {
                "eligible_victims": primary[mode]["victim_count"],
                "pooled_poison_tpr": primary[mode]["pooled_poison_tpr"],
                "pooled_member_clean_fpr": primary[mode]["pooled_member_clean_fpr"],
                "pooled_fresh_clean_fpr": primary[mode]["pooled_fresh_clean_fpr"],
            }
            for mode in protocol["source"]["modes"]
        },
        "scope": "retrospective_development_only_single_query_reference_audit",
        "fresh_confirmation_authorized": bool(
            label == rules["strong_gate"]["label"]
        ),
    }


def report_text(summary: list[dict[str, Any]], decision: dict[str, Any]) -> str:
    primary = [
        row
        for row in summary
        if row["population"] == "stored_test_eligible_victims_per_attack_mode"
    ]
    lines = [
        "# Single-sample reference audit",
        "",
        "This is a retrospective development analysis. Each query is scored independently of all other candidate queries after a trusted reference/calibration state is fixed. It is not fresh confirmation and its clean-calibrated rank values are not unconditional exact conformal p-values for training members.",
        "",
        f"Frozen decision: **{decision['label']}**",
        f"Fresh confirmation authorized: **{'YES' if decision['fresh_confirmation_authorized'] else 'NO'}**",
        "",
        "| attack mode | eligible victims | pooled poison TPR | member-clean FPR | fresh-clean FPR | fused AUC vs member/fresh |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            f"| {row['mode']} | {row['victim_count']} | {row['pooled_poison_tpr']:.4f} | "
            f"{row['pooled_member_clean_fpr']:.4f} | {row['pooled_fresh_clean_fpr']:.4f} | "
            f"{row['mean_poison_auc_against_member_clean']:.4f}/{row['mean_poison_auc_against_fresh_clean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The scorer uses only the fixed trusted state and the current query. The CA/ASR eligibility filter is inherited from the archived test results and is used only for evaluation summaries. A positive retrospective result would require a separately frozen fresh-seed confirmation before entering an efficacy claim.",
            "",
        ]
    )
    return "\n".join(lines)


def verify_sources(protocol: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    source = PAPER14_ROOT / protocol["source"]["root"]
    expected = protocol["source"]
    if sha256_file(source / "manifest.json") != expected["manifest_sha256"]:
        raise RuntimeError("source manifest hash mismatch")
    if sha256_file(source / "victim_metrics.csv") != expected["victim_metrics_sha256"]:
        raise RuntimeError("source victim metric hash mismatch")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    dataset = Path(manifest["dataset_root"]) / "bloodmnist.npz"
    if sha256_file(dataset) != expected["dataset_sha256"]:
        raise RuntimeError("dataset hash mismatch")
    return source, manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    source, source_manifest = verify_sources(protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    victim_source = {
        (row["mode"], int(row["seed"])): row
        for row in read_csv(source / "victim_metrics.csv")
    }
    receipts = {
        (row["mode"], int(row["seed"])): row
        for row in source_manifest["receipts"]
    }
    class_data = heldout.collect_classes(Path(source_manifest["dataset_root"]))
    rgb_trigger, _ = baseline.fixed_rgb_trigger(class_data)
    per_victim: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    feature_receipts: list[dict[str, Any]] = []
    alpha = float(protocol["calibration_and_fusion"]["alpha"])

    for seed in protocol["source"]["seeds"]:
        print(f"seed {seed}: reconstructing fixed single-query channel", flush=True)
        seed_data = pilot.prepare_seed(
            class_data,
            rgb_trigger,
            seed,
            clean_root=Path(source_manifest["clean_qml_root"]),
        )
        trusted_input = seed_data["splits"]["val"]["target"]
        member_input = seed_data["splits"]["train"]["target"]
        fresh_input = seed_data["splits"]["test"]["target"]
        rng = np.random.default_rng(20261003 + seed)
        order = rng.permutation(len(trusted_input))
        split = int(np.floor(0.5 * len(order)))
        reference_indices, calibration_indices = order[:split], order[split:]
        for mode in protocol["source"]["modes"]:
            key = (mode, seed)
            victim = victim_source[key]
            receipt = receipts[key]
            checkpoint_path = source / victim["checkpoint"]
            checkpoint_hash = sha256_file(checkpoint_path)
            if checkpoint_hash != receipt["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint hash mismatch: {mode} {seed}")
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
                raise RuntimeError(f"poison hash mismatch: {mode} {seed}")
            trusted = pilot.checkpoint_drift(
                seed_data["clean_model"], attacked, trusted_input
            )
            member = pilot.checkpoint_drift(
                seed_data["clean_model"], attacked, member_input
            )
            fresh = pilot.checkpoint_drift(
                seed_data["clean_model"], attacked, fresh_input
            )
            poison = pilot.checkpoint_drift(
                seed_data["clean_model"], attacked, poison_input
            )
            if pilot.sha256_array(poison) != receipt["xz_poison_sha256_float32"]:
                raise RuntimeError(f"XZ poison hash mismatch: {mode} {seed}")
            outputs, state_receipt = calibrate_populations(
                trusted[reference_indices],
                trusted[calibration_indices],
                {"member_clean": member, "fresh_clean": fresh, "poison": poison},
                protocol,
            )
            metrics = victim_metrics(outputs, alpha)
            per_victim.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "test_eligible": int(victim["test_eligible"]),
                    "test_balanced_ca": float(victim["test_balanced_ca"]),
                    "test_asr": float(victim["test_asr"]),
                    "reference_count": len(reference_indices),
                    "calibration_count": len(calibration_indices),
                    "member_clean_count": len(member),
                    "fresh_clean_count": len(fresh),
                    "poison_count": len(poison),
                    **metrics,
                }
            )
            per_sample.extend(
                sample_rows(mode=mode, seed=seed, outputs=outputs, alpha=alpha)
            )
            feature_receipts.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "checkpoint": str(checkpoint_path.relative_to(PAPER14_ROOT)),
                    "checkpoint_sha256": checkpoint_hash,
                    "reference_indices_sha256_int64": sha256_int_array(reference_indices),
                    "calibration_indices_sha256_int64": sha256_int_array(calibration_indices),
                    "trusted_xz_sha256_float32": pilot.sha256_array(trusted),
                    "member_clean_xz_sha256_float32": pilot.sha256_array(member),
                    "fresh_clean_xz_sha256_float32": pilot.sha256_array(fresh),
                    "poison_xz_sha256_float32": pilot.sha256_array(poison),
                    **state_receipt,
                }
            )

    summary = aggregate_rows(per_victim, protocol)
    decision = decide(summary, protocol)
    outputs = {
        "per_victim_metrics": args.output / "per_victim_metrics.csv",
        "per_sample_scores": args.output / "per_sample_scores.csv",
        "population_summary": args.output / "population_summary.csv",
        "feature_receipts": args.output / "feature_receipts.json",
        "decision": args.output / "decision.json",
        "report": args.output / "EXPERIMENT_REPORT.md",
    }
    write_csv(outputs["per_victim_metrics"], per_victim)
    write_csv(outputs["per_sample_scores"], per_sample)
    write_csv(outputs["population_summary"], summary)
    write_json(outputs["feature_receipts"], feature_receipts)
    write_json(outputs["decision"], decision)
    outputs["report"].write_text(report_text(summary, decision), encoding="utf-8")
    manifest = {
        "experiment": "single_sample_reference_audit_v1",
        "role": protocol["role"],
        "protocol": str(PROTOCOL.relative_to(PAPER14_ROOT)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner": str(Path(__file__).relative_to(PAPER14_ROOT)),
        "runner_sha256": sha256_file(Path(__file__)),
        "source": str(source.relative_to(PAPER14_ROOT)),
        "source_manifest_sha256": sha256_file(source / "manifest.json"),
        "victim_count": len(per_victim),
        "sample_score_rows": len(per_sample),
        "decision": decision,
        "output_hashes": {
            name: {
                "path": str(path.relative_to(PAPER14_ROOT)),
                "sha256": sha256_file(path),
            }
            for name, path in outputs.items()
        },
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
