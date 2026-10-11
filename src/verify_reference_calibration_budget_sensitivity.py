#!/usr/bin/env python3
"""Independent replay verifier for the frozen CF-CMD-U budget grid."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/top_journal_program_v1/reference_calibration_budget_sensitivity_protocol_v1.json"
RUNNER = ROOT / "src/run_reference_calibration_budget_sensitivity.py"
RESULTS = ROOT / "results/reference_calibration_budget_sensitivity_v1"
SOURCE = ROOT / "results/bloodmnist_covariance_free_confirmation_v1"
DATASET = Path("/private/tmp/blood-ca-data/bloodmnist.npz")
RECEIPT = ROOT / "results/top_journal_program_v1/reference_calibration_budget_sensitivity_verification.json"


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: float | str, right: float | str) -> bool:
    return bool(np.isclose(float(left), float(right), atol=1e-12, rtol=0.0))


def rank_pvalues(calibration: np.ndarray, query: np.ndarray) -> np.ndarray:
    return (
        1.0
        + np.sum(
            calibration.reshape(1, -1) >= query.reshape(-1, 1),
            axis=1,
        )
    ) / (len(calibration) + 1.0)


def replay_cell(
    seed: int,
    trusted: np.ndarray,
    clean: np.ndarray,
    poison: np.ndarray,
    reference_count: int,
    calibration_count: int,
) -> tuple[dict, list[dict]]:
    trusted = np.asarray(trusted, dtype=np.float64)
    features = np.concatenate([clean, poison], axis=0).astype(np.float64)
    labels = np.concatenate(
        [np.zeros(len(clean), dtype=int), np.ones(len(poison), dtype=int)]
    )
    rng = np.random.default_rng(20260928 + 1009 * seed)
    order = rng.permutation(len(trusted))
    reference_indices = order[:reference_count]
    calibration_indices = order[120 : 120 + calibration_count]
    if np.intersect1d(reference_indices, calibration_indices).size:
        raise AssertionError("Trusted pools overlap")
    reference = trusted[reference_indices]
    calibration = trusted[calibration_indices]
    mean = reference.mean(axis=0, keepdims=True)
    centered_reference = reference - mean
    centered_calibration = calibration - mean
    scores = np.empty(len(features), dtype=np.float64)
    pvalues = np.empty(len(features), dtype=np.float64)
    flags = np.zeros(len(features), dtype=bool)
    fallback_count = 0
    for query_indices in np.array_split(rng.permutation(len(features)), 4):
        halves = np.array_split(rng.permutation(query_indices), 2)
        for score_indices, direction_indices in (
            (halves[0], halves[1]),
            (halves[1], halves[0]),
        ):
            direction = (
                (features[direction_indices] - mean).mean(axis=0)
                - centered_reference.mean(axis=0)
            )
            norm = float(np.linalg.norm(direction))
            if norm < 1e-12:
                direction = np.ones(features.shape[1], dtype=np.float64)
                norm = float(np.linalg.norm(direction))
                fallback_count += 1
            direction /= norm
            calibration_scores = centered_calibration @ direction
            query_scores = (features[score_indices] - mean) @ direction
            query_pvalues = rank_pvalues(calibration_scores, query_scores)
            scores[score_indices] = query_scores
            pvalues[score_indices] = query_pvalues
            flags[score_indices] = query_pvalues <= 0.05
    clean_flags, poison_flags = flags[labels == 0], flags[labels == 1]
    fp, tn = int(clean_flags.sum()), int((~clean_flags).sum())
    tp, fn = int(poison_flags.sum()), int((~poison_flags).sum())
    metrics = {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tpr": tp / (tp + fn),
        "fpr": fp / (fp + tn),
        "auc_crossfit_score": float(roc_auc_score(labels, scores)),
        "auc_conformal_rank": float(roc_auc_score(labels, -pvalues)),
        "fallback_direction_count": fallback_count,
        "rank_grid_null_rejection_probability": int(
            np.floor(0.05 * (calibration_count + 1))
        )
        / (calibration_count + 1),
    }
    samples = [
        {
            "sample_index": index,
            "label": int(labels[index]),
            "score": float(scores[index]),
            "p_value": float(pvalues[index]),
            "flagged": int(flags[index]),
        }
        for index in range(len(features))
    ]
    return metrics, samples


def verify_outputs() -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete_checkpoint_only_retrospective_sensitivity":
        raise AssertionError("Manifest status mismatch")
    if manifest["protocol_sha256"] != file_hash(PROTOCOL):
        raise AssertionError("Protocol hash mismatch")
    if manifest["runner_sha256"] != file_hash(RUNNER):
        raise AssertionError("Runner hash mismatch")
    if manifest["dataset_npz_sha256"] != file_hash(DATASET):
        raise AssertionError("Dataset SHA-256 mismatch")
    if manifest["dataset_npz_md5"] != file_hash(DATASET, "md5"):
        raise AssertionError("Dataset MD5 mismatch")
    for name, expected in manifest["output_hashes"].items():
        if file_hash(RESULTS / name) != expected:
            raise AssertionError(f"Output hash mismatch: {name}")
    for row in manifest["cache_files"]:
        if file_hash(RESULTS / row["path"]) != row["sha256"]:
            raise AssertionError(f"Cache hash mismatch: {row['path']}")

    metric_rows = read_csv(RESULTS / "per_seed_grid_metrics.csv")
    sample_rows = read_csv(RESULTS / "per_sample_grid_scores.csv")
    summary_rows = read_csv(RESULTS / "eligible_grid_summary.csv")
    expected_cells = {
        (int(seed), int(reference), int(calibration))
        for seed in protocol["source_experiment"]["seeds"]
        for reference in protocol["grid"]["reference_counts"]
        for calibration in protocol["grid"]["calibration_counts"]
    }
    reported_metrics = {
        (int(row["seed"]), int(row["reference_count"]), int(row["calibration_count"])): row
        for row in metric_rows
    }
    if set(reported_metrics) != expected_cells or len(metric_rows) != 150:
        raise AssertionError("Metric grid mismatch")
    grouped_samples: dict[tuple[int, int, int], dict[int, dict[str, str]]] = {}
    for row in sample_rows:
        key = (int(row["seed"]), int(row["reference_count"]), int(row["calibration_count"]))
        index = int(row["sample_index"])
        if index in grouped_samples.setdefault(key, {}):
            raise AssertionError(f"Duplicate sample row: {key}, {index}")
        grouped_samples[key][index] = row
    if set(grouped_samples) != expected_cells or len(sample_rows) != 99000:
        raise AssertionError("Per-sample grid mismatch")

    replayed: dict[tuple[int, int, int], dict] = {}
    cache_receipts = {int(row["seed"]): row for row in manifest["cache_receipts"]}
    for seed in protocol["source_experiment"]["seeds"]:
        cache_path = RESULTS / "feature_cache" / f"seed_{seed}_xz16_drift.npz"
        with np.load(cache_path) as payload:
            arrays = {name: payload[name] for name in ("trusted", "clean", "poison")}
        receipt = cache_receipts[int(seed)]
        for name in arrays:
            canonical = np.ascontiguousarray(arrays[name].astype("<f4", copy=False))
            if hashlib.sha256(canonical.tobytes()).hexdigest() != receipt[f"{name}_sha256_float32"]:
                raise AssertionError(f"Cache array hash mismatch: seed {seed} {name}")
        for reference_count in protocol["grid"]["reference_counts"]:
            for calibration_count in protocol["grid"]["calibration_counts"]:
                key = (int(seed), int(reference_count), int(calibration_count))
                expected_metrics, expected_samples = replay_cell(
                    int(seed),
                    arrays["trusted"],
                    arrays["clean"],
                    arrays["poison"],
                    int(reference_count),
                    int(calibration_count),
                )
                actual_metrics = reported_metrics[key]
                for field in ("tp", "fn", "fp", "tn", "fallback_direction_count"):
                    if int(actual_metrics[field]) != int(expected_metrics[field]):
                        raise AssertionError(f"Metric mismatch: {key} {field}")
                for field in (
                    "tpr",
                    "fpr",
                    "auc_crossfit_score",
                    "auc_conformal_rank",
                    "rank_grid_null_rejection_probability",
                ):
                    if not close(actual_metrics[field], expected_metrics[field]):
                        raise AssertionError(f"Metric mismatch: {key} {field}")
                actual_samples = grouped_samples[key]
                if set(actual_samples) != set(range(len(expected_samples))):
                    raise AssertionError(f"Sample index mismatch: {key}")
                for expected in expected_samples:
                    actual = actual_samples[expected["sample_index"]]
                    if not (
                        int(actual["label"]) == expected["label"]
                        and int(actual["flagged"]) == expected["flagged"]
                        and close(actual["score"], expected["score"])
                        and close(actual["p_value"], expected["p_value"])
                    ):
                        raise AssertionError(
                            f"Sample replay mismatch: {key} index={expected['sample_index']}"
                        )
                replayed[key] = expected_metrics

    victim_rows = [
        row
        for row in read_csv(SOURCE / "victim_metrics.csv")
        if row["scenario"] == "fixed_pca_additive"
    ]
    eligible = {int(row["seed"]) for row in victim_rows if int(row["eligible"]) == 1}
    if eligible != set(manifest["eligible_seeds"]):
        raise AssertionError("Eligible population mismatch")
    reported_summaries = {
        (int(row["reference_count"]), int(row["calibration_count"])): row
        for row in summary_rows
    }
    for reference_count in protocol["grid"]["reference_counts"]:
        for calibration_count in protocol["grid"]["calibration_counts"]:
            cell = [
                replayed[(seed, reference_count, calibration_count)]
                for seed in sorted(eligible)
            ]
            tp, fn = sum(row["tp"] for row in cell), sum(row["fn"] for row in cell)
            fp, tn = sum(row["fp"] for row in cell), sum(row["tn"] for row in cell)
            tpr = np.asarray([row["tpr"] for row in cell])
            fpr = np.asarray([row["fpr"] for row in cell])
            rng = np.random.default_rng(20261003 + 1000 * reference_count + calibration_count)
            indices = rng.integers(0, len(cell), size=(10000, len(cell)))
            expected = {
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "pooled_tpr": tp / (tp + fn),
                "pooled_fpr": fp / (fp + tn),
                "tpr_ci_low": float(np.quantile(tpr[indices].mean(1), 0.025)),
                "tpr_ci_high": float(np.quantile(tpr[indices].mean(1), 0.975)),
                "fpr_ci_low": float(np.quantile(fpr[indices].mean(1), 0.025)),
                "fpr_ci_high": float(np.quantile(fpr[indices].mean(1), 0.975)),
                "mean_auc_crossfit_score": float(
                    np.mean([row["auc_crossfit_score"] for row in cell])
                ),
                "mean_auc_conformal_rank": float(
                    np.mean([row["auc_conformal_rank"] for row in cell])
                ),
            }
            actual = reported_summaries[(reference_count, calibration_count)]
            for field in ("tp", "fn", "fp", "tn"):
                if int(actual[field]) != expected[field]:
                    raise AssertionError(f"Summary mismatch: {reference_count}/{calibration_count} {field}")
            for field in expected.keys() - {"tp", "fn", "fp", "tn"}:
                if not close(actual[field], expected[field]):
                    raise AssertionError(f"Summary mismatch: {reference_count}/{calibration_count} {field}")

    original = {
        int(row["seed"]): row
        for row in read_csv(SOURCE / "per_config_metrics.csv")
        if row["scenario"] == "fixed_pca_additive"
    }
    for seed, row in original.items():
        replay = replayed[(seed, 120, 80)]
        for field in ("tp", "fn", "fp", "tn"):
            if int(row[field]) != replay[field]:
                raise AssertionError(f"Original cell count mismatch: seed {seed} {field}")
        for field in ("tpr", "fpr", "auc_crossfit_score", "auc_conformal_rank"):
            if not close(row[field], replay[field]):
                raise AssertionError(f"Original cell float mismatch: seed {seed} {field}")

    return {
        "verification": "PASS",
        "protocol_sha256": file_hash(PROTOCOL),
        "runner_sha256": file_hash(RUNNER),
        "verifier_sha256": file_hash(Path(__file__)),
        "dataset_npz_sha256": file_hash(DATASET),
        "seeds": protocol["source_experiment"]["seeds"],
        "eligible_seeds": sorted(eligible),
        "grid_cells": len(protocol["grid"]["reference_counts"]) * len(protocol["grid"]["calibration_counts"]),
        "seed_grid_instances": len(expected_cells),
        "metric_rows": len(metric_rows),
        "sample_rows": len(sample_rows),
        "checks": [
            "protocol, runner, dataset, output and feature-cache hashes",
            "independent trusted permutation, pool disjointness, cross-fit directions and rank p-values",
            "all 99,000 scores, p-values, labels and flags",
            "all 150 per-seed grid metrics and 15 eligible-population summaries",
            "10,000-draw seed-block bootstrap intervals",
            "exact archived 120/80 fixed-scenario replay",
        ],
        "summary": [
            {
                key: (int(value) if key in {"reference_count", "calibration_count", "eligible_seed_count", "tp", "fn", "fp", "tn", "fallback_direction_count"} else float(value) if key not in {"eligible_seeds"} else value)
                for key, value in row.items()
            }
            for row in summary_rows
        ],
        "artifact_sha256": {
            name: file_hash(RESULTS / name)
            for name in (
                "manifest.json",
                "per_seed_grid_metrics.csv",
                "per_sample_grid_scores.csv",
                "eligible_grid_summary.csv",
                "EXPERIMENT_REPORT.md",
            )
        },
    }


def main() -> None:
    receipt = verify_outputs()
    RECEIPT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"PASS: {receipt['grid_cells']} budget cells / {receipt['seed_grid_instances']} seed-grid instances, "
        f"{receipt['metric_rows']} metric rows, "
        f"{receipt['sample_rows']} sample rows"
    )
    print(f"wrote {RECEIPT}")


if __name__ == "__main__":
    main()
