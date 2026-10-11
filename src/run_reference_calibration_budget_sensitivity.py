#!/usr/bin/env python3
"""Checkpoint-only trusted-reference/calibration sensitivity for CF-CMD-U."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_group_coherent_10seeds as baseline
import run_bloodmnist_whitebox_adaptive_pilot as whitebox
import run_mnist_group_coherent_drift as coherent
import run_unified_oneclass_poison_screening as unified
import train_bloodmnist_clean_encoder as clean_encoder


PAPER14_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/reference_calibration_budget_sensitivity_protocol_v1.json"
)
SOURCE_ROOT = PAPER14_ROOT / "results/bloodmnist_covariance_free_confirmation_v1"
DEFAULT_OUTPUT = PAPER14_ROOT / "results/reference_calibration_budget_sensitivity_v1"
SOURCE_MANIFEST = SOURCE_ROOT / "manifest.json"
SOURCE_PROTOCOL = (
    PAPER14_ROOT / "results/top_journal_program_v1/covariance_free_confirmation_protocol.json"
)
SOURCE_RUNNER = PAPER14_ROOT / "src/run_bloodmnist_covariance_free_confirmation.py"
CROSSFIT_HELPER = PAPER14_ROOT / "src/run_bloodmnist_mechanism_ablations.py"
SPLIT_HELPER = PAPER14_ROOT / "src/run_bloodmnist_group_coherent_10seeds.py"
FEATURE_HELPER = PAPER14_ROOT / "src/run_bloodmnist_whitebox_adaptive_pilot.py"


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def load_model(path: Path) -> clean_encoder.CleanQNN:
    if not path.is_file():
        raise FileNotFoundError(f"Missing archived checkpoint; retraining is forbidden: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = clean_encoder.CleanQNN("local", 0)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def budget_crossfit(
    seed: int,
    trusted: np.ndarray,
    clean: np.ndarray,
    poison: np.ndarray,
    reference_count: int,
    calibration_count: int,
    alpha: float,
) -> tuple[dict, list[dict]]:
    """Original unwhitened cross-fit rule with frozen, nested trusted budgets."""

    trusted = np.asarray(trusted, dtype=np.float64)
    features = np.concatenate([clean, poison], axis=0).astype(np.float64)
    labels = np.concatenate(
        [np.zeros(len(clean), dtype=int), np.ones(len(poison), dtype=int)]
    )
    if reference_count > 120 or calibration_count > 80 or len(trusted) < 200:
        raise ValueError("Frozen pools require >=200 trusted rows, ref<=120, calibration<=80")

    rng = np.random.default_rng(20260928 + 1009 * seed)
    order = rng.permutation(len(trusted))
    reference = trusted[order[:reference_count]]
    calibration = trusted[order[120 : 120 + calibration_count]]
    if np.intersect1d(order[:reference_count], order[120 : 120 + calibration_count]).size:
        raise AssertionError("Reference and calibration pools overlap")

    mean = reference.mean(axis=0, keepdims=True)
    center = lambda values: np.asarray(values, dtype=np.float64) - mean
    reference_centered = center(reference)
    calibration_centered = center(calibration)
    scores = np.empty(len(features), dtype=np.float64)
    p_values = np.empty(len(features), dtype=np.float64)
    flags = np.zeros(len(features), dtype=bool)
    folds = np.array_split(rng.permutation(len(features)), 4)
    fallback_count = 0
    for query_indices in folds:
        halves = np.array_split(rng.permutation(query_indices), 2)
        for score_half, direction_half in (
            (halves[0], halves[1]),
            (halves[1], halves[0]),
        ):
            direction = (
                center(features[direction_half]).mean(axis=0)
                - reference_centered.mean(axis=0)
            )
            norm = float(np.linalg.norm(direction))
            if norm < 1e-12:
                direction = np.ones(features.shape[1], dtype=np.float64)
                norm = float(np.linalg.norm(direction))
                fallback_count += 1
            direction /= norm
            calibration_scores = calibration_centered @ direction
            query_scores = center(features[score_half]) @ direction
            query_p = coherent.one_sided_p_values(calibration_scores, query_scores)
            scores[score_half] = query_scores
            p_values[score_half] = query_p
            flags[score_half] = query_p <= alpha

    clean_flags, poison_flags = flags[labels == 0], flags[labels == 1]
    fp, tn = int(clean_flags.sum()), int((~clean_flags).sum())
    tp, fn = int(poison_flags.sum()), int((~poison_flags).sum())
    row = {
        "seed": seed,
        "reference_count": reference_count,
        "calibration_count": calibration_count,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tpr": tp / (tp + fn),
        "fpr": fp / (fp + tn),
        "auc_crossfit_score": float(roc_auc_score(labels, scores)),
        "auc_conformal_rank": float(roc_auc_score(labels, -p_values)),
        "fallback_direction_count": fallback_count,
        "rank_grid_null_rejection_probability": (
            int(np.floor(alpha * (calibration_count + 1)))
            / (calibration_count + 1)
        ),
    }
    samples = [
        {
            "seed": seed,
            "reference_count": reference_count,
            "calibration_count": calibration_count,
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
    return row, samples


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_frozen_inputs(protocol: dict, dataset: Path) -> dict:
    source = protocol["source_experiment"]
    checks = {
        "source_manifest_sha256": (SOURCE_MANIFEST, source["source_manifest_sha256"]),
        "source_protocol_sha256": (SOURCE_PROTOCOL, source["source_protocol_sha256"]),
        "source_runner_sha256": (SOURCE_RUNNER, source["source_runner_sha256"]),
        "crossfit_helper_sha256": (CROSSFIT_HELPER, source["crossfit_helper_sha256"]),
        "split_helper_sha256": (SPLIT_HELPER, source["split_helper_sha256"]),
        "feature_helper_sha256": (FEATURE_HELPER, source["feature_helper_sha256"]),
    }
    observed = {}
    for name, (path, expected) in checks.items():
        actual = file_hash(path)
        if actual != expected:
            raise RuntimeError(f"Frozen hash mismatch for {name}: {actual} != {expected}")
        observed[name] = actual
    if file_hash(dataset) != source["dataset_npz_sha256"]:
        raise RuntimeError("Dataset SHA-256 mismatch")
    if file_hash(dataset, "md5") != source["dataset_npz_md5"]:
        raise RuntimeError("Dataset MD5 mismatch")
    return observed


def compare_original_cell(rows: list[dict]) -> None:
    original = {
        int(row["seed"]): row
        for row in read_csv(SOURCE_ROOT / "per_config_metrics.csv")
        if row["scenario"] == "fixed_pca_additive"
    }
    replay = {
        int(row["seed"]): row
        for row in rows
        if row["reference_count"] == 120 and row["calibration_count"] == 80
    }
    if set(original) != set(replay):
        raise AssertionError("Original-cell seed set does not replay")
    integer_fields = ("tp", "fn", "fp", "tn")
    float_fields = ("tpr", "fpr", "auc_crossfit_score", "auc_conformal_rank")
    for seed in sorted(original):
        for field in integer_fields:
            if int(original[seed][field]) != int(replay[seed][field]):
                raise AssertionError(f"Original-cell {field} mismatch for seed {seed}")
        for field in float_fields:
            if not np.isclose(
                float(original[seed][field]), float(replay[seed][field]), atol=1e-12, rtol=0
            ):
                raise AssertionError(f"Original-cell {field} mismatch for seed {seed}")


def summarize(rows: list[dict], eligible_seeds: set[int], draws: int) -> list[dict]:
    output = []
    cells = sorted({(row["reference_count"], row["calibration_count"]) for row in rows})
    for reference_count, calibration_count in cells:
        selected = sorted(
            [
                row
                for row in rows
                if row["reference_count"] == reference_count
                and row["calibration_count"] == calibration_count
                and int(row["seed"]) in eligible_seeds
            ],
            key=lambda row: int(row["seed"]),
        )
        tp, fn = sum(row["tp"] for row in selected), sum(row["fn"] for row in selected)
        fp, tn = sum(row["fp"] for row in selected), sum(row["tn"] for row in selected)
        tpr = np.asarray([row["tpr"] for row in selected], dtype=float)
        fpr = np.asarray([row["fpr"] for row in selected], dtype=float)
        rng = np.random.default_rng(20261003 + 1000 * reference_count + calibration_count)
        indices = rng.integers(0, len(selected), size=(draws, len(selected)))
        output.append(
            {
                "reference_count": reference_count,
                "calibration_count": calibration_count,
                "eligible_seed_count": len(selected),
                "eligible_seeds": ",".join(str(row["seed"]) for row in selected),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "pooled_tpr": tp / (tp + fn),
                "pooled_fpr": fp / (fp + tn),
                "tpr_ci_low": float(np.quantile(tpr[indices].mean(axis=1), 0.025)),
                "tpr_ci_high": float(np.quantile(tpr[indices].mean(axis=1), 0.975)),
                "fpr_ci_low": float(np.quantile(fpr[indices].mean(axis=1), 0.025)),
                "fpr_ci_high": float(np.quantile(fpr[indices].mean(axis=1), 0.975)),
                "mean_auc_crossfit_score": float(
                    np.mean([row["auc_crossfit_score"] for row in selected])
                ),
                "mean_auc_conformal_rank": float(
                    np.mean([row["auc_conformal_rank"] for row in selected])
                ),
                "rank_grid_null_rejection_probability": selected[0][
                    "rank_grid_null_rejection_probability"
                ],
                "fallback_direction_count": sum(
                    row["fallback_direction_count"] for row in selected
                ),
            }
        )
    return output


def write_report(path: Path, summaries: list[dict], elapsed: float, peak: int) -> None:
    lines = [
        "# CF-CMD-U trusted-reference/calibration budget sensitivity",
        "",
        "Retrospective checkpoint-only replay on the frozen seeds 62--71. All 15 grid cells are reported; no best budget is selected.",
        "",
        "| reference | calibration | eligible seeds | pooled TPR | TPR 95% seed CI | pooled FPR | FPR 95% seed CI | score AUC | rank AUC | null rejection grid |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['reference_count']} | {row['calibration_count']} | {row['eligible_seed_count']} | "
            f"{row['pooled_tpr']:.4f} | [{row['tpr_ci_low']:.4f}, {row['tpr_ci_high']:.4f}] | "
            f"{row['pooled_fpr']:.4f} | [{row['fpr_ci_low']:.4f}, {row['fpr_ci_high']:.4f}] | "
            f"{row['mean_auc_crossfit_score']:.4f} | {row['mean_auc_conformal_rank']:.4f} | "
            f"{row['rank_grid_null_rejection_probability']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The 120/80 cell exactly replays the archived fixed-scenario per-seed confusion counts and AUCs. This table is a sensitivity analysis, not fresh confirmation, and does not replace the original primary endpoint.",
            "",
            f"Wall time: {elapsed:.3f} s. Peak process RSS: {peak / (1024 ** 2):.2f} MiB.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    seeds = tuple(protocol["source_experiment"]["seeds"])
    if tuple(args.seeds) != seeds:
        raise ValueError(f"Seeds are frozen to {seeds}")
    dataset = args.dataset_root / "bloodmnist.npz"
    input_hashes = verify_frozen_inputs(protocol, dataset)
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    source_receipts = {int(row["seed"]): row for row in source_manifest["receipts"]}
    victim_rows = [
        row
        for row in read_csv(SOURCE_ROOT / "victim_metrics.csv")
        if row["scenario"] == "fixed_pca_additive"
    ]
    eligible_seeds = {int(row["seed"]) for row in victim_rows if int(row["eligible"]) == 1}
    if len(eligible_seeds) != 7:
        raise AssertionError("Frozen eligible population must contain exactly seven seeds")

    class_data = heldout.collect_classes(args.dataset_root)
    rgb_trigger, _ = baseline.fixed_rgb_trigger(class_data)
    rows, samples, cache_receipts = [], [], []
    cache_payloads: list[tuple[int, dict[str, np.ndarray]]] = []
    for seed in seeds:
        clean_path = SOURCE_ROOT / "fixed_local/models" / f"seed_{seed}_clean.pt"
        suspect_path = SOURCE_ROOT / "fixed_local/models" / f"seed_{seed}_adaptive.pt"
        receipt = source_receipts[seed]
        if file_hash(clean_path) != receipt["clean_checkpoint_sha256"]:
            raise RuntimeError(f"Clean checkpoint hash mismatch for seed {seed}")
        if file_hash(suspect_path) != receipt["fixed_checkpoint_sha256"]:
            raise RuntimeError(f"Suspect checkpoint hash mismatch for seed {seed}")
        prepared = baseline.preprocess_seed(class_data, seed)
        splits, _ = baseline.seed_splits(prepared, rgb_trigger, seed)
        if whitebox.sha256_array(splits["archived_poison"]) != receipt["poison_sha256_float32"]:
            raise RuntimeError(f"Poison array hash mismatch for seed {seed}")
        clean_model, suspect_model = load_model(clean_path), load_model(suspect_path)
        arrays = {
            "trusted": whitebox.checkpoint_drift(
                clean_model, suspect_model, splits["val"]["target"]
            ),
            "clean": whitebox.checkpoint_drift(
                clean_model, suspect_model, splits["train"]["target"]
            ),
            "poison": whitebox.checkpoint_drift(
                clean_model, suspect_model, splits["archived_poison"]
            ),
        }
        cache_payloads.append((seed, arrays))
        cache_receipts.append(
            {
                "seed": seed,
                "clean_checkpoint_sha256": file_hash(clean_path),
                "suspect_checkpoint_sha256": file_hash(suspect_path),
                "trusted_sha256_float32": whitebox.sha256_array(arrays["trusted"]),
                "clean_sha256_float32": whitebox.sha256_array(arrays["clean"]),
                "poison_sha256_float32": whitebox.sha256_array(arrays["poison"]),
                "trusted_count": len(arrays["trusted"]),
                "clean_count": len(arrays["clean"]),
                "poison_count": len(arrays["poison"]),
            }
        )
        for reference_count in protocol["grid"]["reference_counts"]:
            for calibration_count in protocol["grid"]["calibration_counts"]:
                row, cell_samples = budget_crossfit(
                    seed,
                    arrays["trusted"],
                    arrays["clean"],
                    arrays["poison"],
                    int(reference_count),
                    int(calibration_count),
                    float(protocol["method"]["alpha"]),
                )
                rows.append(row)
                samples.extend(cell_samples)
        print(f"seed={seed}: extracted once, evaluated 15 frozen budget cells", flush=True)

    compare_original_cell(rows)
    summaries = summarize(
        rows, eligible_seeds, int(protocol["statistics"]["bootstrap_draws"])
    )
    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_files = []
    for seed, arrays in cache_payloads:
        path = cache_dir / f"seed_{seed}_xz16_drift.npz"
        np.savez_compressed(path, **arrays)
        cache_files.append(
            {"seed": seed, "path": str(path.relative_to(args.output)), "sha256": file_hash(path)}
        )
    unified.write_csv(args.output / "per_seed_grid_metrics.csv", rows)
    unified.write_csv(args.output / "per_sample_grid_scores.csv", samples)
    unified.write_csv(args.output / "eligible_grid_summary.csv", summaries)
    elapsed = time.perf_counter() - started
    peak = peak_rss_bytes()
    write_report(args.output / "EXPERIMENT_REPORT.md", summaries, elapsed, peak)
    runner_path = Path(__file__).resolve()
    output_hashes = {
        name: file_hash(args.output / name)
        for name in (
            "per_seed_grid_metrics.csv",
            "per_sample_grid_scores.csv",
            "eligible_grid_summary.csv",
            "EXPERIMENT_REPORT.md",
        )
    }
    unified.json_dump(
        args.output / "manifest.json",
        {
            "experiment": "reference_calibration_budget_sensitivity_v1",
            "status": "complete_checkpoint_only_retrospective_sensitivity",
            "protocol": str(PROTOCOL),
            "protocol_sha256": file_hash(PROTOCOL),
            "runner_sha256": file_hash(runner_path),
            "dataset_npz_sha256": file_hash(dataset),
            "dataset_npz_md5": file_hash(dataset, "md5"),
            "seeds": list(seeds),
            "eligible_seeds": sorted(eligible_seeds),
            "reference_counts": protocol["grid"]["reference_counts"],
            "calibration_counts": protocol["grid"]["calibration_counts"],
            "original_120_80_cell_exact_replay": True,
            "input_hashes": input_hashes,
            "cache_receipts": cache_receipts,
            "cache_files": cache_files,
            "output_hashes": output_hashes,
            "elapsed_seconds": elapsed,
            "peak_rss_bytes": peak,
            "platform": sys.platform,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(62, 72)))
    main(parser.parse_args())
