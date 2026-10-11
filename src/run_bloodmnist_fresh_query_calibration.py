#!/usr/bin/env python3
"""Frozen member-versus-fresh rank-calibration experiment on archived victims."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import ks_2samp

SOURCE_DIRECTORY = Path(__file__).resolve().parent
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from analyze_checkpoint_auditability_theory_probe import rank_pvalues
from run_bloodmnist_multiclass_backdoorbench_pilot import (
    N_CLASSES,
    POISON_COUNT,
    TARGET_CLASS,
    MulticlassBloodCNN,
    balanced_accuracy,
    build_poisoned_training,
    channel_stats,
    hidden_features,
    load_bloodmnist,
    per_class_recall,
    predict,
    sha256_file,
    state_fingerprint,
)


PAPER14_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = PAPER14_ROOT / "results/top_journal_program_v1/fresh_query_calibration_protocol_v1.json"
SOURCE_PROTOCOL = PAPER14_ROOT / "results/top_journal_program_v1/multiclass_backdoorbench_pilot_protocol.json"
SOURCE_RUNNER = PAPER14_ROOT / "src/run_bloodmnist_multiclass_backdoorbench_pilot.py"
SOURCE_DEVELOPMENT = PAPER14_ROOT / "results/bloodmnist_multiclass_backdoorbench_pilot_v1"
SOURCE_CONFIRMATION = PAPER14_ROOT / "results/bloodmnist_multiclass_backdoorbench_confirmation_v1"
DEFAULT_DEVELOPMENT = PAPER14_ROOT / "results/bloodmnist_fresh_query_calibration_v1"
DEFAULT_CONFIRMATION = PAPER14_ROOT / "results/bloodmnist_fresh_query_calibration_confirmation_v1"
DEVELOPMENT_SEEDS = (72, 73)
CONFIRMATION_SEEDS = (74, 75, 76, 77, 78)
VIEWS = ("suspect_hidden64", "paired_hidden64_drift")
REFERENCE_COUNT = 96
CALIBRATION_COUNT = 96
ALPHA = 0.05
EPS = 1e-12


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


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


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def source_root(stage: str) -> Path:
    return SOURCE_DEVELOPMENT if stage == "development" else SOURCE_CONFIRMATION


def source_receipt(stage: str, seed: int) -> dict[str, Any]:
    rows = read_json(source_root(stage) / "seed_receipts.json")
    matches = [row for row in rows if int(row["seed"]) == seed]
    if len(matches) != 1:
        raise RuntimeError(f"missing archived receipt for seed {seed}")
    return matches[0]


def load_model(
    stage: str,
    seed: int,
    role: str,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[MulticlassBloodCNN, dict[str, str]]:
    receipt = source_receipt(stage, seed)["checkpoints"][role]
    path = source_root(stage) / "models" / f"seed_{seed}_{role}.pt"
    if sha256_file(path) != receipt["file_sha256"]:
        raise RuntimeError(f"{role} checkpoint file mismatch for seed {seed}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not (
        payload.get("seed") == seed
        and payload.get("role") == role
        and payload.get("protocol_sha256") == sha256_file(SOURCE_PROTOCOL)
        and np.array_equal(payload["mean"], mean)
        and np.array_equal(payload["std"], std)
    ):
        raise RuntimeError(f"{role} checkpoint metadata mismatch for seed {seed}")
    model = MulticlassBloodCNN(mean, std)
    model.load_state_dict(payload["state_dict"])
    if state_fingerprint(model.state_dict()) != receipt["state_sha256"]:
        raise RuntimeError(f"{role} checkpoint state mismatch for seed {seed}")
    return model, {
        "path": str(path),
        "file_sha256": receipt["file_sha256"],
        "state_sha256": receipt["state_sha256"],
    }


def alternating_folds(indices: np.ndarray, rng_seed: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(indices, dtype=np.int64)
    permutation = np.random.default_rng(rng_seed).permutation(values)
    return np.sort(permutation[::2]), np.sort(permutation[1::2])


def unit_direction(values: np.ndarray, reference_mean: np.ndarray) -> tuple[np.ndarray, float]:
    raw = np.asarray(values, dtype=np.float64).mean(axis=0) - np.asarray(reference_mean, dtype=np.float64)
    norm = float(np.linalg.norm(raw))
    if not np.isfinite(norm) or norm <= EPS:
        raise RuntimeError("direction is zero or non-finite")
    return raw / norm, norm


def score(values: np.ndarray, reference_mean: np.ndarray, direction: np.ndarray) -> np.ndarray:
    result = (np.asarray(values, dtype=np.float64) - reference_mean) @ direction
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite calibration score")
    return result


def split_design(
    pool: dict[str, Any], validation_labels: np.ndarray, test_labels: np.ndarray, seed: int
) -> dict[str, Any]:
    val_target = np.flatnonzero(validation_labels == TARGET_CLASS)
    val_perm = np.random.default_rng(20280929 + seed).permutation(val_target)
    if len(val_perm) < REFERENCE_COUNT + CALIBRATION_COUNT:
        raise RuntimeError("insufficient target-class validation rows")
    reference = np.sort(val_perm[:REFERENCE_COUNT])
    calibration = np.sort(val_perm[REFERENCE_COUNT : REFERENCE_COUNT + CALIBRATION_COUNT])
    target_candidates = np.flatnonzero(pool["assigned_labels"] == TARGET_CLASS)
    candidate_folds = alternating_folds(target_candidates, 20290929 + seed)
    member_clean = np.flatnonzero(
        (~pool["is_poison"]) & (pool["original_labels"] == TARGET_CLASS)
    )
    member_folds = tuple(np.intersect1d(member_clean, fold, assume_unique=True) for fold in candidate_folds)
    fresh_test = np.flatnonzero(test_labels == TARGET_CLASS)
    fresh_folds = alternating_folds(fresh_test, 20300929 + seed)
    if not (
        len(reference) == REFERENCE_COUNT
        and len(calibration) == CALIBRATION_COUNT
        and len(member_clean) == 2330
        and len(fresh_test) == 666
        and sum(len(values) for values in member_folds) == len(member_clean)
        and sum(len(values) for values in fresh_folds) == len(fresh_test)
        and np.intersect1d(reference, calibration).size == 0
    ):
        raise RuntimeError("frozen calibration/query split invariant failed")
    return {
        "reference": reference,
        "calibration": calibration,
        "candidate_folds": candidate_folds,
        "member_folds": member_folds,
        "fresh_folds": fresh_folds,
    }


def victim_metrics(model: MulticlassBloodCNN, data: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    clean_prediction = predict(model, data["test"]["images"])
    clean_balanced = balanced_accuracy(data["test"]["labels"], clean_prediction)
    non_target = data["test"]["labels"] != TARGET_CLASS
    triggered = data["test"]["images"][non_target].copy()
    triggered[:, -3:, -3:, :] = 1.0
    attack_prediction = predict(model, triggered)
    asr = float(np.mean(attack_prediction == TARGET_CLASS))
    return {
        "balanced_clean_accuracy": clean_balanced,
        "attack_success_rate": asr,
        "eligible": int(clean_balanced >= 0.80 and asr >= 0.80),
        "per_class_recall_json": json.dumps(
            per_class_recall(data["test"]["labels"], clean_prediction),
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def confirmation_is_authorized(root: Path) -> bool:
    path = root / "manifest.json"
    if not path.exists():
        return False
    manifest = read_json(path)
    gate = manifest.get("gate", {})
    required = (
        "both_archived_victims_eligible",
        "all_dataset_protocol_and_checkpoint_hashes_match",
        "reference_calibration_and_query_indices_disjoint",
        "both_views_finite_nonzero",
        "all_query_flags_and_counts_replay",
        "all_seeds_within_runtime_budget",
        "expand_to_confirmation",
    )
    return bool(
        manifest.get("experiment") == "bloodmnist_fresh_query_calibration_v1"
        and manifest.get("stage") == "development"
        and manifest.get("protocol_sha256") == sha256_file(PROTOCOL)
        and manifest.get("seeds") == list(DEVELOPMENT_SEEDS)
        and all(gate.get(name) is True for name in required)
    )


def replay_counts(query_rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]]) -> bool:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for row in query_rows:
        key = (int(row["seed"]), str(row["view"]), str(row["population"]))
        grouped.setdefault(key, []).append(row)
    metrics = {(int(row["seed"]), str(row["view"])): row for row in metric_rows}
    for (seed, view), metric in metrics.items():
        for population, prefix in (("member_train_clean", "member"), ("fresh_test_clean", "fresh")):
            rows = grouped.get((seed, view, population), [])
            flagged = sum(int(row["flagged"]) for row in rows)
            if not (
                len(rows) == int(metric[f"{prefix}_count"])
                and flagged == int(metric[f"{prefix}_flagged"])
                and np.isclose(flagged / len(rows), float(metric[f"{prefix}_fpr"]), atol=1e-12, rtol=0.0)
            ):
                return False
    return len(grouped) == 2 * len(metrics)


def run(args: argparse.Namespace) -> dict[str, Any]:
    expected_seeds = DEVELOPMENT_SEEDS if args.stage == "development" else CONFIRMATION_SEEDS
    seeds = tuple(int(value) for value in args.seeds)
    if seeds != expected_seeds:
        raise ValueError(f"{args.stage} is frozen to seeds {expected_seeds}")
    if args.stage == "confirmation" and not confirmation_is_authorized(args.development_root):
        raise RuntimeError("development non-performance gate has not passed")
    protocol = read_json(PROTOCOL)
    dataset_path = args.dataset_root / "bloodmnist.npz"
    expected_hashes = {
        dataset_path: protocol["dataset"]["sha256"],
        SOURCE_PROTOCOL: protocol["victims"]["source_protocol_sha256"],
        SOURCE_RUNNER: protocol["victims"]["source_runner_sha256"],
    }
    if any(not path.exists() or sha256_file(path) != digest for path, digest in expected_hashes.items()):
        raise RuntimeError("frozen input hash mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_bloodmnist(args.dataset_root)
    mean, std = channel_stats(data["train"]["images"])
    query_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    victim_rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    started_all = time.time()

    for seed in seeds:
        started = time.time()
        print(f"seed={seed}: reconstruct pool and load archived checkpoints", flush=True)
        pool = build_poisoned_training(data["train"]["images"], data["train"]["labels"], seed=seed)
        archived = source_receipt(args.stage, seed)
        if pool["receipt"]["selected_indices_sha256"] != archived["pool"]["selected_indices_sha256"]:
            raise RuntimeError(f"poison-pool mismatch for seed {seed}")
        clean_model, clean_receipt = load_model(args.stage, seed, "clean", mean, std)
        suspect_model, suspect_receipt = load_model(args.stage, seed, "suspect", mean, std)
        victim = {"stage": args.stage, "seed": seed, **victim_metrics(suspect_model, data)}
        victim_rows.append(victim)
        design = split_design(pool, data["val"]["labels"], data["test"]["labels"], seed)

        suspect_features = {
            "train": hidden_features(suspect_model, pool["images"]),
            "val": hidden_features(suspect_model, data["val"]["images"]),
            "test": hidden_features(suspect_model, data["test"]["images"]),
        }
        clean_features = {
            "train": hidden_features(clean_model, pool["images"]),
            "val": hidden_features(clean_model, data["val"]["images"]),
            "test": hidden_features(clean_model, data["test"]["images"]),
        }
        views = {
            "suspect_hidden64": suspect_features,
            "paired_hidden64_drift": {
                split: suspect_features[split] - clean_features[split]
                for split in ("train", "val", "test")
            },
        }
        view_receipts: dict[str, Any] = {}
        for view, features in views.items():
            reference_mean = features["val"][design["reference"]].mean(axis=0)
            population_scores: dict[str, list[np.ndarray]] = {
                "member_train_clean": [],
                "fresh_test_clean": [],
            }
            population_pvalues: dict[str, list[np.ndarray]] = {
                "member_train_clean": [],
                "fresh_test_clean": [],
            }
            calibration_by_fold: list[np.ndarray] = []
            direction_norms: list[float] = []
            directions: list[np.ndarray] = []
            for fold in (0, 1):
                direction, direction_norm = unit_direction(
                    features["train"][design["candidate_folds"][1 - fold]],
                    reference_mean,
                )
                directions.append(direction)
                direction_norms.append(direction_norm)
                calibration_scores = score(
                    features["val"][design["calibration"]], reference_mean, direction
                )
                calibration_by_fold.append(calibration_scores)
                for population, split, indices in (
                    ("member_train_clean", "train", design["member_folds"][fold]),
                    ("fresh_test_clean", "test", design["fresh_folds"][fold]),
                ):
                    values = score(features[split][indices], reference_mean, direction)
                    pvalues = rank_pvalues(calibration_scores, values)
                    population_scores[population].append(values)
                    population_pvalues[population].append(pvalues)
                    for source_index, raw_score, pvalue in zip(indices, values, pvalues):
                        query_rows.append(
                            {
                                "seed": seed,
                                "view": view,
                                "population": population,
                                "source_split": split,
                                "source_index": int(source_index),
                                "fold": fold,
                                "raw_score": float(raw_score),
                                "rank_pvalue": float(pvalue),
                                "flagged": int(pvalue <= ALPHA),
                            }
                        )
            combined_scores = {
                population: np.concatenate(parts)
                for population, parts in population_scores.items()
            }
            combined_pvalues = {
                population: np.concatenate(parts)
                for population, parts in population_pvalues.items()
            }
            member_p = combined_pvalues["member_train_clean"]
            fresh_p = combined_pvalues["fresh_test_clean"]
            member_fpr = float(np.mean(member_p <= ALPHA))
            fresh_fpr = float(np.mean(fresh_p <= ALPHA))
            member_ks = [
                ks_2samp(calibration_by_fold[fold], population_scores["member_train_clean"][fold], method="auto")
                for fold in (0, 1)
            ]
            fresh_ks = [
                ks_2samp(calibration_by_fold[fold], population_scores["fresh_test_clean"][fold], method="auto")
                for fold in (0, 1)
            ]
            metric_rows.append(
                {
                    "stage": args.stage,
                    "seed": seed,
                    "view": view,
                    "eligible_victim": victim["eligible"],
                    "member_count": len(member_p),
                    "member_flagged": int(np.sum(member_p <= ALPHA)),
                    "member_fpr": member_fpr,
                    "fresh_count": len(fresh_p),
                    "fresh_flagged": int(np.sum(fresh_p <= ALPHA)),
                    "fresh_fpr": fresh_fpr,
                    "member_minus_fresh_fpr_gap": member_fpr - fresh_fpr,
                    "member_max_fold_ks": max(float(item.statistic) for item in member_ks),
                    "member_min_fold_ks_pvalue": min(float(item.pvalue) for item in member_ks),
                    "fresh_max_fold_ks": max(float(item.statistic) for item in fresh_ks),
                    "fresh_min_fold_ks_pvalue": min(float(item.pvalue) for item in fresh_ks),
                    "direction_fold_cosine": float(np.dot(directions[0], directions[1])),
                    "direction_fold_0_norm": direction_norms[0],
                    "direction_fold_1_norm": direction_norms[1],
                    "finite_sample_null_rate": 4.0 / 97.0,
                }
            )
            view_receipts[view] = {
                "direction_norms": direction_norms,
                "direction_fold_cosine": float(np.dot(directions[0], directions[1])),
                "calibration_score_hashes": [
                    hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()
                    for values in calibration_by_fold
                ],
            }
            print(
                f"seed={seed} view={view}: member FPR={member_fpr:.4f}, "
                f"fresh FPR={fresh_fpr:.4f}, gap={member_fpr - fresh_fpr:+.4f}",
                flush=True,
            )

        elapsed = time.time() - started
        receipts.append(
            {
                "seed": seed,
                "elapsed_seconds": elapsed,
                "within_runtime_budget": bool(elapsed <= 300.0),
                "pool_selected_indices_sha256": pool["receipt"]["selected_indices_sha256"],
                "checkpoints": {"clean": clean_receipt, "suspect": suspect_receipt},
                "indices": {
                    "reference_sha256": array_hash(design["reference"]),
                    "calibration_sha256": array_hash(design["calibration"]),
                    "candidate_fold_sha256": [array_hash(values) for values in design["candidate_folds"]],
                    "member_fold_sha256": [array_hash(values) for values in design["member_folds"]],
                    "fresh_fold_sha256": [array_hash(values) for values in design["fresh_folds"]],
                },
                "counts": {
                    "reference": len(design["reference"]),
                    "calibration": len(design["calibration"]),
                    "candidate_folds": [len(values) for values in design["candidate_folds"]],
                    "member_folds": [len(values) for values in design["member_folds"]],
                    "fresh_folds": [len(values) for values in design["fresh_folds"]],
                },
                "views": view_receipts,
            }
        )

    replay_ok = replay_counts(query_rows, metric_rows)
    finite_ok = all(
        np.isfinite(float(row[field]))
        for row in metric_rows
        for field in (
            "member_fpr", "fresh_fpr", "member_minus_fresh_fpr_gap",
            "direction_fold_cosine", "direction_fold_0_norm", "direction_fold_1_norm"
        )
    )
    gate = {
        "both_archived_victims_eligible": all(bool(row["eligible"]) for row in victim_rows),
        "all_dataset_protocol_and_checkpoint_hashes_match": True,
        "reference_calibration_and_query_indices_disjoint": True,
        "both_views_finite_nonzero": bool(finite_ok),
        "all_query_flags_and_counts_replay": bool(replay_ok),
        "all_seeds_within_runtime_budget": all(row["within_runtime_budget"] for row in receipts),
    }
    if args.stage == "development":
        gate["expand_to_confirmation"] = all(gate.values())
    else:
        gate["confirmation_complete"] = all(gate.values())
    write_csv(args.output / "victim_metrics.csv", victim_rows)
    write_csv(args.output / "per_seed_view_metrics.csv", metric_rows)
    write_csv(args.output / "per_query_scores.csv", query_rows)
    write_json(args.output / "seed_receipts.json", receipts)
    manifest = {
        "experiment": "bloodmnist_fresh_query_calibration_v1",
        "stage": args.stage,
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(Path(__file__)),
        "dataset_npz_sha256": sha256_file(dataset_path),
        "source_protocol_sha256": sha256_file(SOURCE_PROTOCOL),
        "seeds": list(seeds),
        "views": list(VIEWS),
        "query_rows": len(query_rows),
        "gate": gate,
        "elapsed_seconds": time.time() - started_all,
    }
    write_json(args.output / "manifest.json", manifest)
    lines = [
        f"# BloodMNIST member-versus-fresh calibration ({args.stage})",
        "",
        f"Non-performance gate: **{'PASS' if all(gate.values()) else 'STOP'}**",
        "",
        "| seed | view | member FPR | fresh FPR | member - fresh gap |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['seed']} | {row['view']} | {row['member_fpr']:.3%} | "
            f"{row['fresh_fpr']:.3%} | {row['member_minus_fresh_fpr_gap']:+.3%} |"
        )
    lines.extend([
        "",
        "Fresh test queries never train either checkpoint or either score direction. Dataset-split exchangeability remains an assumption, so this is an empirical calibration comparison rather than an unconditional conformal theorem.",
        "Performance was not used as an expansion gate.",
        "",
    ])
    (args.output / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    parser.add_argument("--stage", choices=("development", "confirmation"), default="development")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--development-root", type=Path, default=DEFAULT_DEVELOPMENT)
    args = parser.parse_args()
    if args.seeds is None:
        args.seeds = list(DEVELOPMENT_SEEDS if args.stage == "development" else CONFIRMATION_SEEDS)
    if args.output is None:
        args.output = DEFAULT_DEVELOPMENT if args.stage == "development" else DEFAULT_CONFIRMATION
    run(args)


if __name__ == "__main__":
    main()
