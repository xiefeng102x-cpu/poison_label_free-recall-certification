#!/usr/bin/env python3
"""Replay frozen member-versus-fresh calibration scores and artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import run_bloodmnist_fresh_query_calibration as experiment
from analyze_checkpoint_auditability_theory_probe import rank_pvalues
from run_bloodmnist_multiclass_backdoorbench_pilot import (
    TARGET_CLASS,
    build_poisoned_training,
    channel_stats,
    hidden_features,
    load_bloodmnist,
    sha256_file,
)


RECEIPT = experiment.PAPER14_ROOT / "results/top_journal_program_v1/fresh_query_calibration_verification.json"
STAGES = {
    "development": (
        experiment.DEFAULT_DEVELOPMENT,
        experiment.DEVELOPMENT_SEEDS,
    ),
    "confirmation": (
        experiment.DEFAULT_CONFIRMATION,
        experiment.CONFIRMATION_SEEDS,
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def close(left: float | str, right: float | str) -> bool:
    return bool(np.isclose(float(left), float(right), atol=1e-12, rtol=0.0))


def expected_records(
    stage: str,
    seed: int,
    data: dict[str, dict[str, np.ndarray]],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[dict[tuple[str, str, int], dict[str, float | int]], dict[str, dict[str, float | int]]]:
    pool = build_poisoned_training(data["train"]["images"], data["train"]["labels"], seed=seed)
    clean_model, _ = experiment.load_model(stage, seed, "clean", mean, std)
    suspect_model, _ = experiment.load_model(stage, seed, "suspect", mean, std)
    design = experiment.split_design(pool, data["val"]["labels"], data["test"]["labels"], seed)
    suspect = {
        "train": hidden_features(suspect_model, pool["images"]),
        "val": hidden_features(suspect_model, data["val"]["images"]),
        "test": hidden_features(suspect_model, data["test"]["images"]),
    }
    clean = {
        "train": hidden_features(clean_model, pool["images"]),
        "val": hidden_features(clean_model, data["val"]["images"]),
        "test": hidden_features(clean_model, data["test"]["images"]),
    }
    views = {
        "suspect_hidden64": suspect,
        "paired_hidden64_drift": {
            split: suspect[split] - clean[split]
            for split in ("train", "val", "test")
        },
    }
    records: dict[tuple[str, str, int], dict[str, float | int]] = {}
    metrics: dict[str, dict[str, float | int]] = {}
    for view, features in views.items():
        reference_mean = features["val"][design["reference"]].mean(axis=0)
        population_pvalues: dict[str, list[np.ndarray]] = {
            "member_train_clean": [],
            "fresh_test_clean": [],
        }
        for fold in (0, 1):
            direction, _ = experiment.unit_direction(
                features["train"][design["candidate_folds"][1 - fold]], reference_mean
            )
            calibration = experiment.score(
                features["val"][design["calibration"]], reference_mean, direction
            )
            for population, split, indices in (
                ("member_train_clean", "train", design["member_folds"][fold]),
                ("fresh_test_clean", "test", design["fresh_folds"][fold]),
            ):
                scores = experiment.score(features[split][indices], reference_mean, direction)
                pvalues = rank_pvalues(calibration, scores)
                population_pvalues[population].append(pvalues)
                for index, raw_score, pvalue in zip(indices, scores, pvalues):
                    key = (view, population, int(index))
                    if key in records:
                        raise AssertionError(f"duplicate replay key {key}")
                    records[key] = {
                        "fold": fold,
                        "raw_score": float(raw_score),
                        "rank_pvalue": float(pvalue),
                        "flagged": int(pvalue <= experiment.ALPHA),
                    }
        member = np.concatenate(population_pvalues["member_train_clean"])
        fresh = np.concatenate(population_pvalues["fresh_test_clean"])
        metrics[view] = {
            "member_count": len(member),
            "member_flagged": int(np.sum(member <= experiment.ALPHA)),
            "member_fpr": float(np.mean(member <= experiment.ALPHA)),
            "fresh_count": len(fresh),
            "fresh_flagged": int(np.sum(fresh <= experiment.ALPHA)),
            "fresh_fpr": float(np.mean(fresh <= experiment.ALPHA)),
        }
    return records, metrics


def verify_stage(
    stage: str,
    root: Path,
    seeds: tuple[int, ...],
    data: dict[str, dict[str, np.ndarray]],
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, Any]:
    manifest = read_json(root / "manifest.json")
    if not (
        manifest["experiment"] == "bloodmnist_fresh_query_calibration_v1"
        and manifest["stage"] == stage
        and tuple(manifest["seeds"]) == seeds
        and manifest["protocol_sha256"] == sha256_file(experiment.PROTOCOL)
        and manifest["dataset_npz_sha256"]
        == sha256_file(Path("/private/tmp/blood-ca-data/bloodmnist.npz"))
        and all(manifest["gate"].values())
    ):
        raise AssertionError(f"{stage} manifest mismatch")
    query_rows = read_csv(root / "per_query_scores.csv")
    metric_rows = read_csv(root / "per_seed_view_metrics.csv")
    expected_query_count = len(seeds) * 2 * (2330 + 666)
    if len(query_rows) != expected_query_count or len(metric_rows) != len(seeds) * 2:
        raise AssertionError(f"{stage} row count mismatch")
    reported_metrics = {(int(row["seed"]), row["view"]): row for row in metric_rows}
    reported_queries: dict[int, dict[tuple[str, str, int], dict[str, str]]] = {}
    for row in query_rows:
        seed = int(row["seed"])
        key = (row["view"], row["population"], int(row["source_index"]))
        if key in reported_queries.setdefault(seed, {}):
            raise AssertionError(f"duplicate reported query {seed} {key}")
        reported_queries[seed][key] = row

    pooled: dict[str, dict[str, int]] = {
        view: {"member_flagged": 0, "member_count": 0, "fresh_flagged": 0, "fresh_count": 0}
        for view in experiment.VIEWS
    }
    summaries = []
    for seed in seeds:
        expected_queries, expected_metrics = expected_records(stage, seed, data, mean, std)
        actual_queries = reported_queries[seed]
        if set(actual_queries) != set(expected_queries):
            raise AssertionError(f"seed {seed} query provenance mismatch")
        for key, expected in expected_queries.items():
            actual = actual_queries[key]
            if not (
                int(actual["fold"]) == expected["fold"]
                and int(actual["flagged"]) == expected["flagged"]
                and close(actual["raw_score"], expected["raw_score"])
                and close(actual["rank_pvalue"], expected["rank_pvalue"])
                and actual["source_split"]
                == ("train" if key[1] == "member_train_clean" else "test")
            ):
                raise AssertionError(f"seed {seed} score mismatch {key}")
        for view, expected in expected_metrics.items():
            actual = reported_metrics[(seed, view)]
            for name, value in expected.items():
                if name.endswith(("count", "flagged")):
                    if int(actual[name]) != value:
                        raise AssertionError(f"seed {seed} {view} {name} mismatch")
                elif not close(actual[name], value):
                    raise AssertionError(f"seed {seed} {view} {name} mismatch")
            gap = float(expected["member_fpr"]) - float(expected["fresh_fpr"])
            if not close(actual["member_minus_fresh_fpr_gap"], gap):
                raise AssertionError(f"seed {seed} {view} gap mismatch")
            for name in pooled[view]:
                pooled[view][name] += int(expected[name])
            summaries.append(
                {
                    "seed": seed,
                    "view": view,
                    "member_fpr": expected["member_fpr"],
                    "fresh_fpr": expected["fresh_fpr"],
                    "gap": gap,
                }
            )

    pooled_rates = {}
    for view, counts in pooled.items():
        member_rate = counts["member_flagged"] / counts["member_count"]
        fresh_rate = counts["fresh_flagged"] / counts["fresh_count"]
        pooled_rates[view] = {
            **counts,
            "member_fpr": member_rate,
            "fresh_fpr": fresh_rate,
            "member_minus_fresh_gap": member_rate - fresh_rate,
        }
    files = (
        "manifest.json",
        "victim_metrics.csv",
        "per_seed_view_metrics.csv",
        "per_query_scores.csv",
        "seed_receipts.json",
        "EXPERIMENT_REPORT.md",
    )
    return {
        "stage": stage,
        "seeds": list(seeds),
        "query_rows": len(query_rows),
        "pooled": pooled_rates,
        "seed_summaries": summaries,
        "artifact_sha256": {name: sha256_file(root / name) for name in files},
    }


def main() -> None:
    data = load_bloodmnist(Path("/private/tmp/blood-ca-data"))
    mean, std = channel_stats(data["train"]["images"])
    stages = [
        verify_stage(stage, root, seeds, data, mean, std)
        for stage, (root, seeds) in STAGES.items()
    ]
    receipt = {
        "verification": "PASS",
        "protocol_sha256": sha256_file(experiment.PROTOCOL),
        "runner_sha256": sha256_file(experiment.PAPER14_ROOT / "src/run_bloodmnist_fresh_query_calibration.py"),
        "verifier_sha256": sha256_file(Path(__file__)),
        "dataset_npz_sha256": sha256_file(Path("/private/tmp/blood-ca-data/bloodmnist.npz")),
        "checks": [
            "checkpoint, protocol and dataset hashes",
            "deterministic reference, calibration, candidate-fold and fresh-query splits",
            "raw scores, rank p-values and per-query flags from archived checkpoints",
            "per-seed and pooled population counts and FPR gaps",
        ],
        "stages": stages,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    for stage in stages:
        for view, values in stage["pooled"].items():
            print(
                f"{stage['stage']} {view}: member={values['member_fpr']:.6f}, "
                f"fresh={values['fresh_fpr']:.6f}, gap={values['member_minus_fresh_gap']:+.6f}"
            )
    print(f"PASS: wrote {RECEIPT}")


if __name__ == "__main__":
    main()
