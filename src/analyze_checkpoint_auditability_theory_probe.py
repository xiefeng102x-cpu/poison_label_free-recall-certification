#!/usr/bin/env python3
"""Retrospective theory probe on frozen Phase-A Pauli feature caches.

This analysis is deliberately label-informed: train-pair labels estimate an
oracle linear direction, while validation and test pairs remain untouched
during fitting.  It tests representation geometry and calibration shift; it is
not an unsupervised poisoned-sample detector result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score


PAPER14_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/checkpoint_auditability_theory_protocol_v1.json"
)
DEFAULT_SOURCE = PAPER14_ROOT / "results/phase_a_mnist5q_real_v1"
DEFAULT_OUTPUT = PAPER14_ROOT / "results/checkpoint_auditability_theory_probe_v1"
SEEDS = (42, 43, 44)
CHECKPOINTS = ("middle", "output")
REPRESENTATIONS = ("Z", "XYZ", "Full")
VIEWS = ("suspect_absolute", "paired_checkpoint_drift")


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


def pauli_indices(n_qubits: int = 5) -> dict[str, np.ndarray]:
    labels = [
        "".join(chars)
        for chars in itertools.product(("I", "X", "Y", "Z"), repeat=n_qubits)
        if any(char != "I" for char in chars)
    ]
    weights = np.asarray([sum(char != "I" for char in label) for label in labels])
    z = np.asarray(
        [index for index, label in enumerate(labels) if weights[index] == 1 and "Z" in label],
        dtype=np.int64,
    )
    xyz = np.flatnonzero(weights == 1).astype(np.int64)
    return {"Z": z, "XYZ": xyz, "Full": np.arange(len(labels), dtype=np.int64)}


def ridge_discriminant(
    clean: np.ndarray,
    attack: np.ndarray,
    ridge_fraction: float = 0.1,
) -> dict[str, np.ndarray | float]:
    """Fit a ridge Mahalanobis direction using a sample-space solve.

    Sigma is the train-clean covariance plus ``ridge_fraction`` times its mean
    marginal variance.  The Woodbury form avoids a 1023-by-1023 inverse.
    """

    clean = np.asarray(clean, dtype=np.float64)
    attack = np.asarray(attack, dtype=np.float64)
    if clean.ndim != 2 or attack.shape != clean.shape or len(clean) < 2:
        raise ValueError("clean and attack must be aligned 2-D pair matrices")
    if not 0.0 < ridge_fraction:
        raise ValueError("ridge_fraction must be positive")
    mean_clean = clean.mean(axis=0)
    delta = attack.mean(axis=0) - mean_clean
    centered = clean - mean_clean
    mean_variance = float(np.sum(centered * centered) / ((len(clean) - 1) * clean.shape[1]))
    ridge = max(ridge_fraction * mean_variance, 1e-10)
    gram = centered @ centered.T
    system = (len(clean) - 1) * np.eye(len(clean)) + gram / ridge
    correction = centered.T @ np.linalg.solve(system, centered @ delta)
    direction = delta / ridge - correction / (ridge * ridge)
    effect_squared = max(float(delta @ direction), 0.0)
    return {
        "clean_mean": mean_clean,
        "delta": delta,
        "direction": direction,
        "ridge": ridge,
        "ridge_effect_size": float(np.sqrt(effect_squared)),
    }


def score(values: np.ndarray, fit: dict[str, np.ndarray | float]) -> np.ndarray:
    return (
        (np.asarray(values, dtype=np.float64) - np.asarray(fit["clean_mean"]))
        @ np.asarray(fit["direction"])
    )


def safe_cosine(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-15:
        return 0.0
    return float(np.dot(first, second) / denominator)


def rank_pvalues(calibration_scores: np.ndarray, query_scores: np.ndarray) -> np.ndarray:
    calibration = np.asarray(calibration_scores, dtype=np.float64).reshape(-1)
    query = np.asarray(query_scores, dtype=np.float64).reshape(-1)
    if len(calibration) == 0:
        raise ValueError("calibration scores cannot be empty")
    return (1.0 + np.sum(calibration[None, :] >= query[:, None], axis=1)) / (
        len(calibration) + 1.0
    )


def paired_bootstrap_auc_delta(
    labels: np.ndarray,
    drift_scores: np.ndarray,
    absolute_scores: np.ndarray,
    n_pairs: int,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    drift_scores = np.asarray(drift_scores, dtype=np.float64)
    absolute_scores = np.asarray(absolute_scores, dtype=np.float64)
    if labels.shape != (2 * n_pairs,) or drift_scores.shape != labels.shape:
        raise ValueError("expected clean/attack scores for every test pair")
    if absolute_scores.shape != labels.shape:
        raise ValueError("absolute scores must align")
    observed = float(
        roc_auc_score(labels, drift_scores) - roc_auc_score(labels, absolute_scores)
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        chosen = rng.integers(0, n_pairs, size=n_pairs)
        rows = np.concatenate([chosen, n_pairs + chosen])
        draws[repetition] = roc_auc_score(labels[rows], drift_scores[rows]) - roc_auc_score(
            labels[rows], absolute_scores[rows]
        )
    low, high = np.quantile(draws, (0.025, 0.975))
    return {"auc_delta": observed, "ci_low": float(low), "ci_high": float(high)}


def load_split_map(path: Path) -> dict[tuple[int, str], str]:
    output: dict[tuple[int, str], str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["seed"]), row["base_sample_id"])
            if key in output:
                raise ValueError(f"duplicate split row {key}")
            output[key] = row["split"]
    return output


def load_pair_cache(source: Path, seed: int, checkpoint: str) -> dict[str, np.ndarray]:
    paths = {
        "suspect": source / "cache" / f"seed{seed}_backdoor_{checkpoint}_pauli.npz",
        "clean": source / "cache" / f"seed{seed}_clean_{checkpoint}_pauli.npz",
    }
    payloads = {}
    for role, path in paths.items():
        with np.load(path, allow_pickle=False) as values:
            payloads[role] = {key: values[key] for key in values.files}
    if not np.array_equal(payloads["suspect"]["pair_ids"], payloads["clean"]["pair_ids"]):
        raise RuntimeError("clean and suspect cache pair IDs differ")
    return {
        "pair_ids": payloads["suspect"]["pair_ids"],
        "suspect_clean": payloads["suspect"]["clean_full"],
        "suspect_attack": payloads["suspect"]["attack_full"],
        "clean_clean": payloads["clean"]["clean_full"],
        "clean_attack": payloads["clean"]["attack_full"],
        "suspect_path": np.asarray(str(paths["suspect"])),
        "clean_path": np.asarray(str(paths["clean"])),
    }


def split_masks(
    seed: int, pair_ids: np.ndarray, split_map: dict[tuple[int, str], str]
) -> dict[str, np.ndarray]:
    assignments = np.asarray([split_map[(seed, str(pair_id))] for pair_id in pair_ids])
    masks = {name: assignments == name for name in ("train", "validation", "test")}
    if not all(np.any(mask) for mask in masks.values()) or sum(mask.sum() for mask in masks.values()) != len(pair_ids):
        raise RuntimeError("split assignments are incomplete")
    return masks


def analyze_view(
    clean_values: np.ndarray,
    attack_values: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    alpha: float,
    ridge_fraction: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    fitted = ridge_discriminant(
        clean_values[masks["train"]],
        attack_values[masks["train"]],
        ridge_fraction,
    )
    scores = {
        f"{split}_{kind}": score(values[masks[split]], fitted)
        for split in masks
        for kind, values in (("clean", clean_values), ("attack", attack_values))
    }
    aucs = {}
    for split in masks:
        labels = np.concatenate(
            [
                np.zeros(masks[split].sum(), dtype=np.int64),
                np.ones(masks[split].sum(), dtype=np.int64),
            ]
        )
        aucs[split] = float(
            roc_auc_score(
                labels,
                np.concatenate([scores[f"{split}_clean"], scores[f"{split}_attack"]]),
            )
        )
    train_delta = np.asarray(fitted["delta"])
    validation_delta = (
        attack_values[masks["validation"]].mean(axis=0)
        - clean_values[masks["validation"]].mean(axis=0)
    )
    test_delta = (
        attack_values[masks["test"]].mean(axis=0)
        - clean_values[masks["test"]].mean(axis=0)
    )
    calibration = scores["validation_clean"]
    train_clean_p = rank_pvalues(calibration, scores["train_clean"])
    test_clean_p = rank_pvalues(calibration, scores["test_clean"])
    ks = ks_2samp(calibration, scores["test_clean"], method="exact")
    metrics = {
        "train_auc": aucs["train"],
        "validation_auc": aucs["validation"],
        "test_auc": aucs["test"],
        "ridge_effect_size_train": float(fitted["ridge_effect_size"]),
        "ridge_value": float(fitted["ridge"]),
        "train_validation_direction_cosine": safe_cosine(train_delta, validation_delta),
        "train_test_direction_cosine": safe_cosine(train_delta, test_delta),
        "train_clean_rank_fpr": float(np.mean(train_clean_p <= alpha)),
        "test_clean_rank_fpr": float(np.mean(test_clean_p <= alpha)),
        "train_test_clean_fpr_gap": float(
            np.mean(train_clean_p <= alpha) - np.mean(test_clean_p <= alpha)
        ),
        "validation_test_clean_ks_statistic": float(ks.statistic),
        "validation_test_clean_ks_pvalue": float(ks.pvalue),
    }
    return metrics, scores


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if tuple(protocol["source"]["seeds"]) != SEEDS:
        raise RuntimeError("protocol seed mismatch")
    if tuple(args.seeds) != SEEDS:
        raise ValueError(f"theory probe is frozen to seeds {SEEDS}")
    args.output.mkdir(parents=True, exist_ok=True)
    split_map = load_split_map(args.source / "splits.csv")
    indices = pauli_indices(5)
    rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    source_hashes: list[dict[str, Any]] = []

    for seed in SEEDS:
        for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
            cache = load_pair_cache(args.source, seed, checkpoint)
            pair_ids = cache["pair_ids"]
            masks = split_masks(seed, pair_ids, split_map)
            for role in ("suspect_path", "clean_path"):
                path = Path(str(cache[role]))
                source_hashes.append(
                    {"seed": seed, "checkpoint": checkpoint, "role": role, "path": str(path), "sha256": sha256_file(path)}
                )
            for representation_index, representation in enumerate(REPRESENTATIONS):
                columns = indices[representation]
                view_matrices = {
                    "suspect_absolute": (
                        cache["suspect_clean"][:, columns],
                        cache["suspect_attack"][:, columns],
                    ),
                    "paired_checkpoint_drift": (
                        cache["suspect_clean"][:, columns] - cache["clean_clean"][:, columns],
                        cache["suspect_attack"][:, columns] - cache["clean_attack"][:, columns],
                    ),
                }
                view_outputs: dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
                for view, (clean_values, attack_values) in view_matrices.items():
                    metrics, scores = analyze_view(
                        clean_values,
                        attack_values,
                        masks,
                        alpha=float(protocol["outputs"]["rank_alpha"]),
                        ridge_fraction=float(
                            protocol["estimation"]["ridge_fraction_of_mean_variance"]
                        ),
                    )
                    view_outputs[view] = (metrics, scores)
                    rows.append(
                        {
                            "seed": seed,
                            "checkpoint": checkpoint,
                            "representation": representation,
                            "dimension": len(columns),
                            "view": view,
                            "train_pairs": int(masks["train"].sum()),
                            "validation_pairs": int(masks["validation"].sum()),
                            "test_pairs": int(masks["test"].sum()),
                            **metrics,
                        }
                    )
                test_count = int(masks["test"].sum())
                labels = np.concatenate(
                    [np.zeros(test_count, dtype=np.int64), np.ones(test_count, dtype=np.int64)]
                )
                absolute_scores = np.concatenate(
                    [
                        view_outputs["suspect_absolute"][1]["test_clean"],
                        view_outputs["suspect_absolute"][1]["test_attack"],
                    ]
                )
                drift_scores = np.concatenate(
                    [
                        view_outputs["paired_checkpoint_drift"][1]["test_clean"],
                        view_outputs["paired_checkpoint_drift"][1]["test_attack"],
                    ]
                )
                comparison = paired_bootstrap_auc_delta(
                    labels,
                    drift_scores,
                    absolute_scores,
                    test_count,
                    repetitions=int(protocol["outputs"]["bootstrap_repetitions"]),
                    seed=int(protocol["outputs"]["bootstrap_seed"])
                    + seed
                    + 100 * checkpoint_index
                    + 10 * representation_index,
                )
                comparisons.append(
                    {
                        "seed": seed,
                        "checkpoint": checkpoint,
                        "representation": representation,
                        "test_pairs": test_count,
                        **comparison,
                    }
                )

    aggregate = []
    for checkpoint in CHECKPOINTS:
        for representation in REPRESENTATIONS:
            selected = [
                row
                for row in comparisons
                if row["checkpoint"] == checkpoint and row["representation"] == representation
            ]
            aggregate.append(
                {
                    "checkpoint": checkpoint,
                    "representation": representation,
                    "seed_count": len(selected),
                    "mean_auc_delta": float(np.mean([row["auc_delta"] for row in selected])),
                    "minimum_auc_delta": float(np.min([row["auc_delta"] for row in selected])),
                    "maximum_auc_delta": float(np.max([row["auc_delta"] for row in selected])),
                    "all_seeds_nonnegative": bool(all(row["auc_delta"] >= 0.0 for row in selected)),
                }
            )
    positive_both = [
        representation
        for representation in REPRESENTATIONS
        if all(
            next(
                row for row in aggregate if row["checkpoint"] == checkpoint and row["representation"] == representation
            )["mean_auc_delta"]
            > 0.0
            for checkpoint in CHECKPOINTS
        )
    ]
    sign_consistent = [
        representation
        for representation in REPRESENTATIONS
        if all(
            next(
                row for row in aggregate if row["checkpoint"] == checkpoint and row["representation"] == representation
            )["all_seeds_nonnegative"]
            for checkpoint in CHECKPOINTS
        )
    ]
    support = len(positive_both) >= 2 and len(sign_consistent) >= 1
    calibration_warning = any(
        row["test_clean_rank_fpr"] > 0.075
        or abs(row["train_test_clean_fpr_gap"]) > 0.05
        for row in rows
    )
    direction_positive = all(
        row["train_test_direction_cosine"] > 0.0
        for row in rows
        if row["view"] == "paired_checkpoint_drift"
    )
    decision = {
        "supports_conditional_representation_hypothesis": bool(support),
        "positive_mean_delta_both_checkpoints": positive_both,
        "nonnegative_delta_every_seed_both_checkpoints": sign_consistent,
        "calibration_warning": bool(calibration_warning),
        "paired_drift_direction_positive_every_configuration": bool(direction_positive),
        "proceed_to_fresh_warm_start_pilot": bool(support and direction_positive),
        "scope": "retrospective_oracle_mechanism_probe_only",
    }
    write_csv(args.output / "per_view_metrics.csv", rows)
    write_csv(args.output / "paired_view_comparisons.csv", comparisons)
    write_csv(args.output / "aggregate_auc_deltas.csv", aggregate)
    write_json(args.output / "decision.json", decision)
    manifest = {
        "experiment": "checkpoint_auditability_theory_probe_v1",
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(Path(__file__)),
        "source_root": str(args.source),
        "source_manifest_sha256": sha256_file(args.source / "manifest.json"),
        "source_splits_sha256": sha256_file(args.source / "splits.csv"),
        "source_cache_hashes": source_hashes,
        "seeds": list(SEEDS),
        "decision": decision,
    }
    write_json(args.output / "manifest.json", manifest)
    lines = [
        "# Checkpoint auditability theory probe",
        "",
        "This is a retrospective, label-informed representation probe on three archived seeds. It is not confirmatory detector evidence.",
        "",
        f"Proceed to fresh warm-start pilot: **{'YES' if decision['proceed_to_fresh_warm_start_pilot'] else 'NO'}**",
        f"Calibration warning: **{'YES' if decision['calibration_warning'] else 'NO'}**",
        "",
        "| checkpoint | representation | mean drift-minus-absolute test AUC | minimum seed delta | all seeds nonnegative |",
        "|---|---|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['checkpoint']} | {row['representation']} | {row['mean_auc_delta']:.3f} | "
            f"{row['minimum_auc_delta']:.3f} | {int(row['all_seeds_nonnegative'])} |"
        )
    lines.extend(
        [
            "",
            "Mandatory boundary: the clean and backdoor victims were trained independently, so this analysis cannot identify a causal poison-only checkpoint effect.",
            "Full Pauli is an oracle representation; train labels are used only for this mechanism probe.",
            "",
        ]
    )
    (args.output / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
