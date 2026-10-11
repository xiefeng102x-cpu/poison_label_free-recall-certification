#!/usr/bin/env python3
"""Retrain classical and QML victims after poison-screen sanitization."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_cross_attack_pilot as cross
import run_bloodmnist_group_coherent_10seeds as baseline
import run_bloodmnist_whitebox_adaptive_pilot as whitebox
import run_unified_oneclass_poison_screening as unified
import train_bloodmnist_clean_encoder as clean_encoder


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PAPER14_ROOT / "results/bloodmnist_post_sanitization_v1"
PROTOCOL = PAPER14_ROOT / "results/top_journal_program_v1/phase5_post_sanitization_protocol.json"
SOURCE = PAPER14_ROOT / "results/bloodmnist_covariance_free_confirmation_v1"
SEEDS = tuple(range(62, 72))
CONDITIONS = ("detected_removal", "random_matched_removal", "oracle_poison_removal")


def detection_flags() -> dict[int, np.ndarray]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    with (SOURCE / "per_sample_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["scenario"] == "fixed_pca_additive":
                grouped[int(row["seed"])].append(row)
    output = {}
    for seed, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["sample_index"]))
        if len(ordered) != 660:
            raise RuntimeError(f"Seed {seed} has {len(ordered)} detector rows")
        output[seed] = np.asarray([int(row["flagged"]) for row in ordered], dtype=bool)
    return output


def fixed_eligibility() -> dict[int, bool]:
    output = {}
    with (SOURCE / "victim_metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["scenario"] == "fixed_pca_additive":
                output[int(row["seed"])] = bool(int(row["eligible"]))
    return output


def removal_masks(condition: str, flags: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if condition == "detected_removal":
        remove = flags.copy()
    elif condition == "random_matched_removal":
        rng = np.random.default_rng(20260928 + seed)
        remove = np.zeros(660, dtype=bool)
        remove[rng.permutation(660)[: int(flags.sum())]] = True
    elif condition == "oracle_poison_removal":
        remove = np.concatenate([np.zeros(600, dtype=bool), np.ones(60, dtype=bool)])
    else:
        raise ValueError(condition)
    return remove[:600], remove[600:]


def sanitized_splits(splits: dict, remove_clean: np.ndarray, remove_poison: np.ndarray) -> dict:
    output = copy.deepcopy(splits)
    target = splits["train"]["target"][~remove_clean]
    source = splits["train"]["non_target"]
    poison = splits["archived_poison"][~remove_poison]
    output["train"]["target"] = target
    output["train"]["clean_x"] = np.concatenate([target, source], axis=0)
    output["train"]["clean_y"] = np.concatenate(
        [np.ones(len(target), dtype=np.int64), np.zeros(len(source), dtype=np.int64)]
    )
    output["archived_poison"] = poison
    return output


def train_qml(output: Path, splits: dict, seed: int) -> tuple[clean_encoder.CleanQNN, dict]:
    (output / "models").mkdir(parents=True, exist_ok=True)
    clean_model, clean_record = clean_encoder.train_one(
        baseline.clean_arrays(splits), seed, "local", 0, 40, 3e-3, 1e-4, 64, 10
    )
    if len(splits["archived_poison"]) == 0:
        return clean_model, {"selected_epoch": 0, "clean_record": clean_record}
    attacked_model, attacked_record = baseline.load_or_adapt(
        output, clean_model, splits, seed, 5
    )
    return attacked_model, {"selected_epoch": attacked_record["selected_epoch"], "clean_record": clean_record}


def train_classical(splits: dict, seed: int):
    train_x = np.concatenate([splits["train"]["clean_x"], splits["archived_poison"]], axis=0)
    train_y = np.concatenate(
        [splits["train"]["clean_y"], np.ones(len(splits["archived_poison"]), dtype=np.int64)]
    )
    clean_x = splits["train"]["clean_x"]
    mean = clean_x.mean(axis=0, keepdims=True)
    scale = clean_x.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return cross.train_classical_arrays(train_x, train_y, seed=seed, epochs=80, mean=mean, scale=scale)


def classical_metrics(model, splits: dict) -> tuple[float, float]:
    target = cross.predict_classical(model, splits["test"]["target"])
    source = cross.predict_classical(model, splits["test"]["non_target"])
    triggered = cross.predict_classical(model, splits["test"]["triggered_non_target"])
    ca = 0.5 * (float(np.mean(target == 1)) + float(np.mean(source == 0)))
    return ca, float(np.mean(triggered == 1))


def main(args: argparse.Namespace) -> None:
    if tuple(args.seeds) != SEEDS:
        raise ValueError("Post-sanitization experiment is frozen to seeds 62--71")
    args.output.mkdir(parents=True, exist_ok=True)
    flags_by_seed = detection_flags()
    eligibility = fixed_eligibility()
    class_data = heldout.collect_classes(args.dataset_root)
    rgb_trigger, _ = baseline.fixed_rgb_trigger(class_data)
    rows, receipts = [], []
    for seed in args.seeds:
        if not eligibility[seed]:
            continue
        print(f"seed={seed}: post-sanitization retraining", flush=True)
        prepared = baseline.preprocess_seed(class_data, seed)
        splits, _ = baseline.seed_splits(prepared, rgb_trigger, seed)
        flags = flags_by_seed[seed]
        for condition in CONDITIONS:
            remove_clean, remove_poison = removal_masks(condition, flags, seed)
            sanitized = sanitized_splits(splits, remove_clean, remove_poison)
            condition_root = args.output / "models" / condition
            condition_root.mkdir(parents=True, exist_ok=True)
            qml_model, qml_record = train_qml(condition_root, sanitized, seed)
            qml_ca, qml_asr = whitebox.balanced_ca_and_asr(
                qml_model,
                sanitized["test"]["target"],
                sanitized["test"]["non_target"],
                sanitized["test"]["triggered_non_target"],
            )
            classical_model = train_classical(sanitized, seed)
            classical_ca, classical_asr = classical_metrics(classical_model, sanitized)
            row = {
                "seed": seed,
                "condition": condition,
                "clean_candidates_removed": int(remove_clean.sum()),
                "poisons_removed": int(remove_poison.sum()),
                "poisons_remaining": int((~remove_poison).sum()),
                "qml_balanced_ca": qml_ca,
                "qml_asr": qml_asr,
                "qml_selected_adaptation_epoch": qml_record["selected_epoch"],
                "classical_balanced_ca": classical_ca,
                "classical_asr": classical_asr,
            }
            rows.append(row)
            receipts.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "sanitized_clean_sha256": whitebox.sha256_array(sanitized["train"]["clean_x"]),
                    "remaining_poison_sha256": whitebox.sha256_array(sanitized["archived_poison"]),
                }
            )
            print(
                f"seed={seed} {condition}: removed clean/poison={remove_clean.sum()}/{remove_poison.sum()}; "
                f"QML={qml_ca:.3f}/{qml_asr:.3f}; classical={classical_ca:.3f}/{classical_asr:.3f}",
                flush=True,
            )
    summaries = []
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        summaries.append(
            {
                "condition": condition,
                "seed_count": len(selected),
                "mean_clean_removed": float(np.mean([row["clean_candidates_removed"] for row in selected])),
                "mean_poison_removed": float(np.mean([row["poisons_removed"] for row in selected])),
                "mean_qml_balanced_ca": float(np.mean([row["qml_balanced_ca"] for row in selected])),
                "mean_qml_asr": float(np.mean([row["qml_asr"] for row in selected])),
                "mean_classical_balanced_ca": float(np.mean([row["classical_balanced_ca"] for row in selected])),
                "mean_classical_asr": float(np.mean([row["classical_asr"] for row in selected])),
            }
        )
    unified.write_csv(args.output / "per_seed_metrics.csv", rows)
    unified.write_csv(args.output / "summary.csv", summaries)
    unified.json_dump(
        args.output / "manifest.json",
        {
            "experiment": "bloodmnist_post_sanitization_v1",
            "protocol": str(PROTOCOL),
            "protocol_sha256": whitebox.sha256_file(PROTOCOL),
            "dataset_npz_sha256": whitebox.sha256_file(args.dataset_root / "bloodmnist.npz"),
            "seeds": list(args.seeds),
            "eligible_seeds": [seed for seed in args.seeds if eligibility[seed]],
            "conditions": list(CONDITIONS),
            "receipts": receipts,
        },
    )
    lines = [
        "# BloodMNIST post-sanitization retraining",
        "",
        "| condition | seeds | clean removed | poison removed | QML CA | QML ASR | classical CA | classical ASR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['condition']} | {row['seed_count']} | {row['mean_clean_removed']:.1f} | {row['mean_poison_removed']:.1f} | {row['mean_qml_balanced_ca']:.3f} | {row['mean_qml_asr']:.3f} | {row['mean_classical_balanced_ca']:.3f} | {row['mean_classical_asr']:.3f} |"
        )
    lines.append("")
    (args.output / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    main(parser.parse_args())
