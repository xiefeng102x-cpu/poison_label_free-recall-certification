#!/usr/bin/env python3
"""Poison prevalence, trusted-pool contamination and finite-shot boundaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_cross_attack_pilot as cross
import run_bloodmnist_group_coherent_10seeds as baseline
import run_bloodmnist_mechanism_ablations as ablation
import run_bloodmnist_whitebox_adaptive_pilot as whitebox
import run_unified_oneclass_poison_screening as unified
import train_bloodmnist_clean_encoder as clean_encoder


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PAPER14_ROOT / "results/bloodmnist_boundary_study_v1"
PROTOCOL = PAPER14_ROOT / "results/top_journal_program_v1/phase5_boundary_protocol.json"
MODEL_ROOT = PAPER14_ROOT / "results/bloodmnist_covariance_free_confirmation_v1/fixed_local"
VICTIM_METRICS = PAPER14_ROOT / "results/bloodmnist_covariance_free_confirmation_v1/victim_metrics.csv"
SEEDS = tuple(range(62, 72))
FRACTIONS = (0.01, 0.03, 0.05, 0.10, 0.20, 0.30)
CONTAMINATIONS = (0.0, 0.01, 0.03, 0.05, 0.10)
SHOTS = ("exact", 128, 512, 2048)
METHODS = ("original_whitened_cf_cmd", "covariance_free_cf_cmd_u")


def shot_sample(values: np.ndarray, shots: int | str, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if shots == "exact":
        return values.copy()
    probability = np.clip((values + 1.0) / 2.0, 0.0, 1.0)
    counts = rng.binomial(int(shots), probability)
    return (2.0 * counts / int(shots) - 1.0).astype(np.float64)


def read_fixed_eligibility() -> dict[int, bool]:
    output = {}
    with VICTIM_METRICS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["scenario"] == "fixed_pca_additive":
                output[int(row["seed"])] = bool(int(row["eligible"]))
    return output


def model_pair(seed: int) -> tuple[clean_encoder.CleanQNN, clean_encoder.CleanQNN]:
    clean = clean_encoder.CleanQNN("local", 0)
    clean.load_state_dict(
        torch.load(MODEL_ROOT / "models" / f"seed_{seed}_clean.pt", map_location="cpu", weights_only=False)["state_dict"]
    )
    attacked = clean_encoder.CleanQNN("local", 0)
    attacked.load_state_dict(
        torch.load(MODEL_ROOT / "models" / f"seed_{seed}_adaptive.pt", map_location="cpu", weights_only=False)["state_dict"]
    )
    return clean, attacked


def exact_observables(model: clean_encoder.CleanQNN, values: np.ndarray) -> np.ndarray:
    return whitebox.numpy_xz(model, values)


def summarize(rows: list[dict], eligible: dict[int, bool]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if eligible[int(row["seed"])]:
            grouped[
                (
                    row["method"],
                    row["shots"],
                    float(row["poison_fraction"]),
                    float(row["trusted_contamination"]),
                )
            ].append(row)
    output = []
    for key, selected in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        tp, fn = sum(row["tp"] for row in selected), sum(row["fn"] for row in selected)
        fp, tn = sum(row["fp"] for row in selected), sum(row["tn"] for row in selected)
        tpr, fpr = tp / (tp + fn), fp / (fp + tn)
        output.append(
            {
                "method": key[0],
                "shots": key[1],
                "poison_fraction": key[2],
                "trusted_contamination": key[3],
                "eligible_seed_count": len(selected),
                "pooled_tpr": tpr,
                "pooled_fpr": fpr,
                "gate_pass": int(tpr >= 0.90 and fpr <= 0.05),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
            }
        )
    return output


def boundaries(summaries: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in summaries:
        grouped[(row["method"], str(row["shots"]), row["trusted_contamination"])].append(row)
    output = []
    for key, selected in sorted(grouped.items()):
        passing = sorted(row["poison_fraction"] for row in selected if row["gate_pass"])
        output.append(
            {
                "method": key[0],
                "shots": key[1],
                "trusted_contamination": key[2],
                "minimum_passing_poison_fraction": passing[0] if passing else "",
                "passing_poison_fractions": ",".join(map(str, passing)),
                "monotone_from_minimum": int(
                    bool(passing)
                    and passing == [
                        row["poison_fraction"]
                        for row in sorted(selected, key=lambda row: row["poison_fraction"])
                        if row["poison_fraction"] >= passing[0]
                    ]
                ),
            }
        )
    return output


def main(args: argparse.Namespace) -> None:
    if tuple(args.seeds) != SEEDS:
        raise ValueError("Boundary study is frozen to seeds 62--71")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    eligible = read_fixed_eligibility()
    class_data = heldout.collect_classes(args.dataset_root)
    rgb_trigger, _ = baseline.fixed_rgb_trigger(class_data)
    rows = []
    for seed in args.seeds:
        print(f"seed={seed}: prevalence/contamination/shot grid", flush=True)
        prepared = baseline.preprocess_seed(class_data, seed)
        splits, _ = baseline.seed_splits(prepared, rgb_trigger, seed)
        clean_model, attacked_model = model_pair(seed)
        values = {
            "trusted_clean": splits["val"]["target"],
            "trusted_poison": splits["val"]["triggered_non_target"],
            "candidate_clean": splits["train"]["target"],
            "candidate_poison": splits["train"]["triggered_non_target"],
        }
        clean_observables = {name: exact_observables(clean_model, array) for name, array in values.items()}
        attacked_observables = {name: exact_observables(attacked_model, array) for name, array in values.items()}
        poison_order = np.random.default_rng(20260928 + seed).permutation(len(values["candidate_poison"]))
        trusted_clean_order = np.random.default_rng(20260929 + seed).permutation(len(values["trusted_clean"]))
        trusted_poison_order = np.random.default_rng(20260930 + seed).permutation(len(values["trusted_poison"]))

        for shots in SHOTS:
            rng = np.random.default_rng(20261001 + 1009 * seed + (0 if shots == "exact" else int(shots)))
            drift = {
                name: shot_sample(attacked_observables[name], shots, rng)
                - shot_sample(clean_observables[name], shots, rng)
                for name in values
            }
            for contamination in CONTAMINATIONS:
                contaminated_count = int(round(contamination * len(drift["trusted_clean"])))
                trusted = drift["trusted_clean"].copy()
                if contaminated_count:
                    clean_indices = trusted_clean_order[:contaminated_count]
                    poison_indices = trusted_poison_order[:contaminated_count]
                    trusted[clean_indices] = drift["trusted_poison"][poison_indices]
                for fraction in FRACTIONS:
                    poison_count = max(1, int(round(len(drift["candidate_clean"]) * fraction / (1.0 - fraction))))
                    poison = drift["candidate_poison"][poison_order[:poison_count]]
                    config = {
                        "seed": seed,
                        "trigger": "boundary_fixed_pca_additive",
                        "archive_ca": 0.0,
                        "archive_asr": 0.0,
                    }
                    whitened, _ = cross.evaluate_external_reference(
                        config,
                        "original_whitened_cf_cmd",
                        trusted,
                        drift["candidate_clean"],
                        poison,
                        args.alpha,
                    )
                    unwhitened, _ = ablation.custom_crossfit(
                        seed,
                        "covariance_free_cf_cmd_u",
                        trusted,
                        drift["candidate_clean"],
                        poison,
                        args.alpha,
                        whitening=False,
                    )
                    for method, result in (
                        ("original_whitened_cf_cmd", whitened),
                        ("covariance_free_cf_cmd_u", unwhitened),
                    ):
                        rows.append(
                            {
                                "seed": seed,
                                "eligible_victim": int(eligible[seed]),
                                "method": method,
                                "shots": shots,
                                "poison_fraction": fraction,
                                "poison_count": poison_count,
                                "trusted_contamination": contamination,
                                "trusted_contaminated_count": contaminated_count,
                                "tp": result["tp"],
                                "fn": result["fn"],
                                "fp": result["fp"],
                                "tn": result["tn"],
                                "tpr": result["tpr"],
                                "fpr": result["fpr"],
                            }
                        )
    summaries = summarize(rows, eligible)
    boundary_rows = boundaries(summaries)
    unified.write_csv(args.output / "per_config_metrics.csv", rows)
    unified.write_csv(args.output / "pooled_boundary_grid.csv", summaries)
    unified.write_csv(args.output / "boundary_summary.csv", boundary_rows)
    unified.json_dump(
        args.output / "manifest.json",
        {
            "experiment": "bloodmnist_boundary_study_v1",
            "protocol": str(PROTOCOL),
            "protocol_sha256": whitebox.sha256_file(PROTOCOL),
            "dataset_npz_sha256": whitebox.sha256_file(args.dataset_root / "bloodmnist.npz"),
            "seeds": list(args.seeds),
            "eligible_seeds": [seed for seed in args.seeds if eligible[seed]],
            "candidate_poison_fractions": list(FRACTIONS),
            "trusted_contamination": list(CONTAMINATIONS),
            "shots": list(SHOTS),
            "methods": list(METHODS),
            "alpha": args.alpha,
            "per_config_row_count": len(rows),
        },
    )
    lines = [
        "# BloodMNIST operating-boundary study",
        "",
        "The grid uses saved victims; no model is retrained. Results pool only attack-eligible seeds.",
        "",
        "## Exact measurement, clean trusted pool",
        "",
        "| method | poison fraction | pooled TPR | pooled FPR | gate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        if str(row["shots"]) == "exact" and row["trusted_contamination"] == 0.0:
            lines.append(
                f"| {row['method']} | {row['poison_fraction']:.2f} | {row['pooled_tpr']:.3f} | {row['pooled_fpr']:.3f} | {row['gate_pass']} |"
            )
    lines.extend(
        [
            "",
            "`boundary_summary.csv` gives the minimum passing poison fraction for every method × shots × trusted-contamination condition. Non-monotone pass sets are flagged rather than silently collapsed to a single threshold.",
            "",
        ]
    )
    (args.output / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--alpha", type=float, default=0.05)
    main(parser.parse_args())
