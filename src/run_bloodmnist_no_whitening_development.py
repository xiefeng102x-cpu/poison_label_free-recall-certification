#!/usr/bin/env python3
"""Retrospective development check for covariance-free XZ16 scoring."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_group_coherent_10seeds as baseline
import run_bloodmnist_mechanism_ablations as ablation
import run_bloodmnist_strong_baselines as strong
import run_unified_oneclass_poison_screening as unified


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PAPER14_ROOT / "results/bloodmnist_no_whitening_development_v1"
SEEDS = tuple(range(52, 62))


def main(args: argparse.Namespace) -> None:
    if tuple(args.seeds) != SEEDS:
        raise ValueError("Development check is frozen to already-used seeds 52--61")
    args.output.mkdir(parents=True, exist_ok=True)
    class_data = heldout.collect_classes(args.dataset_root)
    rgb_trigger, _ = baseline.fixed_rgb_trigger(class_data)
    rows, victims = [], []
    for seed in args.seeds:
        print(f"seed={seed}: covariance-free cross-scenario development", flush=True)
        prepared = baseline.preprocess_seed(class_data, seed)
        splits, _ = baseline.seed_splits(prepared, rgb_trigger, seed)
        for scenario in strong.SCENARIOS:
            clean_model, attacked_model, ca, asr = strong.load_scenario(scenario, seed, splits)
            victims.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "balanced_ca": ca,
                    "asr": asr,
                    "eligible": int(ca >= 0.90 and asr >= 0.80),
                }
            )
            views, _, _ = strong.views_for_scenario(
                scenario, clean_model, attacked_model, splits
            )
            groups = views["checkpoint_drift_xz16"]
            row, _ = ablation.custom_crossfit(
                seed,
                "covariance_free_xz16",
                groups["trusted"],
                groups["clean"],
                groups["poison"],
                args.alpha,
                whitening=False,
            )
            row["scenario"] = scenario
            rows.append(row)
    summaries = []
    for scenario in strong.SCENARIOS:
        eligible = {
            int(row["seed"])
            for row in victims
            if row["scenario"] == scenario and row["eligible"]
        }
        selected = [
            row
            for row in rows
            if row["scenario"] == scenario and int(row["seed"]) in eligible
        ]
        tp, fn = sum(row["tp"] for row in selected), sum(row["fn"] for row in selected)
        fp, tn = sum(row["fp"] for row in selected), sum(row["tn"] for row in selected)
        summaries.append(
            {
                "scenario": scenario,
                "eligible_seed_count": len(selected),
                "eligible_seeds": ",".join(map(str, sorted(eligible))),
                "pooled_tpr": tp / (tp + fn),
                "pooled_fpr": fp / (fp + tn),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
            }
        )
    unified.write_csv(args.output / "per_config_metrics.csv", rows)
    unified.write_csv(args.output / "victim_metrics.csv", victims)
    unified.write_csv(args.output / "eligible_summary.csv", summaries)
    unified.json_dump(
        args.output / "manifest.json",
        {
            "experiment": "bloodmnist_no_whitening_development_v1",
            "role": "retrospective_development_not_confirmation",
            "seeds": list(args.seeds),
            "scenarios": list(strong.SCENARIOS),
            "alpha": args.alpha,
            "poison_labels_used_for_scoring": False,
        },
    )
    lines = [
        "# Covariance-free XZ16 retrospective development check",
        "",
        "These already-used seeds may motivate a revised protocol but cannot confirm it.",
        "",
        "| scenario | eligible seeds | pooled TPR | pooled FPR |",
        "|---|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['scenario']} | {row['eligible_seed_count']} | {row['pooled_tpr']:.3f} | {row['pooled_fpr']:.3f} |"
        )
    lines.append("")
    (args.output / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--alpha", type=float, default=0.05)
    main(parser.parse_args())
