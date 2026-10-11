#!/usr/bin/env python3
"""Fresh ten-seed confirmation of the frozen BloodMNIST white-box attacks."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_group_coherent_10seeds as baseline
import run_bloodmnist_whitebox_adaptive_pilot as pilot
import run_mnist_group_coherent_drift as coherent
import run_unified_oneclass_poison_screening as unified


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PAPER14_ROOT / "results/bloodmnist_whitebox_adaptive_confirmation_v1"
PROTOCOL = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/phase3_adaptive_confirmation_protocol.json"
)
FRESH_CLEAN_ROOT = PAPER14_ROOT / "results/bloodmnist_xz16_fresh_confirmation_v1"
SEEDS = tuple(range(52, 62))


def eligible_summaries(rows: list[dict], victims: list[dict], draws: int) -> list[dict]:
    eligible = {(row["mode"], int(row["seed"])) for row in victims if row["test_eligible"]}
    rng = np.random.default_rng(20260928)
    output = []
    for mode in pilot.MODES:
        for representation in (pilot.INPUT_NAME, pilot.XZ_NAME, pilot.PRIMARY_NAME):
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["trigger"] == mode
                    and row["representation"] == representation
                    and (mode, int(row["seed"])) in eligible
                ],
                key=lambda row: int(row["seed"]),
            )
            seeds = [int(row["seed"]) for row in selected]
            if not selected:
                output.append(
                    {
                        "mode": mode,
                        "representation": representation,
                        "eligible_seed_count": 0,
                        "eligible_seeds": "",
                        "status": "NO_ELIGIBLE_VICTIM",
                    }
                )
                continue
            summary = coherent.aggregate(selected, "representation")[0]
            tpr = np.asarray([row["tpr"] for row in selected], dtype=float)
            fpr = np.asarray([row["fpr"] for row in selected], dtype=float)
            indices = rng.integers(0, len(selected), size=(draws, len(selected)))
            tpr_draws = tpr[indices].mean(axis=1)
            fpr_draws = fpr[indices].mean(axis=1)
            if representation != pilot.PRIMARY_NAME:
                status = "SECONDARY_ENDPOINT"
            elif len(selected) < 5:
                status = "INSUFFICIENT_ELIGIBLE_VICTIMS"
            elif summary["pooled_tpr"] < 0.90:
                status = (
                    "STRONG_ADAPTIVE_EVASION_CONFIRMED"
                    if float(np.quantile(tpr_draws, 0.975)) < 0.90
                    else "POINT_ADAPTIVE_EVASION_CONFIRMED"
                )
            elif summary["pooled_fpr"] > 0.05:
                status = "DETECTOR_GATE_FAILS_ON_FPR"
            else:
                status = "DETECTOR_GATE_PASSES"
            output.append(
                {
                    "mode": mode,
                    "representation": representation,
                    "eligible_seed_count": len(seeds),
                    "eligible_seeds": ",".join(map(str, seeds)),
                    "status": status,
                    **{key: value for key, value in summary.items() if key != "representation"},
                    "mean_seed_tpr": float(tpr.mean()),
                    "tpr_seed_bootstrap_ci_low": float(np.quantile(tpr_draws, 0.025)),
                    "tpr_seed_bootstrap_ci_high": float(np.quantile(tpr_draws, 0.975)),
                    "mean_seed_fpr": float(fpr.mean()),
                    "fpr_seed_bootstrap_ci_low": float(np.quantile(fpr_draws, 0.025)),
                    "fpr_seed_bootstrap_ci_high": float(np.quantile(fpr_draws, 0.975)),
                }
            )
    return output


def incremental_joint_summary(rows: list[dict], victims: list[dict]) -> dict:
    eligible_by_mode = defaultdict(set)
    for row in victims:
        if row["test_eligible"]:
            eligible_by_mode[row["mode"]].add(int(row["seed"]))
    common = sorted(
        eligible_by_mode["checkpoint_drift_representation_matching"]
        & eligible_by_mode["joint_input_internal_detector_evasion"]
    )
    lookup = {
        (row["trigger"], int(row["seed"])): row
        for row in rows
        if row["representation"] == pilot.PRIMARY_NAME
    }
    differences = [
        lookup[("joint_input_internal_detector_evasion", seed)]["tpr"]
        - lookup[("checkpoint_drift_representation_matching", seed)]["tpr"]
        for seed in common
    ]
    return {
        "common_eligible_seeds": common,
        "n_common": len(common),
        "mean_joint_minus_representation_tpr": (
            float(np.mean(differences)) if differences else None
        ),
        "max_abs_seed_difference": (
            float(np.max(np.abs(differences))) if differences else None
        ),
        "interpretation": (
            "NO_INCREMENTAL_EVASION_BENEFIT"
            if differences and float(np.mean(differences)) >= -0.01
            else "INCREMENTAL_EVASION_BENEFIT"
        ),
    }


def write_report(
    output: Path,
    victims: list[dict],
    summaries: list[dict],
    incremental: dict,
    elapsed: float,
) -> None:
    lines = [
        "# BloodMNIST white-box adaptive attack fresh-seed confirmation",
        "",
        "All attack weights and optimizer settings were frozen after the 42/46/51 pilot. Seeds 52--61 are contiguous untouched confirmation seeds. Test data are used once per selected checkpoint for victim eligibility only.",
        "",
        "## Victim feasibility",
        "",
        "| attack | eligible / 10 | mean test CA | mean test ASR |",
        "|---|---:|---:|---:|",
    ]
    for mode in pilot.MODES:
        selected = [row for row in victims if row["mode"] == mode]
        eligible = [row for row in selected if row["test_eligible"]]
        lines.append(
            f"| {mode} | {len(eligible)} | "
            f"{np.mean([row['test_balanced_ca'] for row in selected]):.3f} | "
            f"{np.mean([row['test_asr'] for row in selected]):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Eligible-victim detector evidence",
            "",
            "| attack | endpoint | n | pooled TPR | TPR 95% seed CI | pooled FPR | FPR 95% seed CI | decision |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summaries:
        if row["eligible_seed_count"] == 0:
            lines.append(
                f"| {row['mode']} | {row['representation']} | 0 | — | — | — | — | {row['status']} |"
            )
        else:
            lines.append(
                f"| {row['mode']} | {row['representation']} | {row['eligible_seed_count']} | "
                f"{row['pooled_tpr']:.3f} | [{row['tpr_seed_bootstrap_ci_low']:.3f}, {row['tpr_seed_bootstrap_ci_high']:.3f}] | "
                f"{row['pooled_fpr']:.3f} | [{row['fpr_seed_bootstrap_ci_low']:.3f}, {row['fpr_seed_bootstrap_ci_high']:.3f}] | {row['status']} |"
            )
    lines.extend(
        [
            "",
            "## Incremental joint-attack check",
            "",
            f"On {incremental['n_common']} commonly eligible seeds, joint minus representation-matching primary TPR is {incremental['mean_joint_minus_representation_tpr']}. Decision: **{incremental['interpretation']}**. The joint attack is not counted as an additional positive mechanism when it does not improve evasion.",
            "",
            f"Runtime: {elapsed:.1f} seconds.",
            "",
        ]
    )
    (output / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    if tuple(args.seeds) != SEEDS:
        raise ValueError("Confirmation is frozen to seeds 52--61")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["fresh_seeds"] != list(SEEDS):
        raise RuntimeError("Protocol seed mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    class_data = heldout.collect_classes(args.dataset_root)
    rgb_trigger, rgb_receipt = baseline.fixed_rgb_trigger(class_data)
    rows, samples, victims, receipts, histories = [], [], [], [], []

    for seed in args.seeds:
        print(f"seed={seed}: preprocessing and loading frozen clean checkpoint", flush=True)
        seed_data = pilot.prepare_seed(
            class_data, rgb_trigger, seed, clean_root=args.clean_qml_root
        )
        for mode in pilot.MODES:
            result = pilot.run_configuration(
                seed_data,
                seed=seed,
                mode=mode,
                evasion_weight=10.0,
                args=args,
            )
            selected_rows, selected_samples, victim, receipt = pilot.save_selected(
                args.output, result, seed_data, args
            )
            rows.extend(selected_rows)
            samples.extend(selected_samples)
            victims.append(victim)
            receipts.append(receipt)
            histories.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "evasion_weight": 10.0,
                    "history": result["history"],
                }
            )
            print(
                f"seed={seed} mode={mode}: CA/ASR={victim['test_balanced_ca']:.3f}/"
                f"{victim['test_asr']:.3f}; primary={selected_rows[-1]['tpr']:.3f}/"
                f"{selected_rows[-1]['fpr']:.3f}",
                flush=True,
            )

    summaries = eligible_summaries(rows, victims, args.bootstrap_draws)
    incremental = incremental_joint_summary(rows, victims)
    unified.write_csv(args.output / "per_config_metrics.csv", rows)
    unified.write_csv(args.output / "per_sample_scores.csv", samples)
    unified.write_csv(args.output / "victim_metrics.csv", victims)
    unified.write_csv(args.output / "eligible_summary.csv", summaries)
    unified.json_dump(args.output / "training_histories.json", histories)
    unified.json_dump(args.output / "incremental_joint_summary.json", incremental)
    unified.json_dump(
        args.output / "manifest.json",
        {
            "experiment": "bloodmnist_whitebox_adaptive_confirmation_v1",
            "role": "fresh_selective_confirmation",
            "protocol": str(PROTOCOL),
            "protocol_sha256": pilot.sha256_file(PROTOCOL),
            "dataset_root": str(args.dataset_root),
            "dataset_npz_sha256": pilot.sha256_file(args.dataset_root / "bloodmnist.npz"),
            "clean_qml_root": str(args.clean_qml_root),
            "seeds": list(args.seeds),
            "modes": list(pilot.MODES),
            "evasion_weight": 10.0,
            "epochs": args.epochs,
            "alpha": args.alpha,
            "victim_gate": {"balanced_ca_min": 0.90, "asr_min": 0.80},
            "detector_gate": {"pooled_tpr_min": 0.90, "pooled_fpr_max": 0.05},
            "bootstrap_draws": args.bootstrap_draws,
            "rgb_trigger_receipt": rgb_receipt,
            "receipts": receipts,
            "elapsed_seconds": time.time() - started,
        },
    )
    write_report(args.output, victims, summaries, incremental, time.time() - started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clean-qml-root", type=Path, default=FRESH_CLEAN_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--trigger-learning-rate", type=float, default=0.002)
    parser.add_argument("--backdoor-weight", type=float, default=10.0)
    parser.add_argument("--distill-weight", type=float, default=20.0)
    parser.add_argument("--anchor-weight", type=float, default=0.001)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--direction-amplitude", type=float, default=0.2)
    parser.add_argument("--trigger-linf", type=float, default=0.8)
    parser.add_argument("--trigger-correction-linf", type=float, default=0.2)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    main(parser.parse_args())
