#!/usr/bin/env python3
"""Fresh seeds 62--71 confirmation of covariance-free CF-CMD-U."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_group_coherent_10seeds as baseline
import run_bloodmnist_mechanism_ablations as ablation
import run_bloodmnist_whitebox_adaptive_pilot as whitebox
import run_unified_oneclass_poison_screening as unified


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PAPER14_ROOT / "results/bloodmnist_covariance_free_confirmation_v1"
PROTOCOL = PAPER14_ROOT / "results/top_journal_program_v1/covariance_free_confirmation_protocol.json"
SEEDS = tuple(range(62, 72))
SCENARIOS = ("fixed_pca_additive", "whitebox_representation_matching")


def summarize(rows: list[dict], victims: list[dict], draws: int) -> list[dict]:
    eligible = {(row["scenario"], int(row["seed"])) for row in victims if row["eligible"]}
    rng = np.random.default_rng(20260928)
    output = []
    for scenario in SCENARIOS:
        selected = sorted(
            [row for row in rows if row["scenario"] == scenario and (scenario, int(row["seed"])) in eligible],
            key=lambda row: int(row["seed"]),
        )
        if not selected:
            output.append({"scenario": scenario, "eligible_seed_count": 0, "status": "NO_ELIGIBLE_VICTIM"})
            continue
        tp, fn = sum(row["tp"] for row in selected), sum(row["fn"] for row in selected)
        fp, tn = sum(row["fp"] for row in selected), sum(row["tn"] for row in selected)
        tpr, fpr = np.asarray([row["tpr"] for row in selected]), np.asarray([row["fpr"] for row in selected])
        indices = rng.integers(0, len(selected), size=(draws, len(selected)))
        pooled_tpr, pooled_fpr = tp / (tp + fn), fp / (fp + tn)
        if scenario == "whitebox_representation_matching" and len(selected) < 5:
            status = "INCONCLUSIVE_ATTACK_STABILITY_BELOW_5_ELIGIBLE"
        else:
            status = (
                "DETECTOR_GATE_PASSES"
                if pooled_tpr >= 0.90 and pooled_fpr <= 0.05
                else "DETECTOR_GATE_FAILS"
            )
        output.append(
            {
                "scenario": scenario,
                "eligible_seed_count": len(selected),
                "eligible_seeds": ",".join(str(row["seed"]) for row in selected),
                "pooled_tpr": pooled_tpr,
                "pooled_fpr": pooled_fpr,
                "tpr_ci_low": float(np.quantile(tpr[indices].mean(1), 0.025)),
                "tpr_ci_high": float(np.quantile(tpr[indices].mean(1), 0.975)),
                "fpr_ci_low": float(np.quantile(fpr[indices].mean(1), 0.025)),
                "fpr_ci_high": float(np.quantile(fpr[indices].mean(1), 0.975)),
                "status": status,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
            }
        )
    return output


def evaluate(
    scenario: str,
    seed: int,
    clean_model,
    attacked_model,
    splits: dict,
    alpha: float,
) -> tuple[dict, list[dict]]:
    trusted = whitebox.checkpoint_drift(clean_model, attacked_model, splits["val"]["target"])
    clean = whitebox.checkpoint_drift(clean_model, attacked_model, splits["train"]["target"])
    poison = whitebox.checkpoint_drift(clean_model, attacked_model, splits["archived_poison"])
    row, samples = ablation.custom_crossfit(
        seed,
        "covariance_free_xz16",
        trusted,
        clean,
        poison,
        alpha,
        whitening=False,
    )
    row["scenario"] = scenario
    for sample in samples:
        sample["scenario"] = scenario
    return row, samples


def main(args: argparse.Namespace) -> None:
    if tuple(args.seeds) != SEEDS:
        raise ValueError("CF-CMD-U confirmation is frozen to seeds 62--71")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["fresh_seeds"] != list(SEEDS):
        raise RuntimeError("Protocol seed mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    fixed_root = args.output / "fixed_local"
    class_data = heldout.collect_classes(args.dataset_root)
    rgb_trigger, rgb_receipt = baseline.fixed_rgb_trigger(class_data)
    rows, samples, victims, receipts, adaptive_histories = [], [], [], [], []
    for seed in args.seeds:
        print(f"seed={seed}: fresh CF-CMD-U fixed and white-box scenarios", flush=True)
        prepared = baseline.preprocess_seed(class_data, seed)
        splits, trigger_receipt = baseline.seed_splits(prepared, rgb_trigger, seed)
        clean_model, clean_record = baseline.load_or_train_clean(
            fixed_root, splits, seed, args.clean_epochs
        )
        fixed_model, fixed_record = baseline.load_or_adapt(
            fixed_root, clean_model, splits, seed, args.fixed_adaptive_epochs
        )
        fixed_ca, fixed_asr = whitebox.balanced_ca_and_asr(
            fixed_model,
            splits["test"]["target"],
            splits["test"]["non_target"],
            splits["test"]["triggered_non_target"],
        )
        fixed_row, fixed_samples = evaluate(
            "fixed_pca_additive", seed, clean_model, fixed_model, splits, args.alpha
        )
        rows.append(fixed_row); samples.extend(fixed_samples)
        victims.append(
            {
                "scenario": "fixed_pca_additive",
                "seed": seed,
                "balanced_ca": fixed_ca,
                "asr": fixed_asr,
                "eligible": int(fixed_ca >= 0.90 and fixed_asr >= 0.80),
                "selected_epoch": fixed_record["selected_epoch"],
            }
        )

        seed_data = whitebox.prepare_seed(
            class_data, rgb_trigger, seed, clean_root=fixed_root
        )
        adaptive_result = whitebox.run_configuration(
            seed_data,
            seed=seed,
            mode="checkpoint_drift_representation_matching",
            evasion_weight=10.0,
            args=args,
        )
        adaptive_model = adaptive_result["model"]
        adaptive_ca, adaptive_asr = whitebox.balanced_ca_and_asr(
            adaptive_model,
            splits["test"]["target"],
            splits["test"]["non_target"],
            splits["test"]["triggered_non_target"],
        )
        adaptive_row, adaptive_samples = evaluate(
            "whitebox_representation_matching",
            seed,
            clean_model,
            adaptive_model,
            splits,
            args.alpha,
        )
        rows.append(adaptive_row); samples.extend(adaptive_samples)
        victims.append(
            {
                "scenario": "whitebox_representation_matching",
                "seed": seed,
                "balanced_ca": adaptive_ca,
                "asr": adaptive_asr,
                "eligible": int(adaptive_ca >= 0.90 and adaptive_asr >= 0.80),
                "selected_epoch": adaptive_result["selected"]["epoch"],
            }
        )
        checkpoint = args.output / "models" / "whitebox" / f"seed_{seed}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": adaptive_model.state_dict(),
                "seed": seed,
                "selection_detector": "covariance_free_xz16",
                "selected_epoch": adaptive_result["selected"]["epoch"],
            },
            checkpoint,
        )
        adaptive_histories.append(
            {"seed": seed, "selected": adaptive_result["selected"], "history": adaptive_result["history"]}
        )
        receipts.append(
            {
                "seed": seed,
                "clean_checkpoint_sha256": whitebox.sha256_file(
                    fixed_root / "models" / f"seed_{seed}_clean.pt"
                ),
                "fixed_checkpoint_sha256": whitebox.sha256_file(
                    fixed_root / "models" / f"seed_{seed}_adaptive.pt"
                ),
                "whitebox_checkpoint_sha256": whitebox.sha256_file(checkpoint),
                "poison_sha256_float32": whitebox.sha256_array(splits["archived_poison"]),
                "trigger_receipt": {
                    **trigger_receipt,
                    "poison_indices": np.asarray(trigger_receipt["poison_indices"], dtype=int).tolist(),
                },
            }
        )
        print(
            f"seed={seed}: fixed CA/ASR/TPR={fixed_ca:.3f}/{fixed_asr:.3f}/{fixed_row['tpr']:.3f}; "
            f"whitebox={adaptive_ca:.3f}/{adaptive_asr:.3f}/{adaptive_row['tpr']:.3f}",
            flush=True,
        )

    summaries = summarize(rows, victims, args.bootstrap_draws)
    unified.write_csv(args.output / "per_config_metrics.csv", rows)
    unified.write_csv(args.output / "per_sample_scores.csv", samples)
    unified.write_csv(args.output / "victim_metrics.csv", victims)
    unified.write_csv(args.output / "eligible_summary.csv", summaries)
    unified.json_dump(args.output / "adaptive_histories.json", adaptive_histories)
    unified.json_dump(
        args.output / "manifest.json",
        {
            "experiment": "bloodmnist_covariance_free_confirmation_v1",
            "protocol": str(PROTOCOL),
            "protocol_sha256": whitebox.sha256_file(PROTOCOL),
            "dataset_root": str(args.dataset_root),
            "dataset_npz_sha256": whitebox.sha256_file(args.dataset_root / "bloodmnist.npz"),
            "seeds": list(args.seeds),
            "scenarios": list(SCENARIOS),
            "selection_detector": args.selection_detector,
            "alpha": args.alpha,
            "bootstrap_draws": args.bootstrap_draws,
            "rgb_trigger_receipt": rgb_receipt,
            "receipts": receipts,
        },
    )
    lines = [
        "# CF-CMD-U fresh-seed confirmation (seeds 62--71)",
        "",
        "The covariance-free rule was frozen after the 52--61 ablation; none of those seeds are reused here.",
        "",
        "| scenario | eligible seeds | pooled TPR | TPR 95% seed CI | pooled FPR | FPR 95% seed CI | decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        if row["eligible_seed_count"]:
            lines.append(
                f"| {row['scenario']} | {row['eligible_seed_count']} | {row['pooled_tpr']:.3f} | [{row['tpr_ci_low']:.3f}, {row['tpr_ci_high']:.3f}] | {row['pooled_fpr']:.3f} | [{row['fpr_ci_low']:.3f}, {row['fpr_ci_high']:.3f}] | {row['status']} |"
            )
        else:
            lines.append(f"| {row['scenario']} | 0 | — | — | — | — | {row['status']} |")
    lines.extend(["", "Residual-reupload is outside this revised rule's claim and remains a confirmed architecture boundary.", ""])
    (args.output / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--clean-epochs", type=int, default=40)
    parser.add_argument("--fixed-adaptive-epochs", type=int, default=5)
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
    parser.add_argument("--selection-detector", default="covariance_free_xz16")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    main(parser.parse_args())
