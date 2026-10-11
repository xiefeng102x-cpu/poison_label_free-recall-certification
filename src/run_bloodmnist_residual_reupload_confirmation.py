#!/usr/bin/env python3
"""Fresh confirmation for the residual entangling-reupload architecture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_cross_attack_pilot as cross
import run_bloodmnist_group_coherent_10seeds as baseline
import run_bloodmnist_residual_reupload_pilot as residual
import run_bloodmnist_whitebox_adaptive_pilot as whitebox
import run_mnist_group_coherent_drift as coherent
import run_unified_oneclass_poison_screening as unified


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PAPER14_ROOT / "results/bloodmnist_residual_reupload_confirmation_v1"
PROTOCOL = PAPER14_ROOT / "results/top_journal_program_v1/phase4_residual_reupload_confirmation_protocol.json"
LOCAL_ROOT = PAPER14_ROOT / "results/bloodmnist_xz16_fresh_confirmation_v1"
SEEDS = tuple(range(52, 62))


def summarize(rows: list[dict], victims: list[dict], draws: int) -> list[dict]:
    eligible = {int(row["seed"]) for row in victims if row["test_eligible"]}
    rng = np.random.default_rng(20260928)
    output = []
    for representation in (residual.INPUT_NAME, residual.XZ_NAME, residual.PRIMARY_NAME):
        selected = sorted(
            [row for row in rows if row["representation"] == representation and int(row["seed"]) in eligible],
            key=lambda row: int(row["seed"]),
        )
        if not selected:
            output.append({"representation": representation, "eligible_seed_count": 0, "eligible_seeds": "", "status": "NO_ELIGIBLE_VICTIM"})
            continue
        aggregate = coherent.aggregate(selected, "representation")[0]
        tpr = np.asarray([row["tpr"] for row in selected])
        fpr = np.asarray([row["fpr"] for row in selected])
        indices = rng.integers(0, len(selected), size=(draws, len(selected)))
        tpr_draws, fpr_draws = tpr[indices].mean(1), fpr[indices].mean(1)
        status = "SECONDARY_ENDPOINT"
        if representation == residual.PRIMARY_NAME:
            status = (
                "DETECTOR_GATE_PASSES"
                if aggregate["pooled_tpr"] >= 0.90 and aggregate["pooled_fpr"] <= 0.05
                else "CROSS_ARCHITECTURE_FAILURE_CONFIRMED"
            )
        output.append(
            {
                "representation": representation,
                "eligible_seed_count": len(selected),
                "eligible_seeds": ",".join(str(row["seed"]) for row in selected),
                "status": status,
                **{key: value for key, value in aggregate.items() if key != "representation"},
                "tpr_seed_bootstrap_ci_low": float(np.quantile(tpr_draws, 0.025)),
                "tpr_seed_bootstrap_ci_high": float(np.quantile(tpr_draws, 0.975)),
                "fpr_seed_bootstrap_ci_low": float(np.quantile(fpr_draws, 0.025)),
                "fpr_seed_bootstrap_ci_high": float(np.quantile(fpr_draws, 0.975)),
            }
        )
    return output


def main(args: argparse.Namespace) -> None:
    if tuple(args.seeds) != SEEDS:
        raise ValueError("Residual-reupload confirmation is frozen to seeds 52--61")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["fresh_seeds"] != list(SEEDS):
        raise RuntimeError("Protocol seed mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    class_data = heldout.collect_classes(args.dataset_root)
    rgb_trigger, rgb_receipt = baseline.fixed_rgb_trigger(class_data)
    rows, samples, victims, receipts = [], [], [], []
    for seed in args.seeds:
        print(f"seed={seed}: residual-reupload fresh confirmation", flush=True)
        prepared = baseline.preprocess_seed(class_data, seed)
        splits, trigger_receipt = baseline.seed_splits(prepared, rgb_trigger, seed)
        local_model, _ = cross.load_reference_clean_model(seed, LOCAL_ROOT)
        clean_model, clean_record = residual.load_or_train_clean(args.output, local_model, splits, seed)
        attacked_model, attack_record = residual.load_or_adapt(args.output, clean_model, splits, seed)
        test_ca, test_asr = whitebox.balanced_ca_and_asr(
            attacked_model,
            splits["test"]["target"],
            splits["test"]["non_target"],
            splits["test"]["triggered_non_target"],
        )
        selected_rows, selected_samples = residual.detector(
            seed, clean_model, attacked_model, splits, args.alpha
        )
        for row in selected_rows:
            row["archive_ca"], row["archive_asr"] = test_ca, test_asr
        rows.extend(selected_rows)
        samples.extend(selected_samples)
        victim = {
            "seed": seed,
            "clean_validation_balanced_ca": clean_record["validation_balanced_ca"],
            "quantum_classifier_weight_fraction": clean_record["quantum_classifier_weight_fraction"],
            "mean_abs_quantum_branch_logit_effect": clean_record["mean_abs_quantum_branch_logit_effect"],
            "adaptive_selected_epoch": attack_record["selected_epoch"],
            "test_balanced_ca": test_ca,
            "test_asr": test_asr,
            "test_eligible": int(test_ca >= 0.90 and test_asr >= 0.80),
        }
        victims.append(victim)
        receipts.append(
            {
                "seed": seed,
                "clean_checkpoint_sha256": clean_record["checkpoint_sha256"],
                "adaptive_checkpoint_sha256": attack_record["checkpoint_sha256"],
                "poison_sha256_float32": whitebox.sha256_array(splits["archived_poison"]),
                "trigger_receipt": {
                    **trigger_receipt,
                    "poison_indices": np.asarray(trigger_receipt["poison_indices"], dtype=int).tolist(),
                },
            }
        )
        print(
            f"seed={seed}: CA/ASR={test_ca:.3f}/{test_asr:.3f}; primary={selected_rows[-1]['tpr']:.3f}/{selected_rows[-1]['fpr']:.3f}",
            flush=True,
        )

    summaries = summarize(rows, victims, args.bootstrap_draws)
    unified.write_csv(args.output / "per_config_metrics.csv", rows)
    unified.write_csv(args.output / "per_sample_scores.csv", samples)
    unified.write_csv(args.output / "victim_metrics.csv", victims)
    unified.write_csv(args.output / "eligible_summary.csv", summaries)
    unified.json_dump(
        args.output / "manifest.json",
        {
            "experiment": "bloodmnist_residual_reupload_confirmation_v1",
            "protocol": str(PROTOCOL),
            "protocol_sha256": whitebox.sha256_file(PROTOCOL),
            "dataset_root": str(args.dataset_root),
            "dataset_npz_sha256": whitebox.sha256_file(args.dataset_root / "bloodmnist.npz"),
            "local_clean_root": str(LOCAL_ROOT),
            "seeds": list(args.seeds),
            "alpha": args.alpha,
            "bootstrap_draws": args.bootstrap_draws,
            "rgb_trigger_receipt": rgb_receipt,
            "receipts": receipts,
        },
    )
    lines = [
        "# BloodMNIST residual-reupload fresh-seed confirmation",
        "",
        "| endpoint | eligible seeds | pooled TPR | TPR 95% seed CI | pooled FPR | FPR 95% seed CI | decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        if row["eligible_seed_count"]:
            lines.append(
                f"| {row['representation']} | {row['eligible_seed_count']} | {row['pooled_tpr']:.3f} | [{row['tpr_seed_bootstrap_ci_low']:.3f}, {row['tpr_seed_bootstrap_ci_high']:.3f}] | {row['pooled_fpr']:.3f} | [{row['fpr_seed_bootstrap_ci_low']:.3f}, {row['fpr_seed_bootstrap_ci_high']:.3f}] | {row['status']} |"
            )
        else:
            lines.append(f"| {row['representation']} | 0 | — | — | — | — | {row['status']} |")
    eligible_count = sum(int(row["test_eligible"]) for row in victims)
    lines.extend(
        [
            "",
            f"Victim eligibility: {eligible_count}/10. Mean quantum-branch classifier weight fraction: {np.mean([row['quantum_classifier_weight_fraction'] for row in victims]):.3f}.",
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
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    main(parser.parse_args())
