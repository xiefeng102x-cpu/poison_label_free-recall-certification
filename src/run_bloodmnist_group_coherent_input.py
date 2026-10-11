#!/usr/bin/env python3
"""BloodMNIST confirmation of cross-fitted coherent input screening."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import run_bloodmnist_unified_oneclass as blood
import run_mnist_group_coherent_drift as coherent
import run_unified_oneclass_poison_screening as unified
import train_bloodmnist_clean_encoder as clean_encoder


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PAPER14_ROOT / "results/bloodmnist_group_coherent_input_v1"


def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    rows, samples = [], []
    for seed in args.seeds:
        checkpoint = torch.load(
            blood.checkpoint_path(seed), map_location="cpu", weights_only=False
        )
        splits, receipt = blood.adaptive.load_data(seed, args.dataset_root, checkpoint)
        adaptive_model = clean_encoder.CleanQNN("local", 0)
        adaptive_model.load_state_dict(checkpoint["state_dict"])
        clean_checkpoint = torch.load(
            blood.clean_checkpoint_path(seed), map_location="cpu", weights_only=False
        )
        clean_model = clean_encoder.CleanQNN("local", 0)
        clean_model.load_state_dict(clean_checkpoint["state_dict"])
        meta = torch.load(
            blood.SOURCE_ROOT
            / f"seed_{seed}/layer_8_grid/t6_vs_t0/pr_0.1"
            / f"poisoned_samples_meta_seed_{seed}_pr_0.1.pt",
            map_location="cpu",
            weights_only=False,
        )
        poison_indices = meta["poison_indices_in_non_target"].numpy().astype(int)
        classical_model = blood.train_classical(
            seed,
            splits["train"]["target"],
            splits["train"]["non_target"],
            splits["archived_poison"],
            poison_indices,
            args.epochs,
        )
        clean_classical = blood.train_clean_classical_companion(
            seed,
            splits["train"]["target"],
            splits["train"]["non_target"],
            classical_model.mean,
            classical_model.scale,
            args.epochs,
        )
        config = {
            "seed": seed,
            "trigger": "t6_vs_t0",
            "non_target": 0,
            "archive_ca": 0.0,
            "archive_asr": 0.0,
        }
        groups = {
            "clean": splits["train"]["target"].astype(float),
            "poison": splits["archived_poison"].astype(float),
        }
        adaptive_z, adaptive_xz = blood.qml_views(adaptive_model, groups)
        clean_z, clean_xz = blood.qml_views(clean_model, groups)
        classical_hidden = {
            name: classical_model.representations(values)[0]
            for name, values in groups.items()
        }
        clean_classical_hidden = {
            name: clean_classical.representations(values)[0]
            for name, values in groups.items()
        }
        classical_margin = {
            name: classical_model.representations(values)[1]
            for name, values in groups.items()
        }
        clean_classical_margin = {
            name: clean_classical.representations(values)[1]
            for name, values in groups.items()
        }
        representations = {
            "coherent_input256": (groups["clean"], groups["poison"]),
            "coherent_classical_hidden_drift16": (
                classical_hidden["clean"] - clean_classical_hidden["clean"],
                classical_hidden["poison"] - clean_classical_hidden["poison"],
            ),
            "coherent_classical_margin_drift1": (
                classical_margin["clean"] - clean_classical_margin["clean"],
                classical_margin["poison"] - clean_classical_margin["poison"],
            ),
            "coherent_qml_z_drift8": (
                adaptive_z["clean"] - clean_z["clean"],
                adaptive_z["poison"] - clean_z["poison"],
            ),
            "coherent_qml_xz_drift16": (
                adaptive_xz["clean"] - clean_xz["clean"],
                adaptive_xz["poison"] - clean_xz["poison"],
            ),
        }
        config_results = {}
        for name, (clean_features, poison_features) in representations.items():
            row, sample_rows = coherent.evaluate_config(
                config, name, clean_features, poison_features, args.alpha
            )
            row["trigger_reconstruction_max_abs_error"] = receipt[
                "triggered_reconstruction_max_abs_error"
            ]
            rows.append(row)
            samples.extend(sample_rows)
            config_results[name] = (row, sample_rows)
            print(
                f"seed={seed} {name}: TPR={row['tpr']:.3f} FPR={row['fpr']:.3f} "
                f"AUC={row['auc_conformal_rank']:.3f}"
            )
        for combined_name, internal_name, internal_dimension in (
            (
                "coherent_bonferroni_input_classical_margin",
                "coherent_classical_margin_drift1",
                1,
            ),
            (
                "coherent_bonferroni_input_qml_z",
                "coherent_qml_z_drift8",
                8,
            ),
        ):
            combined_row, combined_samples = coherent.combine_margin_and_correlator(
                config,
                config_results["coherent_input256"][0],
                config_results["coherent_input256"][1],
                config_results[internal_name][1],
                args.alpha,
                combined_name=combined_name,
                second_dimension=internal_dimension,
            )
            combined_row["trigger_reconstruction_max_abs_error"] = receipt[
                "triggered_reconstruction_max_abs_error"
            ]
            rows.append(combined_row)
            samples.extend(combined_samples)
    summaries = coherent.aggregate(rows, "representation")
    unified.write_csv(args.output / "per_seed_metrics.csv", rows)
    unified.write_csv(args.output / "per_sample_scores.csv", samples)
    unified.write_csv(args.output / "pooled_summary.csv", summaries)
    summary_by_name = {row["representation"]: row for row in summaries}
    report = "\n".join(
        [
            "# BloodMNIST coherent multi-view screening",
            "",
            "The same cross-fitted, label-blind group-direction rule used for MNIST is applied to input and clean-to-backdoor checkpoint-drift views of 600 clean target and 60 archived poisoned training samples on independent seeds 43--46.",
            "",
            "| representation | pooled TPR | pooled FPR | mean AUC | seed pass rate |",
            "|---|---:|---:|---:|---:|",
            *[
                f"| {name} | {row['pooled_tpr']:.3f} | {row['pooled_fpr']:.3f} | {row['mean_auc_conformal_rank']:.3f} | {row['config_pass_rate']:.3f} |"
                for name, row in sorted(summary_by_name.items())
            ],
            "",
            "The direction-estimation half is unlabelled, the scored half does not contribute to its own direction, and thresholds use only known-clean calibration samples.",
            "",
        ]
    )
    (args.output / "EXPERIMENT_REPORT.md").write_text(report, encoding="utf-8")
    unified.json_dump(
        args.output / "manifest.json",
        {
            "experiment": "bloodmnist_group_coherent_input_v1",
            "seeds": args.seeds,
            "clean_target_count_per_seed": 600,
            "poison_count_per_seed": 60,
            "candidate_poison_fraction": 60 / 660,
            "alpha": args.alpha,
            "poison_labels_used_for_fit": False,
            "trusted_clean_reference_required": True,
            "classical_epochs": args.epochs,
        },
    )
    print(
        "pooled: "
        + "; ".join(
            f"{name} TPR={row['pooled_tpr']:.3f} FPR={row['pooled_fpr']:.3f}"
            for name, row in sorted(summary_by_name.items())
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45, 46])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=80)
    main(parser.parse_args())
