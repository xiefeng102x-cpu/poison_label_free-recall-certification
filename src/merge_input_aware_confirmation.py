#!/usr/bin/env python3
"""Merge deterministic Input-Aware confirmation shards into one artifact."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import run_mnist_group_coherent_drift as coherent
import run_unified_oneclass_poison_screening as unified
import verify_top_journal_phase2_pilot as common


PAPER14_ROOT = Path(__file__).resolve().parents[1]
SHARDS = (
    Path("/private/tmp/bloodmnist-ia-confirm-a"),
    Path("/private/tmp/bloodmnist-ia-confirm-b"),
    Path("/private/tmp/bloodmnist-ia-confirm-c"),
)
OUTPUT = PAPER14_ROOT / "results/bloodmnist_input_aware_confirmation_v1"
COUNT_KEYS = {"seed", "dimension", "n_clean", "n_poison", "tn", "fp", "fn", "tp", "config_pass"}
FLOAT_KEYS = {
    "archive_ca",
    "archive_asr",
    "tpr",
    "fpr",
    "auc_crossfit_score",
    "auc_conformal_rank",
}


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric_row(row: dict) -> dict:
    output = dict(row)
    for key in COUNT_KEYS:
        if key in output and str(output[key]).isdigit():
            output[key] = int(output[key])
    for key in FLOAT_KEYS:
        if key in output:
            output[key] = float(output[key])
    return output


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifests = [
        json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
        for shard in SHARDS
    ]
    seeds = sorted(seed for manifest in manifests for seed in manifest["seeds"])
    assert seeds == list(range(52, 62))
    invariant_keys = (
        "classifier_epochs",
        "mask_epochs",
        "attack_epochs",
        "poison_ratio",
        "candidate_count",
        "mask_density",
        "alpha",
        "trusted_reference_split",
        "dataset_npz_sha256",
    )
    for key in invariant_keys:
        assert len({json.dumps(manifest[key], sort_keys=True) for manifest in manifests}) == 1

    metrics = [
        metric_row(row)
        for shard in SHARDS
        for row in csv_rows(shard / "per_config_metrics.csv")
    ]
    samples = [
        row
        for shard in SHARDS
        for row in csv_rows(shard / "per_sample_scores.csv")
    ]
    victims = [
        row for shard in SHARDS for row in csv_rows(shard / "victim_metrics.csv")
    ]
    metrics.sort(key=lambda row: (int(row["seed"]), row["representation"]))
    samples.sort(
        key=lambda row: (
            int(row["seed"]),
            row["representation"],
            int(row["sample_index"]),
        )
    )
    victims.sort(key=lambda row: int(row["seed"]))
    pooled = coherent.aggregate(metrics, "representation")
    victim_map = {int(row["seed"]): row for row in victims}
    eligible_rows = []
    for representation in (
        "coherent_bonferroni_input_cnn_margin",
        "coherent_cnn_hidden_drift64",
    ):
        selected = [
            row
            for row in metrics
            if row["representation"] == representation
            and int(victim_map[int(row["seed"])]["eligible"]) == 1
        ]
        summary = coherent.aggregate(selected, "representation")[0]
        eligible_rows.append(
            {
                "representation": representation,
                "eligible_seed_count": len(selected),
                "eligible_seeds": ",".join(str(row["seed"]) for row in selected),
                **{key: value for key, value in summary.items() if key != "representation"},
            }
        )

    receipts = []
    for shard, manifest in zip(SHARDS, manifests):
        for receipt in manifest["receipts"]:
            seed = int(receipt["seed"])
            attack_source = shard / receipt["checkpoint"]
            attack_target = OUTPUT / "models" / f"seed_{seed}.pt"
            attack_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(attack_source, attack_target)
            assert common.file_sha256(attack_target) == receipt["checkpoint_sha256"]
            clean_source = shard / "models/clean_cnn" / f"seed_{seed}.pt"
            clean_target = OUTPUT / "models/clean_cnn" / f"seed_{seed}.pt"
            clean_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(clean_source, clean_target)
            receipt["checkpoint"] = str(attack_target.relative_to(OUTPUT))
            receipt["clean_checkpoint"] = str(clean_target.relative_to(OUTPUT))
            receipt["clean_checkpoint_sha256"] = common.file_sha256(clean_target)
            receipts.append(receipt)
    receipts.sort(key=lambda row: int(row["seed"]))

    unified.write_csv(OUTPUT / "per_config_metrics.csv", metrics)
    unified.write_csv(OUTPUT / "per_sample_scores.csv", samples)
    unified.write_csv(OUTPUT / "victim_metrics.csv", victims)
    unified.write_csv(OUTPUT / "pooled_summary.csv", pooled)
    unified.write_csv(OUTPUT / "eligible_summary.csv", eligible_rows)
    first = manifests[0]
    unified.json_dump(
        OUTPUT / "manifest.json",
        {
            "experiment": "bloodmnist_input_aware_confirmation_v1",
            "role": "fresh_counterexample_confirmation",
            "protocol": "results/top_journal_program_v1/phase2_counterexample_confirmation_protocol.json",
            "attack_label": first["attack_label"],
            "seeds": seeds,
            "classifier_epochs": first["classifier_epochs"],
            "mask_epochs": first["mask_epochs"],
            "attack_epochs": first["attack_epochs"],
            "poison_ratio": first["poison_ratio"],
            "candidate_count": first["candidate_count"],
            "mask_density": first["mask_density"],
            "alpha": first["alpha"],
            "trusted_reference_split": first["trusted_reference_split"],
            "victim_eligibility": first["victim_eligibility"],
            "dataset_root": first["dataset_root"],
            "dataset_npz_sha256": first["dataset_npz_sha256"],
            "execution": "three isolated shards merged only after per-seed completion",
            "receipts": receipts,
            "elapsed_seconds_sum": sum(
                float(manifest["elapsed_seconds"]) for manifest in manifests
            ),
        },
    )

    lines = [
        "# BloodMNIST Input-Aware-style fresh-seed confirmation",
        "",
        "Seeds 52--61 use the frozen 25-mask/60-joint-epoch attack and the unchanged input+CNN-margin detector. Only CA/ASR-eligible victims enter detector conclusions.",
        "",
        "| seed | balanced CA | ASR | eligible | mask mean | pattern variability |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in victims:
        lines.append(
            f"| {row['seed']} | {float(row['balanced_ca']):.3f} | "
            f"{float(row['asr']):.3f} | {row['eligible']} | "
            f"{float(row['mask_mean']):.4f} | "
            f"{float(row['pattern_samplewise_std_mean']):.4f} |"
        )
    lines.extend(
        [
            "",
            "| endpoint | eligible seeds | pooled TPR | pooled FPR |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in eligible_rows:
        lines.append(
            f"| {row['representation']} | {row['eligible_seed_count']} | "
            f"{row['pooled_tpr']:.3f} | {row['pooled_fpr']:.3f} |"
        )
    (OUTPUT / "EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"seeds": seeds, "eligible": eligible_rows}, indent=2))


if __name__ == "__main__":
    main()
