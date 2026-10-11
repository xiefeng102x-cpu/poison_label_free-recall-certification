#!/usr/bin/env python3
"""Replay the Input-Aware-style fresh counterexample confirmation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_group_coherent_10seeds as baseline
import run_bloodmnist_input_aware_pilot as input_aware
import verify_top_journal_input_aware_pilot as pilot_verify
import verify_top_journal_phase2_pilot as common


PAPER14_ROOT = Path(__file__).resolve().parents[1]
ROOT = PAPER14_ROOT / "results/bloodmnist_input_aware_confirmation_v1"
PROGRAM_ROOT = PAPER14_ROOT / "results/top_journal_program_v1"
OUTPUT = PROGRAM_ROOT / "input_aware_confirmation_receipt.json"
SEEDS = tuple(range(52, 62))
PRIMARY = "coherent_bonferroni_input_cnn_margin"
INTERNAL = "coherent_cnn_hidden_drift64"


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return abs(left - right) <= tolerance


def bootstrap(selected: list[dict]) -> dict:
    counts = np.asarray(
        [[int(row[key]) for key in ("tp", "fn", "fp", "tn")] for row in selected]
    )
    rng = np.random.default_rng(20260928)
    indices = rng.integers(0, len(selected), size=(10000, len(selected)))
    sampled = counts[indices].sum(axis=1)
    tpr = sampled[:, 0] / (sampled[:, 0] + sampled[:, 1])
    fpr = sampled[:, 2] / (sampled[:, 2] + sampled[:, 3])
    return {
        "tpr_ci_low": float(np.quantile(tpr, 0.025)),
        "tpr_ci_high": float(np.quantile(tpr, 0.975)),
        "fpr_ci_low": float(np.quantile(fpr, 0.025)),
        "fpr_ci_high": float(np.quantile(fpr, 0.975)),
    }


def main() -> None:
    protocol = json.loads(
        (PROGRAM_ROOT / "phase2_counterexample_confirmation_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert protocol["frozen_before_execution"] is True
    assert protocol["seeds"] == list(SEEDS)
    assert manifest["role"] == "fresh_counterexample_confirmation"
    assert manifest["seeds"] == list(SEEDS)
    assert manifest["mask_epochs"] == 25
    assert manifest["attack_epochs"] == 60
    assert manifest["poison_ratio"] == 0.1
    assert manifest["candidate_count"] == 60
    assert manifest["mask_density"] == 0.032
    dataset_root = Path(manifest["dataset_root"])
    assert common.file_sha256(dataset_root / "bloodmnist.npz") == manifest[
        "dataset_npz_sha256"
    ]

    metrics = rows(ROOT / "per_config_metrics.csv")
    samples = rows(ROOT / "per_sample_scores.csv")
    victims = rows(ROOT / "victim_metrics.csv")
    eligible = {row["representation"]: row for row in rows(ROOT / "eligible_summary.csv")}
    assert len(metrics) == 40
    assert len(samples) == 26400
    assert len(victims) == 10
    assert len(eligible) == 2

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in samples:
        assert int(row["flagged"]) == (float(row["p_value"]) <= float(row["alpha"]))
        grouped[(row["seed"], row["representation"])].append(row)
    for metric in metrics:
        selected = grouped[(metric["seed"], metric["representation"])]
        clean = [row for row in selected if int(row["label"]) == 0]
        poison = [row for row in selected if int(row["label"]) == 1]
        fp = sum(int(row["flagged"]) for row in clean)
        tp = sum(int(row["flagged"]) for row in poison)
        assert (tp, len(poison) - tp, fp, len(clean) - fp) == tuple(
            int(metric[key]) for key in ("tp", "fn", "fp", "tn")
        )

    victim_map = {int(row["seed"]): row for row in victims}
    eligible_seeds = [
        seed
        for seed, row in victim_map.items()
        if float(row["balanced_ca"]) >= 0.90 and float(row["asr"]) >= 0.80
    ]
    assert eligible_seeds == [53, 56, 57, 58, 61]
    assert all(int(row["eligible"]) == (int(row["seed"]) in eligible_seeds) for row in victims)
    primary = eligible[PRIMARY]
    internal = eligible[INTERNAL]
    assert int(primary["eligible_seed_count"]) == 5
    assert primary["eligible_seeds"] == "53,56,57,58,61"
    assert tuple(int(primary[key]) for key in ("tp", "fn", "fp", "tn")) == (
        284,
        16,
        175,
        2825,
    )
    assert close(float(primary["pooled_tpr"]), 284 / 300)
    assert close(float(primary["pooled_fpr"]), 175 / 3000)
    assert float(primary["pooled_tpr"]) >= 0.90
    assert float(primary["pooled_fpr"]) > 0.05
    assert close(float(internal["pooled_tpr"]), 1.0)
    assert close(float(internal["pooled_fpr"]), 143 / 3000)
    assert float(internal["pooled_fpr"]) <= 0.05

    class_data = heldout.collect_classes(dataset_root)
    fingerprints = 0
    for receipt in manifest["receipts"]:
        checkpoint = ROOT / receipt["checkpoint"]
        clean_checkpoint = ROOT / receipt["clean_checkpoint"]
        assert common.file_sha256(checkpoint) == receipt["checkpoint_sha256"]
        assert common.file_sha256(clean_checkpoint) == receipt[
            "clean_checkpoint_sha256"
        ]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        pattern_generator = input_aware.TinyGenerator(3)
        mask_generator = input_aware.TinyGenerator(1)
        pattern_generator.load_state_dict(payload["pattern_generator"])
        mask_generator.load_state_dict(payload["mask_generator"])
        pattern_generator.eval()
        mask_generator.eval()
        seed = int(receipt["seed"])
        prepared = baseline.preprocess_seed(class_data, seed)
        rng = np.random.default_rng(20260928 + 101 * seed + 5)
        candidate_indices = rng.permutation(
            len(prepared["raw"]["train"][0])
        )[:60]
        assert candidate_indices.tolist() == receipt["candidate_indices"]
        source = prepared["raw"]["train"][0][candidate_indices]
        poison, patterns, masks = pilot_verify.generator_outputs(
            source, pattern_generator, mask_generator
        )
        assert common.array_sha256(poison) == receipt["poison_sha256_float32"]
        assert common.array_sha256(patterns) == receipt["pattern_sha256_float32"]
        assert common.array_sha256(masks) == receipt["mask_sha256_float32"]
        assert np.all(np.isfinite(poison)) and np.all(np.isfinite(masks))
        variability = float(np.mean(np.std(patterns.reshape(60, -1), axis=0)))
        assert close(variability, float(receipt["pattern_samplewise_std_mean"]), 1e-7)
        assert variability > 0.05
        assert 0.0 < float(masks.mean()) < 0.05
        fingerprints += 1

    selected_metrics = [
        row
        for row in metrics
        if row["representation"] == PRIMARY and int(row["seed"]) in eligible_seeds
    ]
    interval = bootstrap(selected_metrics)
    (ROOT / "primary_seed_bootstrap.json").write_text(
        json.dumps(interval, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt = {
        "status": "PASS",
        "meaning": (
            "frozen protocol, 26,400 sample decisions, ten generator-output fingerprints, "
            "victim gates, checkpoint hashes, and eligible seed-block bootstrap replayed"
        ),
        "eligible_victims": 5,
        "eligible_seeds": eligible_seeds,
        "primary_tpr": float(primary["pooled_tpr"]),
        "primary_fpr": float(primary["pooled_fpr"]),
        "primary_seed_bootstrap": interval,
        "internal_tpr": float(internal["pooled_tpr"]),
        "internal_fpr": float(internal["pooled_fpr"]),
        "scientific_verdict": "COUNTEREXAMPLE_CONFIRMED_FOR_FROZEN_DUAL_VIEW",
        "interpretation": (
            "the frozen input+margin rule fails FPR, while hidden64 alone passes; "
            "the failure localizes to the present fusion/input channel rather than universal non-detectability"
        ),
        "attack_stability_limit": "only 5/10 fresh victims satisfy ASR >= 0.80",
        "attack_seed_fingerprints_verified": fingerprints,
    }
    OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
