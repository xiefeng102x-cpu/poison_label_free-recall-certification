#!/usr/bin/env python3
"""Replay the image-native CNN BadNets fresh-seed confirmation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_group_coherent_10seeds as baseline
import verify_top_journal_phase2_pilot as common


PAPER14_ROOT = Path(__file__).resolve().parents[1]
ROOT = PAPER14_ROOT / "results/bloodmnist_cnn_badnets_confirmation_v1"
PROGRAM_ROOT = PAPER14_ROOT / "results/top_journal_program_v1"
OUTPUT = PROGRAM_ROOT / "badnets_confirmation_receipt.json"
SEEDS = tuple(range(52, 62))
PRIMARY = "coherent_bonferroni_input_cnn_margin"


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(left - right) <= tolerance


def main() -> None:
    protocol = json.loads(
        (PROGRAM_ROOT / "phase2_counterexample_confirmation_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert protocol["frozen_before_execution"] is True
    assert protocol["seeds"] == list(SEEDS)
    assert manifest["role"] == "badnets_fresh_confirmation"
    assert manifest["seeds"] == list(SEEDS)
    assert manifest["attacks"] == ["badnets"]
    assert manifest["poison_count"] == 60
    assert manifest["epochs"] == 30
    dataset_root = Path(manifest["dataset_root"])
    assert common.file_sha256(dataset_root / "bloodmnist.npz") == manifest[
        "dataset_npz_sha256"
    ]

    metrics = rows(ROOT / "per_config_metrics.csv")
    samples = rows(ROOT / "per_sample_scores.csv")
    victims = rows(ROOT / "victim_metrics.csv")
    eligible = rows(ROOT / "eligible_summary.csv")
    assert len(metrics) == 40
    assert len(samples) == 26400
    assert len(victims) == 10
    assert len(eligible) == 2
    assert all(int(row["eligible"]) == 1 for row in victims)
    assert all(float(row["balanced_ca"]) >= 0.90 for row in victims)
    assert all(float(row["asr"]) >= 0.80 for row in victims)

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

    primary = next(row for row in eligible if row["representation"] == PRIMARY)
    assert int(primary["eligible_seed_count"]) == 10
    assert tuple(int(primary[key]) for key in ("tp", "fn", "fp", "tn")) == (
        600,
        0,
        275,
        5725,
    )
    assert close(float(primary["pooled_tpr"]), 1.0)
    assert close(float(primary["pooled_fpr"]), 275 / 6000)
    assert float(primary["pooled_fpr"]) <= 0.05

    class_data = heldout.collect_classes(dataset_root)
    fingerprints = 0
    for receipt in manifest["attack_receipts"]:
        seed = int(receipt["seed"])
        prepared = baseline.preprocess_seed(class_data, seed)
        raw_target = prepared["raw"]["train"][6]
        raw_source = prepared["raw"]["train"][0]
        triggered = common.independent_trigger("badnets", raw_source, seed)
        assert common.array_sha256(triggered) == receipt[
            "triggered_source_train_sha256_float32"
        ]
        rng = np.random.default_rng(20260928 + 101 * seed)
        rng.permutation(len(raw_target))
        poison_indices = rng.permutation(len(raw_source))[:60]
        assert poison_indices.tolist() == receipt["poison_indices"]
        assert common.array_sha256(triggered[poison_indices]) == receipt[
            "poison_raw_sha256_float32"
        ]
        checkpoint = ROOT / receipt["checkpoint"]
        assert common.file_sha256(checkpoint) == receipt["checkpoint_sha256"]
        fingerprints += 1

    selected_metrics = [row for row in metrics if row["representation"] == PRIMARY]
    counts = np.asarray(
        [[int(row[key]) for key in ("tp", "fn", "fp", "tn")] for row in selected_metrics]
    )
    rng = np.random.default_rng(20260928)
    indices = rng.integers(0, 10, size=(10000, 10))
    sampled = counts[indices].sum(axis=1)
    tpr = sampled[:, 0] / (sampled[:, 0] + sampled[:, 1])
    fpr = sampled[:, 2] / (sampled[:, 2] + sampled[:, 3])
    bootstrap = {
        "tpr_ci_low": float(np.quantile(tpr, 0.025)),
        "tpr_ci_high": float(np.quantile(tpr, 0.975)),
        "fpr_ci_low": float(np.quantile(fpr, 0.025)),
        "fpr_ci_high": float(np.quantile(fpr, 0.975)),
    }
    (ROOT / "primary_seed_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    receipt = {
        "status": "PASS",
        "meaning": (
            "frozen protocol, 26,400 sample decisions, ten attack/data fingerprints, "
            "victim gates, checkpoint hashes, and seed-block bootstrap replayed"
        ),
        "eligible_victims": 10,
        "primary_tpr": float(primary["pooled_tpr"]),
        "primary_fpr": float(primary["pooled_fpr"]),
        "primary_seed_bootstrap": bootstrap,
        "scientific_verdict": "PASS_COUNTEREXAMPLE_NOT_CONFIRMED",
        "development_fpr": 0.051666666666666666,
        "confirmation_fpr": float(primary["pooled_fpr"]),
    }
    OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
