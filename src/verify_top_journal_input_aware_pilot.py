#!/usr/bin/env python3
"""Replay the Input-Aware-style BloodMNIST resource gate."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_cnn_attack_pilot as cnn
import run_bloodmnist_group_coherent_10seeds as baseline
import run_bloodmnist_input_aware_pilot as input_aware
import verify_top_journal_phase2_pilot as common


PAPER14_ROOT = Path(__file__).resolve().parents[1]
ROOT = PAPER14_ROOT / "results/bloodmnist_input_aware_pilot_v1"
OUTPUT = PAPER14_ROOT / "results/top_journal_program_v1/input_aware_pilot_receipt.json"
SEEDS = (42, 46, 51)


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return abs(left - right) <= tolerance


def generator_outputs(
    source: np.ndarray,
    pattern_generator: input_aware.TinyGenerator,
    mask_generator: input_aware.TinyGenerator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = torch.as_tensor(cnn.nchw(source), dtype=torch.float32)
    with torch.no_grad():
        patterns = pattern_generator(values)
        raw_mask = mask_generator(values)
        masks = (
            torch.tanh(raw_mask * 20.0 - 10.0) / (2.0 + 1e-7) + 0.5
        ).repeat(1, 3, 1, 1)
        poison = values + (patterns - values) * masks

    def flatten(tensor: torch.Tensor) -> np.ndarray:
        return tensor.numpy().transpose(0, 2, 3, 1).reshape(len(tensor), -1)

    return flatten(poison), flatten(patterns), flatten(masks)


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["role"] == "development_resource_gate_not_confirmation"
    assert manifest["attack_label"] == "Input-Aware-style clean-room implementation"
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
    eligible = rows(ROOT / "eligible_summary.csv")
    assert len(metrics) == 12
    assert len(samples) == 7920
    assert len(victims) == 3
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
        assert close(tp / len(poison), float(metric["tpr"]))
        assert close(fp / len(clean), float(metric["fpr"]))

    for row in victims:
        assert float(row["balanced_ca"]) >= 0.90
        assert float(row["asr"]) >= 0.80
        assert int(row["eligible"]) == 1
        assert float(row["mask_mean"]) > 0.0
        assert float(row["mask_fraction_over_0_5"]) > 0.0
        assert float(row["pattern_samplewise_std_mean"]) > 0.05

    primary = next(
        row
        for row in eligible
        if row["representation"] == "coherent_bonferroni_input_cnn_margin"
    )
    assert int(primary["eligible_seed_count"]) == 3
    assert tuple(int(primary[key]) for key in ("tp", "fn", "fp", "tn")) == (
        179,
        1,
        112,
        1688,
    )
    assert close(float(primary["pooled_tpr"]), 179 / 180)
    assert close(float(primary["pooled_fpr"]), 112 / 1800)
    assert float(primary["pooled_fpr"]) > 0.05

    class_data = heldout.collect_classes(dataset_root)
    fingerprints = 0
    for receipt in manifest["receipts"]:
        assert set(receipt["mechanisms"]) == {
            "sample_conditioned_pattern_generator",
            "sample_conditioned_sparse_mask_generator",
            "mask_pretraining_with_norm_and_diversity",
            "joint_classifier_pattern_training",
            "correct_label_cross_samples",
        }
        checkpoint = ROOT / receipt["checkpoint"]
        assert common.file_sha256(checkpoint) == receipt["checkpoint_sha256"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        pattern_generator = input_aware.TinyGenerator(3)
        mask_generator = input_aware.TinyGenerator(1)
        pattern_generator.load_state_dict(payload["pattern_generator"])
        mask_generator.load_state_dict(payload["mask_generator"])
        pattern_generator.eval()
        mask_generator.eval()
        seed = int(receipt["seed"])
        prepared = baseline.preprocess_seed(class_data, seed)
        source = prepared["raw"]["train"][0][receipt["candidate_indices"]]
        poison, patterns, masks = generator_outputs(
            source, pattern_generator, mask_generator
        )
        assert np.all(np.isfinite(poison))
        assert np.all(np.isfinite(patterns))
        assert np.all(np.isfinite(masks))
        assert common.array_sha256(poison) == receipt["poison_sha256_float32"]
        assert common.array_sha256(patterns) == receipt["pattern_sha256_float32"]
        assert common.array_sha256(masks) == receipt["mask_sha256_float32"]
        assert close(float(masks.mean()), float(receipt["mask_mean"]), 1e-7)
        flattened = patterns.reshape(len(patterns), -1)
        variability = float(np.mean(np.std(flattened, axis=0)))
        assert close(
            variability, float(receipt["pattern_samplewise_std_mean"]), 1e-7
        )
        assert variability > 0.05
        assert float(np.linalg.norm(flattened - flattened.mean(axis=0))) > 1.0
        fingerprints += 1

    receipt = {
        "status": "PASS",
        "meaning": (
            "7,920 sample decisions, victim eligibility, sample-conditioned generator "
            "outputs, non-degeneracy statistics, dataset identity, and checkpoint hashes replayed"
        ),
        "attack_seed_fingerprints_verified": fingerprints,
        "eligible_victims": 3,
        "primary_tpr": float(primary["pooled_tpr"]),
        "primary_fpr": float(primary["pooled_fpr"]),
        "scientific_verdict": "COUNTEREXAMPLE_CANDIDATE_REQUIRES_FRESH_SEEDS",
        "reason": (
            "all three victims pass CA/ASR, while the frozen input+margin detector "
            "has FPR 0.06222 > 0.05 with TPR 0.99444"
        ),
        "scope_limit": (
            "clean-room Input-Aware-style implementation preserves the defining mechanisms "
            "but is not asserted bit-identical to the reference implementation"
        ),
    }
    OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
