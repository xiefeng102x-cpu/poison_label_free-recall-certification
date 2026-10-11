#!/usr/bin/env python3
"""Replay the image-native CNN BadNets/WaNet resource gates."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_group_coherent_10seeds as baseline
import run_bloodmnist_cnn_attack_pilot as cnn
import verify_top_journal_phase2_pilot as common


PAPER14_ROOT = Path(__file__).resolve().parents[1]
PROGRAM_ROOT = PAPER14_ROOT / "results/top_journal_program_v1"
OUTPUT = PROGRAM_ROOT / "cnn_pilot_receipt.json"
CASES = (
    {
        "name": "matched_poison_count_60",
        "root": PAPER14_ROOT / "results/bloodmnist_cnn_attack_pilot_v1",
        "attacks": ("badnets", "wanet"),
        "poison_count": 60,
        "metric_rows": 24,
        "sample_rows": 15840,
        "eligible": {"badnets": 3, "wanet": 0},
    },
    {
        "name": "wanet_canonical_total_pratio_0.10",
        "root": PAPER14_ROOT / "results/bloodmnist_cnn_wanet_pratio10_pilot_v1",
        "attacks": ("wanet",),
        "poison_count": 120,
        "metric_rows": 12,
        "sample_rows": 8640,
        "eligible": {"wanet": 2},
    },
    {
        "name": "label_consistent_adversarial_base_pratio_0.10",
        "root": PAPER14_ROOT / "results/bloodmnist_cnn_label_consistent_pilot_v1",
        "attacks": ("label_consistent",),
        "poison_count": 120,
        "metric_rows": 12,
        "sample_rows": 7200,
        "eligible": {"label_consistent": 3},
    },
)


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(left - right) <= tolerance


def independent_label_corners(images: np.ndarray) -> np.ndarray:
    values = images.reshape(-1, 28, 28, 3).astype(np.float32, copy=True)
    positive = {(0, 0), (1, 1), (2, 0), (2, 2)}
    base = [(r, c) for r in range(3) for c in range(3)]
    corners = ((0, 0, 1, 1), (27, 0, -1, 1), (0, 27, 1, -1), (27, 27, -1, -1))
    for anchor_y, anchor_x, direction_y, direction_x in corners:
        for row, col in base:
            y = anchor_y + direction_y * row
            x = anchor_x + direction_x * col
            values[:, y, x, :] += 1.0 if (row, col) in positive else -1.0
    return np.clip(values, 0.0, 1.0).reshape(len(values), -1)


def independent_targeted_pgd(
    model: cnn.SmallBloodCNN, images: np.ndarray
) -> np.ndarray:
    model.eval()
    original = torch.as_tensor(cnn.nchw(images), dtype=torch.float32)
    current = original.clone()
    target = torch.zeros(len(current), dtype=torch.long)
    for _ in range(10):
        current.requires_grad_(True)
        (gradient,) = torch.autograd.grad(
            F.cross_entropy(model(current), target), current
        )
        with torch.no_grad():
            current = current - (2.0 / 255.0) * gradient.sign()
            current = torch.maximum(
                torch.minimum(current, original + 16.0 / 255.0),
                original - 16.0 / 255.0,
            )
            current.clamp_(0.0, 1.0)
    return current.numpy().transpose(0, 2, 3, 1).reshape(len(images), -1)


def verify_fingerprints(root: Path, manifest: dict) -> int:
    dataset_root = Path(manifest["dataset_root"])
    assert common.file_sha256(dataset_root / "bloodmnist.npz") == manifest[
        "dataset_npz_sha256"
    ]
    class_data = heldout.collect_classes(dataset_root)
    count = 0
    for receipt in manifest["attack_receipts"]:
        seed = int(receipt["seed"])
        attack = receipt["attack"]
        poison_count = int(receipt["poison_count"])
        prepared = baseline.preprocess_seed(class_data, seed)
        raw_target = prepared["raw"]["train"][6]
        raw_source = prepared["raw"]["train"][0]
        triggered_source = (
            independent_label_corners(raw_source)
            if attack == "label_consistent"
            else common.independent_trigger(attack, raw_source, seed)
        )
        assert (
            common.array_sha256(triggered_source)
            == receipt["triggered_source_train_sha256_float32"]
        )
        offset = {"badnets": 0, "wanet": 3, "label_consistent": 4}[attack]
        rng = np.random.default_rng(20260928 + 101 * seed + offset)
        target_order = rng.permutation(len(raw_target))
        source_order = rng.permutation(len(raw_source))
        poison_indices = (
            target_order[:poison_count]
            if attack == "label_consistent"
            else source_order[:poison_count]
        )
        assert poison_indices.tolist() == receipt["poison_indices"]
        if attack == "label_consistent":
            clean_checkpoint = Path(receipt["clean_checkpoint"])
            assert common.file_sha256(clean_checkpoint) == receipt[
                "clean_checkpoint_sha256"
            ]
            payload = torch.load(
                clean_checkpoint, map_location="cpu", weights_only=False
            )
            model = cnn.SmallBloodCNN(payload["mean"], payload["std"])
            model.load_state_dict(payload["state_dict"])
            bases = independent_targeted_pgd(model, raw_target[poison_indices])
            assert common.array_sha256(bases) == receipt["adversarial_base"][
                "adversarial_base_sha256_float32"
            ]
            poison = independent_label_corners(bases)
        else:
            poison = triggered_source[poison_indices]
        assert common.array_sha256(poison) == receipt["poison_raw_sha256_float32"]
        if attack == "wanet":
            grid = common.independent_wanet_grid(seed)
            assert common.array_sha256(grid) == receipt["grid_sha256_float32"]
            remaining_source = np.setdiff1d(np.arange(len(raw_source)), poison_indices)
            combined = np.concatenate([raw_target, raw_source[remaining_source]], axis=0)
            cross_count = 2 * poison_count
            cross_indices = rng.permutation(len(combined))[:cross_count]
            generator = torch.Generator(device="cpu").manual_seed(seed + 92003)
            jitter = (
                torch.rand((cross_count, 28, 28, 2), generator=generator) * 2.0 - 1.0
            )
            cross_grids = torch.clamp(
                grid.repeat(cross_count, 1, 1, 1) + jitter / 28.0, -1.0, 1.0
            )
            cross_images = common.independent_warp(combined[cross_indices], cross_grids)
            assert (
                common.array_sha256(cross_images)
                == receipt["cross"]["cross_processed_sha256_float32"]
            )
        checkpoint = root / receipt["checkpoint"]
        assert checkpoint.exists()
        assert common.file_sha256(checkpoint) == receipt["checkpoint_sha256"]
        count += 1
    return count


def verify_case(case: dict) -> dict:
    root = case["root"]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["role"] == "development_resource_gate_not_confirmation"
    assert manifest["seeds"] == [42, 46, 51]
    assert tuple(manifest["attacks"]) == case["attacks"]
    assert manifest["poison_count"] == case["poison_count"]
    metrics = rows(root / "per_config_metrics.csv")
    samples = rows(root / "per_sample_scores.csv")
    victims = rows(root / "victim_metrics.csv")
    eligible = rows(root / "eligible_summary.csv")
    assert len(metrics) == case["metric_rows"]
    assert len(samples) == case["sample_rows"]
    assert len(victims) == 3 * len(case["attacks"])
    assert len(eligible) == 2 * len(case["attacks"])

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in samples:
        assert int(row["flagged"]) == (float(row["p_value"]) <= float(row["alpha"]))
        grouped[(row["seed"], row["trigger"], row["representation"])].append(row)
    for metric in metrics:
        selected = grouped[
            (metric["seed"], metric["trigger"], metric["representation"])
        ]
        clean = [row for row in selected if int(row["label"]) == 0]
        poison = [row for row in selected if int(row["label"]) == 1]
        fp = sum(int(row["flagged"]) for row in clean)
        tp = sum(int(row["flagged"]) for row in poison)
        assert (tp, len(poison) - tp, fp, len(clean) - fp) == tuple(
            int(metric[key]) for key in ("tp", "fn", "fp", "tn")
        )
        assert close(tp / len(poison), float(metric["tpr"]))
        assert close(fp / len(clean), float(metric["fpr"]))

    victim_counts = defaultdict(int)
    for row in victims:
        eligible_flag = (
            float(row["balanced_ca"]) >= 0.90 and float(row["asr"]) >= 0.80
        )
        assert int(row["eligible"]) == eligible_flag
        victim_counts[row["attack"]] += int(eligible_flag)
    assert dict(victim_counts) == case["eligible"]

    eligible_map = {(row["attack"], row["representation"]): row for row in eligible}
    if case["name"] == "matched_poison_count_60":
        badnets = eligible_map[("badnets", "coherent_bonferroni_input_cnn_margin")]
        assert int(badnets["eligible_seed_count"]) == 3
        assert close(float(badnets["pooled_tpr"]), 1.0)
        assert close(float(badnets["pooled_fpr"]), 93 / 1800)
        assert float(badnets["pooled_fpr"]) > 0.05
    else:
        if case["name"] == "wanet_canonical_total_pratio_0.10":
            wanet = eligible_map[("wanet", "coherent_bonferroni_input_cnn_margin")]
            assert int(wanet["eligible_seed_count"]) == 2
            assert close(float(wanet["pooled_tpr"]), 239 / 240)
            assert close(float(wanet["pooled_fpr"]), 36 / 1200)
            assert float(wanet["pooled_fpr"]) <= 0.05
        else:
            lc = eligible_map[
                ("label_consistent", "coherent_bonferroni_input_cnn_margin")
            ]
            assert int(lc["eligible_seed_count"]) == 3
            assert close(float(lc["pooled_tpr"]), 1.0)
            assert close(float(lc["pooled_fpr"]), 71 / 1440)
            assert float(lc["pooled_fpr"]) <= 0.05

    fingerprints = verify_fingerprints(root, manifest)
    return {
        "name": case["name"],
        "metric_rows": len(metrics),
        "sample_rows": len(samples),
        "victim_rows": len(victims),
        "attack_seed_fingerprints_verified": fingerprints,
        "eligible_victims": dict(victim_counts),
    }


def main() -> None:
    case_receipts = [verify_case(case) for case in CASES]
    receipt = {
        "status": "PASS",
        "meaning": (
            "sample decisions, victim gates, attack tensors, dataset identity, "
            "and CNN checkpoint hashes replayed for both poison-rate pilots"
        ),
        "cases": case_receipts,
        "scientific_verdict": {
            "badnets": (
                "3/3 valid CNN victims; input+margin TPR 1.0 but FPR 0.05167, "
                "so the frozen primary gate fails marginally and requires fresh seeds"
            ),
            "wanet_matched_60": "0/3 valid CNN victims; no defense claim",
            "wanet_canonical_120": (
                "2/3 valid CNN victims; eligible input+margin TPR 0.99583/FPR 0.03; "
                "attack stability remains a limitation"
            ),
            "label_consistent": (
                "3/3 valid CNN victims with adversarial bases; input+margin TPR 1.0/"
                "FPR 0.04931 passes narrowly, while hidden64 alone fails FPR"
            ),
        },
    }
    OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
