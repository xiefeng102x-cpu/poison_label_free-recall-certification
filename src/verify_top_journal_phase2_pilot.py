#!/usr/bin/env python3
"""Independent replay of the BloodMNIST cross-attack pilot artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_group_coherent_10seeds as baseline


PAPER14_ROOT = Path(__file__).resolve().parents[1]
ROOT = PAPER14_ROOT / "results/bloodmnist_cross_attack_pilot_v1"
OUTPUT = PAPER14_ROOT / "results/top_journal_program_v1/phase2_pilot_receipt.json"
ATTACKS = ("badnets", "blended", "sig", "wanet")
SEEDS = (42, 46, 51)
ENDPOINTS = {
    "coherent_bonferroni_input_classical_margin": "classical",
    "coherent_qml_xz_drift16": "qml",
    "coherent_bonferroni_input_qml_xz": "qml",
}


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(left - right) <= tolerance


def array_sha256(values: np.ndarray | torch.Tensor) -> str:
    array = (
        values.detach().cpu().numpy()
        if isinstance(values, torch.Tensor)
        else np.asarray(values)
    )
    canonical = np.ascontiguousarray(array.astype("<f4", copy=False))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_raw(prepared: dict, raw_images: np.ndarray) -> np.ndarray:
    standardized = prepared["raw_scaler"].transform(raw_images)
    pca_values = prepared["pca"].transform(standardized)
    return prepared["post_scaler"].transform(pca_values).astype(np.float32)


def independent_wanet_grid(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + 91001)
    coarse = torch.rand((1, 2, 4, 4), generator=generator) * 2.0 - 1.0
    coarse = coarse / torch.mean(torch.abs(coarse))
    noise = F.interpolate(
        coarse, size=(28, 28), mode="bicubic", align_corners=True
    ).permute(0, 2, 3, 1)
    axis = torch.linspace(-1.0, 1.0, steps=28)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    return torch.clamp(identity + 0.5 * noise / 28.0, -1.0, 1.0).float()


def independent_warp(raw_images: np.ndarray, grids: torch.Tensor) -> np.ndarray:
    tensor = torch.from_numpy(raw_images.reshape(-1, 28, 28, 3)).permute(0, 3, 1, 2)
    if grids.shape[0] == 1:
        grids = grids.repeat(len(tensor), 1, 1, 1)
    warped = F.grid_sample(
        tensor, grids, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    result = (
        warped.permute(0, 2, 3, 1)
        .numpy()
        .astype(np.float32)
        .reshape(len(tensor), -1)
    )
    return np.clip(result, 0.0, 1.0)


def independent_trigger(attack: str, raw_images: np.ndarray, seed: int) -> np.ndarray:
    images = raw_images.reshape(-1, 28, 28, 3).astype(np.float32, copy=True)
    if attack == "badnets":
        images[:, -3:, -3:, :] = 1.0
    elif attack == "blended":
        pattern = np.random.default_rng(14014).uniform(
            0.0, 1.0, size=(28, 28, 3)
        ).astype(np.float32)
        images = 0.8 * images + 0.2 * pattern[None]
    elif attack == "sig":
        columns = np.arange(28, dtype=np.float32)
        signal = (40.0 / 255.0) * np.sin(2.0 * np.pi * columns * 6.0 / 28.0)
        images = np.clip(images + signal[None, None, :, None], 0.0, 1.0)
    elif attack == "wanet":
        return independent_warp(raw_images, independent_wanet_grid(seed))
    else:
        raise ValueError(attack)
    return images.astype(np.float32, copy=False).reshape(len(images), -1)


def verify_attack_fingerprints(manifest: dict, root: Path = ROOT) -> int:
    dataset_root = Path(manifest["dataset_root"])
    assert file_sha256(dataset_root / "bloodmnist.npz") == manifest["dataset_npz_sha256"]
    class_data = heldout.collect_classes(dataset_root)
    verified = 0
    for receipt in manifest["attack_receipts"]:
        seed = int(receipt["seed"])
        attack = receipt["attack"]
        prepared = baseline.preprocess_seed(class_data, seed)
        raw_target = prepared["raw"]["train"][6]
        raw_source = prepared["raw"]["train"][0]
        triggered_source = independent_trigger(attack, raw_source, seed)
        assert (
            array_sha256(triggered_source)
            == receipt["triggered_source_train_sha256_float32"]
        )

        rng = np.random.default_rng(20260928 + 101 * seed + ATTACKS.index(attack))
        target_order = rng.permutation(len(raw_target))
        source_order = rng.permutation(len(raw_source))
        if attack == "sig":
            poison_raw = independent_trigger(attack, raw_target[target_order[:60]], seed)
        else:
            poison_raw = triggered_source[source_order[:60]]
        poison_processed = process_raw(prepared, poison_raw)
        assert array_sha256(poison_processed) == receipt["poison_processed_sha256_float32"]

        if attack == "wanet":
            grid = independent_wanet_grid(seed)
            assert array_sha256(grid) == receipt["wanet_grid_sha256_float32"]
            remaining_source = np.setdiff1d(
                np.arange(len(raw_source)), source_order[:60]
            )
            combined_raw = np.concatenate(
                [raw_target, raw_source[remaining_source]], axis=0
            )
            cross_indices = rng.permutation(len(combined_raw))[:120]
            generator = torch.Generator(device="cpu").manual_seed(seed + 92003)
            jitter = torch.rand((120, 28, 28, 2), generator=generator) * 2.0 - 1.0
            cross_grids = torch.clamp(
                grid.repeat(120, 1, 1, 1) + jitter / 28.0, -1.0, 1.0
            )
            cross_raw = independent_warp(combined_raw[cross_indices], cross_grids)
            assert (
                array_sha256(process_raw(prepared, cross_raw))
                == receipt["wanet_cross"]["cross_processed_sha256_float32"]
            )

        checkpoints = receipt["victim_checkpoints"]
        for kind in ("classical", "qml"):
            path = root / checkpoints[kind]
            assert path.exists()
            assert file_sha256(path) == checkpoints[f"{kind}_sha256"]
        verified += 1
    return verified


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["role"] == "development_resource_gate_not_confirmation"
    assert manifest["seeds"] == list(SEEDS)
    assert manifest["attacks"] == list(ATTACKS)
    assert manifest["trusted_reference_split"] == "validation_target"
    assert manifest["victim_eligibility"] == {
        "balanced_ca_min": 0.9,
        "asr_min": 0.8,
    }
    assert set(manifest["blocked_families"]) == {"input_aware", "label_consistent"}

    metrics = csv_rows(ROOT / "per_config_metrics.csv")
    samples = csv_rows(ROOT / "per_sample_scores.csv")
    victims = csv_rows(ROOT / "victim_metrics.csv")
    pooled = csv_rows(ROOT / "pooled_summary.csv")
    eligible = csv_rows(ROOT / "eligible_summary.csv")
    assert len(metrics) == 3 * 4 * 6
    assert len(samples) == 46440
    assert len(victims) == 3 * 4
    assert len(pooled) == 4 * 6
    assert len(eligible) == 4 * 3

    grouped_samples: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in samples:
        assert int(row["flagged"]) == (float(row["p_value"]) <= float(row["alpha"]))
        grouped_samples[(row["seed"], row["trigger"], row["representation"])].append(row)
    for metric in metrics:
        selected = grouped_samples[
            (metric["seed"], metric["trigger"], metric["representation"])
        ]
        clean = [row for row in selected if int(row["label"]) == 0]
        poison = [row for row in selected if int(row["label"]) == 1]
        fp = sum(int(row["flagged"]) for row in clean)
        tp = sum(int(row["flagged"]) for row in poison)
        counts = (tp, len(poison) - tp, fp, len(clean) - fp)
        assert counts == tuple(int(metric[key]) for key in ("tp", "fn", "fp", "tn"))
        assert close(tp / len(poison), float(metric["tpr"]))
        assert close(fp / len(clean), float(metric["fpr"]))

    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in metrics:
        by_group[(row["trigger"], row["representation"])].append(row)
    for summary in pooled:
        selected = by_group[(summary["trigger"], summary["representation"])]
        counts = tuple(
            sum(int(row[key]) for row in selected) for key in ("tp", "fn", "fp", "tn")
        )
        assert counts == tuple(int(summary[key]) for key in ("tp", "fn", "fp", "tn"))
        assert close(counts[0] / (counts[0] + counts[1]), float(summary["pooled_tpr"]))
        assert close(counts[2] / (counts[2] + counts[3]), float(summary["pooled_fpr"]))

    victim_map = {(row["attack"], row["seed"]): row for row in victims}
    expected_eligible = {
        ("badnets", "classical"): 1,
        ("badnets", "qml"): 1,
        ("blended", "classical"): 3,
        ("blended", "qml"): 3,
        ("sig", "classical"): 3,
        ("sig", "qml"): 2,
        ("wanet", "classical"): 0,
        ("wanet", "qml"): 0,
    }
    for row in eligible:
        attack = row["trigger"]
        victim_type = row["victim_type"]
        count = 0
        for seed in map(str, SEEDS):
            victim = victim_map[(attack, seed)]
            if (
                float(victim[f"{victim_type}_balanced_ca"]) >= 0.90
                and float(victim[f"{victim_type}_asr"]) >= 0.80
            ):
                count += 1
        assert int(row["eligible_seed_count"]) == count
        assert count == expected_eligible[(attack, victim_type)]
        assert row["representation"] in ENDPOINTS

    fingerprints_verified = verify_attack_fingerprints(manifest)
    receipt = {
        "status": "PASS",
        "meaning": (
            "sample-level decisions, pooled calculations, victim eligibility, "
            "attack tensors, dataset identity, and checkpoint hashes replayed"
        ),
        "phase2_status": "PILOT_COMPLETE_FOUR_OF_SIX_ATTACKS",
        "scientific_verdict": "PROCEED_SELECTIVELY",
        "metric_rows": len(metrics),
        "sample_rows": len(samples),
        "victim_rows": len(victims),
        "attack_seed_fingerprints_verified": fingerprints_verified,
        "eligible_victim_counts": {
            f"{attack}_{kind}": count
            for (attack, kind), count in sorted(expected_eligible.items())
        },
        "promotion_decision": {
            "blended": "eligible and easy; retain for confirmation",
            "sig": "eligible; retain for confirmation",
            "badnets": "partial eligibility; repair victim architecture/training before scaling",
            "wanet": "no eligible PCA victim; move to image-native CNN before detector claim",
            "input_aware": "not yet implemented",
            "label_consistent": "not yet implemented",
        },
    }
    OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
