#!/usr/bin/env python3
"""Independent replay for the frozen PathMNIST external-validation artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import pathmnist_external_validation as core


PAPER14_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/pathmnist_external_validation_protocol_v1.json"
)
ACQUISITION_PROTOCOL = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/pathmnist_acquisition_protocol_v1.json"
)
ACQUISITION_RECEIPT = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/pathmnist_acquisition_receipt.json"
)
RUNNER = PAPER14_ROOT / "src/run_pathmnist_external_validation.py"
ADAPTER = PAPER14_ROOT / "src/backdoorbench_feature_adapter.py"
UPSTREAM_SCAN = PAPER14_ROOT / "external/BackdoorBench/detection_pretrain/scan.py"
DEFAULT_DATASET = Path("/private/tmp/paper14-data/pathmnist.npz")
DEFAULT_DEVELOPMENT = PAPER14_ROOT / "results/pathmnist_external_validation_v1"
DEFAULT_CONFIRMATION = (
    PAPER14_ROOT / "results/pathmnist_external_validation_confirmation_v1"
)
DEFAULT_RECEIPT = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/pathmnist_external_validation_verification.json"
)
STAGE_SEEDS = {
    "development": (79, 80),
    "confirmation": (81, 82, 83, 84, 85),
}
VIEWS = ("suspect_hidden64", "clean_to_suspect_hidden64_drift")
METHODS = ("cf_cmd_u", "backdoorbench_scan", "backdoorbench_spectral")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        values = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def recompute_subset(labels: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(20260930)
    parts = []
    for label in range(9):
        available = np.flatnonzero(labels == label)
        parts.append(np.sort(rng.choice(available, size=3000, replace=False)))
    return np.sort(np.concatenate(parts)).astype(np.int64, copy=False)


def recompute_poison(subset_labels: np.ndarray, seed: int) -> np.ndarray:
    eligible = np.flatnonzero(subset_labels != 8)
    rng = np.random.default_rng(20260930 + seed)
    return np.sort(rng.choice(eligible, size=1350, replace=False)).astype(
        np.int64, copy=False
    )


def artifact_key(view: str, method: str, value: str) -> str:
    return f"{view}__{method}__{value}"


def verify_hashes(dataset: Path) -> dict[str, str]:
    protocol = read_json(PROTOCOL)
    frozen = protocol["frozen_inputs"]
    observed = {
        "acquisition_protocol_sha256": core.sha256_file(ACQUISITION_PROTOCOL),
        "acquisition_receipt_sha256": core.sha256_file(ACQUISITION_RECEIPT),
        "dataset_sha256": core.sha256_file(dataset),
        "core_implementation_sha256": core.sha256_file(Path(core.__file__)),
        "runner_sha256": core.sha256_file(RUNNER),
        "backdoorbench_adapter_sha256": core.sha256_file(ADAPTER),
        "upstream_scan_sha256": core.sha256_file(UPSTREAM_SCAN),
    }
    mismatches = {
        key: {"expected": frozen.get(key), "observed": value}
        for key, value in observed.items()
        if frozen.get(key) != value
    }
    if mismatches:
        raise AssertionError(f"frozen input mismatch: {mismatches}")
    return observed


def verify_stage(stage: str, source: Path, dataset: Path) -> dict[str, Any]:
    seeds = STAGE_SEEDS[stage]
    manifest = read_json(source / "manifest.json")
    assert manifest["experiment"] == "pathmnist_external_validation_v1"
    assert manifest["stage"] == stage
    assert tuple(manifest["seeds"]) == seeds
    assert manifest["protocol_sha256"] == core.sha256_file(PROTOCOL)
    assert manifest["dataset_sha256"] == core.sha256_file(dataset)
    assert manifest["training_subset_count"] == 27000
    assert manifest["poison_count"] == 1350
    assert tuple(manifest["views"]) == VIEWS
    assert tuple(manifest["methods"]) == METHODS

    metrics = read_csv(source / "per_config_metrics.csv")
    victims = read_csv(source / "victim_metrics.csv")
    seed_receipts = read_json(source / "seed_receipts.json")
    assert len(metrics) == len(seeds) * len(VIEWS) * len(METHODS)
    assert len(victims) == len(seeds)
    assert len(seed_receipts) == len(seeds)
    by_metric = {
        (int(row["seed"]), row["view"], row["method"]): row for row in metrics
    }
    assert len(by_metric) == len(metrics)

    with np.load(dataset, allow_pickle=False) as payload:
        full_train_labels = payload["train_labels"].reshape(-1).astype(np.int64)
    subset_indices = recompute_subset(full_train_labels)
    assert hashlib.sha256(subset_indices.tobytes()).hexdigest() == (
        "42c2b8fe4aa0ce9823df461f0a68022e47769ec1f7e56989a5e77ececd16c060"
    )
    subset_labels = full_train_labels[subset_indices]
    replayed_rows = 0
    checkpoint_count = 0
    pooled: dict[tuple[str, str], dict[str, int]] = {}

    receipt_by_seed = {int(row["seed"]): row for row in seed_receipts}
    assert set(receipt_by_seed) == set(seeds)
    for seed in seeds:
        receipt = receipt_by_seed[seed]
        assert receipt["subset"]["selected_indices_sha256"] == hashlib.sha256(
            subset_indices.tobytes()
        ).hexdigest()
        expected_poison = recompute_poison(subset_labels, seed)
        expected_truth = np.zeros(len(subset_indices), dtype=bool)
        expected_truth[expected_poison] = True
        sample_path = source / "sample_artifacts" / f"seed_{seed}.npz"
        assert core.sha256_file(sample_path) == receipt["sample_artifact"]["sha256"]
        with np.load(sample_path, allow_pickle=False) as payload:
            original_index = payload["original_index"].astype(np.int64)
            original_label = payload["original_label"].astype(np.int64)
            assigned_label = payload["assigned_label"].astype(np.int64)
            truth = payload["is_poison"].astype(bool)
            assert np.array_equal(original_index, subset_indices)
            assert np.array_equal(original_label, subset_labels)
            assert np.array_equal(truth, expected_truth)
            assert np.all(assigned_label[truth] == 8)
            assert np.array_equal(assigned_label[~truth], original_label[~truth])
            for view in VIEWS:
                for method in METHODS:
                    row = by_metric[(seed, view, method)]
                    scores = payload[artifact_key(view, method, "scores")].astype(
                        np.float64
                    )
                    p_values = payload[
                        artifact_key(view, method, "p_values")
                    ].astype(np.float64)
                    flags = payload[artifact_key(view, method, "flagged")].astype(bool)
                    assert scores.shape == p_values.shape == flags.shape == truth.shape
                    defined = assigned_label == 8 if method == "cf_cmd_u" else np.ones(
                        len(truth), dtype=bool
                    )
                    assert np.isfinite(scores[defined]).all()
                    if method == "cf_cmd_u":
                        assert np.isnan(scores[~defined]).all()
                        assert np.all((0.0 <= p_values[defined]) & (p_values[defined] <= 1.0))
                        assert np.array_equal(flags[defined], p_values[defined] <= 0.05)
                        assert not flags[~defined].any()
                    elif method == "backdoorbench_spectral":
                        method_receipt = json.loads(row["receipt_json"])
                        thresholds = method_receipt["thresholds"]
                        replay_flags = np.zeros(len(flags), dtype=bool)
                        for label in range(9):
                            mask = assigned_label == label
                            replay_flags[mask] = scores[mask] > float(thresholds[str(label)])
                        assert np.array_equal(flags, replay_flags)
                    clean = ~truth
                    replay = {
                        "tp": int(np.sum(flags & truth)),
                        "fn": int(np.sum(~flags & truth)),
                        "fp": int(np.sum(flags & clean)),
                        "tn": int(np.sum(~flags & clean)),
                    }
                    assert all(int(row[name]) == value for name, value in replay.items())
                    assert np.isclose(float(row["tpr"]), replay["tp"] / 1350)
                    assert np.isclose(
                        float(row["fpr_all_clean"]), replay["fp"] / (27000 - 1350)
                    )
                    target_clean = clean & (original_label == 8)
                    assert np.isclose(
                        float(row["target_clean_fpr"]), np.mean(flags[target_clean])
                    )
                    accumulator = pooled.setdefault(
                        (view, method), {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
                    )
                    for name, value in replay.items():
                        accumulator[name] += value
                    replayed_rows += len(truth)

        for role in ("clean", "suspect"):
            checkpoint_path = source / "models" / f"seed_{seed}_{role}.pt"
            expected = receipt["checkpoints"][role]
            assert core.sha256_file(checkpoint_path) == expected["file_sha256"]
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            assert checkpoint["seed"] == seed
            assert checkpoint["role"] == role
            assert checkpoint["protocol_sha256"] == core.sha256_file(PROTOCOL)
            assert state_fingerprint(checkpoint["state_dict"]) == expected["state_sha256"]
            checkpoint_count += 1

    pooled_summary = {}
    for (view, method), counts in pooled.items():
        pooled_summary[f"{view}/{method}"] = {
            **counts,
            "tpr": counts["tp"] / (counts["tp"] + counts["fn"]),
            "fpr": counts["fp"] / (counts["fp"] + counts["tn"]),
        }
    return {
        "stage": stage,
        "seeds": list(seeds),
        "sample_decisions_replayed": replayed_rows,
        "checkpoints_verified": checkpoint_count,
        "poison_selections_reconstructed": len(seeds),
        "pooled": pooled_summary,
        "gate": manifest["gate"],
    }


def main(args: argparse.Namespace) -> None:
    observed_hashes = verify_hashes(args.dataset)
    stages = [verify_stage("development", args.development, args.dataset)]
    if args.include_confirmation:
        stages.append(verify_stage("confirmation", args.confirmation, args.dataset))
    receipt = {
        "status": "PASS",
        "protocol": str(PROTOCOL),
        "protocol_sha256": core.sha256_file(PROTOCOL),
        "observed_frozen_hashes": observed_hashes,
        "stages": stages,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--include-confirmation", action="store_true")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    main(parser.parse_args())
