#!/usr/bin/env python3
"""Independent replay for the coordinate-aligned PathMNIST warm-start pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import pathmnist_external_validation as core
import run_pathmnist_external_validation as common
import run_pathmnist_external_validation_v2 as v2
import run_pathmnist_warmstart_pilot as warm
import verify_pathmnist_external_validation as verify_v1


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/pathmnist_warmstart_verification.json"
)


def verify_inputs(dataset: Path) -> dict[str, str]:
    protocol = verify_v1.read_json(warm.PROTOCOL)
    frozen = protocol["frozen_inputs"]
    observed = {
        "dataset_sha256": core.sha256_file(dataset),
        "core_implementation_sha256": core.sha256_file(Path(core.__file__)),
        "common_runner_sha256": core.sha256_file(Path(common.__file__)),
        "v2_runner_sha256": core.sha256_file(Path(v2.__file__)),
        "warmstart_runner_sha256": core.sha256_file(Path(warm.__file__)),
        "source_protocol_sha256": core.sha256_file(warm.SOURCE_PROTOCOL),
        "source_manifest_sha256": core.sha256_file(warm.SOURCE_ROOT / "manifest.json"),
        "source_receipts_sha256": core.sha256_file(
            warm.SOURCE_ROOT / "seed_receipts.json"
        ),
    }
    assert all(frozen[key] == value for key, value in observed.items())
    return observed


def verify_artifacts(source: Path, dataset: Path) -> dict[str, Any]:
    manifest = verify_v1.read_json(source / "manifest.json")
    assert manifest["experiment"] == "pathmnist_warmstart_pilot_v1"
    assert tuple(manifest["seeds"]) == warm.SEEDS
    assert manifest["protocol_sha256"] == core.sha256_file(warm.PROTOCOL)
    assert manifest["source_protocol_sha256"] == core.sha256_file(warm.SOURCE_PROTOCOL)
    metrics = verify_v1.read_csv(source / "per_config_metrics.csv")
    victims = verify_v1.read_csv(source / "victim_metrics.csv")
    receipts = verify_v1.read_json(source / "seed_receipts.json")
    assert len(metrics) == len(warm.SEEDS) * len(warm.VIEWS) * len(warm.METHODS)
    assert len(victims) == len(receipts) == len(warm.SEEDS)
    by_metric = {
        (int(row["seed"]), row["view"], row["method"]): row for row in metrics
    }
    by_receipt = {int(row["seed"]): row for row in receipts}
    eligible = {int(row["seed"]) for row in victims if int(row["eligible"]) == 1}
    with np.load(dataset, allow_pickle=False) as payload:
        train_labels = payload["train_labels"].reshape(-1).astype(np.int64)
    subset_indices = verify_v1.recompute_subset(train_labels)
    subset_labels = train_labels[subset_indices]
    replayed_rows = 0
    spectral_ambiguities = 0
    pooled_eligible: dict[tuple[str, str], dict[str, int]] = {}

    for seed in warm.SEEDS:
        receipt = by_receipt[seed]
        candidates = np.flatnonzero(subset_labels != core.TARGET_CLASS)
        chosen = np.sort(
            np.random.default_rng(20260930 + seed).choice(
                candidates, size=v2.POISON_COUNT, replace=False
            )
        )
        expected_truth = np.zeros(len(subset_indices), dtype=bool)
        expected_truth[chosen] = True
        sample_path = source / "sample_artifacts" / f"seed_{seed}.npz"
        assert core.sha256_file(sample_path) == receipt["sample_artifact"]["sha256"]
        with np.load(sample_path, allow_pickle=False) as payload:
            truth = payload["is_poison"].astype(bool)
            original = payload["original_label"].astype(np.int64)
            assigned = payload["assigned_label"].astype(np.int64)
            assert np.array_equal(payload["original_index"], subset_indices)
            assert np.array_equal(original, subset_labels)
            assert np.array_equal(truth, expected_truth)
            assert np.all(assigned[truth] == core.TARGET_CLASS)
            for view in warm.VIEWS:
                for method in warm.METHODS:
                    row = by_metric[(seed, view, method)]
                    scores = payload[
                        common.artifact_key(view, method, "scores")
                    ].astype(np.float64)
                    p_values = payload[
                        common.artifact_key(view, method, "p_values")
                    ].astype(np.float64)
                    flags = payload[
                        common.artifact_key(view, method, "flagged")
                    ].astype(bool)
                    defined = assigned == core.TARGET_CLASS if method == "cf_cmd_u" else np.ones(len(truth), dtype=bool)
                    assert np.isfinite(scores[defined]).all()
                    if method == "cf_cmd_u":
                        assert np.isnan(scores[~defined]).all()
                        assert np.array_equal(flags[defined], p_values[defined] <= core.ALPHA)
                        assert not flags[~defined].any()
                    elif method == "backdoorbench_spectral":
                        method_receipt = json.loads(row["receipt_json"])
                        replay = np.zeros(len(flags), dtype=bool)
                        distance = np.full(len(flags), np.inf)
                        for label in range(core.N_CLASSES):
                            mask = assigned == label
                            threshold = float(method_receipt["thresholds"][str(label)])
                            replay[mask] = scores[mask] > threshold
                            distance[mask] = np.abs(scores[mask] - threshold)
                        disagreement = flags != replay
                        assert np.all(distance[disagreement] <= 2e-6)
                        spectral_ambiguities += int(disagreement.sum())
                    clean = ~truth
                    counts = {
                        "tp": int(np.sum(flags & truth)),
                        "fn": int(np.sum(~flags & truth)),
                        "fp": int(np.sum(flags & clean)),
                        "tn": int(np.sum(~flags & clean)),
                    }
                    assert all(int(row[name]) == value for name, value in counts.items())
                    if seed in eligible:
                        accumulator = pooled_eligible.setdefault(
                            (view, method), {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
                        )
                        for name, value in counts.items():
                            accumulator[name] += value
                    replayed_rows += len(truth)

        assert receipt["warmstart_training"]["initial_state_sha256"] == receipt[
            "source_clean"
        ]["state_sha256"]
        source_path = Path(receipt["source_clean"]["path"])
        assert core.sha256_file(source_path) == receipt["source_clean"]["file_sha256"]
        checkpoint_path = source / "models" / f"seed_{seed}_warmstart_suspect.pt"
        assert core.sha256_file(checkpoint_path) == receipt["checkpoint"]["file_sha256"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["seed"] == seed and checkpoint["role"] == "warmstart_suspect"
        assert checkpoint["protocol_sha256"] == core.sha256_file(warm.PROTOCOL)
        assert verify_v1.state_fingerprint(checkpoint["state_dict"]) == receipt[
            "checkpoint"
        ]["state_sha256"]

    pooled = {}
    for (view, method), counts in pooled_eligible.items():
        pooled[f"{view}/{method}"] = {
            **counts,
            "tpr": counts["tp"] / (counts["tp"] + counts["fn"]),
            "fpr": counts["fp"] / (counts["fp"] + counts["tn"]),
        }
    return {
        "seeds": list(warm.SEEDS),
        "eligible_seeds": sorted(eligible),
        "sample_decisions_replayed": replayed_rows,
        "source_checkpoints_verified": len(warm.SEEDS),
        "warmstart_checkpoints_verified": len(warm.SEEDS),
        "spectral_float32_boundary_ambiguities": spectral_ambiguities,
        "pooled_eligible_only": pooled,
        "gate": manifest["gate"],
    }


def main(args: argparse.Namespace) -> None:
    receipt = {
        "status": "PASS",
        "protocol": str(warm.PROTOCOL),
        "protocol_sha256": core.sha256_file(warm.PROTOCOL),
        "observed_frozen_hashes": verify_inputs(args.dataset),
        "replay": verify_artifacts(args.source, args.dataset),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=warm.DEFAULT_DATASET)
    parser.add_argument("--source", type=Path, default=warm.DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    main(parser.parse_args())
