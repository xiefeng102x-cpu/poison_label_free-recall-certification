#!/usr/bin/env python3
"""Independent replay for the sequential PathMNIST V2 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import pathmnist_external_validation as core
import run_pathmnist_external_validation as run_v1
import run_pathmnist_external_validation_v2 as run_v2
import verify_pathmnist_external_validation as verify_v1


PAPER14_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/pathmnist_external_validation_protocol_v2.json"
)
DEFAULT_RECEIPT = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/pathmnist_external_validation_verification_v2.json"
)
STAGE_SEEDS = {
    "development": run_v2.DEVELOPMENT_SEEDS,
    "confirmation": run_v2.CONFIRMATION_SEEDS,
}


def verify_frozen_inputs(dataset: Path) -> dict[str, str]:
    protocol = verify_v1.read_json(PROTOCOL)
    frozen = protocol["frozen_inputs"]
    observed = {
        "acquisition_protocol_sha256": core.sha256_file(run_v2.ACQUISITION_PROTOCOL),
        "acquisition_receipt_sha256": core.sha256_file(run_v2.ACQUISITION_RECEIPT),
        "dataset_sha256": core.sha256_file(dataset),
        "core_implementation_sha256": core.sha256_file(Path(core.__file__)),
        "v1_runner_dependency_sha256": core.sha256_file(Path(run_v1.__file__)),
        "v2_runner_sha256": core.sha256_file(Path(run_v2.__file__)),
        "backdoorbench_adapter_sha256": core.sha256_file(
            PAPER14_ROOT / "src/backdoorbench_feature_adapter.py"
        ),
    }
    assert all(frozen[key] == value for key, value in observed.items())
    v1_lock = protocol["v1_evidence_lock"]
    assert v1_lock["v1_protocol_sha256"] == core.sha256_file(run_v2.V1_PROTOCOL)
    assert v1_lock["v1_manifest_sha256"] == core.sha256_file(
        PAPER14_ROOT / "results/pathmnist_external_validation_v1/manifest.json"
    )
    assert v1_lock["v1_victim_metrics_sha256"] == core.sha256_file(
        PAPER14_ROOT / "results/pathmnist_external_validation_v1/victim_metrics.csv"
    )
    assert v1_lock["v1_verification_sha256"] == core.sha256_file(
        PAPER14_ROOT
        / "results/top_journal_program_v1/pathmnist_external_validation_verification.json"
    )
    return observed


def verify_stage(stage: str, source: Path, dataset: Path) -> dict[str, Any]:
    seeds = STAGE_SEEDS[stage]
    manifest = verify_v1.read_json(source / "manifest.json")
    assert manifest["experiment"] == "pathmnist_external_validation_v2"
    assert manifest["stage"] == stage
    assert tuple(manifest["seeds"]) == seeds
    assert manifest["protocol_sha256"] == core.sha256_file(PROTOCOL)
    assert manifest["dataset_sha256"] == core.sha256_file(dataset)
    assert manifest["poison_count"] == run_v2.POISON_COUNT
    assert manifest["patch_size"] == run_v2.PATCH_SIZE
    metrics = verify_v1.read_csv(source / "per_config_metrics.csv")
    victims = verify_v1.read_csv(source / "victim_metrics.csv")
    receipts = verify_v1.read_json(source / "seed_receipts.json")
    assert len(metrics) == len(seeds) * len(run_v2.VIEWS) * len(run_v2.METHODS)
    assert len(victims) == len(receipts) == len(seeds)
    by_metric = {
        (int(row["seed"]), row["view"], row["method"]): row for row in metrics
    }
    eligible = {int(row["seed"]) for row in victims if int(row["eligible"]) == 1}
    assert set(int(row["seed"]) for row in victims) == set(seeds)
    receipt_by_seed = {int(row["seed"]): row for row in receipts}
    with np.load(dataset, allow_pickle=False) as payload:
        train_labels = payload["train_labels"].reshape(-1).astype(np.int64)
    subset_indices = verify_v1.recompute_subset(train_labels)
    subset_labels = train_labels[subset_indices]
    replayed_rows = 0
    spectral_float32_boundary_ambiguities = 0
    pooled: dict[tuple[str, str], dict[str, int]] = {}
    pooled_eligible: dict[tuple[str, str], dict[str, int]] = {}

    for seed in seeds:
        receipt = receipt_by_seed[seed]
        expected_poison = np.zeros(len(subset_indices), dtype=bool)
        candidates = np.flatnonzero(subset_labels != core.TARGET_CLASS)
        rng = np.random.default_rng(20260930 + seed)
        chosen = np.sort(
            rng.choice(candidates, size=run_v2.POISON_COUNT, replace=False)
        )
        expected_poison[chosen] = True
        sample_path = source / "sample_artifacts" / f"seed_{seed}.npz"
        assert core.sha256_file(sample_path) == receipt["sample_artifact"]["sha256"]
        with np.load(sample_path, allow_pickle=False) as payload:
            original_index = payload["original_index"].astype(np.int64)
            original = payload["original_label"].astype(np.int64)
            assigned = payload["assigned_label"].astype(np.int64)
            truth = payload["is_poison"].astype(bool)
            assert np.array_equal(original_index, subset_indices)
            assert np.array_equal(original, subset_labels)
            assert np.array_equal(truth, expected_poison)
            assert np.all(assigned[truth] == core.TARGET_CLASS)
            assert np.array_equal(assigned[~truth], original[~truth])
            for view in run_v2.VIEWS:
                for method in run_v2.METHODS:
                    row = by_metric[(seed, view, method)]
                    scores = payload[
                        run_v1.artifact_key(view, method, "scores")
                    ].astype(np.float64)
                    p_values = payload[
                        run_v1.artifact_key(view, method, "p_values")
                    ].astype(np.float64)
                    flags = payload[
                        run_v1.artifact_key(view, method, "flagged")
                    ].astype(bool)
                    defined = assigned == core.TARGET_CLASS if method == "cf_cmd_u" else np.ones(len(truth), dtype=bool)
                    assert np.isfinite(scores[defined]).all()
                    if method == "cf_cmd_u":
                        assert np.isnan(scores[~defined]).all()
                        assert np.array_equal(flags[defined], p_values[defined] <= core.ALPHA)
                        assert not flags[~defined].any()
                    elif method == "backdoorbench_spectral":
                        method_receipt = json.loads(row["receipt_json"])
                        replay_flags = np.zeros(len(flags), dtype=bool)
                        boundary_distance = np.full(len(flags), np.inf, dtype=np.float64)
                        for label in range(core.N_CLASSES):
                            mask = assigned == label
                            threshold = float(method_receipt["thresholds"][str(label)])
                            replay_flags[mask] = scores[mask] > threshold
                            boundary_distance[mask] = np.abs(scores[mask] - threshold)
                        disagreements = flags != replay_flags
                        assert np.all(boundary_distance[disagreements] <= 2e-6)
                        spectral_float32_boundary_ambiguities += int(disagreements.sum())
                    clean = ~truth
                    counts = {
                        "tp": int(np.sum(flags & truth)),
                        "fn": int(np.sum(~flags & truth)),
                        "fp": int(np.sum(flags & clean)),
                        "tn": int(np.sum(~flags & clean)),
                    }
                    assert all(int(row[name]) == value for name, value in counts.items())
                    key = (view, method)
                    for target in (pooled, pooled_eligible if seed in eligible else None):
                        if target is None:
                            continue
                        accumulator = target.setdefault(
                            key, {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
                        )
                        for name, value in counts.items():
                            accumulator[name] += value
                    replayed_rows += len(truth)
        for role in ("clean", "suspect"):
            checkpoint_path = source / "models" / f"seed_{seed}_{role}.pt"
            expected = receipt["checkpoints"][role]
            assert core.sha256_file(checkpoint_path) == expected["file_sha256"]
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            assert checkpoint["seed"] == seed and checkpoint["role"] == role
            assert checkpoint["protocol_sha256"] == core.sha256_file(PROTOCOL)
            assert verify_v1.state_fingerprint(checkpoint["state_dict"]) == expected[
                "state_sha256"
            ]

    def summarize(values: dict[tuple[str, str], dict[str, int]]) -> dict[str, Any]:
        output = {}
        for (view, method), counts in values.items():
            output[f"{view}/{method}"] = {
                **counts,
                "tpr": counts["tp"] / (counts["tp"] + counts["fn"]),
                "fpr": counts["fp"] / (counts["fp"] + counts["tn"]),
            }
        return output

    return {
        "stage": stage,
        "seeds": list(seeds),
        "eligible_seeds": sorted(eligible),
        "sample_decisions_replayed": replayed_rows,
        "checkpoints_verified": 2 * len(seeds),
        "poison_selections_reconstructed": len(seeds),
        "spectral_float32_boundary_ambiguities": spectral_float32_boundary_ambiguities,
        "pooled_all_attempted": summarize(pooled),
        "pooled_eligible_only": summarize(pooled_eligible),
        "gate": manifest["gate"],
    }


def main(args: argparse.Namespace) -> None:
    hashes = verify_frozen_inputs(args.dataset)
    stages = [verify_stage("development", args.development, args.dataset)]
    if args.include_confirmation:
        stages.append(verify_stage("confirmation", args.confirmation, args.dataset))
    receipt = {
        "status": "PASS",
        "protocol": str(PROTOCOL),
        "protocol_sha256": core.sha256_file(PROTOCOL),
        "observed_frozen_hashes": hashes,
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
    parser.add_argument("--dataset", type=Path, default=run_v2.DEFAULT_DATASET)
    parser.add_argument("--development", type=Path, default=run_v2.DEFAULT_DEVELOPMENT)
    parser.add_argument("--confirmation", type=Path, default=run_v2.DEFAULT_CONFIRMATION)
    parser.add_argument("--include-confirmation", action="store_true")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    main(parser.parse_args())
