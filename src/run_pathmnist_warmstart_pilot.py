#!/usr/bin/env python3
"""Coordinate-aligned warm-start PathMNIST pilot on archived clean victims."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

import pathmnist_external_validation as core
import run_pathmnist_external_validation as common
import run_pathmnist_external_validation_v2 as v2


PAPER14_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/pathmnist_warmstart_protocol_v1.json"
)
SOURCE_PROTOCOL = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/pathmnist_external_validation_protocol_v2.json"
)
SOURCE_ROOT = PAPER14_ROOT / "results/pathmnist_external_validation_v2"
DEFAULT_DATASET = Path("/private/tmp/paper14-data/pathmnist.npz")
DEFAULT_OUTPUT = PAPER14_ROOT / "results/pathmnist_warmstart_pilot_v1"
SEEDS = (86, 87)
EPOCHS = 3
LEARNING_RATE = 3e-4
RUNTIME_BUDGET_SECONDS = 600.0
VIEWS = common.VIEWS
METHODS = common.METHODS


def validate_inputs(dataset: Path) -> dict[str, Any]:
    protocol = common.read_json(PROTOCOL)
    frozen = protocol["frozen_inputs"]
    observed = {
        "dataset_sha256": core.sha256_file(dataset),
        "core_implementation_sha256": core.sha256_file(Path(core.__file__)),
        "common_runner_sha256": core.sha256_file(Path(common.__file__)),
        "v2_runner_sha256": core.sha256_file(Path(v2.__file__)),
        "warmstart_runner_sha256": core.sha256_file(Path(__file__)),
        "source_protocol_sha256": core.sha256_file(SOURCE_PROTOCOL),
        "source_manifest_sha256": core.sha256_file(SOURCE_ROOT / "manifest.json"),
        "source_receipts_sha256": core.sha256_file(SOURCE_ROOT / "seed_receipts.json"),
    }
    mismatches = {
        key: {"expected": frozen.get(key), "actual": value}
        for key, value in observed.items()
        if frozen.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"frozen input mismatch: {mismatches}")
    return protocol


def load_archived_clean(
    seed: int, mean: np.ndarray, std: np.ndarray
) -> tuple[core.PathResidualCNN, dict[str, Any]]:
    receipts = common.read_json(SOURCE_ROOT / "seed_receipts.json")
    receipt = next(row for row in receipts if int(row["seed"]) == seed)
    checkpoint = SOURCE_ROOT / "models" / f"seed_{seed}_clean.pt"
    expected = receipt["checkpoints"]["clean"]
    if core.sha256_file(checkpoint) != expected["file_sha256"]:
        raise RuntimeError(f"source checkpoint hash mismatch for seed {seed}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("seed") != seed
        or payload.get("role") != "clean"
        or payload.get("protocol_sha256") != core.sha256_file(SOURCE_PROTOCOL)
        or not np.array_equal(payload.get("mean"), mean)
        or not np.array_equal(payload.get("std"), std)
    ):
        raise RuntimeError(f"source checkpoint metadata mismatch for seed {seed}")
    model = core.PathResidualCNN(mean, std)
    model.load_state_dict(payload["state_dict"])
    if core.state_fingerprint(model.state_dict()) != expected["state_sha256"]:
        raise RuntimeError(f"source state mismatch for seed {seed}")
    return model, {
        "path": str(checkpoint),
        "file_sha256": expected["file_sha256"],
        "state_sha256": expected["state_sha256"],
        "source_training": payload["training"],
    }


def validation_metrics(
    model: core.PathResidualCNN,
    data: dict[str, dict[str, np.ndarray]],
) -> dict[str, float]:
    prediction = core.predict(model, data["val"]["images"])
    balanced = core.balanced_accuracy(data["val"]["labels"], prediction)
    non_target = data["val"]["labels"] != core.TARGET_CLASS
    triggered = core.apply_patch(data["val"]["images"][non_target], v2.PATCH_SIZE)
    attack_prediction = core.predict(model, triggered)
    return {
        "validation_balanced_accuracy": balanced,
        "validation_attack_success_rate": float(
            np.mean(attack_prediction == core.TARGET_CLASS)
        ),
    }


def fine_tune(
    clean_model: core.PathResidualCNN,
    pool: dict[str, Any],
    data: dict[str, dict[str, np.ndarray]],
    *,
    seed: int,
) -> tuple[core.PathResidualCNN, dict[str, Any]]:
    model = copy.deepcopy(clean_model)
    initial_hash = core.state_fingerprint(model.state_dict())
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(pool["images"].transpose(0, 3, 1, 2)),
        torch.as_tensor(pool["assigned_labels"], dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(20280930 + seed),
    )
    augmentation = torch.Generator().manual_seed(20290930 + seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    history = []
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb.float().div_(255.0)
            horizontal = torch.rand(len(xb), generator=augmentation) < 0.5
            vertical = torch.rand(len(xb), generator=augmentation) < 0.5
            xb[horizontal] = torch.flip(xb[horizontal], dims=(3,))
            xb[vertical] = torch.flip(xb[vertical], dims=(2,))
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(xb)
            count += len(xb)
        history.append(
            {
                "epoch": epoch,
                "train_cross_entropy": loss_sum / count,
                **validation_metrics(model, data),
            }
        )
    return model, {
        "initial_state_sha256": initial_hash,
        "final_state_sha256": core.state_fingerprint(model.state_dict()),
        "epochs_completed": EPOCHS,
        "fixed_final_epoch": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "elapsed_seconds": time.time() - started,
        "history": history,
    }


def save_suspect(
    path: Path,
    model: core.PathResidualCNN,
    training: dict[str, Any],
    source: dict[str, Any],
    *,
    seed: int,
    mean: np.ndarray,
    std: np.ndarray,
    protocol_hash: str,
) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": "path_residual_cnn_hidden64_77977_parameters",
            "seed": seed,
            "role": "warmstart_suspect",
            "mean": mean,
            "std": std,
            "protocol_sha256": protocol_hash,
            "source_clean": source,
            "training": training,
        },
        path,
    )
    return {
        "path": str(path),
        "file_sha256": core.sha256_file(path),
        "state_sha256": core.state_fingerprint(model.state_dict()),
    }


def replay_artifact(
    path: Path, rows: list[dict[str, Any]], seed: int, subset_indices: np.ndarray
) -> bool:
    selected = [row for row in rows if int(row["seed"]) == seed]
    expected = {(row["view"], row["method"]): row for row in selected}
    with np.load(path, allow_pickle=False) as payload:
        truth = payload["is_poison"].astype(bool)
        original = payload["original_label"].astype(np.int64)
        assigned = payload["assigned_label"].astype(np.int64)
        if (
            len(expected) != len(VIEWS) * len(METHODS)
            or not np.array_equal(payload["original_index"], subset_indices)
            or int(truth.sum()) != v2.POISON_COUNT
            or not np.all(assigned[truth] == core.TARGET_CLASS)
            or not np.array_equal(assigned[~truth], original[~truth])
        ):
            return False
        for key, row in expected.items():
            flags = payload[common.artifact_key(*key, "flagged")].astype(bool)
            clean = ~truth
            counts = {
                "tp": int(np.sum(flags & truth)),
                "fn": int(np.sum(~flags & truth)),
                "fp": int(np.sum(flags & clean)),
                "tn": int(np.sum(~flags & clean)),
            }
            if any(int(row[name]) != value for name, value in counts.items()):
                return False
    return True


def run(args: argparse.Namespace) -> dict[str, Any]:
    if tuple(args.seeds) != SEEDS:
        raise ValueError(f"warm-start pilot is frozen to seeds {SEEDS}")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = validate_inputs(args.dataset)
    protocol_hash = core.sha256_file(PROTOCOL)
    data = core.load_pathmnist(args.dataset)
    subset = core.select_balanced_training_subset(
        data["train"]["images"], data["train"]["labels"]
    )
    mean, std = core.channel_stats(data["train"]["images"])
    metric_rows: list[dict[str, Any]] = []
    victim_rows: list[dict[str, Any]] = []
    seed_receipts: list[dict[str, Any]] = []
    finite_outputs = True
    started_all = time.time()

    for seed in SEEDS:
        started = time.time()
        print(f"seed={seed}: load archived clean and fixed 3-epoch warm start", flush=True)
        clean_model, source = load_archived_clean(seed, mean, std)
        pool = core.build_poisoned_training(
            subset["images"],
            subset["labels"],
            seed=seed,
            poison_count=v2.POISON_COUNT,
            patch_size=v2.PATCH_SIZE,
            original_indices=subset["original_indices"],
        )
        suspect_model, training = fine_tune(clean_model, pool, data, seed=seed)
        if training["initial_state_sha256"] != source["state_sha256"]:
            raise RuntimeError("warm-start coordinate alignment failed")
        checkpoint = save_suspect(
            args.output / "models" / f"seed_{seed}_warmstart_suspect.pt",
            suspect_model,
            training,
            source,
            seed=seed,
            mean=mean,
            std=std,
            protocol_hash=protocol_hash,
        )
        clean_prediction = core.predict(clean_model, data["test"]["images"])
        clean_source_ca = core.balanced_accuracy(data["test"]["labels"], clean_prediction)
        victim = v2.victim_metrics(suspect_model, data)
        victim_rows.append(
            {
                "seed": seed,
                "source_clean_external_balanced_ca": clean_source_ca,
                **victim,
                "fixed_finetune_epochs": EPOCHS,
            }
        )
        print(
            f"seed={seed}: source CA={clean_source_ca:.3f}, warm CA={victim['balanced_clean_accuracy']:.3f}, "
            f"ASR={victim['attack_success_rate']:.3f}, eligible={victim['eligible']}",
            flush=True,
        )

        validation_clean = core.hidden_features(clean_model, data["val"]["images"])
        validation_suspect = core.hidden_features(suspect_model, data["val"]["images"])
        candidate_clean = core.hidden_features(clean_model, pool["images"])
        candidate_suspect = core.hidden_features(suspect_model, pool["images"])
        views = {
            "suspect_hidden64": (validation_suspect, candidate_suspect),
            "clean_to_suspect_hidden64_drift": (
                validation_suspect - validation_clean,
                candidate_suspect - candidate_clean,
            ),
        }
        artifact: dict[str, np.ndarray] = {
            "original_index": pool["original_index"],
            "original_label": pool["original_labels"].astype(np.uint8),
            "assigned_label": pool["assigned_labels"].astype(np.uint8),
            "is_poison": pool["is_poison"].astype(np.uint8),
        }
        for view_index, view in enumerate(VIEWS):
            print(f"seed={seed}: detectors on {view}", flush=True)
            reference, candidates = views[view]
            outputs = common.run_methods(
                reference,
                data["val"]["labels"],
                candidates,
                pool["assigned_labels"],
                seed=seed + 1000 * view_index,
            )
            for method in METHODS:
                output = outputs[method]
                flags = np.asarray(output["suspicious_mask"], dtype=bool)
                scores = np.asarray(output["scores"], dtype=np.float64)
                p_values = np.asarray(output["p_values"], dtype=np.float64)
                defined = (
                    pool["assigned_labels"] == core.TARGET_CLASS
                    if method == "cf_cmd_u"
                    else np.ones(len(pool["images"]), dtype=bool)
                )
                finite_outputs = finite_outputs and bool(np.isfinite(scores[defined]).all())
                metrics = core.evaluate_flags(
                    flags,
                    scores,
                    pool["is_poison"],
                    pool["assigned_labels"],
                    pool["original_labels"],
                )
                metric_rows.append(
                    {
                        "seed": seed,
                        "view": view,
                        "method": method,
                        "eligible_victim": victim["eligible"],
                        "dimension": int(candidates.shape[1]),
                        **metrics,
                        "receipt_json": json.dumps(
                            output["receipt"], sort_keys=True, separators=(",", ":")
                        ),
                    }
                )
                artifact[common.artifact_key(view, method, "scores")] = scores.astype(
                    np.float32
                )
                artifact[common.artifact_key(view, method, "p_values")] = p_values.astype(
                    np.float32
                )
                artifact[common.artifact_key(view, method, "flagged")] = flags.astype(
                    np.uint8
                )
        sample_path = args.output / "sample_artifacts" / f"seed_{seed}.npz"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sample_path, **artifact)
        elapsed = time.time() - started
        seed_receipts.append(
            {
                "seed": seed,
                "elapsed_seconds": elapsed,
                "within_runtime_budget": bool(elapsed <= RUNTIME_BUDGET_SECONDS),
                "pool": pool["receipt"],
                "source_clean": source,
                "warmstart_training": training,
                "checkpoint": checkpoint,
                "sample_artifact": {
                    "path": str(sample_path),
                    "sha256": core.sha256_file(sample_path),
                },
            }
        )
        common.write_csv(args.output / "per_config_metrics.partial.csv", metric_rows)
        common.write_csv(args.output / "victim_metrics.partial.csv", victim_rows)
        common.write_json(args.output / "seed_receipts.partial.json", seed_receipts)

    replay = all(
        replay_artifact(
            args.output / "sample_artifacts" / f"seed_{seed}.npz",
            metric_rows,
            seed,
            subset["original_indices"],
        )
        for seed in SEEDS
    )
    gate = {
        "both_source_clean_models_eligible": bool(
            all(row["source_clean_external_balanced_ca"] >= 0.75 for row in victim_rows)
        ),
        "both_warmstart_victims_eligible": bool(
            all(row["eligible"] for row in victim_rows)
        ),
        "coordinate_alignment_verified": bool(
            all(
                row["warmstart_training"]["initial_state_sha256"]
                == row["source_clean"]["state_sha256"]
                for row in seed_receipts
            )
        ),
        "all_method_outputs_finite_where_defined": bool(finite_outputs),
        "sample_artifacts_replay": bool(replay),
        "both_seeds_within_runtime_budget": bool(
            all(row["within_runtime_budget"] for row in seed_receipts)
        ),
    }
    gate["authorize_fresh_seed_confirmation_protocol"] = bool(all(gate.values()))
    common.write_csv(args.output / "per_config_metrics.csv", metric_rows)
    common.write_csv(args.output / "victim_metrics.csv", victim_rows)
    common.write_json(args.output / "seed_receipts.json", seed_receipts)
    manifest = {
        "experiment": "pathmnist_warmstart_pilot_v1",
        "protocol": str(PROTOCOL),
        "protocol_sha256": protocol_hash,
        "source_protocol_sha256": core.sha256_file(SOURCE_PROTOCOL),
        "dataset_sha256": protocol["frozen_inputs"]["dataset_sha256"],
        "seeds": list(SEEDS),
        "fixed_finetune_epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "poison_count": v2.POISON_COUNT,
        "patch_size": v2.PATCH_SIZE,
        "views": list(VIEWS),
        "methods": list(METHODS),
        "runtime_seconds": time.time() - started_all,
        "gate": gate,
        "interpretation": {
            "role": "development-only coordinate-aligned warm-start pilot on previously observed clean checkpoints",
            "not_confirmation": True,
            "fresh_confirmation_required": True,
            "paired_drift_is_oracle": True,
        },
    }
    common.write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    run(parser.parse_args())
