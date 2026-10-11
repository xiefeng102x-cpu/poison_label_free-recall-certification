#!/usr/bin/env python3
"""Run the pre-registered P6-B PathMNIST V2 poison-footprint protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import pathmnist_external_validation as core
import run_pathmnist_external_validation as v1


PAPER14_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/pathmnist_external_validation_protocol_v2.json"
)
V1_PROTOCOL = (
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
DEFAULT_DATASET = Path("/private/tmp/paper14-data/pathmnist.npz")
DEFAULT_DEVELOPMENT = PAPER14_ROOT / "results/pathmnist_external_validation_v2"
DEFAULT_CONFIRMATION = (
    PAPER14_ROOT / "results/pathmnist_external_validation_confirmation_v2"
)
DEVELOPMENT_SEEDS = (86, 87)
CONFIRMATION_SEEDS = (88, 89, 90, 91, 92)
POISON_COUNT = 810
PATCH_SIZE = 5
RUNTIME_BUDGET_SECONDS = 1800.0
VIEWS = v1.VIEWS
METHODS = v1.METHODS


def validate_inputs(dataset: Path) -> dict[str, Any]:
    protocol = v1.read_json(PROTOCOL)
    frozen = protocol["frozen_inputs"]
    observed = {
        "acquisition_protocol_sha256": core.sha256_file(ACQUISITION_PROTOCOL),
        "acquisition_receipt_sha256": core.sha256_file(ACQUISITION_RECEIPT),
        "dataset_sha256": core.sha256_file(dataset),
        "core_implementation_sha256": core.sha256_file(Path(core.__file__)),
        "v1_runner_dependency_sha256": core.sha256_file(Path(v1.__file__)),
        "v2_runner_sha256": core.sha256_file(Path(__file__)),
        "backdoorbench_adapter_sha256": core.sha256_file(
            PAPER14_ROOT / "src/backdoorbench_feature_adapter.py"
        ),
    }
    mismatches = {
        key: {"expected": frozen.get(key), "actual": value}
        for key, value in observed.items()
        if frozen.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"frozen input mismatch: {mismatches}")
    return protocol


def validate_stage_seeds(stage: str, seeds: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    actual = tuple(int(seed) for seed in seeds)
    expected = DEVELOPMENT_SEEDS if stage == "development" else CONFIRMATION_SEEDS
    if actual != expected:
        raise ValueError(f"{stage} is frozen to seeds {expected}")
    return actual


def confirmation_is_authorized(development_root: Path) -> bool:
    path = development_root / "manifest.json"
    if not path.exists():
        return False
    payload = v1.read_json(path)
    gate = payload.get("gate", {})
    required = (
        "both_victims_eligible",
        "all_classes_present",
        "all_method_outputs_finite_where_defined",
        "sample_artifacts_replay",
        "both_seeds_within_runtime_budget",
        "expand_to_confirmation",
    )
    return bool(
        payload.get("experiment") == "pathmnist_external_validation_v2"
        and payload.get("stage") == "development"
        and payload.get("protocol_sha256") == core.sha256_file(PROTOCOL)
        and payload.get("seeds") == list(DEVELOPMENT_SEEDS)
        and all(gate.get(key) is True for key in required)
    )


def victim_metrics(
    model: core.PathResidualCNN, data: dict[str, dict[str, np.ndarray]]
) -> dict[str, Any]:
    prediction = core.predict(model, data["test"]["images"])
    balanced = core.balanced_accuracy(data["test"]["labels"], prediction)
    non_target = data["test"]["labels"] != core.TARGET_CLASS
    triggered = core.apply_patch(data["test"]["images"][non_target], PATCH_SIZE)
    attack_prediction = core.predict(model, triggered)
    asr = float(np.mean(attack_prediction == core.TARGET_CLASS))
    return {
        "balanced_clean_accuracy": balanced,
        "attack_success_rate": asr,
        "eligible": int(balanced >= 0.75 and asr >= 0.80),
        "per_class_recall_json": json.dumps(
            core.per_class_recall(data["test"]["labels"], prediction),
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def replay_seed_artifact(
    path: Path,
    expected_rows: list[dict[str, Any]],
    seed: int,
    subset_indices: np.ndarray,
) -> bool:
    selected = [row for row in expected_rows if int(row["seed"]) == seed]
    expected = {(row["view"], row["method"]): row for row in selected}
    if len(expected) != len(VIEWS) * len(METHODS):
        return False
    with np.load(path, allow_pickle=False) as payload:
        truth = payload["is_poison"].astype(bool)
        original = payload["original_label"].astype(np.int64)
        assigned = payload["assigned_label"].astype(np.int64)
        original_index = payload["original_index"].astype(np.int64)
        if (
            len(truth) != core.TRAIN_PER_CLASS * core.N_CLASSES
            or int(truth.sum()) != POISON_COUNT
            or not np.array_equal(original_index, subset_indices)
            or not np.all(original[truth] != core.TARGET_CLASS)
            or not np.all(assigned[truth] == core.TARGET_CLASS)
            or not np.array_equal(original[~truth], assigned[~truth])
        ):
            return False
        for key, row in expected.items():
            flags = payload[v1.artifact_key(*key, "flagged")].astype(bool)
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
    seeds = validate_stage_seeds(args.stage, args.seeds)
    if args.stage == "confirmation" and not confirmation_is_authorized(
        args.development_root
    ):
        raise RuntimeError("development expansion gate has not passed")
    if args.stage == "confirmation" and args.output.resolve() == args.development_root.resolve():
        raise RuntimeError("confirmation output must not overwrite development artifacts")
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
    all_outputs_valid = True
    started_all = time.time()

    for seed in seeds:
        seed_started = time.time()
        print(f"seed={seed}: constructing V2 poison-footprint pool", flush=True)
        pool = core.build_poisoned_training(
            subset["images"],
            subset["labels"],
            seed=seed,
            poison_count=POISON_COUNT,
            patch_size=PATCH_SIZE,
            original_indices=subset["original_indices"],
        )
        models: dict[str, core.PathResidualCNN] = {}
        training_receipts: dict[str, Any] = {}
        checkpoint_receipts: dict[str, Any] = {}
        reused: dict[str, bool] = {}
        for role, train_x, train_y in (
            ("clean", subset["images"], subset["labels"]),
            ("suspect", pool["images"], pool["assigned_labels"]),
        ):
            print(f"seed={seed}: {role} lightweight residual CNN", flush=True)
            checkpoint = args.output / "models" / f"seed_{seed}_{role}.pt"
            model, training, artifact, was_reused = v1.get_or_train(
                checkpoint,
                train_x,
                train_y,
                data["val"]["images"],
                data["val"]["labels"],
                seed=seed,
                role=role,
                mean=mean,
                std=std,
                protocol_sha256=protocol_hash,
            )
            models[role] = model
            training_receipts[role] = training
            checkpoint_receipts[role] = artifact
            reused[role] = was_reused
        if (
            training_receipts["clean"]["initial_state_sha256"]
            != training_receipts["suspect"]["initial_state_sha256"]
        ):
            raise RuntimeError("paired initialization mismatch")

        victim = victim_metrics(models["suspect"], data)
        victim_rows.append(
            {
                "stage": args.stage,
                "seed": seed,
                **victim,
                "clean_best_epoch": training_receipts["clean"]["best_epoch"],
                "suspect_best_epoch": training_receipts["suspect"]["best_epoch"],
            }
        )
        print(
            f"seed={seed}: external-test balanced CA={victim['balanced_clean_accuracy']:.3f}, "
            f"ASR={victim['attack_success_rate']:.3f}, eligible={victim['eligible']}",
            flush=True,
        )

        validation_clean = core.hidden_features(models["clean"], data["val"]["images"])
        validation_suspect = core.hidden_features(models["suspect"], data["val"]["images"])
        candidate_clean = core.hidden_features(models["clean"], pool["images"])
        candidate_suspect = core.hidden_features(models["suspect"], pool["images"])
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
            print(f"seed={seed}: three detectors on {view}", flush=True)
            reference, candidates = views[view]
            outputs = v1.run_methods(
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
                valid = (
                    flags.shape == (len(pool["images"]),)
                    and scores.shape == flags.shape
                    and p_values.shape == flags.shape
                    and np.isfinite(scores[defined]).all()
                )
                all_outputs_valid = all_outputs_valid and bool(valid)
                metrics = core.evaluate_flags(
                    flags,
                    scores,
                    pool["is_poison"],
                    pool["assigned_labels"],
                    pool["original_labels"],
                )
                metric_rows.append(
                    {
                        "stage": args.stage,
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
                artifact[v1.artifact_key(view, method, "scores")] = scores.astype(
                    np.float32
                )
                artifact[v1.artifact_key(view, method, "p_values")] = p_values.astype(
                    np.float32
                )
                artifact[v1.artifact_key(view, method, "flagged")] = flags.astype(
                    np.uint8
                )
        sample_path = args.output / "sample_artifacts" / f"seed_{seed}.npz"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sample_path, **artifact)
        elapsed = time.time() - seed_started
        seed_receipts.append(
            {
                "seed": seed,
                "elapsed_seconds": elapsed,
                "within_runtime_budget": bool(elapsed <= RUNTIME_BUDGET_SECONDS),
                "subset": subset["receipt"],
                "pool": pool["receipt"],
                "training": training_receipts,
                "checkpoints": checkpoint_receipts,
                "reused_checkpoints": reused,
                "sample_artifact": {
                    "path": str(sample_path),
                    "sha256": core.sha256_file(sample_path),
                },
            }
        )
        v1.write_csv(args.output / "per_config_metrics.partial.csv", metric_rows)
        v1.write_csv(args.output / "victim_metrics.partial.csv", victim_rows)
        v1.write_json(args.output / "seed_receipts.partial.json", seed_receipts)

    sample_replay = all(
        replay_seed_artifact(
            args.output / "sample_artifacts" / f"seed_{seed}.npz",
            metric_rows,
            seed,
            subset["original_indices"],
        )
        for seed in seeds
    )
    all_classes = (
        np.array_equal(
            np.bincount(subset["labels"], minlength=core.N_CLASSES),
            np.full(core.N_CLASSES, core.TRAIN_PER_CLASS),
        )
        and np.all(np.bincount(data["val"]["labels"], minlength=core.N_CLASSES) >= 2)
        and np.all(np.bincount(data["test"]["labels"], minlength=core.N_CLASSES) >= 2)
    )
    gate = {
        "both_victims_eligible": bool(all(row["eligible"] for row in victim_rows)),
        "all_classes_present": bool(all_classes),
        "all_method_outputs_finite_where_defined": bool(all_outputs_valid),
        "sample_artifacts_replay": bool(sample_replay),
        "both_seeds_within_runtime_budget": bool(
            all(row["within_runtime_budget"] for row in seed_receipts)
        ),
    }
    if args.stage == "development":
        gate["expand_to_confirmation"] = bool(all(gate.values()))
    else:
        gate["confirmation_complete"] = bool(all(gate.values()))
    v1.write_csv(args.output / "per_config_metrics.csv", metric_rows)
    v1.write_csv(args.output / "victim_metrics.csv", victim_rows)
    v1.write_json(args.output / "seed_receipts.json", seed_receipts)
    manifest = {
        "experiment": "pathmnist_external_validation_v2",
        "stage": args.stage,
        "protocol": str(PROTOCOL),
        "protocol_sha256": protocol_hash,
        "v1_failure_protocol_sha256": core.sha256_file(V1_PROTOCOL),
        "dataset": str(args.dataset),
        "dataset_sha256": protocol["frozen_inputs"]["dataset_sha256"],
        "seeds": list(seeds),
        "training_subset_count": int(len(subset["labels"])),
        "poison_count": POISON_COUNT,
        "patch_size": PATCH_SIZE,
        "views": list(VIEWS),
        "methods": list(METHODS),
        "runtime_seconds": time.time() - started_all,
        "gate": gate,
        "interpretation": {
            "design_role": "fresh-seed victim-feasibility repair after the retained V1 failure",
            "poison_footprint_proxy_v1": 0.05 * 4 * 4,
            "poison_footprint_proxy_v2": 0.03 * 5 * 5,
            "detection_performance_is_expansion_gate": False,
            "external_test_is_different_clinical_center": True,
            "training_scope": "fixed balanced 27,000-row official-train subset",
            "paired_drift_is_oracle": True,
            "cf_cmd_u_pvalues": "clean-validation rank calibration; exact exchangeability for training members is not claimed",
        },
    }
    v1.write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "confirmation"), default="development")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--development-root", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    arguments = parser.parse_args()
    if arguments.seeds is None:
        arguments.seeds = list(
            DEVELOPMENT_SEEDS if arguments.stage == "development" else CONFIRMATION_SEEDS
        )
    if arguments.output is None:
        arguments.output = (
            DEFAULT_DEVELOPMENT
            if arguments.stage == "development"
            else DEFAULT_CONFIRMATION
        )
    run(arguments)
