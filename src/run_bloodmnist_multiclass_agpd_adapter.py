#!/usr/bin/env python3
"""Run the frozen feature-level AGPD adapter on archived BloodMNIST victims.

No victim is retrained.  Development and confirmation seeds, references,
thresholds, layers, and expansion gates are fixed by the protocol JSON.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import agpd_feature_adapter as agpd
from run_bloodmnist_multiclass_backdoorbench_pilot import (
    N_CLASSES,
    POISON_COUNT,
    TARGET_CLASS,
    MulticlassBloodCNN,
    balanced_accuracy,
    build_poisoned_training,
    channel_stats,
    evaluate_flags,
    load_bloodmnist,
    nchw,
    per_class_recall,
    predict,
    sha256_file,
    state_fingerprint,
)


PAPER14_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/multiclass_agpd_adapter_protocol_v1.json"
)
SOURCE_PROTOCOL = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/multiclass_backdoorbench_pilot_protocol.json"
)
SOURCE_RUNNER = PAPER14_ROOT / "src/run_bloodmnist_multiclass_backdoorbench_pilot.py"
ADAPTER = PAPER14_ROOT / "src/agpd_feature_adapter.py"
UPSTREAM_AGPD = PAPER14_ROOT / "external/BackdoorBench/detection_pretrain/agpd.py"
UPSTREAM_UTILS = PAPER14_ROOT / "external/BackdoorBench/detection_pretrain/agpd_utils.py"
UPSTREAM_CONFIG = (
    PAPER14_ROOT / "external/BackdoorBench/config/detection/agpd/cifar10.yaml"
)
SOURCE_DEVELOPMENT = (
    PAPER14_ROOT / "results/bloodmnist_multiclass_backdoorbench_pilot_v1"
)
SOURCE_CONFIRMATION = (
    PAPER14_ROOT / "results/bloodmnist_multiclass_backdoorbench_confirmation_v1"
)
DEFAULT_DEVELOPMENT = PAPER14_ROOT / "results/bloodmnist_multiclass_agpd_adapter_v1"
DEFAULT_CONFIRMATION = (
    PAPER14_ROOT / "results/bloodmnist_multiclass_agpd_adapter_confirmation_v1"
)
DEVELOPMENT_SEEDS = (72, 73)
CONFIRMATION_SEEDS = (74, 75, 76, 77, 78)
LAYERS = ("features_net.0", "features_net.3")
REFERENCE_PER_CLASS = 12
REFERENCE_SEED_OFFSET = 20270929
RUNTIME_BUDGET_SECONDS = 20.0 * 60.0


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_frozen_inputs(dataset_path: Path) -> dict[str, Any]:
    protocol = read_json(PROTOCOL)
    expected = {
        dataset_path: protocol["dataset"]["sha256"],
        SOURCE_PROTOCOL: protocol["victims"]["source_protocol_sha256"],
        SOURCE_RUNNER: protocol["victims"]["source_runner_sha256"],
        ADAPTER: protocol["adapter"]["implementation_sha256"],
        UPSTREAM_AGPD: protocol["upstream"]["agpd_py_sha256"],
        UPSTREAM_UTILS: protocol["upstream"]["agpd_utils_py_sha256"],
        UPSTREAM_CONFIG: protocol["upstream"]["cifar10_config_sha256"],
    }
    mismatches = {
        str(path): {"expected": digest, "actual": sha256_file(path)}
        for path, digest in expected.items()
        if not path.exists() or sha256_file(path) != digest
    }
    if mismatches:
        raise RuntimeError(f"frozen input hash mismatch: {mismatches}")
    return protocol


def source_root(stage: str) -> Path:
    return SOURCE_DEVELOPMENT if stage == "development" else SOURCE_CONFIRMATION


def validate_source_manifest(stage: str, seeds: tuple[int, ...]) -> None:
    payload = read_json(source_root(stage) / "manifest.json")
    if not (
        payload.get("experiment") == "bloodmnist_multiclass_backdoorbench_pilot_v1"
        and payload.get("stage") == stage
        and tuple(payload.get("seeds", ())) == seeds
        and payload.get("protocol_sha256") == sha256_file(SOURCE_PROTOCOL)
        and payload.get("dataset_npz_sha256")
        == "062023e186f537e26b3c21ea3b2614ddfc475e8a14825dfd20663bbb7e37bddc"
    ):
        raise RuntimeError("archived source manifest does not match the frozen victim set")


def archived_receipt(stage: str, seed: int) -> dict[str, Any]:
    receipts = read_json(source_root(stage) / "seed_receipts.json")
    matches = [row for row in receipts if int(row["seed"]) == int(seed)]
    if len(matches) != 1:
        raise RuntimeError(f"missing or duplicate archived receipt for seed {seed}")
    return matches[0]


def load_archived_model(
    stage: str, seed: int, expected_mean: np.ndarray, expected_std: np.ndarray
) -> tuple[MulticlassBloodCNN, dict[str, Any]]:
    receipt = archived_receipt(stage, seed)
    checkpoint = source_root(stage) / "models" / f"seed_{seed}_suspect.pt"
    expected_checkpoint = receipt["checkpoints"]["suspect"]
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != expected_checkpoint["file_sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch for seed {seed}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("seed") != seed
        or payload.get("role") != "suspect"
        or payload.get("protocol_sha256") != sha256_file(SOURCE_PROTOCOL)
    ):
        raise RuntimeError(f"checkpoint metadata mismatch for seed {seed}")
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    if not np.array_equal(mean, expected_mean) or not np.array_equal(std, expected_std):
        raise RuntimeError(f"checkpoint preprocessing mismatch for seed {seed}")
    model = MulticlassBloodCNN(mean, std)
    model.load_state_dict(payload["state_dict"])
    state_hash = state_fingerprint(model.state_dict())
    if state_hash != expected_checkpoint["state_sha256"]:
        raise RuntimeError(f"checkpoint state mismatch for seed {seed}")
    return model, {
        "path": str(checkpoint),
        "file_sha256": checkpoint_hash,
        "state_sha256": state_hash,
        "source_receipt_pool": receipt["pool"],
    }


def choose_reference(
    model: MulticlassBloodCNN,
    images: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    predictions = predict(model, images)
    rng_seed = REFERENCE_SEED_OFFSET + int(seed)
    rng = np.random.default_rng(rng_seed)
    selected_parts = []
    correct_counts = {}
    for label in range(N_CLASSES):
        eligible = np.flatnonzero((labels == label) & (predictions == label))
        correct_counts[str(label)] = int(len(eligible))
        if len(eligible) < REFERENCE_PER_CLASS:
            raise RuntimeError(
                f"seed {seed} class {label} has {len(eligible)} correct validation rows"
            )
        selected_parts.append(
            np.sort(rng.choice(eligible, size=REFERENCE_PER_CLASS, replace=False))
        )
    selected = np.concatenate(selected_parts).astype(np.int64, copy=False)
    return images[selected], labels[selected], {
        "rng_seed": rng_seed,
        "selected_indices": selected.tolist(),
        "selected_indices_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
        "correct_validation_counts": correct_counts,
        "reference_count": int(len(selected)),
    }


def victim_metrics(
    model: MulticlassBloodCNN,
    data: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    prediction = predict(model, data["test"]["images"])
    clean_balanced = balanced_accuracy(data["test"]["labels"], prediction)
    non_target = data["test"]["labels"] != TARGET_CLASS
    triggered = data["test"]["images"][non_target].copy()
    triggered[:, -3:, -3:, :] = 1.0
    attack_prediction = predict(model, triggered)
    return {
        "balanced_clean_accuracy": clean_balanced,
        "attack_success_rate": float(np.mean(attack_prediction == TARGET_CLASS)),
        "eligible": int(clean_balanced >= 0.80 and np.mean(attack_prediction == TARGET_CLASS) >= 0.80),
        "per_class_recall_json": json.dumps(
            per_class_recall(data["test"]["labels"], prediction),
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def confirmation_is_authorized(development_root: Path) -> bool:
    path = development_root / "manifest.json"
    if not path.exists():
        return False
    payload = read_json(path)
    gate = payload.get("gate", {})
    required = (
        "all_archived_victims_eligible",
        "all_frozen_hashes_match",
        "references_have_12_correct_per_class",
        "all_layer_class_gradients_finite",
        "all_index_mappings_and_confusions_replay",
        "all_seeds_within_runtime_budget",
        "expand_to_confirmation",
    )
    return bool(
        payload.get("experiment") == "bloodmnist_multiclass_agpd_adapter_v1"
        and payload.get("stage") == "development"
        and payload.get("protocol_sha256") == sha256_file(PROTOCOL)
        and payload.get("seeds") == list(DEVELOPMENT_SEEDS)
        and all(gate.get(key) is True for key in required)
    )


def replay_rows(
    sample_rows: list[dict[str, Any]], metrics_rows: list[dict[str, Any]], candidate_count: int
) -> bool:
    by_seed = {int(row["seed"]): row for row in metrics_rows}
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in sample_rows:
        grouped.setdefault(int(row["seed"]), []).append(row)
    if set(by_seed) != set(grouped):
        return False
    for seed, rows in grouped.items():
        if len(rows) != candidate_count:
            return False
        ordered = sorted(rows, key=lambda row: int(row["sample_index"]))
        indices = np.asarray([int(row["sample_index"]) for row in ordered])
        original = np.asarray([int(row["original_index"]) for row in ordered])
        original_label = np.asarray([int(row["original_label"]) for row in ordered])
        assigned = np.asarray([int(row["assigned_label"]) for row in ordered])
        poison = np.asarray([bool(int(row["is_poison"])) for row in ordered])
        flagged = np.asarray([bool(int(row["flagged"])) for row in ordered])
        identity = np.arange(candidate_count)
        if not (
            np.array_equal(indices, identity)
            and np.array_equal(original, identity)
            and int(poison.sum()) == POISON_COUNT
            and np.all(original_label[poison] != TARGET_CLASS)
            and np.all(assigned[poison] == TARGET_CLASS)
            and np.array_equal(assigned[~poison], original_label[~poison])
        ):
            return False
        clean = ~poison
        replayed = {
            "tp": int(np.sum(flagged & poison)),
            "fn": int(np.sum(~flagged & poison)),
            "fp": int(np.sum(flagged & clean)),
            "tn": int(np.sum(~flagged & clean)),
        }
        if any(replayed[key] != int(by_seed[seed][key]) for key in replayed):
            return False
    return True


def run(args: argparse.Namespace) -> dict[str, Any]:
    expected_seeds = DEVELOPMENT_SEEDS if args.stage == "development" else CONFIRMATION_SEEDS
    seeds = tuple(int(value) for value in args.seeds)
    if seeds != expected_seeds:
        raise ValueError(f"{args.stage} is frozen to seeds {expected_seeds}")
    if args.stage == "confirmation" and not confirmation_is_authorized(args.development_root):
        raise RuntimeError("development expansion gate has not passed")
    args.output.mkdir(parents=True, exist_ok=True)
    dataset_path = args.dataset_root / "bloodmnist.npz"
    protocol = validate_frozen_inputs(dataset_path)
    validate_source_manifest(args.stage, seeds)
    data = load_bloodmnist(args.dataset_root)
    mean, std = channel_stats(data["train"]["images"])
    metrics_rows: list[dict[str, Any]] = []
    victim_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    started_all = time.time()

    for seed in seeds:
        started = time.time()
        print(f"seed={seed}: verify archived victim", flush=True)
        pool = build_poisoned_training(data["train"]["images"], data["train"]["labels"], seed=seed)
        model, checkpoint_receipt = load_archived_model(args.stage, seed, mean, std)
        source_pool = checkpoint_receipt["source_receipt_pool"]
        if source_pool["selected_indices_sha256"] != pool["receipt"]["selected_indices_sha256"]:
            raise RuntimeError(f"poison-pool replay mismatch for seed {seed}")
        victim = {"stage": args.stage, "seed": seed, **victim_metrics(model, data)}
        victim_rows.append(victim)
        reference_x, reference_y, reference_receipt = choose_reference(
            model, data["val"]["images"], data["val"]["labels"], seed
        )
        print(f"seed={seed}: collect AGPD activation gradients", flush=True)
        result = agpd.run_agpd_adapter(
            model,
            torch.as_tensor(nchw(pool["images"]), dtype=torch.float32),
            pool["assigned_labels"],
            torch.as_tensor(nchw(reference_x), dtype=torch.float32),
            reference_y,
            LAYERS,
            N_CLASSES,
            tau=float(protocol["adapter"]["progressive_tau"]),
            anomaly_threshold=float(protocol["adapter"]["class_anomaly_threshold"]),
            fallback_rho_threshold=float(protocol["adapter"]["fallback_rho_threshold"]),
            window_size=int(protocol["adapter"]["progressive_window"]),
            batch_size=args.batch_size,
            device=args.device,
        )
        metrics = evaluate_flags(
            result.flags,
            pool["is_poison"],
            pool["assigned_labels"],
            pool["original_labels"],
        )
        elapsed = time.time() - started
        precision = metrics["tp"] / max(1, metrics["tp"] + metrics["fp"])
        metrics_rows.append(
            {
                "stage": args.stage,
                "seed": seed,
                "method": "backdoorbench_agpd_feature_adapter",
                "selected_layer": result.selected_layer or "",
                "target_classes_json": json.dumps(list(result.target_classes), separators=(",", ":")),
                "eligible_victim": victim["eligible"],
                **metrics,
                "precision": precision,
                "repair_activations_json": json.dumps(list(result.repair_activations), separators=(",", ":")),
                "elapsed_seconds": elapsed,
            }
        )
        for layer in LAYERS:
            for label in range(N_CLASSES):
                layer_rows.append(
                    {
                        "seed": seed,
                        "layer": layer,
                        "class": label,
                        "rho": float(result.rho_by_layer[layer][label]),
                        "one_sided_mad": float(result.anomaly_by_layer[layer][label]),
                        "selected_layer": int(result.selected_layer == layer),
                        "selected_target_class": int(label in result.target_classes),
                        "progressive_iterations": result.progressive_iterations.get(label, ""),
                    }
                )
        for index, flag in enumerate(result.flags):
            sample_rows.append(
                {
                    "seed": seed,
                    "sample_index": index,
                    "original_index": int(pool["original_index"][index]),
                    "original_label": int(pool["original_labels"][index]),
                    "assigned_label": int(pool["assigned_labels"][index]),
                    "is_poison": int(pool["is_poison"][index]),
                    "flagged": int(flag),
                }
            )
        receipts.append(
            {
                "seed": seed,
                "elapsed_seconds": elapsed,
                "within_runtime_budget": bool(elapsed <= RUNTIME_BUDGET_SECONDS),
                "checkpoint": checkpoint_receipt,
                "pool": pool["receipt"],
                "reference": reference_receipt,
                "assigned_class_counts": {
                    str(label): int(np.sum(pool["assigned_labels"] == label))
                    for label in range(N_CLASSES)
                },
                "selected_layer": result.selected_layer,
                "target_classes": list(result.target_classes),
                "repair_activations": list(result.repair_activations),
            }
        )
        print(
            f"seed={seed}: TPR={metrics['tpr']:.3f}, FPR={metrics['fpr_all_clean']:.3f}, "
            f"layer={result.selected_layer}, targets={result.target_classes}, elapsed={elapsed:.1f}s",
            flush=True,
        )

    mapping_replay = replay_rows(sample_rows, metrics_rows, len(data["train"]["images"]))
    finite = all(
        np.isfinite(row["rho"]) and np.isfinite(row["one_sided_mad"])
        for row in layer_rows
    )
    gate = {
        "all_archived_victims_eligible": all(bool(row["eligible"]) for row in victim_rows),
        "all_frozen_hashes_match": True,
        "references_have_12_correct_per_class": all(
            row["reference"]["reference_count"] == N_CLASSES * REFERENCE_PER_CLASS
            and min(row["reference"]["correct_validation_counts"].values()) >= REFERENCE_PER_CLASS
            for row in receipts
        ),
        "all_layer_class_gradients_finite": bool(finite),
        "all_index_mappings_and_confusions_replay": bool(mapping_replay),
        "all_seeds_within_runtime_budget": all(row["within_runtime_budget"] for row in receipts),
    }
    if args.stage == "development":
        gate["expand_to_confirmation"] = all(gate.values())
    else:
        gate["confirmation_complete"] = all(gate.values())

    write_csv(args.output / "victim_metrics.csv", victim_rows)
    write_csv(args.output / "per_config_metrics.csv", metrics_rows)
    write_csv(args.output / "per_layer_class_diagnostics.csv", layer_rows)
    write_csv(args.output / "per_sample_flags.csv", sample_rows)
    write_json(args.output / "seed_receipts.json", receipts)
    manifest = {
        "experiment": "bloodmnist_multiclass_agpd_adapter_v1",
        "stage": args.stage,
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(Path(__file__)),
        "adapter_sha256": sha256_file(ADAPTER),
        "dataset_npz_sha256": sha256_file(dataset_path),
        "source_protocol_sha256": sha256_file(SOURCE_PROTOCOL),
        "seeds": list(seeds),
        "layers": list(LAYERS),
        "candidate_count_per_seed": int(len(data["train"]["images"])),
        "reference_count_per_seed": N_CLASSES * REFERENCE_PER_CLASS,
        "gate": gate,
        "elapsed_seconds": time.time() - started_all,
        "fidelity_scope": protocol["upstream"]["scope"],
    }
    write_json(args.output / "manifest.json", manifest)
    lines = [
        f"# BloodMNIST multiclass AGPD feature adapter ({args.stage})",
        "",
        f"Non-performance gate: **{'PASS' if all(gate.values()) else 'STOP'}**",
        "",
        "| seed | balanced CA | ASR | selected layer | target classes | TPR | all-clean FPR | precision |",
        "|---:|---:|---:|---|---|---:|---:|---:|",
    ]
    for victim, metrics in zip(victim_rows, metrics_rows):
        lines.append(
            f"| {victim['seed']} | {victim['balanced_clean_accuracy']:.3f} | "
            f"{victim['attack_success_rate']:.3f} | {metrics['selected_layer'] or 'none'} | "
            f"{metrics['target_classes_json']} | {metrics['tpr']:.3f} | "
            f"{metrics['fpr_all_clean']:.3f} | {metrics['precision']:.3f} |"
        )
    lines.extend(
        [
            "",
            "This is a declared feature-level architecture adapter of pinned BackdoorBench AGPD, not a bit-identical end-to-end run.",
            "Victim weights are archived and unchanged; poison truth is used only after flags are frozen for evaluation.",
            "Detection performance is not an expansion gate.",
            "",
        ]
    )
    (args.output / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    parser.add_argument("--stage", choices=("development", "confirmation"), default="development")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--development-root", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.seeds is None:
        args.seeds = list(DEVELOPMENT_SEEDS if args.stage == "development" else CONFIRMATION_SEEDS)
    if args.output is None:
        args.output = DEFAULT_DEVELOPMENT if args.stage == "development" else DEFAULT_CONFIRMATION
    run(args)


if __name__ == "__main__":
    main()
