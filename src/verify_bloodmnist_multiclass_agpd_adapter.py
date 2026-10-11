#!/usr/bin/env python3
"""Independent artifact replay for the frozen BloodMNIST AGPD adapter runs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_bloodmnist_multiclass_agpd_adapter import (
    ADAPTER,
    CONFIRMATION_SEEDS,
    DEFAULT_CONFIRMATION,
    DEFAULT_DEVELOPMENT,
    DEVELOPMENT_SEEDS,
    N_CLASSES,
    PAPER14_ROOT,
    PROTOCOL,
    REFERENCE_PER_CLASS,
    REFERENCE_SEED_OFFSET,
    SOURCE_CONFIRMATION,
    SOURCE_DEVELOPMENT,
    SOURCE_PROTOCOL,
)
from run_bloodmnist_multiclass_backdoorbench_pilot import (
    POISON_COUNT,
    TARGET_CLASS,
    MulticlassBloodCNN,
    build_poisoned_training,
    channel_stats,
    load_bloodmnist,
    predict,
    sha256_file,
    state_fingerprint,
)


DATASET_ROOT = Path("/private/tmp/blood-ca-data")
RECEIPT = (
    PAPER14_ROOT / "results/top_journal_program_v1/multiclass_agpd_adapter_verification.json"
)
STAGES = {
    "development": (DEFAULT_DEVELOPMENT, SOURCE_DEVELOPMENT, DEVELOPMENT_SEEDS),
    "confirmation": (DEFAULT_CONFIRMATION, SOURCE_CONFIRMATION, CONFIRMATION_SEEDS),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assert_close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
        raise AssertionError(f"{name}: {actual} != {expected}")


def expected_reference_indices(
    model: MulticlassBloodCNN,
    validation_images: np.ndarray,
    validation_labels: np.ndarray,
    seed: int,
) -> np.ndarray:
    predictions = predict(model, validation_images)
    rng = np.random.default_rng(REFERENCE_SEED_OFFSET + seed)
    selected = []
    for label in range(N_CLASSES):
        eligible = np.flatnonzero(
            (validation_labels == label) & (predictions == label)
        )
        if len(eligible) < REFERENCE_PER_CLASS:
            raise AssertionError(f"seed {seed} lacks reference rows for class {label}")
        selected.append(
            np.sort(rng.choice(eligible, size=REFERENCE_PER_CLASS, replace=False))
        )
    return np.concatenate(selected).astype(np.int64, copy=False)


def verify_stage(
    stage: str,
    output_root: Path,
    source_root: Path,
    seeds: tuple[int, ...],
    data: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    manifest = read_json(output_root / "manifest.json")
    if not (
        manifest["experiment"] == "bloodmnist_multiclass_agpd_adapter_v1"
        and manifest["stage"] == stage
        and tuple(manifest["seeds"]) == seeds
        and manifest["protocol_sha256"] == sha256_file(PROTOCOL)
        and manifest["adapter_sha256"] == sha256_file(ADAPTER)
        and manifest["source_protocol_sha256"] == sha256_file(SOURCE_PROTOCOL)
        and manifest["dataset_npz_sha256"]
        == sha256_file(DATASET_ROOT / "bloodmnist.npz")
        and all(manifest["gate"].values())
    ):
        raise AssertionError(f"{stage} manifest mismatch")

    metrics_rows = read_csv(output_root / "per_config_metrics.csv")
    sample_rows = read_csv(output_root / "per_sample_flags.csv")
    diagnostics = read_csv(output_root / "per_layer_class_diagnostics.csv")
    victim_rows = read_csv(output_root / "victim_metrics.csv")
    receipts = read_json(output_root / "seed_receipts.json")
    source_receipts = read_json(source_root / "seed_receipts.json")
    if not (
        len(metrics_rows) == len(seeds)
        and len(victim_rows) == len(seeds)
        and len(receipts) == len(seeds)
        and len(diagnostics) == len(seeds) * 2 * N_CLASSES
        and len(sample_rows) == len(seeds) * len(data["train"]["images"])
    ):
        raise AssertionError(f"{stage} artifact row count mismatch")

    metrics_by_seed = {int(row["seed"]): row for row in metrics_rows}
    victim_by_seed = {int(row["seed"]): row for row in victim_rows}
    receipt_by_seed = {int(row["seed"]): row for row in receipts}
    source_by_seed = {int(row["seed"]): row for row in source_receipts}
    sample_by_seed: dict[int, list[dict[str, str]]] = {}
    for row in sample_rows:
        sample_by_seed.setdefault(int(row["seed"]), []).append(row)
    if not all(set(mapping) == set(seeds) for mapping in (
        metrics_by_seed, victim_by_seed, receipt_by_seed, source_by_seed, sample_by_seed
    )):
        raise AssertionError(f"{stage} seed mapping mismatch")

    pooled = {name: 0 for name in ("tp", "fn", "fp", "tn")}
    seed_summaries = []
    mean, std = channel_stats(data["train"]["images"])
    for seed in seeds:
        pool = build_poisoned_training(
            data["train"]["images"], data["train"]["labels"], seed=seed
        )
        source_receipt = source_by_seed[seed]
        if pool["receipt"]["selected_indices_sha256"] != source_receipt["pool"]["selected_indices_sha256"]:
            raise AssertionError(f"seed {seed} poison construction mismatch")
        checkpoint = source_root / "models" / f"seed_{seed}_suspect.pt"
        if sha256_file(checkpoint) != source_receipt["checkpoints"]["suspect"]["file_sha256"]:
            raise AssertionError(f"seed {seed} checkpoint file mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not np.array_equal(payload["mean"], mean) or not np.array_equal(payload["std"], std):
            raise AssertionError(f"seed {seed} preprocessing mismatch")
        model = MulticlassBloodCNN(mean, std)
        model.load_state_dict(payload["state_dict"])
        if state_fingerprint(model.state_dict()) != source_receipt["checkpoints"]["suspect"]["state_sha256"]:
            raise AssertionError(f"seed {seed} state mismatch")
        reference_indices = expected_reference_indices(
            model, data["val"]["images"], data["val"]["labels"], seed
        )
        reference_receipt = receipt_by_seed[seed]["reference"]
        if not (
            reference_receipt["rng_seed"] == REFERENCE_SEED_OFFSET + seed
            and reference_receipt["selected_indices"] == reference_indices.tolist()
            and reference_receipt["selected_indices_sha256"]
            == hashlib.sha256(reference_indices.tobytes()).hexdigest()
        ):
            raise AssertionError(f"seed {seed} reference selection mismatch")

        rows = sorted(sample_by_seed[seed], key=lambda row: int(row["sample_index"]))
        index = np.asarray([int(row["sample_index"]) for row in rows])
        original_index = np.asarray([int(row["original_index"]) for row in rows])
        original_label = np.asarray([int(row["original_label"]) for row in rows])
        assigned_label = np.asarray([int(row["assigned_label"]) for row in rows])
        poison = np.asarray([bool(int(row["is_poison"])) for row in rows])
        flagged = np.asarray([bool(int(row["flagged"])) for row in rows])
        identity = np.arange(len(rows))
        if not (
            np.array_equal(index, identity)
            and np.array_equal(original_index, identity)
            and np.array_equal(original_label, pool["original_labels"])
            and np.array_equal(assigned_label, pool["assigned_labels"])
            and np.array_equal(poison, pool["is_poison"])
        ):
            raise AssertionError(f"seed {seed} sample provenance mismatch")
        clean = ~poison
        counts = {
            "tp": int(np.sum(flagged & poison)),
            "fn": int(np.sum(~flagged & poison)),
            "fp": int(np.sum(flagged & clean)),
            "tn": int(np.sum(~flagged & clean)),
        }
        row = metrics_by_seed[seed]
        if any(counts[name] != int(row[name]) for name in counts):
            raise AssertionError(f"seed {seed} confusion mismatch")
        tpr = counts["tp"] / (counts["tp"] + counts["fn"])
        fpr = counts["fp"] / (counts["fp"] + counts["tn"])
        target_clean = clean & (original_label == TARGET_CLASS)
        target_fpr = float(np.mean(flagged[target_clean]))
        precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
        assert_close(tpr, float(row["tpr"]), f"seed {seed} TPR")
        assert_close(fpr, float(row["fpr_all_clean"]), f"seed {seed} FPR")
        assert_close(target_fpr, float(row["target_clean_fpr"]), f"seed {seed} target FPR")
        assert_close(precision, float(row["precision"]), f"seed {seed} precision")
        if int(row["flagged_count"]) != int(flagged.sum()):
            raise AssertionError(f"seed {seed} flagged count mismatch")
        for name in pooled:
            pooled[name] += counts[name]
        seed_summaries.append(
            {
                "seed": seed,
                "tpr": tpr,
                "fpr_all_clean": fpr,
                "precision": precision,
                "selected_layer": row["selected_layer"],
                "target_classes": json.loads(row["target_classes_json"]),
            }
        )

    for row in diagnostics:
        if not (
            np.isfinite(float(row["rho"]))
            and np.isfinite(float(row["one_sided_mad"]))
            and row["layer"] in ("features_net.0", "features_net.3")
            and 0 <= int(row["class"]) < N_CLASSES
        ):
            raise AssertionError(f"{stage} nonfinite or malformed diagnostic")

    files = (
        "manifest.json",
        "victim_metrics.csv",
        "per_config_metrics.csv",
        "per_layer_class_diagnostics.csv",
        "per_sample_flags.csv",
        "seed_receipts.json",
        "EXPERIMENT_REPORT.md",
    )
    return {
        "stage": stage,
        "seeds": list(seeds),
        "pooled_tpr": pooled["tp"] / (pooled["tp"] + pooled["fn"]),
        "pooled_fpr_all_clean": pooled["fp"] / (pooled["fp"] + pooled["tn"]),
        "pooled_counts": pooled,
        "seed_summaries": seed_summaries,
        "artifact_sha256": {name: sha256_file(output_root / name) for name in files},
    }


def main() -> None:
    data = load_bloodmnist(DATASET_ROOT)
    summaries = [
        verify_stage(stage, output, source, seeds, data)
        for stage, (output, source, seeds) in STAGES.items()
    ]
    payload = {
        "verification": "PASS",
        "protocol_sha256": sha256_file(PROTOCOL),
        "adapter_sha256": sha256_file(ADAPTER),
        "verifier_sha256": sha256_file(Path(__file__)),
        "dataset_npz_sha256": sha256_file(DATASET_ROOT / "bloodmnist.npz"),
        "checks": [
            "frozen protocol, adapter, source protocol, dataset and checkpoint hashes",
            "poison-pool and validation-reference deterministic reconstruction",
            "per-sample provenance, confusion matrices and reported rates",
            "finite two-layer eight-class diagnostics and non-performance gates",
        ],
        "stages": summaries,
    }
    RECEIPT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for summary in summaries:
        print(
            f"{summary['stage']}: pooled TPR={summary['pooled_tpr']:.6f}, "
            f"FPR={summary['pooled_fpr_all_clean']:.6f}"
        )
    print(f"PASS: wrote {RECEIPT}")


if __name__ == "__main__":
    main()
