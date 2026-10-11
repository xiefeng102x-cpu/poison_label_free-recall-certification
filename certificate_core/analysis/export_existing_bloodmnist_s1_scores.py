#!/usr/bin/env python3
"""Export an auditable S1 score table from archived BloodMNIST victims.

No model is trained.  The script replays the frozen member/fresh experiment,
adds the poison rows that were not written to its query-only CSV, and emits the
six-column hand-off consumed by ``s1_analysis_template.py``.

The exported population is the detector-defined target-assigned stratum, not
the complete eight-class training set.  This is a retrospective feasibility
block and must not be described as the preregistered S1 experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
CERT_ROOT = HERE.parents[1]
PAPER14_ROOT = CERT_ROOT.parent
SRC = PAPER14_ROOT / "src"
sys.path.insert(0, str(SRC))

import run_bloodmnist_fresh_query_calibration as fresh  # noqa: E402
from run_bloodmnist_multiclass_backdoorbench_pilot import (  # noqa: E402
    TARGET_CLASS,
    build_poisoned_training,
    channel_stats,
    hidden_features,
    load_bloodmnist,
)


SEEDS = (74, 75, 76, 77, 78)
VIEWS = ("suspect_hidden64", "paired_hidden64_drift")
SOURCE = PAPER14_ROOT / "results/bloodmnist_fresh_query_calibration_confirmation_v1"
SOURCE_QUERY = SOURCE / "per_query_scores.csv"
SOURCE_MANIFEST = SOURCE / "manifest.json"
SOURCE_RECEIPTS = SOURCE / "seed_receipts.json"
SOURCE_VICTIMS = SOURCE / "victim_metrics.csv"
DEFAULT_OUTPUT = CERT_ROOT / "results_real_s1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_rows(path: Path) -> str:
    return sha256_file(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def expected_clean_query_map() -> dict[tuple[int, str, str, int, int], float]:
    table = pd.read_csv(SOURCE_QUERY)
    if len(table) != 29_960:
        raise RuntimeError(f"unexpected source query row count: {len(table)}")
    if set(table["population"]) != {"member_train_clean", "fresh_test_clean"}:
        raise RuntimeError("unexpected source query populations")
    mapping: dict[tuple[int, str, str, int, int], float] = {}
    for row in table.itertuples(index=False):
        key = (int(row.seed), str(row.view), str(row.source_split), int(row.fold), int(row.source_index))
        if key in mapping:
            raise RuntimeError(f"duplicate source query key: {key}")
        mapping[key] = float(row.raw_score)
    return mapping


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ("victim_id", "fold", "sample_id", "score", "role", "is_poison")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def export(dataset_root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if not (
        source_manifest.get("stage") == "confirmation"
        and source_manifest.get("seeds") == list(SEEDS)
        and all(source_manifest.get("gate", {}).values())
    ):
        raise RuntimeError("source confirmation manifest is incomplete or failed")
    victims = pd.read_csv(SOURCE_VICTIMS)
    if len(victims) != len(SEEDS) or not (victims["eligible"] == 1).all():
        raise RuntimeError("all five archived victims must be eligible")

    data = load_bloodmnist(dataset_root)
    mean, std = channel_stats(data["train"]["images"])
    clean_query = expected_clean_query_map()
    seen_clean_query: set[tuple[int, str, str, int, int]] = set()
    rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []

    for seed in SEEDS:
        pool = build_poisoned_training(data["train"]["images"], data["train"]["labels"], seed=seed)
        clean_model, clean_receipt = fresh.load_model("confirmation", seed, "clean", mean, std)
        suspect_model, suspect_receipt = fresh.load_model("confirmation", seed, "suspect", mean, std)
        design = fresh.split_design(pool, data["val"]["labels"], data["test"]["labels"], seed)
        suspect_features = {
            "train": hidden_features(suspect_model, pool["images"]),
            "val": hidden_features(suspect_model, data["val"]["images"]),
            "test": hidden_features(suspect_model, data["test"]["images"]),
        }
        clean_features = {
            "train": hidden_features(clean_model, pool["images"]),
            "val": hidden_features(clean_model, data["val"]["images"]),
            "test": hidden_features(clean_model, data["test"]["images"]),
        }
        view_features = {
            "suspect_hidden64": suspect_features,
            "paired_hidden64_drift": {
                split: suspect_features[split] - clean_features[split]
                for split in ("train", "val", "test")
            },
        }
        seed_receipt: dict[str, Any] = {
            "seed": seed,
            "pool_selected_indices_sha256": pool["receipt"]["selected_indices_sha256"],
            "checkpoints": {"clean": clean_receipt, "suspect": suspect_receipt},
            "views": {},
        }
        for view, features in view_features.items():
            reference_mean = features["val"][design["reference"]].mean(axis=0)
            view_receipt: dict[str, Any] = {"folds": {}}
            victim_id = f"bloodmnist_badnets_multiclasscnn_{view}_seed{seed}"
            for fold in (0, 1):
                direction, direction_norm = fresh.unit_direction(
                    features["train"][design["candidate_folds"][1 - fold]], reference_mean
                )
                train_indices = design["candidate_folds"][fold]
                fresh_indices = design["fresh_folds"][fold]
                train_scores = fresh.score(features["train"][train_indices], reference_mean, direction)
                fresh_scores = fresh.score(features["test"][fresh_indices], reference_mean, direction)

                clean_train_count = 0
                poison_count = 0
                for source_index, raw_score in zip(train_indices, train_scores):
                    poison = int(pool["is_poison"][source_index])
                    poison_count += poison
                    clean_train_count += 1 - poison
                    if poison == 0:
                        key = (seed, view, "train", fold, int(source_index))
                        expected = clean_query.get(key)
                        if expected is None or not np.isclose(raw_score, expected, atol=1e-10, rtol=0.0):
                            raise RuntimeError(f"member score replay mismatch: {key}")
                        seen_clean_query.add(key)
                    rows.append(
                        {
                            "victim_id": victim_id,
                            "fold": fold,
                            "sample_id": int(source_index),
                            "score": repr(float(raw_score)),
                            "role": "train",
                            "is_poison": poison,
                        }
                    )
                for source_index, raw_score in zip(fresh_indices, fresh_scores):
                    key = (seed, view, "test", fold, int(source_index))
                    expected = clean_query.get(key)
                    if expected is None or not np.isclose(raw_score, expected, atol=1e-10, rtol=0.0):
                        raise RuntimeError(f"fresh score replay mismatch: {key}")
                    seen_clean_query.add(key)
                    rows.append(
                        {
                            "victim_id": victim_id,
                            "fold": fold,
                            "sample_id": 1_000_000 + int(source_index),
                            "score": repr(float(raw_score)),
                            "role": "fresh",
                            "is_poison": 0,
                        }
                    )
                if not (
                    len(train_indices) == 1_465
                    and clean_train_count + poison_count == 1_465
                    and poison_count > 0
                    and len(fresh_indices) == 333
                ):
                    raise RuntimeError("frozen target-stratum count invariant failed")
                view_receipt["folds"][str(fold)] = {
                    "direction_norm": direction_norm,
                    "direction_sha256": array_hash(np.asarray(direction, dtype=np.float64)),
                    "train_indices_sha256": array_hash(np.asarray(train_indices, dtype=np.int64)),
                    "fresh_indices_sha256": array_hash(np.asarray(fresh_indices, dtype=np.int64)),
                    "train_scores_sha256": array_hash(np.asarray(train_scores, dtype=np.float64)),
                    "fresh_scores_sha256": array_hash(np.asarray(fresh_scores, dtype=np.float64)),
                    "train_count": len(train_indices),
                    "clean_train_count": clean_train_count,
                    "poison_count": poison_count,
                    "fresh_count": len(fresh_indices),
                }
            seed_receipt["views"][view] = view_receipt
        receipts.append(seed_receipt)

    if seen_clean_query != set(clean_query):
        missing = len(set(clean_query) - seen_clean_query)
        extra = len(seen_clean_query - set(clean_query))
        raise RuntimeError(f"source clean-query coverage mismatch: missing={missing}, extra={extra}")

    score_path = output / "bloodmnist_s1_scores.csv"
    write_csv(score_path, rows)
    frame = pd.DataFrame(rows)
    duplicates = int(frame.duplicated(["victim_id", "fold", "sample_id"]).sum())
    if duplicates:
        raise RuntimeError(f"duplicate hand-off keys: {duplicates}")
    counts = (
        frame.groupby(["victim_id", "fold", "role", "is_poison"])
        .size()
        .reset_index(name="rows")
    )
    counts.to_csv(output / "bloodmnist_s1_score_counts.csv", index=False)
    manifest = {
        "artifact": "retrospective_bloodmnist_s1_score_handoff_v1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "READY_FOR_RETROSPECTIVE_CERTIFICATE_ANALYSIS",
        "scope": {
            "dataset": "BloodMNIST",
            "attack": "BadNets 3x3 patch, all-to-one target class 6",
            "victim_architecture": "MulticlassBloodCNN",
            "seeds": list(SEEDS),
            "views": list(VIEWS),
            "folds": [0, 1],
            "population": "target-assigned detector-defined stratum only",
            "not_full_training_population": True,
            "retrospective_not_preregistered": True,
        },
        "row_count": len(frame),
        "victim_id_count": int(frame["victim_id"].nunique()),
        "score_table": str(score_path),
        "score_table_sha256": sha256_rows(score_path),
        "counts_sha256": sha256_file(output / "bloodmnist_s1_score_counts.csv"),
        "source_files": {
            str(path): sha256_file(path)
            for path in (SOURCE_QUERY, SOURCE_MANIFEST, SOURCE_RECEIPTS, SOURCE_VICTIMS)
        },
        "source_manifest_sha256_recorded": source_manifest.get("protocol_sha256"),
        "clean_and_fresh_score_replay": "all 29,960 archived query rows matched at atol=1e-10",
        "receipts": receipts,
    }
    write_json(output / "export_manifest.json", manifest)
    (output / "SCOPE_AND_LIMITATIONS.md").write_text(
        "# Scope and limitations\n\n"
        "This is a retrospective feasibility analysis using five archived eligible "
        "BloodMNIST BadNets victims and two frozen score views. It is not the prospectively "
        "preregistered S1 matrix because the victims predate PREREG_S1_S4.md.\n\n"
        "The certificate population is the complete detector-defined target-assigned stratum "
        "for each fold (clean target-class training members plus all 600 poison records). "
        "It is not the complete eight-class training set. Fresh rows are target-class official "
        "test samples that trained neither checkpoint nor score direction.\n\n"
        "All archived member/fresh raw scores were independently replayed; poison scores use "
        "the same frozen opposite-fold direction and reference mean.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = export(args.dataset_root, args.output.resolve())
    print(json.dumps({key: manifest[key] for key in ("status", "row_count", "victim_id_count", "score_table_sha256")}, indent=2))


if __name__ == "__main__":
    main()
