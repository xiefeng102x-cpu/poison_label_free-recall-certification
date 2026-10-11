#!/usr/bin/env python3
"""Export PathMNIST and GTSRB primary scores to the six-column certificate API.

Victim/model protocols are prospective; this conversion is the post-endpoint
analysis lock documented in PREREG_AMENDMENT_03.md.  No model is trained.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


CERT_ROOT = Path(__file__).resolve().parents[1]
PAPER14 = CERT_ROOT.parent
SRC = PAPER14 / "src"
sys.path.insert(0, str(SRC))

import pathmnist_official_resnet18 as path_arch  # noqa: E402
import run_pathmnist_official_weight_detector_confirmation as path_run  # noqa: E402
import run_tifs_native_gtsrb as gtsrb_run  # noqa: E402


DEFAULT_OUTPUT = CERT_ROOT / "results_prospective_blocks_v1"
PATH_SOURCE = PAPER14 / "results/pathmnist_official_weight_detector_confirmation_v1"
GTSRB_SOURCE = PAPER14 / "results/tifs_native_cifar_gtsrb_v1/gtsrb"
GTSRB_PROTOCOL = PAPER14 / "configs/tifs_native_gtsrb_confirmation_v1_1.json"
PATH_DATASET = Path("/private/tmp/paper14-data/pathmnist.npz")
AMENDMENT = CERT_ROOT / "PREREG_AMENDMENT_03.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ("victim_id", "fold", "sample_id", "score", "role", "is_poison")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _crossfit(target_indices: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    shuffled = np.random.default_rng(20271001 + seed).permutation(target_indices)
    left, right = np.array_split(shuffled, 2)
    return left.astype(np.int64), right.astype(np.int64)


def export_pathmnist(dataset: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    manifest = json.loads((PATH_SOURCE / "manifest.json").read_text(encoding="utf-8"))
    replay = json.loads(
        (PAPER14 / "results/top_journal_program_v1/pathmnist_terminal_staged_replay_v2_receipt_2026-10-04.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.get("status") != "COMPLETE_FORMAL" or not manifest.get("all_pairs_eligible"):
        raise RuntimeError("PathMNIST formal source is not complete and fully eligible")
    if replay.get("status") != "PASS" or replay.get("failures"):
        raise RuntimeError("PathMNIST independent replay did not pass")
    with np.load(dataset, allow_pickle=False) as archive:
        test_images = archive["test_images"]
        test_labels = archive["test_labels"].reshape(-1).astype(np.int64)
    fresh_indices = np.flatnonzero(test_labels == path_run.TARGET_CLASS).astype(np.int64)
    if not len(fresh_indices):
        raise RuntimeError("PathMNIST target-class fresh pool is empty")

    rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for seed in path_run.CONFIRMATION_SEEDS:
        feature_path = PATH_SOURCE / "features" / f"seed_{seed}.npz"
        with np.load(feature_path, allow_pickle=False) as feature_file:
            candidates = np.asarray(feature_file["suspect_candidate"], dtype=np.float64)
            reference = np.asarray(feature_file["suspect_reference"], dtype=np.float64)
            assigned = feature_file["assigned_labels"].astype(np.int64)
            original = feature_file["original_labels"].astype(np.int64)
        target = np.flatnonzero(assigned == path_run.TARGET_CLASS).astype(np.int64)
        halves = _crossfit(target, seed)
        ref_mean = reference.mean(axis=0)
        model = path_arch.PathMNISTResNet18()
        state = torch.load(
            PATH_SOURCE / "models" / f"seed_{seed}_suspect_epoch5.pt",
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(state, strict=True)
        model.eval()
        fresh_features = path_run._hidden(model, test_images[fresh_indices], torch.device("cpu"))
        truth = assigned != original
        victim_id = f"pathmnist_badnets_resnet18_primary_cfcmdu_seed{seed}"
        fold_receipts: dict[str, Any] = {}
        for fold, (scoring, fitting) in enumerate(((halves[0], halves[1]), (halves[1], halves[0]))):
            direction = candidates[fitting].mean(axis=0) - ref_mean
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                raise RuntimeError(f"PathMNIST seed {seed} fold {fold}: collapsed direction")
            direction /= norm
            train_scores = (candidates[scoring] - ref_mean) @ direction
            fresh_scores = (fresh_features - ref_mean) @ direction
            for index, score in zip(scoring, train_scores):
                rows.append(
                    {
                        "victim_id": victim_id,
                        "fold": fold,
                        "sample_id": int(index),
                        "score": repr(float(score)),
                        "role": "train",
                        "is_poison": int(truth[index]),
                    }
                )
            for index, score in zip(fresh_indices, fresh_scores):
                rows.append(
                    {
                        "victim_id": victim_id,
                        "fold": fold,
                        "sample_id": 1_000_000 + int(index),
                        "score": repr(float(score)),
                        "role": "fresh",
                        "is_poison": 0,
                    }
                )
            fold_receipts[str(fold)] = {
                "direction_norm": norm,
                "direction_sha256": array_hash(direction.astype(np.float64)),
                "train_indices_sha256": array_hash(scoring),
                "fresh_indices_sha256": array_hash(fresh_indices),
                "train_scores_sha256": array_hash(train_scores.astype(np.float64)),
                "fresh_scores_sha256": array_hash(fresh_scores.astype(np.float64)),
                "train_count": int(len(scoring)),
                "clean_train_count": int((~truth[scoring]).sum()),
                "poison_count": int(truth[scoring].sum()),
                "fresh_count": int(len(fresh_indices)),
            }
        receipts.append(
            {
                "seed": seed,
                "feature_file_sha256": sha256_file(feature_path),
                "suspect_model_sha256": sha256_file(
                    PATH_SOURCE / "models" / f"seed_{seed}_suspect_epoch5.pt"
                ),
                "fresh_inference_device": "cpu",
                "folds": fold_receipts,
            }
        )
    return rows, receipts


def export_gtsrb() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = json.loads(GTSRB_PROTOCOL.read_text(encoding="utf-8"))
    verification = json.loads((GTSRB_SOURCE / "verification_receipt.json").read_text(encoding="utf-8"))
    if verification.get("status") != "PASS" or verification.get("eligible_seed_count") != 10:
        raise RuntimeError("GTSRB formal source is not verified with ten eligible victims")
    data_root, _ = gtsrb_run.validate_dataset(protocol)
    cache, _ = gtsrb_run.build_resized_cache(data_root, lambda _: None)
    rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for seed in protocol["seeds"]:
        seed = int(seed)
        data = gtsrb_run.build_seed_data(cache["train_images"], cache["train_labels"], protocol, seed)
        gtsrb_run.validate_seed_data(data, cache["train_labels"], protocol)
        checkpoint_path = GTSRB_SOURCE / "models" / f"seed_{seed}_suspect.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = gtsrb_run.GTSRBTinyCNN()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        _, candidates = gtsrb_run.predict_and_features(
            model, data.candidate_images, 512, torch.device("cpu"), True
        )
        if candidates is None:
            raise RuntimeError("GTSRB candidate feature extraction failed")
        reference, _calibration, reserved, trusted = gtsrb_run.trusted_features(
            model, cache, protocol, seed, 512, torch.device("cpu")
        )
        target = np.arange(len(candidates), dtype=np.int64)
        halves = _crossfit(target, seed)
        ref_mean = reference.astype(np.float64).mean(axis=0)
        values = candidates.astype(np.float64)
        fresh_values = reserved.astype(np.float64)
        victim_id = f"gtsrb_badnets_tinycnn43_primary_cfcmdu_seed{seed}"
        fold_receipts: dict[str, Any] = {}
        for fold, (scoring, fitting) in enumerate(((halves[0], halves[1]), (halves[1], halves[0]))):
            direction = values[fitting].mean(axis=0) - ref_mean
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                raise RuntimeError(f"GTSRB seed {seed} fold {fold}: collapsed direction")
            direction /= norm
            train_scores = (values[scoring] - ref_mean) @ direction
            fresh_scores = (fresh_values - ref_mean) @ direction
            for local_index, score in zip(scoring, train_scores):
                rows.append(
                    {
                        "victim_id": victim_id,
                        "fold": fold,
                        "sample_id": int(data.candidate_positions[local_index]),
                        "score": repr(float(score)),
                        "role": "train",
                        "is_poison": int(data.candidate_truth[local_index]),
                    }
                )
            for test_index, score in zip(trusted["reserved_indices"], fresh_scores):
                rows.append(
                    {
                        "victim_id": victim_id,
                        "fold": fold,
                        "sample_id": 1_000_000 + int(test_index),
                        "score": repr(float(score)),
                        "role": "fresh",
                        "is_poison": 0,
                    }
                )
            fold_receipts[str(fold)] = {
                "direction_norm": norm,
                "direction_sha256": array_hash(direction.astype(np.float64)),
                "train_indices_sha256": array_hash(scoring),
                "fresh_indices_sha256": hashlib.sha256(
                    np.asarray(trusted["reserved_indices"], dtype=np.int64).tobytes()
                ).hexdigest(),
                "train_scores_sha256": array_hash(train_scores.astype(np.float64)),
                "fresh_scores_sha256": array_hash(fresh_scores.astype(np.float64)),
                "train_count": int(len(scoring)),
                "clean_train_count": int((~data.candidate_truth[scoring]).sum()),
                "poison_count": int(data.candidate_truth[scoring].sum()),
                "fresh_count": int(len(fresh_scores)),
            }
        receipts.append(
            {
                "seed": seed,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "replay_device": "cpu",
                "trusted_reserved_indices_sha256": fold_receipts["0"]["fresh_indices_sha256"],
                "folds": fold_receipts,
            }
        )
    return rows, receipts


def validate_rows(rows: list[dict[str, Any]], expected_victims: int) -> dict[str, Any]:
    keys = [(row["victim_id"], int(row["fold"]), int(row["sample_id"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate six-column keys")
    victims = sorted({str(row["victim_id"]) for row in rows})
    if len(victims) != expected_victims:
        raise RuntimeError("unexpected victim count")
    if {int(row["fold"]) for row in rows} != {0, 1}:
        raise RuntimeError("two-fold coverage mismatch")
    if {str(row["role"]) for row in rows} != {"train", "fresh"}:
        raise RuntimeError("role coverage mismatch")
    if any(str(row["role"]) == "fresh" and int(row["is_poison"]) for row in rows):
        raise RuntimeError("fresh poison row present")
    if not all(np.isfinite(float(row["score"])) for row in rows):
        raise RuntimeError("non-finite score present")
    return {
        "rows": len(rows),
        "victims": len(victims),
        "folds": [0, 1],
        "train_rows": sum(str(row["role"]) == "train" for row in rows),
        "fresh_rows": sum(str(row["role"]) == "fresh" for row in rows),
        "poison_rows": sum(int(row["is_poison"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pathmnist-dataset", type=Path, default=PATH_DATASET)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    path_rows, path_receipts = export_pathmnist(args.pathmnist_dataset.resolve())
    gtsrb_rows, gtsrb_receipts = export_gtsrb()
    path_file = output / "pathmnist_primary_scores.csv"
    gtsrb_file = output / "gtsrb_primary_scores.csv"
    write_csv(path_file, path_rows)
    write_csv(gtsrb_file, gtsrb_rows)
    manifest = {
        "status": "READY_FOR_LOCKED_POSTHOC_CERTIFICATE_ANALYSIS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "post-endpoint certificate integration over separately prospective victim blocks",
        "amendment_sha256": sha256_file(AMENDMENT),
        "exporter_sha256": sha256_file(Path(__file__)),
        "pathmnist": {
            "checks": validate_rows(path_rows, 5),
            "score_table_sha256": sha256_file(path_file),
            "source_manifest_sha256": sha256_file(PATH_SOURCE / "manifest.json"),
            "receipts": path_receipts,
            "certificate_k": [120, 500, 2000],
        },
        "gtsrb": {
            "checks": validate_rows(gtsrb_rows, 10),
            "score_table_sha256": sha256_file(gtsrb_file),
            "source_manifest_sha256": sha256_file(GTSRB_SOURCE / "manifest.json"),
            "source_verification_sha256": sha256_file(GTSRB_SOURCE / "verification_receipt.json"),
            "receipts": gtsrb_receipts,
            "certificate_k": [120, 500],
        },
    }
    atomic_json(output / "export_manifest.json", manifest)
    print(json.dumps({k: manifest[k]["checks"] for k in ("pathmnist", "gtsrb")}, indent=2))


if __name__ == "__main__":
    main()
