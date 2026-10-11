#!/usr/bin/env python3
"""Frozen blind PathMNIST ResNet-18 multi-detector score experiment.

This program stops at a truth-separated score handoff.  The companion
``run_pathmnist_blind_certificate.py`` consumes the handoff and issues certificate
decisions before joining poison truth for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import pathmnist_external_validation as core
import pathmnist_official_resnet18 as architecture
import run_pathmnist_official_weight_detector_confirmation as engine


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pathmnist_blind_e2e_multidetector_certificate_v1.json"
PREREG = ROOT / "auditability_certificate_v1/PREREG_BLIND_E2E_PATHMNIST_V1.md"
FREEZE = ROOT / "auditability_certificate_v1/BLIND_E2E_FREEZE_MANIFEST_V1.json"
DEFAULT_DATASET = Path("/private/tmp/paper14-data/pathmnist.npz")
DEFAULT_OUTPUT = ROOT / "results/pathmnist_blind_e2e_multidetector_certificate_v1"
SMOKE = ROOT / "results/pathmnist_real_input_mps_smoke_v1/smoke_receipt.json"
SMOKE_VERIFY = ROOT / "results/pathmnist_real_input_mps_smoke_v1/smoke_verification.json"
SEEDS = (1201, 1202, 1203, 1204, 1205)
WEIGHT_MAP = {
    1201: "resnet18_28_1.pth",
    1202: "resnet18_28_2.pth",
    1203: "resnet18_28_1.pth",
    1204: "resnet18_28_2.pth",
    1205: "resnet18_28_1.pth",
}
METHODS = (
    "cf_cmd_u",
    "target_centroid",
    "crossfit_spectral",
    "activation_clustering",
    "que_score",
)
ACK = "OPEN_BLIND_PATHMNIST_SEEDS_1201_1205"
TARGET = 8

# Reuse the audited training engine under this protocol.  Its checkpoint contract
# records the engine hash; this orchestrator hash is recorded separately.
engine.PROTOCOL_PATH = CONFIG
engine.CONFIRMATION_SEEDS = SEEDS
engine.WEIGHT_MAP = WEIGHT_MAP


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def merge_csv(inputs: list[Path], output: Path) -> None:
    tmp = output.with_name(output.name + ".tmp")
    with tmp.open("wb") as target:
        for index, path in enumerate(inputs):
            with path.open("rb") as source:
                if index:
                    source.readline()
                shutil.copyfileobj(source, target)
    os.replace(tmp, output)


def load_config() -> dict[str, Any]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if tuple(cfg["formal_seeds"]) != SEEDS:
        raise RuntimeError("formal seed list differs from frozen code")
    if {int(k): v for k, v in cfg["weight_map"].items()} != WEIGHT_MAP:
        raise RuntimeError("weight map differs from frozen code")
    if tuple(cfg["detectors"]) != METHODS:
        raise RuntimeError("detector list differs from frozen code")
    return cfg


def verify_freeze() -> dict[str, Any]:
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    expected = {
        "config_sha256": sha256(CONFIG),
        "prereg_sha256": sha256(PREREG),
        "runner_sha256": sha256(Path(__file__)),
        "certificate_runner_sha256": sha256(ROOT / "auditability_certificate_v1/analysis/run_pathmnist_blind_certificate.py"),
        "certificate_library_sha256": sha256(ROOT / "auditability_certificate_v1/cert/certificate.py"),
    }
    if frozen.get("status") != "FROZEN_BEFORE_UNBLINDING":
        raise RuntimeError("freeze manifest status is not valid")
    if frozen.get("hashes") != expected:
        raise RuntimeError("frozen file hashes do not match the freeze manifest")
    return frozen


def validate_mps_gate(dataset: Path) -> torch.device:
    receipt = json.loads(SMOKE.read_text(encoding="utf-8"))
    verification = json.loads(SMOKE_VERIFY.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS_NONFORMAL_REAL_INPUT_MPS":
        raise RuntimeError("real-input MPS smoke receipt did not pass")
    if verification.get("status") != "PASS_NONFORMAL_REAL_INPUT_MPS" or verification.get("failures"):
        raise RuntimeError("independent MPS smoke verification did not pass")
    if receipt.get("dataset_sha256") != sha256(dataset):
        raise RuntimeError("formal dataset differs from the real-input smoke dataset")
    return engine._mps_device()


def target_validation_indices(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = np.flatnonzero(np.asarray(labels) == TARGET)
    order = np.random.default_rng(20271000).permutation(np.sort(target))
    if len(order) < 1024:
        raise RuntimeError("fewer than 1,024 target-class validation rows")
    return order[:512].astype(np.int64), order[512:1024].astype(np.int64)


def target_folds(assigned: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    target = np.flatnonzero(np.asarray(assigned) == TARGET).astype(np.int64)
    shuffled = np.random.default_rng(20301000 + seed).permutation(target)
    first, second = np.array_split(shuffled, 2)
    return first.astype(np.int64), second.astype(np.int64)


def whitener(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    reference = np.asarray(reference, dtype=np.float64)
    mean = reference.mean(axis=0)
    centered = reference - mean
    covariance = centered.T @ centered / max(len(reference) - 1, 1)
    ridge = max(float(np.trace(covariance) / reference.shape[1]) * 0.1, 1e-6)
    values, vectors = np.linalg.eigh(covariance)
    transform = (vectors * (1.0 / np.sqrt(np.maximum(values, 0.0) + ridge))) @ vectors.T
    return mean, transform, ridge


def fit_score_map(
    method: str,
    reference: np.ndarray,
    fitting: np.ndarray,
    *,
    seed: int,
    whitening: tuple[np.ndarray, np.ndarray, float],
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    reference = np.asarray(reference, dtype=np.float64)
    fitting = np.asarray(fitting, dtype=np.float64)
    ref_mean = reference.mean(axis=0)
    if method == "cf_cmd_u":
        direction = fitting.mean(axis=0) - ref_mean
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            raise RuntimeError("CF-CMD-U direction collapsed")
        direction = direction / norm
        return (lambda values: (np.asarray(values, dtype=np.float64) - ref_mean) @ direction), {
            "direction_norm": norm,
            "direction_sha256": array_hash(direction),
            "transform": "identity",
        }
    if method == "target_centroid":
        return (lambda values: np.linalg.norm(np.asarray(values, dtype=np.float64) - ref_mean, axis=1)), {
            "reference_centroid_sha256": array_hash(ref_mean),
            "transform": "identity",
        }
    if method == "crossfit_spectral":
        center = fitting.mean(axis=0)
        reducer = PCA(n_components=1, svd_solver="randomized", random_state=seed).fit(fitting - center)
        direction = reducer.components_[0].astype(np.float64)
        return (lambda values: ((np.asarray(values, dtype=np.float64) - center) @ direction) ** 2), {
            "fit_center_sha256": array_hash(center),
            "direction_sha256": array_hash(direction),
            "explained_variance_ratio": float(reducer.explained_variance_ratio_[0]),
        }

    white_mean, transform, ridge = whitening
    reference_z = (reference - white_mean) @ transform
    fitting_z = (fitting - white_mean) @ transform
    if method == "activation_clustering":
        dimension = min(16, fitting_z.shape[1], len(reference_z) - 1)
        reducer = PCA(n_components=dimension, svd_solver="randomized", random_state=seed).fit(
            np.concatenate([reference_z, fitting_z], axis=0)
        )
        reference_low = reducer.transform(reference_z)
        fitting_low = reducer.transform(fitting_z)
        cluster = KMeans(n_clusters=2, random_state=seed, n_init=20).fit(fitting_low)
        centers = cluster.cluster_centers_
        reference_center = reference_low.mean(axis=0)
        suspicious = int(np.argmax(np.linalg.norm(centers - reference_center[None, :], axis=1)))
        clean = 1 - suspicious

        def activation_score(values: np.ndarray) -> np.ndarray:
            low = reducer.transform((np.asarray(values, dtype=np.float64) - white_mean) @ transform)
            d_suspicious = np.sum((low - centers[suspicious]) ** 2, axis=1)
            d_clean = np.sum((low - centers[clean]) ** 2, axis=1)
            return d_clean - d_suspicious

        return activation_score, {
            "whitening_ridge": ridge,
            "pca_dimension": dimension,
            "n_clusters": 2,
            "suspicious_component": suspicious,
            "component_sizes": [int(np.sum(cluster.labels_ == c)) for c in range(2)],
        }
    if method == "que_score":
        covariance = fitting_z.T @ fitting_z / max(len(fitting_z) - 1, 1)
        values, vectors = np.linalg.eigh(covariance)
        span = max(float(values[-1] - values[0]), 1e-8)
        weights = np.exp(4.0 * (values - values[-1]) / span)
        weights /= weights.sum()

        def que(values_in: np.ndarray) -> np.ndarray:
            z = (np.asarray(values_in, dtype=np.float64) - white_mean) @ transform
            return np.sum((z @ vectors) ** 2 * weights, axis=1)

        return que, {
            "whitening_ridge": ridge,
            "temperature": 4.0,
            "eigenvalue_min": float(values[0]),
            "eigenvalue_max": float(values[-1]),
            "weights_sha256": array_hash(weights),
        }
    raise ValueError(f"unknown detector: {method}")


def write_seed_scores(
    output: Path,
    seed: int,
    candidate_features: np.ndarray,
    assigned: np.ndarray,
    reference: np.ndarray,
    calibration: np.ndarray,
    fresh_features: np.ndarray,
    fresh_ids: np.ndarray,
    poison_indices: np.ndarray,
) -> tuple[Path, Path, dict[str, Any]]:
    folds = target_folds(assigned, seed)
    whitening = whitener(reference)
    truth = np.zeros(len(assigned), dtype=bool)
    truth[poison_indices] = True
    score_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    receipts: dict[str, Any] = {}
    victim_id = f"pathmnist_badnets_resnet18_seed{seed}"
    target_ids = np.concatenate(folds)
    for sample_id in target_ids:
        truth_rows.append({"victim_id": victim_id, "sample_id": int(sample_id), "role": "train", "is_poison": int(truth[sample_id])})
    for sample_id in fresh_ids:
        truth_rows.append({"victim_id": victim_id, "sample_id": 1_000_000 + int(sample_id), "role": "fresh", "is_poison": 0})

    for method_index, method in enumerate(METHODS):
        method_receipt: dict[str, Any] = {}
        all_flags = np.zeros(len(assigned), dtype=bool)
        for fold, (scoring, fitting) in enumerate(((folds[0], folds[1]), (folds[1], folds[0]))):
            scorer, receipt = fit_score_map(
                method,
                reference,
                candidate_features[fitting],
                seed=20304000 + 101 * seed + 7 * fold + method_index,
                whitening=whitening,
            )
            train_scores = np.asarray(scorer(candidate_features[scoring]), dtype=np.float64)
            fresh_scores = np.asarray(scorer(fresh_features), dtype=np.float64)
            calibration_scores = np.asarray(scorer(calibration), dtype=np.float64)
            if not (np.isfinite(train_scores).all() and np.isfinite(fresh_scores).all() and np.isfinite(calibration_scores).all()):
                raise RuntimeError(f"non-finite score from {method}, seed {seed}, fold {fold}")
            p_values = engine._rank_pvalues(calibration_scores, train_scores)
            flags = p_values <= 0.05
            all_flags[scoring] = flags
            score_rows.extend(
                {
                    "victim_id": victim_id,
                    "detector": method,
                    "fold": fold,
                    "sample_id": int(sample_id),
                    "score": repr(float(score)),
                    "role": "train",
                    "conventional_flag": int(flag),
                }
                for sample_id, score, flag in zip(scoring, train_scores, flags)
            )
            score_rows.extend(
                {
                    "victim_id": victim_id,
                    "detector": method,
                    "fold": fold,
                    "sample_id": 1_000_000 + int(sample_id),
                    "score": repr(float(score)),
                    "role": "fresh",
                    "conventional_flag": "",
                }
                for sample_id, score in zip(fresh_ids, fresh_scores)
            )
            method_receipt[str(fold)] = {
                **receipt,
                "scoring_indices_sha256": array_hash(scoring),
                "fitting_indices_sha256": array_hash(fitting),
                "train_scores_sha256": array_hash(train_scores),
                "fresh_scores_sha256": array_hash(fresh_scores),
                "calibration_scores_sha256": array_hash(calibration_scores),
                "scoring_count": int(len(scoring)),
                "fitting_count": int(len(fitting)),
                "fresh_count": int(len(fresh_scores)),
            }
        receipts[method] = method_receipt
    score_path = output / "blind_scores_parts" / f"seed_{seed}.csv"
    truth_path = output / "private_truth_parts" / f"seed_{seed}.csv"
    atomic_csv(score_path, score_rows)
    atomic_csv(truth_path, truth_rows)
    return score_path, truth_path, receipts


def run_formal(dataset: Path, output: Path, *, resume: bool, ack: str | None) -> dict[str, Any]:
    if ack != ACK:
        raise RuntimeError(f"formal execution requires --ack {ACK}")
    cfg = load_config()
    freeze = verify_freeze()
    if output.exists() and any(output.iterdir()) and not resume:
        raise RuntimeError("formal output is nonempty; exact resume requires --resume")
    if (output / "manifest.json").is_file():
        existing = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if existing.get("status") == "SCORE_HANDOFF_COMPLETE":
            raise RuntimeError("score handoff is already complete")
    device = validate_mps_gate(dataset)
    data = engine._load_train_validation(dataset)
    ref_idx, cal_idx = target_validation_indices(data["val"]["labels"])
    test = None
    acquisition = json.loads(engine.WEIGHT_RECEIPT.read_text(encoding="utf-8"))
    dataset_sha = sha256(dataset)
    output.mkdir(parents=True, exist_ok=True)
    frozen_hashes = {
        "dataset_sha256": dataset_sha,
        "config_sha256": sha256(CONFIG),
        "prereg_sha256": sha256(PREREG),
        "freeze_manifest_sha256": sha256(FREEZE),
        "orchestrator_sha256": sha256(Path(__file__)),
        "training_engine_sha256": sha256(Path(engine.__file__)),
        "architecture_sha256": sha256(Path(architecture.__file__)),
        "core_sha256": sha256(Path(core.__file__)),
        "smoke_receipt_sha256": sha256(SMOKE),
    }
    manifest: dict[str, Any] = {
        "experiment": cfg["protocol_id"],
        "status": "RUNNING_BLINDED",
        "formal": True,
        "freeze": freeze,
        "frozen_hashes": frozen_hashes,
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "device": str(device),
        "torch": torch.__version__,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "reference_indices_sha256": array_hash(ref_idx),
        "calibration_indices_sha256": array_hash(cal_idx),
        "created_unix": time.time(),
        "command_line": sys.argv,
        "completed_seeds": [],
    }
    atomic_json(output / "manifest.json", manifest)
    seed_receipts: list[dict[str, Any]] = []
    score_parts: list[Path] = []
    truth_parts: list[Path] = []
    for seed in SEEDS:
        start = time.time()
        print(f"blind formal seed {seed}: training suspect victim", flush=True)
        weight_path = engine.resolve_weight(acquisition, seed)
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
        initial_model = architecture.PathMNISTResNet18()
        initial_model.load_state_dict(checkpoint["net"], strict=True)
        initial = {k: v.detach().cpu().clone() for k, v in initial_model.state_dict().items()}
        initial_hash = core.state_fingerprint(initial)
        pool = engine.build_suspect_candidate_pool(data["train"]["images"], data["train"]["labels"], seed=seed)
        suspect_x = pool["images"]
        suspect_y = pool["assigned_labels"]
        poison_indices = pool["selected_indices"]
        training_sha = hashlib.sha256(suspect_x.tobytes() + suspect_y.tobytes()).hexdigest()
        weight_sha = sha256(weight_path)
        model, training = engine._train_member(
            suspect_x,
            suspect_y,
            initial,
            seed=seed,
            condition="suspect_blind_e2e",
            device=device,
            order_seed=20300000 + seed,
            output=output,
            checkpoint_path=output / "resume" / f"seed_{seed}_suspect_latest.pt",
            resume=resume,
            dataset_sha256=dataset_sha,
            weight_sha256=weight_sha,
            training_data_sha256=training_sha,
        )
        model_path = output / "models" / f"seed_{seed}_suspect_epoch5.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_model = model_path.with_name(model_path.name + ".tmp")
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, tmp_model)
        os.replace(tmp_model, model_path)

        if test is None:
            # The official test split is first opened only after the first fixed final
            # victim exists; it never affects training, detector fitting, or stopping.
            test = engine._load_test(dataset)
        victim_metrics = engine._test_metrics(model, test["images"], test["labels"], device)
        candidate_features_all = engine._hidden(model, suspect_x, device)
        target_ids = np.flatnonzero(suspect_y == TARGET).astype(np.int64)
        candidate_features = candidate_features_all
        reference = engine._hidden(model, data["val"]["images"][ref_idx], device)
        calibration = engine._hidden(model, data["val"]["images"][cal_idx], device)
        fresh_ids = np.flatnonzero(test["labels"] == TARGET).astype(np.int64)
        fresh_features = engine._hidden(model, test["images"][fresh_ids], device)
        feature_path = output / "features" / f"seed_{seed}_target_features.npz"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_feature = feature_path.with_name(feature_path.name + ".tmp.npz")
        np.savez_compressed(
            tmp_feature,
            candidate_ids=target_ids,
            candidate_features=candidate_features_all[target_ids].astype(np.float32),
            reference=reference.astype(np.float32),
            calibration=calibration.astype(np.float32),
            fresh_ids=fresh_ids,
            fresh_features=fresh_features.astype(np.float32),
            assigned_labels=suspect_y,
        )
        os.replace(tmp_feature, feature_path)
        score_part, truth_part, detector_receipts = write_seed_scores(
            output,
            seed,
            candidate_features,
            suspect_y,
            reference,
            calibration,
            fresh_features,
            fresh_ids,
            poison_indices,
        )
        score_parts.append(score_part)
        truth_parts.append(truth_part)
        receipt = {
            "seed": seed,
            "weight_file": str(weight_path),
            "weight_sha256": weight_sha,
            "initial_state_sha256": initial_hash,
            "training": training,
            "victim_metrics": victim_metrics,
            "eligible": bool(victim_metrics["balanced_accuracy"] >= 0.75 and victim_metrics["asr"] >= 0.80),
            "poison_count": int(len(poison_indices)),
            "poison_indices_sha256": array_hash(poison_indices),
            "target_candidate_count": int(len(target_ids)),
            "target_candidate_indices_sha256": array_hash(target_ids),
            "model_file": str(model_path),
            "model_sha256": sha256(model_path),
            "feature_file": str(feature_path),
            "feature_sha256": sha256(feature_path),
            "blind_score_part_sha256": sha256(score_part),
            "private_truth_part_sha256": sha256(truth_part),
            "detector_receipts": detector_receipts,
            "elapsed_seconds": time.time() - start,
        }
        seed_receipts.append(receipt)
        atomic_json(output / "seed_receipts.partial.json", seed_receipts)
        manifest["completed_seeds"] = [int(x["seed"]) for x in seed_receipts]
        atomic_json(output / "manifest.json", manifest)
        print(f"blind formal seed {seed}: sealed score handoff complete", flush=True)

    merge_csv(score_parts, output / "blind_scores.csv")
    merge_csv(truth_parts, output / "private_truth.csv")
    atomic_json(output / "seed_receipts.json", seed_receipts)
    manifest.update(
        {
            "status": "SCORE_HANDOFF_COMPLETE",
            "completed_seeds": list(SEEDS),
            "all_victims_eligible": all(row["eligible"] for row in seed_receipts),
            "blind_scores_sha256": sha256(output / "blind_scores.csv"),
            "private_truth_sha256": sha256(output / "private_truth.csv"),
            "seed_receipts_sha256": sha256(output / "seed_receipts.json"),
            "completed_unix": time.time(),
            "unblinded": False,
        }
    )
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "completed_seeds": manifest["completed_seeds"]}, sort_keys=True))
    return manifest


def selftest() -> None:
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(512, 32))
    fitting = np.r_[rng.normal(size=(600, 32)), rng.normal(1.2, 1, size=(120, 32))]
    scoring = rng.normal(size=(100, 32))
    white = whitener(reference)
    for index, method in enumerate(METHODS):
        score, _ = fit_score_map(method, reference, fitting, seed=100 + index, whitening=white)
        values = score(scoring)
        if values.shape != (100,) or not np.isfinite(values).all():
            raise RuntimeError(f"selftest failed for {method}")
    print("selftest PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ack")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.formal:
        run_formal(args.dataset, args.output, resume=args.resume, ack=args.ack)
    else:
        parser.error("choose --selftest or --formal")


if __name__ == "__main__":
    main()
