#!/usr/bin/env python3
"""Fresh CIFAR-10/VGG16-BN BadNets victims and truth-separated score handoff."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from blind_score_families import METHODS, array_hash, fit_score_map, rank_pvalues, whitener
from cifar10_vgg16_bn import CIFAR10VGG16BN, parameter_count
import run_cifar10_vgg16_bn_development as development


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cifar10_vgg16_bn_blind_certificate_v1.json"
PREREG = ROOT / "auditability_certificate_v1/PREREG_CIFAR10_VGG16_BLIND_V1.md"
FREEZE = ROOT / "auditability_certificate_v1/CIFAR10_VGG16_BLIND_FREEZE_V1.json"
DEVELOPMENT_REPORT = ROOT / "results/cifar10_vgg16_bn_development_v1/development_report.json"
PREFORMAL_AMENDMENT = ROOT / "auditability_certificate_v1/PREFORMAL_CIFAR_VGG_HORIZON_AMENDMENT_2026-10-08.md"
DEFAULT_OUTPUT = ROOT / "results/cifar10_vgg16_bn_blind_certificate_v1"
ACK = "OPEN_CIFAR10_VGG16_BLIND_SEEDS_4201_4205"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def merge_csv(parts: list[Path], output: Path) -> None:
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as target:
        for index, part in enumerate(parts):
            with part.open("rb") as source:
                if index:
                    source.readline()
                shutil.copyfileobj(source, target)
    os.replace(temporary, output)


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8")); digest.update(str(array.shape).encode("ascii")); digest.update(array.tobytes())
    return digest.hexdigest()


def load_and_verify() -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    dependencies = {
        "config_sha256": sha256(CONFIG),
        "prereg_sha256": sha256(PREREG),
        "runner_sha256": sha256(Path(__file__)),
        "architecture_sha256": sha256(ROOT / "src/cifar10_vgg16_bn.py"),
        "development_engine_sha256": sha256(ROOT / "src/run_cifar10_vgg16_bn_development.py"),
        "score_families_sha256": sha256(ROOT / "src/blind_score_families.py"),
        "certificate_runner_sha256": sha256(ROOT / "auditability_certificate_v1/analysis/run_cifar10_vgg16_blind_certificate.py"),
        "verifier_sha256": sha256(ROOT / "src/verify_cifar10_vgg16_blind.py"),
        "certificate_library_sha256": sha256(ROOT / "auditability_certificate_v1/cert/certificate.py"),
        "development_report_sha256": sha256(DEVELOPMENT_REPORT),
        "preformal_amendment_sha256": sha256(PREFORMAL_AMENDMENT),
        "orchestration_script_sha256": sha256(ROOT / "scripts/run_cifar10_vgg16_blind.zsh"),
    }
    if frozen.get("status") != "FROZEN_BEFORE_FORMAL_SEED_EXECUTION" or frozen.get("hashes") != dependencies:
        raise RuntimeError("formal freeze manifest mismatch")
    if tuple(config["formal_seeds"]) != (4201, 4202, 4203, 4204, 4205):
        raise RuntimeError("unexpected formal seeds")
    if tuple(config["detectors"]) != METHODS:
        raise RuntimeError("unexpected detector set")
    return config, frozen


def model_batch(model: CIFAR10VGG16BN, images: np.ndarray, indices: np.ndarray, device: torch.device, *, poisoned: set[int] | None = None, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    logits: list[np.ndarray] = []
    features: list[np.ndarray] = []
    poisoned = poisoned or set()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            local = indices[start:start + batch_size]
            values = development.uint8_batch(images, local, device)
            mask = np.asarray([int(index) in poisoned for index in local], dtype=bool)
            if mask.any():
                values[torch.from_numpy(mask).to(device)] = development.patch(values[torch.from_numpy(mask).to(device)])
            prediction, representation = model.forward_with_features(values)
            logits.append(prediction.cpu().numpy()); features.append(representation.cpu().numpy())
    return np.concatenate(logits), np.concatenate(features)


def train_seed(
    config: dict[str, Any], seed: int, device: torch.device,
    train_images: np.ndarray, train_labels: np.ndarray, test_images: np.ndarray, test_labels: np.ndarray,
    train_indices: np.ndarray, poison_indices: np.ndarray, output: Path, resume: bool,
) -> tuple[CIFAR10VGG16BN, dict[str, Any]]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    victim = config["victim"]
    target = int(config["attack"]["target_class"])
    poisoned = set(int(x) for x in poison_indices)
    is_poison = np.asarray([int(index) in poisoned for index in train_indices], dtype=bool)
    assigned = train_labels[train_indices].copy(); assigned[is_poison] = target
    model = CIFAR10VGG16BN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(victim["learning_rate"]), weight_decay=float(victim["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(victim["schedule_horizon_epochs"]), eta_min=0.0,
    )
    resume_path = output / "resume" / f"seed_{seed}_latest.pt"
    start_epoch = 1
    history: list[dict[str, Any]] = []
    if resume and resume_path.is_file():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (
            payload.get("seed") != seed
            or payload.get("config_sha256") != sha256(CONFIG)
            or payload.get("freeze_manifest_sha256") != sha256(FREEZE)
        ):
            raise RuntimeError(f"resume contract mismatch for seed {seed}")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        history = list(payload.get("history", []))
    batch_size = int(victim["batch_size"])
    started = time.time()
    for epoch in range(start_epoch, int(victim["training_epochs"]) + 1):
        model.train()
        order = np.random.default_rng(4_242_200 + 1009 * seed + epoch).permutation(len(train_indices))
        augmentation_rng = torch.Generator().manual_seed(4_242_300 + 1009 * seed + epoch)
        loss_sum = 0.0; correct = 0; seen = 0
        for begin in range(0, len(order), batch_size):
            positions = order[begin:begin + batch_size]
            base_indices = train_indices[positions]
            values = development.augment(development.uint8_batch(train_images, base_indices, torch.device("cpu")), augmentation_rng)
            poison_mask = torch.from_numpy(is_poison[positions])
            if bool(poison_mask.any()):
                values[poison_mask] = development.patch(values[poison_mask])
            values = values.to(device); labels = torch.from_numpy(assigned[positions]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values); loss = F.cross_entropy(logits, labels)
            loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * len(positions)
            correct += int((logits.argmax(1) == labels).sum().detach()); seen += len(positions)
        scheduler.step()
        row = {
            "epoch": epoch, "train_loss": loss_sum / seen, "train_accuracy": correct / seen,
            "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.time() - started,
        }
        history.append(row); print(f"cifar-vgg seed={seed} epoch={epoch} loss={row['train_loss']:.4f} acc={row['train_accuracy']:.4f}", flush=True)
        resume_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = resume_path.with_name(resume_path.name + ".tmp")
        torch.save({
            "seed": seed, "epoch": epoch, "config_sha256": sha256(CONFIG),
            "freeze_manifest_sha256": sha256(FREEZE), "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "history": history,
        }, temporary)
        os.replace(temporary, resume_path)
    metrics = development.evaluate(model, test_images, test_labels, device, target, batch_size)
    return model, {"history": history, "victim_metrics": metrics, "model_state_sha256": state_hash(model.state_dict())}


def target_folds(candidate_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    shuffled = np.random.default_rng(4_242_500 + seed).permutation(candidate_ids)
    first, second = np.array_split(shuffled, 2)
    return np.sort(first).astype(np.int64), np.sort(second).astype(np.int64)


def write_scores(
    output: Path, config: dict[str, Any], seed: int, candidate_ids: np.ndarray, candidate_features: np.ndarray,
    reference: np.ndarray, calibration: np.ndarray, fresh_ids: np.ndarray, fresh_features: np.ndarray,
    poison_indices: np.ndarray, original_labels: np.ndarray,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    victim_id = f"cifar10_badnets_vgg16bn_seed{seed}"
    target = int(config["attack"]["target_class"])
    folds = target_folds(candidate_ids, seed)
    index_to_position = {int(sample_id): position for position, sample_id in enumerate(candidate_ids)}
    poisoned = set(int(value) for value in poison_indices)
    whitening = whitener(reference)
    scores: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    trusted: list[dict[str, Any]] = []
    receipts: dict[str, Any] = {}
    for sample_id in candidate_ids:
        truth.append({"victim_id": victim_id, "sample_id": int(sample_id), "role": "train", "is_poison": int(int(sample_id) in poisoned)})
    for sample_id in fresh_ids:
        truth.append({"victim_id": victim_id, "sample_id": 1_000_000 + int(sample_id), "role": "fresh", "is_poison": 0})
    for fold, fold_ids in enumerate(folds):
        known_clean = fold_ids[(original_labels[fold_ids] == target) & ~np.asarray([int(value) in poisoned for value in fold_ids])]
        pool_size = int(config["certificate"]["trusted_pool_rows_per_fold"])
        if len(known_clean) < pool_size:
            raise RuntimeError(f"insufficient known-clean members in fold {fold}")
        pool = np.sort(np.random.default_rng(4_242_600 + 101 * seed + fold).choice(known_clean, size=pool_size, replace=False))
        trusted.extend({"victim_id": victim_id, "fold": fold, "sample_id": int(sample_id)} for sample_id in pool)
    for method_index, method in enumerate(METHODS):
        method_receipts: dict[str, Any] = {}
        for fold, (scoring_ids, fitting_ids) in enumerate(((folds[0], folds[1]), (folds[1], folds[0]))):
            scoring_positions = np.asarray([index_to_position[int(value)] for value in scoring_ids], dtype=np.int64)
            fitting_positions = np.asarray([index_to_position[int(value)] for value in fitting_ids], dtype=np.int64)
            scorer, receipt = fit_score_map(
                method, reference, candidate_features[fitting_positions],
                seed=4_242_700 + 101 * seed + 7 * fold + method_index, whitening=whitening,
            )
            train_scores = np.asarray(scorer(candidate_features[scoring_positions]), dtype=np.float64)
            fresh_scores = np.asarray(scorer(fresh_features), dtype=np.float64)
            calibration_scores = np.asarray(scorer(calibration), dtype=np.float64)
            flags = rank_pvalues(calibration_scores, train_scores) <= float(config["certificate"]["alpha"])
            if not (np.isfinite(train_scores).all() and np.isfinite(fresh_scores).all() and np.isfinite(calibration_scores).all()):
                raise RuntimeError(f"non-finite scores for {method}, fold {fold}")
            scores.extend({
                "victim_id": victim_id, "detector": method, "fold": fold, "sample_id": int(sample_id),
                "score": repr(float(score)), "role": "train", "conventional_flag": int(flag),
            } for sample_id, score, flag in zip(scoring_ids, train_scores, flags))
            scores.extend({
                "victim_id": victim_id, "detector": method, "fold": fold, "sample_id": 1_000_000 + int(sample_id),
                "score": repr(float(score)), "role": "fresh", "conventional_flag": "",
            } for sample_id, score in zip(fresh_ids, fresh_scores))
            method_receipts[str(fold)] = {
                **receipt, "scoring_ids_sha256": array_hash(scoring_ids), "fitting_ids_sha256": array_hash(fitting_ids),
                "train_scores_sha256": array_hash(train_scores), "fresh_scores_sha256": array_hash(fresh_scores),
                "calibration_scores_sha256": array_hash(calibration_scores),
            }
        receipts[method] = method_receipts
    score_path = output / "blind_scores_parts" / f"seed_{seed}.csv"
    truth_path = output / "private_truth_parts" / f"seed_{seed}.csv"
    trusted_path = output / "trusted_member_pool_parts" / f"seed_{seed}.csv"
    atomic_csv(score_path, scores); atomic_csv(truth_path, truth); atomic_csv(trusted_path, trusted)
    return score_path, truth_path, trusted_path, receipts


def run_formal(output: Path, device: torch.device, *, resume: bool, ack: str | None) -> dict[str, Any]:
    if ack != ACK:
        raise RuntimeError(f"formal execution requires --ack {ACK}")
    config, freeze = load_and_verify()
    if output.exists() and any(output.iterdir()) and not resume:
        raise RuntimeError("formal output is nonempty; use --resume for exact continuation")
    if (output / "manifest.json").is_file():
        existing = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if existing.get("status") == "SCORE_HANDOFF_COMPLETE":
            raise RuntimeError("score handoff is already complete")
        if existing.get("freeze_manifest_sha256") != sha256(FREEZE):
            raise RuntimeError("resume output was created under a different freeze manifest")
    if device.type != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("the frozen protocol requires MPS")
    train_images, train_labels, test_images, test_labels, dataset_receipt = development.resolve_dataset(config)
    train_indices, validation_indices = development.fixed_split(
        train_labels, rows_per_class=int(config["dataset"]["validation_rows_per_class"]),
        seed=int(config["dataset"]["validation_split_seed"]),
    )
    target = int(config["attack"]["target_class"])
    target_validation = validation_indices[train_labels[validation_indices] == target]
    order = np.random.default_rng(int(config["reference"]["split_seed"])).permutation(target_validation)
    reference_ids = np.sort(order[:250]); calibration_ids = np.sort(order[250:500])
    fresh_ids = np.flatnonzero(test_labels == target).astype(np.int64)
    output.mkdir(parents=True, exist_ok=True)
    dependency_hashes = freeze["hashes"]
    manifest: dict[str, Any] = {
        "experiment": config["protocol_id"], "status": "RUNNING_BLINDED", "formal": True,
        "freeze": freeze, "freeze_manifest_sha256": sha256(FREEZE), "frozen_hashes": dependency_hashes, "dataset": dataset_receipt,
        "seeds": config["formal_seeds"], "methods": list(METHODS), "device": str(device),
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "train_indices_sha256": array_hash(train_indices), "validation_indices_sha256": array_hash(validation_indices),
        "reference_ids_sha256": array_hash(reference_ids), "calibration_ids_sha256": array_hash(calibration_ids),
        "created_unix": time.time(), "command_line": sys.argv, "completed_seeds": [], "unblinded": False,
    }
    atomic_json(output / "manifest.json", manifest)
    score_parts: list[Path] = []; truth_parts: list[Path] = []; trusted_parts: list[Path] = []; seed_receipts: list[dict[str, Any]] = []
    for seed in (int(value) for value in config["formal_seeds"]):
        start = time.time(); print(f"cifar-vgg blind seed {seed}: training fresh victim", flush=True)
        poison_indices = development.poison_plan(
            train_labels, train_indices, target=target, count=int(config["attack"]["poison_count"]), seed=seed,
        )
        model, training = train_seed(
            config, seed, device, train_images, train_labels, test_images, test_labels,
            train_indices, poison_indices, output, resume,
        )
        model_path = output / "models" / f"seed_{seed}_suspect.pt"; model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = model_path.with_name(model_path.name + ".tmp")
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, temporary); os.replace(temporary, model_path)
        poisoned = set(int(x) for x in poison_indices)
        assigned = train_labels.copy(); assigned[poison_indices] = target
        candidate_ids = train_indices[assigned[train_indices] == target]
        _, candidate_features = model_batch(model, train_images, candidate_ids, device, poisoned=poisoned)
        _, reference = model_batch(model, train_images, reference_ids, device)
        _, calibration = model_batch(model, train_images, calibration_ids, device)
        _, fresh_features = model_batch(model, test_images, fresh_ids, device)
        feature_path = output / "features" / f"seed_{seed}_target_features.npz"; feature_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_feature = feature_path.with_name(feature_path.name + ".tmp.npz")
        np.savez_compressed(
            temporary_feature, candidate_ids=candidate_ids, candidate_features=candidate_features.astype(np.float32),
            reference=reference.astype(np.float32), calibration=calibration.astype(np.float32),
            fresh_ids=fresh_ids, fresh_features=fresh_features.astype(np.float32),
        ); os.replace(temporary_feature, feature_path)
        score_path, truth_path, trusted_path, detector_receipts = write_scores(
            output, config, seed, candidate_ids, candidate_features, reference, calibration,
            fresh_ids, fresh_features, poison_indices, train_labels,
        )
        score_parts.append(score_path); truth_parts.append(truth_path); trusted_parts.append(trusted_path)
        victim_metrics = training["victim_metrics"]
        eligible = victim_metrics["balanced_accuracy"] >= float(config["eligibility"]["balanced_accuracy_min"]) and victim_metrics["attack_success_rate"] >= float(config["eligibility"]["attack_success_rate_min"])
        receipt = {
            "seed": seed, "eligible": bool(eligible), "victim_metrics": victim_metrics,
            "poison_count": int(len(poison_indices)), "poison_indices_sha256": array_hash(poison_indices),
            "candidate_count": int(len(candidate_ids)), "candidate_ids_sha256": array_hash(candidate_ids),
            "model_file": str(model_path), "model_sha256": sha256(model_path), "model_state_sha256": training["model_state_sha256"],
            "feature_file": str(feature_path), "feature_sha256": sha256(feature_path),
            "blind_score_part_sha256": sha256(score_path), "private_truth_part_sha256": sha256(truth_path),
            "trusted_member_pool_part_sha256": sha256(trusted_path), "detector_receipts": detector_receipts,
            "training_history": training["history"], "elapsed_seconds": time.time() - start,
        }
        seed_receipts.append(receipt); atomic_json(output / "seed_receipts.partial.json", seed_receipts)
        manifest["completed_seeds"] = [int(row["seed"]) for row in seed_receipts]; atomic_json(output / "manifest.json", manifest)
        print(f"cifar-vgg blind seed {seed}: sealed score handoff", flush=True)
    merge_csv(score_parts, output / "blind_scores.csv"); merge_csv(truth_parts, output / "private_truth.csv"); merge_csv(trusted_parts, output / "trusted_member_pool.csv")
    atomic_json(output / "seed_receipts.json", seed_receipts)
    manifest.update({
        "status": "SCORE_HANDOFF_COMPLETE", "completed_seeds": config["formal_seeds"],
        "all_victims_eligible": all(row["eligible"] for row in seed_receipts),
        "blind_scores_sha256": sha256(output / "blind_scores.csv"),
        "private_truth_sha256": sha256(output / "private_truth.csv"),
        "trusted_member_pool_sha256": sha256(output / "trusted_member_pool.csv"),
        "seed_receipts_sha256": sha256(output / "seed_receipts.json"), "completed_unix": time.time(),
    })
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "completed_seeds": manifest["completed_seeds"], "all_victims_eligible": manifest["all_victims_eligible"]}, sort_keys=True))
    return manifest


def selftest() -> None:
    rng = np.random.default_rng(4200)
    candidate_ids = np.arange(6750, dtype=np.int64)
    first, second = target_folds(candidate_ids, 4201)
    if len(first) != 3375 or len(second) != 3375 or len(np.intersect1d(first, second)):
        raise RuntimeError("fold selftest failed")
    model = CIFAR10VGG16BN(); logits, features = model.forward_with_features(torch.rand(2, 3, 32, 32))
    if logits.shape != (2, 10) or features.shape != (2, 512) or parameter_count() < 14_000_000:
        raise RuntimeError("model selftest failed")
    if not np.isfinite(rng.normal(size=10)).all():
        raise RuntimeError("numeric selftest failed")
    print("selftest PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ack")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.formal:
        run_formal(args.output, torch.device("mps"), resume=args.resume, ack=args.ack)
    else:
        parser.error("choose --selftest or --formal")


if __name__ == "__main__":
    main()
