#!/usr/bin/env python3
"""Three-seed BloodMNIST cross-attack pilot for the frozen CF-CMD detector.

This is a resource gate, not confirmatory evidence.  It screens four attacks
whose defining mechanisms are implemented faithfully enough for a pilot:
BadNets, Blended, SIG, and WaNet (including correct-label cross samples).
Input-Aware and Label-Consistent remain blocked until their joint/adversarial
training mechanisms are implemented; they are never replaced by static noise.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import bloodmnist_attack_families as attack_ops
import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_group_coherent_10seeds as baseline
import run_bloodmnist_new_encoder_adaptive as adaptive
import run_bloodmnist_unified_oneclass as blood
import run_mnist_group_coherent_drift as coherent
import run_unified_oneclass_poison_screening as unified
import train_bloodmnist_clean_encoder as clean_encoder


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PAPER14_ROOT / "results/bloodmnist_cross_attack_pilot_v1"
REFERENCE_CLEAN_ROOT = PAPER14_ROOT / "results/bloodmnist_group_coherent_10seeds_v1"
FRESH_CLEAN_ROOT = PAPER14_ROOT / "results/bloodmnist_xz16_fresh_confirmation_v1"
ATTACKS = ("badnets", "blended", "sig", "wanet")
PILOT_SEEDS = (42, 46, 51)
CONFIRMATION_SEEDS = tuple(range(52, 62))


def array_sha256(values: np.ndarray | torch.Tensor) -> str:
    array = (
        values.detach().cpu().numpy()
        if isinstance(values, torch.Tensor)
        else np.asarray(values)
    )
    canonical = np.ascontiguousarray(array.astype("<f4", copy=False))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_raw(prepared: dict, raw_images: np.ndarray) -> np.ndarray:
    """Map [0,1] flattened RGB images through the frozen seed preprocessing."""

    standardized = prepared["raw_scaler"].transform(raw_images)
    pca_values = prepared["pca"].transform(standardized)
    return prepared["post_scaler"].transform(pca_values).astype(np.float32)


def base_processed(prepared: dict, split: str, class_id: int) -> np.ndarray:
    return prepared["post_scaler"].transform(
        prepared["pca_splits"][split][class_id]
    ).astype(np.float32)


def trigger_raw(
    attack: str,
    raw_images: np.ndarray,
    *,
    seed: int,
    wanet_grid: torch.Tensor | None,
) -> tuple[np.ndarray, dict]:
    if attack == "badnets":
        values, receipt = attack_ops.apply_badnets_patch(raw_images)
    elif attack == "blended":
        values, receipt = attack_ops.apply_blended(raw_images)
    elif attack == "sig":
        values, receipt = attack_ops.apply_sig(raw_images)
    elif attack == "wanet":
        if wanet_grid is None:
            raise ValueError("WaNet requires a frozen grid")
        values = attack_ops.apply_wanet(raw_images, wanet_grid)
        _, receipt = attack_ops.make_wanet_grid(seed=seed + 91001)
    else:
        raise ValueError(f"Unsupported pilot attack: {attack}")
    return values.astype(np.float32, copy=False), receipt.to_dict()


def build_attack_splits(prepared: dict, attack: str, seed: int) -> tuple[dict, dict]:
    """Construct replacement-style poison training data and held-out attack views."""

    raw = prepared["raw"]
    rng = np.random.default_rng(20260928 + 101 * seed + ATTACKS.index(attack))
    poison_count = 60
    wanet_grid = None
    wanet_receipt = None
    if attack == "wanet":
        wanet_grid, receipt = attack_ops.make_wanet_grid(seed=seed + 91001)
        wanet_receipt = receipt.to_dict()

    triggered_raw: dict[str, np.ndarray] = {}
    trigger_receipt = None
    for split in ("train", "val", "test"):
        values, receipt = trigger_raw(
            attack,
            raw[split][0],
            seed=seed,
            wanet_grid=wanet_grid,
        )
        triggered_raw[split] = values
        trigger_receipt = receipt

    target_train_raw = raw["train"][6]
    source_train_raw = raw["train"][0]
    target_train = base_processed(prepared, "train", 6)
    source_train = base_processed(prepared, "train", 0)
    target_order = rng.permutation(len(target_train))
    source_order = rng.permutation(len(source_train))
    cross_receipt: dict | None = None

    if attack == "sig":
        # SIG is a clean-label attack: poison target-class images and preserve labels.
        poison_indices = target_order[:poison_count]
        poison_raw, trigger_receipt = trigger_raw(
            attack,
            target_train_raw[poison_indices],
            seed=seed,
            wanet_grid=None,
        )
        poison = process_raw(prepared, poison_raw)
        train_target = target_train[np.setdiff1d(np.arange(len(target_train)), poison_indices)]
        train_source = source_train
        candidate_clean = train_target
        poison_origin = "target_clean_label"
    else:
        poison_indices = source_order[:poison_count]
        poison = process_raw(prepared, triggered_raw["train"][poison_indices])
        remaining_source = np.setdiff1d(np.arange(len(source_train)), poison_indices)
        train_target = target_train
        train_source = source_train[remaining_source]
        candidate_clean = train_target
        poison_origin = "source_all_to_one"

        if attack == "wanet":
            # WaNet uses two correct-label cross/noise samples per backdoor sample.
            combined_raw = np.concatenate(
                [target_train_raw, source_train_raw[remaining_source]], axis=0
            )
            combined_y = np.concatenate(
                [
                    np.ones(len(target_train_raw), dtype=np.int64),
                    np.zeros(len(remaining_source), dtype=np.int64),
                ]
            )
            cross_count = 2 * poison_count
            cross_indices = rng.permutation(len(combined_raw))[:cross_count]
            cross_grids = attack_ops.make_wanet_cross_grids(
                wanet_grid, count=cross_count, seed=seed + 92003
            )
            cross_raw = attack_ops.apply_wanet(combined_raw[cross_indices], cross_grids)
            combined_processed = process_raw(prepared, combined_raw)
            cross_processed = process_raw(prepared, cross_raw)
            keep = np.setdiff1d(np.arange(len(combined_raw)), cross_indices)
            victim_clean_x = np.concatenate(
                [combined_processed[keep], cross_processed], axis=0
            )
            victim_clean_y = np.concatenate(
                [combined_y[keep], combined_y[cross_indices]], axis=0
            )
            target_cross_mask = combined_y[cross_indices] == 1
            target_kept = combined_processed[keep][combined_y[keep] == 1]
            target_cross = cross_processed[target_cross_mask]
            candidate_clean = np.concatenate([target_kept, target_cross], axis=0)
            train_target = candidate_clean
            train_source = np.concatenate(
                [
                    combined_processed[keep][combined_y[keep] == 0],
                    cross_processed[~target_cross_mask],
                ],
                axis=0,
            )
            cross_receipt = {
                "cross_ratio": 2.0,
                "cross_count": cross_count,
                "target_cross_count": int(target_cross_mask.sum()),
                "source_cross_count": int((~target_cross_mask).sum()),
                "correct_labels_preserved": True,
                "cross_processed_sha256_float32": array_sha256(cross_processed),
            }

    if attack != "wanet":
        victim_clean_x = np.concatenate([train_target, train_source], axis=0)
        victim_clean_y = np.concatenate(
            [
                np.ones(len(train_target), dtype=np.int64),
                np.zeros(len(train_source), dtype=np.int64),
            ]
        )

    splits: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        target = base_processed(prepared, split, 6)
        source = base_processed(prepared, split, 0)
        triggered = process_raw(prepared, triggered_raw[split])
        splits[split] = {
            "target": target,
            "non_target": source,
            "triggered_non_target": triggered,
            "clean_x": np.concatenate([target, source], axis=0),
            "clean_y": np.concatenate(
                [
                    np.ones(len(target), dtype=np.int64),
                    np.zeros(len(source), dtype=np.int64),
                ]
            ),
        }
    splits["train"]["target"] = train_target
    splits["train"]["non_target"] = train_source
    splits["train"]["clean_x"] = victim_clean_x
    splits["train"]["clean_y"] = victim_clean_y
    splits["archived_poison"] = poison
    splits["candidate_clean"] = candidate_clean

    receipt = {
        "attack": attack,
        "seed": seed,
        "poison_count": poison_count,
        "poison_origin": poison_origin,
        "poison_indices": poison_indices.tolist(),
        "candidate_clean_count": len(candidate_clean),
        "trigger": wanet_receipt if attack == "wanet" else trigger_receipt,
        "wanet_cross": cross_receipt,
        "replacement_style_training": True,
        "triggered_source_train_sha256_float32": array_sha256(
            triggered_raw["train"]
        ),
        "poison_processed_sha256_float32": array_sha256(poison),
    }
    if wanet_grid is not None:
        receipt["wanet_grid_sha256_float32"] = array_sha256(wanet_grid)
    return splits, receipt


def load_reference_clean_model(
    seed: int, root: Path
) -> tuple[clean_encoder.CleanQNN, dict]:
    checkpoint = root / "models" / f"seed_{seed}_clean.pt"
    record = root / "models" / f"seed_{seed}_clean.json"
    if not checkpoint.exists() or not record.exists():
        raise FileNotFoundError(
            f"Missing reusable clean seed-{seed} checkpoint under {root}"
        )
    model = clean_encoder.CleanQNN("local", 0)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    )
    return model, json.loads(record.read_text(encoding="utf-8"))


def train_classical_arrays(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    epochs: int,
    mean: np.ndarray,
    scale: np.ndarray,
) -> unified.ClassicalVictim:
    generator = torch.Generator().manual_seed(unified.RANDOM_SEED + seed)
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor((x - mean) / scale, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, generator=generator
    )
    torch.manual_seed(seed)
    model = unified.BackdoorMLP(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
    return unified.ClassicalVictim(model=model, mean=mean, scale=scale, metrics={})


def load_or_train_poisoned_classical(
    checkpoint: Path,
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    epochs: int,
    mean: np.ndarray,
    scale: np.ndarray,
) -> unified.ClassicalVictim:
    fingerprint = hashlib.sha256(
        (
            array_sha256(x)
            + array_sha256(y.astype(np.float32))
            + str(seed)
            + str(epochs)
        ).encode("ascii")
    ).hexdigest()
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("training_fingerprint") == fingerprint:
            model = unified.BackdoorMLP(x.shape[1])
            model.load_state_dict(payload["state_dict"])
            return unified.ClassicalVictim(
                model=model,
                mean=np.asarray(payload["mean"], dtype=np.float32),
                scale=np.asarray(payload["scale"], dtype=np.float32),
                metrics={},
            )
    victim = train_classical_arrays(
        x,
        y,
        seed=seed,
        epochs=epochs,
        mean=mean,
        scale=scale,
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": victim.model.state_dict(),
            "mean": mean,
            "scale": scale,
            "seed": seed,
            "epochs": epochs,
            "training_fingerprint": fingerprint,
        },
        checkpoint,
    )
    return victim


def load_or_train_clean_classical(
    output: Path,
    target: np.ndarray,
    source: np.ndarray,
    *,
    seed: int,
    epochs: int,
) -> unified.ClassicalVictim:
    cache = output / "clean_classical"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"seed_{seed}.pt"
    x = np.concatenate([target, source], axis=0)
    y = np.concatenate(
        [np.ones(len(target), dtype=np.int64), np.zeros(len(source), dtype=np.int64)]
    )
    mean = x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = unified.BackdoorMLP(x.shape[1])
        model.load_state_dict(payload["state_dict"])
        return unified.ClassicalVictim(
            model=model,
            mean=np.asarray(payload["mean"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
            metrics={},
        )
    victim = train_classical_arrays(
        x, y, seed=seed, epochs=epochs, mean=mean, scale=scale
    )
    torch.save(
        {"state_dict": victim.model.state_dict(), "mean": mean, "scale": scale}, path
    )
    return victim


def predict_classical(victim: unified.ClassicalVictim, x: np.ndarray) -> np.ndarray:
    return blood.predictions(victim, x)


def evaluate_external_reference(
    config: dict,
    name: str,
    trusted_clean: np.ndarray,
    candidate_clean: np.ndarray,
    poison: np.ndarray,
    alpha: float,
) -> tuple[dict, list[dict]]:
    """CF-CMD with a disjoint validation-class trusted reference/calibration pool."""

    trusted = np.asarray(trusted_clean, dtype=np.float64)
    features = np.concatenate([candidate_clean, poison]).astype(np.float64)
    labels = np.concatenate(
        [np.zeros(len(candidate_clean), dtype=int), np.ones(len(poison), dtype=int)]
    )
    rng = np.random.default_rng(20260928 + 1009 * int(config["seed"]))
    trusted_order = rng.permutation(len(trusted))
    n_reference = int(round(0.6 * len(trusted)))
    reference = trusted[trusted_order[:n_reference]]
    calibration = trusted[trusted_order[n_reference:]]
    if len(reference) < 100 or len(calibration) < 60:
        raise RuntimeError("External trusted pool is too small")
    mean, transform = coherent.whitening(reference)

    def whiten(values: np.ndarray) -> np.ndarray:
        return (values - mean) @ transform

    reference_white = whiten(reference)
    calibration_white = whiten(calibration)
    folds = np.array_split(rng.permutation(len(features)), 4)
    scores = np.empty(len(features), dtype=np.float64)
    p_values = np.empty(len(features), dtype=np.float64)
    flags = np.zeros(len(features), dtype=bool)
    fold_receipts = []
    for fold_index, query_indices in enumerate(folds):
        halves = np.array_split(rng.permutation(query_indices), 2)
        fold_receipts.append(
            {
                "fold": fold_index,
                "candidate_count": len(query_indices),
                "candidate_poison_fraction": float(labels[query_indices].mean()),
                "external_reference_count": len(reference),
                "external_calibration_count": len(calibration),
            }
        )
        for score_half, direction_half in (
            (halves[0], halves[1]),
            (halves[1], halves[0]),
        ):
            direction = (
                whiten(features[direction_half]).mean(axis=0)
                - reference_white.mean(axis=0)
            )
            norm = float(np.linalg.norm(direction))
            if norm < 1e-10:
                direction = np.ones(features.shape[1], dtype=np.float64)
                norm = float(np.linalg.norm(direction))
            direction /= norm
            calibration_scores = calibration_white @ direction
            query_scores = whiten(features[score_half]) @ direction
            query_p = coherent.one_sided_p_values(calibration_scores, query_scores)
            scores[score_half] = query_scores
            p_values[score_half] = query_p
            flags[score_half] = query_p <= alpha

    sample_rows = []
    for index, (label, score, p_value, flag) in enumerate(
        zip(labels, scores, p_values, flags)
    ):
        sample_rows.append(
            {
                "seed": config["seed"],
                "trigger": config["trigger"],
                "representation": name,
                "sample_index": index,
                "label": int(label),
                "score": float(score),
                "p_value": float(p_value),
                "flagged": int(flag),
                "alpha": alpha,
            }
        )
    clean_flags, poison_flags = flags[labels == 0], flags[labels == 1]
    fp, tn = int(clean_flags.sum()), int((~clean_flags).sum())
    tp, fn = int(poison_flags.sum()), int((~poison_flags).sum())
    row = {
        "seed": config["seed"],
        "trigger": config["trigger"],
        "representation": name,
        "archive_ca": config["archive_ca"],
        "archive_asr": config["archive_asr"],
        "dimension": features.shape[1],
        "n_clean": len(candidate_clean),
        "n_poison": len(poison),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tpr": tp / (tp + fn),
        "fpr": fp / (fp + tn),
        "auc_crossfit_score": float(roc_auc_score(labels, scores)),
        "auc_conformal_rank": float(roc_auc_score(labels, -p_values)),
        "config_pass": int(tp / (tp + fn) >= 0.80 and fp / (fp + tn) <= 0.10),
        "fold_receipts_json": json.dumps(fold_receipts, separators=(",", ":")),
        "trusted_reference_split": "validation_target",
    }
    return row, sample_rows


def paired_representations(
    clean_classical: unified.ClassicalVictim,
    poisoned_classical: unified.ClassicalVictim,
    clean_qml: clean_encoder.CleanQNN,
    poisoned_qml: clean_encoder.CleanQNN,
    groups: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    clean_margin = {
        key: clean_classical.representations(value)[1] for key, value in groups.items()
    }
    poisoned_margin = {
        key: poisoned_classical.representations(value)[1]
        for key, value in groups.items()
    }
    clean_z, clean_xz = blood.qml_views(clean_qml, groups)
    poisoned_z, poisoned_xz = blood.qml_views(poisoned_qml, groups)
    return {
        "coherent_input256": {key: value.astype(float) for key, value in groups.items()},
        "coherent_classical_margin_drift1": {
            key: poisoned_margin[key] - clean_margin[key] for key in groups
        },
        "coherent_qml_z_drift8": {
            key: poisoned_z[key] - clean_z[key] for key in groups
        },
        "coherent_qml_xz_drift16": {
            key: poisoned_xz[key] - clean_xz[key] for key in groups
        },
    }


def eligible_summaries(
    rows: list[dict], victims: list[dict], attacks: list[str], role: str
) -> list[dict]:
    victim_map = {(row["attack"], int(row["seed"])): row for row in victims}
    endpoints = {
        "coherent_bonferroni_input_classical_margin": "classical",
        "coherent_qml_xz_drift16": "qml",
        "coherent_bonferroni_input_qml_xz": "qml",
    }
    output = []
    for attack in attacks:
        for representation, victim_type in endpoints.items():
            selected = []
            seeds = []
            for row in rows:
                if row["trigger"] != attack or row["representation"] != representation:
                    continue
                victim = victim_map[(attack, int(row["seed"]))]
                if (
                    victim[f"{victim_type}_balanced_ca"] >= 0.90
                    and victim[f"{victim_type}_asr"] >= 0.80
                ):
                    selected.append(row)
                    seeds.append(int(row["seed"]))
            if not selected:
                output.append(
                    {
                        "trigger": attack,
                        "representation": representation,
                        "victim_type": victim_type,
                        "eligible_seed_count": 0,
                        "eligible_seeds": "",
                        "status": "NO_ELIGIBLE_VICTIM",
                    }
                )
                continue
            summary = coherent.aggregate(selected, "representation")[0]
            output.append(
                {
                    "trigger": attack,
                    "representation": representation,
                    "victim_type": victim_type,
                    "eligible_seed_count": len(seeds),
                    "eligible_seeds": ",".join(map(str, sorted(seeds))),
                    "status": (
                        "CONFIRMATORY"
                        if role == "selective_confirmation"
                        else "PILOT_ONLY"
                    ),
                    **{key: value for key, value in summary.items() if key != "representation"},
                }
            )
    return output


def write_report(
    output: Path,
    victims: list[dict],
    eligible: list[dict],
    elapsed: float,
    role: str,
) -> None:
    if role == "selective_confirmation":
        title = "# BloodMNIST cross-attack fresh-seed confirmation"
        scope = (
            "This confirmation uses the contiguous untouched attack block seeds 52--61 "
            "with attack and detector parameters frozen after the 42/46/51 resource gate."
        )
    else:
        title = "# BloodMNIST cross-attack resource-gate pilot"
        scope = "This three-seed run is a development resource gate, not confirmatory evidence."
    lines = [
        title,
        "",
        scope + " Victim eligibility is balanced CA >= 0.90 and ASR >= 0.80. Detector diagnostics use TPR >= 0.80 and FPR <= 0.10; the confirmatory gate is pooled TPR >= 0.90 and FPR <= 0.05.",
        "",
        "## Victim feasibility",
        "",
        "| attack | seed | classical CA | classical ASR | QML CA | QML ASR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in victims:
        lines.append(
            f"| {row['attack']} | {row['seed']} | {row['classical_balanced_ca']:.3f} | "
            f"{row['classical_asr']:.3f} | {row['qml_balanced_ca']:.3f} | "
            f"{row['qml_asr']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Attack-eligible detector results",
            "",
            "| attack | endpoint | victim | eligible seeds | pooled TPR | pooled FPR | diagnostic pass rate |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in eligible:
        if row["eligible_seed_count"] == 0:
            lines.append(
                f"| {row['trigger']} | {row['representation']} | {row['victim_type']} | 0 | — | — | — |"
            )
        else:
            lines.append(
                f"| {row['trigger']} | {row['representation']} | {row['victim_type']} | "
                f"{row['eligible_seed_count']} | {row['pooled_tpr']:.3f} | "
                f"{row['pooled_fpr']:.3f} | {row['config_pass_rate']:.3f} |"
            )
    lines.extend(
        [
            "",
            "Input-Aware and full Label-Consistent are deliberately absent from this experiment until their joint generator/adversarial-base mechanisms are implemented. No static proxy is counted as an attack-family result.",
            "",
            f"Runtime: {elapsed:.1f} seconds.",
            "",
        ]
    )
    (output / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    if any(attack not in ATTACKS for attack in args.attacks):
        raise ValueError(f"Pilot attacks must be selected from {ATTACKS}")
    if args.role == "selective_confirmation":
        if tuple(args.seeds) != CONFIRMATION_SEEDS:
            raise ValueError("selective_confirmation is frozen to seeds 52--61")
        if args.attacks != ["blended", "sig"]:
            raise ValueError(
                "selective_confirmation is frozen to attacks: blended sig"
            )
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    class_data = heldout.collect_classes(args.dataset_root)
    rows, sample_rows, victims, receipts = [], [], [], []

    for seed in args.seeds:
        print(f"seed={seed}: preprocessing and reusable clean victims", flush=True)
        prepared = baseline.preprocess_seed(class_data, seed)
        clean_qml, _ = load_reference_clean_model(seed, args.clean_qml_root)
        original_target = base_processed(prepared, "train", 6)
        original_source = base_processed(prepared, "train", 0)
        clean_classical = load_or_train_clean_classical(
            args.output,
            original_target,
            original_source,
            seed=seed,
            epochs=args.classical_epochs,
        )

        for attack_index, attack in enumerate(args.attacks):
            print(f"seed={seed} attack={attack}: constructing victim", flush=True)
            splits, receipt = build_attack_splits(prepared, attack, seed)
            receipts.append(receipt)
            attack_model_root = args.output / "models" / attack
            (attack_model_root / "models").mkdir(parents=True, exist_ok=True)
            poisoned_qml, adaptive_record = baseline.load_or_adapt(
                attack_model_root,
                clean_qml,
                splits,
                seed,
                args.adaptive_epochs,
            )

            train_x = np.concatenate(
                [splits["train"]["clean_x"], splits["archived_poison"]], axis=0
            )
            train_y = np.concatenate(
                [
                    splits["train"]["clean_y"],
                    np.ones(len(splits["archived_poison"]), dtype=np.int64),
                ]
            )
            classical_checkpoint = (
                args.output / "models" / attack / f"classical_seed_{seed}.pt"
            )
            poisoned_classical = load_or_train_poisoned_classical(
                classical_checkpoint,
                train_x,
                train_y,
                seed=seed,
                epochs=args.classical_epochs,
                mean=clean_classical.mean,
                scale=clean_classical.scale,
            )
            qml_checkpoint = (
                attack_model_root / "models" / f"seed_{seed}_adaptive.pt"
            )
            receipt["victim_checkpoints"] = {
                "classical": str(classical_checkpoint.relative_to(args.output)),
                "classical_sha256": file_sha256(classical_checkpoint),
                "qml": str(qml_checkpoint.relative_to(args.output)),
                "qml_sha256": file_sha256(qml_checkpoint),
            }

            target_test = splits["test"]["target"]
            source_test = splits["test"]["non_target"]
            trigger_test = splits["test"]["triggered_non_target"]
            classical_ca = 0.5 * (
                np.mean(predict_classical(poisoned_classical, target_test) == 1)
                + np.mean(predict_classical(poisoned_classical, source_test) == 0)
            )
            classical_asr = np.mean(
                predict_classical(poisoned_classical, trigger_test) == 1
            )
            qml_test = adaptive_record["test"]
            victim_row = {
                "attack": attack,
                "seed": seed,
                "classical_balanced_ca": float(classical_ca),
                "classical_asr": float(classical_asr),
                "qml_balanced_ca": float(qml_test["clean"]["balanced_accuracy"]),
                "qml_asr": float(qml_test["attack"]["asr"]),
                "qml_selected_epoch": adaptive_record["selected_epoch"],
                "candidate_clean_count": len(splits["candidate_clean"]),
                "poison_count": len(splits["archived_poison"]),
            }
            victims.append(victim_row)

            groups = {
                "trusted": base_processed(prepared, "val", 6),
                "clean": splits["candidate_clean"],
                "poison": splits["archived_poison"],
            }
            representations = paired_representations(
                clean_classical,
                poisoned_classical,
                clean_qml,
                poisoned_qml,
                groups,
            )
            config = {
                "seed": seed,
                "trigger": attack,
                "non_target": 0,
                "archive_ca": 0.0,
                "archive_asr": 0.0,
            }
            config_results = {}
            for name, values in representations.items():
                row, samples = evaluate_external_reference(
                    config,
                    name,
                    values["trusted"],
                    values["clean"],
                    values["poison"],
                    args.alpha,
                )
                rows.append(row)
                sample_rows.extend(samples)
                config_results[name] = (row, samples)
            for combined_name, internal_name, dimension in (
                (
                    "coherent_bonferroni_input_classical_margin",
                    "coherent_classical_margin_drift1",
                    1,
                ),
                (
                    "coherent_bonferroni_input_qml_xz",
                    "coherent_qml_xz_drift16",
                    16,
                ),
            ):
                row, samples = coherent.combine_margin_and_correlator(
                    config,
                    config_results["coherent_input256"][0],
                    config_results["coherent_input256"][1],
                    config_results[internal_name][1],
                    args.alpha,
                    combined_name=combined_name,
                    second_dimension=dimension,
                )
                row["trusted_reference_split"] = "validation_target"
                rows.append(row)
                sample_rows.extend(samples)
            print(
                f"seed={seed} attack={attack}: classical CA/ASR={classical_ca:.3f}/"
                f"{classical_asr:.3f}; QML={victim_row['qml_balanced_ca']:.3f}/"
                f"{victim_row['qml_asr']:.3f}",
                flush=True,
            )

    pooled = coherent.aggregate(rows, "trigger")
    eligible = eligible_summaries(rows, victims, args.attacks, args.role)
    unified.write_csv(args.output / "per_config_metrics.csv", rows)
    unified.write_csv(args.output / "per_sample_scores.csv", sample_rows)
    unified.write_csv(args.output / "pooled_summary.csv", pooled)
    unified.write_csv(args.output / "victim_metrics.csv", victims)
    unified.write_csv(args.output / "eligible_summary.csv", eligible)
    unified.json_dump(
        args.output / "manifest.json",
        {
            "experiment": "bloodmnist_cross_attack_pilot_v1",
            "role": args.role,
            "seeds": args.seeds,
            "attacks": args.attacks,
            "poison_count": 60,
            "alpha": args.alpha,
            "trusted_reference_split": "validation_target",
            "trusted_reference_count_per_config": 120,
            "trusted_calibration_count_per_config": 80,
            "victim_eligibility": {"balanced_ca_min": 0.90, "asr_min": 0.80},
            "diagnostic_gate": {"tpr_min": 0.80, "fpr_max": 0.10},
            "confirmatory_gate_not_applied_to_pilot": {
                "pooled_tpr_min": 0.90,
                "pooled_fpr_max": 0.05,
            },
            "clean_qml_checkpoint_root": str(args.clean_qml_root),
            "dataset_root": str(args.dataset_root),
            "dataset_npz_sha256": file_sha256(args.dataset_root / "bloodmnist.npz"),
            "attack_receipts": receipts,
            "blocked_families": {
                "input_aware": "requires joint pattern/mask generator training",
                "label_consistent": "requires adversarial base-image generation",
            },
            "elapsed_seconds": time.time() - started,
        },
    )
    write_report(args.output, victims, eligible, time.time() - started, args.role)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PILOT_SEEDS))
    parser.add_argument("--attacks", nargs="+", choices=ATTACKS, default=list(ATTACKS))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--adaptive-epochs", type=int, default=5)
    parser.add_argument("--classical-epochs", type=int, default=80)
    parser.add_argument(
        "--role",
        choices=("development_resource_gate_not_confirmation", "selective_confirmation"),
        default="development_resource_gate_not_confirmation",
    )
    parser.add_argument("--clean-qml-root", type=Path, default=REFERENCE_CLEAN_ROOT)
    main(parser.parse_args())
