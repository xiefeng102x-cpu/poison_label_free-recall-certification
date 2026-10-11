#!/usr/bin/env python3
"""Cross-dataset confirmation of clean-only low-dimensional poison screening."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# The reproducibility environment has the official BloodMNIST npz but not the
# small medmnist loader package.  Supply only the iterable interface used by
# evaluate_bloodmnist_checkpoints; no preprocessing behavior is replaced.
if "medmnist" not in sys.modules:
    medmnist_stub = types.ModuleType("medmnist")

    class BloodMNIST:
        def __init__(self, split: str, download: bool, root: str):
            if download:
                raise RuntimeError("This experiment requires the existing local npz")
            payload = np.load(Path(root) / "bloodmnist.npz")
            self.images = payload[f"{split}_images"]
            self.labels = payload[f"{split}_labels"]

        def __iter__(self):
            return iter(zip(self.images, self.labels))

    medmnist_stub.BloodMNIST = BloodMNIST
    sys.modules["medmnist"] = medmnist_stub

import run_bloodmnist_new_encoder_adaptive as adaptive
import train_bloodmnist_clean_encoder as clean_encoder
import run_unified_oneclass_poison_screening as unified


PAPER14_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = unified.PROJECT_ROOT / "data/MedMnist/single_detect_20260411/epsilon_0.8"
ADAPTIVE_ROOT = PAPER14_ROOT / "results/bloodmnist_new_encoder_adaptive_v1"
DEFAULT_OUTPUT = PAPER14_ROOT / "results/bloodmnist_unified_oneclass_v1"


def checkpoint_path(seed: int) -> Path:
    split = "delayed_test" if seed == 42 else "confirmation"
    return ADAPTIVE_ROOT / split / f"seed={seed}_lr=0.01_b=10.0_s=0.0_d=20.0_a=0.001.pt"


def clean_checkpoint_path(seed: int) -> Path:
    split = "development" if seed == 42 else "confirmation"
    return (
        PAPER14_ROOT
        / "results/bloodmnist_clean_encoder_v1"
        / split
        / f"seed_{seed}_local_l0.pt"
    )


def trigger_target(seed: int, checkpoint: dict, target_scaled: np.ndarray) -> np.ndarray:
    meta = torch.load(
        SOURCE_ROOT
        / f"seed_{seed}/layer_8_grid/t6_vs_t0/pr_0.1"
        / f"poisoned_samples_meta_seed_{seed}_pr_0.1.pt",
        map_location="cpu",
        weights_only=False,
    )
    trigger = meta["trigger_tensor"].numpy().astype(np.float32)
    mean = np.asarray(checkpoint["input_scaler_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["input_scaler_scale"], dtype=np.float32)
    target_pca = target_scaled * scale + mean
    return adaptive.transform(np.clip(target_pca + trigger, -3.0, 3.0), mean, scale)


def train_classical(
    seed: int,
    target: np.ndarray,
    source: np.ndarray,
    poison: np.ndarray,
    poison_indices: np.ndarray,
    epochs: int,
) -> unified.ClassicalVictim:
    source_clean_indices = np.setdiff1d(np.arange(len(source)), poison_indices)
    clean_train = np.concatenate([target, source[source_clean_indices]])
    mean = clean_train.mean(axis=0, keepdims=True)
    scale = clean_train.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    x = np.concatenate([target, source[source_clean_indices], poison])
    y = np.concatenate(
        [
            np.ones(len(target), dtype=np.int64),
            np.zeros(len(source_clean_indices), dtype=np.int64),
            np.ones(len(poison), dtype=np.int64),
        ]
    )
    dataset = torch.utils.data.TensorDataset(
        torch.tensor((x - mean) / scale, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(unified.RANDOM_SEED + seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, generator=generator
    )
    torch.manual_seed(seed)
    model = unified.BackdoorMLP(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
    return unified.ClassicalVictim(model=model, mean=mean, scale=scale, metrics={})


def train_clean_classical_companion(
    seed: int,
    target: np.ndarray,
    source: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    epochs: int,
) -> unified.ClassicalVictim:
    x = np.concatenate([target, source])
    y = np.concatenate(
        [np.ones(len(target), dtype=np.int64), np.zeros(len(source), dtype=np.int64)]
    )
    dataset = torch.utils.data.TensorDataset(
        torch.tensor((x - mean) / scale, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(unified.RANDOM_SEED + seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, generator=generator
    )
    # Match the backdoored companion's initialization exactly.
    torch.manual_seed(seed)
    model = unified.BackdoorMLP(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
    return unified.ClassicalVictim(
        model=copy.deepcopy(model), mean=mean, scale=scale, metrics={}
    )


def predictions(victim: unified.ClassicalVictim, x: np.ndarray) -> np.ndarray:
    z = (x - victim.mean) / victim.scale
    victim.model.eval()
    with torch.no_grad():
        return victim.model(torch.tensor(z, dtype=torch.float32)).argmax(1).numpy()


def qml_views(model: clean_encoder.CleanQNN, groups: dict[str, np.ndarray]) -> tuple[dict, dict]:
    z8, xz16 = {}, {}
    for name, x in groups.items():
        xyz = adaptive.numpy_xyz(model, x, batch_size=256)
        z8[name] = xyz[:, 16:24].astype(np.float64)
        xz16[name] = np.concatenate([xyz[:, :8], xyz[:, 16:24]], axis=1).astype(np.float64)
    return z8, xz16


def report(path: Path, summaries: list[dict], victims: list[dict], elapsed: float) -> None:
    lines = [
        "# BloodMNIST unified one-class confirmation",
        "",
        "Independent confirmation seeds 43--46 use official train target samples as reference, validation targets for threshold calibration, and untouched test targets/triggered non-targets for evaluation.",
        "",
        "| representation | pooled TPR | pooled FPR | mean AUC | clean-source flagged | triggered-target flagged |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['representation']} | {row['pooled_tpr']:.3f} | {row['pooled_fpr']:.3f} | "
            f"{row['mean_auc']:.3f} | {row['mean_clean_source_flag_rate']:.3f} | "
            f"{row['mean_trigger_target_flag_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| seed | classical balanced CA | classical ASR | QML balanced CA | QML ASR |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in victims:
        lines.append(
            f"| {row['seed']} | {row['classical_balanced_ca']:.3f} | {row['classical_asr']:.3f} | "
            f"{row['qml_balanced_ca']:.3f} | {row['qml_asr']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Detector fitting and threshold calibration use no triggered samples.",
            "",
            f"Runtime: {elapsed:.1f} seconds.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows, sample_rows, victim_rows = [], [], []
    for seed in args.seeds:
        checkpoint = torch.load(checkpoint_path(seed), map_location="cpu", weights_only=False)
        model = clean_encoder.CleanQNN("local", 0)
        model.load_state_dict(checkpoint["state_dict"])
        clean_checkpoint = torch.load(
            clean_checkpoint_path(seed), map_location="cpu", weights_only=False
        )
        if not np.array_equal(
            np.asarray(checkpoint["input_scaler_mean"]),
            np.asarray(clean_checkpoint["input_scaler_mean"]),
        ) or not np.array_equal(
            np.asarray(checkpoint["input_scaler_scale"]),
            np.asarray(clean_checkpoint["input_scaler_scale"]),
        ):
            raise RuntimeError("Clean/adaptive QML checkpoints use different input scalers")
        clean_qml_model = clean_encoder.CleanQNN("local", 0)
        clean_qml_model.load_state_dict(clean_checkpoint["state_dict"])
        splits, receipt = adaptive.load_data(seed, args.dataset_root, checkpoint)
        x_trigger_target = trigger_target(seed, checkpoint, splits["test"]["target"])

        meta = torch.load(
            SOURCE_ROOT
            / f"seed_{seed}/layer_8_grid/t6_vs_t0/pr_0.1"
            / f"poisoned_samples_meta_seed_{seed}_pr_0.1.pt",
            map_location="cpu",
            weights_only=False,
        )
        poison_indices = meta["poison_indices_in_non_target"].numpy().astype(int)
        victim = train_classical(
            seed,
            splits["train"]["target"],
            splits["train"]["non_target"],
            splits["archived_poison"],
            poison_indices,
            args.epochs,
        )
        clean_classical = train_clean_classical_companion(
            seed,
            splits["train"]["target"],
            splits["train"]["non_target"],
            victim.mean,
            victim.scale,
            args.epochs,
        )
        groups = {
            "reference": splits["train"]["target"],
            "calibration": splits["val"]["target"],
            "clean_target": splits["test"]["target"],
            "poison_source": splits["test"]["triggered_non_target"],
            "clean_source": splits["test"]["non_target"],
            "trigger_target": x_trigger_target,
        }
        hidden = {name: victim.representations(x)[0] for name, x in groups.items()}
        clean_hidden = {
            name: clean_classical.representations(x)[0] for name, x in groups.items()
        }
        margin = {name: victim.representations(x)[1] for name, x in groups.items()}
        clean_margin = {
            name: clean_classical.representations(x)[1] for name, x in groups.items()
        }
        qml_z8, qml_xz16 = qml_views(model, groups)
        clean_qml_z8, clean_qml_xz16 = qml_views(clean_qml_model, groups)
        representations = {
            "input256": {name: x.astype(np.float64) for name, x in groups.items()},
            "classical_hidden16": hidden,
            "classical_hidden_drift16": {
                name: hidden[name] - clean_hidden[name] for name in groups
            },
            "classical_margin_drift1": {
                name: margin[name] - clean_margin[name] for name in groups
            },
            "qml_z8": qml_z8,
            "qml_xz16": qml_xz16,
            "qml_z_drift8": {
                name: qml_z8[name] - clean_qml_z8[name] for name in groups
            },
            "qml_xz_drift16": {
                name: qml_xz16[name] - clean_qml_xz16[name] for name in groups
            },
        }
        for name, sets in representations.items():
            row, samples = unified.evaluate_representation(seed, name, sets, args.alpha)
            rows.append(row)
            sample_rows.extend(samples)

        classical_target = predictions(victim, splits["test"]["target"])
        classical_source = predictions(victim, splits["test"]["non_target"])
        classical_trigger = predictions(victim, splits["test"]["triggered_non_target"])
        qml_result = adaptive.evaluate_split(
            model,
            splits["test"],
            adaptive.numpy_xyz(model, splits["train"]["target"], 256),
            256,
        )
        victim_rows.append(
            {
                "seed": seed,
                "classical_balanced_ca": 0.5
                * (np.mean(classical_target == 1) + np.mean(classical_source == 0)),
                "classical_asr": float(np.mean(classical_trigger == 1)),
                "qml_balanced_ca": qml_result["clean"]["balanced_accuracy"],
                "qml_asr": qml_result["attack"]["asr"],
                "trigger_reconstruction_max_abs_error": receipt[
                    "triggered_reconstruction_max_abs_error"
                ],
            }
        )

    summaries = unified.pooled_summary(rows)
    unified.write_csv(args.output / "per_seed_metrics.csv", rows)
    unified.write_csv(args.output / "per_sample_scores.csv", sample_rows)
    unified.write_csv(args.output / "pooled_summary.csv", summaries)
    unified.write_csv(args.output / "victim_metrics.csv", victim_rows)
    elapsed = time.time() - started
    unified.json_dump(
        args.output / "manifest.json",
        {
            "experiment": "bloodmnist_unified_oneclass_v1",
            "seeds": args.seeds,
            "alpha": args.alpha,
            "classical_epochs": args.epochs,
            "dataset_root": str(args.dataset_root),
            "detector_training_labels": "clean target only; no triggered samples",
            "elapsed_seconds": elapsed,
        },
    )
    report(args.output / "EXPERIMENT_REPORT.md", summaries, victim_rows, elapsed)
    for row in summaries:
        print(
            f"{row['representation']:>20s}: TPR={row['pooled_tpr']:.3f} "
            f"FPR={row['pooled_fpr']:.3f} AUC={row['mean_auc']:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45, 46])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=80)
    main(parser.parse_args())
