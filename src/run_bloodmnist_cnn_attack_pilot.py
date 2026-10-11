#!/usr/bin/env python3
"""Image-native CNN pilot for spatial BloodMNIST attacks.

BadNets and WaNet were not reliably learned by the PCA-MLP/local-QML victims.
This resource gate changes only the victim architecture; attack tensors,
poison count, trusted reference, detector alpha, and eligibility gates remain
fixed.  Ineligible victims are retained but never used for defense claims.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import bloodmnist_attack_families as attack_ops
import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_cross_attack_pilot as cross
import run_bloodmnist_group_coherent_10seeds as baseline
import run_mnist_group_coherent_drift as coherent
import run_unified_oneclass_poison_screening as unified


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PAPER14_ROOT / "results/bloodmnist_cnn_attack_pilot_v1"
ATTACKS = ("badnets", "wanet", "label_consistent")
SEEDS = (42, 46, 51)


class SmallBloodCNN(nn.Module):
    def __init__(self, channel_mean: np.ndarray, channel_std: np.ndarray) -> None:
        super().__init__()
        self.register_buffer(
            "channel_mean", torch.as_tensor(channel_mean.reshape(1, 3, 1, 1))
        )
        self.register_buffer(
            "channel_std", torch.as_tensor(channel_std.reshape(1, 3, 1, 1))
        )
        self.features_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.projector = nn.Sequential(nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU())
        self.head = nn.Linear(64, 2)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.channel_mean) / self.channel_std
        return self.projector(self.features_net(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(x))


def nchw(images: np.ndarray) -> np.ndarray:
    return (
        np.asarray(images, dtype=np.float32)
        .reshape(-1, 28, 28, 3)
        .transpose(0, 3, 1, 2)
        .copy()
    )


def channel_stats(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = nchw(images)
    mean = values.mean(axis=(0, 2, 3)).astype(np.float32)
    std = values.std(axis=(0, 2, 3)).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def train_cnn(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    epochs: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> SmallBloodCNN:
    torch.manual_seed(seed)
    model = SmallBloodCNN(mean, std)
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(nchw(x), dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(20260928 + seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


def load_or_train_clean(
    output: Path,
    target: np.ndarray,
    source: np.ndarray,
    *,
    seed: int,
    epochs: int,
) -> SmallBloodCNN:
    root = output / "models/clean_cnn"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"seed_{seed}.pt"
    all_clean = np.concatenate([target, source], axis=0)
    labels = np.concatenate(
        [np.ones(len(target), dtype=np.int64), np.zeros(len(source), dtype=np.int64)]
    )
    mean, std = channel_stats(all_clean)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("epochs") == epochs:
            model = SmallBloodCNN(mean, std)
            model.load_state_dict(payload["state_dict"])
            return model
    model = train_cnn(
        all_clean, labels, seed=seed, epochs=epochs, mean=mean, std=std
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "seed": seed,
            "epochs": epochs,
            "mean": mean,
            "std": std,
        },
        path,
    )
    return model


def predict(model: SmallBloodCNN, images: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    output = []
    values = nchw(images)
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            logits = model(torch.as_tensor(values[start : start + batch_size]))
            output.append(logits.argmax(1).numpy())
    return np.concatenate(output)


def representations(
    model: SmallBloodCNN, images: np.ndarray, batch_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    hidden_rows, margin_rows = [], []
    values = nchw(images)
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(values[start : start + batch_size])
            hidden = model.hidden(batch)
            logits = model.head(hidden)
            hidden_rows.append(hidden.numpy().astype(np.float64))
            margin_rows.append(
                (logits[:, 1] - logits[:, 0]).numpy().reshape(-1, 1).astype(np.float64)
            )
    return np.concatenate(hidden_rows), np.concatenate(margin_rows)


def targeted_pgd_bases(
    model: SmallBloodCNN,
    images: np.ndarray,
    *,
    epsilon: float = 16.0 / 255.0,
    step_size: float = 2.0 / 255.0,
    steps: int = 10,
) -> np.ndarray:
    """Move target-class images toward the source class while preserving labels."""

    model.eval()
    original = torch.as_tensor(nchw(images), dtype=torch.float32)
    current = original.clone()
    target = torch.zeros(len(current), dtype=torch.long)
    for _ in range(steps):
        current.requires_grad_(True)
        loss = F.cross_entropy(model(current), target)
        (gradient,) = torch.autograd.grad(loss, current)
        with torch.no_grad():
            current = current - step_size * gradient.sign()
            current = torch.maximum(torch.minimum(current, original + epsilon), original - epsilon)
            current.clamp_(0.0, 1.0)
    return current.detach().numpy().transpose(0, 2, 3, 1).reshape(len(images), -1)


def apply_attack(
    attack: str,
    raw_images: np.ndarray,
    *,
    seed: int,
    grid: torch.Tensor | None,
) -> np.ndarray:
    if attack == "badnets":
        return attack_ops.apply_badnets_patch(raw_images)[0]
    if attack == "wanet":
        if grid is None:
            raise ValueError("WaNet grid required")
        return attack_ops.apply_wanet(raw_images, grid)
    if attack == "label_consistent":
        return attack_ops.apply_label_consistent_corners(raw_images)[0]
    raise ValueError(attack)


def build_raw_attack(
    prepared: dict,
    attack: str,
    seed: int,
    poison_count: int,
    clean_model: SmallBloodCNN,
) -> tuple[dict, dict]:
    raw = prepared["raw"]
    attack_offset = {"badnets": 0, "wanet": 3, "label_consistent": 4}[attack]
    rng = np.random.default_rng(20260928 + 101 * seed + attack_offset)
    target_order = rng.permutation(len(raw["train"][6]))
    source_order = rng.permutation(len(raw["train"][0]))
    if poison_count <= 0 or poison_count >= len(source_order):
        raise ValueError("poison_count must be within the non-target train population")
    poison_indices = (
        target_order[:poison_count]
        if attack == "label_consistent"
        else source_order[:poison_count]
    )
    grid, grid_receipt = (None, None)
    if attack == "wanet":
        grid, receipt = attack_ops.make_wanet_grid(seed=seed + 91001)
        grid_receipt = receipt.to_dict()

    triggered = {
        split: apply_attack(attack, raw[split][0], seed=seed, grid=grid)
        for split in ("train", "val", "test")
    }
    if attack == "label_consistent":
        adversarial_bases = targeted_pgd_bases(
            clean_model, raw["train"][6][poison_indices]
        )
        poison = attack_ops.apply_label_consistent_corners(adversarial_bases)[0]
        remaining_target = np.setdiff1d(
            np.arange(len(raw["train"][6])), poison_indices
        )
        train_target = raw["train"][6][remaining_target]
        train_source = raw["train"][0]
        candidate_clean = train_target
        poison_origin = "target_class_adversarial_base_clean_label"
    else:
        poison = triggered["train"][poison_indices]
        remaining_source = np.setdiff1d(
            np.arange(len(raw["train"][0])), poison_indices
        )
        train_target = raw["train"][6]
        train_source = raw["train"][0][remaining_source]
        candidate_clean = train_target
        poison_origin = "source_all_to_one"
    cross_receipt = None

    if attack == "wanet":
        combined = np.concatenate([train_target, train_source], axis=0)
        combined_y = np.concatenate(
            [
                np.ones(len(train_target), dtype=np.int64),
                np.zeros(len(train_source), dtype=np.int64),
            ]
        )
        cross_count = 2 * poison_count
        cross_indices = rng.permutation(len(combined))[:cross_count]
        cross_grids = attack_ops.make_wanet_cross_grids(
            grid, count=cross_count, seed=seed + 92003
        )
        cross_images = attack_ops.apply_wanet(combined[cross_indices], cross_grids)
        keep = np.setdiff1d(np.arange(len(combined)), cross_indices)
        clean_x = np.concatenate([combined[keep], cross_images], axis=0)
        clean_y = np.concatenate([combined_y[keep], combined_y[cross_indices]])
        target_cross = combined_y[cross_indices] == 1
        candidate_clean = np.concatenate(
            [combined[keep][combined_y[keep] == 1], cross_images[target_cross]], axis=0
        )
        cross_receipt = {
            "count": cross_count,
            "target_count": int(target_cross.sum()),
            "source_count": int((~target_cross).sum()),
            "cross_processed_sha256_float32": cross.array_sha256(cross_images),
        }
    else:
        clean_x = np.concatenate([train_target, train_source], axis=0)
        clean_y = np.concatenate(
            [
                np.ones(len(train_target), dtype=np.int64),
                np.zeros(len(train_source), dtype=np.int64),
            ]
        )

    return {
        "clean_x": clean_x,
        "clean_y": clean_y,
        "poison": poison,
        "candidate_clean": candidate_clean,
        "trusted": raw["val"][6],
        "test_target": raw["test"][6],
        "test_source": raw["test"][0],
        "test_triggered": triggered["test"],
    }, {
        "attack": attack,
        "seed": seed,
        "poison_count": poison_count,
        "poison_indices": poison_indices.tolist(),
        "poison_origin": poison_origin,
        "triggered_source_train_sha256_float32": cross.array_sha256(triggered["train"]),
        "poison_raw_sha256_float32": cross.array_sha256(poison),
        "grid": grid_receipt,
        "grid_sha256_float32": None if grid is None else cross.array_sha256(grid),
        "cross": cross_receipt,
        "adversarial_base": (
            {
                "method": "targeted_pgd_to_source_class",
                "epsilon": 16.0 / 255.0,
                "step_size": 2.0 / 255.0,
                "steps": 10,
                "adversarial_base_sha256_float32": cross.array_sha256(
                    adversarial_bases
                ),
            }
            if attack == "label_consistent"
            else None
        ),
    }


def paired_views(
    clean_model: SmallBloodCNN,
    backdoor_model: SmallBloodCNN,
    groups: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    output = {
        "coherent_cnn_margin_drift1": {},
        "coherent_cnn_hidden_drift64": {},
    }
    for name, values in groups.items():
        clean_hidden, clean_margin = representations(clean_model, values)
        backdoor_hidden, backdoor_margin = representations(backdoor_model, values)
        output["coherent_cnn_margin_drift1"][name] = backdoor_margin - clean_margin
        output["coherent_cnn_hidden_drift64"][name] = backdoor_hidden - clean_hidden
    return output


def main(args: argparse.Namespace) -> None:
    if args.role == "badnets_fresh_confirmation":
        if args.seeds != list(range(52, 62)):
            raise ValueError("badnets_fresh_confirmation is frozen to seeds 52--61")
        if args.attacks != ["badnets"] or args.poison_count != 60 or args.epochs != 30:
            raise ValueError(
                "badnets_fresh_confirmation requires badnets, poison_count=60, epochs=30"
            )
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    class_data = heldout.collect_classes(args.dataset_root)
    rows, samples, victims, receipts = [], [], [], []

    for seed in args.seeds:
        prepared = baseline.preprocess_seed(class_data, seed)
        raw = prepared["raw"]
        clean_model = load_or_train_clean(
            args.clean_cache_root or args.output,
            raw["train"][6],
            raw["train"][0],
            seed=seed,
            epochs=args.epochs,
        )
        clean_checkpoint = (
            (args.clean_cache_root or args.output)
            / "models/clean_cnn"
            / f"seed_{seed}.pt"
        )
        for attack in args.attacks:
            print(f"seed={seed} attack={attack}: training CNN victim", flush=True)
            data, receipt = build_raw_attack(
                prepared, attack, seed, args.poison_count, clean_model
            )
            receipt["clean_checkpoint"] = str(clean_checkpoint)
            receipt["clean_checkpoint_sha256"] = cross.file_sha256(clean_checkpoint)
            backdoor_x = np.concatenate([data["clean_x"], data["poison"]], axis=0)
            backdoor_y = np.concatenate(
                [data["clean_y"], np.ones(len(data["poison"]), dtype=np.int64)]
            )
            mean = clean_model.channel_mean.numpy().reshape(3)
            std = clean_model.channel_std.numpy().reshape(3)
            backdoor_model = train_cnn(
                backdoor_x,
                backdoor_y,
                seed=seed,
                epochs=args.epochs,
                mean=mean,
                std=std,
            )
            checkpoint = args.output / "models" / attack / f"seed_{seed}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": backdoor_model.state_dict(),
                    "seed": seed,
                    "attack": attack,
                    "epochs": args.epochs,
                },
                checkpoint,
            )
            receipt["checkpoint"] = str(checkpoint.relative_to(args.output))
            receipt["checkpoint_sha256"] = cross.file_sha256(checkpoint)
            receipts.append(receipt)

            target_pred = predict(backdoor_model, data["test_target"])
            source_pred = predict(backdoor_model, data["test_source"])
            attack_pred = predict(backdoor_model, data["test_triggered"])
            ca = 0.5 * (np.mean(target_pred == 1) + np.mean(source_pred == 0))
            asr = np.mean(attack_pred == 1)
            victims.append(
                {
                    "attack": attack,
                    "seed": seed,
                    "architecture": "small_cnn",
                    "balanced_ca": float(ca),
                    "asr": float(asr),
                    "eligible": int(ca >= 0.90 and asr >= 0.80),
                }
            )

            raw_groups = {
                "trusted": data["trusted"],
                "clean": data["candidate_clean"],
                "poison": data["poison"],
            }
            views = paired_views(clean_model, backdoor_model, raw_groups)
            views["coherent_input256"] = {
                name: cross.process_raw(prepared, values)
                for name, values in raw_groups.items()
            }
            config = {
                "seed": seed,
                "trigger": attack,
                "non_target": 0,
                "archive_ca": 0.0,
                "archive_asr": 0.0,
            }
            config_results = {}
            for name, values in views.items():
                row, row_samples = cross.evaluate_external_reference(
                    config,
                    name,
                    values["trusted"],
                    values["clean"],
                    values["poison"],
                    args.alpha,
                )
                rows.append(row)
                samples.extend(row_samples)
                config_results[name] = (row, row_samples)
            combined, combined_samples = coherent.combine_margin_and_correlator(
                config,
                config_results["coherent_input256"][0],
                config_results["coherent_input256"][1],
                config_results["coherent_cnn_margin_drift1"][1],
                args.alpha,
                combined_name="coherent_bonferroni_input_cnn_margin",
                second_dimension=1,
            )
            combined["trusted_reference_split"] = "validation_target"
            rows.append(combined)
            samples.extend(combined_samples)
            print(
                f"seed={seed} attack={attack}: CA/ASR={ca:.3f}/{asr:.3f}",
                flush=True,
            )

    pooled = coherent.aggregate(rows, "trigger")
    eligible_rows = []
    victim_map = {(row["attack"], row["seed"]): row for row in victims}
    for attack in args.attacks:
        for representation in (
            "coherent_bonferroni_input_cnn_margin",
            "coherent_cnn_hidden_drift64",
        ):
            selected = [
                row
                for row in rows
                if row["trigger"] == attack
                and row["representation"] == representation
                and victim_map[(attack, row["seed"])]["eligible"] == 1
            ]
            if selected:
                summary = coherent.aggregate(selected, "representation")[0]
                eligible_rows.append(
                    {
                        "attack": attack,
                        "representation": representation,
                        "eligible_seed_count": len(selected),
                        **{
                            key: value
                            for key, value in summary.items()
                            if key != "representation"
                        },
                    }
                )
            else:
                eligible_rows.append(
                    {
                        "attack": attack,
                        "representation": representation,
                        "eligible_seed_count": 0,
                    }
                )

    unified.write_csv(args.output / "per_config_metrics.csv", rows)
    unified.write_csv(args.output / "per_sample_scores.csv", samples)
    unified.write_csv(args.output / "victim_metrics.csv", victims)
    unified.write_csv(args.output / "pooled_summary.csv", pooled)
    unified.write_csv(args.output / "eligible_summary.csv", eligible_rows)
    unified.json_dump(
        args.output / "manifest.json",
        {
            "experiment": "bloodmnist_cnn_attack_pilot_v1",
            "role": args.role,
            "seeds": args.seeds,
            "attacks": args.attacks,
            "architecture": "two_conv_small_cnn_64_hidden",
            "epochs": args.epochs,
            "poison_count": args.poison_count,
            "alpha": args.alpha,
            "trusted_reference_split": "validation_target",
            "victim_eligibility": {"balanced_ca_min": 0.90, "asr_min": 0.80},
            "dataset_root": str(args.dataset_root),
            "clean_cache_root": str(args.clean_cache_root or args.output),
            "dataset_npz_sha256": cross.file_sha256(
                args.dataset_root / "bloodmnist.npz"
            ),
            "attack_receipts": receipts,
            "elapsed_seconds": time.time() - started,
        },
    )
    report_title = (
        "# BloodMNIST image-native CNN BadNets fresh-seed confirmation"
        if args.role == "badnets_fresh_confirmation"
        else "# BloodMNIST image-native CNN attack pilot"
    )
    lines = [
        report_title,
        "",
        "Victim eligibility requires balanced CA >= 0.90 and ASR >= 0.80.",
        "",
        "| attack | seed | balanced CA | ASR | eligible |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in victims:
        lines.append(
            f"| {row['attack']} | {row['seed']} | {row['balanced_ca']:.3f} | "
            f"{row['asr']:.3f} | {row['eligible']} |"
        )
    lines.extend(
        [
            "",
            "| attack | endpoint | eligible seeds | pooled TPR | pooled FPR |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in eligible_rows:
        if row["eligible_seed_count"]:
            lines.append(
                f"| {row['attack']} | {row['representation']} | "
                f"{row['eligible_seed_count']} | {row['pooled_tpr']:.3f} | "
                f"{row['pooled_fpr']:.3f} |"
            )
        else:
            lines.append(
                f"| {row['attack']} | {row['representation']} | 0 | — | — |"
            )
    lines.extend(["", f"Runtime: {time.time() - started:.1f} seconds.", ""])
    (args.output / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--attacks", nargs="+", choices=ATTACKS, default=list(ATTACKS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--poison-count", type=int, default=60)
    parser.add_argument("--clean-cache-root", type=Path)
    parser.add_argument(
        "--role",
        choices=("development_resource_gate_not_confirmation", "badnets_fresh_confirmation"),
        default="development_resource_gate_not_confirmation",
    )
    main(parser.parse_args())
