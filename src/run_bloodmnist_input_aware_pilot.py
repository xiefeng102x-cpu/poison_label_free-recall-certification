#!/usr/bin/env python3
"""Clean-room Input-Aware-style BloodMNIST attack and CF-CMD pilot.

The implementation preserves the defining mechanisms: a separately trained
sample-conditioned sparse mask generator, a jointly trained pattern generator,
all-to-one poison samples, correct-label cross samples, and diversity losses.
It is labeled Input-Aware-style rather than claimed bit-identical to the
reference implementation.
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

import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_cnn_attack_pilot as cnn
import run_bloodmnist_cross_attack_pilot as cross
import run_bloodmnist_group_coherent_10seeds as baseline
import run_mnist_group_coherent_drift as coherent
import run_unified_oneclass_poison_screening as unified


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PAPER14_ROOT / "results/bloodmnist_input_aware_pilot_v1"
SEEDS = (42, 46, 51)


class TinyGenerator(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def threshold_mask(values: torch.Tensor) -> torch.Tensor:
    return torch.tanh(values * 20.0 - 10.0) / (2.0 + 1e-7) + 0.5


def diversity_loss(
    inputs1: torch.Tensor,
    inputs2: torch.Tensor,
    outputs1: torch.Tensor,
    outputs2: torch.Tensor,
) -> torch.Tensor:
    input_distance = torch.sqrt(
        torch.mean((inputs1 - inputs2) ** 2, dim=(1, 2, 3)) + 1e-8
    )
    output_distance = torch.sqrt(
        torch.mean((outputs1 - outputs2) ** 2, dim=(1, 2, 3)) + 1e-8
    )
    # The reference uses EPSILON=1e-7.  On this small CPU network, identical
    # early outputs can create explosive gradients, so the clean-room port uses
    # a frozen 1e-4 denominator floor and caps the per-sample ratio at 100.
    ratio = input_distance / (output_distance + 1e-4)
    return torch.mean(torch.clamp(ratio, max=100.0))


def train_mask_generator(
    images: np.ndarray,
    *,
    seed: int,
    epochs: int,
    mask_density: float,
) -> tuple[TinyGenerator, list[dict]]:
    torch.manual_seed(seed + 30001)
    generator = TinyGenerator(1)
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3, betas=(0.5, 0.9))
    values = torch.as_tensor(cnn.nchw(images), dtype=torch.float32)
    rng = np.random.default_rng(20260928 + seed + 30001)
    history = []
    for epoch in range(1, epochs + 1):
        order1 = rng.permutation(len(values))
        order2 = rng.permutation(len(values))
        epoch_norm, epoch_div = [], []
        generator.train()
        for start in range(0, len(values), 64):
            idx1 = order1[start : start + 64]
            idx2 = order2[start : start + len(idx1)]
            first, second = values[idx1], values[idx2]
            mask1 = threshold_mask(generator(first))
            mask2 = threshold_mask(generator(second))
            loss_norm = torch.mean(F.relu(mask1 - mask_density))
            loss_div = diversity_loss(first, second, mask1, mask2)
            loss = 100.0 * loss_norm + loss_div
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            optimizer.step()
            epoch_norm.append(float(loss_norm.detach()))
            epoch_div.append(float(loss_div.detach()))
        history.append(
            {
                "epoch": epoch,
                "mask_norm_loss": float(np.mean(epoch_norm)),
                "mask_diversity_loss": float(np.mean(epoch_div)),
            }
        )
    generator.eval()
    generator.requires_grad_(False)
    return generator, history


def apply_generated(
    images: torch.Tensor,
    pattern_generator: TinyGenerator,
    mask_generator: TinyGenerator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    patterns = pattern_generator(images)
    masks = threshold_mask(mask_generator(images)).repeat(1, 3, 1, 1)
    poisoned = images + (patterns - images) * masks
    return poisoned, patterns, masks


def train_joint_attack(
    images: np.ndarray,
    labels: np.ndarray,
    clean_model: cnn.SmallBloodCNN,
    mask_generator: TinyGenerator,
    *,
    seed: int,
    epochs: int,
    poison_ratio: float,
) -> tuple[cnn.SmallBloodCNN, TinyGenerator, list[dict]]:
    mean = clean_model.channel_mean.numpy().reshape(3)
    std = clean_model.channel_std.numpy().reshape(3)
    torch.manual_seed(seed)
    classifier = cnn.SmallBloodCNN(mean, std)
    torch.manual_seed(seed + 40001)
    pattern_generator = TinyGenerator(3)
    optimizer_c = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer_g = torch.optim.Adam(
        pattern_generator.parameters(), lr=1e-3, betas=(0.5, 0.9)
    )
    x = torch.as_tensor(cnn.nchw(images), dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    rng = np.random.default_rng(20260928 + seed + 40001)
    history = []
    for epoch in range(1, epochs + 1):
        order1 = rng.permutation(len(x))
        order2 = rng.permutation(len(x))
        losses, attack_hits, clean_hits, cross_hits = [], [], [], []
        classifier.train()
        pattern_generator.train()
        for start in range(0, len(x), 64):
            idx1 = order1[start : start + 64]
            idx2 = order2[start : start + len(idx1)]
            first, first_y = x[idx1], y[idx1]
            second = x[idx2]
            num_bd = max(1, int(poison_ratio * len(first)))
            num_cross = min(num_bd, len(first) - num_bd)
            bd, patterns1, _ = apply_generated(
                first[:num_bd], pattern_generator, mask_generator
            )
            cross_condition = second[num_bd : num_bd + num_cross]
            patterns2 = pattern_generator(cross_condition)
            masks2 = threshold_mask(mask_generator(cross_condition)).repeat(
                1, 3, 1, 1
            )
            cross_base = first[num_bd : num_bd + num_cross]
            cross_inputs = cross_base + (patterns2 - cross_base) * masks2
            cross_inputs.clamp_(0.0, 1.0)
            total_x = torch.cat(
                [bd, cross_inputs, first[num_bd + num_cross :]], dim=0
            )
            total_y = torch.cat(
                [
                    torch.ones(num_bd, dtype=torch.long),
                    first_y[num_bd:],
                ]
            )
            logits = classifier(total_x)
            loss_ce = F.cross_entropy(logits, total_y)
            loss_div = diversity_loss(
                first[:num_bd],
                second[num_bd : num_bd + num_bd],
                patterns1,
                patterns2[:num_bd],
            )
            loss = loss_ce + loss_div
            optimizer_c.zero_grad()
            optimizer_g.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(pattern_generator.parameters(), 5.0)
            optimizer_c.step()
            optimizer_g.step()
            prediction = logits.argmax(1)
            losses.append(float(loss.detach()))
            attack_hits.append(float(torch.mean((prediction[:num_bd] == 1).float())))
            cross_hits.append(
                float(
                    torch.mean(
                        (
                            prediction[num_bd : num_bd + num_cross]
                            == first_y[num_bd : num_bd + num_cross]
                        ).float()
                    )
                )
            )
            clean_hits.append(
                float(
                    torch.mean(
                        (
                            prediction[num_bd + num_cross :]
                            == first_y[num_bd + num_cross :]
                        ).float()
                    )
                )
                if len(first) > num_bd + num_cross
                else 1.0
            )
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "train_attack_accuracy": float(np.mean(attack_hits)),
                "train_cross_accuracy": float(np.mean(cross_hits)),
                "train_clean_accuracy": float(np.mean(clean_hits)),
            }
        )
    classifier.eval()
    pattern_generator.eval()
    return classifier, pattern_generator, history


def generated_numpy(
    images: np.ndarray,
    pattern_generator: TinyGenerator,
    mask_generator: TinyGenerator,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_poison, all_pattern, all_mask = [], [], []
    values = cnn.nchw(images)
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(values[start : start + batch_size])
            poison, pattern, mask = apply_generated(
                batch, pattern_generator, mask_generator
            )
            all_poison.append(poison.numpy())
            all_pattern.append(pattern.numpy())
            all_mask.append(mask.numpy())

    def flat(rows: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(rows).transpose(0, 2, 3, 1).reshape(-1, 28 * 28 * 3)

    output = (flat(all_poison), flat(all_pattern), flat(all_mask))
    if not all(np.all(np.isfinite(values)) for values in output):
        raise RuntimeError("Input-Aware generator produced non-finite values")
    return output


def generator_receipt(
    source_images: np.ndarray,
    patterns: np.ndarray,
    masks: np.ndarray,
) -> dict:
    pattern_hwc = patterns.reshape(-1, 28, 28, 3)
    mask_hwc = masks.reshape(-1, 28, 28, 3)
    flattened = pattern_hwc.reshape(len(pattern_hwc), -1)
    centered = flattened - flattened.mean(axis=0, keepdims=True)
    return {
        "mask_mean": float(mask_hwc.mean()),
        "mask_fraction_over_0_5": float(np.mean(mask_hwc > 0.5)),
        "pattern_samplewise_std_mean": float(np.mean(np.std(flattened, axis=0))),
        "pattern_centered_frobenius": float(np.linalg.norm(centered)),
        "input_pattern_correlation": float(
            np.corrcoef(source_images.reshape(-1), patterns.reshape(-1))[0, 1]
        ),
        "poison_sha256_float32": cross.array_sha256(
            source_images + (patterns - source_images) * masks
        ),
        "pattern_sha256_float32": cross.array_sha256(patterns),
        "mask_sha256_float32": cross.array_sha256(masks),
    }


def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    class_data = heldout.collect_classes(args.dataset_root)
    rows, samples, victims, receipts = [], [], [], []
    for seed in args.seeds:
        print(f"seed={seed}: preparing Input-Aware generators", flush=True)
        prepared = baseline.preprocess_seed(class_data, seed)
        raw = prepared["raw"]
        clean_model = cnn.load_or_train_clean(
            args.clean_cache_root or args.output,
            raw["train"][6],
            raw["train"][0],
            seed=seed,
            epochs=args.classifier_epochs,
        )
        train_x = np.concatenate([raw["train"][6], raw["train"][0]], axis=0)
        train_y = np.concatenate(
            [
                np.ones(len(raw["train"][6]), dtype=np.int64),
                np.zeros(len(raw["train"][0]), dtype=np.int64),
            ]
        )
        mask_generator, mask_history = train_mask_generator(
            train_x,
            seed=seed,
            epochs=args.mask_epochs,
            mask_density=args.mask_density,
        )
        backdoor_model, pattern_generator, attack_history = train_joint_attack(
            train_x,
            train_y,
            clean_model,
            mask_generator,
            seed=seed,
            epochs=args.attack_epochs,
            poison_ratio=args.poison_ratio,
        )
        triggered_test, _, _ = generated_numpy(
            raw["test"][0], pattern_generator, mask_generator
        )
        target_prediction = cnn.predict(backdoor_model, raw["test"][6])
        source_prediction = cnn.predict(backdoor_model, raw["test"][0])
        attack_prediction = cnn.predict(backdoor_model, triggered_test)
        ca = 0.5 * (
            np.mean(target_prediction == 1) + np.mean(source_prediction == 0)
        )
        asr = np.mean(attack_prediction == 1)

        rng = np.random.default_rng(20260928 + 101 * seed + 5)
        candidate_indices = rng.permutation(len(raw["train"][0]))[: args.candidate_count]
        candidate_source = raw["train"][0][candidate_indices]
        poison, patterns, masks = generated_numpy(
            candidate_source, pattern_generator, mask_generator
        )
        receipt = generator_receipt(candidate_source, patterns, masks)
        receipt.update(
            {
                "seed": seed,
                "attack": "input_aware_style",
                "candidate_indices": candidate_indices.tolist(),
                "candidate_count": args.candidate_count,
                "poison_ratio_per_batch": args.poison_ratio,
                "mask_density": args.mask_density,
                "mask_epochs": args.mask_epochs,
                "attack_epochs": args.attack_epochs,
                "mechanisms": [
                    "sample_conditioned_pattern_generator",
                    "sample_conditioned_sparse_mask_generator",
                    "mask_pretraining_with_norm_and_diversity",
                    "joint_classifier_pattern_training",
                    "correct_label_cross_samples",
                ],
                "mask_history": mask_history,
                "attack_history": attack_history,
            }
        )
        checkpoint = args.output / "models" / f"seed_{seed}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "classifier": backdoor_model.state_dict(),
                "pattern_generator": pattern_generator.state_dict(),
                "mask_generator": mask_generator.state_dict(),
                "seed": seed,
                "configuration": {
                    "mask_epochs": args.mask_epochs,
                    "attack_epochs": args.attack_epochs,
                    "poison_ratio": args.poison_ratio,
                    "mask_density": args.mask_density,
                },
            },
            checkpoint,
        )
        receipt["checkpoint"] = str(checkpoint.relative_to(args.output))
        receipt["checkpoint_sha256"] = cross.file_sha256(checkpoint)
        receipts.append(receipt)
        victims.append(
            {
                "attack": "input_aware_style",
                "seed": seed,
                "balanced_ca": float(ca),
                "asr": float(asr),
                "eligible": int(ca >= 0.90 and asr >= 0.80),
                "mask_mean": receipt["mask_mean"],
                "mask_fraction_over_0_5": receipt["mask_fraction_over_0_5"],
                "pattern_samplewise_std_mean": receipt[
                    "pattern_samplewise_std_mean"
                ],
            }
        )

        raw_groups = {
            "trusted": raw["val"][6],
            "clean": raw["train"][6],
            "poison": poison,
        }
        views = cnn.paired_views(clean_model, backdoor_model, raw_groups)
        views["coherent_input256"] = {
            name: cross.process_raw(prepared, values) for name, values in raw_groups.items()
        }
        config = {
            "seed": seed,
            "trigger": "input_aware_style",
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
            f"seed={seed}: CA/ASR={ca:.3f}/{asr:.3f}; "
            f"mask_mean={receipt['mask_mean']:.4f}; "
            f"pattern_std={receipt['pattern_samplewise_std_mean']:.4f}",
            flush=True,
        )

    pooled = coherent.aggregate(rows, "trigger")
    victim_map = {int(row["seed"]): row for row in victims}
    eligible_rows = []
    for representation in (
        "coherent_bonferroni_input_cnn_margin",
        "coherent_cnn_hidden_drift64",
    ):
        selected = [
            row
            for row in rows
            if row["representation"] == representation
            and victim_map[int(row["seed"])]["eligible"] == 1
        ]
        if selected:
            summary = coherent.aggregate(selected, "representation")[0]
            eligible_rows.append(
                {
                    "representation": representation,
                    "eligible_seed_count": len(selected),
                    **{key: value for key, value in summary.items() if key != "representation"},
                }
            )
        else:
            eligible_rows.append(
                {"representation": representation, "eligible_seed_count": 0}
            )

    unified.write_csv(args.output / "per_config_metrics.csv", rows)
    unified.write_csv(args.output / "per_sample_scores.csv", samples)
    unified.write_csv(args.output / "victim_metrics.csv", victims)
    unified.write_csv(args.output / "pooled_summary.csv", pooled)
    unified.write_csv(args.output / "eligible_summary.csv", eligible_rows)
    unified.json_dump(
        args.output / "manifest.json",
        {
            "experiment": "bloodmnist_input_aware_pilot_v1",
            "role": "development_resource_gate_not_confirmation",
            "attack_label": "Input-Aware-style clean-room implementation",
            "seeds": args.seeds,
            "classifier_epochs": args.classifier_epochs,
            "mask_epochs": args.mask_epochs,
            "attack_epochs": args.attack_epochs,
            "poison_ratio": args.poison_ratio,
            "candidate_count": args.candidate_count,
            "mask_density": args.mask_density,
            "alpha": args.alpha,
            "trusted_reference_split": "validation_target",
            "victim_eligibility": {"balanced_ca_min": 0.90, "asr_min": 0.80},
            "dataset_root": str(args.dataset_root),
            "dataset_npz_sha256": cross.file_sha256(
                args.dataset_root / "bloodmnist.npz"
            ),
            "receipts": receipts,
            "elapsed_seconds": time.time() - started,
        },
    )
    lines = [
        "# BloodMNIST Input-Aware-style pilot",
        "",
        "This clean-room implementation preserves sample-conditioned pattern/mask generators, mask sparsity/diversity training, joint pattern/classifier training, and correct-label cross samples. It is not claimed bit-identical to the reference code.",
        "",
        "| seed | balanced CA | ASR | eligible | mask mean | pattern variability |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in victims:
        lines.append(
            f"| {row['seed']} | {row['balanced_ca']:.3f} | {row['asr']:.3f} | "
            f"{row['eligible']} | {row['mask_mean']:.4f} | "
            f"{row['pattern_samplewise_std_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| endpoint | eligible seeds | pooled TPR | pooled FPR |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in eligible_rows:
        if row["eligible_seed_count"]:
            lines.append(
                f"| {row['representation']} | {row['eligible_seed_count']} | "
                f"{row['pooled_tpr']:.3f} | {row['pooled_fpr']:.3f} |"
            )
        else:
            lines.append(f"| {row['representation']} | 0 | — | — |")
    lines.extend(["", f"Runtime: {time.time() - started:.1f} seconds.", ""])
    (args.output / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clean-cache-root", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--classifier-epochs", type=int, default=30)
    parser.add_argument("--mask-epochs", type=int, default=25)
    parser.add_argument("--attack-epochs", type=int, default=30)
    parser.add_argument("--poison-ratio", type=float, default=0.10)
    parser.add_argument("--candidate-count", type=int, default=60)
    parser.add_argument("--mask-density", type=float, default=0.032)
    parser.add_argument("--alpha", type=float, default=0.05)
    main(parser.parse_args())
