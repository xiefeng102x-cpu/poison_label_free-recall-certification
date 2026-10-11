#!/usr/bin/env python3
"""Mechanism and observable-dimension ablations for BloodMNIST CF-CMD."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_cross_attack_pilot as cross
import run_bloodmnist_group_coherent_10seeds as baseline
import run_bloodmnist_new_encoder_adaptive as adaptive
import run_bloodmnist_whitebox_adaptive_pilot as whitebox
import run_mnist_group_coherent_drift as coherent
import run_unified_oneclass_poison_screening as unified
import train_bloodmnist_clean_encoder as clean_encoder


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PAPER14_ROOT / "results/bloodmnist_mechanism_ablations_v1"
PROTOCOL = PAPER14_ROOT / "results/top_journal_program_v1/phase4_ablation_protocol.json"
MODEL_ROOT = PAPER14_ROOT / "results/bloodmnist_xz16_fresh_confirmation_v1"
SEEDS = tuple(range(52, 62))


def numpy_xz(model: clean_encoder.CleanQNN, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = adaptive.numpy_xyz(model, values, 256)
    return xyz[:, :8], xyz[:, 16:24]


def pauli_views(model: clean_encoder.CleanQNN, values: np.ndarray) -> dict[str, np.ndarray]:
    x, z = numpy_xz(model, values)
    zz, pairs = [], []
    for left in range(8):
        for right in range(left + 1, 8):
            zz.append(z[:, left] * z[:, right])
            pairs.extend(
                [
                    x[:, left] * x[:, right],
                    x[:, left] * z[:, right],
                    z[:, left] * x[:, right],
                    z[:, left] * z[:, right],
                ]
            )
    return {
        "z8": z,
        "xz16": np.concatenate([x, z], axis=1),
        "z_zz36": np.concatenate([z, np.stack(zz, axis=1)], axis=1),
        "pauli128": np.concatenate([x, z, np.stack(pairs, axis=1)], axis=1),
    }


def custom_crossfit(
    seed: int,
    name: str,
    trusted: np.ndarray,
    clean: np.ndarray,
    poison: np.ndarray,
    alpha: float,
    *,
    whitening: bool = True,
    sample_exclusion: bool = True,
    conformal: bool = True,
) -> tuple[dict, list[dict]]:
    trusted = np.asarray(trusted, dtype=np.float64)
    features = np.concatenate([clean, poison], axis=0).astype(np.float64)
    labels = np.concatenate([np.zeros(len(clean), dtype=int), np.ones(len(poison), dtype=int)])
    rng = np.random.default_rng(20260928 + 1009 * seed)
    order = rng.permutation(len(trusted))
    reference, calibration = trusted[order[:120]], trusted[order[120:]]
    if whitening:
        mean, transform = coherent.whitening(reference)
    else:
        mean = reference.mean(axis=0, keepdims=True)
        transform = np.eye(reference.shape[1])
    whiten = lambda values: (values - mean) @ transform
    reference_w, calibration_w = whiten(reference), whiten(calibration)
    scores = np.empty(len(features), dtype=np.float64)
    p_values = np.empty(len(features), dtype=np.float64)
    flags = np.zeros(len(features), dtype=bool)
    folds = np.array_split(rng.permutation(len(features)), 4)
    global_direction = whiten(features).mean(axis=0) - reference_w.mean(axis=0)
    global_direction /= max(float(np.linalg.norm(global_direction)), 1e-12)
    for query_indices in folds:
        halves = np.array_split(rng.permutation(query_indices), 2)
        for score_half, direction_half in ((halves[0], halves[1]), (halves[1], halves[0])):
            direction = (
                whiten(features[direction_half]).mean(axis=0) - reference_w.mean(axis=0)
                if sample_exclusion
                else global_direction
            )
            norm = float(np.linalg.norm(direction))
            if norm < 1e-12:
                direction = np.ones(features.shape[1], dtype=np.float64)
                norm = float(np.linalg.norm(direction))
            direction /= norm
            calibration_scores = calibration_w @ direction
            query_scores = whiten(features[score_half]) @ direction
            if conformal:
                query_p = coherent.one_sided_p_values(calibration_scores, query_scores)
                query_flags = query_p <= alpha
            else:
                threshold = float(calibration_scores.mean() + 1.6448536269514722 * calibration_scores.std(ddof=1))
                query_flags = query_scores >= threshold
                query_p = 1.0 - np.asarray(
                    [
                        0.5 * (1.0 + math.erf((score - calibration_scores.mean()) / (calibration_scores.std(ddof=1) * np.sqrt(2.0))))
                        for score in query_scores
                    ]
                )
            scores[score_half], p_values[score_half], flags[score_half] = query_scores, query_p, query_flags
    clean_flags, poison_flags = flags[labels == 0], flags[labels == 1]
    fp, tn = int(clean_flags.sum()), int((~clean_flags).sum())
    tp, fn = int(poison_flags.sum()), int((~poison_flags).sum())
    row = {
        "seed": seed,
        "representation": name,
        "dimension": features.shape[1],
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tpr": tp / (tp + fn),
        "fpr": fp / (fp + tn),
        "auc_crossfit_score": float(roc_auc_score(labels, scores)),
        "auc_conformal_rank": float(roc_auc_score(labels, -p_values)),
        "config_pass": int(tp / (tp + fn) >= 0.80 and fp / (fp + tn) <= 0.10),
    }
    samples = [
        {
            "seed": seed,
            "representation": name,
            "sample_index": index,
            "label": int(label),
            "score": float(score),
            "p_value": float(p_value),
            "flagged": int(flag),
        }
        for index, (label, score, p_value, flag) in enumerate(zip(labels, scores, p_values, flags))
    ]
    return row, samples


def main(args: argparse.Namespace) -> None:
    if tuple(args.seeds) != SEEDS:
        raise ValueError("Ablations are frozen to seeds 52--61")
    args.output.mkdir(parents=True, exist_ok=True)
    class_data = heldout.collect_classes(args.dataset_root)
    rgb_trigger, _ = baseline.fixed_rgb_trigger(class_data)
    rows, samples, victims = [], [], []
    for seed in args.seeds:
        print(f"seed={seed}: mechanism and Pauli ablations", flush=True)
        prepared = baseline.preprocess_seed(class_data, seed)
        splits, _ = baseline.seed_splits(prepared, rgb_trigger, seed)
        clean_model, _ = cross.load_reference_clean_model(seed, MODEL_ROOT)
        attacked_model = clean_encoder.CleanQNN("local", 0)
        attacked_model.load_state_dict(
            torch.load(MODEL_ROOT / "models" / f"seed_{seed}_adaptive.pt", map_location="cpu", weights_only=False)["state_dict"]
        )
        ca, asr = whitebox.balanced_ca_and_asr(
            attacked_model,
            splits["test"]["target"],
            splits["test"]["non_target"],
            splits["test"]["triggered_non_target"],
        )
        victims.append({"seed": seed, "balanced_ca": ca, "asr": asr, "eligible": int(ca >= 0.90 and asr >= 0.80)})
        groups = {
            "trusted": splits["val"]["target"],
            "clean": splits["train"]["target"],
            "poison": splits["archived_poison"],
        }
        clean_views = {name: pauli_views(clean_model, values) for name, values in groups.items()}
        attacked_views = {name: pauli_views(attacked_model, values) for name, values in groups.items()}
        drift = {
            view: {name: attacked_views[name][view] - clean_views[name][view] for name in groups}
            for view in ("z8", "xz16", "z_zz36", "pauli128")
        }
        config = {"seed": seed, "trigger": "fixed_pca_additive", "archive_ca": ca, "archive_asr": asr}
        evaluated = {}
        for view, label in (
            ("z8", "dimension_z8"),
            ("xz16", "internal_xz16_only"),
            ("z_zz36", "dimension_z_plus_zz36"),
            ("pauli128", "dimension_pauli128"),
        ):
            result = cross.evaluate_external_reference(
                config, label, drift[view]["trusted"], drift[view]["clean"], drift[view]["poison"], args.alpha
            )
            rows.append(result[0]); samples.extend(result[1]); evaluated[label] = result
        input_result = cross.evaluate_external_reference(
            config, "input_only", groups["trusted"], groups["clean"], groups["poison"], args.alpha
        )
        rows.append(input_result[0]); samples.extend(input_result[1]); evaluated["input_only"] = input_result
        primary = coherent.combine_margin_and_correlator(
            config,
            input_result[0],
            input_result[1],
            evaluated["internal_xz16_only"][1],
            args.alpha,
            combined_name="bonferroni_input_plus_xz16",
            second_dimension=16,
        )
        rows.append(primary[0]); samples.extend(primary[1])
        absolute = cross.evaluate_external_reference(
            config,
            "no_checkpoint_difference_absolute_xz16",
            attacked_views["trusted"]["xz16"],
            attacked_views["clean"]["xz16"],
            attacked_views["poison"]["xz16"],
            args.alpha,
        )
        rows.append(absolute[0]); samples.extend(absolute[1])
        candidate_reference = coherent.evaluate_config(
            {**config, "non_target": 0},
            "candidate_reference_xz16",
            drift["xz16"]["clean"],
            drift["xz16"]["poison"],
            args.alpha,
        )
        rows.append(candidate_reference[0]); samples.extend(candidate_reference[1])
        for label, options in (
            ("no_sample_exclusion_xz16", {"sample_exclusion": False}),
            ("no_whitening_xz16", {"whitening": False}),
            ("no_conformal_gaussian_xz16", {"conformal": False}),
        ):
            result = custom_crossfit(
                seed,
                label,
                drift["xz16"]["trusted"],
                drift["xz16"]["clean"],
                drift["xz16"]["poison"],
                args.alpha,
                **options,
            )
            rows.append(result[0]); samples.extend(result[1])

    eligible = {int(row["seed"]) for row in victims if row["eligible"]}
    summaries = []
    for representation in sorted({row["representation"] for row in rows}):
        selected = [row for row in rows if row["representation"] == representation and int(row["seed"]) in eligible]
        if selected:
            summary = coherent.aggregate(selected, "representation")[0]
            summaries.append({"representation": representation, "eligible_seed_count": len(selected), **{key: value for key, value in summary.items() if key != "representation"}})
    unified.write_csv(args.output / "per_config_metrics.csv", rows)
    unified.write_csv(args.output / "per_sample_scores.csv", samples)
    unified.write_csv(args.output / "victim_metrics.csv", victims)
    unified.write_csv(args.output / "eligible_summary.csv", summaries)
    unified.json_dump(
        args.output / "manifest.json",
        {
            "experiment": "bloodmnist_mechanism_ablations_v1",
            "role": "retrospective_mechanism_ablation_not_new_confirmation",
            "protocol": str(PROTOCOL),
            "protocol_sha256": whitebox.sha256_file(PROTOCOL),
            "dataset_npz_sha256": whitebox.sha256_file(args.dataset_root / "bloodmnist.npz"),
            "seeds": list(args.seeds),
            "alpha": args.alpha,
        },
    )
    lines = [
        "# BloodMNIST CF-CMD mechanism and dimension ablations",
        "",
        "These are retrospective mechanism ablations on already-used seeds, not confirmation of a revised deployment rule.",
        "",
        "| ablation | eligible seeds | pooled TPR | pooled FPR |",
        "|---|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(f"| {row['representation']} | {row['eligible_seed_count']} | {row['pooled_tpr']:.3f} | {row['pooled_fpr']:.3f} |")
    lines.append("")
    (args.output / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--alpha", type=float, default=0.05)
    main(parser.parse_args())
