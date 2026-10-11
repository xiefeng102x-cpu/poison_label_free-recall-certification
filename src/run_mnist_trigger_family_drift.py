#!/usr/bin/env python3
"""Fixed-rule checkpoint-drift screening across eligible MNIST triggers.

Primary positives are the clean-label poisoned target samples that actually
entered training. Triggered non-target samples are reported only as an attack
transfer control. Detector fitting and calibration use clean target samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import run_unified_oneclass_poison_screening as unified
import fast_circuit as FAST


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER14_ROOT = PROJECT_ROOT / "paper14"
DATA_ROOT = PROJECT_ROOT / "data/Mnist/detail_single_samle"
ELIGIBILITY = PAPER14_ROOT / "results/mnist_trigger_family_compression_v1/eligibility.csv"
DEFAULT_OUTPUT = PAPER14_ROOT / "results/mnist_trigger_family_drift_v1"
RANDOM_SEED = 20260928


def all_configs() -> list[dict]:
    with ELIGIBILITY.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "seed": int(row["seed"]),
            "trigger": row["trigger"],
            "non_target": int(row["trigger"].split("t7_vs_t")[1]),
            "archive_ca": float(row["clean_accuracy_percent"]) / 100.0,
            "archive_asr": float(row["attack_success_rate_percent"]) / 100.0,
            "archive_eligible": row["eligible"].lower() == "true",
        }
        for row in rows
    ]


def eligible_configs() -> list[dict]:
    return [row for row in all_configs() if row["archive_eligible"]]


def config_root(seed: int, trigger: str) -> Path:
    return DATA_ROOT / f"seed_{seed}/layer_12_grid/{trigger}/pr_0.3"


def load_config(config: dict) -> tuple[dict[str, np.ndarray], dict, dict]:
    root = config_root(config["seed"], config["trigger"])
    meta = torch.load(
        root / f"poisoned_samples_meta_seed_{config['seed']}_pr_0.3.pt",
        map_location="cpu",
        weights_only=False,
    )
    clean = torch.load(
        root / "models" / f"clean_model_seed_{config['seed']}_epoch_200.pt",
        map_location="cpu",
        weights_only=False,
    )
    backdoor = torch.load(
        root / "models" / f"backdoor_model_seed_{config['seed']}_pr_0.3.pt",
        map_location="cpu",
        weights_only=False,
    )
    poison_source_indices = meta["poison_indices_in_non_target"].numpy().astype(int)
    arrays = {
        "clean_target_all": meta["target_clean_data"].numpy().astype(np.float64),
        "poison_target": meta["poisoned_target_data"].numpy().astype(np.float64),
        "clean_source": meta["non_target_clean_data"].numpy().astype(np.float64)[
            poison_source_indices
        ],
        "trigger_source": meta["poisoned_non_target_data"].numpy().astype(np.float64),
        "clean_source_all": meta["non_target_clean_data"].numpy().astype(np.float64),
    }
    return arrays, clean, backdoor


def split_clean_target(n: int, seed: int, non_target: int) -> dict[str, np.ndarray]:
    if n < 420:
        raise ValueError("MNIST protocol requires 420 unpoisoned target samples")
    order = np.random.default_rng(RANDOM_SEED + 101 * seed + non_target).permutation(n)
    return {
        "reference": order[:210],
        "calibration": order[210:315],
        "clean_target": order[315:420],
    }


def circuit_features(samples: np.ndarray, weights_flat: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(samples, dtype=np.float64).reshape(len(samples), -1)
    norms = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    psi = (x / norms).astype(np.complex128)
    weights = np.asarray(weights_flat, dtype=np.float64).reshape(12, 5, 2)
    for layer in range(12):
        for qubit in range(5):
            theta, phi = weights[layer, qubit]
            cosine, sine = np.cos(theta / 2), np.sin(theta / 2)
            ry = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.complex128)
            psi = FAST._apply_single_qubit(psi, ry, qubit)
            phase = np.exp(1j * phi / 2)
            rz = np.asarray([[1 / phase, 0], [0, phase]], dtype=np.complex128)
            psi = FAST._apply_single_qubit(psi, rz, qubit)
        for qubit in range(5):
            psi = FAST._apply_cnot(psi, qubit, (qubit + 1) % 5)
    z_evals, zz_evals, _ = FAST._make_evals(5)
    probabilities = np.abs(psi) ** 2
    z = probabilities @ z_evals.T
    zz = probabilities @ zz_evals.T
    x_values = np.stack(
        [
            np.abs(FAST._apply_single_qubit(psi, FAST._H_GATE, qubit)) ** 2
            @ z_evals[qubit]
            for qubit in range(5)
        ],
        axis=1,
    )
    return {
        "z5": z.astype(np.float64),
        "xz10": np.concatenate([x_values, z], axis=1).astype(np.float64),
        "z_zz15": np.concatenate([z, zz], axis=1).astype(np.float64),
    }


def classifier_margin(z: np.ndarray, state: dict) -> np.ndarray:
    w1 = state["classifier.0.weight"].detach().cpu().numpy()
    b1 = state["classifier.0.bias"].detach().cpu().numpy()
    w2 = state["classifier.3.weight"].detach().cpu().numpy()
    b2 = state["classifier.3.bias"].detach().cpu().numpy()
    hidden = np.maximum(z @ w1.T + b1, 0.0)
    logits = hidden @ w2.T + b2
    return (logits[:, 1] - logits[:, 0]).reshape(-1, 1).astype(np.float64)


def load_or_compute_views(
    output: Path,
    config: dict,
    arrays: dict[str, np.ndarray],
    clean_state: dict,
    backdoor_state: dict,
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    cache = output / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{config['trigger']}_seed{config['seed']}_checkpoint_views.npz"
    group_order = (
        "clean_target_all",
        "poison_target",
        "clean_source",
        "trigger_source",
        "clean_source_all",
    )
    lengths = [len(arrays[group]) for group in group_order]
    if path.exists():
        payload = np.load(path)
        stored = {key: payload[key] for key in payload.files}
    else:
        combined = np.concatenate([arrays[group] for group in group_order])
        clean_features = circuit_features(combined, clean_state["weights"].numpy())
        backdoor_features = circuit_features(combined, backdoor_state["weights"].numpy())
        stored = {}
        offsets = np.cumsum([0] + lengths)
        for model_name, features in (
            ("clean", clean_features),
            ("backdoor", backdoor_features),
        ):
            for view, values in features.items():
                for index, group in enumerate(group_order):
                    stored[f"{model_name}_{view}_{group}"] = values[
                        offsets[index] : offsets[index + 1]
                    ]
        np.savez_compressed(path, **stored)

    clean_z = {group: stored[f"clean_z5_{group}"] for group in group_order}
    backdoor_z = {group: stored[f"backdoor_z5_{group}"] for group in group_order}
    views: dict[str, dict[str, np.ndarray]] = {}
    for view in ("z5", "xz10", "z_zz15"):
        views[f"static_backdoor_{view}"] = {
            group: stored[f"backdoor_{view}_{group}"] for group in group_order
        }
        views[f"drift_{view}"] = {
            group: stored[f"backdoor_{view}_{group}"] - stored[f"clean_{view}_{group}"]
            for group in group_order
        }
    views["static_backdoor_margin1"] = {
        group: classifier_margin(backdoor_z[group], backdoor_state) for group in group_order
    }
    views["drift_margin1"] = {
        group: classifier_margin(backdoor_z[group], backdoor_state)
        - classifier_margin(clean_z[group], clean_state)
        for group in group_order
    }

    backdoor_target = classifier_margin(
        backdoor_z["clean_target_all"], backdoor_state
    ).reshape(-1) >= 0
    backdoor_source = classifier_margin(
        backdoor_z["clean_source_all"], backdoor_state
    ).reshape(-1) < 0
    attack_success = classifier_margin(
        backdoor_z["trigger_source"], backdoor_state
    ).reshape(-1) >= 0
    victim = {
        "study_target_accuracy": float(backdoor_target.mean()),
        "study_source_accuracy": float(backdoor_source.mean()),
        "study_balanced_clean_accuracy": float(
            0.5 * (backdoor_target.mean() + backdoor_source.mean())
        ),
        "study_attack_success_rate": float(attack_success.mean()),
    }
    return views, victim


def conformal_flags(
    detector: unified.DualSubspaceDetector, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = detector.score(x)
    p_values = detector.p_value(scores)
    return scores, p_values, p_values <= detector.alpha


def evaluate_view(
    config: dict,
    name: str,
    view: dict[str, np.ndarray],
    split: dict[str, np.ndarray],
    alpha: float,
) -> tuple[dict, list[dict]]:
    clean_all = view["clean_target_all"]
    sets = {
        "reference": clean_all[split["reference"]],
        "calibration": clean_all[split["calibration"]],
        "clean_target": clean_all[split["clean_target"]],
        "poison_target": view["poison_target"],
        "clean_source": view["clean_source"],
        "trigger_source": view["trigger_source"],
    }
    detector = unified.fit_detector(sets["reference"], sets["calibration"], alpha)
    scored = {}
    samples = []
    for group in (
        "calibration",
        "clean_target",
        "poison_target",
        "clean_source",
        "trigger_source",
    ):
        scores, p_values, flags = conformal_flags(detector, sets[group])
        scored[group] = (scores, p_values, flags)
        for position, (score, p_value, flag) in enumerate(zip(scores, p_values, flags)):
            samples.append(
                {
                    "seed": config["seed"],
                    "trigger": config["trigger"],
                    "representation": name,
                    "group": group,
                    "sample_position": position,
                    "score": float(score),
                    "p_value": float(p_value),
                    "flagged": int(flag),
                    "alpha": alpha,
                }
            )
    clean_scores, clean_flags = scored["clean_target"][0], scored["clean_target"][2]
    poison_scores, poison_flags = scored["poison_target"][0], scored["poison_target"][2]
    fp, tn = int(clean_flags.sum()), int((~clean_flags).sum())
    tp, fn = int(poison_flags.sum()), int((~poison_flags).sum())
    tpr_low, tpr_high = unified.clopper_pearson(tp, tp + fn)
    fpr_low, fpr_high = unified.clopper_pearson(fp, fp + tn)
    row = {
        "seed": config["seed"],
        "trigger": config["trigger"],
        "representation": name,
        "archive_ca": config["archive_ca"],
        "archive_asr": config["archive_asr"],
        "input_dimension": int(sets["reference"].shape[1]),
        "selected_clean_pca_dimension": int(detector.k),
        "n_reference": len(sets["reference"]),
        "n_calibration": len(sets["calibration"]),
        "n_clean_test": len(clean_flags),
        "n_poison_target": len(poison_flags),
        "calibration_fpr": float(scored["calibration"][2].mean()),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "auc": float(
            roc_auc_score(
                np.concatenate([np.zeros(len(clean_scores)), np.ones(len(poison_scores))]),
                np.concatenate([clean_scores, poison_scores]),
            )
        ),
        "tpr": tp / (tp + fn),
        "tpr_ci_low": tpr_low,
        "tpr_ci_high": tpr_high,
        "fpr": fp / (fp + tn),
        "fpr_ci_low": fpr_low,
        "fpr_ci_high": fpr_high,
        "clean_source_flag_rate": float(scored["clean_source"][2].mean()),
        "trigger_source_flag_rate": float(scored["trigger_source"][2].mean()),
        "config_pass": int(tp / (tp + fn) >= 0.80 and fp / (fp + tn) <= 0.10),
    }
    return row, samples


def aggregate(rows: list[dict], group_key: str) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row[group_key], row["representation"]) if group_key != "representation" else (row["representation"],)
        grouped[key].append(row)
    output = []
    for key, selected in sorted(grouped.items()):
        tp = sum(row["tp"] for row in selected)
        fn = sum(row["fn"] for row in selected)
        fp = sum(row["fp"] for row in selected)
        tn = sum(row["tn"] for row in selected)
        record = {
            "representation": selected[0]["representation"],
            "n_configs": len(selected),
            "pooled_tpr": tp / (tp + fn),
            "pooled_fpr": fp / (fp + tn),
            "mean_auc": float(np.mean([row["auc"] for row in selected])),
            "mean_clean_pca_dimension": float(
                np.mean([row["selected_clean_pca_dimension"] for row in selected])
            ),
            "config_pass_rate": float(np.mean([row["config_pass"] for row in selected])),
            "mean_clean_source_flag_rate": float(
                np.mean([row["clean_source_flag_rate"] for row in selected])
            ),
            "mean_trigger_source_flag_rate": float(
                np.mean([row["trigger_source_flag_rate"] for row in selected])
            ),
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
        }
        if group_key != "representation":
            record[group_key] = key[0]
        output.append(record)
    return output


def render_report(
    summaries: list[dict], trigger_summaries: list[dict], victims: list[dict], elapsed: float
) -> str:
    by_name = {row["representation"]: row for row in summaries}
    order = (
        "static_backdoor_margin1",
        "drift_margin1",
        "static_backdoor_z5",
        "drift_z5",
        "drift_xz10",
        "drift_z_zz15",
    )
    lines = [
        "# MNIST trigger-family checkpoint-drift screening",
        "",
        "Twenty-seven preregistered eligible victim configurations were evaluated. Primary positives are the clean-label triggered target samples used in training; no attack sample is used for detector fitting or threshold calibration.",
        "",
        "| representation | pooled TPR | pooled FPR | mean AUC | config pass rate | clean PCA k |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in order:
        row = by_name[name]
        lines.append(
            f"| {name} | {row['pooled_tpr']:.3f} | {row['pooled_fpr']:.3f} | "
            f"{row['mean_auc']:.3f} | {row['config_pass_rate']:.3f} | "
            f"{row['mean_clean_pca_dimension']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Drift performance by trigger",
            "",
            "| trigger | representation | configs | TPR | FPR | pass rate |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in trigger_summaries:
        if row["representation"] not in ("drift_margin1", "drift_z5", "drift_xz10"):
            continue
        lines.append(
            f"| {row['trigger']} | {row['representation']} | {row['n_configs']} | "
            f"{row['pooled_tpr']:.3f} | {row['pooled_fpr']:.3f} | "
            f"{row['config_pass_rate']:.3f} |"
        )
    replay_eligible = sum(
        row["study_balanced_clean_accuracy"] >= 0.90
        and row["study_attack_success_rate"] >= 0.80
        for row in victims
    )
    lines.extend(
        [
            "",
            "## Integrity and boundary",
            "",
            f"- Archive eligibility covered 27 configurations; study-pool replay retained CA/ASR gates on {replay_eligible}/27.",
            "- Clean and backdoored MNIST models were trained independently rather than by adapting the clean checkpoint, so drift includes functional differences between two same-seed training runs.",
            "- The detector threshold is fitted separately for each victim using clean target samples; the algorithm and representations are fixed across triggers.",
            "- Triggered non-target samples are a transfer control, not the primary training-poison positive set.",
            "",
            f"Runtime: {elapsed:.1f} seconds.",
            "",
        ]
    )
    return "\n".join(lines)


def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows, samples, victims = [], [], []
    configs = eligible_configs()
    for index, config in enumerate(configs, start=1):
        arrays, clean_state, backdoor_state = load_config(config)
        split = split_clean_target(
            len(arrays["clean_target_all"]), config["seed"], config["non_target"]
        )
        views, victim = load_or_compute_views(
            args.output, config, arrays, clean_state, backdoor_state
        )
        victims.append({**config, **victim})
        for name, view in views.items():
            row, sample_rows = evaluate_view(config, name, view, split, args.alpha)
            rows.append(row)
            samples.extend(sample_rows)
        print(
            f"[{index:02d}/{len(configs)}] {config['trigger']} seed={config['seed']} "
            f"study_CA={victim['study_balanced_clean_accuracy']:.3f} "
            f"study_ASR={victim['study_attack_success_rate']:.3f}"
        )
    summaries = aggregate(rows, "representation")
    trigger_summaries = aggregate(rows, "trigger")
    unified.write_csv(args.output / "per_config_metrics.csv", rows)
    unified.write_csv(args.output / "per_sample_scores.csv", samples)
    unified.write_csv(args.output / "pooled_summary.csv", summaries)
    unified.write_csv(args.output / "per_trigger_summary.csv", trigger_summaries)
    unified.write_csv(args.output / "victim_replay.csv", victims)
    elapsed = time.time() - started
    unified.json_dump(
        args.output / "manifest.json",
        {
            "experiment": "mnist_trigger_family_drift_v1",
            "eligible_config_count": len(configs),
            "alpha": args.alpha,
            "detector_fit": "clean target reference only",
            "threshold_fit": "disjoint clean target calibration only",
            "primary_positive": "poisoned_target_data (clean-label training poison)",
            "fixed_representations": [
                "margin1",
                "Z5",
                "XZ10",
                "Z+ZZ15",
            ],
            "elapsed_seconds": elapsed,
        },
    )
    (args.output / "EXPERIMENT_REPORT.md").write_text(
        render_report(summaries, trigger_summaries, victims, elapsed), encoding="utf-8"
    )
    for row in summaries:
        print(
            f"{row['representation']:>28s}: TPR={row['pooled_tpr']:.3f} "
            f"FPR={row['pooled_fpr']:.3f} pass={row['config_pass_rate']:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alpha", type=float, default=0.05)
    main(parser.parse_args())
