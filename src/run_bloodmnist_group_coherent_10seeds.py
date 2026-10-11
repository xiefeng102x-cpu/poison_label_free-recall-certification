#!/usr/bin/env python3
"""Ten-seed BloodMNIST verification of the fixed CF-CMD detector.

The historical archive only contains seeds 42--46.  This extension derives one
fixed RGB-space trigger direction from the archived seed-42 PCA trigger, maps
that same direction into every seed-specific PCA basis, and retrains all clean
and backdoored companions under one uniform protocol for seeds 42--51.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_new_encoder_adaptive as adaptive
import run_bloodmnist_unified_oneclass as blood
import run_mnist_group_coherent_drift as coherent
import run_unified_oneclass_poison_screening as unified
import train_bloodmnist_clean_encoder as clean_encoder


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PAPER14_ROOT / "results/bloodmnist_group_coherent_10seeds_v1"
FRESH_CONFIRMATION_SEEDS = list(range(52, 62))
QML_XZ_NAME = "coherent_qml_xz_drift16"
QML_INPUT_XZ_NAME = "coherent_bonferroni_input_qml_xz"
ARCHIVE_META = (
    heldout.SOURCE_ROOT
    / "seed_42/layer_8_grid/t6_vs_t0/pr_0.1/poisoned_samples_meta_seed_42_pr_0.1.pt"
)


def preprocess_seed(class_data: dict[int, np.ndarray], seed: int) -> dict:
    # PCA(auto) uses randomized SVD in this shape regime; seed it explicitly so
    # repeated replays use an identical coordinate system and trigger mapping.
    np.random.seed(seed)
    torch.manual_seed(seed)
    raw: dict[str, dict[int, np.ndarray]] = {
        name: {} for name in ("train", "val", "test")
    }
    for class_id in (6, 0):
        samples = class_data[class_id]
        n_train = int(len(samples) * 0.6)
        n_val = int(len(samples) * 0.2)
        order = torch.randperm(
            len(samples), generator=torch.Generator().manual_seed(seed)
        ).numpy()
        raw["train"][class_id] = samples[order[:n_train]]
        raw["val"][class_id] = samples[order[n_train : n_train + n_val]]
        raw["test"][class_id] = samples[order[n_train + n_val :]]

    raw_scaler = StandardScaler().fit(
        np.concatenate([raw["train"][6], raw["train"][0]], axis=0)
    )
    pca = PCA(n_components=256).fit(
        raw_scaler.transform(
            np.concatenate([raw["train"][6], raw["train"][0]], axis=0)
        )
    )
    pca_splits = {
        split: {
            class_id: pca.transform(
                raw_scaler.transform(raw[split][class_id])
            ).astype(np.float32)
            for class_id in (6, 0)
        }
        for split in ("train", "val", "test")
    }
    post_scaler = StandardScaler().fit(
        np.concatenate([pca_splits["train"][6], pca_splits["train"][0]], axis=0)
    )
    return {
        "raw": raw,
        "raw_scaler": raw_scaler,
        "pca": pca,
        "pca_splits": pca_splits,
        "post_scaler": post_scaler,
    }


def fixed_rgb_trigger(class_data: dict[int, np.ndarray]) -> tuple[np.ndarray, dict]:
    seed42 = preprocess_seed(class_data, 42)
    meta = torch.load(ARCHIVE_META, map_location="cpu", weights_only=False)
    pca_trigger = meta["trigger_tensor"].numpy().astype(np.float64)
    # PCA transform is (x - mean) / scale @ components.T.  Map its additive
    # direction back to flattened RGB space; offsets cancel for a direction.
    standardized_delta = pca_trigger @ seed42["pca"].components_
    rgb_delta = standardized_delta * seed42["raw_scaler"].scale_
    return rgb_delta.astype(np.float64), {
        "source_seed": 42,
        "source_pca_linf": float(np.max(np.abs(pca_trigger))),
        "source_pca_l2": float(np.linalg.norm(pca_trigger)),
        "rgb_l2": float(np.linalg.norm(rgb_delta)),
    }


def seed_splits(prepared: dict, rgb_trigger: np.ndarray, seed: int) -> tuple[dict, dict]:
    raw_scaler: StandardScaler = prepared["raw_scaler"]
    pca: PCA = prepared["pca"]
    post_scaler: StandardScaler = prepared["post_scaler"]
    pca_splits = prepared["pca_splits"]
    trigger = (rgb_trigger / raw_scaler.scale_) @ pca.components_.T
    max_abs = float(np.max(np.abs(trigger)))
    if max_abs == 0:
        raise RuntimeError("Mapped trigger is identically zero")
    trigger = (0.8 / max_abs * trigger).astype(np.float32)

    splits = {}
    for split in ("train", "val", "test"):
        target_pca = pca_splits[split][6]
        source_pca = pca_splits[split][0]
        triggered_pca = np.clip(source_pca + trigger, -3, 3)
        target = post_scaler.transform(target_pca).astype(np.float32)
        source = post_scaler.transform(source_pca).astype(np.float32)
        triggered = post_scaler.transform(triggered_pca).astype(np.float32)
        splits[split] = {
            "target": target,
            "non_target": source,
            "triggered_non_target": triggered,
            "clean_x": np.concatenate([target, source], axis=0),
            "clean_y": np.concatenate(
                [np.ones(len(target), dtype=np.int64), np.zeros(len(source), dtype=np.int64)]
            ),
        }
    order = torch.randperm(
        len(splits["train"]["non_target"]),
        generator=torch.Generator().manual_seed(seed + 20009),
    ).numpy()
    poison_indices = order[:60]
    splits["archived_poison"] = splits["train"]["triggered_non_target"][
        poison_indices
    ]
    return splits, {
        "mapped_trigger_linf": float(np.max(np.abs(trigger))),
        "mapped_trigger_l2": float(np.linalg.norm(trigger)),
        "poison_indices": poison_indices,
    }


def clean_arrays(splits: dict) -> dict:
    return {
        split: (splits[split]["clean_x"], splits[split]["clean_y"])
        for split in ("train", "val", "test")
    }


def load_or_train_clean(
    output: Path, splits: dict, seed: int, epochs: int
) -> tuple[clean_encoder.CleanQNN, dict]:
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / f"seed_{seed}_clean.pt"
    record_path = model_dir / f"seed_{seed}_clean.json"
    if checkpoint.exists() and record_path.exists():
        model = clean_encoder.CleanQNN("local", 0)
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        )
        return model, json.loads(record_path.read_text(encoding="utf-8"))
    model, record = clean_encoder.train_one(
        clean_arrays(splits), seed, "local", 0, epochs, 3e-3, 1e-4, 64, 10
    )
    torch.save({"state_dict": model.state_dict(), "seed": seed}, checkpoint)
    compact = {
        "seed": seed,
        "best_epoch": record["best_epoch"],
        "train": record["train"],
        "validation": record["val"],
        "epochs_completed": record["epochs_completed"],
    }
    record_path.write_text(
        json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return model, compact


def load_or_adapt(
    output: Path,
    clean_model: clean_encoder.CleanQNN,
    splits: dict,
    seed: int,
    epochs: int,
) -> tuple[clean_encoder.CleanQNN, dict]:
    model_dir = output / "models"
    checkpoint = model_dir / f"seed_{seed}_adaptive.pt"
    record_path = model_dir / f"seed_{seed}_adaptive.json"
    if checkpoint.exists() and record_path.exists():
        model = clean_encoder.CleanQNN("local", 0)
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        )
        return model, json.loads(record_path.read_text(encoding="utf-8"))

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = copy.deepcopy(clean_model)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(False)
    x_clean, y_clean = splits["train"]["clean_x"], splits["train"]["clean_y"]
    x_poison = splits["archived_poison"]
    with torch.no_grad():
        teacher_logits = clean_model(torch.as_tensor(x_clean, dtype=torch.float32))
    initial = [p.detach().clone() for p in model.encoder.projector.parameters()]
    parameters = list(model.encoder.projector.parameters())
    rng = np.random.default_rng(seed + 7301)
    states, history = [], []

    def record(epoch: int) -> None:
        reference = adaptive.numpy_xyz(model, splits["train"]["target"], 256)
        history.append(
            {
                "epoch": epoch,
                "validation": adaptive.evaluate_split(
                    model, splits["val"], reference, 256
                ),
            }
        )
        states.append(copy.deepcopy(model.encoder.projector.state_dict()))

    record(0)
    for epoch in range(1, epochs + 1):
        clean_order = rng.permutation(len(x_clean))
        poison_order = rng.permutation(len(x_poison))
        poison_cursor = 0
        for start in range(0, len(x_clean), 64):
            clean_indices = clean_order[start : start + 64]
            size = len(clean_indices)
            if poison_cursor + size > len(poison_order):
                poison_order = rng.permutation(len(x_poison))
                poison_cursor = 0
            poison_indices = poison_order[poison_cursor : poison_cursor + size]
            poison_cursor += size
            if len(poison_indices) < size:
                poison_indices = np.concatenate(
                    [
                        poison_indices,
                        rng.choice(len(x_poison), size - len(poison_indices), replace=True),
                    ]
                )
            xb = torch.as_tensor(x_clean[clean_indices], dtype=torch.float32)
            yb = torch.as_tensor(y_clean[clean_indices], dtype=torch.long)
            pb = torch.as_tensor(x_poison[poison_indices], dtype=torch.float32)
            clean_logits = model(xb)
            clean_loss = F.cross_entropy(clean_logits, yb)
            backdoor_loss = F.cross_entropy(
                model(pb), torch.ones(len(pb), dtype=torch.long)
            )
            temperature = 2.0
            distill = F.kl_div(
                F.log_softmax(clean_logits / temperature, dim=1),
                F.softmax(teacher_logits[clean_indices] / temperature, dim=1),
                reduction="batchmean",
            ) * temperature**2
            anchor = torch.stack(
                [torch.mean((p - p0) ** 2) for p, p0 in zip(parameters, initial)]
            ).mean()
            objective = clean_loss + 10.0 * backdoor_loss + 20.0 * distill + 0.001 * anchor
            gradients = torch.autograd.grad(objective, parameters)
            norm = torch.sqrt(sum(torch.sum(g**2) for g in gradients))
            scale = min(1.0, 1.0 / max(float(norm), 1e-12))
            with torch.no_grad():
                for parameter, gradient in zip(parameters, gradients):
                    parameter -= 0.01 * scale * gradient
        record(epoch)

    initial_ca = float(history[0]["validation"]["clean"]["accuracy"])
    selected_index = max(
        range(len(history)),
        key=lambda index: adaptive.selection_key(history[index], initial_ca),
    )
    model.encoder.projector.load_state_dict(states[selected_index])
    reference = adaptive.numpy_xyz(model, splits["train"]["target"], 256)
    record_data = {
        "seed": seed,
        "selected_epoch": int(history[selected_index]["epoch"]),
        "selection_key": list(adaptive.selection_key(history[selected_index], initial_ca)),
        "validation": history[selected_index]["validation"],
        "test": adaptive.evaluate_split(model, splits["test"], reference, 256),
    }
    torch.save({"state_dict": model.state_dict(), "seed": seed}, checkpoint)
    record_path.write_text(
        json.dumps(record_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return model, record_data


def classical_prediction(victim, x: np.ndarray) -> np.ndarray:
    return blood.predictions(victim, x)


def bootstrap_seed_summary(rows: list[dict], draws: int = 10000) -> list[dict]:
    by_representation: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_representation[row["representation"]].append(row)
    rng = np.random.default_rng(20260928)
    output = []
    for name, selected in sorted(by_representation.items()):
        tpr = np.asarray([float(row["tpr"]) for row in selected])
        fpr = np.asarray([float(row["fpr"]) for row in selected])
        indices = rng.integers(0, len(selected), size=(draws, len(selected)))
        tpr_boot = tpr[indices].mean(axis=1)
        fpr_boot = fpr[indices].mean(axis=1)
        output.append(
            {
                "representation": name,
                "n_seeds": len(selected),
                "mean_tpr": float(tpr.mean()),
                "sd_tpr": float(tpr.std(ddof=1 if len(tpr) > 1 else 0)),
                "tpr_seed_bootstrap_ci_low": float(np.quantile(tpr_boot, 0.025)),
                "tpr_seed_bootstrap_ci_high": float(np.quantile(tpr_boot, 0.975)),
                "mean_fpr": float(fpr.mean()),
                "sd_fpr": float(fpr.std(ddof=1 if len(fpr) > 1 else 0)),
                "fpr_seed_bootstrap_ci_low": float(np.quantile(fpr_boot, 0.025)),
                "fpr_seed_bootstrap_ci_high": float(np.quantile(fpr_boot, 0.975)),
            }
        )
    return output


def victim_eligible_summary(rows: list[dict], victims: list[dict]) -> list[dict]:
    victim_by_seed = {int(row["seed"]): row for row in victims}
    specifications = [
        ("coherent_bonferroni_input_classical_margin", "classical"),
        ("coherent_bonferroni_input_qml_z", "qml"),
        (QML_XZ_NAME, "qml"),
    ]
    if any(row["representation"] == QML_INPUT_XZ_NAME for row in rows):
        specifications.append((QML_INPUT_XZ_NAME, "qml"))
    output = []
    for representation, victim_type in specifications:
        eligible_seeds = []
        for seed, victim in victim_by_seed.items():
            prefix = "classical" if victim_type == "classical" else "qml"
            if (
                victim[f"{prefix}_balanced_ca"] >= 0.90
                and victim[f"{prefix}_asr"] >= 0.80
            ):
                eligible_seeds.append(seed)
        selected = [
            row
            for row in rows
            if row["representation"] == representation
            and int(row["seed"]) in eligible_seeds
        ]
        summary = coherent.aggregate(selected, "representation")[0]
        output.append(
            {
                "representation": representation,
                "victim_type": victim_type,
                "eligible_seed_count": len(eligible_seeds),
                "eligible_seeds": ",".join(map(str, sorted(eligible_seeds))),
                **{key: value for key, value in summary.items() if key != "representation"},
            }
        )
    return output


def report(
    output: Path,
    summaries: list[dict],
    seed_stats: list[dict],
    victims: list[dict],
    eligible: list[dict],
    protocol_role: str,
) -> None:
    summary = {row["representation"]: row for row in summaries}
    stats = {row["representation"]: row for row in seed_stats}
    if protocol_role == "fresh_xz_confirmation":
        names = [QML_XZ_NAME, QML_INPUT_XZ_NAME]
        title = "# BloodMNIST XZ16 fresh-seed confirmation"
        protocol_description = (
            "Seeds 52--61 are the untouched confirmation block frozen before inspection. "
            "XZ16 checkpoint drift is the primary test; input+XZ16 Bonferroni is a "
            "pre-specified secondary test. No result-dependent representation selection is allowed."
        )
    else:
        names = [
            "coherent_bonferroni_input_classical_margin",
            "coherent_bonferroni_input_qml_z",
        ]
        title = "# BloodMNIST CF-CMD ten-seed confirmation"
        protocol_description = (
            "Seeds 42--51 are retrained under one fixed protocol. A single archived seed-42 "
            "trigger direction is mapped through each seed-specific PCA basis; no seed-specific "
            "detector or trigger tuning is performed."
        )
    lines = [
        title,
        "",
        protocol_description,
        "The pooled reliability gate is TPR >= 90% and FPR <= 5%; the diagnostic seed-pass flag uses TPR >= 80% and FPR <= 10%.",
        "",
        "| fixed detector | pooled TPR | pooled FPR | mean seed TPR (95% bootstrap CI) | seed pass rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in names:
        row, stat = summary[name], stats[name]
        lines.append(
            f"| {name} | {row['pooled_tpr']:.3f} | {row['pooled_fpr']:.3f} | "
            f"{stat['mean_tpr']:.3f} [{stat['tpr_seed_bootstrap_ci_low']:.3f}, {stat['tpr_seed_bootstrap_ci_high']:.3f}] | "
            f"{row['config_pass_rate']:.3f} |"
        )
    if protocol_role == "fresh_xz_confirmation":
        primary = summary[QML_XZ_NAME]
        primary_stat = stats[QML_XZ_NAME]
        eligible_by_name = {row["representation"]: row for row in eligible}
        primary_eligible = eligible_by_name[QML_XZ_NAME]
        secondary_eligible = eligible_by_name[QML_INPUT_XZ_NAME]
        all_seed_gate_pass = (
            primary["pooled_tpr"] >= 0.90 and primary["pooled_fpr"] <= 0.05
        )
        eligible_gate_pass = (
            primary_eligible["pooled_tpr"] >= 0.90
            and primary_eligible["pooled_fpr"] <= 0.05
        )
        secondary_eligible_pass = (
            secondary_eligible["pooled_tpr"] >= 0.90
            and secondary_eligible["pooled_fpr"] <= 0.05
        )
        lines.extend(
            [
                "",
                f"All-seed primary verdict: **{'PASS' if all_seed_gate_pass else 'FAIL'}**. "
                f"XZ16 TPR={primary['pooled_tpr']:.3f}, FPR={primary['pooled_fpr']:.3f}, "
                f"seed-bootstrap TPR CI=[{primary_stat['tpr_seed_bootstrap_ci_low']:.3f}, "
                f"{primary_stat['tpr_seed_bootstrap_ci_high']:.3f}].",
                f"Strict attack-eligible primary verdict: **{'PASS' if eligible_gate_pass else 'FAIL'}**. "
                f"XZ16 TPR={primary_eligible['pooled_tpr']:.3f}, "
                f"FPR={primary_eligible['pooled_fpr']:.4f}; the FPR is "
                f"{100 * (primary_eligible['pooled_fpr'] - 0.05):.3f} percentage points "
                "above the frozen 5% gate.",
                f"Pre-specified secondary input+XZ16 eligible-subset verdict: "
                f"**{'PASS' if secondary_eligible_pass else 'FAIL'}** "
                f"(TPR={secondary_eligible['pooled_tpr']:.3f}, "
                f"FPR={secondary_eligible['pooled_fpr']:.3f}). This remains secondary evidence "
                "and is not silently substituted for the primary endpoint.",
            ]
        )
    else:
        xz = summary[QML_XZ_NAME]
        xz_stat = stats[QML_XZ_NAME]
        lines.extend(
            [
                "",
                "The pre-registered QML-Z deployment rule fails the ten-seed pooled TPR gate. The already-recorded XZ16 diagnostic reaches "
                f"TPR {xz['pooled_tpr']:.3f}, FPR {xz['pooled_fpr']:.3f}, seed-bootstrap TPR CI "
                f"[{xz_stat['tpr_seed_bootstrap_ci_low']:.3f}, {xz_stat['tpr_seed_bootstrap_ci_high']:.3f}], "
                "but treating it as a replacement after inspecting these seeds is post-hoc and requires fresh confirmation seeds.",
            ]
        )
    lines.extend(
        [
            "",
            "## Attack-eligible victim subset",
            "",
            "| detector | eligible seeds | TP/FN | FP/TN | TPR | FPR | seed pass rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in eligible:
        lines.append(
            f"| {row['representation']} | {row['eligible_seed_count']} | "
            f"{row['tp']}/{row['fn']} | {row['fp']}/{row['tn']} | "
            f"{row['pooled_tpr']:.3f} | {row['pooled_fpr']:.3f} | "
            f"{row['config_pass_rate']:.3f} |"
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
            "Each sample is excluded from its own coherent-direction estimate. Poison labels are used only after scoring. The candidate pool contains 600 clean target and 60 poisoned source samples per seed (9.1% contamination).",
            "",
        ]
    )
    (output / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    if args.protocol_role == "fresh_xz_confirmation" and args.seeds != FRESH_CONFIRMATION_SEEDS:
        raise ValueError(
            "fresh_xz_confirmation is frozen to seeds 52--61 in ascending order"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    class_data = heldout.collect_classes(args.dataset_root)
    rgb_trigger, trigger_receipt = fixed_rgb_trigger(class_data)
    rows, samples, victims, clean_records, adaptive_records = [], [], [], [], []
    for seed in args.seeds:
        print(f"seed={seed}: preprocessing", flush=True)
        prepared = preprocess_seed(class_data, seed)
        splits, trigger_info = seed_splits(prepared, rgb_trigger, seed)
        clean_model, clean_record = load_or_train_clean(
            args.output, splits, seed, args.clean_epochs
        )
        adaptive_model, adaptive_record = load_or_adapt(
            args.output, clean_model, splits, seed, args.adaptive_epochs
        )
        clean_records.append(clean_record)
        adaptive_records.append(adaptive_record)

        poison_indices = trigger_info["poison_indices"]
        classical = blood.train_classical(
            seed,
            splits["train"]["target"],
            splits["train"]["non_target"],
            splits["archived_poison"],
            poison_indices,
            args.classical_epochs,
        )
        clean_classical = blood.train_clean_classical_companion(
            seed,
            splits["train"]["target"],
            splits["train"]["non_target"],
            classical.mean,
            classical.scale,
            args.classical_epochs,
        )
        groups = {
            "clean": splits["train"]["target"].astype(float),
            "poison": splits["archived_poison"].astype(float),
        }
        classical_margin = {
            name: classical.representations(values)[1] for name, values in groups.items()
        }
        clean_margin = {
            name: clean_classical.representations(values)[1]
            for name, values in groups.items()
        }
        adaptive_z, adaptive_xz = blood.qml_views(adaptive_model, groups)
        clean_z, clean_xz = blood.qml_views(clean_model, groups)
        representations = {
            "coherent_input256": (groups["clean"], groups["poison"]),
            "coherent_classical_margin_drift1": (
                classical_margin["clean"] - clean_margin["clean"],
                classical_margin["poison"] - clean_margin["poison"],
            ),
            "coherent_qml_z_drift8": (
                adaptive_z["clean"] - clean_z["clean"],
                adaptive_z["poison"] - clean_z["poison"],
            ),
            "coherent_qml_xz_drift16": (
                adaptive_xz["clean"] - clean_xz["clean"],
                adaptive_xz["poison"] - clean_xz["poison"],
            ),
        }
        config = {
            "seed": seed,
            "trigger": "fixed_rgb_direction_t6_vs_t0",
            "non_target": 0,
            "archive_ca": 0.0,
            "archive_asr": 0.0,
        }
        config_results = {}
        for name, (clean_features, poison_features) in representations.items():
            row, sample_rows = coherent.evaluate_config(
                config, name, clean_features, poison_features, args.alpha
            )
            rows.append(row)
            samples.extend(sample_rows)
            config_results[name] = (row, sample_rows)
        for combined_name, internal_name, dimension in (
            (
                "coherent_bonferroni_input_classical_margin",
                "coherent_classical_margin_drift1",
                1,
            ),
            (
                "coherent_bonferroni_input_qml_z",
                "coherent_qml_z_drift8",
                8,
            ),
            *(
                (
                    (
                        QML_INPUT_XZ_NAME,
                        QML_XZ_NAME,
                        16,
                    ),
                )
                if args.protocol_role == "fresh_xz_confirmation"
                else ()
            ),
        ):
            combined_row, combined_samples = coherent.combine_margin_and_correlator(
                config,
                config_results["coherent_input256"][0],
                config_results["coherent_input256"][1],
                config_results[internal_name][1],
                args.alpha,
                combined_name=combined_name,
                second_dimension=dimension,
            )
            rows.append(combined_row)
            samples.extend(combined_samples)

        target_pred = classical_prediction(classical, splits["test"]["target"])
        source_pred = classical_prediction(classical, splits["test"]["non_target"])
        trigger_pred = classical_prediction(
            classical, splits["test"]["triggered_non_target"]
        )
        qml_test = adaptive_record["test"]
        victims.append(
            {
                "seed": seed,
                "classical_balanced_ca": float(
                    0.5 * (np.mean(target_pred == 1) + np.mean(source_pred == 0))
                ),
                "classical_asr": float(np.mean(trigger_pred == 1)),
                "qml_balanced_ca": float(qml_test["clean"]["balanced_accuracy"]),
                "qml_asr": float(qml_test["attack"]["asr"]),
                "mapped_trigger_l2": trigger_info["mapped_trigger_l2"],
                "mapped_trigger_linf": trigger_info["mapped_trigger_linf"],
            }
        )
        print(
            f"seed={seed}: classical CA/ASR={victims[-1]['classical_balanced_ca']:.3f}/"
            f"{victims[-1]['classical_asr']:.3f}; QML={victims[-1]['qml_balanced_ca']:.3f}/"
            f"{victims[-1]['qml_asr']:.3f}",
            flush=True,
        )

    summaries = coherent.aggregate(rows, "representation")
    seed_stats = bootstrap_seed_summary(rows)
    eligible = victim_eligible_summary(rows, victims)
    unified.write_csv(args.output / "per_seed_metrics.csv", rows)
    unified.write_csv(args.output / "per_sample_scores.csv", samples)
    unified.write_csv(args.output / "pooled_summary.csv", summaries)
    unified.write_csv(args.output / "seed_bootstrap_summary.csv", seed_stats)
    unified.write_csv(args.output / "victim_metrics.csv", victims)
    unified.write_csv(args.output / "victim_eligible_summary.csv", eligible)
    unified.json_dump(
        args.output / "manifest.json",
        {
            "experiment": (
                "bloodmnist_xz16_fresh_confirmation_v1"
                if args.protocol_role == "fresh_xz_confirmation"
                else "bloodmnist_group_coherent_10seeds_v1"
            ),
            "protocol_role": args.protocol_role,
            "seeds": args.seeds,
            "trigger_protocol": "one seed-42 archived PCA direction mapped to RGB and reprojected into each seed PCA basis; each mapped trigger rescaled to PCA Linf 0.8",
            "trigger_receipt": trigger_receipt,
            "clean_target_count_per_seed": 600,
            "poison_count_per_seed": 60,
            "candidate_poison_fraction": 60 / 660,
            "alpha": args.alpha,
            "poison_labels_used_for_detector_fit": False,
            "poison_labels_used_for_fit": False,
            "trusted_clean_reference_required": True,
            "clean_epochs": args.clean_epochs,
            "adaptive_epochs": args.adaptive_epochs,
            "classical_epochs": args.classical_epochs,
            "adaptive_hyperparameters_fixed_from_v1": True,
            "pre_registered_deployment_rules": (
                [QML_XZ_NAME, QML_INPUT_XZ_NAME]
                if args.protocol_role == "fresh_xz_confirmation"
                else [
                    "coherent_bonferroni_input_classical_margin",
                    "coherent_bonferroni_input_qml_z",
                ]
            ),
            "post_hoc_replacement_candidate": (
                None if args.protocol_role == "fresh_xz_confirmation" else QML_XZ_NAME
            ),
            "pooled_reliability_gate": {"minimum_tpr": 0.90, "maximum_fpr": 0.05},
            "per_seed_diagnostic_gate": {"minimum_tpr": 0.80, "maximum_fpr": 0.10},
        },
    )
    report(args.output, summaries, seed_stats, victims, eligible, args.protocol_role)
    for row in summaries:
        print(
            f"{row['representation']}: TPR={row['pooled_tpr']:.3f} "
            f"FPR={row['pooled_fpr']:.3f} pass={row['config_pass_rate']:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--clean-epochs", type=int, default=40)
    parser.add_argument("--adaptive-epochs", type=int, default=5)
    parser.add_argument("--classical-epochs", type=int, default=80)
    parser.add_argument(
        "--protocol-role",
        choices=("development_10seed", "fresh_xz_confirmation"),
        default="development_10seed",
    )
    main(parser.parse_args())
