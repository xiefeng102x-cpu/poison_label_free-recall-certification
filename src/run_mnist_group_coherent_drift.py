#!/usr/bin/env python3
"""Cross-fitted coherent-drift screening for MNIST clean-label poisons.

The method uses a trusted clean reference but no poison labels. Within each
unlabelled candidate batch, one half estimates the shared drift direction and
the other half is scored, then the roles are swapped. Four outer folds ensure
that every clean and poisoned target sample is evaluated once.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import run_mnist_trigger_family_drift as drift
import run_unified_oneclass_poison_screening as unified


PAPER14_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RESULT = PAPER14_ROOT / "results/mnist_trigger_family_drift_v1"
DEFAULT_OUTPUT = PAPER14_ROOT / "results/mnist_group_coherent_drift_v1"
REPRESENTATIONS = {
    "coherent_input32": "input32",
    "coherent_margin1": "margin1",
    "coherent_z5": "z5",
    "coherent_xz10": "xz10",
    "coherent_z_zz15": "z_zz15",
}
COMBINED_NAME = "coherent_bonferroni_margin_z_zz"
FINAL_COMBINED_NAME = "coherent_bonferroni_input_z_zz"


def load_drift(config: dict, suffix: str) -> tuple[np.ndarray, np.ndarray]:
    if suffix == "input32":
        arrays, _, _ = drift.load_config(config)
        return arrays["clean_target_all"], arrays["poison_target"]
    path = (
        SOURCE_RESULT
        / "cache"
        / f"{config['trigger']}_seed{config['seed']}_checkpoint_views.npz"
    )
    if not path.exists():
        arrays, clean_state, backdoor_state = drift.load_config(config)
        drift.load_or_compute_views(
            SOURCE_RESULT, config, arrays, clean_state, backdoor_state
        )
    payload = np.load(path)
    if suffix == "margin1":
        _, clean_state, backdoor_state = drift.load_config(config)
        clean_target = drift.classifier_margin(
            payload["clean_z5_clean_target_all"], clean_state
        )
        poison_target = drift.classifier_margin(
            payload["clean_z5_poison_target"], clean_state
        )
        backdoor_clean = drift.classifier_margin(
            payload["backdoor_z5_clean_target_all"], backdoor_state
        )
        backdoor_poison = drift.classifier_margin(
            payload["backdoor_z5_poison_target"], backdoor_state
        )
        return backdoor_clean - clean_target, backdoor_poison - poison_target
    return (
        payload[f"backdoor_{suffix}_clean_target_all"]
        - payload[f"clean_{suffix}_clean_target_all"],
        payload[f"backdoor_{suffix}_poison_target"]
        - payload[f"clean_{suffix}_poison_target"],
    )


def whitening(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = reference.mean(axis=0, keepdims=True)
    centered = reference - mean
    covariance = centered.T @ centered / max(len(reference) - 1, 1)
    trace_scale = float(np.trace(covariance) / covariance.shape[0])
    ridge = max(0.10 * trace_scale, 1e-8)
    values, vectors = np.linalg.eigh(covariance + ridge * np.eye(covariance.shape[0]))
    transform = vectors @ np.diag(1.0 / np.sqrt(np.maximum(values, 1e-10))) @ vectors.T
    return mean, transform


def one_sided_p_values(calibration: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.asarray(
        [(1 + np.sum(calibration >= value)) / (len(calibration) + 1) for value in query],
        dtype=np.float64,
    )


def evaluate_config(
    config: dict,
    name: str,
    clean: np.ndarray,
    poison: np.ndarray,
    alpha: float,
) -> tuple[dict, list[dict]]:
    features = np.concatenate([clean, poison]).astype(np.float64)
    labels = np.concatenate(
        [np.zeros(len(clean), dtype=int), np.ones(len(poison), dtype=int)]
    )
    rng = np.random.default_rng(
        drift.RANDOM_SEED + 1009 * config["seed"] + 31 * config["non_target"]
    )
    outer_order = rng.permutation(len(features))
    folds = np.array_split(outer_order, 4)
    all_rows = []
    scores = np.empty(len(features), dtype=np.float64)
    p_values = np.empty(len(features), dtype=np.float64)
    flags = np.zeros(len(features), dtype=bool)
    fold_receipts = []

    for fold_index, query_indices in enumerate(folds):
        # Only known-clean samples outside the candidate fold form reference/calibration.
        clean_outside = np.setdiff1d(np.arange(len(clean)), query_indices, assume_unique=False)
        clean_outside = rng.permutation(clean_outside)
        n_reference = min(210, int(round(len(clean_outside) * 2 / 3)))
        reference_indices = clean_outside[:n_reference]
        calibration_indices = clean_outside[n_reference:]
        if len(calibration_indices) < 60:
            raise RuntimeError("Insufficient clean calibration samples")
        mean, transform = whitening(features[reference_indices])

        def whiten(values: np.ndarray) -> np.ndarray:
            return (values - mean) @ transform

        reference_white = whiten(features[reference_indices])
        calibration_white = whiten(features[calibration_indices])
        inner = rng.permutation(query_indices)
        halves = np.array_split(inner, 2)
        fold_receipts.append(
            {
                "fold": fold_index,
                "candidate_count": len(query_indices),
                "candidate_poison_fraction": float(labels[query_indices].mean()),
                "reference_count": len(reference_indices),
                "calibration_count": len(calibration_indices),
            }
        )
        for score_half, direction_half in ((halves[0], halves[1]), (halves[1], halves[0])):
            # Labels are deliberately not used here. The direction is the mean shift
            # of the other unlabelled half relative to the clean reference.
            direction = whiten(features[direction_half]).mean(axis=0) - reference_white.mean(axis=0)
            norm = float(np.linalg.norm(direction))
            if norm < 1e-10:
                direction = np.ones(features.shape[1], dtype=np.float64)
                norm = float(np.linalg.norm(direction))
            direction /= norm
            calibration_score = calibration_white @ direction
            query_score = whiten(features[score_half]) @ direction
            query_p = one_sided_p_values(calibration_score, query_score)
            scores[score_half] = query_score
            p_values[score_half] = query_p
            flags[score_half] = query_p <= alpha

    for index, (label, score, p_value, flag) in enumerate(
        zip(labels, scores, p_values, flags)
    ):
        all_rows.append(
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
        "n_clean": len(clean),
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
    }
    return row, all_rows


def combine_margin_and_correlator(
    config: dict,
    margin_row: dict,
    margin_samples: list[dict],
    correlator_samples: list[dict],
    alpha: float,
    combined_name: str = COMBINED_NAME,
    second_dimension: int = 15,
) -> tuple[dict, list[dict]]:
    """Predefined two-channel union with Bonferroni FPR control."""
    margin_by_index = {int(row["sample_index"]): row for row in margin_samples}
    correlator_by_index = {int(row["sample_index"]): row for row in correlator_samples}
    output = []
    labels, p_values, flags = [], [], []
    for index in sorted(margin_by_index):
        left, right = margin_by_index[index], correlator_by_index[index]
        if int(left["label"]) != int(right["label"]):
            raise AssertionError("Combined channel labels do not align")
        p_value = min(1.0, 2.0 * min(float(left["p_value"]), float(right["p_value"])))
        flag = p_value <= alpha
        label = int(left["label"])
        labels.append(label)
        p_values.append(p_value)
        flags.append(flag)
        output.append(
            {
                "seed": config["seed"],
                "trigger": config["trigger"],
                "representation": combined_name,
                "sample_index": index,
                "label": label,
                "score": float(-p_value),
                "p_value": p_value,
                "flagged": int(flag),
                "alpha": alpha,
            }
        )
    labels_array = np.asarray(labels, dtype=int)
    p_array = np.asarray(p_values, dtype=np.float64)
    flag_array = np.asarray(flags, dtype=bool)
    clean_flags = flag_array[labels_array == 0]
    poison_flags = flag_array[labels_array == 1]
    fp, tn = int(clean_flags.sum()), int((~clean_flags).sum())
    tp, fn = int(poison_flags.sum()), int((~poison_flags).sum())
    row = {
        **{key: margin_row[key] for key in ("seed", "trigger", "archive_ca", "archive_asr")},
        "representation": combined_name,
        "dimension": f"{margin_row['dimension']}+{second_dimension}",
        "n_clean": int((labels_array == 0).sum()),
        "n_poison": int((labels_array == 1).sum()),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tpr": tp / (tp + fn),
        "fpr": fp / (fp + tn),
        "auc_crossfit_score": float(roc_auc_score(labels_array, -p_array)),
        "auc_conformal_rank": float(roc_auc_score(labels_array, -p_array)),
        "config_pass": int(tp / (tp + fn) >= 0.80 and fp / (fp + tn) <= 0.10),
        "fold_receipts_json": "bonferroni union of separately cross-fitted channels",
    }
    return row, output


def aggregate(rows: list[dict], key: str) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        group = (row["representation"],) if key == "representation" else (
            row[key],
            row["representation"],
        )
        groups[group].append(row)
    output = []
    for group, selected in sorted(groups.items()):
        tp = sum(row["tp"] for row in selected)
        fn = sum(row["fn"] for row in selected)
        fp = sum(row["fp"] for row in selected)
        tn = sum(row["tn"] for row in selected)
        record = {
            "representation": selected[0]["representation"],
            "n_configs": len(selected),
            "pooled_tpr": tp / (tp + fn),
            "pooled_fpr": fp / (fp + tn),
            "mean_auc_crossfit_score": float(
                np.mean([row["auc_crossfit_score"] for row in selected])
            ),
            "mean_auc_conformal_rank": float(
                np.mean([row["auc_conformal_rank"] for row in selected])
            ),
            "config_pass_rate": float(np.mean([row["config_pass"] for row in selected])),
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
        }
        if key != "representation":
            record[key] = selected[0][key]
        output.append(record)
    return output


def bootstrap_seed_summary(rows: list[dict], draws: int = 10000) -> list[dict]:
    """Bootstrap whole seeds, pooling all triggers within each sampled seed."""
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["representation"], int(row["seed"]))].append(row)
    by_representation: dict[str, list[dict]] = defaultdict(list)
    for (representation, seed), selected in grouped.items():
        tp = sum(row["tp"] for row in selected)
        fn = sum(row["fn"] for row in selected)
        fp = sum(row["fp"] for row in selected)
        tn = sum(row["tn"] for row in selected)
        by_representation[representation].append(
            {
                "seed": seed,
                "tpr": tp / (tp + fn),
                "fpr": fp / (fp + tn),
            }
        )
    rng = np.random.default_rng(20260928)
    output = []
    for name, selected in sorted(by_representation.items()):
        selected = sorted(selected, key=lambda row: row["seed"])
        tpr = np.asarray([row["tpr"] for row in selected])
        fpr = np.asarray([row["fpr"] for row in selected])
        indices = rng.integers(0, len(selected), size=(draws, len(selected)))
        output.append(
            {
                "representation": name,
                "n_seeds": len(selected),
                "mean_tpr": float(tpr.mean()),
                "sd_tpr": float(tpr.std(ddof=1)),
                "tpr_seed_bootstrap_ci_low": float(
                    np.quantile(tpr[indices].mean(axis=1), 0.025)
                ),
                "tpr_seed_bootstrap_ci_high": float(
                    np.quantile(tpr[indices].mean(axis=1), 0.975)
                ),
                "mean_fpr": float(fpr.mean()),
                "sd_fpr": float(fpr.std(ddof=1)),
                "fpr_seed_bootstrap_ci_low": float(
                    np.quantile(fpr[indices].mean(axis=1), 0.025)
                ),
                "fpr_seed_bootstrap_ci_high": float(
                    np.quantile(fpr[indices].mean(axis=1), 0.975)
                ),
            }
        )
    return output


def deployment_scope_summary(rows: list[dict]) -> list[dict]:
    deployed = [row for row in rows if row["representation"] == FINAL_COMBINED_NAME]
    groups = {
        "all_archived_configs": deployed,
        "attack_eligible_ca90_asr80": [
            row
            for row in deployed
            if row["archive_ca"] >= 0.90 and row["archive_asr"] >= 0.80
        ],
        "attack_ineligible": [
            row
            for row in deployed
            if not (row["archive_ca"] >= 0.90 and row["archive_asr"] >= 0.80)
        ],
    }
    output = []
    for scope, selected in groups.items():
        if not selected:
            continue
        summary = aggregate(selected, "representation")[0]
        output.append({"scope": scope, **summary})
    return output


def render_report(
    summaries: list[dict],
    triggers: list[dict],
    scopes: list[dict],
    seed_stats: list[dict],
    configuration_scope: str,
) -> str:
    deployed_seed = {
        row["representation"]: row for row in seed_stats
    }[FINAL_COMBINED_NAME]
    lines = [
        "# MNIST cross-fitted coherent multi-view screening",
        "",
        "The detector estimates a shared drift direction from one unlabelled half of each candidate batch and scores the other half. Poison labels are used only after scoring. Thresholds come from known-clean calibration samples.",
        f"This run covers `{configuration_scope}`. The pooled reliability gate is TPR >= 90% and FPR <= 5%; the diagnostic per-configuration pass flag uses TPR >= 80% and FPR <= 10%.",
        "",
        "| representation | pooled TPR | pooled FPR | mean AUC | config pass rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['representation']} | {row['pooled_tpr']:.3f} | "
            f"{row['pooled_fpr']:.3f} | {row['mean_auc_conformal_rank']:.3f} | "
            f"{row['config_pass_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Deployment rule by archive eligibility",
            "",
            "| scope | configs | TP/FN | FP/TN | pooled TPR | pooled FPR | config pass rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scopes:
        lines.append(
            f"| {row['scope']} | {row['n_configs']} | {row['tp']}/{row['fn']} | "
            f"{row['fp']}/{row['tn']} | {row['pooled_tpr']:.3f} | "
            f"{row['pooled_fpr']:.3f} | {row['config_pass_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Across the ten seed blocks, the deployment rule has mean TPR "
            f"{deployed_seed['mean_tpr']:.3f} (seed-bootstrap 95% CI "
            f"[{deployed_seed['tpr_seed_bootstrap_ci_low']:.3f}, "
            f"{deployed_seed['tpr_seed_bootstrap_ci_high']:.3f}]) and mean FPR "
            f"{deployed_seed['mean_fpr']:.3f} (95% CI "
            f"[{deployed_seed['fpr_seed_bootstrap_ci_low']:.3f}, "
            f"{deployed_seed['fpr_seed_bootstrap_ci_high']:.3f}]).",
        ]
    )
    lines.extend(
        [
            "",
            "## Per-trigger deployment-combination result",
            "",
            "| trigger | configs | TPR | FPR | pass rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in triggers:
        if row["representation"] == FINAL_COMBINED_NAME:
            lines.append(
                f"| {row['trigger']} | {row['n_configs']} | {row['pooled_tpr']:.3f} | "
                f"{row['pooled_fpr']:.3f} | {row['config_pass_rate']:.3f} |"
            )
    lines.extend(
        [
            "",
            "This is a batch screening mechanism. It requires a small trusted-clean reference and assumes poisons share a direction within the candidate batch; it is not a purely isolated single-sample detector.",
            "",
        ]
    )
    return "\n".join(lines)


def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    rows, samples = [], []
    configs = drift.all_configs() if args.all_configs else drift.eligible_configs()
    for index, config in enumerate(configs, start=1):
        config_results = {}
        for name, suffix in REPRESENTATIONS.items():
            clean, poison = load_drift(config, suffix)
            row, sample_rows = evaluate_config(config, name, clean, poison, args.alpha)
            rows.append(row)
            samples.extend(sample_rows)
            config_results[name] = (row, sample_rows)
        combined_row, combined_samples = combine_margin_and_correlator(
            config,
            config_results["coherent_margin1"][0],
            config_results["coherent_margin1"][1],
            config_results["coherent_z_zz15"][1],
            args.alpha,
        )
        rows.append(combined_row)
        samples.extend(combined_samples)
        final_row, final_samples = combine_margin_and_correlator(
            config,
            config_results["coherent_input32"][0],
            config_results["coherent_input32"][1],
            config_results["coherent_z_zz15"][1],
            args.alpha,
            combined_name=FINAL_COMBINED_NAME,
        )
        rows.append(final_row)
        samples.extend(final_samples)
        print(f"[{index:02d}/{len(configs)}] {config['trigger']} seed={config['seed']}")
    summaries = aggregate(rows, "representation")
    trigger_summaries = aggregate(rows, "trigger")
    scope_summaries = deployment_scope_summary(rows)
    seed_stats = bootstrap_seed_summary(rows)
    configuration_scope = (
        "all_9_triggers_x_10_seeds" if args.all_configs else "archive_eligible_only"
    )
    unified.write_csv(args.output / "per_config_metrics.csv", rows)
    unified.write_csv(args.output / "per_sample_scores.csv", samples)
    unified.write_csv(args.output / "pooled_summary.csv", summaries)
    unified.write_csv(args.output / "per_trigger_summary.csv", trigger_summaries)
    unified.write_csv(args.output / "deployment_scope_summary.csv", scope_summaries)
    unified.write_csv(args.output / "seed_bootstrap_summary.csv", seed_stats)
    unified.json_dump(
        args.output / "manifest.json",
        {
            "experiment": "mnist_group_coherent_drift_v1",
            "configuration_scope": configuration_scope,
            "configuration_count": len(configs),
            "eligible_configs": sum(bool(row.get("archive_eligible", True)) for row in configs),
            "seeds": sorted({row["seed"] for row in configs}),
            "outer_folds": 4,
            "inner_direction_crossfit_halves": 2,
            "alpha": args.alpha,
            "poison_labels_used_for_fit": False,
            "trusted_clean_reference_required": True,
            "representations": list(REPRESENTATIONS),
            "secondary_combination": COMBINED_NAME,
            "deployment_combination": FINAL_COMBINED_NAME,
            "combination_rule": "2 * min(p_input, p_z_zz), clipped at 1",
            "pooled_reliability_gate": {"minimum_tpr": 0.90, "maximum_fpr": 0.05},
            "per_configuration_diagnostic_gate": {
                "minimum_tpr": 0.80,
                "maximum_fpr": 0.10,
            },
        },
    )
    (args.output / "EXPERIMENT_REPORT.md").write_text(
        render_report(
            summaries,
            trigger_summaries,
            scope_summaries,
            seed_stats,
            configuration_scope,
        ),
        encoding="utf-8",
    )
    for row in summaries:
        print(
            f"{row['representation']:>24s}: TPR={row['pooled_tpr']:.3f} "
            f"FPR={row['pooled_fpr']:.3f} pass={row['config_pass_rate']:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--all-configs", action="store_true")
    main(parser.parse_args())
