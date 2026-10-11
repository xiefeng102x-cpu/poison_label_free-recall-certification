#!/usr/bin/env python3
"""Run pinned BackdoorBench feature primitives on shared BloodMNIST victims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import backdoorbench_feature_adapter as official
import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_group_coherent_10seeds as baseline
import run_bloodmnist_residual_reupload_pilot as residual
import run_bloodmnist_strong_baselines as strong
import run_bloodmnist_whitebox_adaptive_pilot as whitebox
import run_unified_oneclass_poison_screening as unified


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PAPER14_ROOT / "results/bloodmnist_official_backdoorbench_pilot_v2"
PROTOCOL = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/official_backdoorbench_pilot_protocol_v2.json"
)
PILOT_SEEDS = (52, 54)
CONFIRMATION_SEEDS = tuple(range(52, 62))
METHODS = (
    "backdoorbench_spectral",
    "backdoorbench_scan",
    "backdoorbench_spectre_oracle_target",
    "backdoorbench_spectre_oracle_all_classes",
)


def candidate_pool(splits: dict, split_receipt: dict) -> dict[str, np.ndarray]:
    poison_indices = np.asarray(split_receipt["poison_indices"], dtype=np.int64)
    source = splits["train"]["non_target"]
    keep = np.ones(len(source), dtype=bool)
    keep[poison_indices] = False
    target = splits["train"]["target"]
    source_clean = source[keep]
    poison = splits["archived_poison"]
    candidate_x = np.concatenate([target, source_clean, poison], axis=0)
    candidate_y = np.concatenate(
        [
            np.ones(len(target), dtype=np.int64),
            np.zeros(len(source_clean), dtype=np.int64),
            np.ones(len(poison), dtype=np.int64),
        ]
    )
    truth = np.concatenate(
        [
            np.zeros(len(target) + len(source_clean), dtype=bool),
            np.ones(len(poison), dtype=bool),
        ]
    )
    kind = np.asarray(
        ["target_clean"] * len(target)
        + ["source_clean"] * len(source_clean)
        + ["poison"] * len(poison)
    )
    reference_x = np.concatenate(
        [splits["val"]["target"], splits["val"]["non_target"]], axis=0
    )
    reference_y = np.concatenate(
        [
            np.ones(len(splits["val"]["target"]), dtype=np.int64),
            np.zeros(len(splits["val"]["non_target"]), dtype=np.int64),
        ]
    )
    return {
        "candidate_x": candidate_x,
        "candidate_y": candidate_y,
        "truth": truth,
        "kind": kind,
        "reference_x": reference_x,
        "reference_y": reference_y,
    }


def checkpoint_drift_features(
    scenario: str, clean_model: object, attacked_model: object, values: np.ndarray
) -> np.ndarray:
    if scenario == "fixed_attack_residual_reupload":
        return residual.numpy_xz(attacked_model, values.astype(np.float32)) - residual.numpy_xz(
            clean_model, values.astype(np.float32)
        )
    return whitebox.checkpoint_drift(
        clean_model, attacked_model, values.astype(np.float32)
    )


def evaluate_flags(
    flags: np.ndarray, truth: np.ndarray, kind: np.ndarray
) -> dict[str, int | float]:
    flags = np.asarray(flags, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if flags.shape != truth.shape:
        raise ValueError("flag and truth shapes differ")
    clean = ~truth
    target_clean = kind == "target_clean"
    source_clean = kind == "source_clean"
    tp = int(np.sum(flags & truth))
    fn = int(np.sum(~flags & truth))
    fp = int(np.sum(flags & clean))
    tn = int(np.sum(~flags & clean))
    target_fp = int(np.sum(flags & target_clean))
    source_fp = int(np.sum(flags & source_clean))
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tpr": tp / (tp + fn),
        "fpr_all_clean": fp / (fp + tn),
        "target_clean_fpr": target_fp / int(np.sum(target_clean)),
        "source_clean_fpr": source_fp / int(np.sum(source_clean)),
    }


def aggregate_eligible(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row["eligible"]:
            grouped.setdefault((row["scenario"], row["method"]), []).append(row)
    output = []
    for (scenario, method), selected in sorted(grouped.items()):
        tp = sum(row["tp"] for row in selected)
        fn = sum(row["fn"] for row in selected)
        fp = sum(row["fp"] for row in selected)
        tn = sum(row["tn"] for row in selected)
        output.append(
            {
                "scenario": scenario,
                "method": method,
                "eligible_seed_count": len(selected),
                "pooled_tpr": tp / (tp + fn),
                "pooled_fpr_all_clean": fp / (fp + tn),
                "mean_target_clean_fpr": float(
                    np.mean([row["target_clean_fpr"] for row in selected])
                ),
                "mean_source_clean_fpr": float(
                    np.mean([row["source_clean_fpr"] for row in selected])
                ),
            }
        )
    return output


def run_methods(
    reference: np.ndarray,
    reference_y: np.ndarray,
    candidates: np.ndarray,
    candidate_y: np.ndarray,
    seed: int,
) -> dict[str, official.FeatureAdapterResult]:
    n_poison = 60
    return {
        "backdoorbench_spectral": official.spectral_signatures(
            candidates, candidate_y, percentile=85.0
        ),
        "backdoorbench_scan": official.scan(
            reference, reference_y, candidates, candidate_y, n_classes=2, seed=seed
        ),
        "backdoorbench_spectre_oracle_target": official.spectre(
            candidates, candidate_y, {0: 0, 1: n_poison}, pca_dimension=16
        ),
        "backdoorbench_spectre_oracle_all_classes": official.spectre(
            candidates, candidate_y, {0: n_poison, 1: n_poison}, pca_dimension=16
        ),
    }


def run(args: argparse.Namespace) -> dict:
    allowed = PILOT_SEEDS if args.stage == "pilot" else CONFIRMATION_SEEDS
    if tuple(args.seeds) != allowed:
        raise ValueError(f"{args.stage} stage is frozen to seeds {allowed}")
    args.output.mkdir(parents=True, exist_ok=True)
    class_data = heldout.collect_classes(args.dataset_root)
    rgb_trigger, _ = baseline.fixed_rgb_trigger(class_data)
    rows: list[dict] = []
    samples: list[dict] = []
    victims: list[dict] = []
    count_checks: list[dict] = []
    deterministic_repeat_passed = False

    for seed in args.seeds:
        prepared = baseline.preprocess_seed(class_data, seed)
        splits, split_receipt = baseline.seed_splits(prepared, rgb_trigger, seed)
        pool = candidate_pool(splits, split_receipt)
        counts = {
            name: int(np.sum(pool["kind"] == name))
            for name in ("target_clean", "source_clean", "poison")
        }
        counts.update(
            {
                "seed": seed,
                "reference_target": int(np.sum(pool["reference_y"] == 1)),
                "reference_source": int(np.sum(pool["reference_y"] == 0)),
            }
        )
        count_checks.append(counts)

        for scenario_index, scenario in enumerate(strong.SCENARIOS):
            print(f"seed={seed} scenario={scenario}", flush=True)
            clean_model, attacked_model, ca, asr = strong.load_scenario(
                scenario, seed, splits
            )
            eligible = int(ca >= 0.90 and asr >= 0.80)
            victims.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "balanced_ca": ca,
                    "asr": asr,
                    "eligible": eligible,
                }
            )
            reference = checkpoint_drift_features(
                scenario, clean_model, attacked_model, pool["reference_x"]
            )
            candidates = checkpoint_drift_features(
                scenario, clean_model, attacked_model, pool["candidate_x"]
            )
            method_seed = seed + 1000 * scenario_index
            outputs = run_methods(
                reference,
                pool["reference_y"],
                candidates,
                pool["candidate_y"],
                method_seed,
            )
            if seed == args.seeds[0] and scenario_index == 0:
                repeated = official.scan(
                    reference,
                    pool["reference_y"],
                    candidates,
                    pool["candidate_y"],
                    n_classes=2,
                    seed=method_seed,
                )
                original = outputs["backdoorbench_scan"]
                deterministic_repeat_passed = bool(
                    np.array_equal(original.suspicious_mask, repeated.suspicious_mask)
                    and np.array_equal(original.scores, repeated.scores)
                )

            for method, result in outputs.items():
                metrics = evaluate_flags(result.suspicious_mask, pool["truth"], pool["kind"])
                rows.append(
                    {
                        "stage": args.stage,
                        "seed": seed,
                        "scenario": scenario,
                        "eligible": eligible,
                        "view": "checkpoint_drift_xz16",
                        "method": method,
                        "dimension": candidates.shape[1],
                        **metrics,
                        "receipt_json": json.dumps(
                            result.receipt, sort_keys=True, separators=(",", ":")
                        ),
                    }
                )
                for index, (flag, score, label, truth, kind) in enumerate(
                    zip(
                        result.suspicious_mask,
                        result.scores,
                        pool["candidate_y"],
                        pool["truth"],
                        pool["kind"],
                    )
                ):
                    samples.append(
                        {
                            "seed": seed,
                            "scenario": scenario,
                            "method": method,
                            "sample_index": index,
                            "assigned_label": int(label),
                            "is_poison": int(truth),
                            "sample_kind": str(kind),
                            "score": float(score),
                            "flagged": int(flag),
                        }
                    )

    expected_counts = all(
        row["target_clean"] == 600
        and row["source_clean"] == 451
        and row["poison"] == 60
        and row["reference_target"] == 200
        and row["reference_source"] == 170
        for row in count_checks
    )
    victims_eligible = all(row["eligible"] for row in victims)
    finite_exposed_scores = all(
        np.isfinite(row["score"])
        for row in samples
        if row["method"] != "backdoorbench_spectre_oracle_target"
        and row["method"] != "backdoorbench_spectre_oracle_all_classes"
    )
    gate = {
        "pool_counts_match": expected_counts,
        "scan_deterministic_repeat": deterministic_repeat_passed,
        "finite_exposed_scores": finite_exposed_scores,
    }
    if args.stage == "pilot":
        gate["all_development_victims_eligible"] = victims_eligible
        gate["expand_to_confirmation"] = all(gate.values())
    else:
        gate["confirmation_complete"] = all(gate.values())

    summaries = aggregate_eligible(rows)
    unified.write_csv(args.output / "per_config_metrics.csv", rows)
    unified.write_csv(args.output / "per_sample_flags.csv", samples)
    unified.write_csv(args.output / "victim_metrics.csv", victims)
    unified.write_csv(args.output / "pool_count_checks.csv", count_checks)
    unified.write_csv(args.output / "eligible_summary.csv", summaries)
    manifest = {
        "experiment": (
            "bloodmnist_official_backdoorbench_pilot_v2"
            if args.stage == "pilot"
            else "bloodmnist_official_backdoorbench_confirmation_v1"
        ),
        "stage": args.stage,
        "protocol": str(PROTOCOL),
        "protocol_sha256": whitebox.sha256_file(PROTOCOL),
        "dataset_npz_sha256": whitebox.sha256_file(args.dataset_root / "bloodmnist.npz"),
        "adapter_sha256": whitebox.sha256_file(
            PAPER14_ROOT / "src/backdoorbench_feature_adapter.py"
        ),
        "upstream_commit": official.UPSTREAM_COMMIT,
        "seeds": list(args.seeds),
        "scenarios": list(strong.SCENARIOS),
        "methods": list(METHODS),
        "gate": gate,
    }
    unified.json_dump(args.output / "manifest.json", manifest)
    lines = [
        f"# Official BackdoorBench feature audit ({args.stage})",
        "",
        (
            f"Expansion gate: **{'PASS' if gate['expand_to_confirmation'] else 'STOP'}**"
            if args.stage == "pilot"
            else f"Confirmation integrity: **{'PASS' if gate['confirmation_complete'] else 'FAIL'}**"
        ),
        "",
        "| seed | scenario | method | eligible | TPR | all-clean FPR | target-clean FPR | source-clean FPR |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['scenario']} | {row['method']} | {row['eligible']} | "
            f"{row['tpr']:.3f} | {row['fpr_all_clean']:.3f} | "
            f"{row['target_clean_fpr']:.3f} | {row['source_clean_fpr']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Eligible-victim pooled summary",
            "",
            "| scenario | method | eligible seeds | pooled TPR | pooled all-clean FPR |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['scenario']} | {row['method']} | {row['eligible_seed_count']} | "
            f"{row['pooled_tpr']:.3f} | {row['pooled_fpr_all_clean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "SCAn is unmodified at the feature primitive level; its across-class anomaly index may degenerate on a two-class task.",
            "SPECTRE variants use explicit oracle contamination budgets and are therefore advantaged baselines.",
            "",
        ]
    )
    (args.output / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    parser.add_argument("--stage", choices=("pilot", "confirmation"), default="pilot")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.seeds is None:
        args.seeds = list(PILOT_SEEDS if args.stage == "pilot" else CONFIRMATION_SEEDS)
    run(args)


if __name__ == "__main__":
    main()
