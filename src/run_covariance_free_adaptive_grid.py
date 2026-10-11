#!/usr/bin/env python3
"""Validation-only grid for covariance-free adaptive attack parameters.

The driver never evaluates a test split.  It reuses the existing white-box
training routine, writes a record after every completed seed/configuration,
and can safely resume an interrupted grid with the same run signature.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
import time
from pathlib import Path
from typing import NamedTuple, Sequence

import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_group_coherent_10seeds as baseline
import run_bloodmnist_whitebox_adaptive_pilot as pilot


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PAPER14_ROOT / "results/covariance_free_adaptive_grid_v1/grid.json"
MODE = "checkpoint_drift_representation_matching"
SELECTION_DETECTOR = "covariance_free_xz16"
CA_GATE = 0.90
ASR_GATE = 0.80


class ParameterCombo(NamedTuple):
    backdoor_weight: float
    evasion_weight: float
    distill_weight: float


def build_combos(
    backdoor_weights: Sequence[float],
    evasion_weights: Sequence[float],
    distill_weights: Sequence[float],
) -> list[ParameterCombo]:
    return [
        ParameterCombo(float(backdoor), float(evasion), float(distill))
        for backdoor, evasion, distill in itertools.product(
            backdoor_weights, evasion_weights, distill_weights
        )
    ]


def record_key(record: dict) -> tuple[int, float, float, float]:
    return (
        int(record["seed"]),
        float(record["backdoor_weight"]),
        float(record["evasion_weight"]),
        float(record["distill_weight"]),
    )


def combo_key(combo: ParameterCombo, seed: int) -> tuple[int, float, float, float]:
    return (int(seed), *map(float, combo))


def is_eligible(record: dict) -> bool:
    return (
        float(record["validation_balanced_ca"]) >= CA_GATE
        and float(record["validation_asr"]) >= ASR_GATE
    )


def summarize_combo(combo: ParameterCombo, records: Sequence[dict]) -> dict:
    selected = sorted(records, key=lambda row: int(row["seed"]))
    eligible = [row for row in selected if is_eligible(row)]
    margins = [
        min(
            float(row["validation_balanced_ca"]) / CA_GATE,
            float(row["validation_asr"]) / ASR_GATE,
        )
        for row in selected
    ]
    return {
        "backdoor_weight": combo.backdoor_weight,
        "evasion_weight": combo.evasion_weight,
        "distill_weight": combo.distill_weight,
        "seed_count": len(selected),
        "seeds": [int(row["seed"]) for row in selected],
        "eligible_seed_count": len(eligible),
        "eligible_seeds": [int(row["seed"]) for row in eligible],
        "mean_covariance_free_tpr_eligible": (
            sum(float(row["covariance_free_tpr"]) for row in eligible) / len(eligible)
            if eligible
            else None
        ),
        "mean_covariance_free_fpr_eligible": (
            sum(float(row["covariance_free_fpr"]) for row in eligible) / len(eligible)
            if eligible
            else None
        ),
        "worst_validation_ca": (
            min(float(row["validation_balanced_ca"]) for row in selected)
            if selected
            else None
        ),
        "worst_validation_asr": (
            min(float(row["validation_asr"]) for row in selected)
            if selected
            else None
        ),
        "fallback_gate_margin": min(margins) if margins else None,
    }


def rank_key(summary: dict) -> tuple:
    """Ascending key implementing the frozen best-first ordering."""

    eligible = int(summary["eligible_seed_count"])
    parameters = (
        float(summary["backdoor_weight"]),
        float(summary["evasion_weight"]),
        float(summary["distill_weight"]),
    )
    if eligible:
        secondary = (
            0,
            float(summary["mean_covariance_free_tpr_eligible"]),
            -float(summary["worst_validation_ca"]),
            -float(summary["worst_validation_asr"]),
        )
    else:
        margin = summary["fallback_gate_margin"]
        secondary = (1, -float(margin) if margin is not None else float("inf"), 0.0, 0.0)
    return (-eligible, secondary, parameters)


def rank_combos(summaries: Sequence[dict]) -> list[dict]:
    ordered = sorted((copy.deepcopy(row) for row in summaries), key=rank_key)
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
    return ordered


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"metadata": {}, "runs": [], "combo_summaries": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
        raise ValueError(f"Invalid resume file: {path}")
    return payload


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_runs(runs: Sequence[dict], combos: Sequence[ParameterCombo]) -> list[dict]:
    output = []
    for combo in combos:
        matching = [
            row
            for row in runs
            if record_key(row)[1:] == tuple(map(float, combo))
        ]
        output.append(summarize_combo(combo, matching))
    return rank_combos(output)


def persist(path: Path, metadata: dict, runs: Sequence[dict], combos: Sequence[ParameterCombo]) -> list[dict]:
    summaries = summarize_runs(runs, combos)
    payload = {
        "metadata": metadata,
        "runs": list(runs),
        "combo_summaries": summaries,
        "selected_config": summaries[0] if summaries else None,
    }
    atomic_json(path, payload)
    write_csv(path.with_suffix(".csv"), summaries)
    return summaries


def build_record(seed: int, combo: ParameterCombo, result: dict) -> dict:
    selected = result["selected"]
    return {
        "seed": int(seed),
        "backdoor_weight": combo.backdoor_weight,
        "evasion_weight": combo.evasion_weight,
        "distill_weight": combo.distill_weight,
        "selected_epoch": int(selected["epoch"]),
        "validation_balanced_ca": float(selected["validation_balanced_ca"]),
        "validation_asr": float(selected["validation_asr"]),
        "covariance_free_tpr": float(selected["covariance_free_tpr"]),
        "covariance_free_fpr": float(selected["covariance_free_fpr"]),
        "eligible_on_validation": bool(selected["eligible_on_validation"]),
    }


def run_signature(args: argparse.Namespace, combos: Sequence[ParameterCombo]) -> dict:
    return {
        "seeds": list(map(int, args.seeds)),
        "parameter_combos": [list(map(float, combo)) for combo in combos],
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "alpha": float(args.alpha),
        "learning_rate": float(args.learning_rate),
        "anchor_weight": float(args.anchor_weight),
        "gradient_clip": float(args.gradient_clip),
    }


def main(args: argparse.Namespace) -> None:
    combos = build_combos(
        args.backdoor_weights, args.evasion_weights, args.distill_weights
    )
    signature = run_signature(args, combos)
    state = load_state(args.output)
    previous_signature = state.get("metadata", {}).get("run_signature")
    if previous_signature and previous_signature != signature:
        raise ValueError("Resume file run signature does not match requested grid")
    runs = list(state.get("runs", []))
    completed = {record_key(row) for row in runs}
    metadata = {
        "experiment": "covariance_free_adaptive_grid_v1",
        "role": "development_only",
        "mode": MODE,
        "selection_detector": SELECTION_DETECTOR,
        "test_split_used": False,
        "gate": {"balanced_ca_min": CA_GATE, "asr_min": ASR_GATE},
        "run_signature": signature,
    }

    class_data = heldout.collect_classes(args.dataset_root)
    rgb_trigger, _ = baseline.fixed_rgb_trigger(class_data)
    started = time.time()
    for seed in args.seeds:
        seed_data = pilot.prepare_seed(
            class_data, rgb_trigger, seed, clean_root=args.clean_root
        )
        for combo in combos:
            key = combo_key(combo, seed)
            if key in completed:
                continue
            run_args = copy.deepcopy(args)
            run_args.backdoor_weight = combo.backdoor_weight
            run_args.distill_weight = combo.distill_weight
            run_args.selection_detector = SELECTION_DETECTOR
            result = pilot.run_configuration(
                seed_data,
                seed=seed,
                mode=MODE,
                evasion_weight=combo.evasion_weight,
                args=run_args,
            )
            runs.append(build_record(seed, combo, result))
            completed.add(key)
            metadata["elapsed_seconds"] = time.time() - started
            persist(args.output, metadata, runs, combos)

    metadata["elapsed_seconds"] = time.time() - started
    persist(args.output, metadata, runs, combos)


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    output.add_argument("--clean-root", type=Path, required=True)
    output.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    output.add_argument("--seeds", nargs="+", type=int, required=True)
    output.add_argument("--backdoor-weights", nargs="+", type=float, required=True)
    output.add_argument("--evasion-weights", nargs="+", type=float, required=True)
    output.add_argument("--distill-weights", nargs="+", type=float, required=True)
    output.add_argument("--epochs", type=int, default=8)
    output.add_argument("--batch-size", type=int, default=64)
    output.add_argument("--alpha", type=float, default=0.05)
    output.add_argument("--learning-rate", type=float, default=0.01)
    output.add_argument("--trigger-learning-rate", type=float, default=0.002)
    output.add_argument("--anchor-weight", type=float, default=0.001)
    output.add_argument("--gradient-clip", type=float, default=1.0)
    output.add_argument("--direction-amplitude", type=float, default=0.2)
    output.add_argument("--trigger-linf", type=float, default=0.8)
    output.add_argument("--trigger-correction-linf", type=float, default=0.2)
    return output


if __name__ == "__main__":
    main(parser().parse_args())

