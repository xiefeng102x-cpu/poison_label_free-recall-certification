#!/usr/bin/env python3
"""P6-D v5 prospective development: confidence-selected poisons + utility guard.

Only seeds 134--136 are accessible in this runner.  Confirmation seeds are
listed in the protocol but deliberately have no execution path until a
hash-bound development decision authorizes a separately frozen runner.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import run_p6d_final_observable_adaptive_v3 as v3


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "results/top_journal_program_v1/p6d_final_observable_adaptive_protocol_v5.json"
DEFAULT_OUTPUT = ROOT / "results/p6d_final_observable_adaptive_v5"
PROTOCOL_ID = "p6d_final_observable_adaptive_protocol_v5_confidence_guard"
DEVELOPMENT_SEEDS = (134, 135, 136)
CONFIRMATION_SEEDS = tuple(range(137, 147))
CONFIDENCE_RULES = ("hard60", "uncertain60", "stratified60")
EVASION_WEIGHT = 0.5
VALIDATION_CA_GUARD = 0.92
VALIDATION_ASR_GUARD = 0.90
FORMAL_ACKNOWLEDGEMENT = "CONSUME_P6D_V5_DEVELOPMENT_SEEDS_134_136"


def sha256_file(path: Path) -> str:
    return v3.sha256_file(path)


def load_protocol() -> dict[str, Any]:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("P6-D v5 protocol identifier mismatch")
    audit = payload["source_audit"]
    if audit["development_runner_sha256"] != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("P6-D v5 development runner is not source-frozen")
    if tuple(payload["seed_audit"]["development_seeds"]) != DEVELOPMENT_SEEDS:
        raise RuntimeError("P6-D v5 development seed lock mismatch")
    if tuple(payload["seed_audit"]["confirmation_seeds"]) != CONFIRMATION_SEEDS:
        raise RuntimeError("P6-D v5 confirmation seed lock mismatch")
    if tuple(payload["development_selection"]["grid"]) != CONFIDENCE_RULES:
        raise RuntimeError("P6-D v5 confidence-rule grid mismatch")
    return payload


def confidence_selected_indices(confidence: np.ndarray, rule: str, count: int = 60) -> np.ndarray:
    values = np.asarray(confidence, dtype=np.float64)
    if values.ndim != 1 or len(values) < count or not np.isfinite(values).all():
        raise ValueError("Confidence vector is invalid")
    index = np.arange(len(values), dtype=np.int64)
    ranked = np.lexsort((index, values))
    if rule == "hard60":
        selected = ranked[:count]
    elif rule == "uncertain60":
        selected = np.lexsort((index, np.abs(values - 0.5)))[:count]
    elif rule == "stratified60":
        positions = np.rint(np.linspace(0, len(values) - 1, count)).astype(np.int64)
        selected = ranked[positions]
    else:
        raise ValueError(f"Unknown confidence rule: {rule}")
    if len(np.unique(selected)) != count:
        raise AssertionError("Confidence selection returned duplicate members")
    return selected.astype(np.int64)


def utility_margin(row: Mapping[str, Any]) -> float:
    ca = (float(row["validation_balanced_ca"]) - VALIDATION_CA_GUARD) / 0.02
    asr = (float(row["validation_asr"]) - VALIDATION_ASR_GUARD) / 0.05
    return min(ca, asr)


def guarded_select_epoch(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not history:
        raise ValueError("Cannot select an epoch from empty history")
    if any(any(str(key).startswith("test_") for key in row) for row in history):
        raise RuntimeError("Test firewall: epoch selection received a test field")
    guarded = [
        row for row in history
        if float(row["validation_balanced_ca"]) >= VALIDATION_CA_GUARD
        and float(row["validation_asr"]) >= VALIDATION_ASR_GUARD
    ]
    if guarded:
        selected = min(
            guarded,
            key=lambda row: (
                float(row["tpr"]),
                -float(row["mean_poison_p_value"]),
                -utility_margin(row),
                int(row["epoch"]),
            ),
        )
        eligible = True
    else:
        selected = min(
            history,
            key=lambda row: (-utility_margin(row), float(row["tpr"]), int(row["epoch"])),
        )
        eligible = False
    result = copy.deepcopy(dict(selected))
    result["validation_eligible"] = eligible
    result["normalized_guard_band_margin"] = utility_margin(result)
    return result


def summarize_configuration(rule: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if sorted(int(row["seed"]) for row in rows) != list(DEVELOPMENT_SEEDS):
        raise ValueError("Configuration summary requires the three frozen development seeds")
    guarded = [row for row in rows if bool(row["validation_eligible"])]
    return {
        "configuration_id": rule,
        "guarded_seed_count": len(guarded),
        "worst_normalized_utility_margin": min(float(row["normalized_guard_band_margin"]) for row in rows),
        "mean_tpr_guarded": float(np.mean([float(row["tpr"]) for row in guarded])) if guarded else None,
        "worst_validation_asr": min(float(row["validation_asr"]) for row in rows),
        "worst_validation_balanced_ca": min(float(row["validation_balanced_ca"]) for row in rows),
        "selected_epochs": {str(int(row["seed"])): int(row["epoch"]) for row in rows},
    }


def configuration_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    mean_tpr = row["mean_tpr_guarded"]
    return (
        -int(row["guarded_seed_count"]),
        -float(row["worst_normalized_utility_margin"]),
        float(mean_tpr) if mean_tpr is not None else float("inf"),
        -float(row["worst_validation_asr"]),
        CONFIDENCE_RULES.index(str(row["configuration_id"])),
    )


def confidence_vector(model: torch.nn.Module, triggered: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(triggered, dtype=torch.float32))
        result = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().astype(np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite confidence score")
    return result


def run_development(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_protocol()
    if args.acknowledge_formal_seed_consumption != FORMAL_ACKNOWLEDGEMENT:
        raise PermissionError("Development requires --acknowledge-formal-seed-consumption " + FORMAL_ACKNOWLEDGEMENT)
    dataset_path = args.dataset_root / "bloodmnist.npz"
    if sha256_file(dataset_path) != protocol["source_audit"]["dataset_sha256"]:
        raise RuntimeError("P6-D v5 BloodMNIST hash mismatch")
    output = args.output / "development"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("P6-D v5 development output already exists; formal seeds may not be rerun")
    output.mkdir(parents=True, exist_ok=True)

    import evaluate_bloodmnist_checkpoints as heldout
    import run_bloodmnist_group_coherent_10seeds as baseline
    import run_bloodmnist_whitebox_adaptive_pilot as whitebox
    import train_bloodmnist_clean_encoder as clean_encoder

    classes = heldout.collect_classes(args.dataset_root)
    rgb_trigger, _ = baseline.fixed_rgb_trigger(classes)
    v3._runtime.FORMAL_SEEDS = DEVELOPMENT_SEEDS + CONFIRMATION_SEEDS
    original_selector = v3._runtime.select_epoch
    v3._runtime.select_epoch = guarded_select_epoch
    histories: list[dict[str, Any]] = []
    selection_rows: dict[str, list[dict[str, Any]]] = {rule: [] for rule in CONFIDENCE_RULES}
    selection_receipts: list[dict[str, Any]] = []
    stored: dict[tuple[int, str], dict[str, Any]] = {}
    test_builders: dict[int, Any] = {}
    try:
        for seed in DEVELOPMENT_SEEDS:
            splits, trigger_receipt, build_test_once, _ = v3._p6d_prepare_seed(classes, rgb_trigger, seed, baseline)
            test_builders[seed] = build_test_once
            clean_model, clean_record = clean_encoder.train_one(
                {"train": (splits["train"]["clean_x"], splits["train"]["clean_y"]),
                 "val": (splits["val"]["clean_x"], splits["val"]["clean_y"])},
                seed, "local", 0, 40, 3e-3, 1e-4, 64, 10,
            )
            confidence = confidence_vector(clean_model, splits["train"]["triggered_non_target"])
            for rule in CONFIDENCE_RULES:
                selected_ids = confidence_selected_indices(confidence, rule)
                attack_splits = {
                    "train": splits["train"],
                    "val": splits["val"],
                    "archived_poison": splits["train"]["triggered_non_target"][selected_ids],
                }

                def validation_metric(model: torch.nn.Module) -> tuple[float, float]:
                    return whitebox.balanced_ca_and_asr(
                        model, splits["val"]["target"], splits["val"]["non_target"],
                        splits["val"]["triggered_non_target"],
                    )

                result = v3.train_attack_configuration(
                    seed=seed, evasion_weight=EVASION_WEIGHT, clean_model=clean_model,
                    splits=attack_splits, validation_metric=validation_metric,
                )
                row = {
                    **result["selected"], "seed": seed, "configuration_id": rule,
                    "evasion_weight": EVASION_WEIGHT,
                }
                selection_rows[rule].append(row)
                stored[(seed, rule)] = result
                for history in result["history"]:
                    histories.append({
                        "seed": seed, "configuration_id": rule, "evasion_weight": EVASION_WEIGHT,
                        **{key: value for key, value in history.items() if key != "objectives"},
                        "selected_epoch": int(history["epoch"]) == int(result["selected"]["epoch"]),
                        "objectives_json": json.dumps(history["objectives"], sort_keys=True),
                    })
                selection_receipts.append({
                    "seed": seed, "configuration_id": rule,
                    "selected_train_indices": selected_ids.astype(int).tolist(),
                    "selected_indices_sha256": v3.stable_json_sha256(selected_ids.astype(int).tolist()),
                    "selected_confidence_min": float(np.min(confidence[selected_ids])),
                    "selected_confidence_mean": float(np.mean(confidence[selected_ids])),
                    "selected_confidence_max": float(np.max(confidence[selected_ids])),
                    "full_confidence_sha256_float64": v3.sha256_array(confidence, dtype="<f8"),
                    "original_random_poison_indices_sha256": v3.stable_json_sha256(
                        np.asarray(trigger_receipt["poison_indices"], dtype=int).tolist()
                    ),
                    "clean_best_epoch": int(clean_record["best_epoch"]),
                })
    finally:
        v3._runtime.select_epoch = original_selector

    summaries = [summarize_configuration(rule, selection_rows[rule]) for rule in CONFIDENCE_RULES]
    ranked = sorted(summaries, key=configuration_key)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    selected_rule = str(ranked[0]["configuration_id"])
    victim_rows: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    for seed in DEVELOPMENT_SEEDS:
        result = stored[(seed, selected_rule)]
        test = test_builders[seed]()
        ca, asr = whitebox.balanced_ca_and_asr(
            result["model"], test["target"], test["non_target"], test["triggered_non_target"]
        )
        exact = v3.numpy_cfcmdu_score(
            seed, result["selected_observables"]["trusted"],
            result["selected_observables"]["clean"], result["selected_observables"]["poison"],
            partition=result["partition"],
        )
        eligible = bool(ca >= 0.90 and asr >= 0.80)
        victim_rows.append({
            "seed": seed, "configuration_id": selected_rule, "selected_epoch": int(result["selected"]["epoch"]),
            "test_balanced_ca": float(ca), "test_asr": float(asr), "eligible": eligible,
        })
        per_seed.append({
            "seed": seed, "configuration_id": selected_rule, "eligible": eligible,
            "tpr": float(exact["tpr"]), "fpr": float(exact["fpr"]),
            "tp": int(exact["tp"]), "fn": int(exact["fn"]),
            "fp": int(exact["fp"]), "tn": int(exact["tn"]),
        })
        checkpoint = output / "models" / f"seed_{seed}_{selected_rule}.pt"
        v3.atomic_torch_save(checkpoint, {
            "seed": seed, "configuration_id": selected_rule,
            "selected_epoch": int(result["selected"]["epoch"]),
            "model_state": result["model"].state_dict(),
        })

    viable = sum(bool(row["eligible"]) for row in victim_rows)
    evasive = sum(bool(row["eligible"]) and float(row["tpr"]) < 0.90 and float(row["fpr"]) <= 0.05 for row in per_seed)
    go = viable == 3 and evasive >= 2
    selection = {
        "protocol_id": PROTOCOL_ID,
        "selection_uses_test": False,
        "selected_configuration_id": selected_rule,
        "ranked_configurations": ranked,
        "ranking_rule": protocol["development_selection"]["configuration_ranking"],
    }
    summary = {
        "protocol_id": PROTOCOL_ID, "development_seeds": list(DEVELOPMENT_SEEDS),
        "selected_configuration_id": selected_rule, "attack_viable_count": viable,
        "evasive_and_viable_count": evasive,
        "confirmation_authorized": go,
        "decision": "AUTHORIZE_CONFIRMATION_FREEZE" if go else "STOP_NO_CONFIRMATION",
    }
    v3.atomic_csv(output / "epoch_history.csv", histories)
    v3.atomic_csv(output / "victim_metrics.csv", victim_rows)
    v3.atomic_csv(output / "per_seed_metrics.csv", per_seed)
    v3.atomic_json(output / "confidence_selection_receipts.json", selection_receipts)
    v3.atomic_json(output / "development_selection_receipt.json", selection)
    v3._atomic_bytes(
        output / "development_selection_receipt.json.sha256",
        (sha256_file(output / "development_selection_receipt.json") + "\n").encode(),
    )
    v3.atomic_json(output / "development_summary.json", summary)
    artifacts = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    manifest = {
        "status": "complete", "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "dataset_sha256": sha256_file(dataset_path), "artifacts": artifacts,
    }
    v3.atomic_json(output / "manifest.json", manifest)
    return summary


def self_check() -> dict[str, Any]:
    protocol = load_protocol()
    example = np.linspace(0.0, 1.0, 100)
    selections = {rule: confidence_selected_indices(example, rule).tolist() for rule in CONFIDENCE_RULES}
    if any(len(set(values)) != 60 for values in selections.values()):
        raise AssertionError("P6-D v5 selector self-check failed")
    rows = [
        {"epoch": 0, "validation_balanced_ca": 0.93, "validation_asr": 0.89,
         "tpr": 0.1, "mean_poison_p_value": 0.4},
        {"epoch": 1, "validation_balanced_ca": 0.93, "validation_asr": 0.91,
         "tpr": 0.3, "mean_poison_p_value": 0.5},
    ]
    if guarded_select_epoch(rows)["epoch"] != 1:
        raise AssertionError("P6-D v5 utility guard self-check failed")
    return {"status": "PASS", "protocol_id": protocol["protocol_id"], "development_seeds": list(DEVELOPMENT_SEEDS)}


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    sub = output.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check")
    run = sub.add_parser("run-development")
    run.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--acknowledge-formal-seed-consumption")
    return output


def main() -> None:
    args = parser().parse_args()
    if args.command == "self-check":
        result = self_check()
    else:
        args.dataset_root = args.dataset_root.resolve()
        args.output = args.output.resolve()
        result = run_development(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
