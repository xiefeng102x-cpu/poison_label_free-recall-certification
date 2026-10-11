#!/usr/bin/env python3
"""Audit low-dimensional trigger observability across all MNIST t7_vs_t* pairs.

The audit uses the shared amplitude-encoding checkpoint, so no victim circuit is
rerun.  For every trigger/seed pair it reconstructs the same 180 matched pairs
and 60/20/20 pair-grouped split used by Phase A, computes exact five-qubit
Pauli expectations, and evaluates Z, XYZ, Full, and train-only PCA probes.
Seed 42 selects one PCA dimension per trigger; seeds 43 and 44 confirm it.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER14_ROOT = PROJECT_ROOT / "paper14"
DATA_ROOT = PROJECT_ROOT / "data/Mnist/detail_single_samle"
DEFAULT_OUTPUT = PAPER14_ROOT / "results/mnist_trigger_family_compression_v1"
REFERENCE_CACHE = PAPER14_ROOT / "results/phase_a_mnist5q_real_v1/cache"
BASE_SEEDS = (42, 43, 44)
ALL_SEEDS = tuple(range(42, 52))
NON_TARGETS = (0, 1, 2, 3, 4, 5, 6, 8, 9)
PCA_DIMS = (1, 2, 4, 8, 16, 32, 64)
C_GRID = (0.01, 0.1, 1.0, 10.0)
RANDOM_SEED = 20260927


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pauli_labels(n_qubits: int) -> list[str]:
    return [
        "".join(chars)
        for chars in itertools.product("IXYZ", repeat=n_qubits)
        if any(char != "I" for char in chars)
    ]


def pauli_actions(labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    n_qubits = len(labels[0])
    dim = 2**n_qubits
    basis = np.arange(dim, dtype=np.int64)
    permutations = np.empty((len(labels), dim), dtype=np.int64)
    phases = np.ones((len(labels), dim), dtype=np.complex128)
    for row, label in enumerate(labels):
        permutation = basis.copy()
        phase = np.ones(dim, dtype=np.complex128)
        for qubit, char in enumerate(label):
            if char == "I":
                continue
            bit_position = n_qubits - 1 - qubit
            bit = (basis >> bit_position) & 1
            if char == "X":
                permutation ^= 1 << bit_position
            elif char == "Y":
                phase *= 1j * (1 - 2 * bit)
                permutation ^= 1 << bit_position
            elif char == "Z":
                phase *= 1 - 2 * bit
        permutations[row] = permutation
        phases[row] = phase
    return permutations, phases


def pauli_expectations(
    states: np.ndarray, permutations: np.ndarray, phases: np.ndarray
) -> np.ndarray:
    output = np.empty((len(states), len(permutations)), dtype=np.float32)
    for start in range(0, len(permutations), 128):
        stop = min(start + 128, len(permutations))
        values = np.einsum(
            "bfd,fd,bd->bf",
            np.conjugate(states[:, permutations[start:stop]]),
            phases[start:stop],
            states,
            optimize=True,
        )
        if np.max(np.abs(values.imag)) > 1e-8:
            raise AssertionError("imaginary Pauli residual exceeds tolerance")
        output[:, start:stop] = values.real.astype(np.float32)
    return output


def source_dir(seed: int, non_target: int) -> Path:
    return DATA_ROOT / f"seed_{seed}/layer_12_grid/t7_vs_t{non_target}/pr_0.3"


def eligibility_record(seed: int, non_target: int) -> dict:
    root = source_dir(seed, non_target)
    eligibility = json.loads((root / "enhanced_scb_comprehensive_results_fixed.json").read_text())
    result = eligibility["aggregated_results"]["0.3"]
    ca = float(result["clean_accuracy"]["mean"])
    asr = float(result["attack_success_rate"]["mean"])
    return {
        "seed": seed,
        "trigger": f"t7_vs_t{non_target}",
        "clean_accuracy_percent": ca,
        "attack_success_rate_percent": asr,
        "eligible": ca >= 90.0 and asr >= 80.0,
    }


def load_pairs(seed: int, non_target: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    root = source_dir(seed, non_target)
    meta = torch.load(
        root / f"poisoned_samples_meta_seed_{seed}_pr_0.3.pt",
        map_location="cpu",
        weights_only=False,
    )
    original_indices = meta["poison_indices_in_non_target"].cpu().numpy().astype(int)
    clean_all = meta["non_target_clean_data"].cpu().numpy()
    attack_all = meta["poisoned_non_target_data"].cpu().numpy()
    rng = np.random.default_rng(RANDOM_SEED + seed)
    positions = np.sort(rng.choice(len(original_indices), size=min(180, len(original_indices)), replace=False))
    selected_indices = original_indices[positions]
    clean = clean_all[selected_indices]
    attack = attack_all[positions]
    trigger = meta["trigger_tensor"].cpu().numpy()
    reconstruction_error = float(np.max(np.abs(np.clip(clean + trigger, -3.0, 3.0) - attack)))
    if reconstruction_error > 1e-6:
        raise AssertionError(f"pair reconstruction error {reconstruction_error:.3e}")
    pair_ids = np.asarray([f"seed{seed}_nt{index}" for index in selected_indices])
    receipt = {
        **eligibility_record(seed, non_target),
        "available_pairs": int(len(original_indices)),
        "selected_pairs": int(len(pair_ids)),
        "pair_reconstruction_max_abs_error": reconstruction_error,
    }
    return clean, attack, pair_ids, receipt


def split_map(pair_ids: np.ndarray, seed: int) -> dict[str, str]:
    shuffled = pair_ids.copy()
    np.random.default_rng(RANDOM_SEED + seed).shuffle(shuffled)
    return {
        **{str(value): "train" for value in shuffled[:108]},
        **{str(value): "validation" for value in shuffled[108:144]},
        **{str(value): "test" for value in shuffled[144:]},
    }


def normalized_states(samples: np.ndarray) -> np.ndarray:
    flat = np.asarray(samples, dtype=np.float64).reshape(len(samples), -1)
    if flat.shape[1] < 32:
        flat = np.pad(flat, ((0, 0), (0, 32 - flat.shape[1])))
    elif flat.shape[1] > 32:
        flat = flat[:, :32]
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    return (flat / (norms + 1e-12)).astype(np.complex128)


def load_or_compute_features(
    output: Path,
    seed: int,
    non_target: int,
    clean: np.ndarray,
    attack: np.ndarray,
    pair_ids: np.ndarray,
    permutations: np.ndarray,
    phases: np.ndarray,
) -> dict[str, np.ndarray]:
    cache_dir = output / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"t7_vs_t{non_target}_seed{seed}_encoding_pauli.npz"
    if cache_path.exists():
        loaded = np.load(cache_path)
        return {key: loaded[key] for key in loaded.files}
    reference_path = REFERENCE_CACHE / f"seed{seed}_shared_encoding_pauli.npz"
    if non_target == 1 and reference_path.exists():
        reference = np.load(reference_path)
        if np.array_equal(reference["pair_ids"].astype(str), pair_ids.astype(str)):
            payload = {key: reference[key] for key in reference.files}
            np.savez_compressed(cache_path, **payload)
            return payload
    states = normalized_states(np.concatenate([clean, attack]))
    full = pauli_expectations(states, permutations, phases)
    purity_error = float(np.max(np.abs(np.sum(full.astype(float) ** 2, axis=1) - 31.0)))
    payload = {
        "clean_full": full[: len(pair_ids)],
        "attack_full": full[len(pair_ids) :],
        "pair_ids": pair_ids.astype("U"),
        "state_norm_max_abs_error": np.asarray(
            float(np.max(np.abs(np.sum(np.abs(states) ** 2, axis=1) - 1.0)))
        ),
        "pauli_purity_max_abs_error": np.asarray(purity_error),
    }
    np.savez_compressed(cache_path, **payload)
    return payload


def model(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("center", StandardScaler(with_mean=True, with_std=False)),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    penalty="l2",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def evaluate(features: dict[str, np.ndarray], labels: dict[str, np.ndarray], seed: int) -> dict:
    candidates = []
    for c_value in C_GRID:
        fitted = model(c_value, seed).fit(features["train"], labels["train"])
        auc = roc_auc_score(labels["validation"], fitted.decision_function(features["validation"]))
        candidates.append((float(auc), c_value))
    validation_auc, selected_c = sorted(candidates, key=lambda row: (-row[0], row[1]))[0]
    fitted = model(selected_c, seed).fit(
        np.concatenate([features["train"], features["validation"]]),
        np.concatenate([labels["train"], labels["validation"]]),
    )
    test_auc = float(roc_auc_score(labels["test"], fitted.decision_function(features["test"])))
    return {"selected_c": selected_c, "validation_auc": validation_auc, "test_auc": test_auc}


def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    labels_full = pauli_labels(5)
    weights = np.asarray([sum(char != "I" for char in label) for label in labels_full])
    z_indices = np.asarray(
        [i for i, label in enumerate(labels_full) if weights[i] == 1 and "Z" in label]
    )
    xyz_indices = np.flatnonzero(weights == 1)
    permutations, phases = pauli_actions(labels_full)
    metric_rows: list[dict] = []
    eligibility_rows = [
        eligibility_record(seed, non_target)
        for non_target in NON_TARGETS
        for seed in ALL_SEEDS
    ]
    eligibility_lookup = {
        (row["trigger"], int(row["seed"])): bool(row["eligible"])
        for row in eligibility_rows
    }
    receipts: list[dict] = []

    for non_target in NON_TARGETS:
        trigger = f"t7_vs_t{non_target}"
        evaluation_seeds = sorted(
            set(BASE_SEEDS)
            | {
                seed
                for seed in ALL_SEEDS
                if eligibility_lookup[(trigger, seed)]
            }
        )
        for seed in evaluation_seeds:
            clean, attack, pair_ids, eligibility = load_pairs(seed, non_target)
            payload = load_or_compute_features(
                args.output, seed, non_target, clean, attack, pair_ids, permutations, phases
            )
            mapping = split_map(pair_ids, seed)
            full: dict[str, np.ndarray] = {}
            target: dict[str, np.ndarray] = {}
            for split in ("train", "validation", "test"):
                mask = np.asarray([mapping[str(pair_id)] == split for pair_id in pair_ids])
                full[split] = np.concatenate(
                    [payload["clean_full"][mask], payload["attack_full"][mask]]
                )
                target[split] = np.concatenate(
                    [np.zeros(mask.sum(), dtype=int), np.ones(mask.sum(), dtype=int)]
                )

            def add(path: str, representation: str, values: dict[str, np.ndarray], **extra) -> None:
                result = evaluate(values, target, seed)
                metric_rows.append(
                    {
                        "trigger": trigger,
                        "non_target": non_target,
                        "seed": seed,
                        "eligible": eligibility["eligible"],
                        "clean_accuracy_percent": eligibility["clean_accuracy_percent"],
                        "attack_success_rate_percent": eligibility["attack_success_rate_percent"],
                        "path": path,
                        "representation": representation,
                        "dimension": values["train"].shape[1],
                        **result,
                        **extra,
                    }
                )

            add("local", "Z", {split: values[:, z_indices] for split, values in full.items()})
            add("local", "XYZ", {split: values[:, xyz_indices] for split, values in full.items()})
            add("full", "Full", full)
            pca = PCA(n_components=max(PCA_DIMS), svd_solver="randomized", random_state=seed + 4301)
            pca.fit(full["train"])
            transformed = {split: pca.transform(values) for split, values in full.items()}
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            for dimension in PCA_DIMS:
                add(
                    "pca",
                    f"PCA_{dimension}",
                    {split: values[:, :dimension] for split, values in transformed.items()},
                    explained_variance=float(cumulative[dimension - 1]),
                )
            receipts.append(
                {
                    "trigger": trigger,
                    "seed": seed,
                    "state_norm_max_abs_error": float(payload["state_norm_max_abs_error"]),
                    "pauli_purity_max_abs_error": float(payload["pauli_purity_max_abs_error"]),
                    "selected_pairs": len(pair_ids),
                }
            )
            print(f"{trigger} seed={seed} complete eligible={eligibility['eligible']}", flush=True)

    summary_rows = []
    for non_target in NON_TARGETS:
        trigger = f"t7_vs_t{non_target}"
        trigger_rows = [row for row in metric_rows if row["trigger"] == trigger]
        development_full = next(
            row for row in trigger_rows if row["seed"] == 42 and row["representation"] == "Full"
        )
        threshold = float(development_full["validation_auc"]) - 0.02
        candidates = sorted(
            (
                row
                for row in trigger_rows
                if row["seed"] == 42
                and row["path"] == "pca"
                and float(row["validation_auc"]) >= threshold
            ),
            key=lambda row: int(row["dimension"]),
        )
        selected_dimension = int(candidates[0]["dimension"]) if candidates else None
        selected_name = f"PCA_{selected_dimension}" if selected_dimension else None
        selected_rows = [row for row in trigger_rows if row["representation"] == selected_name]
        full_by_seed = {
            int(row["seed"]): row for row in trigger_rows if row["representation"] == "Full"
        }
        z_rows = [row for row in trigger_rows if row["representation"] == "Z"]
        xyz_rows = [row for row in trigger_rows if row["representation"] == "XYZ"]
        eligible_selected = [row for row in selected_rows if row["eligible"]]
        summary_rows.append(
            {
                "trigger": trigger,
                "eligible_seed_count": sum(
                    row["eligible"] for row in eligibility_rows if row["trigger"] == trigger
                ),
                "eligible_seeds": ",".join(
                    str(row["seed"])
                    for row in eligibility_rows
                    if row["trigger"] == trigger and row["eligible"]
                ),
                "selected_pca_dimension": selected_dimension if selected_dimension else "over_64",
                "development_validation_threshold": threshold,
                "selected_test_auc_mean_all_evaluated": float(
                    np.mean([row["test_auc"] for row in selected_rows])
                ) if selected_rows else "",
                "selected_within_002_full_all_evaluated": bool(
                    selected_rows
                    and all(
                        float(row["test_auc"])
                        >= float(full_by_seed[int(row["seed"])]["test_auc"]) - 0.02
                        for row in selected_rows
                    )
                ),
                "selected_within_002_full_all_eligible": bool(
                    eligible_selected
                    and all(
                        float(row["test_auc"])
                        >= float(full_by_seed[int(row["seed"])]["test_auc"]) - 0.02
                        for row in eligible_selected
                    )
                ),
                "z_test_auc_mean": float(np.mean([row["test_auc"] for row in z_rows])),
                "z_within_002_full_all_evaluated": all(
                    float(row["test_auc"])
                    >= float(full_by_seed[int(row["seed"])]["test_auc"]) - 0.02
                    for row in z_rows
                ),
                "xyz_test_auc_mean": float(np.mean([row["test_auc"] for row in xyz_rows])),
                "xyz_within_002_full_all_evaluated": all(
                    float(row["test_auc"])
                    >= float(full_by_seed[int(row["seed"])]["test_auc"]) - 0.02
                    for row in xyz_rows
                ),
                "full_test_auc_mean": float(
                    np.mean([row["test_auc"] for row in full_by_seed.values()])
                ),
            }
        )

    summary = {
        "status": "PASS",
        "protocol": {
            "checkpoint": "shared amplitude encoding before learned QNN layers",
            "triggers": [f"t7_vs_t{value}" for value in NON_TARGETS],
            "eligibility_seeds": list(ALL_SEEDS),
            "feature_evaluation_rule": "all seeds 42-44 plus every eligible seed 45-51",
            "pairs_per_seed_trigger": 180,
            "split_pairs": {"train": 108, "validation": 36, "test": 36},
            "development_seed": 42,
            "confirmation_seeds": "all evaluated seeds other than 42; effective-attack claims use eligible seeds only",
            "pca_fit_scope": "train only",
            "probe_c_selection": "validation only; refit train+validation before test",
            "low_dimension_tolerance_auc": 0.02,
            "victim_eligibility": "CA >= 90% and ASR >= 80%",
        },
        "counts": {
            "trigger_families": len(NON_TARGETS),
            "seed_trigger_eligibility_configurations": len(eligibility_rows),
            "feature_evaluated_configurations": len(receipts),
            "eligible_configurations": sum(row["eligible"] for row in eligibility_rows),
        },
        "all_trigger_summaries": summary_rows,
        "numeric_receipts": receipts,
    }
    write_csv(args.output / "eligibility.csv", eligibility_rows)
    write_csv(args.output / "probe_metrics.csv", metric_rows)
    write_csv(args.output / "trigger_summary.csv", summary_rows)
    (args.output / "trigger_family_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["counts"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output = args.output.resolve()
    main(args)
