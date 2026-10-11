#!/usr/bin/env python3
"""Post-Stage-1 QML diagnostic using local Pauli views at victim checkpoints."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pennylane as qml
import torch

import run_unified_oneclass_poison_screening as unified


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PAPER14_ROOT / "results/unified_qml_checkpoint_diagnostic_v1"
PHASE_CACHE = PAPER14_ROOT / "results/phase_a_mnist5q_real_v1/cache"
CHECKPOINTS = {"middle": 6, "output": 12}


def pauli_labels() -> list[str]:
    return [
        "".join(chars)
        for chars in itertools.product("IXYZ", repeat=5)
        if any(char != "I" for char in chars)
    ]


def local_indices() -> tuple[np.ndarray, np.ndarray]:
    labels = pauli_labels()
    lookup = {label: index for index, label in enumerate(labels)}
    ordered = []
    for qubit in range(5):
        for axis in ("X", "Z"):
            chars = ["I"] * 5
            chars[qubit] = axis
            ordered.append(lookup["".join(chars)])
    xz = np.asarray(ordered, dtype=int)
    z = xz[1::2]
    return z, xz


def build_local_circuit(weights_flat: np.ndarray, checkpoint_layers: int):
    weights = np.asarray(weights_flat, dtype=float).reshape(12, 5, 2)
    device = qml.device("default.qubit", wires=5)

    @qml.qnode(device, interface="numpy")
    def circuit(sample):
        vector = np.asarray(sample, dtype=float).reshape(-1)
        if vector.size < 32:
            vector = np.pad(vector, (0, 32 - vector.size))
        elif vector.size > 32:
            vector = vector[:32]
        vector = vector / (np.linalg.norm(vector) + 1e-12)
        qml.AmplitudeEmbedding(vector, wires=range(5), normalize=False)
        for layer in range(checkpoint_layers):
            for qubit in range(5):
                qml.RY(weights[layer, qubit, 0], wires=qubit)
                qml.RZ(weights[layer, qubit, 1], wires=qubit)
            for qubit in range(5):
                qml.CNOT(wires=[qubit, (qubit + 1) % 5])
        return tuple(
            qml.expval(observable(qubit))
            for qubit in range(5)
            for observable in (qml.PauliX, qml.PauliZ)
        )

    return circuit


def extract_local(samples: np.ndarray, circuit) -> np.ndarray:
    return np.stack([np.asarray(circuit(sample), dtype=np.float64) for sample in samples])


def checkpoint_path(seed: int) -> Path:
    return (
        unified.source_dir(seed)
        / "models"
        / f"backdoor_model_seed_{seed}_pr_0.3.pt"
    )


def load_cached_source_attack(seed: int, checkpoint: str) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(PHASE_CACHE / f"seed{seed}_backdoor_{checkpoint}_pauli.npz")
    z_indices, xz_indices = local_indices()
    clean = payload["clean_full"]
    attack = payload["attack_full"]
    return (
        np.asarray(clean[:, xz_indices], dtype=np.float64),
        np.asarray(attack[:, xz_indices], dtype=np.float64),
    )


def target_features(
    output: Path,
    seed: int,
    checkpoint: str,
    layers: int,
    target: np.ndarray,
    trigger_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cache = output / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"seed{seed}_backdoor_{checkpoint}_target_local_xz.npz"
    if path.exists():
        payload = np.load(path)
        return payload["target_xz"], payload["trigger_target_xz"]
    state = torch.load(checkpoint_path(seed), map_location="cpu", weights_only=False)
    circuit = build_local_circuit(state["weights"].detach().cpu().numpy(), layers)
    target_xz = extract_local(target, circuit)
    trigger_target_xz = extract_local(trigger_target, circuit)
    np.savez_compressed(path, target_xz=target_xz, trigger_target_xz=trigger_target_xz)
    return target_xz, trigger_target_xz


def make_sets(
    target_xz: np.ndarray,
    trigger_target_xz: np.ndarray,
    source_xz: np.ndarray,
    attack_xz: np.ndarray,
    split: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "reference": target_xz[split["target_reference"]][:, indices],
        "calibration": target_xz[split["target_calibration"]][:, indices],
        "clean_target": target_xz[split["target_test"]][:, indices],
        "poison_source": attack_xz[split["poison_test"]][:, indices],
        "clean_source": source_xz[split["poison_test"]][:, indices],
        "trigger_target": trigger_target_xz[split["trigger_target_test"]][:, indices],
    }


def write_report(path: Path, summaries: list[dict], elapsed: float) -> None:
    lines = [
        "# QML checkpoint diagnostic",
        "",
        "This is a post-Stage-1 diagnostic. It does not change the original Stage-1 decision.",
        "",
        "| representation | pooled TPR | pooled FPR | mean AUC | clean-source flagged | triggered-target flagged |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['representation']} | {row['pooled_tpr']:.3f} | "
            f"{row['pooled_fpr']:.3f} | {row['mean_auc']:.3f} | "
            f"{row['mean_clean_source_flag_rate']:.3f} | "
            f"{row['mean_trigger_target_flag_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "All PCA dimensions and thresholds were selected from clean target reference/calibration samples only.",
            "",
            f"Runtime: {elapsed:.1f} seconds.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows: list[dict] = []
    samples: list[dict] = []
    # Cached source/attack features are ordered X0,Z0,...,X4,Z4.
    z_columns = np.asarray([1, 3, 5, 7, 9])
    xz_columns = np.arange(10)
    for seed in args.seeds:
        data = unified.load_seed(seed)
        split = unified.make_splits(data, seed)
        for checkpoint, layers in CHECKPOINTS.items():
            target_xz, trigger_target_xz = target_features(
                args.output,
                seed,
                checkpoint,
                layers,
                data["target"],
                data["trigger_target"],
            )
            source_xz, attack_xz = load_cached_source_attack(seed, checkpoint)
            for suffix, columns in (("z5", z_columns), ("xz10", xz_columns)):
                sets = make_sets(
                    target_xz,
                    trigger_target_xz,
                    source_xz,
                    attack_xz,
                    split,
                    columns,
                )
                row, sample_rows = unified.evaluate_representation(
                    seed, f"qml_{checkpoint}_{suffix}", sets, args.alpha
                )
                rows.append(row)
                samples.extend(sample_rows)
    summaries = unified.pooled_summary(rows)
    unified.write_csv(args.output / "per_seed_metrics.csv", rows)
    unified.write_csv(args.output / "per_sample_scores.csv", samples)
    unified.write_csv(args.output / "pooled_summary.csv", summaries)
    elapsed = time.time() - started
    unified.json_dump(
        args.output / "manifest.json",
        {
            "experiment": "unified_qml_checkpoint_diagnostic_v1",
            "post_stage1_diagnostic": True,
            "seeds": args.seeds,
            "alpha": args.alpha,
            "checkpoints": CHECKPOINTS,
            "elapsed_seconds": elapsed,
        },
    )
    write_report(args.output / "EXPERIMENT_REPORT.md", summaries, elapsed)
    for row in summaries:
        print(
            f"{row['representation']:>24s}: TPR={row['pooled_tpr']:.3f} "
            f"FPR={row['pooled_fpr']:.3f} AUC={row['mean_auc']:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--seeds",
        type=lambda value: [int(item) for item in value.split(",")],
        default=list(unified.SEEDS),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    main(parser.parse_args())
