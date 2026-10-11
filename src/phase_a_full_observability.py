#!/usr/bin/env python3
"""Leakage-safe Phase A validation for Paper 14.

The runner reuses frozen MNIST QNN victims and matched clean/triggered inputs,
extracts exact statevectors at three circuit checkpoints, expands them in the
complete non-identity Pauli basis, and evaluates Z/XYZ/Full representations on
an untouched pair-grouped test split.

The full Pauli representation is an oracle. Results from this script establish
held-out discriminability and measurement accessibility, not deployability or
causal mechanism.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pennylane as qml
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAULI_CHARS = ("I", "X", "Y", "Z")


def json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def csv_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def pauli_labels(n_qubits: int) -> list[str]:
    return [
        "".join(chars)
        for chars in itertools.product(PAULI_CHARS, repeat=n_qubits)
        if any(char != "I" for char in chars)
    ]


@dataclass(frozen=True)
class PauliActions:
    labels: tuple[str, ...]
    permutations: np.ndarray
    phases: np.ndarray
    weights: np.ndarray


def build_pauli_actions(n_qubits: int) -> PauliActions:
    """Build P|j> = phase[P,j] |permutation[P,j]> in PennyLane wire order."""
    labels = pauli_labels(n_qubits)
    dim = 2**n_qubits
    basis = np.arange(dim, dtype=np.int64)
    permutations = np.empty((len(labels), dim), dtype=np.int64)
    phases = np.ones((len(labels), dim), dtype=np.complex128)
    weights = np.empty(len(labels), dtype=np.int16)

    for row, label in enumerate(labels):
        permutation = basis.copy()
        phase = np.ones(dim, dtype=np.complex128)
        weight = 0
        for qubit, char in enumerate(label):
            if char == "I":
                continue
            weight += 1
            bit_position = n_qubits - 1 - qubit
            bit = (basis >> bit_position) & 1
            if char == "X":
                permutation ^= 1 << bit_position
            elif char == "Y":
                phase *= 1j * (1 - 2 * bit)
                permutation ^= 1 << bit_position
            elif char == "Z":
                phase *= 1 - 2 * bit
            else:  # pragma: no cover - labels are generated internally
                raise ValueError(char)
        permutations[row] = permutation
        phases[row] = phase
        weights[row] = weight
    return PauliActions(tuple(labels), permutations, phases, weights)


def pauli_expectations(
    states: np.ndarray,
    actions: PauliActions,
    feature_chunk_size: int = 128,
) -> np.ndarray:
    """Return exact real Pauli expectations for normalized pure states."""
    states = np.asarray(states, dtype=np.complex128)
    out = np.empty((states.shape[0], len(actions.labels)), dtype=np.float64)
    for start in range(0, len(actions.labels), feature_chunk_size):
        stop = min(start + feature_chunk_size, len(actions.labels))
        permutation = actions.permutations[start:stop]
        phase = actions.phases[start:stop]
        # <psi|P|psi> = sum_j conj(psi[P(j)]) phase(P,j) psi[j]
        values = np.einsum(
            "bfd,fd,bd->bf",
            np.conjugate(states[:, permutation]),
            phase,
            states,
            optimize=True,
        )
        imag_max = float(np.max(np.abs(values.imag)))
        if imag_max > 1e-8:
            raise AssertionError(f"Pauli expectation has imaginary residual {imag_max:.3e}")
        out[:, start:stop] = values.real
    return out


def build_state_circuit(
    weights_flat: np.ndarray,
    n_qubits: int,
    n_layers: int,
    checkpoint_layers: int,
):
    if not 0 <= checkpoint_layers <= n_layers:
        raise ValueError(f"checkpoint {checkpoint_layers} outside [0, {n_layers}]")
    weights = np.asarray(weights_flat, dtype=float).reshape(n_layers, n_qubits, 2)
    dim = 2**n_qubits
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device, interface="numpy")
    def circuit(sample):
        vector = np.asarray(sample, dtype=float).copy()
        if vector.size < dim:
            vector = np.pad(vector, (0, dim - vector.size))
        elif vector.size > dim:
            vector = vector[:dim]
        vector = vector / (np.linalg.norm(vector) + 1e-12)
        qml.AmplitudeEmbedding(vector, wires=range(n_qubits), normalize=False)
        for layer in range(checkpoint_layers):
            for qubit in range(n_qubits):
                qml.RY(weights[layer, qubit, 0], wires=qubit)
                qml.RZ(weights[layer, qubit, 1], wires=qubit)
            for qubit in range(n_qubits):
                qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])
        return qml.state()

    return circuit


def extract_states(samples: np.ndarray, circuit) -> np.ndarray:
    return np.stack([np.asarray(circuit(sample), dtype=np.complex128) for sample in samples])


def source_paths(config: dict, seed: int) -> dict[str, Path]:
    pr = float(config["poison_ratio"])
    base = (
        PROJECT_ROOT
        / config["data_root"]
        / f"seed_{seed}"
        / f"layer_{config['n_layers']}_grid"
        / config["pair"]
        / f"pr_{pr:.1f}"
    )
    return {
        "base": base,
        "meta": base / f"poisoned_samples_meta_seed_{seed}_pr_{pr:.1f}.pt",
        "backdoor": base / "models" / f"backdoor_model_seed_{seed}_pr_{pr:.1f}.pt",
        "clean": base / "models" / f"clean_model_seed_{seed}_epoch_200.pt",
        "eligibility": base / "enhanced_scb_comprehensive_results_fixed.json",
    }


def load_matched_pairs(config: dict, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    paths = source_paths(config, seed)
    meta = torch.load(paths["meta"], map_location="cpu", weights_only=False)
    original_indices = meta["poison_indices_in_non_target"].detach().cpu().numpy().astype(int)
    clean_all = meta["non_target_clean_data"].detach().cpu().numpy()
    attack_all = meta["poisoned_non_target_data"].detach().cpu().numpy()
    if len(original_indices) != len(attack_all):
        raise AssertionError("matched non-target index count does not equal triggered sample count")

    max_pairs = min(int(config["max_pairs_per_seed"]), len(original_indices))
    rng = np.random.default_rng(int(config["random_seed"]) + seed)
    selected_positions = np.sort(rng.choice(len(original_indices), size=max_pairs, replace=False))
    selected_indices = original_indices[selected_positions]
    clean = clean_all[selected_indices]
    attack = attack_all[selected_positions]

    trigger = meta["trigger_tensor"].detach().cpu().numpy()
    reconstructed = np.clip(clean + trigger, -3.0, 3.0)
    pair_error = float(np.max(np.abs(reconstructed - attack)))
    if pair_error > 1e-6:
        raise AssertionError(f"clean/trigger pairing reconstruction error={pair_error:.3e}")

    pair_ids = np.asarray([f"seed{seed}_nt{idx}" for idx in selected_indices])
    audit = {
        "seed": seed,
        "available_pairs": int(len(original_indices)),
        "selected_pairs": int(max_pairs),
        "pair_reconstruction_max_abs_error": pair_error,
        "target_class": int(meta["target_class"]),
        "non_target_class": int(meta["non_target_class"]),
    }
    return clean, attack, pair_ids, audit


def split_pairs(pair_ids: np.ndarray, config: dict, seed: int) -> dict[str, str]:
    fractions = config["split_fractions"]
    if not np.isclose(sum(float(v) for v in fractions.values()), 1.0):
        raise ValueError("split fractions must sum to one")
    shuffled = np.asarray(pair_ids).copy()
    rng = np.random.default_rng(int(config["split_seed"]) + seed)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * float(fractions["train"])))
    n_validation = int(round(n * float(fractions["validation"])))
    n_train = min(n_train, n - 2)
    n_validation = min(n_validation, n - n_train - 1)
    mapping = {}
    for pair_id in shuffled[:n_train]:
        mapping[str(pair_id)] = "train"
    for pair_id in shuffled[n_train : n_train + n_validation]:
        mapping[str(pair_id)] = "validation"
    for pair_id in shuffled[n_train + n_validation :]:
        mapping[str(pair_id)] = "test"
    return mapping


def local_feature_indices(labels: Iterable[str]) -> dict[str, np.ndarray]:
    labels = list(labels)
    weights = np.asarray([sum(char != "I" for char in label) for label in labels])
    z = np.asarray(
        [i for i, label in enumerate(labels) if weights[i] == 1 and "Z" in label],
        dtype=int,
    )
    xyz = np.asarray([i for i in range(len(labels)) if weights[i] == 1], dtype=int)
    return {"Z": z, "XYZ": xyz, "Full": np.arange(len(labels), dtype=int)}


def classifier_predictions(features_z: np.ndarray, checkpoint: dict) -> np.ndarray:
    w1 = checkpoint["classifier.0.weight"].detach().cpu().numpy()
    b1 = checkpoint["classifier.0.bias"].detach().cpu().numpy()
    w2 = checkpoint["classifier.3.weight"].detach().cpu().numpy()
    b2 = checkpoint["classifier.3.bias"].detach().cpu().numpy()
    hidden = np.maximum(0.0, features_z @ w1.T + b1)
    logits = hidden @ w2.T + b2
    return np.argmax(logits, axis=1)


def make_estimator(detector: str, c_value: float, random_seed: int):
    # Centering preserves the Euclidean/orthogonal geometry of Pauli space;
    # per-coordinate variance scaling would destroy full-space unitary invariance.
    if detector == "logistic":
        model = LogisticRegression(
            C=c_value,
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            random_state=random_seed,
        )
    elif detector == "rbf_svm":
        model = SVC(C=c_value, kernel="rbf", gamma="scale")
    else:
        raise ValueError(f"unknown detector: {detector}")
    return Pipeline(
        [
            ("center", StandardScaler(with_mean=True, with_std=False)),
            ("model", model),
        ]
    )


def decision_scores(estimator, features: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(features), dtype=float)
    return np.asarray(estimator.predict_proba(features)[:, 1], dtype=float)


def fit_with_validation(
    detector: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    c_grid: list[float],
    random_seed: int,
):
    candidates = []
    for c_value in c_grid:
        estimator = make_estimator(detector, float(c_value), random_seed)
        estimator.fit(x_train, y_train)
        scores = decision_scores(estimator, x_validation)
        candidates.append((float(roc_auc_score(y_validation, scores)), float(c_value), estimator))
    # Tie-break toward stronger regularization (smaller C), fixed before test.
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2], candidates[0][1], candidates[0][0]


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, limit: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    eligible = tpr[fpr <= limit + 1e-15]
    return float(np.max(eligible)) if eligible.size else 0.0


def prediction_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    predicted = (scores >= 0.0).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "aucpr": float(average_precision_score(y_true, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "tpr_at_fpr_01": tpr_at_fpr(y_true, scores, 0.01),
        "tpr_at_fpr_05": tpr_at_fpr(y_true, scores, 0.05),
    }


def pair_groups(pair_ids: np.ndarray) -> list[np.ndarray]:
    unique = list(dict.fromkeys(str(value) for value in pair_ids))
    groups = [np.flatnonzero(pair_ids == pair_id) for pair_id in unique]
    if any(len(group) != 2 for group in groups):
        raise AssertionError("every held-out base_sample_id must have exactly two rows")
    return groups


def cluster_bootstrap_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
    pair_ids: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups = pair_groups(pair_ids)
    values = np.empty(repetitions, dtype=float)
    for rep in range(repetitions):
        chosen = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[index] for index in chosen])
        values[rep] = roc_auc_score(y_true[rows], scores[rows])
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def cluster_bootstrap_auc_delta(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    pair_ids: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    groups = pair_groups(pair_ids)
    values = np.empty(repetitions, dtype=float)
    for rep in range(repetitions):
        chosen = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[index] for index in chosen])
        values[rep] = roc_auc_score(y_true[rows], scores_a[rows]) - roc_auc_score(
            y_true[rows], scores_b[rows]
        )
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "delta_auc": float(roc_auc_score(y_true, scores_a) - roc_auc_score(y_true, scores_b)),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def paired_permutation_pvalue(
    y_true: np.ndarray,
    scores: np.ndarray,
    pair_ids: np.ndarray,
    repetitions: int,
    seed: int,
) -> float:
    observed = float(roc_auc_score(y_true, scores))
    rng = np.random.default_rng(seed)
    groups = pair_groups(pair_ids)
    null = np.empty(repetitions, dtype=float)
    for rep in range(repetitions):
        permuted = y_true.copy()
        for group in groups:
            if rng.random() < 0.5:
                permuted[group] = permuted[group[::-1]]
        null[rep] = roc_auc_score(permuted, scores)
    return float((1 + np.sum(null >= observed)) / (1 + repetitions))


def extract_or_load_features(
    config: dict,
    output: Path,
    seed: int,
    victim_kind: str,
    checkpoint_name: str,
    checkpoint_layers: int,
    clean_samples: np.ndarray,
    attack_samples: np.ndarray,
    pair_ids: np.ndarray,
    actions: PauliActions,
    force: bool,
) -> dict:
    cache_dir = output / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_kind = "shared" if checkpoint_layers == 0 else victim_kind
    cache_path = cache_dir / f"seed{seed}_{cache_kind}_{checkpoint_name}_pauli.npz"
    if cache_path.exists() and not force:
        cached = np.load(cache_path, allow_pickle=False)
        return {key: cached[key] for key in cached.files}

    paths = source_paths(config, seed)
    state_dict = torch.load(paths[victim_kind], map_location="cpu", weights_only=False)
    circuit = build_state_circuit(
        state_dict["weights"].detach().cpu().numpy(),
        int(config["n_qubits"]),
        int(config["n_layers"]),
        checkpoint_layers,
    )
    samples = np.concatenate([clean_samples, attack_samples], axis=0)
    states = extract_states(samples, circuit)
    state_norm_error = float(np.max(np.abs(np.sum(np.abs(states) ** 2, axis=1) - 1.0)))
    features = pauli_expectations(states, actions).astype(np.float32)
    purity_target = float(2 ** int(config["n_qubits"]) - 1)
    purity_error = float(np.max(np.abs(np.sum(features.astype(float) ** 2, axis=1) - purity_target)))
    n_pairs = len(pair_ids)
    payload = {
        "clean_full": features[:n_pairs],
        "attack_full": features[n_pairs:],
        "pair_ids": pair_ids.astype("U"),
        "state_norm_max_abs_error": np.asarray(state_norm_error),
        "pauli_purity_max_abs_error": np.asarray(purity_error),
    }
    np.savez_compressed(cache_path, **payload)
    return payload


def eligibility_record(config: dict, seed: int) -> dict:
    path = source_paths(config, seed)["eligibility"]
    source = load_json(path)
    key = f"{float(config['poison_ratio']):.1f}"
    metrics = source["aggregated_results"][key]
    return {
        "seed": seed,
        "clean_accuracy_percent": float(metrics["clean_accuracy"]["mean"]),
        "attack_success_rate_percent": float(metrics["attack_success_rate"]["mean"]),
        "eligible_ca_ge_90": bool(float(metrics["clean_accuracy"]["mean"]) >= 90.0),
        "eligible_asr_ge_80": bool(float(metrics["attack_success_rate"]["mean"]) >= 80.0),
        "source": str(path.relative_to(PROJECT_ROOT)),
        "source_sha256": sha256_file(path),
    }


def run(config_path: Path, output: Path, force: bool) -> None:
    started = time.time()
    config = load_json(config_path)
    output.mkdir(parents=True, exist_ok=True)
    json_dump(output / "resolved_config.json", config)

    n_qubits = int(config["n_qubits"])
    actions = build_pauli_actions(n_qubits)
    indices = local_feature_indices(actions.labels)
    if len(indices["Z"]) != n_qubits or len(indices["XYZ"]) != 3 * n_qubits:
        raise AssertionError("local Z/XYZ basis dimensions are incorrect")

    eligibility_rows = [eligibility_record(config, int(seed)) for seed in config["seeds"]]
    if not all(row["eligible_ca_ge_90"] and row["eligible_asr_ge_80"] for row in eligibility_rows):
        raise RuntimeError("one or more frozen victims fail the preregistered CA/ASR gate")
    csv_write(output / "victim_eligibility.csv", eligibility_rows)

    split_rows: list[dict] = []
    pair_audit_rows: list[dict] = []
    metric_rows: list[dict] = []
    score_rows: list[dict] = []
    activation_rows: list[dict] = []
    feature_store: dict[tuple, dict] = {}

    for seed_value in config["seeds"]:
        seed = int(seed_value)
        clean_samples, attack_samples, pair_ids, pair_audit = load_matched_pairs(config, seed)
        pair_audit_rows.append(pair_audit)
        split_map = split_pairs(pair_ids, config, seed)
        split_rows.extend(
            {"seed": seed, "base_sample_id": str(pair_id), "split": split_map[str(pair_id)]}
            for pair_id in pair_ids
        )

        for victim_kind in ("backdoor", "clean"):
            for checkpoint_name, checkpoint_value in config["checkpoints"].items():
                checkpoint_layers = int(checkpoint_value)
                payload = extract_or_load_features(
                    config,
                    output,
                    seed,
                    victim_kind,
                    checkpoint_name,
                    checkpoint_layers,
                    clean_samples,
                    attack_samples,
                    pair_ids,
                    actions,
                    force,
                )
                feature_store[(seed, victim_kind, checkpoint_name)] = payload
                full = np.concatenate([payload["clean_full"], payload["attack_full"]], axis=0)
                labels = np.concatenate(
                    [np.zeros(len(pair_ids), dtype=int), np.ones(len(pair_ids), dtype=int)]
                )
                row_pair_ids = np.concatenate([pair_ids, pair_ids])
                row_splits = np.asarray([split_map[str(pair_id)] for pair_id in row_pair_ids])

                if checkpoint_name == "output":
                    checkpoint = torch.load(
                        source_paths(config, seed)[victim_kind],
                        map_location="cpu",
                        weights_only=False,
                    )
                    predictions = classifier_predictions(full[:, indices["Z"]], checkpoint)
                    clean_predictions = predictions[: len(pair_ids)]
                    attack_predictions = predictions[len(pair_ids) :]
                    activation_rows.append(
                        {
                            "seed": seed,
                            "victim_kind": victim_kind,
                            "n_pairs": len(pair_ids),
                            "clean_non_target_rate": float(np.mean(clean_predictions == 0)),
                            "triggered_target_rate": float(np.mean(attack_predictions == 1)),
                        }
                    )

                for representation, feature_idx in indices.items():
                    x = full[:, feature_idx]
                    masks = {name: row_splits == name for name in ("train", "validation", "test")}
                    for detector in config["detectors"]:
                        estimator, selected_c, validation_auc = fit_with_validation(
                            detector,
                            x[masks["train"]],
                            labels[masks["train"]],
                            x[masks["validation"]],
                            labels[masks["validation"]],
                            [float(value) for value in config["c_grid"]],
                            int(config["random_seed"]) + seed,
                        )
                        test_scores = decision_scores(estimator, x[masks["test"]])
                        test_y = labels[masks["test"]]
                        test_pair_ids = row_pair_ids[masks["test"]]
                        metrics = prediction_metrics(test_y, test_scores)
                        ci_low, ci_high = cluster_bootstrap_auc(
                            test_y,
                            test_scores,
                            test_pair_ids,
                            int(config["bootstrap_repetitions"]),
                            int(config["random_seed"]) + seed * 100 + len(metric_rows),
                        )
                        permutation_p = paired_permutation_pvalue(
                            test_y,
                            test_scores,
                            test_pair_ids,
                            int(config["permutation_repetitions"]),
                            int(config["random_seed"]) + seed * 1000 + len(metric_rows),
                        )
                        metric_rows.append(
                            {
                                "seed": seed,
                                "victim_kind": victim_kind,
                                "checkpoint": checkpoint_name,
                                "checkpoint_layers": checkpoint_layers,
                                "representation": representation,
                                "dimension": int(len(feature_idx)),
                                "detector": detector,
                                "selected_c": selected_c,
                                "validation_auc": validation_auc,
                                "test_pairs": int(len(np.unique(test_pair_ids))),
                                **metrics,
                                "auc_ci_low": ci_low,
                                "auc_ci_high": ci_high,
                                "paired_permutation_p": permutation_p,
                            }
                        )
                        test_indices = np.flatnonzero(masks["test"])
                        for local_row, global_row in enumerate(test_indices):
                            score_rows.append(
                                {
                                    "seed": seed,
                                    "victim_kind": victim_kind,
                                    "checkpoint": checkpoint_name,
                                    "representation": representation,
                                    "detector": detector,
                                    "base_sample_id": str(row_pair_ids[global_row]),
                                    "label": int(labels[global_row]),
                                    "score": float(test_scores[local_row]),
                                }
                            )

    csv_write(output / "pair_audit.csv", pair_audit_rows)
    csv_write(output / "splits.csv", split_rows)
    csv_write(output / "victim_activation_on_study_pairs.csv", activation_rows)
    csv_write(output / "heldout_metrics.csv", metric_rows)
    csv_write(output / "heldout_scores.csv", score_rows)

    invariance_rows = []
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        for victim_kind in ("backdoor", "clean"):
            reference_payload = feature_store[(seed, victim_kind, "encoding")]
            reference = np.concatenate(
                [reference_payload["clean_full"], reference_payload["attack_full"]], axis=0
            ).astype(float)
            reference_gram = reference @ reference.T
            for checkpoint_name in config["checkpoints"]:
                payload = feature_store[(seed, victim_kind, checkpoint_name)]
                features = np.concatenate(
                    [payload["clean_full"], payload["attack_full"]], axis=0
                ).astype(float)
                gram = features @ features.T
                difference = gram - reference_gram
                invariance_rows.append(
                    {
                        "seed": seed,
                        "victim_kind": victim_kind,
                        "checkpoint": checkpoint_name,
                        "state_norm_max_abs_error": float(payload["state_norm_max_abs_error"]),
                        "pauli_purity_max_abs_error": float(payload["pauli_purity_max_abs_error"]),
                        "full_pauli_gram_max_abs_delta_vs_encoding": float(
                            np.max(np.abs(difference))
                        ),
                        "full_pauli_gram_relative_fro_delta_vs_encoding": float(
                            np.linalg.norm(difference) / max(np.linalg.norm(reference_gram), 1e-15)
                        ),
                    }
                )
    csv_write(output / "unitary_invariance_checks.csv", invariance_rows)

    score_lookup = {}
    for row in score_rows:
        key = (
            int(row["seed"]),
            row["victim_kind"],
            row["checkpoint"],
            row["representation"],
            row["detector"],
        )
        score_lookup.setdefault(key, []).append(row)

    comparison_rows = []
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        for victim_kind in ("backdoor", "clean"):
            for checkpoint_name in config["checkpoints"]:
                for detector in config["detectors"]:
                    full_rows = score_lookup[(seed, victim_kind, checkpoint_name, "Full", detector)]
                    xyz_rows = score_lookup[(seed, victim_kind, checkpoint_name, "XYZ", detector)]
                    full_rows = sorted(full_rows, key=lambda row: (row["base_sample_id"], row["label"]))
                    xyz_rows = sorted(xyz_rows, key=lambda row: (row["base_sample_id"], row["label"]))
                    identifiers = [(row["base_sample_id"], row["label"]) for row in full_rows]
                    if identifiers != [(row["base_sample_id"], row["label"]) for row in xyz_rows]:
                        raise AssertionError("Full and XYZ held-out rows are not aligned")
                    y = np.asarray([int(row["label"]) for row in full_rows])
                    pairs = np.asarray([row["base_sample_id"] for row in full_rows])
                    full_scores = np.asarray([float(row["score"]) for row in full_rows])
                    xyz_scores = np.asarray([float(row["score"]) for row in xyz_rows])
                    comparison = cluster_bootstrap_auc_delta(
                        y,
                        full_scores,
                        xyz_scores,
                        pairs,
                        int(config["bootstrap_repetitions"]),
                        int(config["random_seed"]) + seed * 10_000 + len(comparison_rows),
                    )
                    comparison_rows.append(
                        {
                            "seed": seed,
                            "victim_kind": victim_kind,
                            "checkpoint": checkpoint_name,
                            "detector": detector,
                            "comparison": "Full_minus_XYZ",
                            **comparison,
                        }
                    )

        for representation in ("Z", "XYZ", "Full"):
            for detector in config["detectors"]:
                backdoor_rows = sorted(
                    score_lookup[(seed, "backdoor", "output", representation, detector)],
                    key=lambda row: (row["base_sample_id"], row["label"]),
                )
                clean_rows = sorted(
                    score_lookup[(seed, "clean", "output", representation, detector)],
                    key=lambda row: (row["base_sample_id"], row["label"]),
                )
                identifiers = [(row["base_sample_id"], row["label"]) for row in backdoor_rows]
                if identifiers != [(row["base_sample_id"], row["label"]) for row in clean_rows]:
                    raise AssertionError("backdoor and clean-model held-out rows are not aligned")
                y = np.asarray([int(row["label"]) for row in backdoor_rows])
                pairs = np.asarray([row["base_sample_id"] for row in backdoor_rows])
                backdoor_scores = np.asarray([float(row["score"]) for row in backdoor_rows])
                clean_scores = np.asarray([float(row["score"]) for row in clean_rows])
                comparison = cluster_bootstrap_auc_delta(
                    y,
                    backdoor_scores,
                    clean_scores,
                    pairs,
                    int(config["bootstrap_repetitions"]),
                    int(config["random_seed"]) + seed * 20_000 + len(comparison_rows),
                )
                comparison_rows.append(
                    {
                        "seed": seed,
                        "victim_kind": "backdoor_minus_clean_control",
                        "checkpoint": "output",
                        "detector": detector,
                        "comparison": f"{representation}_backdoor_minus_clean",
                        **comparison,
                    }
                )
    csv_write(output / "paired_auc_comparisons.csv", comparison_rows)

    primary = [
        row
        for row in metric_rows
        if row["victim_kind"] == "backdoor"
        and row["checkpoint"] == "output"
        and row["detector"] == "logistic"
    ]
    primary_by_seed_rep = {(int(row["seed"]), row["representation"]): row for row in primary}
    full_auc = [primary_by_seed_rep[(int(seed), "Full")]["auc"] for seed in config["seeds"]]
    xyz_auc = [primary_by_seed_rep[(int(seed), "XYZ")]["auc"] for seed in config["seeds"]]
    deltas = [full - xyz for full, xyz in zip(full_auc, xyz_auc)]
    if min(full_auc) >= 0.85 and min(deltas) >= 0.10:
        decision = "GO_A"
        rationale = "Full AUC >= 0.85 and Full-XYZ >= 0.10 for every pilot seed."
    elif min(full_auc) >= 0.75:
        decision = "GO_B"
        rationale = "Full-space discriminability exists, but the Full-XYZ gap is not uniformly >= 0.10."
    elif max(full_auc) <= 0.60:
        decision = "NO_GO_STATIC"
        rationale = "Full-space linear discriminability is at or below the preregistered no-go range."
    else:
        decision = "AMBIGUOUS"
        rationale = "Pilot results fall between the preregistered engineering gates."
    json_dump(
        output / "phase_a_gate_decision.json",
        {
            "decision": decision,
            "rationale": rationale,
            "primary_analysis": "backdoor/output/logistic",
            "full_auc_by_seed": {str(seed): float(value) for seed, value in zip(config["seeds"], full_auc)},
            "xyz_auc_by_seed": {str(seed): float(value) for seed, value in zip(config["seeds"], xyz_auc)},
            "full_minus_xyz_by_seed": {
                str(seed): float(value) for seed, value in zip(config["seeds"], deltas)
            },
            "interpretation_guardrail": (
                "The clean-model trigger control and unitary invariance checks must be considered before "
                "attributing any Full-space signal to learned backdoor dynamics."
            ),
        },
    )

    manifest = {
        "experiment_id": config["experiment_id"],
        "config": str(config_path.resolve()),
        "project_root": str(PROJECT_ROOT),
        "elapsed_seconds": round(time.time() - started, 3),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "pennylane": qml.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "pauli_dimensions": {
            "Z": int(len(indices["Z"])),
            "XYZ": int(len(indices["XYZ"])),
            "Full": int(len(indices["Full"])),
        },
        "source_checkpoints": [
            {
                "seed": int(seed),
                "kind": kind,
                "path": str(source_paths(config, int(seed))[kind].relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(source_paths(config, int(seed))[kind]),
            }
            for seed in config["seeds"]
            for kind in ("backdoor", "clean")
        ],
    }
    json_dump(output / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "decision": decision, "output": str(output)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="recompute cached state/Pauli features")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config.resolve(), args.output.resolve(), args.force)
