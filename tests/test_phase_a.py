from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "phase_a_full_observability.py"
SPEC = importlib.util.spec_from_file_location("paper14_phase_a", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def dense_pauli(label: str) -> np.ndarray:
    matrices = {
        "I": np.eye(2, dtype=complex),
        "X": np.asarray([[0, 1], [1, 0]], dtype=complex),
        "Y": np.asarray([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.asarray([[1, 0], [0, -1]], dtype=complex),
    }
    out = np.asarray([[1.0 + 0.0j]])
    for char in label:
        out = np.kron(out, matrices[char])
    return out


def test_pauli_basis_counts_and_local_dimensions():
    for n_qubits in (1, 2, 3, 5):
        actions = MODULE.build_pauli_actions(n_qubits)
        assert len(actions.labels) == 4**n_qubits - 1
        indices = MODULE.local_feature_indices(actions.labels)
        assert len(indices["Z"]) == n_qubits
        assert len(indices["XYZ"]) == 3 * n_qubits


def test_fast_pauli_expectations_match_dense_reference():
    rng = np.random.default_rng(7)
    n_qubits = 3
    dim = 2**n_qubits
    states = rng.normal(size=(4, dim)) + 1j * rng.normal(size=(4, dim))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    actions = MODULE.build_pauli_actions(n_qubits)
    fast = MODULE.pauli_expectations(states, actions, feature_chunk_size=7)
    dense = np.asarray(
        [
            [np.vdot(state, dense_pauli(label) @ state).real for label in actions.labels]
            for state in states
        ]
    )
    np.testing.assert_allclose(fast, dense, atol=1e-12, rtol=1e-12)


def test_pure_state_pauli_norm_identity():
    rng = np.random.default_rng(11)
    n_qubits = 4
    dim = 2**n_qubits
    states = rng.normal(size=(5, dim)) + 1j * rng.normal(size=(5, dim))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    actions = MODULE.build_pauli_actions(n_qubits)
    features = MODULE.pauli_expectations(states, actions)
    np.testing.assert_allclose(
        np.sum(features**2, axis=1),
        np.full(states.shape[0], 2**n_qubits - 1),
        atol=1e-10,
        rtol=1e-10,
    )


def test_pair_split_has_no_cross_split_leakage():
    pair_ids = np.asarray([f"p{i}" for i in range(100)])
    config = {
        "split_seed": 19,
        "split_fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
    }
    mapping = MODULE.split_pairs(pair_ids, config, seed=42)
    assert set(mapping) == set(pair_ids)
    counts = {name: sum(value == name for value in mapping.values()) for name in set(mapping.values())}
    assert counts == {"train": 60, "validation": 20, "test": 20}


def test_cluster_groups_require_exact_matched_pairs():
    pair_ids = np.asarray(["a", "b", "a", "b"])
    groups = MODULE.pair_groups(pair_ids)
    assert [group.tolist() for group in groups] == [[0, 2], [1, 3]]
