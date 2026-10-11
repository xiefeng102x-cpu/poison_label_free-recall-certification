from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import fast_circuit  # noqa: E402


def dense_reference(
    x_batch: np.ndarray,
    weights: np.ndarray,
    n_layers: int,
    n_qubits: int,
) -> np.ndarray:
    """Small independent dense-matrix reference for the portable evaluator."""
    identity = np.eye(2, dtype=np.complex128)
    reshaped = weights.reshape(n_layers, n_qubits, 2)
    cnot_matrices: list[np.ndarray] = []
    dim = 2**n_qubits
    for control in range(n_qubits):
        target = (control + 1) % n_qubits
        matrix = np.zeros((dim, dim), dtype=np.complex128)
        for source in range(dim):
            control_bit = (source >> (n_qubits - 1 - control)) & 1
            destination = (
                source ^ (1 << (n_qubits - 1 - target))
                if control_bit
                else source
            )
            matrix[destination, source] = 1.0
        cnot_matrices.append(matrix)

    outputs: list[np.ndarray] = []
    for row in x_batch:
        psi = row.astype(np.complex128)
        psi /= max(np.linalg.norm(psi), 1e-12)
        for layer in range(n_layers):
            for qubit in range(n_qubits):
                theta, phi = reshaped[layer, qubit]
                cosine, sine = np.cos(theta / 2), np.sin(theta / 2)
                ry = np.array(
                    [[cosine, -sine], [sine, cosine]], dtype=np.complex128
                )
                phase = np.exp(1j * phi / 2)
                rz = np.array(
                    [[1 / phase, 0], [0, phase]], dtype=np.complex128
                )
                gate = np.array([[1.0 + 0j]])
                for position in range(n_qubits):
                    gate = np.kron(gate, rz @ ry if position == qubit else identity)
                psi = gate @ psi
            for matrix in cnot_matrices:
                psi = matrix @ psi

        probabilities = np.abs(psi) ** 2
        states = np.arange(dim, dtype=np.int64)
        z_columns: list[float] = []
        for qubit in range(n_qubits):
            bits = (states >> (n_qubits - 1 - qubit)) & 1
            z_columns.append(float(probabilities @ (1 - 2 * bits)))
        for left, right in itertools.combinations(range(n_qubits), 2):
            left_bits = (states >> (n_qubits - 1 - left)) & 1
            right_bits = (states >> (n_qubits - 1 - right)) & 1
            z_columns.append(
                float(probabilities @ ((1 - 2 * left_bits) * (1 - 2 * right_bits)))
            )
        outputs.append(np.asarray(z_columns, dtype=np.float32))
    return np.stack(outputs)


def test_portable_fast_circuit_matches_independent_dense_reference() -> None:
    rng = np.random.RandomState(20261003)
    inputs = rng.randn(3, 8)
    weights = rng.randn(2 * 3 * 2)
    observed = fast_circuit.run_circuit_batch(
        inputs,
        weights,
        n_layers=2,
        n_qubits=3,
    )
    expected = dense_reference(inputs, weights, n_layers=2, n_qubits=3)
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)
