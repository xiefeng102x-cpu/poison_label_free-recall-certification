"""Portable NumPy state-vector evaluator used by the paper14 replay code.

This is the minimal executable dependency previously loaded from the sibling
``qml_adaptive_nc_project`` tree.  The implementation is preserved from the
historical source whose SHA-256 is
``471a052ca4c755015ac4ca30e0799fdac18a58ea7edd52b8828ca0435d39058d``.
Keeping the replay copy inside paper14 makes the reviewer package independent
of the author's surrounding workspace.
"""

from __future__ import annotations

import itertools

import numpy as np


N_QUBITS = 5
ZZ_PAIRS = list(itertools.combinations(range(N_QUBITS), 2))


def _apply_single_qubit(
    psi: np.ndarray,
    gate: np.ndarray,
    qubit: int,
) -> np.ndarray:
    """Apply a 2-by-2 gate to one qubit in a batch of state vectors."""
    dim = psi.shape[1]
    step = dim >> (qubit + 1)
    groups = dim // (2 * step)
    psi_r = psi.reshape(-1, groups, 2, step)
    out = np.einsum("ij,bkjl->bkil", gate, psi_r, optimize=True)
    return out.reshape(-1, dim)


def _apply_cnot(psi: np.ndarray, ctrl: int, tgt: int) -> np.ndarray:
    """Apply a CNOT permutation using PennyLane's most-significant-bit order."""
    dim = psi.shape[1]
    n_qubits = int(np.log2(dim))
    ctrl_bit = n_qubits - 1 - ctrl
    tgt_bit = n_qubits - 1 - tgt
    states = np.arange(dim, dtype=np.int64)
    ctrl_on = (states >> ctrl_bit) & 1
    source = np.where(ctrl_on, states ^ (1 << tgt_bit), states)
    return psi[:, source]


_eval_cache: dict[int, tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]] = {}


def _make_evals(
    n_qubits: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Return cached Z and ZZ eigenvalue tables for ``n_qubits``."""
    if n_qubits in _eval_cache:
        return _eval_cache[n_qubits]

    zz_pairs = list(itertools.combinations(range(n_qubits), 2))
    dim = 2**n_qubits
    states = np.arange(dim, dtype=np.int64)

    z_evals = np.zeros((n_qubits, dim), dtype=np.float64)
    for qubit in range(n_qubits):
        bit = (states >> (n_qubits - 1 - qubit)) & 1
        z_evals[qubit] = 1 - 2 * bit

    zz_evals = np.zeros((len(zz_pairs), dim), dtype=np.float64)
    for index, (left, right) in enumerate(zz_pairs):
        bit_left = (states >> (n_qubits - 1 - left)) & 1
        bit_right = (states >> (n_qubits - 1 - right)) & 1
        zz_evals[index] = (1 - 2 * bit_left) * (1 - 2 * bit_right)

    result = (z_evals, zz_evals, zz_pairs)
    _eval_cache[n_qubits] = result
    return result


_Z_EVALS, _ZZ_EVALS, _ = _make_evals(N_QUBITS)
_H_GATE = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
_HSDAG = np.array([[1, -1j], [1, 1j]], dtype=np.complex128) / np.sqrt(2)


def run_circuit_batch(
    x_batch: np.ndarray,
    weights: np.ndarray,
    n_layers: int,
    n_qubits: int = N_QUBITS,
) -> np.ndarray:
    """Evaluate Z and pairwise-Z observables for a batch of QNN inputs."""
    norms = np.linalg.norm(x_batch, axis=1, keepdims=True).clip(min=1e-12)
    psi = (x_batch / norms).astype(np.complex128)
    z_evals, zz_evals, _ = _make_evals(n_qubits)
    reshaped = weights.reshape(n_layers, n_qubits, 2)

    for layer in range(n_layers):
        for qubit in range(n_qubits):
            theta, phi = (
                float(reshaped[layer, qubit, 0]),
                float(reshaped[layer, qubit, 1]),
            )
            cosine = np.cos(theta / 2)
            sine = np.sin(theta / 2)
            ry = np.array(
                [[cosine, -sine], [sine, cosine]],
                dtype=np.complex128,
            )
            psi = _apply_single_qubit(psi, ry, qubit)

            phase = np.exp(1j * phi / 2)
            rz = np.array(
                [[1 / phase, 0], [0, phase]],
                dtype=np.complex128,
            )
            psi = _apply_single_qubit(psi, rz, qubit)

        for qubit in range(n_qubits):
            psi = _apply_cnot(psi, qubit, (qubit + 1) % n_qubits)

    probabilities = np.abs(psi) ** 2
    z_values = probabilities @ z_evals.T
    zz_values = probabilities @ zz_evals.T
    return np.concatenate([z_values, zz_values], axis=1).astype(np.float32)
