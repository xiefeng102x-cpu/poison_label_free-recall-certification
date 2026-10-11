from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import analyze_checkpoint_auditability_theory_probe as probe


def test_pauli_indices_have_expected_dimensions():
    indices = probe.pauli_indices(5)
    assert {name: len(values) for name, values in indices.items()} == {
        "Z": 5,
        "XYZ": 15,
        "Full": 1023,
    }


def test_ridge_discriminant_orients_toward_attack_mean():
    clean = np.asarray([[0.0, -1.0], [0.0, 1.0], [0.1, 0.0]])
    attack = clean + np.asarray([2.0, 0.0])
    fitted = probe.ridge_discriminant(clean, attack)
    clean_scores = probe.score(clean, fitted)
    attack_scores = probe.score(attack, fitted)
    assert fitted["ridge_effect_size"] > 0
    assert attack_scores.mean() > clean_scores.mean()


def test_rank_pvalues_use_finite_sample_plus_one_rule():
    calibration = np.asarray([1.0, 2.0, 3.0])
    query = np.asarray([4.0, 2.0, 0.0])
    assert probe.rank_pvalues(calibration, query).tolist() == [0.25, 0.75, 1.0]


def test_safe_cosine_handles_zero_and_matching_vectors():
    assert probe.safe_cosine(np.zeros(2), np.ones(2)) == 0.0
    assert probe.safe_cosine(np.asarray([1.0, 2.0]), np.asarray([2.0, 4.0])) == pytest.approx(1.0)


def test_paired_bootstrap_delta_is_deterministic_and_positive():
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    absolute = np.asarray([0.2, 0.4, 0.6, 0.3, 0.5, 0.7])
    drift = np.asarray([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    first = probe.paired_bootstrap_auc_delta(
        labels, drift, absolute, 3, repetitions=50, seed=9
    )
    second = probe.paired_bootstrap_auc_delta(
        labels, drift, absolute, 3, repetitions=50, seed=9
    )
    assert first == second
    assert first["auc_delta"] > 0


def test_analyze_view_keeps_test_out_of_fit():
    clean = np.arange(18, dtype=float).reshape(9, 2) / 20.0
    attack = clean + np.asarray([0.5, 0.0])
    masks = {
        "train": np.asarray([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool),
        "validation": np.asarray([0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=bool),
        "test": np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool),
    }
    metrics, scores = probe.analyze_view(
        clean, attack, masks, alpha=0.05, ridge_fraction=0.1
    )
    assert metrics["test_auc"] == pytest.approx(1.0)
    assert scores["test_clean"].shape == (3,)
