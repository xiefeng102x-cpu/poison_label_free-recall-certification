from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import analyze_auditability_geometry_diagnostic as diagnostic  # noqa: E402


def test_distribution_distances_are_zero_for_identity_and_positive_for_shift() -> None:
    values = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float
    )
    assert diagnostic.energy_distance_v(values, values) == 0.0
    assert diagnostic.rbf_mmd2_biased(values, values)[0] == 0.0
    assert diagnostic.energy_distance_v(values, values + 2.0) > 1.0
    assert diagnostic.rbf_mmd2_biased(values, values + 2.0)[0] > 0.1


def test_symmetric_poison_cloud_has_lower_resultant_length() -> None:
    rng = np.random.default_rng(17)
    clean = rng.normal(scale=0.2, size=(200, 4))
    one_direction = rng.normal(scale=0.1, size=(60, 4)) + np.asarray([2.0, 0.0, 0.0, 0.0])
    symmetric = np.concatenate(
        [
            rng.normal(scale=0.1, size=(30, 4)) + np.asarray([2.0, 0.0, 0.0, 0.0]),
            rng.normal(scale=0.1, size=(30, 4)) - np.asarray([2.0, 0.0, 0.0, 0.0]),
        ],
        axis=0,
    )
    fixed = {
        "ridge_fraction_of_mean_clean_variance": 0.1,
        "cross_validation_folds": 5,
        "cross_validation_seed": 20261002,
        "rbf_svc_c": 1.0,
    }
    one = diagnostic.geometry_metrics(clean, one_direction, fixed)
    multi = diagnostic.geometry_metrics(clean, symmetric, fixed)
    assert one["mean_resultant_length"] > 0.95
    assert multi["mean_resultant_length"] < 0.15
    assert multi["rbf_mmd2_biased"] > 0.0


def test_crossfit_auc_and_seed_bootstrap_are_deterministic() -> None:
    rng = np.random.default_rng(91)
    clean = rng.normal(size=(120, 5))
    poison = rng.normal(loc=1.0, size=(30, 5))
    first = diagnostic.crossfit_aucs(
        clean,
        poison,
        folds=5,
        random_seed=20261002,
        ridge_fraction=0.1,
        svc_c=1.0,
    )
    second = diagnostic.crossfit_aucs(
        clean,
        poison,
        folds=5,
        random_seed=20261002,
        ridge_fraction=0.1,
        svc_c=1.0,
    )
    assert first == second
    assert all(0.5 < value <= 1.0 for value in first)
    values = np.asarray([0.7, 0.8, 0.9])
    assert diagnostic.bootstrap_mean_ci(values, 1000, 123, 0.95) == diagnostic.bootstrap_mean_ci(
        values, 1000, 123, 0.95
    )


def test_frozen_decision_rejects_the_intended_cancellation_mechanism() -> None:
    decision = json.loads(
        (
            PAPER14
            / "results/auditability_geometry_diagnostic_v1/decision.json"
        ).read_text(encoding="utf-8")
    )
    assert decision["label"] == "NO_CLEAN_CANCELLATION_DIAGNOSIS"
    assert decision["first_order_cancellation_gate"] is False
    assert decision["residual_nonlinear_information_gate"] is True
    assert decision["common_eligible_seeds"] == [55, 56, 58, 60, 61]
