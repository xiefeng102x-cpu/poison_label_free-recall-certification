from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_bloodmnist_mechanism_ablations as original  # noqa: E402
from run_reference_calibration_budget_sensitivity import budget_crossfit  # noqa: E402


def synthetic_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(9182)
    trusted = rng.normal(size=(200, 16))
    clean = rng.normal(size=(60, 16))
    poison = rng.normal(loc=0.6, size=(12, 16))
    return trusted, clean, poison


def test_original_120_80_cell_is_bitwise_decision_compatible() -> None:
    trusted, clean, poison = synthetic_features()
    expected_row, expected_samples = original.custom_crossfit(
        67,
        "covariance_free_xz16",
        trusted,
        clean,
        poison,
        0.05,
        whitening=False,
    )
    actual_row, actual_samples = budget_crossfit(
        67, trusted, clean, poison, 120, 80, 0.05
    )
    for field in ("tp", "fn", "fp", "tn"):
        assert actual_row[field] == expected_row[field]
    for field in ("tpr", "fpr", "auc_crossfit_score", "auc_conformal_rank"):
        assert actual_row[field] == pytest.approx(expected_row[field], abs=1e-14)
    assert [row["flagged"] for row in actual_samples] == [
        row["flagged"] for row in expected_samples
    ]
    assert np.allclose(
        [row["p_value"] for row in actual_samples],
        [row["p_value"] for row in expected_samples],
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("calibration_count", "expected"),
    [(20, 1 / 21), (40, 2 / 41), (80, 4 / 81)],
)
def test_rank_grid_resolution_is_prespecified(
    calibration_count: int, expected: float
) -> None:
    trusted, clean, poison = synthetic_features()
    row, _ = budget_crossfit(
        67, trusted, clean, poison, 8, calibration_count, 0.05
    )
    assert row["rank_grid_null_rejection_probability"] == pytest.approx(expected)


def test_protocol_grid_and_pool_boundary_are_frozen() -> None:
    protocol_path = (
        ROOT
        / "results/top_journal_program_v1/reference_calibration_budget_sensitivity_protocol_v1.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["status"] == "preregistered_retrospective_checkpoint_replay_before_results"
    assert protocol["grid"]["reference_counts"] == [8, 16, 32, 64, 120]
    assert protocol["grid"]["calibration_counts"] == [20, 40, 80]
    assert protocol["method"]["reference_pool"].startswith("nested prefixes")
    assert "trusted_order[120" in protocol["method"]["calibration_pool"]


def test_invalid_budget_cannot_cross_frozen_pool_boundary() -> None:
    trusted, clean, poison = synthetic_features()
    with pytest.raises(ValueError):
        budget_crossfit(67, trusted, clean, poison, 121, 80, 0.05)
    with pytest.raises(ValueError):
        budget_crossfit(67, trusted, clean, poison, 120, 81, 0.05)
