from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_bloodmnist_fresh_query_calibration import (  # noqa: E402
    alternating_folds,
    replay_counts,
    split_design,
    unit_direction,
)


def test_alternating_folds_are_deterministic_disjoint_and_complete() -> None:
    values = np.arange(101, dtype=np.int64)
    first = alternating_folds(values, 1234)
    second = alternating_folds(values, 1234)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert np.intersect1d(first[0], first[1]).size == 0
    assert np.array_equal(np.sort(np.concatenate(first)), values)
    assert abs(len(first[0]) - len(first[1])) <= 1


def test_unit_direction_normalizes_mean_shift() -> None:
    values = np.asarray([[2.0, 0.0], [4.0, 0.0]])
    direction, norm = unit_direction(values, np.asarray([1.0, 0.0]))
    assert norm == pytest.approx(2.0)
    assert np.allclose(direction, np.asarray([1.0, 0.0]))
    with pytest.raises(RuntimeError):
        unit_direction(np.ones((2, 3)), np.ones(3))


def test_frozen_split_design_preserves_population_provenance() -> None:
    original = np.r_[np.zeros(600, dtype=np.int64), np.full(2330, 6, dtype=np.int64)]
    assigned = np.full(2930, 6, dtype=np.int64)
    poison = np.r_[np.ones(600, dtype=bool), np.zeros(2330, dtype=bool)]
    pool = {
        "original_labels": original,
        "assigned_labels": assigned,
        "is_poison": poison,
    }
    validation = np.full(333, 6, dtype=np.int64)
    test = np.full(666, 6, dtype=np.int64)
    design = split_design(pool, validation, test, seed=74)
    assert len(design["reference"]) == 96
    assert len(design["calibration"]) == 96
    assert np.intersect1d(design["reference"], design["calibration"]).size == 0
    assert sum(len(values) for values in design["candidate_folds"]) == 2930
    assert sum(len(values) for values in design["member_folds"]) == 2330
    assert sum(len(values) for values in design["fresh_folds"]) == 666


def test_replay_counts_detects_exact_flags() -> None:
    query_rows = []
    for population, values in (("member_train_clean", [0, 1, 0]), ("fresh_test_clean", [1, 0])):
        for flagged in values:
            query_rows.append(
                {"seed": 74, "view": "suspect_hidden64", "population": population, "flagged": flagged}
            )
    metrics = [
        {
            "seed": 74,
            "view": "suspect_hidden64",
            "member_count": 3,
            "member_flagged": 1,
            "member_fpr": 1 / 3,
            "fresh_count": 2,
            "fresh_flagged": 1,
            "fresh_fpr": 0.5,
        }
    ]
    assert replay_counts(query_rows, metrics)
    metrics[0]["fresh_flagged"] = 0
    assert not replay_counts(query_rows, metrics)
