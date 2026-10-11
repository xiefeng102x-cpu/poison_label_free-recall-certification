from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_bloodmnist_official_backdoorbench_pilot as pilot


def test_candidate_pool_replaces_poisoned_source_rows():
    splits = {
        "train": {
            "target": np.asarray([[10.0], [11.0]]),
            "non_target": np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        },
        "val": {
            "target": np.asarray([[12.0]]),
            "non_target": np.asarray([[4.0]]),
        },
        "archived_poison": np.asarray([[101.0], [103.0]]),
    }
    pool = pilot.candidate_pool(splits, {"poison_indices": np.asarray([1, 3])})
    assert pool["candidate_x"].reshape(-1).tolist() == [10, 11, 0, 2, 101, 103]
    assert pool["candidate_y"].tolist() == [1, 1, 0, 0, 1, 1]
    assert pool["truth"].tolist() == [False, False, False, False, True, True]


def test_evaluate_flags_reports_class_conditional_false_positives():
    flags = np.asarray([True, False, True, False, True])
    truth = np.asarray([False, False, False, False, True])
    kind = np.asarray(
        ["target_clean", "target_clean", "source_clean", "source_clean", "poison"]
    )
    row = pilot.evaluate_flags(flags, truth, kind)
    assert row["tp"] == 1
    assert row["fp"] == 2
    assert row["target_clean_fpr"] == 0.5
    assert row["source_clean_fpr"] == 0.5
