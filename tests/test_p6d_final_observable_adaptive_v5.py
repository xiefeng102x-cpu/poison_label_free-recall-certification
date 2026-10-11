from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_p6d_final_observable_adaptive_v5 as v5


def test_confidence_rules_are_deterministic_unique_and_distinct():
    confidence = np.linspace(0.01, 0.99, 180)
    selections = [v5.confidence_selected_indices(confidence, rule) for rule in v5.CONFIDENCE_RULES]
    assert all(len(values) == 60 and len(np.unique(values)) == 60 for values in selections)
    assert np.array_equal(selections[0], np.arange(60))
    assert len({tuple(values) for values in selections}) == 3


def test_guarded_epoch_selection_prioritizes_utility_feasibility():
    rows = [
        {"epoch": 0, "validation_balanced_ca": 0.95, "validation_asr": 0.89,
         "tpr": 0.00, "mean_poison_p_value": 0.9},
        {"epoch": 1, "validation_balanced_ca": 0.93, "validation_asr": 0.91,
         "tpr": 0.40, "mean_poison_p_value": 0.5},
    ]
    selected = v5.guarded_select_epoch(rows)
    assert selected["epoch"] == 1
    assert selected["validation_eligible"] is True


def test_guarded_epoch_selection_rejects_test_leakage():
    rows = [{"epoch": 0, "validation_balanced_ca": 0.95, "validation_asr": 0.95,
             "tpr": 0.2, "mean_poison_p_value": 0.5, "test_asr": 0.9}]
    with pytest.raises(RuntimeError, match="Test firewall"):
        v5.guarded_select_epoch(rows)


def test_configuration_ranking_prioritizes_viability_margin_before_evasion():
    low_tpr_weak_margin = {
        "configuration_id": "hard60", "guarded_seed_count": 3,
        "worst_normalized_utility_margin": 0.1, "mean_tpr_guarded": 0.1,
        "worst_validation_asr": 0.91,
    }
    higher_tpr_strong_margin = {
        "configuration_id": "uncertain60", "guarded_seed_count": 3,
        "worst_normalized_utility_margin": 0.5, "mean_tpr_guarded": 0.4,
        "worst_validation_asr": 0.93,
    }
    assert v5.configuration_key(higher_tpr_strong_margin) < v5.configuration_key(low_tpr_weak_margin)
