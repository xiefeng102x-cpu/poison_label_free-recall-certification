from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_covariance_free_adaptive_grid as grid


def record(seed, backdoor, evasion, distill, ca, asr, tpr, fpr=0.03):
    return {
        "seed": seed,
        "backdoor_weight": backdoor,
        "evasion_weight": evasion,
        "distill_weight": distill,
        "validation_balanced_ca": ca,
        "validation_asr": asr,
        "covariance_free_tpr": tpr,
        "covariance_free_fpr": fpr,
    }


def test_rank_prefers_more_eligible_seeds_then_lower_tpr():
    a = grid.ParameterCombo(10, 5, 20)
    b = grid.ParameterCombo(20, 5, 20)
    rows_a = [record(1, *a, 0.95, 0.85, 0.70), record(2, *a, 0.95, 0.70, 0.10)]
    rows_b = [record(1, *b, 0.95, 0.85, 0.80), record(2, *b, 0.95, 0.85, 0.80)]
    ranked = grid.rank_combos([
        grid.summarize_combo(a, rows_a),
        grid.summarize_combo(b, rows_b),
    ])
    assert ranked[0]["backdoor_weight"] == 20
    assert ranked[0]["eligible_seed_count"] == 2


def test_zero_eligible_uses_worst_normalized_gate_margin():
    a = grid.ParameterCombo(10, 5, 20)
    b = grid.ParameterCombo(20, 5, 20)
    rows_a = [record(1, *a, 0.95, 0.79, 0.20)]
    rows_b = [record(1, *b, 0.95, 0.70, 0.10)]
    ranked = grid.rank_combos([
        grid.summarize_combo(b, rows_b),
        grid.summarize_combo(a, rows_a),
    ])
    assert ranked[0]["backdoor_weight"] == 10


def test_parameter_tuple_breaks_exact_tie_deterministically():
    a = grid.ParameterCombo(10, 5, 20)
    b = grid.ParameterCombo(20, 5, 20)
    rows_a = [record(1, *a, 0.95, 0.85, 0.70)]
    rows_b = [record(1, *b, 0.95, 0.85, 0.70)]
    ranked = grid.rank_combos([
        grid.summarize_combo(b, rows_b),
        grid.summarize_combo(a, rows_a),
    ])
    assert [row["backdoor_weight"] for row in ranked] == [10, 20]


def test_load_state_supports_resume_and_rejects_malformed(tmp_path):
    path = tmp_path / "grid.json"
    payload = {"metadata": {"x": 1}, "runs": [{"seed": 52}]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert grid.load_state(path) == payload
    path.write_text(json.dumps({"runs": {"seed": 52}}), encoding="utf-8")
    with pytest.raises(ValueError):
        grid.load_state(path)

