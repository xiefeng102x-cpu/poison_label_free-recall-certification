from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import analyze_single_sample_reference_audit as audit  # noqa: E402


def protocol() -> dict:
    return json.loads(audit.PROTOCOL.read_text(encoding="utf-8"))


def test_rank_pvalues_are_one_sided_and_discrete() -> None:
    calibration = np.asarray([1.0, 2.0, 3.0, 4.0])
    query = np.asarray([0.0, 2.5, 5.0])
    observed = audit.rank_pvalues(calibration, query)
    np.testing.assert_allclose(observed, [1.0, 0.6, 0.2])
    assert np.all(np.diff(observed) < 0.0)


def test_three_way_bonferroni_with_100_calibrators_is_an_extreme_value_gate() -> None:
    rank_grid = np.arange(1, 102, dtype=float) / 101.0
    fused_grid = np.minimum(1.0, 3.0 * rank_grid)
    passing = fused_grid[fused_grid <= 0.05]
    assert passing.tolist() == [3.0 / 101.0]
    assert fused_grid[1] == 6.0 / 101.0
    assert fused_grid[0] < 0.05 < fused_grid[1]


def test_query_score_is_independent_of_other_queries() -> None:
    rng = np.random.default_rng(3)
    reference = rng.normal(size=(80, 6))
    state = audit.fit_reference_state(reference, ridge_fraction=0.1, neighbors=5)
    query = rng.normal(size=(1, 6))
    distractors = rng.normal(loc=20.0, size=(19, 6))
    alone = audit.score_queries(state, query)
    together = audit.score_queries(state, np.concatenate([query, distractors], axis=0))
    for scorer in audit.SCORERS:
        assert alone[scorer][0] == together[scorer][0]


def test_reference_only_fusion_flags_strong_outliers_without_high_clean_fpr() -> None:
    rng = np.random.default_rng(11)
    trusted = rng.normal(size=(160, 5))
    populations = {
        "member_clean": rng.normal(size=(200, 5)),
        "fresh_clean": rng.normal(size=(160, 5)),
        "poison": rng.normal(loc=4.0, size=(40, 5)),
    }
    outputs, _ = audit.calibrate_populations(
        trusted[:80], trusted[80:], populations, protocol()
    )
    metrics = audit.victim_metrics(outputs, 0.05)
    assert metrics["poison_tpr"] == 1.0
    assert metrics["member_clean_fpr"] <= 0.075
    assert metrics["fresh_clean_fpr"] <= 0.075
    assert metrics["poison_auc_against_member_clean"] > 0.98


def test_frozen_decision_has_strong_partial_and_negative_branches() -> None:
    p = protocol()
    modes = p["source"]["modes"]

    def rows(tpr: float, member: float, fresh: float) -> list[dict]:
        return [
            {
                "mode": mode,
                "population": "stored_test_eligible_victims_per_attack_mode",
                "victim_count": 5,
                "pooled_poison_tpr": tpr,
                "pooled_member_clean_fpr": member,
                "pooled_fresh_clean_fpr": fresh,
            }
            for mode in modes
        ]

    assert audit.decide(rows(0.91, 0.04, 0.04), p)["label"] == (
        "PROMISING_RETROSPECTIVE_SINGLE_SAMPLE_CHANNEL"
    )
    partial = rows(0.81, 0.07, 0.07)
    partial[-1]["pooled_poison_tpr"] = 0.5
    assert audit.decide(partial, p)["label"] == (
        "PARTIAL_RETROSPECTIVE_SINGLE_SAMPLE_CHANNEL"
    )
    assert audit.decide(rows(0.5, 0.2, 0.2), p)["label"] == (
        "NO_RELIABLE_SINGLE_SAMPLE_CHANNEL_IN_THIS_ARCHIVE"
    )
