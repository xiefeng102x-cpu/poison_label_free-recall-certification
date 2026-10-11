from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_p6d_final_observable_adaptive_v4 as v4
import verify_p6d_final_observable_adaptive_v4 as verifier


def _row(seed: int, *, eligible: bool, tpr: float, fpr: float) -> dict:
    tp = int(round(60 * tpr))
    fp = int(round(600 * fpr))
    return {
        "seed": seed,
        "eligible": eligible,
        "tpr": tp / 60,
        "fpr": fp / 600,
        "tp": tp,
        "fn": 60 - tp,
        "fp": fp,
        "tn": 600 - fp,
    }


def test_v4_source_and_seed_locks_are_exact():
    protocol = json.loads(v4.PROTOCOL_PATH.read_text())
    assert v4.CONFIRMATION_SEEDS == tuple(range(124, 134))
    assert set(v4.CONFIRMATION_SEEDS).isdisjoint(set(range(42, 124)))
    assert protocol["seed_audit"]["replacement_seeds"] == []
    assert v4.sha256_file(Path(v4.__file__)) == protocol["source_audit"]["v4_runner_sha256"]
    assert verifier.sha256_file(Path(verifier.__file__)) == protocol["source_audit"]["v4_verifier_sha256"]


def test_v4_evasion_decision_requires_viability_sign_interval_and_fpr():
    rows = [
        _row(seed, eligible=index < 8, tpr=0.20 + 0.05 * (index % 4), fpr=0.01)
        for index, seed in enumerate(v4.CONFIRMATION_SEEDS)
    ]
    summary = v4.summarize_outcomes(rows, draws=2_000, rng_seed=124)
    independently_replayed = verifier.replay_summary(rows, draws=2_000, rng_seed=124)
    assert summary == independently_replayed
    assert summary["attack_viable_count"] == 8
    assert summary["joint_adaptive_evasion_count"] == 8
    assert np.isclose(summary["sign_test_pvalue_one_sided"], 1 / 256)
    assert summary["decision"] == "ADAPTIVE_EVASION_SUPPORTED"


def test_v4_keeps_failed_attacks_in_intention_to_evaluate_denominator():
    rows = [
        _row(seed, eligible=index < 7, tpr=0.10, fpr=0.0)
        for index, seed in enumerate(v4.CONFIRMATION_SEEDS)
    ]
    summary = v4.summarize_outcomes(rows, draws=1_000, rng_seed=125)
    assert summary["all_seed_count"] == 10
    assert summary["attack_viability_rate"] == 0.7
    assert summary["joint_adaptive_evasion_rate"] == 0.7
    assert summary["decision"] == "INSUFFICIENT_ATTACK_VIABILITY"


def test_v4_does_not_reclassify_high_tpr_as_evasion():
    rows = [
        _row(seed, eligible=True, tpr=0.95, fpr=0.01)
        for seed in v4.CONFIRMATION_SEEDS
    ]
    summary = v4.summarize_outcomes(rows, draws=2_000, rng_seed=126)
    assert summary["joint_adaptive_evasion_count"] == 0
    assert summary["decision"] == "DETECTOR_ROBUSTNESS_SUPPORTED"
