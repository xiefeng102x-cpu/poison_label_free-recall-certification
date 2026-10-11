from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "auditability_certificate_v1/analysis"))

import run_trusted_identity_noninterference_v1 as identity


def test_keyed_rng_is_deterministic_and_domain_separated():
    key = hashlib.sha256(b"test-key").digest()
    left = identity.keyed_rng(key, "left", 1, 2).integers(0, 2**31, size=16)
    repeat = identity.keyed_rng(key, "left", 1, 2).integers(0, 2**31, size=16)
    right = identity.keyed_rng(key, "right", 1, 2).integers(0, 2**31, size=16)
    assert np.array_equal(left, repeat)
    assert not np.array_equal(left, right)


def test_rank_metrics_oracle_and_reverse():
    selected = np.array([0, 1, 2], dtype=np.int64)
    oracle = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    reverse = oracle[::-1]
    oracle_metrics = identity.rank_metrics(oracle, selected, attack_budget=2)
    reverse_metrics = identity.rank_metrics(reverse, selected, attack_budget=2)
    assert oracle_metrics[0] == 1.0
    assert oracle_metrics[2] == 1.0
    assert reverse_metrics[0] == 0.0
    assert reverse_metrics[2] == 0.0


def test_random_overlap_centers_on_exact_baseline():
    key = hashlib.sha256(b"selection-key").digest()
    n, k, budget = 200, 50, 20
    order = np.arange(n, dtype=np.int64)
    lifts = []
    for draw in range(2000):
        selected = identity.keyed_rng(key, "toy", draw).choice(n, k, replace=False)
        precision, _, _ = identity.rank_metrics(order, selected, attack_budget=budget)
        lifts.append(precision / (k / n))
    assert abs(float(np.mean(lifts)) - 1.0) < 0.03
