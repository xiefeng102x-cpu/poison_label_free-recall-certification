import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from run_liu_matched_baseline import (
    liu_algorithm1,
    liu_corrections,
    liu_minimum_balanced_n,
)


def test_algorithm1_matches_direct_empirical_cdf_search():
    clean = np.array([0.0, 1.0, 2.0, 3.0])
    mixture = np.array([0.5, 1.5, 4.0, 5.0])
    prevalence = 0.5
    q = 0.25
    threshold, fhat0 = liu_algorithm1(clean, mixture, prevalence, q)

    support = np.unique(np.r_[clean, mixture])
    f_alien = []
    for value in support:
        fm = np.mean(mixture <= value)
        f0 = np.mean(clean <= value)
        f_alien.append((fm - (1.0 - prevalence) * f0) / prevalence)
    expected = support[np.asarray(f_alien) <= q].max()
    assert threshold == expected
    assert fhat0 == np.mean(clean <= expected)


def test_corrections_shrink_with_sample_size():
    small = liu_corrections(2_000, 0.2)
    large = liu_corrections(10_000, 0.2)
    assert large[0] < small[0]
    assert large[1] < small[1]


def test_rare_poison_requires_more_samples():
    assert liu_minimum_balanced_n(0.01) > liu_minimum_balanced_n(0.05)
    assert liu_minimum_balanced_n(0.05) > liu_minimum_balanced_n(0.20)
