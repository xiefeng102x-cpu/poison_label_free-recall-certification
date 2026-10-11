from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "auditability_certificate_v1/analysis"))

from blind_score_families import METHODS, fit_score_map, rank_pvalues, whitener
from cifar10_vgg16_bn import CIFAR10VGG16BN, parameter_count
from run_cifar10_vgg16_blind_certificate import certificate_draw


def test_vgg16_bn_shape_and_size() -> None:
    model = CIFAR10VGG16BN()
    logits, features = model.forward_with_features(torch.rand(2, 3, 32, 32))
    assert logits.shape == (2, 10)
    assert features.shape == (2, 512)
    assert parameter_count() == 14_724_042


def test_all_frozen_score_families_are_finite() -> None:
    rng = np.random.default_rng(12)
    reference = rng.normal(size=(64, 16))
    fitting = np.r_[rng.normal(size=(80, 16)), rng.normal(1.0, 1.0, size=(20, 16))]
    query = rng.normal(size=(30, 16))
    whitening = whitener(reference)
    for index, method in enumerate(METHODS):
        scorer, _ = fit_score_map(method, reference, fitting, seed=100 + index, whitening=whitening)
        scores = scorer(query)
        assert scores.shape == (30,)
        assert np.isfinite(scores).all()
        pvalues = rank_pvalues(scores[:20], scores[20:])
        assert np.all((pvalues > 0.0) & (pvalues <= 1.0))


def test_certificate_draw_is_exactly_replayable() -> None:
    scores = np.linspace(-2.0, 2.0, 80)
    sample_ids = np.arange(100, 180, dtype=np.int64)
    trusted_ids = sample_ids[:60]
    first = certificate_draw(scores, sample_ids, trusted_ids, 20, np.random.default_rng(42))
    second = certificate_draw(scores, sample_ids, trusted_ids, 20, np.random.default_rng(42))
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2:] == second[2:]


def test_formal_config_has_familywise_primary_and_fresh_seeds() -> None:
    config = json.loads((ROOT / "configs/cifar10_vgg16_bn_blind_certificate_v1.json").read_text())
    assert config["formal_seeds"] == [4201, 4202, 4203, 4204, 4205]
    assert config["development_provenance"]["seed"] not in config["formal_seeds"]
    assert config["certificate"]["delta_per_detector_fold"] == 0.005
    assert len(config["detectors"]) == 5
    assert config["certificate"]["trusted_member_budgets"] == [120, 500, 2000]
