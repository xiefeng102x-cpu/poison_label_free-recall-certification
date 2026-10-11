from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pathmnist_external_validation as pathv  # noqa: E402
import run_pathmnist_external_validation_v2 as pathv2  # noqa: E402


def toy_images(count: int) -> np.ndarray:
    return np.zeros((count, 28, 28, 3), dtype=np.uint8)


def test_poison_pool_is_deterministic_in_place_and_targeted() -> None:
    images = toy_images(180)
    labels = np.tile(np.arange(9, dtype=np.int64), 20)
    first = pathv.build_poisoned_training(images, labels, seed=79, poison_count=18)
    second = pathv.build_poisoned_training(images, labels, seed=79, poison_count=18)
    assert np.array_equal(first["selected_indices"], second["selected_indices"])
    selected = first["selected_indices"]
    assert len(selected) == 18
    assert np.all(labels[selected] != pathv.TARGET_CLASS)
    assert np.all(first["assigned_labels"][selected] == pathv.TARGET_CLASS)
    assert np.all(first["images"][selected, -4:, -4:, :] == 255)
    assert np.array_equal(first["images"][~first["is_poison"]], images[~first["is_poison"]])


def test_balanced_subset_preserves_original_indices() -> None:
    images = toy_images(180)
    labels = np.tile(np.arange(9, dtype=np.int64), 20)
    subset = pathv.select_balanced_training_subset(images, labels, per_class=10, seed=7)
    assert len(subset["labels"]) == 90
    assert np.array_equal(np.bincount(subset["labels"], minlength=9), np.full(9, 10))
    assert np.array_equal(subset["images"], images[subset["original_indices"]])


def test_residual_model_has_frozen_hidden_shape_and_is_lightweight() -> None:
    mean = np.asarray([0.5, 0.4, 0.3], dtype=np.float32)
    std = np.asarray([0.2, 0.2, 0.2], dtype=np.float32)
    model = pathv.make_model(79, mean, std)
    values = torch.rand(3, 3, 28, 28)
    assert model.hidden(values).shape == (3, 64)
    assert model(values).shape == (3, 9)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 50_000 < parameter_count < 100_000


def test_rank_pvalues_use_greater_equal_finite_sample_rule() -> None:
    calibration = np.asarray([1.0, 2.0, 3.0])
    query = np.asarray([0.0, 2.0, 4.0])
    assert np.allclose(pathv._rank_pvalues(calibration, query), [1.0, 0.75, 0.25])


def test_cf_cmd_u_is_deterministic_and_only_scores_target_class() -> None:
    rng = np.random.default_rng(9)
    dimension = 5
    reference = rng.normal(size=(1100, dimension))
    reference_y = np.full(1100, pathv.TARGET_CLASS, dtype=np.int64)
    clean = rng.normal(size=(1200, dimension))
    poison = rng.normal(loc=3.0, size=(200, dimension))
    candidates = np.concatenate([clean, poison], axis=0)
    candidate_y = np.full(len(candidates), pathv.TARGET_CLASS, dtype=np.int64)
    non_target = rng.normal(size=(50, dimension))
    candidates = np.concatenate([candidates, non_target], axis=0)
    candidate_y = np.concatenate([candidate_y, np.zeros(50, dtype=np.int64)])
    first = pathv.cf_cmd_u(reference, reference_y, candidates, candidate_y, seed=79)
    second = pathv.cf_cmd_u(reference, reference_y, candidates, candidate_y, seed=79)
    assert np.array_equal(first["suspicious_mask"], second["suspicious_mask"])
    assert np.allclose(first["p_values"], second["p_values"])
    assert np.isnan(first["scores"][-50:]).all()
    assert not first["suspicious_mask"][-50:].any()
    assert first["suspicious_mask"][1200:1400].mean() > 0.95


def test_evaluate_flags_replays_confusion_and_target_auc() -> None:
    truth = np.asarray([0, 0, 1, 1], dtype=bool)
    flags = np.asarray([0, 1, 1, 0], dtype=bool)
    scores = np.asarray([0.1, 0.2, 0.9, 0.8])
    assigned = np.full(4, pathv.TARGET_CLASS)
    original = np.asarray([8, 8, 1, 2])
    metrics = pathv.evaluate_flags(flags, scores, truth, assigned, original)
    assert (metrics["tp"], metrics["fn"], metrics["fp"], metrics["tn"]) == (1, 1, 1, 1)
    assert metrics["target_class_score_auc"] == pytest.approx(1.0)


def test_v2_poison_footprint_repair_is_fixed_and_budget_matched() -> None:
    assert pathv2.DEVELOPMENT_SEEDS == (86, 87)
    assert pathv2.CONFIRMATION_SEEDS == (88, 89, 90, 91, 92)
    assert pathv2.POISON_COUNT == 810
    assert pathv2.PATCH_SIZE == 5
    assert 0.03 * pathv2.PATCH_SIZE**2 == pytest.approx(0.75)
    assert 0.05 * pathv.PATCH_SIZE**2 == pytest.approx(0.80)
