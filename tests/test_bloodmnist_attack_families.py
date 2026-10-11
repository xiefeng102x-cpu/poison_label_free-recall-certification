from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import bloodmnist_attack_families as attacks


def sample_images(count: int = 5) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.uniform(0.1, 0.9, size=(count, 28 * 28 * 3)).astype(np.float32)


def test_badnets_patch_changes_only_bottom_right_support() -> None:
    images = sample_images()
    triggered, receipt = attacks.apply_badnets_patch(images, patch_size=3)
    delta = (triggered - images).reshape(-1, 28, 28, 3)
    outside = delta.copy()
    outside[:, -3:, -3:, :] = 0
    assert np.count_nonzero(outside) == 0
    assert np.all(triggered.reshape(-1, 28, 28, 3)[:, -3:, -3:, :] == 1)
    assert receipt.family == "badnets_bottom_right_patch"


def test_blended_is_deterministic_and_matches_formula() -> None:
    images = sample_images()
    first, _ = attacks.apply_blended(images, alpha=0.2, pattern_seed=99)
    second, _ = attacks.apply_blended(images, alpha=0.2, pattern_seed=99)
    pattern = attacks.make_fixed_blend_pattern(seed=99).reshape(1, -1)
    np.testing.assert_allclose(first, second, atol=0, rtol=0)
    np.testing.assert_allclose(first, 0.8 * images + 0.2 * pattern, atol=1e-7)


def test_sig_matches_column_formula_without_clipping() -> None:
    images = np.full((2, 28, 28, 3), 0.5, dtype=np.float32)
    triggered, receipt = attacks.apply_sig(images, delta=0.1, frequency=2)
    expected = 0.5 + 0.1 * np.sin(2 * np.pi * np.arange(28) * 2 / 28)
    np.testing.assert_allclose(triggered[0, 4, :, 1], expected, atol=1e-6)
    assert receipt.parameters["frequency"] == 2


def test_wanet_grid_and_cross_samples_are_reproducible_and_distinct() -> None:
    grid_a, receipt = attacks.make_wanet_grid(seed=13)
    grid_b, _ = attacks.make_wanet_grid(seed=13)
    torch.testing.assert_close(grid_a, grid_b, rtol=0, atol=0)
    assert tuple(grid_a.shape) == (1, 28, 28, 2)
    assert float(grid_a.min()) >= -1 and float(grid_a.max()) <= 1
    cross_a = attacks.make_wanet_cross_grids(grid_a, count=4, seed=29)
    cross_b = attacks.make_wanet_cross_grids(grid_a, count=4, seed=29)
    torch.testing.assert_close(cross_a, cross_b, rtol=0, atol=0)
    assert not torch.equal(cross_a[0], grid_a[0])
    warped = attacks.apply_wanet(sample_images(4), cross_a)
    assert warped.shape == (4, 28 * 28 * 3)
    assert 0 <= warped.min() <= warped.max() <= 1
    assert receipt.parameters["cross_ratio"] == 2.0


def test_label_consistent_requires_adversarial_bases_and_target_class() -> None:
    images = sample_images(8)
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    with pytest.raises(attacks.AdversarialBaseRequiredError):
        attacks.build_label_consistent_poison(
            images,
            labels,
            target_label=1,
            poison_count=2,
            seed=4,
            adversarial_base_images=None,
        )
    poisoned, indices, receipt = attacks.build_label_consistent_poison(
        images,
        labels,
        target_label=1,
        poison_count=3,
        seed=4,
        adversarial_base_images=images,
    )
    assert np.all(labels[indices] == 1)
    assert poisoned.shape == (3, 28 * 28 * 3)
    assert receipt.family == "label_consistent_all_corners"


def test_input_aware_applies_generator_outputs_but_rejects_static_proxy() -> None:
    images = sample_images(2).reshape(2, 28, 28, 3)
    patterns = np.ones_like(images)
    masks = np.full_like(images, 0.25)
    result = attacks.apply_input_aware_generated(images, patterns, masks)
    np.testing.assert_allclose(result, images * 0.75 + 0.25, atol=1e-7)
    with pytest.raises(attacks.JointTrainingRequiredError):
        attacks.reject_static_input_aware()


@pytest.mark.parametrize(
    "transform",
    [
        lambda x: attacks.apply_badnets_patch(x)[0],
        lambda x: attacks.apply_blended(x)[0],
        lambda x: attacks.apply_sig(x)[0],
        lambda x: attacks.apply_label_consistent_corners(x)[0],
    ],
)
def test_static_transforms_preserve_shape_dtype_and_range(transform) -> None:
    images = sample_images()
    result = transform(images)
    assert result.shape == images.shape
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))
    assert 0 <= result.min() <= result.max() <= 1
