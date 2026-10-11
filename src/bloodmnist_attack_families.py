#!/usr/bin/env python3
"""Clean-room BloodMNIST image-space attack primitives.

The functions in this module intentionally separate image transforms from
poison-set construction.  That prevents a static trigger from being presented
as a faithful implementation of attacks whose defining mechanism is joint
training (Input-Aware) or adversarial base-image generation
(Label-Consistent).

Inputs are float images in [0, 1], either flattened ``(N, H*W*C)`` arrays or
``(N, H, W, C)`` arrays.  Returned images preserve the input layout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


BLOODMNIST_SHAPE = (28, 28, 3)


class JointTrainingRequiredError(RuntimeError):
    """Raised when a dynamic attack is requested as a static transform."""


class AdversarialBaseRequiredError(RuntimeError):
    """Raised when Label-Consistent lacks its adversarial base images."""


@dataclass(frozen=True)
class AttackReceipt:
    family: str
    parameters: dict[str, Any]
    training_mechanism: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_hwc_batch(
    images: np.ndarray, image_shape: tuple[int, int, int] = BLOODMNIST_SHAPE
) -> tuple[np.ndarray, bool]:
    array = np.asarray(images, dtype=np.float32)
    was_flat = array.ndim == 2
    if was_flat:
        expected = int(np.prod(image_shape))
        if array.shape[1] != expected:
            raise ValueError(f"Expected flattened width {expected}, got {array.shape[1]}")
        array = array.reshape(len(array), *image_shape)
    if array.ndim != 4 or tuple(array.shape[1:]) != image_shape:
        raise ValueError(
            f"Expected (N, {image_shape[0]}, {image_shape[1]}, {image_shape[2]}) "
            f"or flattened equivalent, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Images contain non-finite values")
    return array, was_flat


def _restore_layout(images: np.ndarray, was_flat: bool) -> np.ndarray:
    images = np.asarray(images, dtype=np.float32)
    if was_flat:
        return images.reshape(len(images), -1)
    return images


def _validate_unit_interval(images: np.ndarray) -> None:
    if len(images) and (float(images.min()) < 0.0 or float(images.max()) > 1.0):
        raise ValueError("Expected input images scaled to [0, 1]")


def apply_badnets_patch(
    images: np.ndarray,
    *,
    patch_size: int = 3,
    value: float = 1.0,
    image_shape: tuple[int, int, int] = BLOODMNIST_SHAPE,
) -> tuple[np.ndarray, AttackReceipt]:
    """Apply a solid bottom-right BadNets patch."""

    batch, was_flat = _as_hwc_batch(images, image_shape)
    _validate_unit_interval(batch)
    height, width, _ = image_shape
    if patch_size <= 0 or patch_size > min(height, width):
        raise ValueError("patch_size must fit inside the image")
    if not 0.0 <= value <= 1.0:
        raise ValueError("value must be in [0, 1]")
    result = batch.copy()
    result[:, height - patch_size :, width - patch_size :, :] = value
    receipt = AttackReceipt(
        family="badnets_bottom_right_patch",
        parameters={"patch_size": patch_size, "value": value},
        training_mechanism="all-to-one label overwrite on selected non-target samples",
    )
    return _restore_layout(result, was_flat), receipt


def make_fixed_blend_pattern(
    *,
    seed: int = 14014,
    image_shape: tuple[int, int, int] = BLOODMNIST_SHAPE,
) -> np.ndarray:
    """Return the frozen full-frame trigger used by the Blended attack.

    A seeded random pattern avoids importing BackdoorBench's bundled image and
    makes the exact pattern reproducible from a compact protocol receipt.
    """

    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=image_shape).astype(np.float32)


def apply_blended(
    images: np.ndarray,
    *,
    alpha: float = 0.2,
    pattern: np.ndarray | None = None,
    pattern_seed: int = 14014,
    image_shape: tuple[int, int, int] = BLOODMNIST_SHAPE,
) -> tuple[np.ndarray, AttackReceipt]:
    """Apply ``(1-alpha)*image + alpha*fixed_pattern``."""

    batch, was_flat = _as_hwc_batch(images, image_shape)
    _validate_unit_interval(batch)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if pattern is None:
        pattern = make_fixed_blend_pattern(seed=pattern_seed, image_shape=image_shape)
    pattern = np.asarray(pattern, dtype=np.float32)
    if pattern.shape != image_shape:
        raise ValueError(f"Expected blend pattern shape {image_shape}, got {pattern.shape}")
    _validate_unit_interval(pattern[None])
    result = (1.0 - alpha) * batch + alpha * pattern[None]
    receipt = AttackReceipt(
        family="blended_full_frame",
        parameters={"alpha": alpha, "pattern_seed": pattern_seed},
        training_mechanism="all-to-one label overwrite on selected non-target samples",
    )
    return _restore_layout(result, was_flat), receipt


def apply_sig(
    images: np.ndarray,
    *,
    delta: float = 40.0 / 255.0,
    frequency: float = 6.0,
    image_shape: tuple[int, int, int] = BLOODMNIST_SHAPE,
) -> tuple[np.ndarray, AttackReceipt]:
    """Apply the column-wise sinusoidal SIG trigger.

    The original SIG/BackdoorBench amplitude 40 on uint8 images is represented
    as ``40/255`` for BloodMNIST's [0, 1] arrays.
    """

    batch, was_flat = _as_hwc_batch(images, image_shape)
    _validate_unit_interval(batch)
    if delta < 0 or frequency <= 0:
        raise ValueError("delta must be non-negative and frequency must be positive")
    width = image_shape[1]
    columns = np.arange(width, dtype=np.float32)
    signal = delta * np.sin(2.0 * np.pi * columns * frequency / width)
    pattern = signal[None, None, :, None]
    result = np.clip(batch + pattern, 0.0, 1.0)
    receipt = AttackReceipt(
        family="sig_sinusoidal_clean_label",
        parameters={"delta": delta, "frequency": frequency},
        training_mechanism="clean-label poisoning of selected target-class samples",
    )
    return _restore_layout(result, was_flat), receipt


def make_wanet_grid(
    *,
    seed: int,
    height: int = 28,
    width: int = 28,
    k: int = 4,
    strength: float = 0.5,
    grid_rescale: float = 1.0,
) -> tuple[torch.Tensor, AttackReceipt]:
    """Create the frozen WaNet identity-plus-smooth-noise sampling grid."""

    if height != width:
        raise ValueError("This protocol currently requires square inputs")
    if k < 2 or strength < 0 or grid_rescale <= 0:
        raise ValueError("Invalid WaNet grid parameters")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    coarse = torch.rand((1, 2, k, k), generator=generator) * 2.0 - 1.0
    coarse = coarse / torch.mean(torch.abs(coarse))
    noise = F.interpolate(
        coarse, size=(height, width), mode="bicubic", align_corners=True
    ).permute(0, 2, 3, 1)
    axis_y = torch.linspace(-1.0, 1.0, steps=height)
    axis_x = torch.linspace(-1.0, 1.0, steps=width)
    yy, xx = torch.meshgrid(axis_y, axis_x, indexing="ij")
    identity = torch.stack((xx, yy), dim=-1)[None]
    grid = torch.clamp(
        (identity + strength * noise / float(height)) * grid_rescale, -1.0, 1.0
    )
    receipt = AttackReceipt(
        family="wanet_smooth_warp",
        parameters={
            "grid_seed": seed,
            "height": height,
            "width": width,
            "k": k,
            "strength": strength,
            "grid_rescale": grid_rescale,
            "cross_ratio": 2.0,
        },
        training_mechanism=(
            "all-to-one warped poisons plus correct-label cross/noise warped samples"
        ),
    )
    return grid.to(dtype=torch.float32), receipt


def make_wanet_cross_grids(
    base_grid: torch.Tensor,
    *,
    count: int,
    seed: int,
) -> torch.Tensor:
    """Add WaNet's per-sample uniform grid jitter for cross/noise examples."""

    if base_grid.ndim != 4 or base_grid.shape[0] != 1 or base_grid.shape[-1] != 2:
        raise ValueError("base_grid must have shape (1, H, W, 2)")
    if count < 0:
        raise ValueError("count must be non-negative")
    height = int(base_grid.shape[1])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    jitter = torch.rand(
        (count, base_grid.shape[1], base_grid.shape[2], 2), generator=generator
    ) * 2.0 - 1.0
    return torch.clamp(base_grid.repeat(count, 1, 1, 1) + jitter / height, -1.0, 1.0)


def apply_wanet(
    images: np.ndarray,
    grid: torch.Tensor,
    *,
    image_shape: tuple[int, int, int] = BLOODMNIST_SHAPE,
) -> np.ndarray:
    """Warp a batch with a shared or per-sample WaNet sampling grid."""

    batch, was_flat = _as_hwc_batch(images, image_shape)
    _validate_unit_interval(batch)
    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2)
    grid = torch.as_tensor(grid, dtype=torch.float32)
    if grid.shape[0] == 1:
        grid = grid.repeat(len(batch), 1, 1, 1)
    expected = (len(batch), image_shape[0], image_shape[1], 2)
    if tuple(grid.shape) != expected:
        raise ValueError(f"Expected grid shape {expected}, got {tuple(grid.shape)}")
    warped = F.grid_sample(
        tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    result = warped.permute(0, 2, 3, 1).numpy().astype(np.float32, copy=False)
    return _restore_layout(np.clip(result, 0.0, 1.0), was_flat)


def apply_label_consistent_corners(
    images: np.ndarray,
    *,
    reduced_amplitude: float = 1.0,
    image_shape: tuple[int, int, int] = BLOODMNIST_SHAPE,
) -> tuple[np.ndarray, AttackReceipt]:
    """Apply the four-corner Label-Consistent sign pattern.

    This function only applies the published corner pattern.  Use
    :func:`build_label_consistent_poison` to enforce the adversarial-base and
    target-class requirements of the full training attack.
    """

    batch, was_flat = _as_hwc_batch(images, image_shape)
    _validate_unit_interval(batch)
    if not 0.0 <= reduced_amplitude <= 1.0:
        raise ValueError("reduced_amplitude must be in [0, 1]")
    positive = {(0, 0), (1, 1), (2, 0), (2, 2)}
    base = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    corners = [
        (0, 0, 1, 1),
        (image_shape[0] - 1, 0, -1, 1),
        (0, image_shape[1] - 1, 1, -1),
        (image_shape[0] - 1, image_shape[1] - 1, -1, -1),
    ]
    result = batch.copy()
    for anchor_y, anchor_x, direction_y, direction_x in corners:
        for row, col in base:
            y = anchor_y + direction_y * row
            x = anchor_x + direction_x * col
            sign = 1.0 if (row, col) in positive else -1.0
            result[:, y, x, :] += sign * reduced_amplitude
    result = np.clip(result, 0.0, 1.0)
    receipt = AttackReceipt(
        family="label_consistent_all_corners",
        parameters={"reduced_amplitude": reduced_amplitude},
        training_mechanism=(
            "target-class-only clean-label poisoning on externally generated adversarial bases"
        ),
    )
    return _restore_layout(result, was_flat), receipt


def build_label_consistent_poison(
    clean_images: np.ndarray,
    clean_labels: np.ndarray,
    *,
    target_label: int,
    poison_count: int,
    seed: int,
    adversarial_base_images: np.ndarray | None,
    reduced_amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, AttackReceipt]:
    """Build target-class, clean-label poisons from adversarial base images."""

    labels = np.asarray(clean_labels)
    target_indices = np.flatnonzero(labels == target_label)
    if poison_count <= 0 or poison_count > len(target_indices):
        raise ValueError("poison_count must be within the target-class population")
    if adversarial_base_images is None:
        raise AdversarialBaseRequiredError(
            "Full Label-Consistent poisoning requires adversarial base images; "
            "a target-only corner patch is not a faithful substitute."
        )
    bases = np.asarray(adversarial_base_images, dtype=np.float32)
    if len(bases) != len(clean_images):
        raise ValueError("adversarial_base_images must align with clean_images")
    selected = np.random.default_rng(seed).permutation(target_indices)[:poison_count]
    poisoned, receipt = apply_label_consistent_corners(
        bases[selected], reduced_amplitude=reduced_amplitude
    )
    return poisoned, selected.astype(np.int64), receipt


def apply_input_aware_generated(
    images: np.ndarray,
    patterns: np.ndarray,
    masks: np.ndarray,
    *,
    image_shape: tuple[int, int, int] = BLOODMNIST_SHAPE,
) -> np.ndarray:
    """Apply outputs of jointly trained Input-Aware pattern/mask generators."""

    batch, was_flat = _as_hwc_batch(images, image_shape)
    pattern_batch, _ = _as_hwc_batch(patterns, image_shape)
    mask_batch, _ = _as_hwc_batch(masks, image_shape)
    if len(pattern_batch) != len(batch) or len(mask_batch) != len(batch):
        raise ValueError("patterns and masks must align one-to-one with images")
    _validate_unit_interval(batch)
    _validate_unit_interval(pattern_batch)
    _validate_unit_interval(mask_batch)
    result = batch * (1.0 - mask_batch) + pattern_batch * mask_batch
    return _restore_layout(np.clip(result, 0.0, 1.0), was_flat)


def reject_static_input_aware() -> None:
    """Reject an invalid static-noise approximation of Input-Aware."""

    raise JointTrainingRequiredError(
        "Input-Aware requires jointly trained, sample-conditioned pattern and mask "
        "generators with diversity and mask-density constraints."
    )
