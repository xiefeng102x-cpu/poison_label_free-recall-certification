#!/usr/bin/env python3
"""Architecture adapter for pinned BackdoorBench AGPD feature primitives.

This module preserves AGPD's activation-gradient, class-anomaly, and
progressive-filtering equations while making the convolution-layer list and
input tensors explicit.  Compatibility repairs are frozen in
``multiclass_agpd_adapter_protocol_v1.json`` and are not tuned by poison labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon


EPS = 1e-12


@dataclass(frozen=True)
class AGPDResult:
    flags: np.ndarray
    selected_layer: str | None
    target_classes: tuple[int, ...]
    rho_by_layer: dict[str, np.ndarray]
    anomaly_by_layer: dict[str, np.ndarray]
    progressive_iterations: dict[int, int]
    repair_activations: tuple[str, ...]


def _as_float_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def cosine_similarity_rows(values: np.ndarray, template: np.ndarray) -> np.ndarray:
    matrix = _as_float_matrix(values, "values")
    vector = np.asarray(template, dtype=np.float64).reshape(-1)
    if vector.shape[0] != matrix.shape[1] or not np.isfinite(vector).all():
        raise ValueError("template shape or values do not match")
    numerator = matrix @ vector
    denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    result = np.zeros(matrix.shape[0], dtype=np.float64)
    valid = denominator > EPS
    result[valid] = numerator[valid] / denominator[valid]
    return np.clip(result, -1.0, 1.0)


def farthest_from_reference(values: np.ndarray, clean_reference: np.ndarray) -> int:
    similarities = cosine_similarity_rows(values, clean_reference)
    return int(np.argmin(similarities))


def upstream_rho(
    values: np.ndarray, clean_reference: np.ndarray, suspect_indices: Sequence[int]
) -> float:
    matrix = _as_float_matrix(values, "values")
    suspects = np.unique(np.asarray(suspect_indices, dtype=np.int64))
    if suspects.size == 0 or suspects.min() < 0 or suspects.max() >= len(matrix):
        raise ValueError("suspect_indices must identify at least one matrix row")
    keep = np.ones(len(matrix), dtype=bool)
    keep[suspects] = False
    if not np.any(keep):
        return 0.0
    poison_mean = matrix[suspects].mean(axis=0)
    clean_similarity = cosine_similarity_rows(matrix[keep], clean_reference)
    poison_similarity = cosine_similarity_rows(matrix[keep], poison_mean)
    return float(np.linalg.norm(clean_similarity - poison_similarity) / np.sqrt(len(matrix)))


def one_sided_mad(values: Iterable[float]) -> np.ndarray:
    scores = np.asarray(tuple(values), dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("values must be a finite non-empty vector")
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    result = np.abs(scores - median) / (1.4826 * (mad + 1e-6))
    result[scores < median] = 0.0
    return result


def _rate_vector(
    matrix: np.ndarray, clean_reference: np.ndarray, known: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    remaining = np.setdiff1d(np.arange(len(matrix), dtype=np.int64), known)
    if remaining.size == 0:
        return remaining, np.empty(0, dtype=np.float64)
    poison_mean = matrix[known].mean(axis=0)
    sim_poison = cosine_similarity_rows(matrix[remaining], poison_mean)
    sim_clean = cosine_similarity_rows(matrix[remaining], clean_reference)
    distance_poison = 1.0 - sim_poison
    distance_clean = 1.0 - sim_clean
    denominator = distance_poison + distance_clean
    rates = np.full(remaining.size, 0.5, dtype=np.float64)
    valid = np.abs(denominator) > EPS
    rates[valid] = distance_poison[valid] / denominator[valid]
    return remaining, rates


def _upstream_stage(
    matrix: np.ndarray,
    clean_reference: np.ndarray,
    known: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    remaining, rates = _rate_vector(matrix, clean_reference, known)
    if rates.size == 0:
        return np.empty(0, dtype=np.int64), rates
    span = float(rates.max() - rates.min())
    if span <= EPS:
        selected = remaining.copy()
    else:
        normalized = (rates - rates.min()) / span
        selected = remaining[normalized <= tau]
    retained_rates = rates[~np.isin(remaining, selected)]
    return selected.astype(np.int64, copy=False), retained_rates


def _js_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 1.0
    bins = np.linspace(0.0, 1.0, 100)
    left_hist, _ = np.histogram(left, bins=bins, density=False)
    right_hist, _ = np.histogram(right, bins=bins, density=False)
    if left_hist.sum() == 0 or right_hist.sum() == 0:
        return 1.0
    return float(jensenshannon(left_hist, right_hist))


def progressive_filter(
    values: np.ndarray,
    clean_reference: np.ndarray,
    tau: float = 0.5,
    window_size: int = 10,
) -> tuple[np.ndarray, int, tuple[str, ...]]:
    matrix = _as_float_matrix(values, "values")
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    if window_size < 1:
        raise ValueError("window_size must be positive")

    known = np.asarray([farthest_from_reference(matrix, clean_reference)], dtype=np.int64)
    groups = [known.copy()]
    _, previous_rates = _rate_vector(matrix, clean_reference, known)
    js_updates: list[float] = []

    while len(known) < len(matrix):
        selected, current_rates = _upstream_stage(matrix, clean_reference, known, tau)
        if selected.size == 0:
            break
        groups.append(selected)
        known = np.unique(np.concatenate([known, selected]))
        js_updates.append(_js_distance(previous_rates, current_rates))
        previous_rates = current_rates

    repairs: list[str] = []
    if not js_updates:
        return groups[0], len(groups), ("retain_initial_farthest_sample",)

    effective_window = min(window_size, len(js_updates))
    if effective_window < window_size:
        repairs.append("short_js_window")
    window_scores = []
    for start in range(len(js_updates) - effective_window + 1):
        window = np.asarray(js_updates[start : start + effective_window], dtype=np.float64)
        window_scores.append(float(window.mean() + 5.0 * window.std()))
    upstream_start = int(np.argmin(window_scores))
    group_count = max(1, upstream_start)
    if upstream_start == 0:
        repairs.append("retain_initial_farthest_sample")
    final = np.unique(np.concatenate(groups[:group_count])).astype(np.int64, copy=False)
    return final, len(groups), tuple(repairs)


def collect_activation_gradients(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    layer_name: str,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    if len(images) != len(labels) or len(images) == 0:
        raise ValueError("images and labels must be aligned and non-empty")
    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise KeyError(f"unknown layer {layer_name!r}")
    layer = modules[layer_name]
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    rows: list[np.ndarray] = []

    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size].to(device)
        batch_labels = labels[start : start + batch_size].to(device=device, dtype=torch.long)
        captured: list[torch.Tensor] = []

        def capture(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            captured.append(output)

        handle = layer.register_forward_hook(capture)
        try:
            logits = model(batch_images)
            if logits.ndim != 2 or logits.shape[0] != len(batch_images):
                raise ValueError("model must return an [N, C] logit matrix")
            if batch_labels.min() < 0 or batch_labels.max() >= logits.shape[1]:
                raise ValueError("labels fall outside model output classes")
            selected = logits.gather(1, batch_labels[:, None]).sum()
            if len(captured) != 1:
                raise RuntimeError("declared layer must execute exactly once per forward pass")
            gradient = torch.autograd.grad(selected, captured[0], retain_graph=False)[0]
        finally:
            handle.remove()
        if gradient.ndim != 4:
            raise ValueError("AGPD convolution gradient must have NCHW shape")
        rows.append(gradient.mean(dim=(2, 3)).detach().cpu().numpy().astype(np.float64))

    result = np.concatenate(rows, axis=0)
    if not np.isfinite(result).all():
        raise ValueError("activation gradients contain non-finite values")
    return result


def run_agpd_adapter(
    model: torch.nn.Module,
    candidate_images: torch.Tensor,
    assigned_labels: np.ndarray,
    reference_images: torch.Tensor,
    reference_labels: np.ndarray,
    layer_names: Sequence[str],
    num_classes: int,
    tau: float = 0.5,
    anomaly_threshold: float = float(np.exp(2.0)),
    fallback_rho_threshold: float = 0.3,
    window_size: int = 10,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
) -> AGPDResult:
    labels = np.asarray(assigned_labels, dtype=np.int64)
    ref_labels = np.asarray(reference_labels, dtype=np.int64)
    if len(candidate_images) != len(labels) or len(reference_images) != len(ref_labels):
        raise ValueError("image and label arrays must be aligned")
    if set(np.unique(ref_labels)) != set(range(num_classes)):
        raise ValueError("reference must contain every class")
    if not layer_names:
        raise ValueError("at least one convolution layer is required")

    candidate_label_tensor = torch.as_tensor(labels, dtype=torch.long)
    reference_label_tensor = torch.as_tensor(ref_labels, dtype=torch.long)
    gradients_by_layer: dict[str, np.ndarray] = {}
    references_by_layer: dict[str, np.ndarray] = {}
    rho_by_layer: dict[str, np.ndarray] = {}
    anomaly_by_layer: dict[str, np.ndarray] = {}

    for layer_name in layer_names:
        candidate_gradients = collect_activation_gradients(
            model,
            candidate_images,
            candidate_label_tensor,
            layer_name,
            batch_size=batch_size,
            device=device,
        )
        reference_gradients = collect_activation_gradients(
            model,
            reference_images,
            reference_label_tensor,
            layer_name,
            batch_size=batch_size,
            device=device,
        )
        class_references = np.stack(
            [reference_gradients[ref_labels == label].mean(axis=0) for label in range(num_classes)]
        )
        rhos = []
        for label in range(num_classes):
            class_gradients = candidate_gradients[labels == label]
            if len(class_gradients) < 2:
                raise ValueError(f"class {label} has fewer than two candidate samples")
            initial = farthest_from_reference(class_gradients, class_references[label])
            rhos.append(upstream_rho(class_gradients, class_references[label], [initial]))
        gradients_by_layer[layer_name] = candidate_gradients
        references_by_layer[layer_name] = class_references
        rho_by_layer[layer_name] = np.asarray(rhos, dtype=np.float64)
        anomaly_by_layer[layer_name] = one_sided_mad(rhos)

    max_anomalies = np.asarray(
        [anomaly_by_layer[layer].max() for layer in layer_names], dtype=np.float64
    )
    repair_activations: list[str] = []
    if float(max_anomalies.max()) >= anomaly_threshold:
        selected_layer = str(layer_names[int(np.argmax(max_anomalies))])
        target_classes = tuple(
            int(value)
            for value in np.flatnonzero(anomaly_by_layer[selected_layer] > anomaly_threshold)
        )
    else:
        repair_activations.append("short_layer_fallback")
        fallback_targets = [
            tuple(int(value) for value in np.flatnonzero(rho_by_layer[layer] > fallback_rho_threshold))
            for layer in layer_names
        ]
        counts = np.asarray([len(values) for values in fallback_targets])
        if counts.max(initial=0) == 0:
            selected_layer = None
            target_classes = ()
        else:
            selected_index = int(np.argmax(counts))
            selected_layer = str(layer_names[selected_index])
            target_classes = fallback_targets[selected_index]

    flags = np.zeros(len(labels), dtype=np.int8)
    iterations: dict[int, int] = {}
    if selected_layer is not None:
        for target_class in target_classes:
            global_indices = np.flatnonzero(labels == target_class)
            class_gradients = gradients_by_layer[selected_layer][global_indices]
            local_flags, iteration_count, repairs = progressive_filter(
                class_gradients,
                references_by_layer[selected_layer][target_class],
                tau=tau,
                window_size=window_size,
            )
            flags[global_indices[local_flags]] = 1
            iterations[target_class] = int(iteration_count)
            repair_activations.extend(repairs)

    return AGPDResult(
        flags=flags,
        selected_layer=selected_layer,
        target_classes=target_classes,
        rho_by_layer=rho_by_layer,
        anomaly_by_layer=anomaly_by_layer,
        progressive_iterations=iterations,
        repair_activations=tuple(sorted(set(repair_activations))),
    )
