#!/usr/bin/env python3
"""Detector-agnostic score families for prospective blind handoffs."""

from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


METHODS = (
    "cf_cmd_u",
    "target_centroid",
    "crossfit_spectral",
    "activation_clustering",
    "que_score",
)


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def whitener(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    reference = np.asarray(reference, dtype=np.float64)
    mean = reference.mean(axis=0)
    centered = reference - mean
    covariance = centered.T @ centered / max(len(reference) - 1, 1)
    ridge = max(float(np.trace(covariance) / reference.shape[1]) * 0.1, 1e-6)
    values, vectors = np.linalg.eigh(covariance)
    transform = (vectors * (1.0 / np.sqrt(np.maximum(values, 0.0) + ridge))) @ vectors.T
    return mean, transform, ridge


def rank_pvalues(calibration_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    calibration = np.asarray(calibration_scores, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    # Conservative upper-tail rank p-values, including the query itself.
    return (1.0 + np.sum(calibration[None, :] >= values[:, None], axis=1)) / (len(calibration) + 1.0)


def fit_score_map(
    method: str,
    reference: np.ndarray,
    fitting: np.ndarray,
    *,
    seed: int,
    whitening: tuple[np.ndarray, np.ndarray, float],
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    reference = np.asarray(reference, dtype=np.float64)
    fitting = np.asarray(fitting, dtype=np.float64)
    ref_mean = reference.mean(axis=0)
    if method == "cf_cmd_u":
        direction = fitting.mean(axis=0) - ref_mean
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            raise RuntimeError("CF-CMD-U direction collapsed")
        direction /= norm
        return (lambda values: (np.asarray(values, dtype=np.float64) - ref_mean) @ direction), {
            "direction_norm": norm, "direction_sha256": array_hash(direction), "transform": "identity",
        }
    if method == "target_centroid":
        return (lambda values: np.linalg.norm(np.asarray(values, dtype=np.float64) - ref_mean, axis=1)), {
            "reference_centroid_sha256": array_hash(ref_mean), "transform": "identity",
        }
    if method == "crossfit_spectral":
        center = fitting.mean(axis=0)
        reducer = PCA(n_components=1, svd_solver="randomized", random_state=seed).fit(fitting - center)
        direction = reducer.components_[0].astype(np.float64)
        return (lambda values: ((np.asarray(values, dtype=np.float64) - center) @ direction) ** 2), {
            "fit_center_sha256": array_hash(center), "direction_sha256": array_hash(direction),
            "explained_variance_ratio": float(reducer.explained_variance_ratio_[0]),
        }
    white_mean, transform, ridge = whitening
    reference_z = (reference - white_mean) @ transform
    fitting_z = (fitting - white_mean) @ transform
    if method == "activation_clustering":
        dimension = min(16, fitting_z.shape[1], len(reference_z) - 1)
        reducer = PCA(n_components=dimension, svd_solver="randomized", random_state=seed).fit(
            np.concatenate([reference_z, fitting_z], axis=0)
        )
        reference_low = reducer.transform(reference_z)
        fitting_low = reducer.transform(fitting_z)
        cluster = KMeans(n_clusters=2, random_state=seed, n_init=20).fit(fitting_low)
        centers = cluster.cluster_centers_
        reference_center = reference_low.mean(axis=0)
        suspicious = int(np.argmax(np.linalg.norm(centers - reference_center[None, :], axis=1)))
        clean = 1 - suspicious

        def score(values: np.ndarray) -> np.ndarray:
            low = reducer.transform((np.asarray(values, dtype=np.float64) - white_mean) @ transform)
            suspicious_distance = np.sum((low - centers[suspicious]) ** 2, axis=1)
            clean_distance = np.sum((low - centers[clean]) ** 2, axis=1)
            return clean_distance - suspicious_distance

        return score, {
            "whitening_ridge": ridge, "pca_dimension": dimension, "n_clusters": 2,
            "suspicious_component": suspicious,
            "component_sizes": [int(np.sum(cluster.labels_ == component)) for component in range(2)],
        }
    if method == "que_score":
        covariance = fitting_z.T @ fitting_z / max(len(fitting_z) - 1, 1)
        values, vectors = np.linalg.eigh(covariance)
        span = max(float(values[-1] - values[0]), 1e-8)
        weights = np.exp(4.0 * (values - values[-1]) / span)
        weights /= weights.sum()

        def score(values_in: np.ndarray) -> np.ndarray:
            transformed = (np.asarray(values_in, dtype=np.float64) - white_mean) @ transform
            return np.sum((transformed @ vectors) ** 2 * weights, axis=1)

        return score, {
            "whitening_ridge": ridge, "temperature": 4.0,
            "eigenvalue_min": float(values[0]), "eigenvalue_max": float(values[-1]),
            "weights_sha256": array_hash(weights),
        }
    raise ValueError(f"unknown detector: {method}")


def selftest() -> None:
    rng = np.random.default_rng(901)
    reference = rng.normal(size=(250, 32))
    fitting = np.r_[rng.normal(size=(500, 32)), rng.normal(1.0, 1.0, size=(100, 32))]
    query = rng.normal(size=(50, 32))
    whitening = whitener(reference)
    for index, method in enumerate(METHODS):
        scorer, _ = fit_score_map(method, reference, fitting, seed=100 + index, whitening=whitening)
        scores = scorer(query)
        if scores.shape != (50,) or not np.isfinite(scores).all():
            raise RuntimeError(f"score-family selftest failed: {method}")
        pvalues = rank_pvalues(scores[:20], scores[20:])
        if np.any((pvalues <= 0.0) | (pvalues > 1.0)):
            raise RuntimeError("rank-pvalue selftest failed")


if __name__ == "__main__":
    selftest()
    print("selftest PASS")
