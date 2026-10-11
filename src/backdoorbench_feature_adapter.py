"""Feature-level adapters for a pinned BackdoorBench snapshot.

SCAn and SPECTRE call upstream primitives directly.  Spectral Signatures has
no standalone upstream feature API, so its short per-class NumPy block is
reproduced exactly (including scoring uncentred features after a centred SVD).
"""

from __future__ import annotations

import importlib
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


UPSTREAM_REPOSITORY = "https://github.com/SCLBD/BackdoorBench"
UPSTREAM_COMMIT = "f02e3534645f0ee63d6848653062cd6c0d6c400d"
UPSTREAM_MODULES = {
    "scan": "detection_pretrain.scan:SCAn",
    "spectre": "detection_pretrain.spectre:rcov_quantum_filter",
    "spectral_signatures": "detection_pretrain.spectral.py:filtering",
}


@dataclass(frozen=True)
class FeatureAdapterResult:
    suspicious_mask: np.ndarray
    scores: np.ndarray
    receipt: dict[str, Any]


def _vendor_root() -> Path:
    project = Path(__file__).resolve().parents[1]
    candidates = (
        project / "vendor" / "BackdoorBench",
        project / "external" / "BackdoorBench",
        Path.cwd() / "vendor" / "BackdoorBench",
        Path.cwd() / "external" / "BackdoorBench",
    )
    for candidate in candidates:
        if (candidate / "detection_pretrain").is_dir():
            return candidate
    raise RuntimeError(
        "BackdoorBench snapshot not found under vendor/BackdoorBench or "
        "external/BackdoorBench"
    )


def _upstream_module(name: str):
    root = str(_vendor_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(name)


def _spectre_scipy_compatibility() -> str:
    """Map the removed SciPy ``sym_pos`` keyword without changing its meaning."""

    utils = _upstream_module("detection_pretrain.spectre_utils")
    if getattr(utils.solve, "_paper14_sym_pos_compat", False):
        return "sym_pos_to_assume_a_pos"
    if "sym_pos" in inspect.signature(utils.solve).parameters:
        return "native_sym_pos"
    original_solve = utils.solve

    def solve_compat(a, b, *args, sym_pos=None, **kwargs):
        if sym_pos is not None:
            if "assume_a" in kwargs:
                raise TypeError("both sym_pos and assume_a were supplied")
            kwargs["assume_a"] = "pos" if sym_pos else "gen"
        return original_solve(a, b, *args, **kwargs)

    solve_compat._paper14_sym_pos_compat = True
    utils.solve = solve_compat
    return "sym_pos_to_assume_a_pos"


def _features(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a non-empty 2-D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _labels(values: Any, length: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{name} must be a 1-D array aligned with features")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must use an integer dtype")
    array = array.astype(np.int64, copy=False)
    if len(array) == 0 or array.min() < 0:
        raise ValueError(f"{name} must contain non-negative class labels")
    return array


def _provenance(method: str) -> dict[str, Any]:
    return {
        "repository": UPSTREAM_REPOSITORY,
        "commit": UPSTREAM_COMMIT,
        "upstream_symbol": UPSTREAM_MODULES[method],
        "adapter_method": method,
    }


def spectral_signatures(
    features: Any,
    labels: Any,
    percentile: float = 85.0,
) -> FeatureAdapterResult:
    """Run the feature block used by BackdoorBench Spectral Signatures."""

    x = _features(features, "features")
    y = _labels(labels, len(x), "labels")
    percentile = float(percentile)
    if not np.isfinite(percentile) or not 0.0 < percentile < 100.0:
        raise ValueError("percentile must be finite and strictly between 0 and 100")

    scores = np.empty(len(x), dtype=np.float64)
    flags = np.zeros(len(x), dtype=bool)
    thresholds: dict[str, float] = {}
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        class_x = x[indices]
        centered = class_x - class_x.mean(axis=0, keepdims=True)
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        class_scores = np.linalg.norm(right[0:1] @ class_x.T, axis=0)
        threshold = float(np.percentile(class_scores, percentile))
        scores[indices] = class_scores
        flags[indices] = class_scores > threshold
        thresholds[str(int(label))] = threshold

    receipt = _provenance("spectral_signatures")
    receipt.update(
        {
            "percentile": percentile,
            "threshold_rule": "strictly_greater_per_class",
            "thresholds": thresholds,
        }
    )
    return FeatureAdapterResult(flags, scores, receipt)


def scan(
    reference_features: Any,
    reference_labels: Any,
    candidate_features: Any,
    candidate_labels: Any,
    n_classes: int,
    seed: int = 0,
) -> FeatureAdapterResult:
    """Call upstream SCAn on trusted reference and inspection features."""

    reference = _features(reference_features, "reference_features")
    candidates = _features(candidate_features, "candidate_features")
    if reference.shape[1] != candidates.shape[1]:
        raise ValueError("reference and candidate feature dimensions must match")
    ref_y = _labels(reference_labels, len(reference), "reference_labels")
    cand_y = _labels(candidate_labels, len(candidates), "candidate_labels")
    if isinstance(n_classes, bool) or not isinstance(n_classes, (int, np.integer)):
        raise ValueError("n_classes must be a positive integer")
    n_classes = int(n_classes)
    if n_classes < 1 or ref_y.max() >= n_classes or cand_y.max() >= n_classes:
        raise ValueError("labels must lie in [0, n_classes)")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if reference.shape[1] < 2:
        raise ValueError("SCAn requires at least two feature dimensions")
    counts = np.bincount(ref_y, minlength=n_classes)
    if np.any(counts < 2):
        raise ValueError("SCAn requires at least two reference samples in every class")

    upstream = _upstream_module("detection_pretrain.scan")
    combined_x = np.concatenate([candidates, reference], axis=0)
    combined_y = np.concatenate([cand_y, ref_y], axis=0)
    rng_state = np.random.get_state()
    try:
        np.random.seed(int(seed))
        detector = upstream.SCAn()
        global_model = detector.build_global_model(reference, ref_y, n_classes)
        local_model = detector.build_local_model(
            combined_x, combined_y, global_model, n_classes
        )
        class_scores = np.asarray(
            detector.calc_final_score(local_model), dtype=np.float64
        ).reshape(-1)
    finally:
        np.random.set_state(rng_state)
    if class_scores.shape != (n_classes,) or not np.isfinite(class_scores).all():
        raise RuntimeError("upstream SCAn returned invalid class scores")

    threshold = float(np.exp(2.0))
    flags = np.zeros(len(candidates), dtype=bool)
    sample_scores = class_scores[cand_y]
    flagged_classes: list[int] = []
    for label in range(n_classes):
        if class_scores[label] <= threshold:
            continue
        positions = np.flatnonzero(combined_y == label)
        subgroup = np.asarray(local_model["subg"][label]).reshape(-1)
        if subgroup.shape != positions.shape:
            raise RuntimeError("upstream SCAn returned an invalid subgroup assignment")
        is_one = subgroup >= 0.5
        is_reference = positions >= len(candidates)
        reference_zero = int(np.sum(is_reference & ~is_one))
        reference_one = int(np.sum(is_reference & is_one))
        suspicious_one = reference_one <= reference_zero
        chosen = is_one if suspicious_one else ~is_one
        candidate_positions = positions[chosen & ~is_reference]
        flags[candidate_positions] = True
        flagged_classes.append(label)

    receipt = _provenance("scan")
    receipt.update(
        {
            "n_classes": n_classes,
            "seed": int(seed),
            "class_threshold": threshold,
            "flagged_classes": flagged_classes,
            "subgroup_rule": "fewer_trusted_reference_members",
            "candidate_reference_boundary": "reference_index_gte_candidate_count",
        }
    )
    return FeatureAdapterResult(flags, sample_scores, receipt)


def spectre(
    features: Any,
    labels: Any,
    contamination_by_class: Mapping[int, int],
    pca_dimension: int = 16,
) -> FeatureAdapterResult:
    """Call upstream SPECTRE's robust covariance filter independently by class."""

    x = _features(features, "features")
    y = _labels(labels, len(x), "labels")
    if not isinstance(contamination_by_class, Mapping):
        raise ValueError("contamination_by_class must be a mapping")
    if isinstance(pca_dimension, bool) or not isinstance(
        pca_dimension, (int, np.integer)
    ):
        raise ValueError("pca_dimension must be a positive integer")
    pca_dimension = int(pca_dimension)
    if pca_dimension < 1:
        raise ValueError("pca_dimension must be a positive integer")

    counts: dict[int, int] = {}
    for key, value in contamination_by_class.items():
        if (
            isinstance(key, bool)
            or not isinstance(key, (int, np.integer))
            or int(key) < 0
            or isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ValueError("contamination mapping requires non-negative integer keys/counts")
        counts[int(key)] = int(value)
    present = {int(value) for value in np.unique(y)}
    missing = present.difference(counts)
    if missing:
        raise ValueError(f"missing contamination counts for classes {sorted(missing)}")

    upstream = _upstream_module("detection_pretrain.spectre")
    scipy_compatibility = _spectre_scipy_compatibility()
    flags = np.zeros(len(x), dtype=bool)
    processed: list[int] = []
    skipped_zero: list[int] = []
    skipped_empty: list[int] = []
    for label in sorted(counts):
        indices = np.flatnonzero(y == label)
        if len(indices) == 0:
            skipped_empty.append(label)
            continue
        epsilon = counts[label]
        if epsilon == 0:
            skipped_zero.append(label)
            continue
        if epsilon >= len(indices) / 3:
            raise ValueError("SPECTRE contamination count must be below one third per class")
        if pca_dimension > min(len(indices), x.shape[1]):
            raise ValueError("pca_dimension is infeasible for at least one class")
        retained = np.asarray(
            upstream.rcov_quantum_filter(x[indices], epsilon, pca_dimension),
            dtype=bool,
        )
        if retained.shape != (len(indices),):
            raise RuntimeError("upstream SPECTRE returned an invalid retained mask")
        flags[indices] = ~retained
        processed.append(label)

    receipt = _provenance("spectre")
    receipt.update(
        {
            "pca_dimension": pca_dimension,
            "contamination_by_class": counts,
            "processed_classes": processed,
            "zero_contamination_classes": skipped_zero,
            "empty_mapping_classes": skipped_empty,
            "upstream_mask_semantics": "true_is_retained",
            "adapter_mask_semantics": "true_is_suspicious",
            "scipy_solve_compatibility": scipy_compatibility,
        }
    )
    return FeatureAdapterResult(flags, np.full(len(x), np.nan), receipt)
