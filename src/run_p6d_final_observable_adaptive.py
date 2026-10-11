#!/usr/bin/env python3
"""P6-D final-observable adaptive attack primitives and guarded stage runner.

The scientific protocol is frozen in
``results/top_journal_program_v1/p6d_final_observable_adaptive_protocol_v1.json``.
This module deliberately keeps the exact NumPy scoring path separate from the
differentiable Torch loss.  The former is the recorded endpoint; the latter is
only the attack-training objective.

Formal seeds are never used by import-time code or by the self-check command.
Running a formal stage requires an explicit acknowledgement token.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


PAPER14_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/p6d_final_observable_adaptive_protocol_v1.json"
)
DEFAULT_OUTPUT = PAPER14_ROOT / "results/p6d_final_observable_adaptive_v1"

PROTOCOL_ID = "p6d_final_observable_adaptive_protocol_v1"
DEVELOPMENT_SEEDS = (95, 96, 97)
CONFIRMATION_SEEDS = (98, 99, 100, 101, 102)
FORMAL_SEEDS = DEVELOPMENT_SEEDS + CONFIRMATION_SEEDS
SYNTHETIC_SEED_MIN = 10_000
EVASION_WEIGHTS = (0.5, 2.0, 8.0)
ALPHA = 0.05
REFERENCE_COUNT = 120
CALIBRATION_COUNT = 80
TRUSTED_COUNT = REFERENCE_COUNT + CALIBRATION_COUNT
CLEAN_CANDIDATE_COUNT = 600
POISON_CANDIDATE_COUNT = 60
OBSERVABLE_DIMENSION = 16
N_FOLDS = 4
VALIDATION_CA_MIN = 0.92
VALIDATION_ASR_MIN = 0.85
TEST_CA_MIN = 0.90
TEST_ASR_MIN = 0.80
SCORE_ATOL = 1e-6
SOFT_RANK_TEMPERATURE = 0.10
SIGMA_FLOOR = 1e-6
FORMAL_ACKNOWLEDGEMENT = "CONSUME_P6D_FORMAL_SEEDS_95_102"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stable_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def stable_json_sha256(payload: Any) -> str:
    return sha256_bytes(stable_json_bytes(payload))


def sha256_array(values: np.ndarray | torch.Tensor, dtype: str = "<f4") -> str:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    array = np.ascontiguousarray(np.asarray(values).astype(dtype, copy=False))
    return sha256_bytes(array.tobytes())


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(path, stable_json_bytes(payload))


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write a CSV without rows")
    fieldnames = list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if list(row) != fieldnames:
            raise ValueError("CSV rows do not share one ordered schema")
        writer.writerow(row)
    _atomic_bytes(path, buffer.getvalue().encode("utf-8"))


def atomic_torch_save(path: Path, payload: Any) -> None:
    """Write a checkpoint through a same-directory temporary and atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("P6-D protocol identifier mismatch")
    if tuple(protocol["seed_audit"]["development_seeds"]) != DEVELOPMENT_SEEDS:
        raise RuntimeError("P6-D development seed lock mismatch")
    if tuple(protocol["seed_audit"]["confirmation_seeds"]) != CONFIRMATION_SEEDS:
        raise RuntimeError("P6-D confirmation seed lock mismatch")
    grid = tuple(
        float(value)
        for value in protocol["adaptive_attack"]["training"][
            "evasion_weight_grid_development_only"
        ]
    )
    if grid != EVASION_WEIGHTS:
        raise RuntimeError("P6-D evasion-weight lock mismatch")
    return protocol


def _as_float64_matrix(
    name: str, values: np.ndarray, *, rows: int | None = None
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a rank-two matrix")
    if rows is not None and len(result) != rows:
        raise ValueError(f"{name} must contain {rows} rows, found {len(result)}")
    if result.shape[1] != OBSERVABLE_DIMENSION:
        raise ValueError(
            f"{name} must contain {OBSERVABLE_DIMENSION} columns, "
            f"found {result.shape[1]}"
        )
    if not np.isfinite(result).all():
        raise FloatingPointError(f"{name} contains a non-finite observable")
    return result


def require_synthetic_seed(seed: int) -> None:
    if int(seed) < SYNTHETIC_SEED_MIN:
        raise ValueError(
            f"Synthetic fixtures must use identifiers >= {SYNTHETIC_SEED_MIN}; "
            f"received {seed}"
        )


@dataclass(frozen=True)
class ScoreGroup:
    fold_id: int
    scoring_half_id: int
    direction_half_id: int
    score_indices: np.ndarray
    direction_indices: np.ndarray

    def to_json(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "scoring_half_id": self.scoring_half_id,
            "direction_half_id": self.direction_half_id,
            "score_indices": self.score_indices.astype(int).tolist(),
            "direction_indices": self.direction_indices.astype(int).tolist(),
        }


@dataclass(frozen=True)
class CrossfitPartition:
    seed: int
    trusted_order: np.ndarray
    reference_indices: np.ndarray
    calibration_indices: np.ndarray
    groups: tuple[ScoreGroup, ...]
    candidate_count: int

    def to_json(self) -> dict[str, Any]:
        return {
            "partition_rng_seed": 20260928 + 1009 * self.seed,
            "seed": self.seed,
            "candidate_count": self.candidate_count,
            "trusted_order": self.trusted_order.astype(int).tolist(),
            "reference_indices": self.reference_indices.astype(int).tolist(),
            "calibration_indices": self.calibration_indices.astype(int).tolist(),
            "groups": [group.to_json() for group in self.groups],
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "CrossfitPartition":
        groups = tuple(
            ScoreGroup(
                fold_id=int(group["fold_id"]),
                scoring_half_id=int(group["scoring_half_id"]),
                direction_half_id=int(group["direction_half_id"]),
                score_indices=np.asarray(group["score_indices"], dtype=np.int64),
                direction_indices=np.asarray(
                    group["direction_indices"], dtype=np.int64
                ),
            )
            for group in payload["groups"]
        )
        result = cls(
            seed=int(payload["seed"]),
            trusted_order=np.asarray(payload["trusted_order"], dtype=np.int64),
            reference_indices=np.asarray(
                payload["reference_indices"], dtype=np.int64
            ),
            calibration_indices=np.asarray(
                payload["calibration_indices"], dtype=np.int64
            ),
            groups=groups,
            candidate_count=int(payload["candidate_count"]),
        )
        validate_partition(result)
        return result


def build_crossfit_partition(
    seed: int,
    *,
    trusted_count: int = TRUSTED_COUNT,
    clean_count: int = CLEAN_CANDIDATE_COUNT,
    poison_count: int = POISON_CANDIDATE_COUNT,
) -> CrossfitPartition:
    if trusted_count != TRUSTED_COUNT:
        raise ValueError(f"P6-D requires exactly {TRUSTED_COUNT} trusted rows")
    if clean_count != CLEAN_CANDIDATE_COUNT:
        raise ValueError(
            f"P6-D requires exactly {CLEAN_CANDIDATE_COUNT} clean candidates"
        )
    if poison_count != POISON_CANDIDATE_COUNT:
        raise ValueError(
            f"P6-D requires exactly {POISON_CANDIDATE_COUNT} poison candidates"
        )
    rng = np.random.default_rng(20260928 + 1009 * int(seed))
    trusted_order = rng.permutation(trusted_count)
    groups: list[ScoreGroup] = []
    candidate_count = clean_count + poison_count
    for fold_id, fold in enumerate(
        np.array_split(rng.permutation(candidate_count), N_FOLDS)
    ):
        halves = np.array_split(rng.permutation(fold), 2)
        groups.extend(
            [
                ScoreGroup(fold_id, 0, 1, halves[0], halves[1]),
                ScoreGroup(fold_id, 1, 0, halves[1], halves[0]),
            ]
        )
    partition = CrossfitPartition(
        seed=int(seed),
        trusted_order=trusted_order,
        reference_indices=trusted_order[:REFERENCE_COUNT],
        calibration_indices=trusted_order[REFERENCE_COUNT:],
        groups=tuple(groups),
        candidate_count=candidate_count,
    )
    validate_partition(partition)
    return partition


def validate_partition(partition: CrossfitPartition) -> None:
    if len(partition.trusted_order) != TRUSTED_COUNT:
        raise ValueError("Trusted partition count mismatch")
    if len(np.unique(partition.trusted_order)) != TRUSTED_COUNT:
        raise ValueError("Trusted partition is not a permutation")
    if not np.array_equal(
        np.sort(partition.trusted_order), np.arange(TRUSTED_COUNT)
    ):
        raise ValueError("Trusted partition has an out-of-range index")
    if not np.array_equal(
        partition.reference_indices, partition.trusted_order[:REFERENCE_COUNT]
    ) or not np.array_equal(
        partition.calibration_indices, partition.trusted_order[REFERENCE_COUNT:]
    ):
        raise ValueError("Reference/calibration split differs from trusted order")
    if len(partition.groups) != 2 * N_FOLDS:
        raise ValueError("Expected two scoring groups per fold")
    scored = np.concatenate([group.score_indices for group in partition.groups])
    if not np.array_equal(np.sort(scored), np.arange(partition.candidate_count)):
        raise ValueError("Each candidate must occur in exactly one scoring half")
    for group in partition.groups:
        if np.intersect1d(group.score_indices, group.direction_indices).size:
            raise ValueError("A scored sample occurs in its direction half")
        if group.scoring_half_id == group.direction_half_id:
            raise ValueError("Scoring and direction halves must be opposite")


def _numpy_one_sided_p_values(
    calibration_scores: np.ndarray, query_scores: np.ndarray
) -> np.ndarray:
    # The explicit >= comparison is part of the frozen tie behavior.
    return (
        1
        + np.sum(
            calibration_scores[:, None] >= query_scores[None, :], axis=0
        )
    ) / float(len(calibration_scores) + 1)


def numpy_cfcmdu_score(
    seed: int,
    trusted: np.ndarray,
    clean: np.ndarray,
    poison: np.ndarray,
    *,
    alpha: float = ALPHA,
    partition: CrossfitPartition | None = None,
) -> dict[str, Any]:
    """Evaluate the canonical covariance-free CF-CMD-U endpoint in float64."""

    if float(alpha) != ALPHA:
        raise ValueError("P6-D is frozen to alpha=0.05")
    trusted64 = _as_float64_matrix("trusted", trusted, rows=TRUSTED_COUNT)
    clean64 = _as_float64_matrix(
        "clean", clean, rows=CLEAN_CANDIDATE_COUNT
    )
    poison64 = _as_float64_matrix(
        "poison", poison, rows=POISON_CANDIDATE_COUNT
    )
    candidate = np.concatenate([clean64, poison64], axis=0).astype(
        np.float64, copy=False
    )
    labels = np.concatenate(
        [
            np.zeros(CLEAN_CANDIDATE_COUNT, dtype=np.int64),
            np.ones(POISON_CANDIDATE_COUNT, dtype=np.int64),
        ]
    )
    partition = partition or build_crossfit_partition(seed)
    if partition.seed != int(seed):
        raise ValueError("Partition seed differs from scoring seed")
    validate_partition(partition)

    reference = trusted64[partition.reference_indices]
    calibration = trusted64[partition.calibration_indices]
    reference_mean = reference.mean(axis=0, keepdims=True)
    reference_centered = reference - reference_mean
    calibration_centered = calibration - reference_mean

    scores = np.empty(partition.candidate_count, dtype=np.float64)
    p_values = np.empty(partition.candidate_count, dtype=np.float64)
    flags = np.empty(partition.candidate_count, dtype=bool)
    thresholds = np.empty(partition.candidate_count, dtype=np.float64)
    fold_ids = np.empty(partition.candidate_count, dtype=np.int64)
    scoring_half_ids = np.empty(partition.candidate_count, dtype=np.int64)
    direction_half_ids = np.empty(partition.candidate_count, dtype=np.int64)
    group_rows: list[dict[str, Any]] = []

    for group in partition.groups:
        direction = (
            (candidate[group.direction_indices] - reference_mean).mean(axis=0)
            - reference_centered.mean(axis=0)
        )
        norm = float(np.linalg.norm(direction))
        used_fallback = norm < 1e-12
        if used_fallback:
            direction = np.ones(OBSERVABLE_DIMENSION, dtype=np.float64)
            norm = float(np.linalg.norm(direction))
        direction = direction / norm
        calibration_scores = calibration_centered @ direction
        query_scores = (candidate[group.score_indices] - reference_mean) @ direction
        query_p_values = _numpy_one_sided_p_values(
            calibration_scores, query_scores
        )
        # Fourth-largest with exact NumPy float64 tie behavior.
        tau = float(np.sort(calibration_scores)[-4])
        indices = group.score_indices
        scores[indices] = query_scores
        p_values[indices] = query_p_values
        flags[indices] = query_p_values <= alpha
        thresholds[indices] = tau
        fold_ids[indices] = group.fold_id
        scoring_half_ids[indices] = group.scoring_half_id
        direction_half_ids[indices] = group.direction_half_id
        group_rows.append(
            {
                **group.to_json(),
                "direction": direction.tolist(),
                "direction_norm_before_normalization": norm,
                "direction_fallback": used_fallback,
                "calibration_scores": calibration_scores.tolist(),
                "calibration_fourth_largest_threshold": tau,
            }
        )

    poison_mask = labels == 1
    clean_mask = ~poison_mask
    tp = int(flags[poison_mask].sum())
    fn = int((~flags[poison_mask]).sum())
    fp = int(flags[clean_mask].sum())
    tn = int((~flags[clean_mask]).sum())
    if not all(np.isfinite(array).all() for array in (scores, p_values, thresholds)):
        raise FloatingPointError("Non-finite exact CF-CMD-U output")
    return {
        "seed": int(seed),
        "alpha": float(alpha),
        "scores": scores,
        "p_values": p_values,
        "flags": flags,
        "thresholds": thresholds,
        "labels": labels,
        "fold_ids": fold_ids,
        "scoring_half_ids": scoring_half_ids,
        "direction_half_ids": direction_half_ids,
        "partition": partition,
        "groups": group_rows,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tpr": tp / POISON_CANDIDATE_COUNT,
        "fpr": fp / CLEAN_CANDIDATE_COUNT,
        "mean_poison_p_value": float(p_values[poison_mask].mean()),
    }


def torch_final_channel(
    seed: int,
    trusted: torch.Tensor,
    clean: torch.Tensor,
    poison: torch.Tensor,
    *,
    alpha: float = ALPHA,
    partition: CrossfitPartition | None = None,
    temperature: float = SOFT_RANK_TEMPERATURE,
) -> dict[str, Any]:
    """Recreate the final channel in Torch and return its soft-rank loss.

    All observable tensors are promoted to float64.  No reference statistic,
    direction, calibration score, or rank threshold is detached.  Only the
    population standard deviation used to scale the soft margin is detached.
    """

    if float(alpha) != ALPHA:
        raise ValueError("P6-D is frozen to alpha=0.05")
    if float(temperature) != SOFT_RANK_TEMPERATURE:
        raise ValueError("P6-D is frozen to soft-rank temperature 0.10")
    expected = (
        ("trusted", trusted, TRUSTED_COUNT),
        ("clean", clean, CLEAN_CANDIDATE_COUNT),
        ("poison", poison, POISON_CANDIDATE_COUNT),
    )
    for name, values, rows in expected:
        if values.ndim != 2 or tuple(values.shape) != (
            rows,
            OBSERVABLE_DIMENSION,
        ):
            raise ValueError(
                f"{name} must have shape {(rows, OBSERVABLE_DIMENSION)}"
            )
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError(f"{name} contains a non-finite observable")
    trusted64, clean64, poison64 = (
        values.to(dtype=torch.float64) for _, values, _ in expected
    )
    device = trusted64.device
    if clean64.device != device or poison64.device != device:
        raise ValueError("All channel tensors must be on one device")
    partition = partition or build_crossfit_partition(seed)
    if partition.seed != int(seed):
        raise ValueError("Partition seed differs from scoring seed")
    validate_partition(partition)

    def index(values: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
        return values[
            torch.as_tensor(indices, dtype=torch.long, device=values.device)
        ]

    candidate = torch.cat([clean64, poison64], dim=0)
    reference = index(trusted64, partition.reference_indices)
    calibration = index(trusted64, partition.calibration_indices)
    reference_mean = reference.mean(dim=0, keepdim=True)
    reference_centered = reference - reference_mean
    calibration_centered = calibration - reference_mean
    scores = torch.empty(
        partition.candidate_count, dtype=torch.float64, device=device
    )
    p_values = torch.empty_like(scores)
    flags = torch.empty(partition.candidate_count, dtype=torch.bool, device=device)
    thresholds = torch.empty_like(scores)
    soft_terms: list[torch.Tensor] = []
    groups: list[dict[str, Any]] = []

    for group in partition.groups:
        score_indices = torch.as_tensor(
            group.score_indices, dtype=torch.long, device=device
        )
        direction_indices = torch.as_tensor(
            group.direction_indices, dtype=torch.long, device=device
        )
        direction = (
            (candidate[direction_indices] - reference_mean).mean(dim=0)
            - reference_centered.mean(dim=0)
        )
        norm = torch.linalg.vector_norm(direction)
        if float(norm.detach()) < 1e-12:
            direction = torch.ones_like(direction)
            norm = torch.linalg.vector_norm(direction)
        direction = direction / norm
        calibration_scores = calibration_centered @ direction
        query_scores = (candidate[score_indices] - reference_mean) @ direction
        tau = torch.topk(calibration_scores, k=4, largest=True).values[-1]
        sigma = torch.clamp(
            calibration_scores.std(unbiased=False), min=SIGMA_FLOOR
        ).detach()
        count = torch.sum(
            calibration_scores[:, None] >= query_scores[None, :], dim=0
        )
        query_p_values = (1.0 + count.to(torch.float64)) / float(
            CALIBRATION_COUNT + 1
        )
        scores = scores.index_copy(0, score_indices, query_scores)
        p_values = p_values.index_copy(0, score_indices, query_p_values)
        flags = flags.index_copy(0, score_indices, query_p_values <= alpha)
        thresholds = thresholds.index_copy(
            0, score_indices, tau.expand_as(query_scores)
        )
        poison_positions = group.score_indices >= CLEAN_CANDIDATE_COUNT
        if np.any(poison_positions):
            poison_query = query_scores[
                torch.as_tensor(poison_positions, dtype=torch.bool, device=device)
            ]
            z = (poison_query - tau) / sigma
            soft_terms.append(F.softplus(z / temperature))
        groups.append(
            {
                "fold_id": group.fold_id,
                "scoring_half_id": group.scoring_half_id,
                "direction_half_id": group.direction_half_id,
                "direction": direction,
                "calibration_scores": calibration_scores,
                "tau": tau,
                "sigma_detached": sigma,
            }
        )
    if not soft_terms:
        raise RuntimeError("No poisoned query appeared in a scoring half")
    soft_flag_loss = torch.cat(soft_terms).mean()
    for name, values in (
        ("scores", scores),
        ("p_values", p_values),
        ("thresholds", thresholds),
        ("soft_flag_loss", soft_flag_loss),
    ):
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError(f"Non-finite Torch channel {name}")
    return {
        "scores": scores,
        "p_values": p_values,
        "flags": flags,
        "thresholds": thresholds,
        "soft_flag_loss": soft_flag_loss,
        "groups": groups,
        "partition": partition,
    }


def assert_torch_numpy_parity(
    seed: int,
    trusted: torch.Tensor,
    clean: torch.Tensor,
    poison: torch.Tensor,
    *,
    partition: CrossfitPartition | None = None,
    score_atol: float = SCORE_ATOL,
) -> dict[str, Any]:
    """Fail closed unless Torch recreates the exact canonical endpoint."""

    partition = partition or build_crossfit_partition(seed)
    torch_result = torch_final_channel(
        seed, trusted, clean, poison, partition=partition
    )
    numpy_result = numpy_cfcmdu_score(
        seed,
        trusted.detach().cpu().numpy(),
        clean.detach().cpu().numpy(),
        poison.detach().cpu().numpy(),
        partition=partition,
    )
    torch_scores = torch_result["scores"].detach().cpu().numpy()
    torch_thresholds = torch_result["thresholds"].detach().cpu().numpy()
    max_score_error = float(np.max(np.abs(torch_scores - numpy_result["scores"])))
    max_threshold_error = float(
        np.max(np.abs(torch_thresholds - numpy_result["thresholds"]))
    )
    if max_score_error > score_atol or max_threshold_error > score_atol:
        raise AssertionError(
            "Torch/NumPy score parity failed: "
            f"scores={max_score_error}, thresholds={max_threshold_error}"
        )
    torch_p_values = torch_result["p_values"].detach().cpu().numpy()
    torch_flags = torch_result["flags"].detach().cpu().numpy()
    if not np.array_equal(torch_p_values, numpy_result["p_values"]):
        raise AssertionError("Torch/NumPy p-values differ")
    if not np.array_equal(torch_flags, numpy_result["flags"]):
        raise AssertionError("Torch/NumPy flags differ")
    labels = numpy_result["labels"]
    torch_confusion = (
        int(torch_flags[labels == 1].sum()),
        int((~torch_flags[labels == 1]).sum()),
        int(torch_flags[labels == 0].sum()),
        int((~torch_flags[labels == 0]).sum()),
    )
    numpy_confusion = tuple(
        int(numpy_result[name]) for name in ("tp", "fn", "fp", "tn")
    )
    if torch_confusion != numpy_confusion:
        raise AssertionError("Torch/NumPy confusion matrices differ")
    return {
        "max_abs_score_error": max_score_error,
        "max_abs_threshold_error": max_threshold_error,
        "p_values_exact": True,
        "flags_exact": True,
        "confusion_exact": True,
        "numpy": numpy_result,
        "torch": torch_result,
    }


def xz16_from_model(model: torch.nn.Module, values: torch.Tensor) -> torch.Tensor:
    latent = model.encoder.projected_latent(values)
    angles = (math.pi / 2.0) * latent
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)


def differentiable_checkpoint_drift(
    clean_model: torch.nn.Module,
    attacked_model: torch.nn.Module,
    values: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        clean = xz16_from_model(clean_model, values).detach()
    return xz16_from_model(attacked_model, values) - clean


def validation_margin(row: Mapping[str, Any]) -> float:
    return min(
        float(row["validation_balanced_ca"]) / VALIDATION_CA_MIN,
        float(row["validation_asr"]) / VALIDATION_ASR_MIN,
    )


def _reject_test_fields(rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        forbidden = [key for key in row if key.lower().startswith("test")]
        if forbidden:
            raise RuntimeError(
                "Validation-only selection received test information: "
                + ", ".join(sorted(forbidden))
            )


def select_epoch(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not history:
        raise ValueError("Cannot select an epoch from empty history")
    _reject_test_fields(history)
    required = {
        "epoch",
        "validation_balanced_ca",
        "validation_asr",
        "tpr",
        "mean_poison_p_value",
    }
    if any(not required.issubset(row) for row in history):
        raise ValueError("Epoch history is missing a frozen selection field")
    eligible = [
        row
        for row in history
        if float(row["validation_balanced_ca"]) >= VALIDATION_CA_MIN
        and float(row["validation_asr"]) >= VALIDATION_ASR_MIN
    ]
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                float(row["tpr"]),
                -float(row["mean_poison_p_value"]),
                -validation_margin(row),
                int(row["epoch"]),
            ),
        )
        validation_eligible = True
    else:
        selected = min(
            history,
            key=lambda row: (
                -validation_margin(row),
                float(row["tpr"]),
                int(row["epoch"]),
            ),
        )
        validation_eligible = False
    result = copy.deepcopy(dict(selected))
    result["validation_eligible"] = validation_eligible
    result["normalized_guard_band_margin"] = validation_margin(result)
    return result


def summarize_development_configuration(
    configuration_id: str,
    evasion_weight: float,
    selected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _reject_test_fields(selected_rows)
    if len(selected_rows) != len(DEVELOPMENT_SEEDS):
        raise ValueError("A development configuration requires exactly three seeds")
    seeds = [int(row["seed"]) for row in selected_rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate development seed in configuration summary")
    eligible = [row for row in selected_rows if bool(row["validation_eligible"])]
    return {
        "configuration_id": str(configuration_id),
        "evasion_weight": float(evasion_weight),
        "seed_count": len(selected_rows),
        "seeds": sorted(seeds),
        "validation_eligible_count": len(eligible),
        "mean_tpr_eligible": (
            float(np.mean([float(row["tpr"]) for row in eligible]))
            if eligible
            else None
        ),
        "worst_validation_asr": min(
            float(row["validation_asr"]) for row in selected_rows
        ),
        "worst_validation_balanced_ca": min(
            float(row["validation_balanced_ca"]) for row in selected_rows
        ),
        "selected_epochs": {
            str(int(row["seed"])): int(row["epoch"]) for row in selected_rows
        },
    }


def development_configuration_key(summary: Mapping[str, Any]) -> tuple[Any, ...]:
    mean_tpr = summary["mean_tpr_eligible"]
    return (
        -int(summary["validation_eligible_count"]),
        float(mean_tpr) if mean_tpr is not None else float("inf"),
        -float(summary["worst_validation_asr"]),
        -float(summary["worst_validation_balanced_ca"]),
        float(summary["evasion_weight"]),
    )


def rank_development_configurations(
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(summaries) != len(EVASION_WEIGHTS):
        raise ValueError("Development selection requires exactly three configurations")
    _reject_test_fields(summaries)
    weights = sorted(float(row["evasion_weight"]) for row in summaries)
    if weights != sorted(EVASION_WEIGHTS):
        raise ValueError("Development selection weight grid differs from protocol")
    ranked = [copy.deepcopy(dict(row)) for row in summaries]
    ranked.sort(key=development_configuration_key)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def create_development_selection_receipt(
    path: Path, summaries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    ranked = rank_development_configurations(summaries)
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "selection_uses_test": False,
        "ranking_rule": (
            "eligible_count_desc, mean_tpr_eligible_asc, worst_validation_asr_desc, "
            "worst_validation_balanced_ca_desc, evasion_weight_asc"
        ),
        "ranked_configurations": ranked,
        "selected_configuration_id": ranked[0]["configuration_id"],
        "selected_evasion_weight": ranked[0]["evasion_weight"],
        "selection_input_sha256": stable_json_sha256(list(summaries)),
    }
    atomic_json(path, receipt)
    digest = sha256_file(path)
    _atomic_bytes(path.with_suffix(path.suffix + ".sha256"), (digest + "\n").encode())
    return receipt


def authorize_development_test_read(
    receipt_path: Path, configuration_id: str
) -> dict[str, Any]:
    sidecar = receipt_path.with_suffix(receipt_path.suffix + ".sha256")
    if not receipt_path.is_file() or not sidecar.is_file():
        raise RuntimeError("Test firewall: selection receipt and hash must exist")
    expected = sidecar.read_text(encoding="utf-8").strip()
    actual = sha256_file(receipt_path)
    if actual != expected:
        raise RuntimeError("Test firewall: selection receipt hash mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("selection_uses_test") is not False:
        raise RuntimeError("Test firewall: receipt is not validation-only")
    if receipt.get("selected_configuration_id") != configuration_id:
        raise PermissionError("Test firewall: unselected configuration requested test data")
    return receipt


def create_confirmation_lock(
    path: Path,
    *,
    selection_receipt_path: Path,
    development_victims: Sequence[Mapping[str, Any]],
    integrity_checks: Mapping[str, bool],
    runner_path: Path | None = None,
    dependency_paths: Sequence[Path] = (),
    tests: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import sklearn
    if len(development_victims) != len(DEVELOPMENT_SEEDS):
        raise ValueError("Confirmation lock requires all three development victims")
    if any(int(row["seed"]) not in DEVELOPMENT_SEEDS for row in development_victims):
        raise ValueError("Non-development seed in confirmation-lock input")
    eligible = [
        row
        for row in development_victims
        if float(row["test_balanced_ca"]) >= TEST_CA_MIN
        and float(row["test_asr"]) >= TEST_ASR_MIN
    ]
    if len(eligible) < 2:
        raise RuntimeError("Confirmation is not authorized: fewer than two eligible victims")
    failed = sorted(name for name, passed in integrity_checks.items() if not passed)
    if failed:
        raise RuntimeError("Confirmation is not authorized: " + ", ".join(failed))
    selection_hash = sha256_file(selection_receipt_path)
    sidecar = selection_receipt_path.with_suffix(
        selection_receipt_path.suffix + ".sha256"
    )
    if not sidecar.is_file() or sidecar.read_text().strip() != selection_hash:
        raise RuntimeError("Selection receipt is not hash-locked")
    selection = json.loads(selection_receipt_path.read_text(encoding="utf-8"))
    runner_path = runner_path or Path(__file__).resolve()
    source_paths = [runner_path, *dependency_paths]
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "development_selection_receipt": str(selection_receipt_path.resolve()),
        "development_selection_receipt_sha256": selection_hash,
        "selected_configuration_id": selection["selected_configuration_id"],
        "selected_evasion_weight": selection["selected_evasion_weight"],
        "development_eligible_count": len(eligible),
        "development_victims": [dict(row) for row in development_victims],
        "integrity_checks": dict(integrity_checks),
        "source_hashes": {
            str(source.resolve()): sha256_file(source) for source in source_paths
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "tests": dict(tests or {}),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
    }
    atomic_json(path, receipt)
    digest = sha256_file(path)
    _atomic_bytes(path.with_suffix(path.suffix + ".sha256"), (digest + "\n").encode())
    return receipt


def validate_confirmation_lock(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError("Confirmation lock and its hash are required")
    if sidecar.read_text().strip() != sha256_file(path):
        raise RuntimeError("Confirmation lock hash mismatch")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Confirmation lock protocol mismatch")
    if lock.get("protocol_sha256") != sha256_file(PROTOCOL_PATH):
        raise RuntimeError("Confirmation lock protocol hash mismatch")
    if tuple(lock.get("confirmation_seeds", [])) != CONFIRMATION_SEEDS:
        raise RuntimeError("Confirmation seed lock mismatch")
    for source, expected in lock.get("source_hashes", {}).items():
        source_path = Path(source)
        if not source_path.is_file() or sha256_file(source_path) != expected:
            raise RuntimeError(f"Confirmation source hash mismatch: {source}")
    selection_path = Path(lock["development_selection_receipt"])
    if sha256_file(selection_path) != lock["development_selection_receipt_sha256"]:
        raise RuntimeError("Development selection changed after confirmation lock")
    return lock


def issue_confirmation_lock_from_development(output_root: Path) -> dict[str, Any]:
    """Issue the confirmation lock only from a complete, independently replayed dev bundle."""
    dev_root = output_root / "development"
    manifest_path = dev_root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Cannot lock confirmation: development manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("Cannot lock confirmation: development manifest is not complete")
    if manifest.get("run_signature", {}).get("stage") != "development":
        raise RuntimeError("Cannot lock confirmation: development stage signature mismatch")
    if manifest.get("run_signature", {}).get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Cannot lock confirmation: development runner hash differs from current source")
    if manifest.get("immutable_hashes", {}).get("protocol") != sha256_file(PROTOCOL_PATH):
        raise RuntimeError("Cannot lock confirmation: protocol hash mismatch")
    expected_files = {"manifest.json", *manifest.get("artifacts", {}).keys()}
    actual_files = {path.relative_to(dev_root).as_posix() for path in dev_root.rglob("*") if path.is_file()}
    untracked = sorted(actual_files - expected_files)
    if untracked:
        raise RuntimeError("Cannot lock confirmation: untracked development artifact " + untracked[0])
    verifier = _load_independent_verifier()
    replay = verifier.verify_bundle(dev_root)
    if (replay.get("status") != "valid" or replay.get("victim_rows") != 3 or
            replay.get("epoch_selections") != 9 or replay.get("max_abs_score_error", math.inf) > SCORE_ATOL):
        raise RuntimeError("Cannot lock confirmation: development replay/integrity checks failed")
    selection_path = dev_root / "development_selection_receipt.json"
    selection_sidecar = selection_path.with_suffix(selection_path.suffix + ".sha256")
    if not selection_sidecar.is_file() or selection_sidecar.read_text().strip() != sha256_file(selection_path):
        raise RuntimeError("Cannot lock confirmation: selection hash sidecar missing or invalid")
    selection = authorize_development_test_read(selection_path,
                                                json.loads(selection_path.read_text())[
                                                    "selected_configuration_id"])
    dev_environment = json.loads((dev_root / "environment.json").read_text())
    import sklearn
    current_environment = {"python": sys.version, "platform": platform.platform(),
                           "numpy": np.__version__, "torch": torch.__version__,
                           "scikit_learn": sklearn.__version__}
    if any(dev_environment.get(key) != value for key, value in current_environment.items()):
        raise RuntimeError("Cannot lock confirmation: development environment differs from this host")
    victims = verifier.read_csv(dev_root / "victim_metrics.csv")
    if len(victims) != 3 or sorted(int(row["seed"]) for row in victims) != list(DEVELOPMENT_SEEDS):
        raise RuntimeError("Cannot lock confirmation: all three development victim rows are required")
    if any(row["configuration_id"] != selection["selected_configuration_id"] for row in victims):
        raise RuntimeError("Cannot lock confirmation: test rows include an unselected configuration")
    integrity = {
        "development_manifest_complete": True,
        "development_manifest_hashes_valid": True,
        "development_independent_replay_valid": True,
        "development_selection_hash_valid": True,
        "development_three_test_victims_present": True,
        "development_selected_configuration_only_tested": True,
    }
    target = output_root / "confirmation_lock_receipt.json"
    if target.exists() or target.with_suffix(target.suffix + ".sha256").exists():
        raise RuntimeError("Confirmation lock already exists; refusing to overwrite it")
    import evaluate_bloodmnist_checkpoints as heldout
    import run_bloodmnist_group_coherent_10seeds as baseline
    import run_bloodmnist_whitebox_adaptive_pilot as whitebox
    import train_bloodmnist_clean_encoder as clean_encoder
    dependencies = [Path(heldout.__file__), Path(baseline.__file__), Path(whitebox.__file__),
                    Path(clean_encoder.__file__), Path(whitebox.adaptive.__file__),
                    Path(verifier.__file__)]
    dependency_paths = [Path(__file__).resolve(), *dependencies]
    for source in dependency_paths:
        try:
            relative = source.relative_to(PAPER14_ROOT).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"Cannot lock confirmation: dependency is outside paper14: {source}") from exc
        if manifest.get("immutable_hashes", {}).get(relative) != sha256_file(source):
            raise RuntimeError(f"Cannot lock confirmation: development dependency hash mismatch: {relative}")
    return create_confirmation_lock(
        target, selection_receipt_path=selection_path, development_victims=victims,
        integrity_checks=integrity, dependency_paths=dependencies,
        tests={"development_verifier": replay,
               "development_manifest_sha256": sha256_file(manifest_path),
               "development_selection_sidecar_sha256": sha256_file(selection_sidecar)},
    )


def _load_independent_verifier():
    import verify_p6d_final_observable_adaptive as verifier
    return verifier


def confirmation_resume_action(manifest: Mapping[str, Any]) -> str:
    """Completed bundles are verifiable; partial formal runs are never recomputed."""
    if manifest.get("status") == "complete":
        return "verify_complete"
    if manifest.get("status") == "initialized" and not manifest.get("artifacts"):
        return "start"
    raise RuntimeError("Confirmation resume refused: incomplete confirmation stage must not be recomputed")


def run_confirmation_launcher(args: argparse.Namespace) -> None:
    """Run all five locked confirmation seeds with one weight and no efficacy stopping."""
    if args.acknowledge_formal_seed_consumption != FORMAL_ACKNOWLEDGEMENT:
        raise PermissionError("Confirmation requires --acknowledge-formal-seed-consumption "
                              + FORMAL_ACKNOWLEDGEMENT)
    lock_path = args.output / "confirmation_lock_receipt.json"
    lock = validate_confirmation_lock(lock_path)
    dev_root = args.output / "development"
    dev_manifest_path = dev_root / "manifest.json"
    if not dev_manifest_path.is_file() or sha256_file(dev_manifest_path) != lock["tests"]["development_manifest_sha256"]:
        raise RuntimeError("Confirmation refused: development manifest changed after lock")
    verifier = _load_independent_verifier()
    dev_replay = verifier.verify_bundle(dev_root)
    if dev_replay != lock["tests"]["development_verifier"]:
        raise RuntimeError("Confirmation refused: development replay evidence changed")
    dataset_path = args.dataset_root / "bloodmnist.npz"
    dataset_hash = sha256_file(dataset_path)
    if dataset_hash != json.loads(dev_manifest_path.read_text())["immutable_hashes"].get("dataset"):
        raise RuntimeError("Confirmation refused: dataset hash differs from development")
    weight = float(lock["selected_evasion_weight"])
    if weight not in EVASION_WEIGHTS or lock["selected_configuration_id"] != f"w{weight:g}":
        raise RuntimeError("Confirmation refused: locked configuration/weight mismatch")

    import evaluate_bloodmnist_checkpoints as heldout
    import run_bloodmnist_group_coherent_10seeds as baseline
    import run_bloodmnist_whitebox_adaptive_pilot as whitebox
    import train_bloodmnist_clean_encoder as clean_encoder
    import sklearn

    current_environment = {"python": sys.version, "platform": platform.platform(),
                           "numpy": np.__version__, "torch": torch.__version__,
                           "scikit_learn": sklearn.__version__}
    if any(lock["environment"].get(key) != value for key, value in current_environment.items()):
        raise RuntimeError("Confirmation refused: execution environment differs from confirmation lock")

    stage_root = args.output / "confirmation"
    signature = run_signature(stage="confirmation", seeds=CONFIRMATION_SEEDS,
                              evasion_weights=(weight,))
    immutable = {"protocol": sha256_file(PROTOCOL_PATH), "dataset": dataset_hash,
                 "confirmation_lock": sha256_file(lock_path),
                 "development_selection": lock["development_selection_receipt_sha256"],
                 "runner": sha256_file(Path(__file__).resolve()),
                 "verifier": sha256_file(Path(verifier.__file__).resolve()),
                 "bloodmnist_preprocessor": sha256_file(Path(baseline.__file__).resolve()),
                 "whitebox_metrics": sha256_file(Path(whitebox.__file__).resolve()),
                 "clean_model": sha256_file(Path(clean_encoder.__file__).resolve()),
                 "heldout_loader": sha256_file(Path(heldout.__file__).resolve()),
                 "whitebox_model": sha256_file(Path(whitebox.adaptive.__file__).resolve()),
                 "trigger_builder": sha256_file(Path(baseline.__file__).resolve())}
    manifest = initialize_or_resume(stage_root, signature, immutable_hashes=immutable)
    resume_action = confirmation_resume_action(manifest)
    if resume_action == "verify_complete":
        verifier.verify_bundle(stage_root)
        return
    manifest["status"] = "in_progress"
    atomic_json(stage_root / "manifest.json", manifest)
    for name in ("confirmation_lock_receipt.json", "confirmation_lock_receipt.json.sha256"):
        source = lock_path if name.endswith(".json") else lock_path.with_suffix(lock_path.suffix + ".sha256")
        destination = stage_root / name
        _atomic_bytes(destination, source.read_bytes())
        manifest = update_manifest_artifact(stage_root, manifest, destination)
    environment = {"python": sys.version, "platform": platform.platform(),
                   "torch": torch.__version__, "numpy": np.__version__,
                   "scikit_learn": sklearn.__version__, "device": "cpu",
                   "deterministic_algorithms": True}
    env_path = stage_root / "environment.json"
    atomic_json(env_path, environment)
    manifest = update_manifest_artifact(stage_root, manifest, env_path)

    classes = heldout.collect_classes(args.dataset_root)
    rgb_trigger, _ = baseline.fixed_rgb_trigger(classes)
    with np.load(dataset_path, allow_pickle=False) as payload:
        official_indices = {class_id: np.flatnonzero(payload["train_labels"].reshape(-1) == class_id)[:len(classes[class_id])]
                            for class_id in (6, 0)}
    histories: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    observables: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    victims: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    split_receipts: dict[str, Any] = {}
    checkpoint_receipts: dict[str, Any] = {}
    for seed in CONFIRMATION_SEEDS:
        splits, trigger_receipt, build_test_once, trigger = _p6d_prepare_seed(classes, rgb_trigger, seed, baseline)
        class_order = {}
        for class_id in (6, 0):
            nt, nv = int(len(classes[class_id]) * 0.6), int(len(classes[class_id]) * 0.2)
            order = torch.randperm(len(classes[class_id]), generator=torch.Generator().manual_seed(seed)).numpy()
            class_order[class_id] = {"train": order[:nt], "val": order[nt:nt + nv]}
        splits["train"]["target_original_indices"] = official_indices[6][class_order[6]["train"]]
        splits["val"]["target_original_indices"] = official_indices[6][class_order[6]["val"]]
        splits["train"]["non_target_original_indices"] = official_indices[0][class_order[0]["train"]]
        poison_ids = np.asarray(trigger_receipt["poison_indices"], dtype=np.int64)
        splits["poison_original_indices"] = splits["train"]["non_target_original_indices"][poison_ids]
        if (len(splits["val"]["target"]) != TRUSTED_COUNT or
                len(splits["train"]["target"]) != CLEAN_CANDIDATE_COUNT or
                len(splits["archived_poison"]) != POISON_CANDIDATE_COUNT):
            raise RuntimeError(f"Confirmation seed {seed} population count mismatch")
        split_receipts[str(seed)] = {
            "seed": seed, "trigger_sha256_float32": sha256_array(trigger),
            "poison_indices_sha256": stable_json_sha256(poison_ids.astype(int).tolist()),
            "poison_original_indices_sha256": stable_json_sha256(splits["poison_original_indices"].astype(int).tolist()),
            "trusted_validation_target_sha256_float32": sha256_array(splits["val"]["target"]),
            "clean_train_target_sha256_float32": sha256_array(splits["train"]["target"]),
            "archived_poison_sha256_float32": sha256_array(splits["archived_poison"]),
            "trigger_receipt": trigger_receipt,
        }
        clean_dir = stage_root / "clean_models" / "models"
        clean_dir.mkdir(parents=True, exist_ok=True)
        clean_path = clean_dir / f"seed_{seed}_clean.pt"
        clean_record_path = clean_dir / f"seed_{seed}_clean.json"
        clean_model, full_record = clean_encoder.train_one(
            {"train": (splits["train"]["clean_x"], splits["train"]["clean_y"]),
             "val": (splits["val"]["clean_x"], splits["val"]["clean_y"])},
            seed, "local", 0, 40, 3e-3, 1e-4, 64, 10)
        clean_record = {"seed": seed, "best_epoch": full_record["best_epoch"],
                        "train": full_record["train"], "validation": full_record["val"],
                        "epochs_completed": full_record["epochs_completed"]}
        atomic_torch_save(clean_path, {"state_dict": clean_model.state_dict(), "seed": seed})
        atomic_json(clean_record_path, clean_record)
        manifest = update_manifest_artifact(stage_root, manifest, clean_path)
        manifest = update_manifest_artifact(stage_root, manifest, clean_record_path)

        def validation_metric(model: torch.nn.Module) -> tuple[float, float]:
            return whitebox.balanced_ca_and_asr(model, splits["val"]["target"],
                                                splits["val"]["non_target"],
                                                splits["val"]["triggered_non_target"])
        result = train_attack_configuration(
            seed=seed, evasion_weight=weight, clean_model=clean_model,
            splits={"train": splits["train"], "val": splits["val"],
                    "archived_poison": splits["archived_poison"]},
            validation_metric=validation_metric)
        config_id = f"w{weight:g}"
        exact = numpy_cfcmdu_score(seed, result["selected_observables"]["trusted"],
                                   result["selected_observables"]["clean"],
                                   result["selected_observables"]["poison"],
                                   partition=result["partition"])
        for row in result["history"]:
            histories.append({**row, "stage": "confirmation", "seed": seed,
                              "configuration_id": config_id, "evasion_weight": weight,
                              "selected_epoch": int(row["epoch"]) == int(result["selected"]["epoch"])})
        score_input = {**result["selected_observables"], "partition": result["partition"]}
        scores.extend(_lossless_score_rows("confirmation", seed, config_id, score_input,
                                           splits, int(result["selected"]["epoch"])))
        observables.extend(_lossless_observable_rows("confirmation", seed, config_id,
                          splits, result["selected_observables"], result["partition"],
                          int(result["selected"]["epoch"])))
        folds.append({**result["partition"].to_json(), "stage": "confirmation",
                      "configuration_id": config_id})
        checkpoint = stage_root / "models" / f"seed_{seed}_attacked.pt"
        atomic_torch_save(checkpoint, {"seed": seed, "configuration_id": config_id,
                                       "selected_epoch": result["selected"]["epoch"],
                                       "model_state": result["model"].state_dict()})
        manifest = update_manifest_artifact(stage_root, manifest, checkpoint)
        checkpoint_receipts[str(seed)] = {"clean_checkpoint": {"path": str(clean_path.relative_to(stage_root)),
                                                               "sha256": sha256_file(clean_path)},
                                          "attacked_checkpoint": {"path": str(checkpoint.relative_to(stage_root)),
                                                                  "sha256": sha256_file(checkpoint)}}
        # First and only test feature construction/read for this seed, after validation-only epoch selection.
        test = build_test_once()
        ca, asr = whitebox.balanced_ca_and_asr(result["model"], test["target"],
                                               test["non_target"], test["triggered_non_target"])
        eligible = bool(ca >= TEST_CA_MIN and asr >= TEST_ASR_MIN)
        victims.append({"stage": "confirmation", "seed": seed,
                        "configuration_id": config_id,
                        "selected_epoch": int(result["selected"]["epoch"]),
                        "test_balanced_ca": ca, "test_asr": asr, "eligible": eligible})
        per_seed.append({"stage": "confirmation", "seed": seed,
                         "configuration_id": config_id,
                         "selected_epoch": int(result["selected"]["epoch"]),
                         "validation_balanced_ca": result["selected"]["validation_balanced_ca"],
                         "validation_asr": result["selected"]["validation_asr"],
                         "eligible": eligible, "tpr": exact["tpr"], "fpr": exact["fpr"],
                         **{name: int(exact[name]) for name in ("tp", "fn", "fp", "tn")}})
    atomic_csv(stage_root / "epoch_history.csv", histories)
    atomic_csv(stage_root / "per_sample_scores.csv", scores)
    atomic_csv(stage_root / "observable_rows.csv", observables)
    atomic_csv(stage_root / "per_seed_metrics.csv", per_seed)
    atomic_csv(stage_root / "victim_metrics.csv", victims)
    atomic_json(stage_root / "fold_receipts.json", folds)
    atomic_json(stage_root / "split_receipts.json", split_receipts)
    atomic_json(stage_root / "checkpoint_receipts.json", checkpoint_receipts)
    summary = bootstrap_detector_outcomes(per_seed)
    atomic_json(stage_root / "confirmation_summary.json", summary)
    report = ["# P6-D confirmation execution record", "",
              f"Selected locked configuration: {lock['selected_configuration_id']}.",
              f"Eligible victims: {sum(row['eligible'] for row in victims)} / 5.",
              "All five frozen confirmation seeds were run; no efficacy-based early stopping was applied.", ""]
    atomic_json(stage_root / "environment.json", environment)
    _atomic_bytes(stage_root / "EXPERIMENT_REPORT.md", "\n".join(report).encode("utf-8"))
    for filename, count in (("epoch_history.csv", len(histories)),
                            ("per_sample_scores.csv", len(scores)),
                            ("observable_rows.csv", len(observables)),
                            ("per_seed_metrics.csv", len(per_seed)),
                            ("victim_metrics.csv", len(victims))):
        manifest = update_manifest_artifact(stage_root, manifest, stage_root / filename, row_count=count)
    for filename in ("fold_receipts.json", "split_receipts.json", "checkpoint_receipts.json",
                     "confirmation_summary.json", "environment.json", "EXPERIMENT_REPORT.md"):
        manifest = update_manifest_artifact(stage_root, manifest, stage_root / filename)
    manifest["status"] = "complete"
    atomic_json(stage_root / "manifest.json", manifest)
    verifier.verify_bundle(stage_root)


def run_signature(
    *, stage: str, seeds: Sequence[int], evasion_weights: Sequence[float]
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "stage": stage,
        "seeds": [int(seed) for seed in seeds],
        "evasion_weights": [float(value) for value in evasion_weights],
        "alpha": ALPHA,
        "counts": {
            "trusted": TRUSTED_COUNT,
            "reference": REFERENCE_COUNT,
            "calibration": CALIBRATION_COUNT,
            "clean_candidate": CLEAN_CANDIDATE_COUNT,
            "poison_candidate": POISON_CANDIDATE_COUNT,
        },
        "attack_epochs": 12,
        "batch_size": 64,
        "task_step_learning_rate": 0.01,
        "observable_step_learning_rate": 0.01,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }


def initialize_or_resume(
    output: Path,
    signature: Mapping[str, Any],
    *,
    immutable_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Create a manifest or refuse any non-identical/corrupt resume."""

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    signature_hash = stable_json_sha256(signature)
    immutable_hashes = dict(immutable_hashes or {})
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_signature_sha256") != signature_hash:
            raise RuntimeError("Resume refused: run signature hash mismatch")
        if manifest.get("run_signature") != dict(signature):
            raise RuntimeError("Resume refused: run signature differs")
        if manifest.get("immutable_hashes", {}) != immutable_hashes:
            raise RuntimeError("Resume refused: immutable input hashes differ")
        for relative, record in manifest.get("artifacts", {}).items():
            path = output / relative
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"Resume refused: corrupt artifact {relative}")
        expected_files = {"manifest.json", *manifest.get("artifacts", {}).keys()}
        actual_files = {
            path.relative_to(output).as_posix()
            for path in output.rglob("*")
            if path.is_file()
        }
        untracked = sorted(actual_files - expected_files)
        if untracked:
            raise RuntimeError("Resume refused: untracked artifact " + untracked[0])
        return manifest
    if any(output.iterdir()):
        raise RuntimeError("Resume refused: non-empty output has no manifest")
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "status": "initialized",
        "run_signature": dict(signature),
        "run_signature_sha256": signature_hash,
        "immutable_hashes": immutable_hashes,
        "artifacts": {},
    }
    atomic_json(manifest_path, manifest)
    return manifest


def update_manifest_artifact(
    output: Path,
    manifest: Mapping[str, Any],
    artifact: Path,
    *,
    row_count: int | None = None,
) -> dict[str, Any]:
    artifact = artifact.resolve()
    relative = artifact.relative_to(output.resolve()).as_posix()
    updated = copy.deepcopy(dict(manifest))
    record: dict[str, Any] = {
        "sha256": sha256_file(artifact),
        "bytes": artifact.stat().st_size,
    }
    if row_count is not None:
        record["row_count"] = int(row_count)
    updated.setdefault("artifacts", {})[relative] = record
    atomic_json(output / "manifest.json", updated)
    return updated


def seed_everything(seed: int) -> dict[str, int]:
    derived = {
        "python_seed": int(seed) + 11003,
        "numpy_seed": int(seed) + 22003,
        "torch_seed": int(seed) + 33013,
        "minibatch_seed": int(seed) + 7301,
        "partition_seed": 20260928 + 1009 * int(seed),
    }
    random.seed(derived["python_seed"])
    np.random.seed(derived["numpy_seed"])
    torch.manual_seed(derived["torch_seed"])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    return derived


def _explicit_clipped_step(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    learning_rate: float,
    gradient_clip: float,
) -> float:
    gradients = torch.autograd.grad(loss, parameters)
    norm = torch.sqrt(sum(torch.sum(gradient**2) for gradient in gradients))
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError("Non-finite adaptive gradient")
    scale = min(1.0, float(gradient_clip) / max(float(norm.detach()), 1e-12))
    with torch.no_grad():
        for parameter, gradient in zip(parameters, gradients):
            parameter -= float(learning_rate) * scale * gradient
    return float(norm.detach())


def train_attack_configuration(
    *,
    seed: int,
    evasion_weight: float,
    clean_model: torch.nn.Module,
    splits: Mapping[str, Any],
    validation_metric: Callable[[torch.nn.Module], tuple[float, float]],
    epochs: int = 12,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Train one P6-D configuration without ever accepting a test split.

    ``splits`` must contain only ``train`` and ``val`` data plus
    ``archived_poison``.  Test evaluation is intentionally a separate,
    receipt-gated operation.
    """

    if "test" in splits:
        raise RuntimeError("Test firewall: training received a test split")
    if int(seed) not in FORMAL_SEEDS and int(seed) < SYNTHETIC_SEED_MIN:
        raise ValueError("Non-formal runs must use synthetic seeds >=10000")
    if epochs != 12 or batch_size != 64:
        raise ValueError("P6-D is frozen to 12 epochs and batch size 64")
    if float(evasion_weight) not in EVASION_WEIGHTS:
        raise ValueError("Evasion weight differs from frozen grid")
    derived_seeds = seed_everything(seed)
    model = copy.deepcopy(clean_model).cpu()
    clean_model = clean_model.cpu().eval()
    for parameter in clean_model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    projector_parameters = list(model.encoder.projector.parameters())
    for parameter in projector_parameters:
        parameter.requires_grad_(True)
    initial_parameters = [parameter.detach().clone() for parameter in projector_parameters]

    train = splits["train"]
    val = splits["val"]
    x_clean = np.asarray(train["clean_x"], dtype=np.float32)
    y_clean = np.asarray(train["clean_y"], dtype=np.int64)
    trusted_x = torch.as_tensor(val["target"], dtype=torch.float32)
    candidate_clean_x = torch.as_tensor(train["target"], dtype=torch.float32)
    poison_x = torch.as_tensor(splits["archived_poison"], dtype=torch.float32)
    if tuple(map(len, (trusted_x, candidate_clean_x, poison_x))) != (
        TRUSTED_COUNT,
        CLEAN_CANDIDATE_COUNT,
        POISON_CANDIDATE_COUNT,
    ):
        raise ValueError("P6-D detector population count mismatch")
    clean_tensor = torch.as_tensor(x_clean, dtype=torch.float32)
    with torch.no_grad():
        teacher_logits = clean_model(clean_tensor).detach()
        clean_observables = {
            "trusted": xz16_from_model(clean_model, trusted_x).detach(),
            "clean": xz16_from_model(clean_model, candidate_clean_x).detach(),
            "poison": xz16_from_model(clean_model, poison_x).detach(),
        }
    partition = build_crossfit_partition(seed)
    rng = np.random.default_rng(derived_seeds["minibatch_seed"])
    history: list[dict[str, Any]] = []
    states: list[dict[str, torch.Tensor]] = []
    observables: list[dict[str, np.ndarray]] = []
    epoch_zero_loss: torch.Tensor | None = None

    def anchor_loss() -> torch.Tensor:
        return torch.stack(
            [
                torch.mean((parameter - initial) ** 2)
                for parameter, initial in zip(projector_parameters, initial_parameters)
            ]
        ).mean()

    def channel() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        drift = {
            "trusted": xz16_from_model(model, trusted_x)
            - clean_observables["trusted"],
            "clean": xz16_from_model(model, candidate_clean_x)
            - clean_observables["clean"],
            "poison": xz16_from_model(model, poison_x)
            - clean_observables["poison"],
        }
        result = torch_final_channel(
            seed,
            drift["trusted"],
            drift["clean"],
            drift["poison"],
            partition=partition,
        )
        return drift, result

    def record(epoch: int, objectives: Mapping[str, Any] | None) -> None:
        nonlocal epoch_zero_loss
        model.eval()
        drift, torch_result = channel()
        if epoch_zero_loss is None:
            epoch_zero_loss = torch_result["soft_flag_loss"].detach()
        parity = assert_torch_numpy_parity(
            seed,
            drift["trusted"],
            drift["clean"],
            drift["poison"],
            partition=partition,
        )
        exact = parity["numpy"]
        validation_ca, validation_asr = validation_metric(model)
        if not np.isfinite([validation_ca, validation_asr]).all():
            raise FloatingPointError("Non-finite validation metric")
        history.append(
            {
                "epoch": int(epoch),
                "validation_balanced_ca": float(validation_ca),
                "validation_asr": float(validation_asr),
                "tpr": float(exact["tpr"]),
                "fpr": float(exact["fpr"]),
                "mean_poison_p_value": float(exact["mean_poison_p_value"]),
                "soft_flag_loss": float(torch_result["soft_flag_loss"].detach()),
                "max_abs_score_error": parity["max_abs_score_error"],
                "max_abs_threshold_error": parity["max_abs_threshold_error"],
                "objectives": dict(objectives or {}),
            }
        )
        states.append(copy.deepcopy(model.encoder.projector.state_dict()))
        observables.append(
            {
                name: values.detach().cpu().numpy().astype(np.float32)
                for name, values in drift.items()
            }
        )
        model.train()

    record(0, None)
    assert epoch_zero_loss is not None
    normalization = epoch_zero_loss + 1e-12
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x_clean))
        poison_order = rng.permutation(len(poison_x))
        poison_cursor = 0
        task_sums = {
            "clean_ce": 0.0,
            "backdoor_ce": 0.0,
            "distillation": 0.0,
            "anchor": 0.0,
            "task_total": 0.0,
            "task_gradient_norm_max": 0.0,
            "task_updates": 0,
        }
        model.train()
        for start in range(0, len(x_clean), batch_size):
            clean_indices = order[start : start + batch_size]
            size = len(clean_indices)
            if poison_cursor + size > len(poison_order):
                poison_order = rng.permutation(len(poison_x))
                poison_cursor = 0
            poison_indices = poison_order[poison_cursor : poison_cursor + size]
            poison_cursor += size
            if len(poison_indices) < size:
                poison_indices = np.concatenate(
                    [
                        poison_indices,
                        rng.choice(
                            len(poison_x), size - len(poison_indices), replace=True
                        ),
                    ]
                )
            clean_batch = clean_tensor[clean_indices]
            labels = torch.as_tensor(y_clean[clean_indices], dtype=torch.long)
            poison_batch = poison_x[poison_indices]
            clean_logits = model(clean_batch)
            poison_logits = model(poison_batch)
            clean_ce = F.cross_entropy(clean_logits, labels)
            backdoor_ce = F.cross_entropy(
                poison_logits, torch.ones(size, dtype=torch.long)
            )
            temperature = 2.0
            distillation = F.kl_div(
                F.log_softmax(clean_logits / temperature, dim=1),
                F.softmax(teacher_logits[clean_indices] / temperature, dim=1),
                reduction="batchmean",
            ) * temperature**2
            anchor = anchor_loss()
            task_total = (
                clean_ce
                + 12.0 * backdoor_ce
                + 20.0 * distillation
                + 0.001 * anchor
            )
            gradient_norm = _explicit_clipped_step(
                task_total, projector_parameters, 0.01, 1.0
            )
            for name, value in (
                ("clean_ce", clean_ce),
                ("backdoor_ce", backdoor_ce),
                ("distillation", distillation),
                ("anchor", anchor),
                ("task_total", task_total),
            ):
                task_sums[name] += float(value.detach())
            task_sums["task_gradient_norm_max"] = max(
                task_sums["task_gradient_norm_max"], gradient_norm
            )
            task_sums["task_updates"] += 1

        # Exactly one full-population differentiable final-observable step.
        _, differentiable = channel()
        observable_anchor = anchor_loss()
        normalized_soft_loss = differentiable["soft_flag_loss"] / normalization
        observable_total = (
            float(evasion_weight) * normalized_soft_loss
            + 0.001 * observable_anchor
        )
        observable_gradient_norm = _explicit_clipped_step(
            observable_total, projector_parameters, 0.01, 1.0
        )
        updates = int(task_sums["task_updates"])
        objective_summary = {
            key: value / updates
            for key, value in task_sums.items()
            if key not in {"task_updates", "task_gradient_norm_max"}
        }
        objective_summary.update(
            {
                "task_updates": updates,
                "task_gradient_norm_max": task_sums["task_gradient_norm_max"],
                "observable_soft_flag_loss": float(
                    differentiable["soft_flag_loss"].detach()
                ),
                "observable_normalized_soft_flag_loss": float(
                    normalized_soft_loss.detach()
                ),
                "observable_anchor": float(observable_anchor.detach()),
                "observable_total": float(observable_total.detach()),
                "observable_gradient_norm": observable_gradient_norm,
                "observable_updates": 1,
            }
        )
        record(epoch, objective_summary)

    selected = select_epoch(history)
    selected_index = next(
        index
        for index, row in enumerate(history)
        if int(row["epoch"]) == int(selected["epoch"])
    )
    model.encoder.projector.load_state_dict(states[selected_index])
    return {
        "seed": int(seed),
        "evasion_weight": float(evasion_weight),
        "derived_seeds": derived_seeds,
        "partition": partition,
        "history": history,
        "selected": selected,
        "selected_observables": observables[selected_index],
        "model": model,
        "projector_state": copy.deepcopy(states[selected_index]),
    }


def bootstrap_detector_outcomes(
    per_seed: Sequence[Mapping[str, Any]],
    *,
    draws: int = 10_000,
    rng_seed: int = 20260930,
) -> dict[str, Any]:
    eligible = [row for row in per_seed if bool(row["eligible"])]
    if len(eligible) < 5:
        return {
            "eligible_victim_count": len(eligible),
            "decision": "INSUFFICIENT_ELIGIBLE_VICTIMS",
            "efficacy_pooled": False,
        }
    if len(eligible) != 5:
        raise ValueError("The confirmation block contains exactly five victims")
    counts = np.asarray(
        [
            [int(row[name]) for name in ("tp", "fn", "fp", "tn")]
            for row in eligible
        ],
        dtype=np.int64,
    )
    total = counts.sum(axis=0)
    tp, fn, fp, tn = map(int, total)
    tpr, fpr = tp / (tp + fn), fp / (fp + tn)
    rng = np.random.default_rng(rng_seed)
    selections = rng.integers(0, len(eligible), size=(draws, len(eligible)))
    resampled = counts[selections].sum(axis=1)
    tpr_draws = resampled[:, 0] / (resampled[:, 0] + resampled[:, 1])
    fpr_draws = resampled[:, 2] / (resampled[:, 2] + resampled[:, 3])
    tpr_low, tpr_high = np.quantile(tpr_draws, [0.025, 0.975])
    fpr_low, fpr_high = np.quantile(fpr_draws, [0.025, 0.975])
    if tpr_high < 0.90:
        decision = "STRONG_EVASION"
    elif tpr < 0.90:
        decision = "POINT_EVASION"
    elif tpr_low >= 0.90 and fpr_high <= 0.05:
        decision = "INTERVAL_ROBUSTNESS_SUPPORT"
    else:
        decision = "POINT_ROBUSTNESS_ONLY"
    return {
        "eligible_victim_count": 5,
        "efficacy_pooled": True,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "pooled_tpr": tpr,
        "pooled_fpr": fpr,
        "tpr_ci_low": float(tpr_low),
        "tpr_ci_high": float(tpr_high),
        "fpr_ci_low": float(fpr_low),
        "fpr_ci_high": float(fpr_high),
        "decision": decision,
        "fpr_decision": "FPR_GATE_FAILURE" if fpr > 0.05 else "FPR_GATE_PASSES",
        "bootstrap_draws": int(draws),
        "bootstrap_rng_seed": int(rng_seed),
    }


def _formal_scope(stage: str) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if stage == "development":
        return DEVELOPMENT_SEEDS, EVASION_WEIGHTS
    if stage == "confirmation":
        return CONFIRMATION_SEEDS, ()
    raise ValueError(stage)


def guarded_formal_stage(args: argparse.Namespace) -> None:
    """Initialize a formal stage only after all safety/lock checks.

    Data/model training is exposed through :func:`train_attack_configuration`.
    The CLI intentionally stops after creating the hash-locked stage manifest;
    a data-bearing launcher must supply the audited BloodMNIST acquisition path
    and call the primitive.  This prevents an implementation-validation command
    from silently opening seeds 95--102.
    """

    if args.acknowledge_formal_seed_consumption != FORMAL_ACKNOWLEDGEMENT:
        raise PermissionError(
            "Formal P6-D execution requires --acknowledge-formal-seed-consumption "
            + FORMAL_ACKNOWLEDGEMENT
        )
    load_protocol()
    seeds, weights = _formal_scope(args.stage)
    if args.stage == "confirmation":
        validate_confirmation_lock(args.output / "confirmation_lock_receipt.json")
        lock = json.loads(
            (args.output / "confirmation_lock_receipt.json").read_text()
        )
        weights = (float(lock["selected_evasion_weight"]),)
    signature = run_signature(stage=args.stage, seeds=seeds, evasion_weights=weights)
    stage_root = args.output / args.stage
    initialize_or_resume(
        stage_root,
        signature,
        immutable_hashes={"protocol": sha256_file(PROTOCOL_PATH)},
    )
    print(
        json.dumps(
            {
                "status": "FORMAL_STAGE_INITIALIZED_NOT_RUN",
                "stage": args.stage,
                "seeds": seeds,
                "output": str(stage_root),
            },
            indent=2,
        )
    )


def _lossless_observable_rows(stage: str, seed: int, configuration_id: str,
                             splits: Mapping[str, Any],
                             observables: Mapping[str, np.ndarray],
                             partition: CrossfitPartition, epoch: int = 0) -> list[dict[str, Any]]:
    """Flatten selected float32 delta-XZ16 arrays with source-row provenance."""
    rows: list[dict[str, Any]] = []
    trusted = np.ascontiguousarray(observables["trusted"], dtype="<f4")
    clean = np.ascontiguousarray(observables["clean"], dtype="<f4")
    poison = np.ascontiguousarray(observables["poison"], dtype="<f4")
    if (trusted.shape != (TRUSTED_COUNT, OBSERVABLE_DIMENSION) or
            clean.shape != (CLEAN_CANDIDATE_COUNT, OBSERVABLE_DIMENSION) or
            poison.shape != (POISON_CANDIDATE_COUNT, OBSERVABLE_DIMENSION)):
        raise ValueError("Observable population shape mismatch")
    trusted_ids = np.asarray(splits["val"]["target_original_indices"], dtype=np.int64)
    populations = [("trusted", "reference", local, int(trusted_ids[i]), trusted[i], i)
                   for local, i in enumerate(partition.reference_indices)]
    populations.extend(("trusted", "calibration", local, int(trusted_ids[i]), trusted[i], i)
                       for local, i in enumerate(partition.calibration_indices))
    populations.extend(("clean_candidate", "clean_candidate", i,
                        int(splits["train"]["target_original_indices"][i]), clean[i], None)
                       for i in range(CLEAN_CANDIDATE_COUNT))
    populations.extend(("poison_candidate", "poison_candidate", i,
                        int(splits["poison_original_indices"][i]), poison[i], None)
                       for i in range(POISON_CANDIDATE_COUNT))
    for _group, population, local_index, original_index, values, trusted_index in populations:
            row = {"stage": stage, "seed": int(seed), "configuration_id": configuration_id,
                   "epoch": int(epoch),
                   "population": population, "sample_index": local_index,
                   "original_dataset_index": int(original_index)}
            if trusted_index is not None:
                row["trusted_sample_index"] = int(trusted_index)
            row.update({f"delta_xz16_{j:02d}": float(v) for j, v in enumerate(values)})
            rows.append(row)
    return rows


def _lossless_score_rows(stage: str, seed: int, configuration_id: str,
                         result: Mapping[str, Any], splits: Mapping[str, Any],
                         epoch: int = 0) -> list[dict[str, Any]]:
    exact = numpy_cfcmdu_score(seed, result["trusted"], result["clean"], result["poison"],
                               partition=result["partition"])
    orig = np.concatenate((np.asarray(splits["train"]["target_original_indices"], dtype=np.int64)[:CLEAN_CANDIDATE_COUNT],
                           np.asarray(splits["poison_original_indices"], dtype=np.int64)))
    rows = []
    for i in range(CLEAN_CANDIDATE_COUNT + POISON_CANDIDATE_COUNT):
        rows.append({"stage": stage, "seed": int(seed), "configuration_id": configuration_id,
                     "epoch": int(epoch),
                     "sample_index": i, "original_dataset_index": int(orig[i]),
                     "label": int(exact["labels"][i]), "fold_id": int(exact["fold_ids"][i]),
                     "scoring_half_id": int(exact["scoring_half_ids"][i]),
                     "direction_half_id": int(exact["direction_half_ids"][i]),
                     "score": float(exact["scores"][i]),
                     "calibration_fourth_largest_threshold": float(exact["thresholds"][i]),
                     "p_value": float(exact["p_values"][i]), "flagged": bool(exact["flags"][i])})
    return rows


def write_selected_artifacts(output: Path, stage: str, seed: int,
                             configuration_id: str, result: Mapping[str, Any],
                             splits: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Persist replay-complete selected checkpoint and float32 observables atomically."""
    seed_root = output / f"seed_{seed}" / configuration_id
    seed_root.mkdir(parents=True, exist_ok=True)
    checkpoint = seed_root / "attacked_projector.pt"
    atomic_torch_save(checkpoint, {"seed": int(seed), "configuration_id": configuration_id,
                                  "projector_state": result["projector_state"]})
    observable_rows = _lossless_observable_rows(stage, seed, configuration_id,
                                                splits, result["selected_observables"],
                                                result["partition"], int(result["selected"]["epoch"]))
    observable_path = seed_root / "observable_rows.csv"
    atomic_csv(observable_path, observable_rows)
    score_input = {**result["selected_observables"], "partition": result["partition"]}
    score_rows = _lossless_score_rows(stage, seed, configuration_id, score_input, splits,
                                      int(result["selected"]["epoch"]))
    score_path = seed_root / "per_sample_scores.csv"
    atomic_csv(score_path, score_rows)
    history_path = seed_root / "epoch_history.csv"
    atomic_csv(history_path, result["history"])
    fold_path = seed_root / "fold_receipt.json"
    atomic_json(fold_path, result["partition"].to_json())
    for artifact, count in ((checkpoint, None), (observable_path, len(observable_rows)),
                            (score_path, len(score_rows)), (history_path, len(result["history"])),
                            (fold_path, None)):
        manifest = update_manifest_artifact(output, manifest, artifact, row_count=count)
    return dict(manifest)


def _p6d_prepare_seed(classes: Mapping[int, np.ndarray], rgb_trigger: np.ndarray,
                     seed: int, baseline: Any) -> tuple[dict[str, Any], dict[str, Any], Callable[[], dict[str, np.ndarray]], np.ndarray]:
    """Use the audited PCA/split algorithm while deferring all test feature reads."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    np.random.seed(seed)
    torch.manual_seed(seed)
    raw_train: dict[int, np.ndarray] = {}
    raw_val: dict[int, np.ndarray] = {}
    test_indices: dict[int, np.ndarray] = {}
    for class_id in (6, 0):
        samples = classes[class_id]
        n_train, n_val = int(len(samples) * 0.6), int(len(samples) * 0.2)
        order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed)).numpy()
        raw_train[class_id] = samples[order[:n_train]]
        raw_val[class_id] = samples[order[n_train:n_train + n_val]]
        test_indices[class_id] = order[n_train + n_val:]
    fit_rows = np.concatenate([raw_train[6], raw_train[0]], axis=0)
    raw_scaler = StandardScaler().fit(fit_rows)
    pca = PCA(n_components=256).fit(raw_scaler.transform(fit_rows))
    pca_train = {k: pca.transform(raw_scaler.transform(v)).astype(np.float32)
                 for k, v in raw_train.items()}
    pca_val = {k: pca.transform(raw_scaler.transform(v)).astype(np.float32)
               for k, v in raw_val.items()}
    post_scaler = StandardScaler().fit(np.concatenate([pca_train[6], pca_train[0]], axis=0))
    trigger = (rgb_trigger / raw_scaler.scale_) @ pca.components_.T
    trigger = (0.8 / float(np.max(np.abs(trigger))) * trigger).astype(np.float32)

    def build_split(pca_values: Mapping[int, np.ndarray]) -> dict[str, np.ndarray]:
        target_pca, source_pca = pca_values[6], pca_values[0]
        target = post_scaler.transform(target_pca).astype(np.float32)
        source = post_scaler.transform(source_pca).astype(np.float32)
        triggered = post_scaler.transform(np.clip(source_pca + trigger, -3, 3)).astype(np.float32)
        return {"target": target, "non_target": source, "triggered_non_target": triggered,
                "clean_x": np.concatenate([target, source], axis=0),
                "clean_y": np.concatenate([np.ones(len(target), dtype=np.int64),
                                           np.zeros(len(source), dtype=np.int64)])}

    splits = {"train": build_split(pca_train), "val": build_split(pca_val)}
    order = torch.randperm(len(splits["train"]["non_target"]),
                           generator=torch.Generator().manual_seed(seed + 20009)).numpy()
    poison_indices = order[:POISON_CANDIDATE_COUNT]
    splits["archived_poison"] = splits["train"]["triggered_non_target"][poison_indices]
    rebuilt = post_scaler.transform(np.clip(pca_train[0][poison_indices] + trigger, -3, 3)).astype(np.float32)
    trigger_error = float(np.max(np.abs(rebuilt - splits["archived_poison"])))
    if trigger_error > 1e-5:
        raise RuntimeError(f"Seed {seed} trigger reconstruction mismatch: {trigger_error}")
    trigger_receipt = {"poison_indices": poison_indices,
                       "mapped_trigger_linf": float(np.max(np.abs(trigger))),
                       "mapped_trigger_l2": float(np.linalg.norm(trigger)),
                       "trigger_reconstruction_max_abs_error": trigger_error}
    def build_test_once() -> dict[str, np.ndarray]:
        pca_test = {k: pca.transform(raw_scaler.transform(classes[k][test_indices[k]])).astype(np.float32)
                    for k in (6, 0)}
        return build_split(pca_test)
    return splits, trigger_receipt, build_test_once, trigger


def run_development_launcher(args: argparse.Namespace) -> None:
    """Execute the frozen development grid; never touch confirmation seeds."""
    if args.acknowledge_formal_seed_consumption != FORMAL_ACKNOWLEDGEMENT:
        raise PermissionError("Formal execution requires --acknowledge-formal-seed-consumption "
                              + FORMAL_ACKNOWLEDGEMENT)
    if args.stage != "development":
        raise PermissionError("This data launcher currently supports development only")
    protocol = load_protocol()
    import evaluate_bloodmnist_checkpoints as heldout
    import run_bloodmnist_group_coherent_10seeds as baseline
    import run_bloodmnist_whitebox_adaptive_pilot as whitebox
    import train_bloodmnist_clean_encoder as clean_encoder
    import verify_p6d_final_observable_adaptive as independent_verifier
    import sklearn

    dataset_path = args.dataset_root / "bloodmnist.npz"
    if sha256_file(dataset_path) != protocol["dataset_and_split_lock"]["dataset_npz_sha256"]:
        raise RuntimeError("P6-D dataset SHA-256 mismatch; refusing stage")
    stage_root = args.output / "development"
    signature = run_signature(stage="development", seeds=DEVELOPMENT_SEEDS,
                              evasion_weights=EVASION_WEIGHTS)
    dependency_paths = [Path(__file__).resolve(), Path(heldout.__file__).resolve(),
                        Path(baseline.__file__).resolve(), Path(whitebox.__file__).resolve(),
                        Path(clean_encoder.__file__).resolve(), Path(independent_verifier.__file__).resolve(),
                        Path(whitebox.adaptive.__file__).resolve()]
    immutable = {str(path.relative_to(PAPER14_ROOT)): sha256_file(path) for path in dependency_paths}
    immutable.update({"protocol": sha256_file(PROTOCOL_PATH), "dataset": sha256_file(dataset_path)})
    manifest = initialize_or_resume(stage_root, signature, immutable_hashes=immutable)
    environment = {"python": sys.version, "platform": platform.platform(),
                   "torch": torch.__version__, "numpy": np.__version__,
                   "scikit_learn": sklearn.__version__,
                   "device": "cpu", "deterministic_algorithms": True}
    env_path = stage_root / "environment.json"
    if env_path.exists() and json.loads(env_path.read_text()) != environment:
        raise RuntimeError("Resume refused: environment receipt differs")
    atomic_json(env_path, environment)
    manifest = update_manifest_artifact(stage_root, manifest, env_path)

    classes = heldout.collect_classes(args.dataset_root)
    rgb_trigger, rgb_receipt = baseline.fixed_rgb_trigger(classes)
    with np.load(dataset_path, allow_pickle=False) as dataset_payload:
        official_indices = {class_id: np.flatnonzero(
            dataset_payload["train_labels"].reshape(-1) == class_id)[:len(classes[class_id])]
                            for class_id in (6, 0)}
    all_results: dict[tuple[int, float], dict[str, Any]] = {}
    all_splits: dict[int, dict[str, Any]] = {}
    deferred_tests: dict[int, Callable[[], dict[str, np.ndarray]]] = {}
    summary_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    observable_rows: list[dict[str, Any]] = []
    fold_receipts: list[dict[str, Any]] = []
    split_receipts: dict[str, Any] = {}
    checkpoint_receipts: dict[str, Any] = {}
    for seed in DEVELOPMENT_SEEDS:
        splits, trigger_receipt, build_test_once, initial_trigger = _p6d_prepare_seed(
            classes, rgb_trigger, seed, baseline)
        deferred_tests[seed] = build_test_once
        # The canonical preprocessing helper preserves class-local row order.
        # Store unambiguous source coordinates as (class id, class-local index).
        class_order = {}
        for class_id in (6, 0):
            n_train = int(len(classes[class_id]) * 0.6)
            n_val = int(len(classes[class_id]) * 0.2)
            order = torch.randperm(len(classes[class_id]), generator=torch.Generator().manual_seed(seed)).numpy()
            class_order[class_id] = {name: order[sl].tolist() for name, sl in (
                ("train", slice(0, n_train)), ("val", slice(n_train, n_train + n_val)))}
        splits["train"]["target_original_indices"] = official_indices[6][np.asarray(class_order[6]["train"], dtype=np.int64)]
        splits["val"]["target_original_indices"] = official_indices[6][np.asarray(class_order[6]["val"], dtype=np.int64)]
        splits["train"]["non_target_original_indices"] = official_indices[0][np.asarray(class_order[0]["train"], dtype=np.int64)]
        poison_ids = np.asarray(trigger_receipt["poison_indices"], dtype=np.int64)
        splits["poison_original_indices"] = splits["train"]["non_target_original_indices"][poison_ids]
        if (len(splits["val"]["target"]) != TRUSTED_COUNT or
                len(splits["train"]["target"]) < CLEAN_CANDIDATE_COUNT or
                len(splits["archived_poison"]) != POISON_CANDIDATE_COUNT):
            raise RuntimeError(f"Seed {seed} population count mismatch")
        split_receipts[str(seed)] = {
            "seed": seed,
            "trigger_sha256_float32": sha256_array(initial_trigger),
            "poison_indices_sha256": stable_json_sha256(poison_ids.astype(int).tolist()),
            "poison_original_indices_sha256": stable_json_sha256(splits["poison_original_indices"].astype(int).tolist()),
            "trusted_validation_target_sha256_float32": sha256_array(splits["val"]["target"]),
            "clean_train_target_sha256_float32": sha256_array(splits["train"]["target"][:CLEAN_CANDIDATE_COUNT]),
            "archived_poison_sha256_float32": sha256_array(splits["archived_poison"]),
            "trigger_receipt": trigger_receipt,
        }
        clean_root = stage_root / "clean_models"
        clean_checkpoint_path = clean_root / "models" / f"seed_{seed}_clean.pt"
        clean_record_path = clean_root / "models" / f"seed_{seed}_clean.json"
        clean_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if clean_checkpoint_path.is_file() and clean_record_path.is_file():
            clean_model = clean_encoder.CleanQNN("local", 0)
            clean_model.load_state_dict(torch.load(clean_checkpoint_path, map_location="cpu",
                                                   weights_only=False)["state_dict"])
            clean_record = json.loads(clean_record_path.read_text(encoding="utf-8"))
        else:
            clean_model, full_record = clean_encoder.train_one(
                {"train": (splits["train"]["clean_x"], splits["train"]["clean_y"]),
                 "val": (splits["val"]["clean_x"], splits["val"]["clean_y"])},
                seed, "local", 0, 40, 3e-3, 1e-4, 64, 10)
            clean_record = {"seed": seed, "best_epoch": full_record["best_epoch"],
                            "train": full_record["train"], "validation": full_record["val"],
                            "epochs_completed": full_record["epochs_completed"]}
            atomic_torch_save(clean_checkpoint_path, {"state_dict": clean_model.state_dict(), "seed": seed})
            atomic_json(clean_record_path, clean_record)
        clean_checkpoint = clean_root / "models" / f"seed_{seed}_clean.pt"
        clean_record_path = clean_root / "models" / f"seed_{seed}_clean.json"
        if not clean_checkpoint.is_file():
            raise RuntimeError(f"Seed {seed} clean checkpoint missing after training/load")
        manifest = update_manifest_artifact(stage_root, manifest, clean_checkpoint)
        manifest = update_manifest_artifact(stage_root, manifest, clean_record_path)
        all_splits[seed] = splits

        def validation_metric(model: torch.nn.Module) -> tuple[float, float]:
            return whitebox.balanced_ca_and_asr(model, splits["val"]["target"],
                                                splits["val"]["non_target"],
                                                splits["val"]["triggered_non_target"])

        for weight in EVASION_WEIGHTS:
            configuration_id = f"w{weight:g}"
            attack_splits = {"train": splits["train"], "val": splits["val"],
                             "archived_poison": splits["archived_poison"]}
            result = train_attack_configuration(seed=seed, evasion_weight=weight,
                                                clean_model=clean_model, splits=attack_splits,
                                                validation_metric=validation_metric)
            all_results[(seed, weight)] = result
            selected = {**result["selected"], "seed": seed, "evasion_weight": weight,
                        "configuration_id": configuration_id, "stage": "development",
                        "selected_epoch": int(result["selected"]["epoch"])}
            exact = numpy_cfcmdu_score(seed, result["selected_observables"]["trusted"],
                                       result["selected_observables"]["clean"],
                                       result["selected_observables"]["poison"],
                                       partition=result["partition"])
            selected.update({name: int(exact[name]) for name in ("tp", "fn", "fp", "tn")})
            summary_rows.append(selected)
            history_rows.extend({**row, "seed": seed, "stage": "development",
                                 "evasion_weight": weight, "configuration_id": configuration_id,
                                 "selected_epoch": int(row["epoch"]) == int(result["selected"]["epoch"])}
                                for row in result["history"])
            score_result = {**result["selected_observables"], "partition": result["partition"]}
            score_rows.extend(_lossless_score_rows("development", seed, configuration_id,
                                                   score_result, splits, int(result["selected"]["epoch"])))
            observable_rows.extend(_lossless_observable_rows("development", seed,
                                      configuration_id, splits, result["selected_observables"],
                                      result["partition"], int(result["selected"]["epoch"])))
            fold_receipts.append({**result["partition"].to_json(), "stage": "development",
                                  "configuration_id": configuration_id})
    summaries = []
    for weight in EVASION_WEIGHTS:
        configuration_id = f"w{weight:g}"
        selected_rows = [{**all_results[(seed, weight)]["selected"], "seed": seed}
                         for seed in DEVELOPMENT_SEEDS]
        summaries.append(summarize_development_configuration(configuration_id, weight, selected_rows))
    receipt = create_development_selection_receipt(stage_root / "development_selection_receipt.json", summaries)
    manifest = update_manifest_artifact(stage_root, manifest, stage_root / "development_selection_receipt.json")
    manifest = update_manifest_artifact(stage_root, manifest,
                                       stage_root / "development_selection_receipt.json.sha256")
    atomic_csv(stage_root / "epoch_history.csv", history_rows)
    atomic_csv(stage_root / "per_sample_scores.csv", score_rows)
    atomic_csv(stage_root / "observable_rows.csv", observable_rows)
    atomic_json(stage_root / "fold_receipts.json", fold_receipts)
    atomic_json(stage_root / "split_receipts.json", split_receipts)
    atomic_csv(stage_root / "per_seed_metrics.csv", summary_rows)
    for artifact, count in ((stage_root / "epoch_history.csv", len(history_rows)),
                            (stage_root / "per_sample_scores.csv", len(score_rows)),
                            (stage_root / "observable_rows.csv", len(observable_rows)),
                            (stage_root / "fold_receipts.json", None),
                            (stage_root / "split_receipts.json", None),
                            (stage_root / "per_seed_metrics.csv", len(summary_rows))):
        manifest = update_manifest_artifact(stage_root, manifest, artifact, row_count=count)
    selected_weight = float(receipt["selected_evasion_weight"])
    authorize_development_test_read(stage_root / "development_selection_receipt.json",
                                   f"w{selected_weight:g}")
    victims = []
    for seed in DEVELOPMENT_SEEDS:
        result = all_results[(seed, selected_weight)]
        configuration_id = f"w{selected_weight:g}"
        # Save the selected projector before the one permitted test read.
        checkpoint = stage_root / f"seed_{seed}" / configuration_id / "attacked_projector.pt"
        atomic_torch_save(checkpoint, {"seed": seed, "configuration_id": configuration_id,
                                       "projector_state": result["projector_state"]})
        manifest = update_manifest_artifact(stage_root, manifest, checkpoint)
        checkpoint_receipts[str(seed)] = {
            "clean_checkpoint": {"path": str((stage_root / "clean_models" / "models" / f"seed_{seed}_clean.pt").relative_to(stage_root)),
                                 "sha256": sha256_file(stage_root / "clean_models" / "models" / f"seed_{seed}_clean.pt")},
            "attacked_projector": {"path": str(checkpoint.relative_to(stage_root)),
                                   "sha256": sha256_file(checkpoint)},
        }
        test_split = deferred_tests[seed]()
        ca, asr = whitebox.balanced_ca_and_asr(result["model"], test_split["target"],
                                              test_split["non_target"],
                                              test_split["triggered_non_target"])
        victims.append({"stage": "development", "seed": seed,
                        "configuration_id": configuration_id, "selected_epoch": result["selected"]["epoch"],
                        "test_balanced_ca": ca, "test_asr": asr,
                        "eligible": bool(ca >= TEST_CA_MIN and asr >= TEST_ASR_MIN)})
    atomic_csv(stage_root / "victim_metrics.csv", victims)
    manifest = update_manifest_artifact(stage_root, manifest, stage_root / "victim_metrics.csv", row_count=len(victims))
    atomic_json(stage_root / "checkpoint_receipts.json", checkpoint_receipts)
    manifest = update_manifest_artifact(stage_root, manifest, stage_root / "checkpoint_receipts.json")
    report = ["# P6-D development execution record", "",
              f"Selected configuration: {receipt['selected_configuration_id']} (validation-only ranking).",
              f"Eligible development victims: {sum(row['eligible'] for row in victims)} / {len(victims)}.",
              "This report records the frozen development stage and does not authorize a scientific claim.", ""]
    report_path = stage_root / "EXPERIMENT_REPORT.md"
    _atomic_bytes(report_path, "\n".join(report).encode("utf-8"))
    manifest = update_manifest_artifact(stage_root, manifest, report_path)
    manifest["status"] = "complete"
    atomic_json(stage_root / "manifest.json", manifest)
    verification = independent_verifier.verify_bundle(stage_root)
    print(json.dumps({"status": "development_complete", "selected_evasion_weight": selected_weight,
                      "eligible_victims": sum(row["eligible"] for row in victims),
                      "expansion_gate": "PASS" if sum(row["eligible"] for row in victims) >= 2 else "FAIL",
                      "independent_replay": verification,
                      "manifest": str(stage_root / "manifest.json")}, indent=2))


def self_check(seed: int) -> dict[str, Any]:
    require_synthetic_seed(seed)
    rng = np.random.default_rng(seed)
    trusted = torch.tensor(
        rng.normal(size=(TRUSTED_COUNT, OBSERVABLE_DIMENSION)),
        dtype=torch.float64,
        requires_grad=True,
    )
    clean = torch.tensor(
        rng.normal(size=(CLEAN_CANDIDATE_COUNT, OBSERVABLE_DIMENSION)),
        dtype=torch.float64,
        requires_grad=True,
    )
    poison = torch.tensor(
        rng.normal(size=(POISON_CANDIDATE_COUNT, OBSERVABLE_DIMENSION)),
        dtype=torch.float64,
        requires_grad=True,
    )
    parity = assert_torch_numpy_parity(seed, trusted, clean, poison)
    parity["torch"]["soft_flag_loss"].backward()
    return {
        "seed": seed,
        "max_abs_score_error": parity["max_abs_score_error"],
        "max_abs_threshold_error": parity["max_abs_threshold_error"],
        "trusted_gradient_nonzero": bool(torch.any(trusted.grad != 0)),
        "clean_gradient_nonzero": bool(torch.any(clean.grad != 0)),
        "poison_gradient_nonzero": bool(torch.any(poison.grad != 0)),
    }


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    subparsers = output.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("self-check")
    check.add_argument("--seed", type=int, default=10_000)
    formal = subparsers.add_parser("formal-stage")
    formal.add_argument("--stage", choices=("development", "confirmation"), required=True)
    formal.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    formal.add_argument("--dataset-root", type=Path)
    formal.add_argument("--acknowledge-formal-seed-consumption")
    formal.add_argument("--issue-confirmation-lock", action="store_true",
                        help="Independently verify development artifacts and issue the one-shot confirmation lock")
    return output


def main() -> None:
    args = parser().parse_args()
    if args.command == "self-check":
        print(json.dumps(self_check(args.seed), indent=2, sort_keys=True))
    elif getattr(args, "issue_confirmation_lock", False):
        if args.acknowledge_formal_seed_consumption != FORMAL_ACKNOWLEDGEMENT:
            raise PermissionError("Confirmation lock issuance requires --acknowledge-formal-seed-consumption "
                                  + FORMAL_ACKNOWLEDGEMENT)
        print(json.dumps(issue_confirmation_lock_from_development(args.output), indent=2, sort_keys=True))
    elif getattr(args, "dataset_root", None) is not None:
        if args.stage == "confirmation":
            run_confirmation_launcher(args)
        else:
            run_development_launcher(args)
    else:
        guarded_formal_stage(args)


if __name__ == "__main__":
    main()
