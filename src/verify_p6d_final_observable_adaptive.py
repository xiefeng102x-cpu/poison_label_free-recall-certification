#!/usr/bin/env python3
"""Independent replay verifier for P6-D artifacts.

This module intentionally does not import the P6-D runner and contains no
training-time soft-rank implementation.  It reconstructs the frozen float64
endpoint, validation-only selections, victim eligibility, and confirmation
bootstrap directly from saved tables and receipts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PAPER14_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    PAPER14_ROOT
    / "results/top_journal_program_v1/p6d_final_observable_adaptive_protocol_v1.json"
)
DEFAULT_OUTPUT = PAPER14_ROOT / "results/p6d_final_observable_adaptive_v1"
PROTOCOL_ID = "p6d_final_observable_adaptive_protocol_v1"
DEVELOPMENT_SEEDS = (95, 96, 97)
CONFIRMATION_SEEDS = (98, 99, 100, 101, 102)
EVASION_WEIGHTS = (0.5, 2.0, 8.0)
REFERENCE_COUNT = 120
CALIBRATION_COUNT = 80
TRUSTED_COUNT = 200
CLEAN_COUNT = 600
POISON_COUNT = 60
DIMENSION = 16
ALPHA = 0.05
SCORE_ATOL = 1e-6
VALIDATION_CA_MIN = 0.92
VALIDATION_ASR_MIN = 0.85
TEST_CA_MIN = 0.90
TEST_ASR_MIN = 0.80


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _partition(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(20260928 + 1009 * int(seed))
    trusted_order = rng.permutation(TRUSTED_COUNT)
    groups: list[dict[str, Any]] = []
    for fold_id, fold in enumerate(
        np.array_split(rng.permutation(CLEAN_COUNT + POISON_COUNT), 4)
    ):
        halves = np.array_split(rng.permutation(fold), 2)
        groups.extend(
            [
                {
                    "fold_id": fold_id,
                    "scoring_half_id": 0,
                    "direction_half_id": 1,
                    "score_indices": halves[0],
                    "direction_indices": halves[1],
                },
                {
                    "fold_id": fold_id,
                    "scoring_half_id": 1,
                    "direction_half_id": 0,
                    "score_indices": halves[1],
                    "direction_indices": halves[0],
                },
            ]
        )
    return {
        "seed": int(seed),
        "trusted_order": trusted_order,
        "reference_indices": trusted_order[:REFERENCE_COUNT],
        "calibration_indices": trusted_order[REFERENCE_COUNT:],
        "groups": groups,
    }


def _array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.int64)


def validate_partition_receipt(seed: int, receipt: Mapping[str, Any]) -> dict[str, Any]:
    expected = _partition(seed)
    if int(receipt["seed"]) != int(seed):
        raise AssertionError("Fold receipt seed mismatch")
    for name in ("trusted_order", "reference_indices", "calibration_indices"):
        if not np.array_equal(_array(receipt[name]), expected[name]):
            raise AssertionError(f"Fold receipt differs from frozen RNG: {name}")
    received_groups = receipt["groups"]
    if len(received_groups) != len(expected["groups"]):
        raise AssertionError("Fold receipt group count mismatch")
    for received, frozen in zip(received_groups, expected["groups"]):
        for name in ("fold_id", "scoring_half_id", "direction_half_id"):
            if int(received[name]) != int(frozen[name]):
                raise AssertionError(
                    f"Fold receipt differs from frozen RNG group: {name}"
                )
        for name in ("score_indices", "direction_indices"):
            if not np.array_equal(_array(received[name]), frozen[name]):
                raise AssertionError(
                    f"Fold receipt differs from frozen RNG group: {name}"
                )
    return expected


def _matrix(name: str, values: np.ndarray, rows: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (rows, DIMENSION):
        raise AssertionError(f"{name} shape mismatch: {result.shape}")
    if not np.isfinite(result).all():
        raise AssertionError(f"{name} contains a non-finite value")
    return result


def replay_numpy_cfcmdu(
    seed: int,
    trusted: np.ndarray,
    clean: np.ndarray,
    poison: np.ndarray,
    *,
    partition_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Independent exact float64 reconstruction of canonical custom_crossfit."""

    trusted = _matrix("trusted", trusted, TRUSTED_COUNT)
    clean = _matrix("clean", clean, CLEAN_COUNT)
    poison = _matrix("poison", poison, POISON_COUNT)
    partition = _partition(seed)
    if partition_receipt is not None:
        validate_partition_receipt(seed, partition_receipt)
    features = np.concatenate([clean, poison], axis=0).astype(np.float64, copy=False)
    labels = np.concatenate(
        [np.zeros(CLEAN_COUNT, dtype=int), np.ones(POISON_COUNT, dtype=int)]
    )
    reference = trusted[partition["reference_indices"]]
    calibration = trusted[partition["calibration_indices"]]
    mean = reference.mean(axis=0, keepdims=True)
    reference_centered = reference - mean
    calibration_centered = calibration - mean
    scores = np.empty(CLEAN_COUNT + POISON_COUNT, dtype=np.float64)
    p_values = np.empty_like(scores)
    thresholds = np.empty_like(scores)
    flags = np.empty(CLEAN_COUNT + POISON_COUNT, dtype=bool)
    fold_ids = np.empty(CLEAN_COUNT + POISON_COUNT, dtype=np.int64)
    scoring_half_ids = np.empty_like(fold_ids)
    direction_half_ids = np.empty_like(fold_ids)
    for group in partition["groups"]:
        score_indices = group["score_indices"]
        direction = (
            (features[group["direction_indices"]] - mean).mean(axis=0)
            - reference_centered.mean(axis=0)
        )
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            direction = np.ones(DIMENSION, dtype=np.float64)
            norm = float(np.linalg.norm(direction))
        direction /= norm
        calibration_scores = calibration_centered @ direction
        query_scores = (features[score_indices] - mean) @ direction
        query_p = (
            1
            + np.sum(
                calibration_scores[:, None] >= query_scores[None, :], axis=0
            )
        ) / float(CALIBRATION_COUNT + 1)
        tau = float(np.sort(calibration_scores)[-4])
        scores[score_indices] = query_scores
        p_values[score_indices] = query_p
        thresholds[score_indices] = tau
        flags[score_indices] = query_p <= ALPHA
        fold_ids[score_indices] = int(group["fold_id"])
        scoring_half_ids[score_indices] = int(group["scoring_half_id"])
        direction_half_ids[score_indices] = int(group["direction_half_id"])
    poison_mask = labels == 1
    clean_mask = ~poison_mask
    return {
        "scores": scores,
        "p_values": p_values,
        "thresholds": thresholds,
        "flags": flags,
        "labels": labels,
        "fold_ids": fold_ids,
        "scoring_half_ids": scoring_half_ids,
        "direction_half_ids": direction_half_ids,
        "tp": int(flags[poison_mask].sum()),
        "fn": int((~flags[poison_mask]).sum()),
        "fp": int(flags[clean_mask].sum()),
        "tn": int((~flags[clean_mask]).sum()),
        "tpr": float(flags[poison_mask].mean()),
        "fpr": float(flags[clean_mask].mean()),
        "mean_poison_p_value": float(p_values[poison_mask].mean()),
    }


def verify_score_rows(
    replay: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(rows) != CLEAN_COUNT + POISON_COUNT:
        raise AssertionError("Per-sample score row count mismatch")
    ordered = sorted(rows, key=lambda row: int(row["sample_index"]))
    indices = np.asarray([int(row["sample_index"]) for row in ordered])
    if not np.array_equal(indices, np.arange(CLEAN_COUNT + POISON_COUNT)):
        raise AssertionError("Per-sample indices are not a complete permutation")
    score_error = float(
        np.max(
            np.abs(
                np.asarray([float(row["score"]) for row in ordered])
                - replay["scores"]
            )
        )
    )
    threshold_error = float(
        np.max(
            np.abs(
                np.asarray(
                    [
                        float(row["calibration_fourth_largest_threshold"])
                        for row in ordered
                    ]
                )
                - replay["thresholds"]
            )
        )
    )
    if score_error > SCORE_ATOL or threshold_error > SCORE_ATOL:
        raise AssertionError(
            f"Exact-score replay failed: score={score_error}, tau={threshold_error}"
        )
    p_values = np.asarray([float(row["p_value"]) for row in ordered])
    if not np.array_equal(p_values, replay["p_values"]):
        raise AssertionError("Exact p-value replay failed")
    flags = np.asarray([_boolean(row["flagged"]) for row in ordered])
    if not np.array_equal(flags, replay["flags"]):
        raise AssertionError("Exact flag replay failed")
    for column, expected in (
        ("label", replay["labels"]),
        ("fold_id", replay["fold_ids"]),
        ("scoring_half_id", replay["scoring_half_ids"]),
        ("direction_half_id", replay["direction_half_ids"]),
    ):
        received = np.asarray([int(row[column]) for row in ordered])
        if not np.array_equal(received, expected):
            raise AssertionError(f"Per-sample partition replay failed: {column}")
    return {
        "max_abs_score_error": score_error,
        "max_abs_threshold_error": threshold_error,
        "p_values_exact": True,
        "flags_exact": True,
    }


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ValueError(f"Not a Boolean value: {value!r}")


def _margin(row: Mapping[str, Any]) -> float:
    return min(
        float(row["validation_balanced_ca"]) / VALIDATION_CA_MIN,
        float(row["validation_asr"]) / VALIDATION_ASR_MIN,
    )


def replay_epoch_selection(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise AssertionError("Missing epoch history")
    if any(any(str(key).lower().startswith("test") for key in row) for row in rows):
        raise AssertionError("Test information leaked into epoch selection")
    eligible = [
        row
        for row in rows
        if float(row["validation_balanced_ca"]) >= VALIDATION_CA_MIN
        and float(row["validation_asr"]) >= VALIDATION_ASR_MIN
    ]
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                float(row["tpr"]),
                -float(row["mean_poison_p_value"]),
                -_margin(row),
                int(row["epoch"]),
            ),
        )
        validation_eligible = True
    else:
        selected = min(
            rows,
            key=lambda row: (-_margin(row), float(row["tpr"]), int(row["epoch"])),
        )
        validation_eligible = False
    return {
        "epoch": int(selected["epoch"]),
        "validation_eligible": validation_eligible,
    }


def _configuration_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    mean_tpr = row["mean_tpr_eligible"]
    return (
        -int(row["validation_eligible_count"]),
        float(mean_tpr) if mean_tpr is not None else float("inf"),
        -float(row["worst_validation_asr"]),
        -float(row["worst_validation_balanced_ca"]),
        float(row["evasion_weight"]),
    )


def replay_development_selection(
    summaries: Sequence[Mapping[str, Any]], receipt: Mapping[str, Any]
) -> str:
    if len(summaries) != 3:
        raise AssertionError("Expected three development configurations")
    if sorted(float(row["evasion_weight"]) for row in summaries) != list(
        EVASION_WEIGHTS
    ):
        raise AssertionError("Development grid differs from frozen weights")
    if any(any(str(key).lower().startswith("test") for key in row) for row in summaries):
        raise AssertionError("Test information leaked into configuration selection")
    ranked = sorted(summaries, key=_configuration_key)
    selected = str(ranked[0]["configuration_id"])
    if receipt.get("selection_uses_test") is not False:
        raise AssertionError("Selection receipt does not assert the test firewall")
    if str(receipt["selected_configuration_id"]) != selected:
        raise AssertionError("Development configuration selection replay failed")
    ranked_receipt = receipt.get("ranked_configurations", [])
    if [str(row["configuration_id"]) for row in ranked_receipt] != [
        str(row["configuration_id"]) for row in ranked
    ]:
        raise AssertionError("Development ranking order differs from receipt")
    for recomputed, recorded in zip(ranked, ranked_receipt):
        for field in (
            "evasion_weight",
            "validation_eligible_count",
            "mean_tpr_eligible",
            "worst_validation_asr",
            "worst_validation_balanced_ca",
        ):
            left, right = recomputed[field], recorded[field]
            if left is None or right is None:
                if left is not None or right is not None:
                    raise AssertionError(
                        f"Development summary mismatch: {field}"
                    )
            elif not np.isclose(float(left), float(right), atol=1e-12, rtol=0.0):
                raise AssertionError(f"Development summary mismatch: {field}")
    return selected


def replay_bootstrap(
    rows: Sequence[Mapping[str, Any]], draws: int = 10_000, seed: int = 20260930
) -> dict[str, Any]:
    eligible = [row for row in rows if _boolean(row["eligible"])]
    if len(eligible) < 5:
        return {
            "eligible_victim_count": len(eligible),
            "decision": "INSUFFICIENT_ELIGIBLE_VICTIMS",
            "efficacy_pooled": False,
        }
    if len(eligible) != 5:
        raise AssertionError("Confirmation has more than five eligible victims")
    counts = np.asarray(
        [[int(row[name]) for name in ("tp", "fn", "fp", "tn")] for row in eligible]
    )
    total = counts.sum(axis=0)
    tp, fn, fp, tn = map(int, total)
    tpr, fpr = tp / (tp + fn), fp / (fp + tn)
    rng = np.random.default_rng(seed)
    selections = rng.integers(0, 5, size=(draws, 5))
    sampled = counts[selections].sum(axis=1)
    tpr_draws = sampled[:, 0] / (sampled[:, 0] + sampled[:, 1])
    fpr_draws = sampled[:, 2] / (sampled[:, 2] + sampled[:, 3])
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
    }


def _verify_manifest(output: Path) -> dict[str, Any]:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise AssertionError("Manifest protocol identifier mismatch")
    signature = manifest.get("run_signature", {})
    if signature and signature.get("protocol_sha256") != sha256_file(PROTOCOL_PATH):
        raise AssertionError("Manifest protocol hash mismatch")
    for relative, record in manifest.get("artifacts", {}).items():
        artifact = output / relative
        if not artifact.is_file():
            raise AssertionError(f"Manifest artifact is missing: {relative}")
        if artifact.stat().st_size != int(record["bytes"]):
            raise AssertionError(f"Manifest byte count mismatch: {relative}")
        if sha256_file(artifact) != record["sha256"]:
            raise AssertionError(f"Manifest artifact hash mismatch: {relative}")
        if "row_count" in record and artifact.suffix == ".csv":
            if len(read_csv(artifact)) != int(record["row_count"]):
                raise AssertionError(f"Manifest row count mismatch: {relative}")
    return manifest


def _coordinates(rows: Sequence[Mapping[str, str]]) -> list[str]:
    if not rows:
        raise AssertionError("Observable table is empty")
    candidates = [
        key
        for key in rows[0]
        if key.startswith("delta_xz16_") or key.startswith("observable_")
    ]
    if len(candidates) != DIMENSION:
        raise AssertionError("Observable table must contain 16 coordinate columns")
    return sorted(candidates)


def _observable_matrix(
    rows: Sequence[Mapping[str, str]], population: str, coordinates: Sequence[str]
) -> np.ndarray:
    selected = [row for row in rows if row["population"] == population]
    selected.sort(key=lambda row: int(row["sample_index"]))
    return np.asarray(
        [[float(row[column]) for column in coordinates] for row in selected],
        dtype=np.float64,
    )


def _group(
    rows: Iterable[Mapping[str, str]], fields: Sequence[str]
) -> dict[tuple[str, ...], list[Mapping[str, str]]]:
    output: dict[tuple[str, ...], list[Mapping[str, str]]] = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        output.setdefault(key, []).append(row)
    return output


def verify_bundle(output: Path) -> dict[str, Any]:
    """Verify a completed artifact bundle without invoking training code."""

    output = output.resolve()
    manifest = _verify_manifest(output)
    required = (
        "environment.json",
        "victim_metrics.csv",
        "per_seed_metrics.csv",
        "epoch_history.csv",
        "per_sample_scores.csv",
        "observable_rows.csv",
        "fold_receipts.json",
        "checkpoint_receipts.json",
    )
    for relative in required:
        if not (output / relative).is_file():
            raise AssertionError(f"Required artifact is missing: {relative}")
    fold_payload = json.loads((output / "fold_receipts.json").read_text())
    fold_rows = fold_payload.get("receipts", fold_payload)
    if not isinstance(fold_rows, list):
        raise AssertionError("Fold receipts must be a list")
    fold_by_key = {
        (
            str(row["stage"]),
            str(int(row["seed"])),
            str(row["configuration_id"]),
        ): row
        for row in fold_rows
    }
    observable_rows = read_csv(output / "observable_rows.csv")
    score_rows = read_csv(output / "per_sample_scores.csv")
    coordinates = _coordinates(observable_rows)
    key_fields = ("stage", "seed", "configuration_id", "epoch")
    observable_groups = _group(observable_rows, key_fields)
    score_groups = _group(score_rows, key_fields)
    replay_reports = []
    exact_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for key, rows in observable_groups.items():
        stage_seed_config = key[:3]
        if key not in score_groups:
            raise AssertionError(f"Observable group has no scores: {key}")
        seed = int(key[1])
        fold_receipt = fold_by_key.get(stage_seed_config)
        if fold_receipt is None:
            raise AssertionError(f"Missing fold receipt: {stage_seed_config}")
        trusted = _observable_matrix(rows, "trusted", coordinates)
        # A lossless table may store already partitioned trusted rows.  Rebuild
        # the original 200-row order from their trusted_sample_index field.
        if trusted.shape != (TRUSTED_COUNT, DIMENSION):
            reference = _observable_matrix(rows, "reference", coordinates)
            calibration = _observable_matrix(rows, "calibration", coordinates)
            if reference.shape != (REFERENCE_COUNT, DIMENSION) or calibration.shape != (
                CALIBRATION_COUNT,
                DIMENSION,
            ):
                raise AssertionError("Observable trusted population mismatch")
            trusted = np.empty((TRUSTED_COUNT, DIMENSION), dtype=np.float64)
            receipt = validate_partition_receipt(seed, fold_receipt)
            trusted[receipt["reference_indices"]] = reference
            trusted[receipt["calibration_indices"]] = calibration
        clean = _observable_matrix(rows, "clean_candidate", coordinates)
        poison = _observable_matrix(rows, "poison_candidate", coordinates)
        replay = replay_numpy_cfcmdu(
            seed,
            trusted,
            clean,
            poison,
            partition_receipt=fold_receipt,
        )
        exact_by_key[key] = replay
        report = verify_score_rows(replay, score_groups[key])
        report.update({"stage": key[0], "seed": seed, "configuration_id": key[2], "epoch": int(key[3])})
        replay_reports.append(report)

    history = read_csv(output / "epoch_history.csv")
    history_groups = _group(history, ("stage", "seed", "configuration_id"))
    selection_replays: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, rows in history_groups.items():
        selection_replays[key] = replay_epoch_selection(rows)
        selected_values = {_boolean(row["selected_epoch"]) for row in rows}
        if True not in selected_values:
            raise AssertionError(f"No selected epoch marker for {key}")
        marked = [int(row["epoch"]) for row in rows if _boolean(row["selected_epoch"])]
        if marked != [selection_replays[key]["epoch"]]:
            raise AssertionError(f"Epoch selection receipt mismatch for {key}")

    victim_rows = read_csv(output / "victim_metrics.csv")
    for row in victim_rows:
        expected = (
            float(row["test_balanced_ca"]) >= TEST_CA_MIN
            and float(row["test_asr"]) >= TEST_ASR_MIN
        )
        if _boolean(row["eligible"]) != expected:
            raise AssertionError(f"Victim eligibility replay failed for seed {row['seed']}")

    per_seed_rows = read_csv(output / "per_seed_metrics.csv")
    for row in per_seed_rows:
        key = (
            str(row["stage"]),
            str(int(row["seed"])),
            str(row["configuration_id"]),
            str(int(row["selected_epoch"])),
        )
        if key not in exact_by_key:
            raise AssertionError(f"Per-seed metric has no observable replay: {key}")
        exact = exact_by_key[key]
        for field in ("tp", "fn", "fp", "tn"):
            if int(row[field]) != int(exact[field]):
                raise AssertionError(f"Confusion replay mismatch {key}: {field}")
        for field in ("tpr", "fpr"):
            if not np.isclose(
                float(row[field]), float(exact[field]), atol=1e-12, rtol=0.0
            ):
                raise AssertionError(f"Rate replay mismatch {key}: {field}")

    report: dict[str, Any] = {
        "status": "valid",
        "observable_score_groups": len(replay_reports),
        "epoch_selections": len(selection_replays),
        "victim_rows": len(victim_rows),
        "max_abs_score_error": max(
            (row["max_abs_score_error"] for row in replay_reports), default=0.0
        ),
    }
    stages = {row["stage"] for row in history}
    if "development" in stages:
        receipt_path = output / "development_selection_receipt.json"
        sidecar = receipt_path.with_suffix(receipt_path.suffix + ".sha256")
        if not receipt_path.is_file() or not sidecar.is_file():
            raise AssertionError("Development selection receipt/hash is missing")
        if sidecar.read_text().strip() != sha256_file(receipt_path):
            raise AssertionError("Development selection receipt hash mismatch")
        receipt = json.loads(receipt_path.read_text())
        development_groups = {
            key: rows
            for key, rows in history_groups.items()
            if key[0] == "development"
        }
        by_configuration: dict[str, list[tuple[dict[str, Any], Mapping[str, Any]]]] = {}
        for key, rows in development_groups.items():
            replayed = selection_replays[key]
            selected_row = next(
                row for row in rows if int(row["epoch"]) == replayed["epoch"]
            )
            by_configuration.setdefault(key[2], []).append((replayed, selected_row))
        summaries = []
        for configuration_id, selected_rows in by_configuration.items():
            if len(selected_rows) != 3:
                raise AssertionError(
                    f"Development configuration lacks three seeds: {configuration_id}"
                )
            eligible_rows = [
                row
                for replayed, row in selected_rows
                if replayed["validation_eligible"]
            ]
            weights = {float(row["evasion_weight"]) for _, row in selected_rows}
            if len(weights) != 1:
                raise AssertionError("A configuration has inconsistent evasion weights")
            summaries.append(
                {
                    "configuration_id": configuration_id,
                    "evasion_weight": weights.pop(),
                    "validation_eligible_count": len(eligible_rows),
                    "mean_tpr_eligible": (
                        float(np.mean([float(row["tpr"]) for row in eligible_rows]))
                        if eligible_rows
                        else None
                    ),
                    "worst_validation_asr": min(
                        float(row["validation_asr"]) for _, row in selected_rows
                    ),
                    "worst_validation_balanced_ca": min(
                        float(row["validation_balanced_ca"])
                        for _, row in selected_rows
                    ),
                }
            )
        selected = replay_development_selection(summaries, receipt)
        unselected_test = [
            row
            for row in victim_rows
            if row["stage"] == "development"
            and str(row["configuration_id"]) != selected
        ]
        if unselected_test:
            raise AssertionError("Test firewall failed: unselected configuration tested")
        report["development_selected_configuration_id"] = selected
    if "confirmation" in stages:
        lock_path = output / "confirmation_lock_receipt.json"
        sidecar = lock_path.with_suffix(lock_path.suffix + ".sha256")
        if not lock_path.is_file() or not sidecar.is_file():
            raise AssertionError("Confirmation ran without a hash-locked receipt")
        if sidecar.read_text().strip() != sha256_file(lock_path):
            raise AssertionError("Confirmation lock hash mismatch")
        confirmation_victims = [
            row for row in victim_rows if row["stage"] == "confirmation"
        ]
        if sorted(int(row["seed"]) for row in confirmation_victims) != list(
            CONFIRMATION_SEEDS
        ):
            raise AssertionError("Confirmation victim block is incomplete")
        confirmation_metrics = [
            row for row in per_seed_rows if row["stage"] == "confirmation"
        ]
        eligibility = {
            int(row["seed"]): _boolean(row["eligible"])
            for row in confirmation_victims
        }
        for row in confirmation_metrics:
            row["eligible"] = eligibility[int(row["seed"])]
        report["confirmation"] = replay_bootstrap(confirmation_metrics)
    if manifest.get("status") not in {"complete", "verified", "synthetic_complete"}:
        raise AssertionError(f"Bundle status is not complete: {manifest.get('status')}")
    return report


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return output


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(verify_bundle(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
