#!/usr/bin/env python3
"""Independent replay, truth join, and audit for protocol-correct reruns."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CERT_ROOT = ROOT / "auditability_certificate_v1"
CONFIG = ROOT / "configs/protocol_correct_certificate_rerun_v1.json"
FREEZE = CERT_ROOT / "PROTOCOL_CORRECT_RERUN_FREEZE_V1.json"
REGISTRAR = CERT_ROOT / "analysis/prepare_protocol_correct_observations.py"
ISSUER = CERT_ROOT / "analysis/issue_protocol_correct_certificates.py"
COMMON = CERT_ROOT / "analysis/protocol_correct_common.py"

import sys
sys.path.insert(0, str(CERT_ROOT))
sys.path.insert(0, str(CERT_ROOT / "analysis"))
from cert.certificate import conformal_rank, certify_realized_counts
from protocol_correct_common import atomic_json, ids_hash, keyed_rng, load_key, opaque_token, sha256, victim_seed

PUBLIC_FIELDS = ["protocol", "trial_token", "k", "conformal_rank", "candidate_count", "flagged_count"]
PRIVATE_FIELDS = [
    "protocol", "trial_token", "victim_id", "detector", "fold", "k", "draw",
    "part", "condition", "epsilon", "calibration_ids_sha256",
    "selected_contaminated_count", "threshold",
]
DECISION_FIELDS = PUBLIC_FIELDS + [
    "interval", "n_poison_min", "n_poison_max", "recall_lo", "recall_hi",
    "clean_flag_rate_hi", "caught_poison_lo", "decision",
]
EVALUATED_FIELDS = PRIVATE_FIELDS + [
    "interval", "n_poison_min", "n_poison_max", "conformal_rank", "candidate_count",
    "flagged_count", "recall_lo", "recall_hi", "clean_flag_rate_hi",
    "caught_poison_lo", "decision", "actual_n_poison_candidate",
    "actual_n_poison_total", "declared_interval_valid", "actual_candidate_recall",
    "actual_system_recall", "actual_clean_flag_rate", "failed_candidate_target",
    "failed_system_target", "failure_recognized", "false_go_candidate",
    "valid_false_go_candidate", "false_go_system", "valid_false_go_system",
]


@lru_cache(maxsize=None)
def cached_certificate(k: int, rank: int, flagged_count: int, candidate_count: int,
                       n_min: int, n_max: int, tau: float, delta: float,
                       alpha_cap: float, fpr_mode: str) -> dict[str, Any]:
    return certify_realized_counts(
        k, rank, flagged_count, candidate_count, tau=tau,
        n_poison_min=n_min, n_poison_max=n_max, delta=delta,
        alpha_cap=alpha_cap, fpr_mode=fpr_mode,
    )


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def close(left: Any, right: Any, message: str) -> None:
    if not math.isclose(float(left), float(right), rel_tol=1e-11, abs_tol=1e-11):
        raise AssertionError(f"{message}: {left} != {right}")


def next_required(reader: csv.DictReader, label: str) -> dict[str, str]:
    try:
        return next(reader)
    except StopIteration as error:
        raise AssertionError(f"{label} ended early") from error


def require_exhausted(reader: csv.DictReader, label: str) -> None:
    try:
        next(reader)
    except StopIteration:
        return
    raise AssertionError(f"{label} contains extra rows")


def position_lookup(ids: np.ndarray, selected_ids: np.ndarray) -> np.ndarray:
    lookup = {int(value): index for index, value in enumerate(ids)}
    try:
        return np.asarray([lookup[int(value)] for value in selected_ids], dtype=np.int64)
    except KeyError as error:
        raise AssertionError("replay id is absent from score fold") from error


def replay_threshold(scores: np.ndarray, selected: np.ndarray, jitter: np.ndarray, alpha: float) -> tuple[int, np.ndarray, np.ndarray, float]:
    rank = conformal_rank(len(selected), alpha)
    order = np.lexsort((jitter[selected], scores[selected]))
    threshold_position = int(selected[order[rank - 1]])
    threshold = float(scores[threshold_position])
    threshold_jitter = float(jitter[threshold_position])
    candidate = np.ones(len(scores), dtype=bool)
    candidate[selected] = False
    flagged = candidate & ((scores > threshold) | ((scores == threshold) & (jitter > threshold_jitter)))
    return rank, candidate, flagged, threshold


def trusted_pool(key: bytes, protocol: str, group: pd.DataFrame, pool_size: int, victim: str, fold: int) -> np.ndarray:
    clean_ids = np.sort(group.loc[group["is_poison"] == 0, "sample_id"].to_numpy(np.int64))
    require(len(clean_ids) > pool_size, "trusted pool is not non-exhaustive")
    chosen = np.sort(keyed_rng(key, "trusted-pool", protocol, victim_seed(victim), fold).choice(clean_ids, pool_size, replace=False))
    return position_lookup(group["sample_id"].to_numpy(np.int64), chosen)


def trial_stream_path(joined: pd.DataFrame, key: bytes, cfg: dict[str, Any]) -> Iterator[dict[str, Any]]:
    protocol = "pathmnist_blind"
    section = cfg["pathmnist_blind"]
    train = joined.loc[joined["role"] == "train"]
    detectors = sorted(train["detector"].unique())
    for (victim, detector, fold), group in train.groupby(["victim_id", "detector", "fold"], sort=True):
        group = group.sort_values("sample_id").reset_index(drop=True)
        ids = group["sample_id"].to_numpy(np.int64)
        scores = group["score"].to_numpy(float)
        poison = group["is_poison"].to_numpy(int).astype(bool)
        pool = trusted_pool(key, protocol, group, int(section["trusted_pool_size_per_victim_fold"]), str(victim), int(fold))
        seed = victim_seed(str(victim))
        detector_index = detectors.index(str(detector))
        for k in map(int, section["trusted_member_budgets"]):
            for draw in range(int(section["draws_per_cell"])):
                selected = keyed_rng(key, "path-selection", seed, detector_index, int(fold), k, draw).choice(pool, k, replace=False)
                jitter = keyed_rng(key, "path-ties", seed, detector_index, int(fold), k, draw).random(len(scores))
                rank, candidate, flagged, threshold = replay_threshold(scores, selected, jitter, float(cfg["common"]["alpha"]))
                yield {
                    "protocol": protocol, "trial_token": opaque_token(key, protocol, victim, detector, fold, k, draw),
                    "victim_id": victim, "detector": detector, "fold": int(fold), "k": k, "draw": draw,
                    "part": "ordinary", "condition": "protected", "epsilon": 0.0,
                    "calibration_ids_sha256": ids_hash(ids[selected]), "selected_contaminated_count": 0,
                    "threshold": threshold, "rank": rank, "candidate": candidate, "flagged": flagged, "poison": poison,
                }


def trial_stream_redteam(joined: pd.DataFrame, key: bytes, cfg: dict[str, Any]) -> Iterator[dict[str, Any]]:
    protocol = "certificate_aware_redteam"
    section = cfg["certificate_aware_redteam"]
    train = joined.loc[joined["role"] == "train"]
    detectors = sorted(train["detector"].unique())
    for (victim, detector, fold), group in train.groupby(["victim_id", "detector", "fold"], sort=True):
        group = group.sort_values("sample_id").reset_index(drop=True)
        ids = group["sample_id"].to_numpy(np.int64)
        scores = group["score"].to_numpy(float)
        poison = group["is_poison"].to_numpy(int).astype(bool)
        clean_pool = trusted_pool(key, protocol, group, int(section["trusted_pool_size_per_victim_fold"]), str(victim), int(fold))
        poison_ids = np.sort(group.loc[group["is_poison"] == 1, "sample_id"].to_numpy(np.int64))
        private_poison_ids = np.sort(keyed_rng(key, "attacker-pool", victim_seed(str(victim)), int(fold)).choice(poison_ids, int(section["private_attacker_pool_size_per_victim_fold"]), replace=False))
        poison_pool = position_lookup(ids, private_poison_ids)
        seed = victim_seed(str(victim))
        detector_index = detectors.index(str(detector))
        for k in map(int, section["trusted_member_budgets"]):
            draws = int(section["primary_draws_per_cell"] if k == int(section["primary_k"]) else section["secondary_draws_per_cell"])
            for draw in range(draws):
                protected = keyed_rng(key, "protected-selection", seed, detector_index, int(fold), k, draw).choice(clean_pool, k, replace=False)
                low_order = np.lexsort((ids[clean_pool], scores[clean_pool]))
                low_pool = clean_pool[low_order[: int(section["low_score_pool_multiple"]) * k]]
                leaked = keyed_rng(key, "leaked-selection", seed, detector_index, int(fold), k, draw).choice(low_pool, k, replace=False)
                base_clean = keyed_rng(key, "contamination-clean", seed, detector_index, int(fold), k, draw).choice(clean_pool, k, replace=False)
                replacement_order = keyed_rng(key, "contamination-replacement", seed, detector_index, int(fold), k, draw).permutation(k)
                poison_order = keyed_rng(key, "contamination-poison", seed, detector_index, int(fold), k, draw).permutation(poison_pool)
                conditions: list[tuple[str, str, float, np.ndarray, int]] = [
                    ("a1", "protected", 0.0, protected, 0),
                    ("a1", "leaked", 0.0, leaked, 0),
                ]
                for epsilon in map(float, section["contamination_fractions"]):
                    count = int(np.floor(epsilon * k))
                    selected = base_clean.copy()
                    if count:
                        selected[replacement_order[:count]] = poison_order[:count]
                    conditions.append(("a2", "contamination", epsilon, selected, count))
                for part, condition, epsilon, selected, contaminated in conditions:
                    jitter = keyed_rng(key, "redteam-ties", seed, detector_index, int(fold), k, draw, part, condition, epsilon).random(len(scores))
                    rank, candidate, flagged, threshold = replay_threshold(scores, selected, jitter, float(cfg["common"]["alpha"]))
                    yield {
                        "protocol": protocol,
                        "trial_token": opaque_token(key, protocol, victim, detector, fold, k, draw, part, condition, epsilon),
                        "victim_id": victim, "detector": detector, "fold": int(fold), "k": k, "draw": draw,
                        "part": part, "condition": condition, "epsilon": epsilon,
                        "calibration_ids_sha256": ids_hash(ids[selected]), "selected_contaminated_count": contaminated,
                        "threshold": threshold, "rank": rank, "candidate": candidate, "flagged": flagged, "poison": poison,
                    }


def verify_frozen_files(key: bytes) -> dict[str, Any]:
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    require(frozen["status"] == "POST_UNBLINDING_REPLAY_FROZEN_BEFORE_FRESH_SECRET_DRAWS", "freeze status mismatch")
    require(frozen["key_commitment_sha256"] == __import__("hashlib").sha256(key).hexdigest(), "key commitment mismatch")
    expected = {
        "config_sha256": sha256(CONFIG),
        "protocol_sha256": sha256(CERT_ROOT / "PROTOCOL_CORRECT_RERUN_V1.md"),
        "common_sha256": sha256(COMMON),
        "registrar_sha256": sha256(REGISTRAR),
        "issuer_sha256": sha256(ISSUER),
        "verifier_sha256": sha256(Path(__file__)),
        "certificate_library_sha256": sha256(CERT_ROOT / "cert/certificate.py"),
    }
    for name, value in expected.items():
        require(frozen["hashes"].get(name) == value, f"frozen hash mismatch: {name}")
    issuer_text = ISSUER.read_text(encoding="utf-8")
    for banned in ("private_truth", "is_poison", "pool_role", "selected_clean", "selected_poison"):
        require(banned not in issuer_text, f"issuer contains prohibited identifier: {banned}")
    return frozen


def update_stats(bucket: dict[str, float], decision: str, valid: bool, false_candidate: bool, false_system: bool, failed: bool, recall: float, clean_rate: float) -> None:
    bucket["rows"] += 1
    bucket[f"decision_{decision}"] += 1
    bucket["valid"] += int(valid)
    if valid:
        bucket["false_candidate"] += int(false_candidate)
        bucket["false_system"] += int(false_system)
        bucket["failed"] += int(failed)
        bucket["failure_recognized"] += int(failed and decision != "GO")
        bucket["sum_recall"] += recall
        bucket["sum_clean_rate"] += clean_rate


def verify(mode: str, source: Path, output: Path, key_path: Path) -> dict[str, Any]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    key = load_key(key_path)
    frozen = verify_frozen_files(key)
    protocol = "pathmnist_blind" if mode == "path" else "certificate_aware_redteam"
    section = cfg["pathmnist_blind" if mode == "path" else "certificate_aware_redteam"]
    expected_score_hash = frozen["hashes"]["path_score_sha256" if mode == "path" else "redteam_score_sha256"]
    expected_eval_hash = frozen["hashes"]["path_evaluation_sha256" if mode == "path" else "redteam_evaluation_sha256"]
    require(sha256(source / "blind_scores.csv") == expected_score_hash, "frozen score input drift")
    require(sha256(source / "private_truth.csv") == expected_eval_hash, "frozen evaluation input drift")

    public_manifest_path = output / "public/issuer_manifest.json"
    registrar_manifest_path = output / "private/registrar_manifest.json"
    observation_path = output / "public/issuer_observations.csv"
    map_path = output / "private/trial_map.csv"
    seal_path = output / "sealed/issuer_seal_manifest.json"
    decision_path = output / "sealed/certificate_decisions_sealed.csv"
    public_manifest = json.loads(public_manifest_path.read_text(encoding="utf-8"))
    registrar_manifest = json.loads(registrar_manifest_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    require(public_manifest["protocol"] == protocol and seal["protocol"] == protocol, "protocol mismatch")
    require(public_manifest["allowed_schema"] == PUBLIC_FIELDS, "public schema allow-list mismatch")
    require(public_manifest["prohibited_fields_absent"] is True, "registrar did not attest field isolation")
    require(sha256(observation_path) == public_manifest["issuer_observations_sha256"], "public observations drift")
    require(sha256(map_path) == registrar_manifest["trial_map_sha256"], "private trial map drift")
    require(sha256(decision_path) == seal["certificate_decisions_sealed_sha256"], "sealed decision drift")
    require(float(seal["sealed_unix"]) >= float(public_manifest["created_unix"]), "decisions predate observation seal")
    require(seal["key_commitment_sha256"] == public_manifest["key_commitment_sha256"] == frozen["key_commitment_sha256"], "commitment chain mismatch")

    scores = pd.read_csv(source / "blind_scores.csv")
    truth = pd.read_csv(source / "private_truth.csv")
    require(not scores.duplicated(["victim_id", "detector", "fold", "sample_id"]).any(), "duplicate score key")
    require(not truth.duplicated(["victim_id", "sample_id"]).any(), "duplicate evaluation key")
    require(np.isfinite(scores["score"].to_numpy(float)).all(), "non-finite scores")
    joined = scores.merge(truth, on=["victim_id", "sample_id", "role"], how="left", validate="many_to_one")
    require(not joined["is_poison"].isna().any(), "incomplete evaluation join")

    verified_dir = output / "verified"
    verified_dir.mkdir(parents=True, exist_ok=True)
    evaluated_path = verified_dir / "certificate_decisions_evaluated.csv"
    evaluated_tmp = evaluated_path.with_name(evaluated_path.name + ".tmp")
    stats: dict[tuple[Any, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    primary: dict[tuple[Any, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    tokens: set[str] = set()
    checked_trials = 0
    checked_decisions = 0
    stream = trial_stream_path(joined, key, cfg) if mode == "path" else trial_stream_redteam(joined, key, cfg)
    intervals = {name: tuple(map(int, bounds)) for name, bounds in section["declared_intervals_per_fold"].items()}
    with observation_path.open(newline="", encoding="utf-8") as observation_handle, map_path.open(newline="", encoding="utf-8") as map_handle, decision_path.open(newline="", encoding="utf-8") as decision_handle, evaluated_tmp.open("w", newline="", encoding="utf-8") as evaluated_handle:
        observation_reader = csv.DictReader(observation_handle)
        map_reader = csv.DictReader(map_handle)
        decision_reader = csv.DictReader(decision_handle)
        require(observation_reader.fieldnames == PUBLIC_FIELDS, "public observation schema mismatch")
        require(map_reader.fieldnames == PRIVATE_FIELDS, "private map schema mismatch")
        require(decision_reader.fieldnames == DECISION_FIELDS, "sealed decision schema mismatch")
        evaluated_writer = csv.DictWriter(evaluated_handle, fieldnames=EVALUATED_FIELDS)
        evaluated_writer.writeheader()
        for expected in stream:
            observed = next_required(observation_reader, "public observations")
            mapped = next_required(map_reader, "private trial map")
            token = expected["trial_token"]
            require(token not in tokens, "duplicate opaque trial token")
            tokens.add(token)
            for name in ("protocol", "trial_token"):
                require(observed[name] == str(expected[name]), f"observation mismatch: {name}")
                require(mapped[name] == str(expected[name]), f"private map mismatch: {name}")
            for name in ("victim_id", "detector", "part", "condition"):
                require(mapped[name] == str(expected[name]), f"private map mismatch: {name}")
            for name in ("fold", "k", "draw", "selected_contaminated_count"):
                require(int(mapped[name]) == int(expected[name]), f"private map mismatch: {name}")
            close(mapped["epsilon"], expected["epsilon"], "private map mismatch: epsilon")
            close(mapped["threshold"], expected["threshold"], "private map mismatch: threshold")
            require(mapped["calibration_ids_sha256"] == expected["calibration_ids_sha256"], "calibration hash mismatch")
            rank = int(expected["rank"])
            candidate = expected["candidate"]
            flagged = expected["flagged"]
            poison = expected["poison"]
            require(int(observed["k"]) == int(expected["k"]), "public k mismatch")
            require(int(observed["conformal_rank"]) == rank, "public rank mismatch")
            require(int(observed["candidate_count"]) == int(candidate.sum()), "public candidate count mismatch")
            require(int(observed["flagged_count"]) == int(flagged.sum()), "public flagged count mismatch")

            total_poison = int(poison.sum())
            candidate_poison = int(np.sum(candidate & poison))
            caught = int(np.sum(flagged & poison))
            candidate_recall = caught / max(candidate_poison, 1)
            system_recall = caught / max(total_poison, 1)
            clean_rate = int(np.sum(flagged & ~poison)) / max(int(np.sum(candidate & ~poison)), 1)
            for interval, (n_min, n_max) in intervals.items():
                decision_row = next_required(decision_reader, "sealed decisions")
                for name in PUBLIC_FIELDS:
                    require(decision_row[name] == observed[name], f"sealed/public mismatch: {name}")
                require(decision_row["interval"] == interval, "interval order mismatch")
                require(int(decision_row["n_poison_min"]) == n_min and int(decision_row["n_poison_max"]) == n_max, "declared interval mismatch")
                result = cached_certificate(
                    int(expected["k"]), rank, int(flagged.sum()), int(candidate.sum()),
                    n_min, n_max, float(cfg["common"]["tau"]),
                    float(section["delta_per_cell"]), float(cfg["common"]["alpha_cap"]),
                    str(cfg["common"]["fpr_mode"]),
                )
                decision = {1: "GO", -1: "NO_GO", 0: "ABSTAIN"}[int(result["decision"])]
                require(decision_row["decision"] == decision, "certificate decision mismatch")
                close(decision_row["recall_lo"], result["recall_lo"], "recall lower mismatch")
                close(decision_row["recall_hi"], result["recall_hi"], "recall upper mismatch")
                close(decision_row["clean_flag_rate_hi"], result["clean_flag_rate_hi"], "clean-rate bound mismatch")
                valid = n_min <= candidate_poison <= n_max
                failed_candidate = candidate_recall < float(cfg["common"]["tau"]) or clean_rate > float(cfg["common"]["alpha_cap"])
                failed_system = system_recall < float(cfg["common"]["tau"]) or clean_rate > float(cfg["common"]["alpha_cap"])
                false_candidate = decision == "GO" and failed_candidate
                false_system = decision == "GO" and failed_system
                evaluated_writer.writerow({
                    **{name: mapped[name] for name in PRIVATE_FIELDS},
                    "interval": interval, "n_poison_min": n_min, "n_poison_max": n_max,
                    "conformal_rank": rank, "candidate_count": int(candidate.sum()), "flagged_count": int(flagged.sum()),
                    "recall_lo": decision_row["recall_lo"], "recall_hi": decision_row["recall_hi"],
                    "clean_flag_rate_hi": decision_row["clean_flag_rate_hi"], "caught_poison_lo": decision_row["caught_poison_lo"],
                    "decision": decision, "actual_n_poison_candidate": candidate_poison,
                    "actual_n_poison_total": total_poison, "declared_interval_valid": int(valid),
                    "actual_candidate_recall": repr(candidate_recall), "actual_system_recall": repr(system_recall),
                    "actual_clean_flag_rate": repr(clean_rate), "failed_candidate_target": int(failed_candidate),
                    "failed_system_target": int(failed_system), "failure_recognized": int(failed_candidate and decision != "GO"),
                    "false_go_candidate": int(false_candidate), "valid_false_go_candidate": int(false_candidate and valid),
                    "false_go_system": int(false_system), "valid_false_go_system": int(false_system and valid),
                })
                cell_key = (expected["part"], expected["condition"], float(expected["epsilon"]), expected["detector"], int(expected["fold"]), int(expected["k"]), interval)
                update_stats(stats[cell_key], decision, valid, false_candidate, false_system, failed_candidate, candidate_recall, clean_rate)
                if mode == "redteam" and expected["part"] == "a2" and int(expected["k"]) == int(section["primary_k"]) and interval == "primary":
                    update_stats(primary[(float(expected["epsilon"]), expected["detector"], int(expected["fold"]))], decision, valid, false_candidate, false_system, failed_candidate, candidate_recall, clean_rate)
                checked_decisions += 1
            checked_trials += 1
        require_exhausted(observation_reader, "public observations")
        require_exhausted(map_reader, "private trial map")
        require_exhausted(decision_reader, "sealed decisions")
    os.replace(evaluated_tmp, evaluated_path)
    require(checked_trials == int(public_manifest["rows"]), "verified trial count mismatch")
    require(checked_decisions == int(seal["decision_rows"]), "verified decision count mismatch")

    summary_rows: list[dict[str, Any]] = []
    for cell_key, bucket in sorted(stats.items()):
        part, condition, epsilon, detector, fold, k, interval = cell_key
        rows = int(bucket["rows"])
        valid_rows = int(bucket["valid"])
        summary_rows.append({
            "part": part, "condition": condition, "epsilon": epsilon, "detector": detector,
            "fold": fold, "k": k, "interval": interval, "rows": rows, "valid_rows": valid_rows,
            "go_rate": bucket["decision_GO"] / rows, "nogo_rate": bucket["decision_NO_GO"] / rows,
            "abstain_rate": bucket["decision_ABSTAIN"] / rows,
            "valid_false_go_candidate_count": int(bucket["false_candidate"]),
            "valid_false_go_candidate_rate": bucket["false_candidate"] / max(valid_rows, 1),
            "valid_false_go_system_count": int(bucket["false_system"]),
            "failure_recognition_rate": bucket["failure_recognized"] / max(bucket["failed"], 1),
            "mean_candidate_recall": bucket["sum_recall"] / max(valid_rows, 1),
            "mean_clean_flag_rate": bucket["sum_clean_rate"] / max(valid_rows, 1),
        })
    summary_path = verified_dir / "verification_cell_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    valid_false_go = int(sum(bucket["false_candidate"] for bucket in stats.values()))
    valid_rows = int(sum(bucket["valid"] for bucket in stats.values()))
    report: dict[str, Any] = {
        "status": "PASS_PROTOCOL_CORRECT_INDEPENDENT_REPLAY",
        "protocol": protocol,
        "scope": "post-unblinding protocol replay over archived sealed scores",
        "checked_trials": checked_trials,
        "checked_decisions": checked_decisions,
        "unique_trial_tokens": len(tokens),
        "valid_false_go_candidate_count_all_cells": valid_false_go,
        "valid_decision_rows_all_cells": valid_rows,
        "valid_false_go_candidate_rate_all_cells": valid_false_go / max(valid_rows, 1),
        "issuer_static_prohibited_identifier_scan": "PASS",
        "issuer_input_schema_allowlist": "PASS",
        "decision_sealed_before_verification": True,
        "non_exhaustive_trusted_pool": True,
        "hashes": {
            "public_manifest_sha256": sha256(public_manifest_path),
            "registrar_manifest_sha256": sha256(registrar_manifest_path),
            "issuer_seal_manifest_sha256": sha256(seal_path),
            "certificate_decisions_sealed_sha256": sha256(decision_path),
            "certificate_decisions_evaluated_sha256": sha256(evaluated_path),
            "verification_cell_summary_sha256": sha256(summary_path),
            "freeze_manifest_sha256": sha256(FREEZE),
            "verifier_sha256": sha256(Path(__file__)),
        },
        "verified_unix": time.time(),
    }
    if mode == "redteam":
        primary_rows: list[dict[str, Any]] = []
        statuses: list[str] = []
        for cell_key, bucket in sorted(primary.items()):
            epsilon, detector, fold = cell_key
            trials = int(bucket["valid"])
            failures = int(bucket["false_candidate"])
            rate = failures / max(trials, 1)
            upper = 1.0 - 0.05 ** (1.0 / trials) if failures == 0 else min(1.0, rate + math.sqrt(math.log(20.0) / (2.0 * trials)))
            status = "SUPPORTED" if failures == 0 and upper <= float(section["delta_per_cell"]) else "INCONCLUSIVE"
            statuses.append(status)
            primary_rows.append({
                "epsilon": epsilon, "detector": detector, "fold": fold, "valid_draws": trials,
                "false_go_candidate_count": failures, "false_go_candidate_rate": rate,
                "false_go_one_sided_95pct_upper": upper, "delta_cell": float(section["delta_per_cell"]),
                "evidence_status": status, "go_rate": bucket["decision_GO"] / max(bucket["rows"], 1),
            })
        primary_path = verified_dir / "redteam_primary_cell_inference.csv"
        pd.DataFrame(primary_rows).to_csv(primary_path, index=False)
        report["primary_safety_cells"] = len(primary_rows)
        report["primary_supported_cells"] = int(sum(value == "SUPPORTED" for value in statuses))
        report["scientific_primary_status"] = "SUPPORTED" if statuses and all(value == "SUPPORTED" for value in statuses) else "INCONCLUSIVE"
        report["hashes"]["redteam_primary_cell_inference_sha256"] = sha256(primary_path)
        a1 = {}
        for name in ("protected", "leaked"):
            selected = [bucket for key, bucket in stats.items() if key[0] == "a1" and key[1] == name and key[5] == int(section["primary_k"]) and key[6] == "primary"]
            failures = int(sum(bucket["false_candidate"] for bucket in selected))
            trials = int(sum(bucket["valid"] for bucket in selected))
            a1[name] = {"valid_false_go_count": failures, "valid_draws": trials, "valid_false_go_rate": failures / max(trials, 1)}
        report["a1_mechanism_ablation"] = a1
    else:
        primary_cells = [bucket for key, bucket in stats.items() if key[6] == "primary"]
        primary_failures = int(sum(bucket["false_candidate"] for bucket in primary_cells))
        primary_valid = int(sum(bucket["valid"] for bucket in primary_cells))
        report["primary_false_go_count"] = primary_failures
        report["primary_valid_decision_rows"] = primary_valid
        report["primary_false_go_rate"] = primary_failures / max(primary_valid, 1)

    revealed_key_path = output / "private/revealed_after_seal_key.hex"
    revealed_key_path.write_text(key.hex() + "\n", encoding="ascii")
    report["hashes"]["revealed_key_sha256"] = sha256(revealed_key_path)
    report_path = verified_dir / "verification_report.json"
    atomic_json(report_path, report)
    final_manifest = {
        "status": report["status"],
        "protocol": protocol,
        "issuer_seal_preserved": True,
        "verification_report_sha256": sha256(report_path),
        "verified_unix": report["verified_unix"],
    }
    atomic_json(output / "verification_manifest.json", final_manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("path", "redteam"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()
    verify(args.mode, args.source, args.output, args.key_file)


if __name__ == "__main__":
    main()
