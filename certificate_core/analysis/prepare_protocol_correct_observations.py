#!/usr/bin/env python3
"""Truth-aware registrar that emits only issuer sufficient statistics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CERT_ROOT = ROOT / "auditability_certificate_v1"
CONFIG = ROOT / "configs/protocol_correct_certificate_rerun_v1.json"

import sys
sys.path.insert(0, str(CERT_ROOT))
sys.path.insert(0, str(CERT_ROOT / "analysis"))
from cert.certificate import conformal_rank
from protocol_correct_common import atomic_json, ids_hash, keyed_rng, load_key, opaque_token, sha256, victim_seed

PUBLIC_FIELDS = ["protocol", "trial_token", "k", "conformal_rank", "candidate_count", "flagged_count"]
PRIVATE_FIELDS = [
    "protocol", "trial_token", "victim_id", "detector", "fold", "k", "draw",
    "part", "condition", "epsilon", "calibration_ids_sha256",
    "selected_contaminated_count", "threshold",
]


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    if count == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("refusing to seal an empty registrar output")
    os.replace(temporary, path)
    return count


def validate_source(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    score_path = source / "blind_scores.csv"
    truth_path = source / "private_truth.csv"
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scores = pd.read_csv(score_path)
    truth = pd.read_csv(truth_path)
    required_scores = {"victim_id", "detector", "fold", "sample_id", "score", "role"}
    if not required_scores.issubset(scores.columns):
        raise RuntimeError("score schema mismatch")
    if set(truth.columns) != {"victim_id", "sample_id", "role", "is_poison"}:
        raise RuntimeError("private evaluation schema mismatch")
    if scores.duplicated(["victim_id", "detector", "fold", "sample_id"]).any():
        raise RuntimeError("duplicate score key")
    if truth.duplicated(["victim_id", "sample_id"]).any():
        raise RuntimeError("duplicate evaluation key")
    if not np.isfinite(scores["score"].to_numpy(float)).all():
        raise RuntimeError("non-finite score")
    if "blind_scores_sha256" in manifest and sha256(score_path) != manifest["blind_scores_sha256"]:
        raise RuntimeError("score hash mismatch")
    if "private_truth_sha256" in manifest and sha256(truth_path) != manifest["private_truth_sha256"]:
        raise RuntimeError("evaluation hash mismatch")
    return scores, truth, manifest


def threshold_statistics(scores: np.ndarray, selected: np.ndarray, jitter: np.ndarray, alpha: float) -> tuple[int, int, int, float]:
    rank = conformal_rank(len(selected), alpha)
    order = np.lexsort((jitter[selected], scores[selected]))
    threshold_position = int(selected[order[rank - 1]])
    threshold = float(scores[threshold_position])
    threshold_jitter = float(jitter[threshold_position])
    candidate = np.ones(len(scores), dtype=bool)
    candidate[selected] = False
    flagged = candidate & ((scores > threshold) | ((scores == threshold) & (jitter > threshold_jitter)))
    return rank, int(candidate.sum()), int(flagged.sum()), threshold


def clean_pool_for_fold(key: bytes, protocol: str, group: pd.DataFrame, pool_size: int, victim: str, fold: int) -> np.ndarray:
    clean = np.sort(group.loc[group["is_poison"] == 0, "sample_id"].to_numpy(np.int64))
    if len(clean) <= pool_size:
        raise RuntimeError("trusted pool must be a strict, non-exhaustive subset")
    return np.sort(keyed_rng(key, "trusted-pool", protocol, victim_seed(victim), fold).choice(clean, pool_size, replace=False))


def position_lookup(ids: np.ndarray, selected_ids: np.ndarray) -> np.ndarray:
    lookup = {int(value): index for index, value in enumerate(ids)}
    try:
        return np.asarray([lookup[int(value)] for value in selected_ids], dtype=np.int64)
    except KeyError as error:
        raise RuntimeError("registrar selected an id absent from the score fold") from error


def emit_pair(public_handle: Any, private_handle: Any, public: dict[str, Any], private: dict[str, Any]) -> None:
    public_handle.writerow(public)
    private_handle.writerow(private)


def register_path(scores: pd.DataFrame, truth: pd.DataFrame, key: bytes, cfg: dict[str, Any], public_writer: Any, private_writer: Any) -> int:
    protocol = "pathmnist_blind"
    joined = scores.merge(truth, on=["victim_id", "sample_id", "role"], how="left", validate="many_to_one")
    if joined["is_poison"].isna().any():
        raise RuntimeError("incomplete registrar join")
    rows = 0
    alpha = float(cfg["common"]["alpha"])
    section = cfg["pathmnist_blind"]
    train = joined.loc[joined["role"] == "train"]
    for (victim, detector, fold), group in train.groupby(["victim_id", "detector", "fold"], sort=True):
        group = group.sort_values("sample_id").reset_index(drop=True)
        ids = group["sample_id"].to_numpy(np.int64)
        values = group["score"].to_numpy(float)
        pool_ids = clean_pool_for_fold(key, protocol, group, int(section["trusted_pool_size_per_victim_fold"]), str(victim), int(fold))
        pool = position_lookup(ids, pool_ids)
        seed = victim_seed(str(victim))
        detector_index = sorted(train["detector"].unique()).index(str(detector))
        for k in map(int, section["trusted_member_budgets"]):
            for draw in range(int(section["draws_per_cell"])):
                selected = keyed_rng(key, "path-selection", seed, detector_index, int(fold), k, draw).choice(pool, k, replace=False)
                jitter = keyed_rng(key, "path-ties", seed, detector_index, int(fold), k, draw).random(len(values))
                rank, candidate_count, flagged_count, threshold = threshold_statistics(values, selected, jitter, alpha)
                token = opaque_token(key, protocol, victim, detector, fold, k, draw)
                emit_pair(public_writer, private_writer,
                    {"protocol": protocol, "trial_token": token, "k": k, "conformal_rank": rank, "candidate_count": candidate_count, "flagged_count": flagged_count},
                    {"protocol": protocol, "trial_token": token, "victim_id": victim, "detector": detector, "fold": int(fold), "k": k, "draw": draw, "part": "ordinary", "condition": "protected", "epsilon": 0.0, "calibration_ids_sha256": ids_hash(ids[selected]), "selected_contaminated_count": 0, "threshold": repr(threshold)},
                )
                rows += 1
    return rows


def register_redteam(scores: pd.DataFrame, truth: pd.DataFrame, key: bytes, cfg: dict[str, Any], public_writer: Any, private_writer: Any) -> int:
    protocol = "certificate_aware_redteam"
    joined = scores.merge(truth, on=["victim_id", "sample_id", "role"], how="left", validate="many_to_one")
    if joined["is_poison"].isna().any():
        raise RuntimeError("incomplete registrar join")
    rows = 0
    alpha = float(cfg["common"]["alpha"])
    section = cfg["certificate_aware_redteam"]
    train = joined.loc[joined["role"] == "train"]
    detectors = sorted(train["detector"].unique())
    for (victim, detector, fold), group in train.groupby(["victim_id", "detector", "fold"], sort=True):
        group = group.sort_values("sample_id").reset_index(drop=True)
        ids = group["sample_id"].to_numpy(np.int64)
        values = group["score"].to_numpy(float)
        pool_ids = clean_pool_for_fold(key, protocol, group, int(section["trusted_pool_size_per_victim_fold"]), str(victim), int(fold))
        clean_pool = position_lookup(ids, pool_ids)
        poison_ids = np.sort(group.loc[group["is_poison"] == 1, "sample_id"].to_numpy(np.int64))
        private_poison_ids = np.sort(keyed_rng(key, "attacker-pool", victim_seed(str(victim)), int(fold)).choice(poison_ids, int(section["private_attacker_pool_size_per_victim_fold"]), replace=False))
        poison_pool = position_lookup(ids, private_poison_ids)
        seed = victim_seed(str(victim))
        detector_index = detectors.index(str(detector))
        for k in map(int, section["trusted_member_budgets"]):
            draws = int(section["primary_draws_per_cell"] if k == int(section["primary_k"]) else section["secondary_draws_per_cell"])
            for draw in range(draws):
                protected = keyed_rng(key, "protected-selection", seed, detector_index, int(fold), k, draw).choice(clean_pool, k, replace=False)
                low_order = np.lexsort((ids[clean_pool], values[clean_pool]))
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
                    jitter = keyed_rng(key, "redteam-ties", seed, detector_index, int(fold), k, draw, part, condition, epsilon).random(len(values))
                    rank, candidate_count, flagged_count, threshold = threshold_statistics(values, selected, jitter, alpha)
                    token = opaque_token(key, protocol, victim, detector, fold, k, draw, part, condition, epsilon)
                    emit_pair(public_writer, private_writer,
                        {"protocol": protocol, "trial_token": token, "k": k, "conformal_rank": rank, "candidate_count": candidate_count, "flagged_count": flagged_count},
                        {"protocol": protocol, "trial_token": token, "victim_id": victim, "detector": detector, "fold": int(fold), "k": k, "draw": draw, "part": part, "condition": condition, "epsilon": epsilon, "calibration_ids_sha256": ids_hash(ids[selected]), "selected_contaminated_count": contaminated, "threshold": repr(threshold)},
                    )
                    rows += 1
    return rows


def run(mode: str, source: Path, output: Path, key_path: Path) -> dict[str, Any]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    scores, truth, source_manifest = validate_source(source)
    key = load_key(key_path)
    public_dir = output / "public"
    private_dir = output / "private"
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    observation_path = public_dir / "issuer_observations.csv"
    map_path = private_dir / "trial_map.csv"
    observation_tmp = observation_path.with_name(observation_path.name + ".tmp")
    map_tmp = map_path.with_name(map_path.name + ".tmp")
    with observation_tmp.open("w", newline="", encoding="utf-8") as public_handle, map_tmp.open("w", newline="", encoding="utf-8") as private_handle:
        public_writer = csv.DictWriter(public_handle, fieldnames=PUBLIC_FIELDS)
        private_writer = csv.DictWriter(private_handle, fieldnames=PRIVATE_FIELDS)
        public_writer.writeheader()
        private_writer.writeheader()
        if mode == "path":
            rows = register_path(scores, truth, key, cfg, public_writer, private_writer)
            protocol = "pathmnist_blind"
        else:
            rows = register_redteam(scores, truth, key, cfg, public_writer, private_writer)
            protocol = "certificate_aware_redteam"
    os.replace(observation_tmp, observation_path)
    os.replace(map_tmp, map_path)
    public_manifest = {
        "status": "ISSUER_OBSERVATIONS_SEALED",
        "protocol": protocol,
        "rows": rows,
        "key_commitment_sha256": __import__("hashlib").sha256(key).hexdigest(),
        "config_sha256": sha256(CONFIG),
        "score_sha256": sha256(source / "blind_scores.csv"),
        "issuer_observations_sha256": sha256(observation_path),
        "created_unix": time.time(),
        "allowed_schema": PUBLIC_FIELDS,
        "prohibited_fields_absent": True,
    }
    atomic_json(public_dir / "issuer_manifest.json", public_manifest)
    private_manifest = {
        **public_manifest,
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "evaluation_sha256": sha256(source / "private_truth.csv"),
        "trial_map_sha256": sha256(map_path),
        "registrar_sha256": sha256(Path(__file__)),
        "source_manifest_status": source_manifest.get("status"),
    }
    atomic_json(private_dir / "registrar_manifest.json", private_manifest)
    print(json.dumps(private_manifest, indent=2, sort_keys=True))
    return private_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("path", "redteam"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()
    run(args.mode, args.source, args.output, args.key_file)


if __name__ == "__main__":
    main()
