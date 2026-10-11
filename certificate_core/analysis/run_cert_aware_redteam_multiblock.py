#!/usr/bin/env python3
"""Truth-separated formal issuer for the certificate-aware A1/A2 red-team."""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CERT_ROOT = ROOT / "auditability_certificate_v1"
CONFIG = ROOT / "configs/cert_aware_redteam_multiblock_v2.json"
SOURCE_CONFIG = ROOT / "configs/pathmnist_certaware_redteam_source_v2.json"
PREREG = CERT_ROOT / "PREREG_CERT_AWARE_REDTEAM_MULTIBLOCK_V1.md"
AMENDMENT = CERT_ROOT / "CERT_AWARE_REDTEAM_OPERATIONAL_AMENDMENT_V2.md"
FREEZE = CERT_ROOT / "CERT_AWARE_REDTEAM_MULTIBLOCK_FREEZE_V2.json"
SOURCE_RUNNER = ROOT / "src/run_pathmnist_certaware_redteam_source.py"
VERIFIER = CERT_ROOT / "analysis/verify_cert_aware_redteam_multiblock.py"

import sys
sys.path.insert(0, str(CERT_ROOT))
from cert.certificate import conformal_rank, certify_realized_counts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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
        raise RuntimeError(f"refusing empty CSV: {path}")
    os.replace(temporary, path)
    return count


def victim_seed(victim_id: str) -> int:
    suffix = str(victim_id).rsplit("seed", 1)
    if len(suffix) != 2 or not suffix[1].isdigit():
        raise RuntimeError(f"victim id lacks terminal seed: {victim_id}")
    return int(suffix[1])


def load_key(path: Path, frozen: dict[str, Any]) -> bytes:
    raw = path.read_bytes()
    if len(raw) != 32:
        raise RuntimeError("defender key must contain exactly 32 raw bytes")
    if hashlib.sha256(raw).hexdigest() != frozen.get("defender_key_sha256"):
        raise RuntimeError("defender key does not match frozen commitment")
    return raw


def keyed_rng(key: bytes, domain: str, *parts: Any) -> np.random.Generator:
    message = "|".join([domain, *(str(value) for value in parts)]).encode("utf-8")
    digest = hmac.new(key, message, hashlib.sha256).digest()
    entropy = np.frombuffer(digest[:16], dtype=np.uint32).astype(np.uint64).tolist()
    return np.random.default_rng(np.random.SeedSequence(entropy))


def verify_freeze() -> dict[str, Any]:
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    required = {
        "config_sha256": sha256(CONFIG),
        "source_config_sha256": sha256(SOURCE_CONFIG),
        "prereg_sha256": sha256(PREREG),
        "operational_amendment_sha256": sha256(AMENDMENT),
        "runner_sha256": sha256(Path(__file__)),
        "verifier_sha256": sha256(VERIFIER),
        "source_runner_sha256": sha256(SOURCE_RUNNER),
        "certificate_library_sha256": sha256(CERT_ROOT / "cert/certificate.py"),
    }
    if frozen.get("status") != "FROZEN_BEFORE_FORMAL_VICTIM_TRAINING":
        raise RuntimeError("red-team freeze status is invalid")
    for name, value in required.items():
        if frozen.get("hashes", {}).get(name) != value:
            raise RuntimeError(f"red-team frozen hash mismatch: {name}")
    return frozen


def validate_source(source: Path, config: dict[str, Any], frozen: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = source / "manifest.json"
    score_path = source / "blind_scores.csv"
    truth_path = source / "private_truth.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "SCORE_HANDOFF_COMPLETE" or manifest.get("unblinded") is not False:
        raise RuntimeError("formal source is not a sealed score handoff")
    if manifest.get("redteam_freeze_manifest_sha256") != sha256(FREEZE):
        raise RuntimeError("formal source is not bound to this freeze")
    if float(manifest.get("victim_training_started_unix", 0.0)) < float(frozen["created_unix"]):
        raise RuntimeError("formal victim training predates the freeze")
    if sha256(score_path) != manifest.get("blind_scores_sha256"):
        raise RuntimeError("source score hash mismatch")
    if sha256(truth_path) != manifest.get("private_truth_sha256"):
        raise RuntimeError("source truth hash mismatch")
    scores = pd.read_csv(score_path)
    required = set(config["formal_input_contract"]["required_score_columns"])
    if not required.issubset(scores.columns):
        raise RuntimeError("source score schema mismatch")
    if scores.duplicated(["victim_id", "detector", "fold", "sample_id"]).any():
        raise RuntimeError("duplicate score key")
    if not np.isfinite(scores["score"].to_numpy(float)).all():
        raise RuntimeError("non-finite source score")
    if set(scores["detector"].unique()) != set(config["detectors"]):
        raise RuntimeError("detector coverage mismatch")
    if set(scores["fold"].astype(int).unique()) != set(config["folds"]):
        raise RuntimeError("fold coverage mismatch")
    seeds = sorted(victim_seed(value) for value in scores["victim_id"].unique())
    if seeds != list(config["formal_seeds"]):
        raise RuntimeError("formal seed coverage mismatch")
    return scores, manifest


POOL_FIELDS = ["victim_id", "fold", "pool_role", "sample_id"]


def prepare_pools(source: Path, output: Path, key_path: Path, ack: str | None) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frozen = verify_freeze()
    if ack != config["formal_ack"]:
        raise RuntimeError("formal acknowledgement token missing or incorrect")
    key = load_key(key_path, frozen)
    scores, source_manifest = validate_source(source, config, frozen)
    truth = pd.read_csv(source / "private_truth.csv")
    if set(config["formal_input_contract"]["required_truth_columns"]) != set(truth.columns):
        raise RuntimeError("source truth schema mismatch")
    base = scores.loc[(scores["role"] == "train") & (scores["detector"] == config["detectors"][0]), ["victim_id", "fold", "sample_id", "role"]]
    joined = base.merge(truth, on=["victim_id", "sample_id", "role"], how="left", validate="many_to_one")
    if joined["is_poison"].isna().any():
        raise RuntimeError("pool planner truth join is incomplete")
    poison_size = int(config["redteam"]["private_attacker_poison_pool_rows_per_fold"])

    def rows() -> Iterable[dict[str, Any]]:
        for (victim_id, fold), group in joined.groupby(["victim_id", "fold"], sort=True):
            clean = np.sort(group.loc[group["is_poison"] == 0, "sample_id"].to_numpy(np.int64))
            poison = np.sort(group.loc[group["is_poison"] == 1, "sample_id"].to_numpy(np.int64))
            if len(clean) < 2 * max(map(int, config["certificate"]["trusted_member_budgets"])) or len(poison) < poison_size:
                raise RuntimeError("formal fold lacks rows for frozen private pools")
            seed = victim_seed(str(victim_id))
            chosen_poison = np.sort(keyed_rng(key, "private-poison-pool", seed, int(fold)).choice(poison, poison_size, replace=False))
            for sample_id in clean:
                yield {"victim_id": victim_id, "fold": int(fold), "pool_role": "verified_clean", "sample_id": int(sample_id)}
            for sample_id in chosen_poison:
                yield {"victim_id": victim_id, "fold": int(fold), "pool_role": "attacker_poison", "sample_id": int(sample_id)}

    output.mkdir(parents=True, exist_ok=True)
    pool_path = output / "sealed_member_pools.csv"
    pool_rows = atomic_csv(pool_path, rows(), POOL_FIELDS)
    report = {
        "status": "MEMBER_POOLS_SEALED_AFTER_SCORE_HANDOFF", "formal": True,
        "truth_read_by_pool_planner": True, "truth_available_to_certificate_issuer": False,
        "key_commitment_sha256": frozen["defender_key_sha256"], "pool_rows": pool_rows,
        "sealed_member_pools_sha256": sha256(pool_path),
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "blind_scores_sha256": source_manifest["blind_scores_sha256"],
        "private_truth_sha256": source_manifest["private_truth_sha256"],
        "freeze_manifest_sha256": sha256(FREEZE), "created_unix": time.time(),
    }
    atomic_json(output / "pool_plan_manifest.json", report)
    return report


DECISION_FIELDS = [
    "part", "condition", "epsilon", "victim_id", "detector", "fold", "k", "draw", "interval",
    "n_poison_min", "n_poison_max", "calibration_ids_sha256", "selected_clean_ids_sha256",
    "selected_poison_ids_sha256", "selected_contaminated_count", "threshold", "conformal_rank",
    "candidate_count", "flagged_count", "recall_lo", "recall_hi", "clean_flag_rate_hi",
    "caught_poison_lo", "decision",
]


def _id_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.sort(np.asarray(values, dtype=np.int64)).tobytes()).hexdigest()


def _selected_positions(sample_ids: np.ndarray, selected_ids: np.ndarray) -> np.ndarray:
    lookup = {int(value): index for index, value in enumerate(sample_ids)}
    try:
        return np.asarray([lookup[int(value)] for value in selected_ids], dtype=np.int64)
    except KeyError as error:
        raise RuntimeError("sealed pool id is absent from score fold") from error


def _threshold(scores: np.ndarray, selected: np.ndarray, jitter: np.ndarray, alpha: float) -> tuple[np.ndarray, int, float, int]:
    k = len(selected); rank = conformal_rank(k, alpha)
    order = np.lexsort((jitter[selected], scores[selected])); threshold_position = int(selected[order[rank - 1]])
    candidate = np.ones(len(scores), dtype=bool); candidate[selected] = False
    flagged = candidate & ((scores > scores[threshold_position]) | ((scores == scores[threshold_position]) & (jitter > jitter[threshold_position])))
    return flagged, rank, float(scores[threshold_position]), int(candidate.sum())


def _draw_count(config: dict[str, Any], k: int) -> int:
    if int(k) == int(config["primary_endpoint"]["k"]):
        return int(config["certificate"]["primary_draws_per_cell"])
    return int(config["certificate"]["secondary_draws_per_cell"])


@lru_cache(maxsize=None)
def _certificate_cached(k: int, rank: int, flagged: int, candidates: int,
                        n_min: int, n_max: int, tau: float, delta: float,
                        alpha_cap: float, fpr_mode: str) -> dict[str, Any]:
    return certify_realized_counts(
        k, rank, flagged, candidates, tau=tau, n_poison_min=n_min,
        n_poison_max=n_max, delta=delta, alpha_cap=alpha_cap,
        fpr_mode=fpr_mode,
    )


def issue(source: Path, output: Path, key_path: Path, ack: str | None) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8")); frozen = verify_freeze()
    if ack != config["formal_ack"]:
        raise RuntimeError("formal acknowledgement token missing or incorrect")
    key = load_key(key_path, frozen)
    scores, source_manifest = validate_source(source, config, frozen)
    pool_manifest = json.loads((output / "pool_plan_manifest.json").read_text(encoding="utf-8"))
    pool_path = output / "sealed_member_pools.csv"
    if pool_manifest.get("status") != "MEMBER_POOLS_SEALED_AFTER_SCORE_HANDOFF" or sha256(pool_path) != pool_manifest.get("sealed_member_pools_sha256"):
        raise RuntimeError("sealed member-pool handoff is invalid")
    if pool_manifest.get("source_manifest_sha256") != sha256(source / "manifest.json"):
        raise RuntimeError("member pools refer to a different source")
    pools = pd.read_csv(pool_path)
    if list(pools.columns) != POOL_FIELDS or pools.duplicated().any():
        raise RuntimeError("sealed pool schema/uniqueness mismatch")
    cert = config["certificate"]
    intervals = {name: tuple(map(int, bounds)) for name, bounds in cert["declared_intervals_per_fold"].items()}
    detector_index = {name: index for index, name in enumerate(config["detectors"])}

    def rows() -> Iterable[dict[str, Any]]:
        train_scores = scores.loc[scores["role"] == "train"]
        for (victim_id, detector, fold), group in train_scores.groupby(["victim_id", "detector", "fold"], sort=True):
            group = group.sort_values("sample_id").reset_index(drop=True)
            values = group["score"].to_numpy(float); ids = group["sample_id"].to_numpy(np.int64)
            pool = pools.loc[(pools["victim_id"] == victim_id) & (pools["fold"] == int(fold))]
            clean_positions = _selected_positions(ids, pool.loc[pool["pool_role"] == "verified_clean", "sample_id"].to_numpy(np.int64))
            poison_positions = _selected_positions(ids, pool.loc[pool["pool_role"] == "attacker_poison", "sample_id"].to_numpy(np.int64))
            clean_position_set = set(map(int, clean_positions)); poison_position_set = set(map(int, poison_positions))
            seed = victim_seed(str(victim_id)); dindex = detector_index[str(detector)]; fold = int(fold)
            for k in map(int, cert["trusted_member_budgets"]):
                for draw in range(_draw_count(config, k)):
                    protected = keyed_rng(key, "protected-selection", seed, dindex, fold, k, draw).choice(clean_positions, k, replace=False)
                    protected_jitter = keyed_rng(key, "protected-ties", seed, dindex, fold, k, draw).random(len(values))
                    low_order = np.lexsort((ids[clean_positions], values[clean_positions]))
                    leaked_pool = clean_positions[low_order[: int(config["redteam"]["a1_leaked_pool_multiple"]) * k]]
                    leaked = keyed_rng(key, "leaked-selection", seed, dindex, fold, k, draw).choice(leaked_pool, k, replace=False)
                    leaked_jitter = keyed_rng(key, "leaked-ties", seed, dindex, fold, k, draw).random(len(values))
                    conditions: list[tuple[str, str, float, np.ndarray, np.ndarray]] = [
                        ("a1", "protected", 0.0, protected, protected_jitter),
                        ("a1", "leaked", 0.0, leaked, leaked_jitter),
                    ]
                    base_clean = keyed_rng(key, "a2-clean-selection", seed, dindex, fold, k, draw).choice(clean_positions, k, replace=False)
                    replace_order = keyed_rng(key, "a2-replacement-order", seed, dindex, fold, k, draw).permutation(k)
                    poison_order = keyed_rng(key, "a2-poison-order", seed, dindex, fold, k, draw).permutation(poison_positions)
                    a2_jitter = keyed_rng(key, "a2-ties", seed, dindex, fold, k, draw).random(len(values))
                    for epsilon in map(float, config["redteam"]["a2_contamination_fractions"]):
                        contaminated = int(np.floor(epsilon * k)); selected = base_clean.copy()
                        if contaminated:
                            selected[replace_order[:contaminated]] = poison_order[:contaminated]
                        conditions.append(("a2", "contamination", epsilon, selected, a2_jitter))
                    for part, condition, epsilon, selected, jitter in conditions:
                        flagged, rank, threshold, candidate_count = _threshold(values, selected, jitter, float(cert["alpha"]))
                        selected_clean = np.asarray([value for value in selected if int(value) in clean_position_set], dtype=np.int64)
                        selected_poison = np.asarray([value for value in selected if int(value) in poison_position_set], dtype=np.int64)
                        for interval, (n_min, n_max) in intervals.items():
                            result = _certificate_cached(
                                k, rank, int(flagged.sum()), candidate_count, n_min, n_max,
                                float(cert["tau"]), float(cert["delta_per_familywise_cell"]),
                                float(cert["alpha_cap"]), str(cert["fpr_mode"]),
                            )
                            yield {
                                "part": part, "condition": condition, "epsilon": epsilon,
                                "victim_id": victim_id, "detector": detector, "fold": fold, "k": k, "draw": draw,
                                "interval": interval, "n_poison_min": n_min, "n_poison_max": n_max,
                                "calibration_ids_sha256": _id_hash(ids[selected]),
                                "selected_clean_ids_sha256": _id_hash(ids[selected_clean]),
                                "selected_poison_ids_sha256": _id_hash(ids[selected_poison]),
                                "selected_contaminated_count": int(len(selected_poison)),
                                "threshold": repr(threshold), "conformal_rank": rank,
                                "candidate_count": candidate_count, "flagged_count": int(flagged.sum()),
                                "recall_lo": repr(float(result["recall_lo"])), "recall_hi": repr(float(result["recall_hi"])),
                                "clean_flag_rate_hi": repr(float(result["clean_flag_rate_hi"])),
                                "caught_poison_lo": repr(float(result["flagged_poisons_lo"])),
                                "decision": {1: "GO", -1: "NO_GO", 0: "ABSTAIN"}[int(result["decision"])],
                            }

    decision_path = output / "redteam_decisions_blind.csv"
    decision_rows = atomic_csv(decision_path, rows(), DECISION_FIELDS)
    report = {
        "status": "CERTIFICATE_DECISIONS_SEALED_BEFORE_TRUTH_JOIN", "formal": True,
        "private_truth_read_by_issuer": False, "decision_rows": decision_rows,
        "key_commitment_sha256": frozen["defender_key_sha256"],
        "redteam_decisions_blind_sha256": sha256(decision_path),
        "sealed_member_pools_sha256": sha256(pool_path),
        "blind_scores_sha256": source_manifest["blind_scores_sha256"],
        "freeze_manifest_sha256": sha256(FREEZE), "sealed_unix": time.time(),
    }
    report_path = output / "certificate_blind_report.json"; atomic_json(report_path, report)
    manifest = {
        "status": report["status"], "formal": True, "unblinded": False,
        "source": str(source.resolve()), "source_manifest_sha256": sha256(source / "manifest.json"),
        "pool_plan_manifest_sha256": sha256(output / "pool_plan_manifest.json"),
        "certificate_blind_report_sha256": sha256(report_path),
        "redteam_decisions_blind_sha256": report["redteam_decisions_blind_sha256"],
        "sealed_member_pools_sha256": report["sealed_member_pools_sha256"],
        "freeze_manifest_sha256": sha256(FREEZE), "runner_sha256": sha256(Path(__file__)),
        "blind_scores_sha256": source_manifest["blind_scores_sha256"], "private_truth_sha256": source_manifest["private_truth_sha256"],
    }
    atomic_json(output / "manifest.json", manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare-pools", "issue"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--ack", required=True)
    args = parser.parse_args()
    result = prepare_pools(args.source, args.output, args.key_file, args.ack) if args.stage == "prepare-pools" else issue(args.source, args.output, args.key_file, args.ack)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
