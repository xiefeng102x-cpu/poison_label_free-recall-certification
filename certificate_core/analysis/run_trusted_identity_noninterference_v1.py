#!/usr/bin/env python3
"""Seal key-blind rankings for trusted-member identity inference V1."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "auditability_certificate_v1"
CONFIG = ROOT / "configs/trusted_identity_noninterference_v1.json"
PREREG = CERT / "PREREG_TRUSTED_IDENTITY_NONINTERFERENCE_V1.md"
FREEZE = CERT / "TRUSTED_IDENTITY_NONINTERFERENCE_FREEZE_V1.json"
VERIFIER = CERT / "analysis/verify_trusted_identity_noninterference_v1.py"
DEFAULT_OUTPUT = ROOT / "results/trusted_identity_noninterference_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def keyed_rng(key: bytes, domain: str, *parts: Any) -> np.random.Generator:
    message = "|".join([domain, *(str(value) for value in parts)]).encode("utf-8")
    digest = hmac.new(key, message, hashlib.sha256).digest()
    entropy = np.frombuffer(digest[:16], dtype=np.uint32).astype(np.uint64).tolist()
    return np.random.default_rng(np.random.SeedSequence(entropy))


def victim_seed(victim_id: str) -> int:
    suffix = str(victim_id).rsplit("seed", 1)
    if len(suffix) != 2 or not suffix[1].isdigit():
        raise RuntimeError(f"victim id lacks terminal seed: {victim_id}")
    return int(suffix[1])


def rank_metrics(order: np.ndarray, selected: np.ndarray, *, attack_budget: int) -> tuple[float, float, float]:
    """Return precision@budget, trusted-set recall, and AUROC."""
    order = np.asarray(order, dtype=np.int64)
    selected = np.asarray(selected, dtype=np.int64)
    n = len(order); k = len(selected); budget = min(int(attack_budget), n)
    if n <= k or k <= 0 or sorted(order.tolist()) != list(range(n)):
        raise ValueError("invalid ranking or selected set")
    selected_mask = np.zeros(n, dtype=bool); selected_mask[selected] = True
    hits = int(selected_mask[order[:budget]].sum())
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n, 0, -1, dtype=np.int64)
    rank_sum = int(ranks[selected].sum())
    auc = (rank_sum - k * (k + 1) / 2.0) / (k * (n - k))
    return hits / budget, hits / k, float(auc)


def verify_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    source = ROOT / config["source"]
    pool = ROOT / config["candidate_pool"]
    required = {
        "config_sha256": sha256(CONFIG),
        "prereg_sha256": sha256(PREREG),
        "runner_sha256": sha256(Path(__file__)),
        "verifier_sha256": sha256(VERIFIER),
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "blind_scores_sha256": sha256(source / "blind_scores.csv"),
        "candidate_pool_sha256": sha256(pool),
    }
    if frozen.get("status") != "FROZEN_BEFORE_FORMAL_IDENTITY_DRAW":
        raise RuntimeError("identity freeze is not active")
    if frozen.get("target_identities_opened_at_freeze") is not False:
        raise RuntimeError("identity targets were opened before freeze")
    for name, value in required.items():
        if frozen.get("hashes", {}).get(name) != value:
            raise RuntimeError(f"frozen hash mismatch: {name}")
    return config, frozen


def public_hash_scores(ids: np.ndarray, *parts: Any) -> np.ndarray:
    prefix = "|".join(str(value) for value in parts)
    values = []
    for sample_id in ids:
        digest = hashlib.sha256(f"{prefix}|{int(sample_id)}".encode("utf-8")).digest()
        values.append(int.from_bytes(digest[:8], "big") / 2**64)
    return np.asarray(values, dtype=np.float64)


def order_from_scores(values: np.ndarray, tie: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or len(values) != len(tie):
        raise RuntimeError("invalid attack score")
    return np.lexsort((tie, -values)).astype(np.int32)


def standardized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scale = values.std(axis=0, ddof=0)
    scale[scale < 1e-12] = 1.0
    return (values - values.mean(axis=0)) / scale


def ridge_predict(features: np.ndarray, target: np.ndarray, penalty: float) -> np.ndarray:
    x = standardized(features)
    design = np.column_stack([np.ones(len(x)), x])
    gram = design.T @ design
    regularizer = np.eye(gram.shape[0]) * float(penalty); regularizer[0, 0] = 0.0
    beta = np.linalg.solve(gram + regularizer, design.T @ np.asarray(target, dtype=np.float64))
    return design @ beta


def build_group(scores: pd.DataFrame, pools: pd.DataFrame, victim_id: str, fold: int,
                detectors: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_ids = np.sort(pools.loc[
        (pools["victim_id"] == victim_id) & (pools["fold"] == fold) &
        (pools["pool_role"] == "verified_clean"), "sample_id"
    ].to_numpy(np.int64))
    if len(candidate_ids) <= 2000 or len(np.unique(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("invalid verified-clean candidate universe")
    group = scores.loc[
        (scores["victim_id"] == victim_id) & (scores["fold"] == fold) &
        (scores["role"] == "train") & (scores["sample_id"].isin(candidate_ids))
    ]
    score_wide = group.pivot(index="sample_id", columns="detector", values="score").reindex(candidate_ids)
    flag_wide = group.pivot(index="sample_id", columns="detector", values="conventional_flag").reindex(candidate_ids)
    score_wide = score_wide.reindex(columns=detectors); flag_wide = flag_wide.reindex(columns=detectors)
    if score_wide.isna().any().any() or flag_wide.isna().any().any():
        raise RuntimeError("candidate score matrix is incomplete")
    return candidate_ids, score_wide.to_numpy(float), flag_wide.to_numpy(float)


def run(output: Path, ack: str | None) -> dict[str, Any]:
    config, frozen = verify_freeze()
    if ack != config["formal_ack"]:
        raise RuntimeError("formal acknowledgement token missing or incorrect")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("formal output is not empty")
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / config["source"]
    pool_path = ROOT / config["candidate_pool"]
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("status") != "SCORE_HANDOFF_COMPLETE":
        raise RuntimeError("source score handoff is not complete")
    scores = pd.read_csv(source / "blind_scores.csv")
    pools = pd.read_csv(pool_path)
    detectors = list(config["detectors"])
    if set(scores["detector"].unique()) != set(detectors):
        raise RuntimeError("detector coverage mismatch")
    arrays: dict[str, np.ndarray] = {}
    groups: list[dict[str, Any]] = []
    attack_methods = list(config["attack_methods"])
    shadow_count = int(config["shadow_key_count"])
    budgets = list(map(int, config["trusted_member_budgets"]))
    for victim_id in sorted(scores["victim_id"].unique()):
        seed = victim_seed(victim_id)
        for fold in map(int, config["folds"]):
            ids, raw_scores, flags = build_group(scores, pools, victim_id, fold, detectors)
            zscores = standardized(raw_scores)
            features = np.column_stack([zscores, flags])
            for dindex, detector in enumerate(detectors):
                prefix = f"g{len(groups):03d}"
                arrays[f"{prefix}_ids"] = ids.astype(np.int64)
                groups.append({
                    "prefix": prefix, "victim_id": victim_id, "seed": seed,
                    "detector": detector, "detector_index": dindex, "fold": fold,
                    "candidate_count": int(len(ids)),
                })
                for k in budgets:
                    tie = public_hash_scores(ids, "tie", seed, dindex, fold, k)
                    shadow_counts = np.zeros(len(ids), dtype=np.float64)
                    for shadow in range(shadow_count):
                        shadow_key = hashlib.sha256(f"public-shadow-key-{shadow}".encode("ascii")).digest()
                        chosen = keyed_rng(
                            shadow_key, "formal-trusted-identity", seed, dindex, fold, k, 0
                        ).choice(len(ids), k, replace=False)
                        shadow_counts[chosen] += 1.0
                    shadow_frequency = shadow_counts / shadow_count
                    predictions = {
                        "public_hash": public_hash_scores(ids, "public-attack", seed, dindex, fold, k),
                        "same_detector_low": -zscores[:, dindex],
                        "same_detector_high": zscores[:, dindex],
                        "multiscore_extreme": np.max(np.abs(zscores), axis=1),
                        "shadow_frequency": shadow_frequency,
                        "shadow_ridge": ridge_predict(features, shadow_frequency, float(config["ridge_penalty"])),
                    }
                    if list(predictions) != attack_methods:
                        raise RuntimeError("attack method order drift")
                    for method, prediction in predictions.items():
                        arrays[f"{prefix}_k{k}_{method}"] = order_from_scores(prediction, tie)
    ranking_path = output / "attacker_rankings.npz"
    temporary = ranking_path.with_name(ranking_path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, ranking_path)
    report = {
        "status": "ATTACK_RANKINGS_SEALED_BEFORE_KEY_OPEN",
        "formal": True,
        "defender_key_read_by_attacker": False,
        "target_membership_read_by_attacker": False,
        "source_scores_open_before_new_identity_draw": True,
        "groups": groups,
        "attack_methods": attack_methods,
        "trusted_member_budgets": budgets,
        "ranking_arrays": len(arrays),
        "attacker_rankings_sha256": sha256(ranking_path),
        "freeze_manifest_sha256": sha256(FREEZE),
        "defender_key_commitment_sha256": frozen["defender_key_sha256"],
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "blind_scores_sha256": sha256(source / "blind_scores.csv"),
        "candidate_pool_sha256": sha256(pool_path),
        "sealed_unix": time.time(),
    }
    atomic_json(output / "attack_manifest.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def selftest() -> None:
    key = hashlib.sha256(b"selftest").digest()
    selected = keyed_rng(key, "selection", 1).choice(100, 20, replace=False)
    order = np.arange(100, dtype=np.int64)
    precision, recall, auc = rank_metrics(order, selected, attack_budget=10)
    if not (0.0 <= precision <= 1.0 and 0.0 <= recall <= 1.0 and 0.0 <= auc <= 1.0):
        raise RuntimeError("rank metric selftest failed")
    print("trusted identity attacker selftest PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ack")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.formal:
        run(args.output, args.ack)
    else:
        parser.error("choose --selftest or --formal")


if __name__ == "__main__":
    main()
