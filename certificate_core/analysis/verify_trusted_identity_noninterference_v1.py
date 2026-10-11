#!/usr/bin/env python3
"""Open the new key and independently evaluate sealed identity rankings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

import run_trusted_identity_noninterference_v1 as attack


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/trusted_identity_noninterference_v1"


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def load_key(path: Path, commitment: str) -> bytes:
    key = path.read_bytes()
    if len(key) != 32 or hashlib.sha256(key).hexdigest() != commitment:
        raise RuntimeError("defender key does not match the frozen commitment")
    return key


def update(bucket: dict[str, float], precision: float, recall: float, auc: float,
           precision_lift: float) -> None:
    bucket["draws"] += 1
    bucket["precision"] += precision
    bucket["recall"] += recall
    bucket["auc"] += auc
    bucket["precision_lift"] += precision_lift


def mean_ci(values: np.ndarray, alpha: float) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=len(values) - 1))
    half = critical * float(values.std(ddof=1)) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def verify(output: Path, key_path: Path) -> dict[str, Any]:
    config, frozen = attack.verify_freeze()
    manifest_path = output / "attack_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ATTACK_RANKINGS_SEALED_BEFORE_KEY_OPEN":
        raise RuntimeError("attack rankings were not sealed")
    if manifest.get("defender_key_read_by_attacker") is not False:
        raise RuntimeError("attacker reports defender-key access")
    if manifest.get("freeze_manifest_sha256") != attack.sha256(attack.FREEZE):
        raise RuntimeError("attack manifest freeze mismatch")
    ranking_path = output / "attacker_rankings.npz"
    if attack.sha256(ranking_path) != manifest.get("attacker_rankings_sha256"):
        raise RuntimeError("sealed attacker ranking drift")
    key = load_key(key_path, frozen["defender_key_sha256"])
    rankings = np.load(ranking_path, allow_pickle=False)
    methods = list(config["attack_methods"])
    negative = str(config["negative_control"])
    all_methods = methods + [negative]
    budgets = list(map(int, config["trusted_member_budgets"]))
    attack_budget = int(config["attacker_corruption_budget"])
    draws = {int(k): int(v) for k, v in config["draws"].items()}
    per_draw_path = output / "identity_inference_per_draw.csv"
    temporary = per_draw_path.with_name(per_draw_path.name + ".tmp")
    fields = [
        "victim_id", "detector", "fold", "k", "draw", "method", "candidate_count",
        "attack_budget", "random_precision", "random_target_corruption",
        "precision_at_budget", "target_corruption", "precision_lift", "auroc",
    ]
    victim_stats: dict[tuple[Any, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    pooled_primary: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    total_rows = 0
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for group in manifest["groups"]:
            prefix = str(group["prefix"]); ids = rankings[f"{prefix}_ids"]
            n = len(ids); seed = int(group["seed"]); dindex = int(group["detector_index"])
            fold = int(group["fold"]); victim_id = str(group["victim_id"]); detector = str(group["detector"])
            if n != int(group["candidate_count"]) or len(np.unique(ids)) != n:
                raise RuntimeError("ranking candidate universe mismatch")
            for k in budgets:
                if not 0 < k < n:
                    raise RuntimeError("invalid trusted-member budget")
                orders = {}
                top_masks = {}
                score_ranks = {}
                for method in methods:
                    order = rankings[f"{prefix}_k{k}_{method}"].astype(np.int64)
                    if len(order) != n or len(np.unique(order)) != n or order.min() != 0 or order.max() != n - 1:
                        raise RuntimeError("invalid sealed ranking permutation")
                    orders[method] = order
                    top_mask = np.zeros(n, dtype=bool)
                    top_mask[order[:min(attack_budget, n)]] = True
                    top_masks[method] = top_mask
                    rank = np.empty(n, dtype=np.int32)
                    rank[order] = np.arange(n, 0, -1, dtype=np.int32)
                    score_ranks[method] = rank
                random_precision = k / n
                random_corruption = min(attack_budget, n) / n
                for draw in range(draws[k]):
                    selected = attack.keyed_rng(
                        key, "formal-trusted-identity", seed, dindex, fold, k, draw
                    ).choice(n, k, replace=False)
                    for method in methods:
                        budget = min(attack_budget, n)
                        hits = int(top_masks[method][selected].sum())
                        precision = hits / budget; recall = hits / k
                        rank_sum = int(score_ranks[method][selected].sum())
                        auc = (rank_sum - k * (k + 1) / 2.0) / (k * (n - k))
                        lift = precision / random_precision
                        row = {
                            "victim_id": victim_id, "detector": detector, "fold": fold,
                            "k": k, "draw": draw, "method": method, "candidate_count": n,
                            "attack_budget": min(attack_budget, n),
                            "random_precision": repr(random_precision),
                            "random_target_corruption": repr(random_corruption),
                            "precision_at_budget": repr(precision),
                            "target_corruption": repr(recall),
                            "precision_lift": repr(lift), "auroc": repr(auc),
                        }
                        writer.writerow(row); total_rows += 1
                        update(victim_stats[(victim_id, detector, fold, k, method)], precision, recall, auc, lift)
                        if k == int(config["primary_endpoint"]["k"]):
                            update(pooled_primary[method], precision, recall, auc, lift)
                    budget = min(attack_budget, n)
                    hits = min(budget, k)
                    oracle_precision = hits / budget; oracle_recall = hits / k
                    oracle_lift = oracle_precision / random_precision; oracle_auc = 1.0
                    writer.writerow({
                        "victim_id": victim_id, "detector": detector, "fold": fold,
                        "k": k, "draw": draw, "method": negative, "candidate_count": n,
                        "attack_budget": budget, "random_precision": repr(random_precision),
                        "random_target_corruption": repr(random_corruption),
                        "precision_at_budget": repr(oracle_precision),
                        "target_corruption": repr(oracle_recall),
                        "precision_lift": repr(oracle_lift), "auroc": repr(oracle_auc),
                    }); total_rows += 1
                    update(victim_stats[(victim_id, detector, fold, k, negative)], oracle_precision, oracle_recall, oracle_auc, oracle_lift)
                    if k == int(config["primary_endpoint"]["k"]):
                        update(pooled_primary[negative], oracle_precision, oracle_recall, oracle_auc, oracle_lift)
    os.replace(temporary, per_draw_path)

    victim_rows: list[dict[str, Any]] = []
    for group_key, bucket in sorted(victim_stats.items()):
        victim_id, detector, fold, k, method = group_key; count = int(bucket["draws"])
        victim_rows.append({
            "victim_id": victim_id, "detector": detector, "fold": fold, "k": k,
            "method": method, "draws": count,
            "mean_precision_at_budget": bucket["precision"] / count,
            "mean_target_corruption": bucket["recall"] / count,
            "mean_precision_lift": bucket["precision_lift"] / count,
            "mean_auroc": bucket["auc"] / count,
        })
    victim_path = output / "identity_inference_victim_summary.csv"
    atomic_csv(victim_path, victim_rows, list(victim_rows[0]))

    primary_k = int(config["primary_endpoint"]["k"])
    family_cells = int(config["primary_endpoint"]["familywise_cells"])
    alpha_cell = float(config["primary_endpoint"]["familywise_alpha"]) / family_cells
    lift_margin = float(config["primary_endpoint"]["precision_lift_equivalence_margin"])
    auc_margin = float(config["primary_endpoint"]["auroc_equivalence_margin"])
    primary_rows: list[dict[str, Any]] = []
    statuses: list[str] = []
    for method in methods:
        for detector in config["detectors"]:
            for fold in map(int, config["folds"]):
                subset = [row for row in victim_rows if row["k"] == primary_k and row["method"] == method and row["detector"] == detector and row["fold"] == fold]
                if len(subset) != len(config["formal_seeds"]):
                    raise RuntimeError("primary victim coverage mismatch")
                lifts = np.asarray([row["mean_precision_lift"] for row in subset], dtype=float)
                aucs = np.asarray([row["mean_auroc"] for row in subset], dtype=float)
                corruptions = np.asarray([row["mean_target_corruption"] for row in subset], dtype=float)
                lift_mean, lift_lo, lift_hi = mean_ci(lifts, alpha_cell)
                auc_mean, auc_lo, auc_hi = mean_ci(aucs, alpha_cell)
                supported = (
                    lift_lo >= 1.0 - lift_margin and lift_hi <= 1.0 + lift_margin and
                    auc_lo >= 0.5 - auc_margin and auc_hi <= 0.5 + auc_margin
                )
                contradicted = (
                    lift_hi < 1.0 - lift_margin or lift_lo > 1.0 + lift_margin or
                    auc_hi < 0.5 - auc_margin or auc_lo > 0.5 + auc_margin
                )
                status = "SUPPORTED" if supported else "CONTRADICTED" if contradicted else "INCONCLUSIVE"
                statuses.append(status)
                primary_rows.append({
                    "method": method, "detector": detector, "fold": fold,
                    "victims": len(subset), "draws_per_victim": draws[primary_k],
                    "mean_precision_lift": lift_mean, "precision_lift_ci_lo": lift_lo,
                    "precision_lift_ci_hi": lift_hi, "mean_auroc": auc_mean,
                    "auroc_ci_lo": auc_lo, "auroc_ci_hi": auc_hi,
                    "mean_target_corruption": float(corruptions.mean()),
                    "familywise_alpha_per_cell": alpha_cell,
                    "evidence_status": status,
                })
    if len(primary_rows) != family_cells:
        raise RuntimeError("familywise primary cell count mismatch")
    primary_path = output / "identity_inference_primary_cells.csv"
    atomic_csv(primary_path, primary_rows, list(primary_rows[0]))

    pooled_rows = []
    for method in all_methods:
        bucket = pooled_primary[method]; count = int(bucket["draws"])
        pooled_rows.append({
            "method": method, "draws": count,
            "mean_precision_at_budget": bucket["precision"] / count,
            "mean_target_corruption": bucket["recall"] / count,
            "mean_precision_lift": bucket["precision_lift"] / count,
            "mean_auroc": bucket["auc"] / count,
        })
    pooled_path = output / "identity_inference_primary_pooled.csv"
    atomic_csv(pooled_path, pooled_rows, list(pooled_rows[0]))
    max_corruption = max(row["mean_target_corruption"] for row in pooled_rows if row["method"] != negative)
    oracle = next(row for row in pooled_rows if row["method"] == negative)
    oracle_pass = math.isclose(oracle["mean_auroc"], 1.0, abs_tol=1e-15) and oracle["mean_precision_lift"] > 1.0 + lift_margin
    scientific = (
        "SUPPORTED" if all(status == "SUPPORTED" for status in statuses) and
        max_corruption <= float(config["primary_endpoint"]["maximum_mean_target_corruption"]) and oracle_pass
        else "CONTRADICTED" if any(status == "CONTRADICTED" for status in statuses)
        else "INCONCLUSIVE"
    )
    revealed_path = output / "revealed_identity_key.hex"
    revealed_path.write_text(key.hex() + "\n", encoding="ascii")
    report = {
        "status": "PASS_INDEPENDENT_TRUSTED_IDENTITY_VERIFICATION",
        "scientific_primary_status": scientific,
        "formal": True,
        "attack_methods": methods,
        "negative_control": negative,
        "recomputed_draw_method_rows": total_rows,
        "primary_safety_cells": len(primary_rows),
        "primary_supported_cells": sum(status == "SUPPORTED" for status in statuses),
        "primary_inconclusive_cells": sum(status == "INCONCLUSIVE" for status in statuses),
        "primary_contradicted_cells": sum(status == "CONTRADICTED" for status in statuses),
        "maximum_nonoracle_mean_target_corruption": max_corruption,
        "maximum_allowed_mean_target_corruption": float(config["primary_endpoint"]["maximum_mean_target_corruption"]),
        "oracle_negative_control_pass": oracle_pass,
        "interpretation": "The secret draw is computationally independent of all declared attacker observables; this does not cover key or side-channel disclosure.",
        "hashes": {
            "attacker_rankings_sha256": attack.sha256(ranking_path),
            "per_draw_sha256": attack.sha256(per_draw_path),
            "victim_summary_sha256": attack.sha256(victim_path),
            "primary_cells_sha256": attack.sha256(primary_path),
            "primary_pooled_sha256": attack.sha256(pooled_path),
            "revealed_key_sha256": attack.sha256(revealed_path),
        },
        "verified_unix": time.time(),
    }
    report_path = output / "independent_identity_verification.json"
    attack.atomic_json(report_path, report)
    final_manifest = dict(manifest)
    final_manifest.update({
        "status": report["status"], "unblinded": True,
        "scientific_primary_status": scientific,
        "independent_verification_sha256": attack.sha256(report_path),
    })
    attack.atomic_json(output / "manifest.json", final_manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()
    verify(args.results, args.key_file)


if __name__ == "__main__":
    main()
