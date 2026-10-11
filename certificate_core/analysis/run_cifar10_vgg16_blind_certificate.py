#!/usr/bin/env python3
"""Truth-free certificate decisions for the frozen CIFAR-10/VGG16-BN block."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CERT_ROOT = ROOT / "auditability_certificate_v1"
sys.path.insert(0, str(CERT_ROOT))
from cert.certificate import conformal_rank, certify_realized_counts


CONFIG = ROOT / "configs/cifar10_vgg16_bn_blind_certificate_v1.json"
FREEZE = CERT_ROOT / "CIFAR10_VGG16_BLIND_FREEZE_V1.json"
DEFAULT_RESULTS = ROOT / "results/cifar10_vgg16_bn_blind_certificate_v1"
METHODS = ("cf_cmd_u", "target_centroid", "crossfit_spectral", "activation_clustering", "que_score")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("empty certificate output")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def victim_seed(victim_id: str) -> int:
    return int(victim_id.rsplit("seed", 1)[1])


def draw_rng(base: int, seed: int, method_index: int, fold: int, k: int, draw: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([base, seed, method_index, fold, k, draw]))


def certificate_draw(
    scores: np.ndarray, sample_ids: np.ndarray, trusted_ids: np.ndarray, k: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, float, int, int]:
    trusted_positions = np.flatnonzero(np.isin(sample_ids, trusted_ids, assume_unique=True))
    if len(trusted_positions) != len(trusted_ids):
        raise RuntimeError("trusted-member pool is not a subset of its score fold")
    selected = rng.choice(trusted_positions, size=k, replace=False)
    candidate = np.ones(len(scores), dtype=bool); candidate[selected] = False
    jitter = rng.random(len(scores))
    j = conformal_rank(k, 0.05)
    order = np.lexsort((jitter[selected], scores[selected]))
    threshold_position = selected[order[j - 1]]
    threshold_score = float(scores[threshold_position]); threshold_jitter = float(jitter[threshold_position])
    flagged = candidate & ((scores > threshold_score) | ((scores == threshold_score) & (jitter > threshold_jitter)))
    return selected, flagged, threshold_score, threshold_jitter, j, int(flagged.sum())


def validate_inputs(results: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "SCORE_HANDOFF_COMPLETE" or manifest.get("unblinded") is not False:
        raise RuntimeError("sealed score handoff is not available")
    scores_path = results / "blind_scores.csv"; trusted_path = results / "trusted_member_pool.csv"
    if sha256(scores_path) != manifest["blind_scores_sha256"] or sha256(trusted_path) != manifest["trusted_member_pool_sha256"]:
        raise RuntimeError("blind input hash mismatch")
    if sha256(FREEZE) != manifest.get("freeze_manifest_sha256"):
        raise RuntimeError("freeze manifest hash mismatch")
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen.get("hashes", {}).get("certificate_runner_sha256") != sha256(Path(__file__)):
        raise RuntimeError("certificate runner differs from the frozen source")
    if frozen.get("hashes", {}).get("config_sha256") != sha256(CONFIG):
        raise RuntimeError("certificate configuration differs from the frozen source")
    scores = pd.read_csv(scores_path); trusted = pd.read_csv(trusted_path)
    if set(scores.columns) != {"victim_id", "detector", "fold", "sample_id", "score", "role", "conventional_flag"}:
        raise RuntimeError("blind score schema mismatch")
    if set(trusted.columns) != {"victim_id", "fold", "sample_id"}:
        raise RuntimeError("trusted-member schema mismatch")
    if scores.duplicated(["victim_id", "detector", "fold", "sample_id"]).any() or trusted.duplicated().any():
        raise RuntimeError("duplicate blind input key")
    if set(scores["detector"].unique()) != set(METHODS) or set(scores["fold"].unique()) != {0, 1}:
        raise RuntimeError("detector/fold coverage mismatch")
    if scores["victim_id"].nunique() != len(config["formal_seeds"]) or not np.isfinite(scores["score"]).all():
        raise RuntimeError("victim or numeric coverage mismatch")
    return scores, trusted, manifest


def run(results: Path) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    scores, trusted, manifest = validate_inputs(results, config)
    cert = config["certificate"]
    intervals = {name: tuple(int(value) for value in bounds) for name, bounds in cert["declared_intervals_per_fold"].items()}
    rows: list[dict[str, Any]] = []
    train_scores = scores.loc[scores["role"] == "train"].copy()
    for (victim_id, detector, fold), group in train_scores.groupby(["victim_id", "detector", "fold"], sort=True):
        group = group.sort_values("sample_id").reset_index(drop=True)
        values = group["score"].to_numpy(dtype=np.float64); sample_ids = group["sample_id"].to_numpy(dtype=np.int64)
        pool = trusted.loc[(trusted["victim_id"] == victim_id) & (trusted["fold"] == fold), "sample_id"].to_numpy(dtype=np.int64)
        if len(pool) != int(cert["trusted_pool_rows_per_fold"]):
            raise RuntimeError("trusted pool size mismatch")
        seed = victim_seed(str(victim_id)); method_index = METHODS.index(str(detector))
        for k in (int(value) for value in cert["trusted_member_budgets"]):
            for draw in range(int(cert["draws_per_cell"])):
                rng = draw_rng(int(cert["trusted_draw_seed_base"]), seed, method_index, int(fold), k, draw)
                selected, flagged, threshold, threshold_jitter, j, K = certificate_draw(values, sample_ids, pool, k, rng)
                selected_ids = np.sort(sample_ids[selected])
                candidate_count = len(values) - k
                for interval_name, (n_min, n_max) in intervals.items():
                    result = certify_realized_counts(
                        k, j, K, candidate_count, tau=float(cert["tau"]), n_poison_min=n_min, n_poison_max=n_max,
                        delta=float(cert["delta_per_detector_fold"]), alpha_cap=float(cert["alpha_cap"]), fpr_mode=str(cert["fpr_mode"]),
                    )
                    code = int(result["decision"])
                    rows.append({
                        "victim_id": victim_id, "detector": detector, "fold": int(fold), "k": k, "draw": draw,
                        "interval": interval_name, "n_poison_min": n_min, "n_poison_max": n_max,
                        "calibration_ids_sha256": hashlib.sha256(selected_ids.tobytes()).hexdigest(),
                        "threshold": threshold, "threshold_jitter": threshold_jitter, "conformal_rank": j,
                        "candidate_count": candidate_count, "flagged_count": K,
                        "recall_lo": float(result["recall_lo"]), "recall_hi": float(result["recall_hi"]),
                        "clean_flag_rate_hi": float(result["clean_flag_rate_hi"]),
                        "caught_poison_lo": float(result["flagged_poisons_lo"]),
                        "decision": {1: "GO", -1: "NO_GO", 0: "ABSTAIN"}[code],
                    })
    decision_path = results / "certificate_decisions_blind.csv"; atomic_csv(decision_path, rows)
    decision_frame = pd.DataFrame(rows)
    deployment_rows: list[dict[str, Any]] = []
    for keys, group in decision_frame.groupby(["victim_id", "detector", "k", "draw", "interval"], sort=True):
        if set(group["fold"].astype(int)) != {0, 1}:
            raise RuntimeError("deployment decision is missing a cross-fit fold")
        by_fold = group.set_index("fold")
        fold_decisions = [str(by_fold.loc[fold, "decision"]) for fold in (0, 1)]
        if fold_decisions == ["GO", "GO"]:
            decision = "GO"
        elif fold_decisions == ["NO_GO", "NO_GO"]:
            decision = "NO_GO"
        else:
            decision = "ABSTAIN"
        deployment_rows.append({
            "victim_id": keys[0], "detector": keys[1], "k": int(keys[2]), "draw": int(keys[3]),
            "interval": keys[4], "fold0_decision": fold_decisions[0], "fold1_decision": fold_decisions[1],
            "decision": decision, "min_fold_recall_lo": float(group["recall_lo"].min()),
            "max_fold_clean_flag_rate_hi": float(group["clean_flag_rate_hi"].max()),
        })
    deployment_path = results / "certificate_deployment_decisions_blind.csv"; atomic_csv(deployment_path, deployment_rows)
    report = {
        "status": "CERTIFICATE_DECISIONS_SEALED_BEFORE_TRUTH_JOIN", "formal": True,
        "private_truth_read_by_this_program": False, "familywise_primary": True,
        "delta_family": float(cert["delta_family"]), "delta_per_detector_fold": float(cert["delta_per_detector_fold"]),
        "victims": int(scores["victim_id"].nunique()), "detectors": int(scores["detector"].nunique()),
        "decision_rows": len(rows), "deployment_decision_rows": len(deployment_rows),
        "blind_scores_sha256": manifest["blind_scores_sha256"],
        "trusted_member_pool_sha256": manifest["trusted_member_pool_sha256"],
        "certificate_decisions_blind_sha256": sha256(decision_path),
        "certificate_deployment_decisions_blind_sha256": sha256(deployment_path),
    }
    report_path = results / "certificate_blind_report.json"; atomic_json(report_path, report)
    manifest.update({
        "status": report["status"], "unblinded": False,
        "certificate_decisions_blind_sha256": report["certificate_decisions_blind_sha256"],
        "certificate_deployment_decisions_blind_sha256": report["certificate_deployment_decisions_blind_sha256"],
        "certificate_blind_report_sha256": sha256(report_path),
    })
    atomic_json(results / "manifest.json", manifest)
    print(json.dumps(report, indent=2, sort_keys=True)); return report


def selftest() -> None:
    scores = np.linspace(-1.0, 1.0, 30); ids = np.arange(30); trusted = np.arange(20)
    selected, flagged, _, _, j, K = certificate_draw(scores, ids, trusted, 20, np.random.default_rng(3))
    if len(selected) != 20 or flagged.shape != (30,) or j != conformal_rank(20, 0.05) or K != int(flagged.sum()):
        raise RuntimeError("certificate analyzer selftest failed")
    print("selftest PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--selftest", action="store_true"); parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(); selftest() if args.selftest else run(args.results)


if __name__ == "__main__":
    main()
