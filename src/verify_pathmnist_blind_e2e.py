#!/usr/bin/env python3
"""Independent replay verifier for the blind PathMNIST certificate block."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CERT_ROOT = ROOT / "auditability_certificate_v1"
import sys
sys.path.insert(0, str(CERT_ROOT))
from cert.certificate import conformal_rank, certify_realized_counts

DEFAULT_RESULTS = ROOT / "results/pathmnist_blind_e2e_multidetector_certificate_v1"
CONFIG = ROOT / "configs/pathmnist_blind_e2e_multidetector_certificate_v1.json"
FREEZE = CERT_ROOT / "BLIND_E2E_FREEZE_MANIFEST_V1.json"
METHODS = (
    "cf_cmd_u", "target_centroid", "crossfit_spectral",
    "activation_clustering", "que_score",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def seed_from_victim(value: str) -> int:
    return int(value.rsplit("seed", 1)[1])


def rng_for(base: int, seed: int, method_index: int, fold: int, k: int, draw: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([base, seed, method_index, fold, k, draw]))


def replay_threshold(scores: np.ndarray, clean: np.ndarray, k: int, rng: np.random.Generator):
    selected = rng.choice(clean, size=k, replace=False)
    candidate = np.ones(len(scores), dtype=bool); candidate[selected] = False
    jitter = rng.random(len(scores))
    j = conformal_rank(k, 0.05)
    order = np.lexsort((jitter[selected], scores[selected]))
    threshold_position = selected[order[j - 1]]
    threshold = float(scores[threshold_position])
    tie = float(jitter[threshold_position])
    flags = candidate & ((scores > threshold) | ((scores == threshold) & (jitter > tie)))
    return selected, candidate, threshold, j, flags


def verify(results: Path) -> dict[str, Any]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    manifest_path = results / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("status") == "COMPLETE_BLIND_END_TO_END_CERTIFICATE", "incomplete manifest")
    require(manifest.get("unblinded") is True, "manifest unblinding state mismatch")
    require(manifest.get("freeze") == frozen, "embedded freeze differs from frozen file")
    require(tuple(manifest.get("seeds", [])) == tuple(cfg["formal_seeds"]), "seed mismatch")
    require(tuple(manifest.get("methods", [])) == METHODS, "method mismatch")

    paths = {
        "scores": results / "blind_scores.csv",
        "truth": results / "private_truth.csv",
        "decisions": results / "certificate_decisions_blind.csv",
        "evaluated": results / "certificate_decisions_evaluated.csv",
        "cells": results / "certificate_cell_summary.csv",
        "detector_summary": results / "certificate_detector_summary.csv",
        "conventional": results / "conventional_detector_metrics.csv",
        "report": results / "blind_e2e_certificate_report.json",
        "receipts": results / "seed_receipts.json",
    }
    for name, path in paths.items():
        require(path.is_file(), f"missing {name}")
    require(sha256(paths["scores"]) == manifest["blind_scores_sha256"], "score hash mismatch")
    require(sha256(paths["truth"]) == manifest["private_truth_sha256"], "truth hash mismatch")
    require(sha256(paths["decisions"]) == manifest["certificate_decisions_blind_sha256"], "decision hash mismatch")
    require(sha256(paths["report"]) == manifest["certificate_report_sha256"], "report hash mismatch")

    scores = pd.read_csv(paths["scores"])
    truth = pd.read_csv(paths["truth"])
    decisions = pd.read_csv(paths["decisions"])
    evaluated = pd.read_csv(paths["evaluated"])
    receipts = json.loads(paths["receipts"].read_text(encoding="utf-8"))
    require(len(receipts) == 5, "receipt count mismatch")
    require(scores["victim_id"].nunique() == 5 and scores["detector"].nunique() == 5, "score coverage mismatch")
    require(set(scores["fold"].unique()) == {0, 1}, "fold coverage mismatch")
    require(not scores.duplicated(["victim_id", "detector", "fold", "sample_id"]).any(), "duplicate scores")
    require(not truth.duplicated(["victim_id", "sample_id"]).any(), "duplicate truth")
    require(np.isfinite(scores["score"]).all(), "non-finite scores")
    require(not ((truth["role"] == "fresh") & (truth["is_poison"] == 1)).any(), "poisoned fresh row")

    expected_rows = 0
    expected_truth = 0
    for receipt in receipts:
        feature_path = Path(receipt["feature_file"])
        model_path = Path(receipt["model_file"])
        require(feature_path.is_file() and sha256(feature_path) == receipt["feature_sha256"], "feature hash mismatch")
        require(model_path.is_file() and sha256(model_path) == receipt["model_sha256"], "model hash mismatch")
        with np.load(feature_path, allow_pickle=False) as feature:
            target_count = len(feature["candidate_ids"])
            fresh_count = len(feature["fresh_ids"])
        require(target_count == receipt["target_candidate_count"], "target count mismatch")
        expected_rows += len(METHODS) * (target_count + 2 * fresh_count)
        expected_truth += target_count + fresh_count
    require(len(scores) == expected_rows, "score row count mismatch")
    require(len(truth) == expected_truth, "truth row count mismatch")

    ks = tuple(cfg["certificate"]["trusted_member_budgets"])
    intervals = cfg["certificate"]["declared_intervals_per_fold"]
    R = int(cfg["certificate"]["draws_per_cell"])
    expected_decisions = 5 * len(METHODS) * 2 * len(ks) * R * len(intervals)
    require(len(decisions) == expected_decisions == len(evaluated), "certificate row count mismatch")
    require(not decisions.duplicated(["victim_id", "detector", "fold", "k", "draw", "interval"]).any(), "duplicate decisions")

    joined = scores.merge(truth, on=["victim_id", "sample_id", "role"], validate="many_to_one")
    eval_key = ["victim_id", "detector", "fold", "k", "draw", "interval"]
    eval_index = evaluated.set_index(eval_key)
    decision_index = decisions.set_index(eval_key)
    checked = 0
    for (victim, detector, fold), group in joined[joined["role"] == "train"].groupby(["victim_id", "detector", "fold"], sort=True):
        group = group.reset_index(drop=True)
        values = group["score"].to_numpy(float)
        poison = group["is_poison"].to_numpy(int).astype(bool)
        clean = np.flatnonzero(~poison)
        seed = seed_from_victim(str(victim)); method_index = METHODS.index(str(detector))
        for k in ks:
            for draw in range(R):
                rng = rng_for(int(cfg["certificate"]["trusted_draw_seed_base"]), seed, method_index, int(fold), int(k), draw)
                selected, candidate, threshold, j, flags = replay_threshold(values, clean, int(k), rng)
                ids = np.sort(group.loc[selected, "sample_id"].to_numpy(np.int64))
                cal_hash = hashlib.sha256(ids.tobytes()).hexdigest()
                actual_np = int(poison.sum())
                actual_recall = float(np.sum(flags & poison) / actual_np)
                actual_clean = float(np.sum(flags & ~poison) / np.sum(candidate & ~poison))
                for interval, bounds in intervals.items():
                    key = (victim, detector, int(fold), int(k), draw, interval)
                    row = decision_index.loc[key]
                    erow = eval_index.loc[key]
                    require(row["calibration_ids_sha256"] == cal_hash, "calibration draw mismatch")
                    require(abs(float(row["threshold"]) - threshold) <= 1e-12, "threshold mismatch")
                    require(int(row["conformal_rank"]) == j and int(row["flagged_count"]) == int(flags.sum()), "tail count mismatch")
                    cert = certify_realized_counts(
                        int(k), j, int(flags.sum()), int(candidate.sum()), tau=0.8,
                        n_poison_min=int(bounds[0]), n_poison_max=int(bounds[1]),
                        delta=0.025, alpha_cap=0.1, fpr_mode="tight",
                    )
                    expected_label = {1: "GO", -1: "NO_GO", 0: "ABSTAIN"}[int(cert["decision"])]
                    require(row["decision"] == expected_label, "decision mismatch")
                    require(abs(float(row["recall_lo"]) - float(cert["recall_lo"])) <= 1e-12, "recall lower bound mismatch")
                    require(abs(float(erow["actual_recall"]) - actual_recall) <= 1e-12, "actual recall mismatch")
                    require(abs(float(erow["actual_clean_flag_rate"]) - actual_clean) <= 1e-12, "actual clean rate mismatch")
                    false_go = int(expected_label == "GO" and (actual_recall < 0.8 or actual_clean > 0.1))
                    require(int(erow["false_go"]) == false_go, "false-GO mismatch")
                    checked += 1

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    require(report["decision_rows"] == checked, "reported decision count mismatch")
    false_go = int(evaluated.loc[evaluated["declared_interval_valid"] == 1, "valid_false_go"].sum())
    require(report["primary_false_go_count"] == int(evaluated.loc[(evaluated["interval"] == "primary") & (evaluated["declared_interval_valid"] == 1), "valid_false_go"].sum()), "primary false-GO report mismatch")
    verification = {
        "status": "PASS",
        "checked_certificate_rows": checked,
        "score_rows": int(len(scores)),
        "truth_rows": int(len(truth)),
        "victims": 5,
        "detectors": 5,
        "all_models_and_features_hash_verified": True,
        "valid_false_go_count_all_intervals": false_go,
        "manifest_sha256": sha256(manifest_path),
        "report_sha256": sha256(paths["report"]),
    }
    atomic_json(results / "independent_verification.json", verification)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(); verify(args.results)


if __name__ == "__main__":
    main()
