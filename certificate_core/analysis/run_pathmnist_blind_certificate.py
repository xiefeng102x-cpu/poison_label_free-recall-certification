#!/usr/bin/env python3
"""Issue the frozen PathMNIST certificate, then evaluate it after decisions exist."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
CERT_ROOT = ROOT / "auditability_certificate_v1"
import sys
sys.path.insert(0, str(CERT_ROOT))
from cert.certificate import conformal_rank, certify_realized_counts


CONFIG = ROOT / "configs/pathmnist_blind_e2e_multidetector_certificate_v1.json"
FREEZE = CERT_ROOT / "BLIND_E2E_FREEZE_MANIFEST_V1.json"
DEFAULT_RESULTS = ROOT / "results/pathmnist_blind_e2e_multidetector_certificate_v1"
METHODS = (
    "cf_cmd_u",
    "target_centroid",
    "crossfit_spectral",
    "activation_clustering",
    "que_score",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty output: {path}")
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def upper_binomial(successes: int, trials: int, confidence: float = 0.95) -> float:
    if successes >= trials:
        return 1.0
    return float(stats.beta.ppf(confidence, successes + 1, trials - successes))


def victim_seed(victim_id: str) -> int:
    return int(victim_id.rsplit("seed", 1)[1])


def draw_rng(base: int, seed: int, method_index: int, fold: int, k: int, draw: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([base, seed, method_index, fold, k, draw]))


def threshold_and_flags(scores: np.ndarray, clean_positions: np.ndarray, k: int, rng: np.random.Generator) -> tuple[np.ndarray, float, int, int]:
    selected = rng.choice(clean_positions, size=k, replace=False)
    candidate_mask = np.ones(len(scores), dtype=bool)
    candidate_mask[selected] = False
    jitter = rng.random(len(scores))
    j = conformal_rank(k, 0.05)
    order = np.lexsort((jitter[selected], scores[selected]))
    threshold_position = selected[order[j - 1]]
    threshold_score = float(scores[threshold_position])
    threshold_jitter = float(jitter[threshold_position])
    flagged = candidate_mask & (
        (scores > threshold_score)
        | ((scores == threshold_score) & (jitter > threshold_jitter))
    )
    return selected, threshold_score, j, int(flagged.sum()), flagged


def validate_inputs(results: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest_path = results / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "SCORE_HANDOFF_COMPLETE" or manifest.get("unblinded") is not False:
        raise RuntimeError("score handoff is not in its sealed pre-certificate state")
    scores_path = results / "blind_scores.csv"
    truth_path = results / "private_truth.csv"
    if sha256(scores_path) != manifest["blind_scores_sha256"]:
        raise RuntimeError("blind score hash mismatch")
    if sha256(truth_path) != manifest["private_truth_sha256"]:
        raise RuntimeError("private truth hash mismatch")
    if sha256(FREEZE) != manifest["frozen_hashes"]["freeze_manifest_sha256"]:
        raise RuntimeError("freeze manifest hash mismatch")
    scores = pd.read_csv(scores_path)
    truth = pd.read_csv(truth_path)
    required_scores = {"victim_id", "detector", "fold", "sample_id", "score", "role", "conventional_flag"}
    if set(scores.columns) != required_scores:
        raise RuntimeError("blind score schema mismatch")
    if set(truth.columns) != {"victim_id", "sample_id", "role", "is_poison"}:
        raise RuntimeError("truth schema mismatch")
    if scores.duplicated(["victim_id", "detector", "fold", "sample_id"]).any():
        raise RuntimeError("duplicate score key")
    if truth.duplicated(["victim_id", "sample_id"]).any():
        raise RuntimeError("duplicate truth key")
    if set(scores["detector"].unique()) != set(METHODS):
        raise RuntimeError("detector coverage mismatch")
    if set(scores["fold"].unique()) != {0, 1}:
        raise RuntimeError("fold coverage mismatch")
    if not np.isfinite(scores["score"]).all():
        raise RuntimeError("non-finite scores")
    if scores["victim_id"].nunique() != len(config["formal_seeds"]):
        raise RuntimeError("victim coverage mismatch")
    return scores, truth, manifest


def run(results: Path) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    scores, truth, manifest = validate_inputs(results, config)
    cert_cfg = config["certificate"]
    intervals = {
        name: tuple(int(x) for x in bounds)
        for name, bounds in cert_cfg["declared_intervals_per_fold"].items()
    }
    ks = [int(x) for x in cert_cfg["trusted_member_budgets"]]
    R = int(cert_cfg["draws_per_cell"])
    decisions: list[dict[str, Any]] = []
    evaluated: list[dict[str, Any]] = []

    # Truth is joined here only to simulate independent verification of trusted
    # members.  Neither the score map nor any count interval is estimated from it.
    joined = scores.merge(truth, on=["victim_id", "sample_id", "role"], how="left", validate="many_to_one")
    if joined["is_poison"].isna().any():
        raise RuntimeError("score/truth join is incomplete")
    for (victim_id, detector, fold), group in joined.groupby(["victim_id", "detector", "fold"], sort=True):
        train = group.loc[group["role"] == "train"].copy().reset_index(drop=True)
        if train.empty:
            raise RuntimeError("empty train score group")
        values = train["score"].to_numpy(dtype=float)
        poison = train["is_poison"].to_numpy(dtype=int).astype(bool)
        clean_positions = np.flatnonzero(~poison)
        seed = victim_seed(str(victim_id))
        method_index = METHODS.index(str(detector))
        actual_n_poison = int(poison.sum())
        for k in ks:
            if len(clean_positions) <= k:
                raise RuntimeError(f"insufficient verified clean members for k={k}")
            for draw in range(R):
                rng = draw_rng(int(cert_cfg["trusted_draw_seed_base"]), seed, method_index, int(fold), k, draw)
                selected, threshold, j, K, flagged = threshold_and_flags(values, clean_positions, k, rng)
                candidate = np.ones(len(values), dtype=bool)
                candidate[selected] = False
                selected_ids = np.sort(train.loc[selected, "sample_id"].to_numpy(dtype=np.int64))
                actual_recall = float(np.sum(flagged & poison) / max(actual_n_poison, 1))
                actual_clean_rate = float(np.sum(flagged & ~poison) / max(np.sum(candidate & ~poison), 1))
                for interval_name, (n_min, n_max) in intervals.items():
                    certificate = certify_realized_counts(
                        k,
                        j,
                        K,
                        int(candidate.sum()),
                        tau=float(cert_cfg["tau"]),
                        n_poison_min=n_min,
                        n_poison_max=n_max,
                        delta=float(cert_cfg["delta_per_fold"]),
                        alpha_cap=float(cert_cfg["alpha_cap"]),
                        fpr_mode=str(cert_cfg["fpr_mode"]),
                    )
                    code = int(certificate["decision"])
                    base = {
                        "victim_id": victim_id,
                        "detector": detector,
                        "fold": int(fold),
                        "k": k,
                        "draw": draw,
                        "interval": interval_name,
                        "n_poison_min": n_min,
                        "n_poison_max": n_max,
                        "calibration_ids_sha256": hashlib.sha256(selected_ids.tobytes()).hexdigest(),
                        "threshold": threshold,
                        "conformal_rank": j,
                        "candidate_count": int(candidate.sum()),
                        "flagged_count": K,
                        "recall_lo": float(certificate["recall_lo"]),
                        "recall_hi": float(certificate["recall_hi"]),
                        "clean_flag_rate_hi": float(certificate["clean_flag_rate_hi"]),
                        "caught_poison_lo": float(certificate["flagged_poisons_lo"]),
                        "decision": {1: "GO", -1: "NO_GO", 0: "ABSTAIN"}[code],
                    }
                    decisions.append(base)
                    interval_valid = n_min <= actual_n_poison <= n_max
                    false_go = bool(
                        code == 1
                        and (actual_recall < float(cert_cfg["tau"]) or actual_clean_rate > float(cert_cfg["alpha_cap"]))
                    )
                    evaluated.append(
                        {
                            **base,
                            "actual_n_poison": actual_n_poison,
                            "declared_interval_valid": int(interval_valid),
                            "actual_recall": actual_recall,
                            "actual_clean_flag_rate": actual_clean_rate,
                            "false_go": int(false_go),
                            "valid_false_go": int(false_go and interval_valid),
                            "failed_operating_target": int(actual_recall < float(cert_cfg["tau"]) or actual_clean_rate > float(cert_cfg["alpha_cap"])),
                            "failure_recognized": int(code != 1),
                        }
                    )

    # Seal the truth-free decisions before emitting any evaluated result.
    decision_path = results / "certificate_decisions_blind.csv"
    atomic_csv(decision_path, decisions)
    evaluated_path = results / "certificate_decisions_evaluated.csv"
    atomic_csv(evaluated_path, evaluated)
    evaluated_frame = pd.DataFrame(evaluated)

    cells: list[dict[str, Any]] = []
    for keys, group in evaluated_frame.groupby(["victim_id", "detector", "fold", "k", "interval"], sort=True):
        valid = bool(group["declared_interval_valid"].all())
        false_count = int(group["valid_false_go"].sum()) if valid else 0
        trials = int(len(group))
        cells.append(
            {
                "victim_id": keys[0],
                "detector": keys[1],
                "fold": int(keys[2]),
                "k": int(keys[3]),
                "interval": keys[4],
                "R": trials,
                "declared_interval_valid": int(valid),
                "actual_n_poison": int(group["actual_n_poison"].iloc[0]),
                "mean_actual_recall": float(group["actual_recall"].mean()),
                "mean_actual_clean_flag_rate": float(group["actual_clean_flag_rate"].mean()),
                "go_rate": float((group["decision"] == "GO").mean()),
                "nogo_rate": float((group["decision"] == "NO_GO").mean()),
                "abstain_rate": float((group["decision"] == "ABSTAIN").mean()),
                "false_go_count": false_count,
                "false_go_rate": float(false_count / trials) if valid else float("nan"),
                "false_go_one_sided_95pct_upper": upper_binomial(false_count, trials) if valid else float("nan"),
                "failure_recognition_rate": float(group.loc[group["failed_operating_target"] == 1, "failure_recognized"].mean()) if group["failed_operating_target"].any() else float("nan"),
                "mean_recall_lo": float(group["recall_lo"].mean()),
                "mean_caught_poison_lo": float(group["caught_poison_lo"].mean()),
            }
        )
    cell_path = results / "certificate_cell_summary.csv"
    atomic_csv(cell_path, cells)

    conventional = joined.loc[joined["role"] == "train"].copy()
    conventional["conventional_flag"] = conventional["conventional_flag"].astype(int)
    conventional_rows: list[dict[str, Any]] = []
    for (victim_id, detector), group in conventional.groupby(["victim_id", "detector"], sort=True):
        poison = group["is_poison"].to_numpy(dtype=int).astype(bool)
        flag = group["conventional_flag"].to_numpy(dtype=int).astype(bool)
        conventional_rows.append(
            {
                "victim_id": victim_id,
                "detector": detector,
                "tp": int(np.sum(flag & poison)),
                "n_poison": int(poison.sum()),
                "fp_target_clean": int(np.sum(flag & ~poison)),
                "n_target_clean": int((~poison).sum()),
                "tpr": float(np.sum(flag & poison) / max(poison.sum(), 1)),
                "target_clean_fpr": float(np.sum(flag & ~poison) / max((~poison).sum(), 1)),
            }
        )
    conventional_path = results / "conventional_detector_metrics.csv"
    atomic_csv(conventional_path, conventional_rows)

    cell_frame = pd.DataFrame(cells)
    aggregate_rows: list[dict[str, Any]] = []
    for (detector, k, interval), group in cell_frame.groupby(["detector", "k", "interval"], sort=True):
        valid_group = group.loc[group["declared_interval_valid"] == 1]
        aggregate_rows.append(
            {
                "detector": detector,
                "k": int(k),
                "interval": interval,
                "cells": int(len(group)),
                "valid_cells": int(len(valid_group)),
                "mean_actual_recall": float(group["mean_actual_recall"].mean()),
                "mean_actual_clean_flag_rate": float(group["mean_actual_clean_flag_rate"].mean()),
                "mean_go_rate": float(group["go_rate"].mean()),
                "mean_nogo_rate": float(group["nogo_rate"].mean()),
                "mean_abstain_rate": float(group["abstain_rate"].mean()),
                "pooled_false_go_count": int(valid_group["false_go_count"].sum()),
                "pooled_valid_draws": int(valid_group["R"].sum()),
                "pooled_false_go_rate": float(valid_group["false_go_count"].sum() / max(valid_group["R"].sum(), 1)),
                "min_failure_recognition_rate": float(group["failure_recognition_rate"].min()) if group["failure_recognition_rate"].notna().any() else float("nan"),
            }
        )
    aggregate_path = results / "certificate_detector_summary.csv"
    atomic_csv(aggregate_path, aggregate_rows)

    report = {
        "status": "COMPLETE_BLIND_END_TO_END_CERTIFICATE",
        "freeze_manifest_sha256": sha256(FREEZE),
        "blind_scores_sha256": sha256(results / "blind_scores.csv"),
        "private_truth_sha256": sha256(results / "private_truth.csv"),
        "certificate_decisions_blind_sha256": sha256(decision_path),
        "certificate_decisions_evaluated_sha256": sha256(evaluated_path),
        "certificate_cell_summary_sha256": sha256(cell_path),
        "certificate_detector_summary_sha256": sha256(aggregate_path),
        "conventional_detector_metrics_sha256": sha256(conventional_path),
        "victims": int(scores["victim_id"].nunique()),
        "detectors": int(scores["detector"].nunique()),
        "decision_rows": len(decisions),
        "evaluation_rows": len(evaluated),
        "all_primary_intervals_valid": bool(cell_frame.loc[cell_frame["interval"] == "primary", "declared_interval_valid"].all()),
        "primary_false_go_count": int(cell_frame.loc[cell_frame["interval"] == "primary", "false_go_count"].sum()),
        "primary_valid_draws": int(cell_frame.loc[(cell_frame["interval"] == "primary") & (cell_frame["declared_interval_valid"] == 1), "R"].sum()),
        "scope": "one dataset, one strong architecture, one BadNets attack, five unseen seeds; no cross-domain universality claim",
    }
    report["primary_false_go_rate"] = report["primary_false_go_count"] / max(report["primary_valid_draws"], 1)
    report_path = results / "blind_e2e_certificate_report.json"
    atomic_json(report_path, report)
    manifest.update(
        {
            "status": "COMPLETE_BLIND_END_TO_END_CERTIFICATE",
            "unblinded": True,
            "certificate_report_sha256": sha256(report_path),
            "certificate_decisions_blind_sha256": report["certificate_decisions_blind_sha256"],
        }
    )
    atomic_json(results / "manifest.json", manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    run(args.results)


if __name__ == "__main__":
    main()
