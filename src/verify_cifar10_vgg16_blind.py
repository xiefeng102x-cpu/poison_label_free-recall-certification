#!/usr/bin/env python3
"""Independent truth-join verifier for the CIFAR-10/VGG16-BN blind block."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
CERT_ROOT = ROOT / "auditability_certificate_v1"
sys.path.insert(0, str(CERT_ROOT / "analysis")); sys.path.insert(0, str(CERT_ROOT))
from cert.certificate import certify_realized_counts
from run_cifar10_vgg16_blind_certificate import METHODS, certificate_draw, draw_rng, sha256, victim_seed


CONFIG = ROOT / "configs/cifar10_vgg16_bn_blind_certificate_v1.json"
DEFAULT_RESULTS = ROOT / "results/cifar10_vgg16_bn_blind_certificate_v1"


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp"); temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("empty verifier output")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def upper_binomial(successes: int, trials: int, confidence: float = 0.95) -> float:
    if successes >= trials:
        return 1.0
    return float(stats.beta.ppf(confidence, successes + 1, trials - successes))


def run(results: Path) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8")); cert = config["certificate"]
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "CERTIFICATE_DECISIONS_SEALED_BEFORE_TRUTH_JOIN" or manifest.get("unblinded") is not False:
        raise RuntimeError("truth-free certificate decisions were not sealed")
    if manifest.get("frozen_hashes", {}).get("verifier_sha256") != sha256(Path(__file__)):
        raise RuntimeError("verifier differs from the frozen source")
    paths = {name: results / filename for name, filename in {
        "scores": "blind_scores.csv", "trusted": "trusted_member_pool.csv", "truth": "private_truth.csv",
        "decisions": "certificate_decisions_blind.csv", "deployments": "certificate_deployment_decisions_blind.csv",
    }.items()}
    expected = {
        "scores": manifest["blind_scores_sha256"], "trusted": manifest["trusted_member_pool_sha256"],
        "truth": manifest["private_truth_sha256"], "decisions": manifest["certificate_decisions_blind_sha256"],
        "deployments": manifest["certificate_deployment_decisions_blind_sha256"],
    }
    if any(sha256(paths[name]) != digest for name, digest in expected.items()):
        raise RuntimeError("verifier input hash mismatch")
    scores = pd.read_csv(paths["scores"]); trusted = pd.read_csv(paths["trusted"]); truth = pd.read_csv(paths["truth"]); decisions = pd.read_csv(paths["decisions"]); deployments = pd.read_csv(paths["deployments"])
    joined = scores.merge(truth, on=["victim_id", "sample_id", "role"], how="left", validate="many_to_one")
    if joined["is_poison"].isna().any():
        raise RuntimeError("incomplete score/truth join")
    evaluated: list[dict[str, Any]] = []
    grouped_scores = {
        key: frame.sort_values("sample_id").reset_index(drop=True)
        for key, frame in joined.loc[joined["role"] == "train"].groupby(["victim_id", "detector", "fold"], sort=True)
    }
    for row in decisions.itertuples(index=False):
        key = (row.victim_id, row.detector, int(row.fold)); group = grouped_scores[key]
        values = group["score"].to_numpy(dtype=float); sample_ids = group["sample_id"].to_numpy(dtype=np.int64)
        poison = group["is_poison"].to_numpy(dtype=int).astype(bool)
        pool = trusted.loc[(trusted["victim_id"] == row.victim_id) & (trusted["fold"] == row.fold), "sample_id"].to_numpy(dtype=np.int64)
        rng = draw_rng(int(cert["trusted_draw_seed_base"]), victim_seed(str(row.victim_id)), METHODS.index(str(row.detector)), int(row.fold), int(row.k), int(row.draw))
        selected, flagged, threshold, threshold_jitter, j, K = certificate_draw(values, sample_ids, pool, int(row.k), rng)
        selected_ids = np.sort(sample_ids[selected]); selected_hash = hashlib.sha256(selected_ids.tobytes()).hexdigest()
        if selected_hash != row.calibration_ids_sha256 or j != int(row.conformal_rank) or K != int(row.flagged_count) or not math.isclose(threshold, float(row.threshold), rel_tol=0.0, abs_tol=1e-12) or not math.isclose(threshold_jitter, float(row.threshold_jitter), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("independent draw replay mismatch")
        candidate = np.ones(len(values), dtype=bool); candidate[selected] = False
        actual_n = int(poison.sum()); actual_recall = float(np.sum(flagged & poison) / max(actual_n, 1))
        actual_clean_rate = float(np.sum(flagged & ~poison) / max(np.sum(candidate & ~poison), 1))
        valid = int(int(row.n_poison_min) <= actual_n <= int(row.n_poison_max))
        failed = actual_recall < float(cert["tau"]) or actual_clean_rate > float(cert["alpha_cap"])
        false_go = bool(row.decision == "GO" and failed)
        evaluated.append({
            **row._asdict(), "actual_n_poison": actual_n, "declared_interval_valid": valid,
            "actual_recall": actual_recall, "actual_clean_flag_rate": actual_clean_rate,
            "actual_n_clean_candidate": int(np.sum(candidate & ~poison)),
            "false_go": int(false_go), "valid_false_go": int(false_go and valid),
            "failed_operating_target": int(failed), "failure_recognized": int(row.decision != "GO"),
        })
    evaluated_path = results / "certificate_decisions_evaluated.csv"; atomic_csv(evaluated_path, evaluated)
    frame = pd.DataFrame(evaluated)
    deployment_evaluated: list[dict[str, Any]] = []
    deployment_keys = ["victim_id", "detector", "k", "draw", "interval"]
    for deployment in deployments.itertuples(index=False):
        mask = np.ones(len(frame), dtype=bool)
        for key in deployment_keys:
            mask &= frame[key].to_numpy() == getattr(deployment, key)
        group = frame.loc[mask]
        if len(group) != 2 or set(group["fold"].astype(int)) != {0, 1}:
            raise RuntimeError("deployment/per-fold decision join mismatch")
        expected = group.set_index("fold")["decision"].to_dict()
        if expected[0] != deployment.fold0_decision or expected[1] != deployment.fold1_decision:
            raise RuntimeError("sealed deployment decision does not match sealed folds")
        poison_total = int(group["actual_n_poison"].sum())
        caught = float(np.sum(group["actual_recall"] * group["actual_n_poison"]))
        clean_total = int(group["actual_n_clean_candidate"].sum())
        clean_flagged = float(np.sum(group["actual_clean_flag_rate"] * group["actual_n_clean_candidate"]))
        recall = caught / max(poison_total, 1); clean_rate = clean_flagged / max(clean_total, 1)
        valid = int(group["declared_interval_valid"].all())
        failed = recall < float(cert["tau"]) or clean_rate > float(cert["alpha_cap"])
        false_go = bool(deployment.decision == "GO" and failed)
        deployment_evaluated.append({
            **deployment._asdict(), "actual_n_poison": poison_total, "declared_interval_valid": valid,
            "actual_recall": recall, "actual_clean_flag_rate": clean_rate,
            "failed_operating_target": int(failed), "failure_recognized": int(deployment.decision != "GO"),
            "false_go": int(false_go), "valid_false_go": int(false_go and valid),
        })
    deployment_evaluated_path = results / "certificate_deployment_decisions_evaluated.csv"
    atomic_csv(deployment_evaluated_path, deployment_evaluated)
    deployment_frame = pd.DataFrame(deployment_evaluated)
    deployment_summary_rows: list[dict[str, Any]] = []
    for (detector, k, interval), group in deployment_frame.groupby(["detector", "k", "interval"], sort=True):
        valid = group.loc[group["declared_interval_valid"] == 1]
        failures = group.loc[group["failed_operating_target"] == 1]
        deployment_summary_rows.append({
            "detector": detector, "k": int(k), "interval": interval, "rows": len(group), "valid_rows": len(valid),
            "go_rate": float((group["decision"] == "GO").mean()), "nogo_rate": float((group["decision"] == "NO_GO").mean()),
            "abstain_rate": float((group["decision"] == "ABSTAIN").mean()),
            "valid_false_go_count": int(valid["valid_false_go"].sum()),
            "valid_false_go_rate": float(valid["valid_false_go"].mean()) if len(valid) else float("nan"),
            "failure_recognition_rate": float(failures["failure_recognized"].mean()) if len(failures) else float("nan"),
        })
    deployment_summary_path = results / "certificate_deployment_summary.csv"; atomic_csv(deployment_summary_path, deployment_summary_rows)
    cells: list[dict[str, Any]] = []
    for keys, group in frame.groupby(["victim_id", "detector", "fold", "k", "interval"], sort=True):
        valid = bool(group["declared_interval_valid"].all()); false_count = int(group["valid_false_go"].sum()) if valid else 0; trials = len(group)
        failures = group.loc[group["failed_operating_target"] == 1]
        cells.append({
            "victim_id": keys[0], "detector": keys[1], "fold": int(keys[2]), "k": int(keys[3]), "interval": keys[4],
            "R": trials, "declared_interval_valid": int(valid), "actual_n_poison": int(group["actual_n_poison"].iloc[0]),
            "mean_actual_recall": float(group["actual_recall"].mean()), "mean_actual_clean_flag_rate": float(group["actual_clean_flag_rate"].mean()),
            "go_rate": float((group["decision"] == "GO").mean()), "nogo_rate": float((group["decision"] == "NO_GO").mean()),
            "abstain_rate": float((group["decision"] == "ABSTAIN").mean()), "false_go_count": false_count,
            "false_go_rate": float(false_count / trials) if valid else float("nan"),
            "false_go_one_sided_95pct_upper": upper_binomial(false_count, trials) if valid else float("nan"),
            "failure_recognition_rate": float(failures["failure_recognized"].mean()) if len(failures) else float("nan"),
            "mean_recall_lo": float(group["recall_lo"].mean()), "mean_caught_poison_lo": float(group["caught_poison_lo"].mean()),
        })
    cell_path = results / "certificate_cell_summary.csv"; atomic_csv(cell_path, cells); cell_frame = pd.DataFrame(cells)
    conventional_rows: list[dict[str, Any]] = []
    conventional = joined.loc[joined["role"] == "train"].copy(); conventional["conventional_flag"] = conventional["conventional_flag"].astype(int)
    for (victim_id, detector), group in conventional.groupby(["victim_id", "detector"], sort=True):
        poison = group["is_poison"].to_numpy(dtype=int).astype(bool); flag = group["conventional_flag"].to_numpy(dtype=int).astype(bool)
        conventional_rows.append({
            "victim_id": victim_id, "detector": detector, "tp": int(np.sum(flag & poison)), "n_poison": int(poison.sum()),
            "fp_target_clean": int(np.sum(flag & ~poison)), "n_target_clean": int((~poison).sum()),
            "tpr": float(np.sum(flag & poison) / max(poison.sum(), 1)), "target_clean_fpr": float(np.sum(flag & ~poison) / max((~poison).sum(), 1)),
        })
    conventional_path = results / "conventional_detector_metrics.csv"; atomic_csv(conventional_path, conventional_rows)
    aggregate_rows: list[dict[str, Any]] = []
    for (detector, k, interval), group in cell_frame.groupby(["detector", "k", "interval"], sort=True):
        valid = group.loc[group["declared_interval_valid"] == 1]
        aggregate_rows.append({
            "detector": detector, "k": int(k), "interval": interval, "cells": len(group), "valid_cells": len(valid),
            "mean_actual_recall": float(group["mean_actual_recall"].mean()), "mean_actual_clean_flag_rate": float(group["mean_actual_clean_flag_rate"].mean()),
            "mean_go_rate": float(group["go_rate"].mean()), "mean_nogo_rate": float(group["nogo_rate"].mean()), "mean_abstain_rate": float(group["abstain_rate"].mean()),
            "pooled_false_go_count": int(valid["false_go_count"].sum()), "pooled_valid_draws": int(valid["R"].sum()),
            "pooled_false_go_rate": float(valid["false_go_count"].sum() / max(valid["R"].sum(), 1)),
            "min_failure_recognition_rate": float(group["failure_recognition_rate"].min()) if group["failure_recognition_rate"].notna().any() else float("nan"),
        })
    aggregate_path = results / "certificate_detector_summary.csv"; atomic_csv(aggregate_path, aggregate_rows)
    primary_all = deployment_frame.loc[deployment_frame["interval"] == "primary"]
    primary = primary_all.loc[primary_all["declared_interval_valid"] == 1]
    false_go_count = int(primary["valid_false_go"].sum())
    all_primary_valid = bool(primary_all["declared_interval_valid"].all())
    all_victims_eligible = bool(manifest.get("all_victims_eligible"))
    if false_go_count:
        status = "FAIL_INDEPENDENT_BLIND_VERIFICATION"
    elif not all_primary_valid or not all_victims_eligible:
        status = "INCONCLUSIVE_INDEPENDENT_BLIND_VERIFICATION"
    else:
        status = "PASS_INDEPENDENT_BLIND_VERIFICATION"
    report = {
        "status": status,
        "formal": True, "familywise_primary": True, "victims": int(scores["victim_id"].nunique()), "detectors": int(scores["detector"].nunique()),
        "primary_decision_unit": "victim-detector-budget-draw with both cross-fit folds jointly required for GO",
        "all_victims_eligible": all_victims_eligible, "all_primary_intervals_valid": all_primary_valid,
        "primary_valid_rows": len(primary), "primary_false_go_count": false_go_count,
        "primary_false_go_rate": float(primary["valid_false_go"].mean()) if len(primary) else float("nan"),
        "evaluated_sha256": sha256(evaluated_path), "deployment_evaluated_sha256": sha256(deployment_evaluated_path),
        "deployment_summary_sha256": sha256(deployment_summary_path), "cell_summary_sha256": sha256(cell_path),
        "detector_summary_sha256": sha256(aggregate_path), "conventional_metrics_sha256": sha256(conventional_path),
    }
    report_path = results / "independent_verification.json"; atomic_json(report_path, report)
    manifest.update({"status": report["status"], "unblinded": True, "independent_verification_sha256": sha256(report_path)}); atomic_json(results / "manifest.json", manifest)
    print(json.dumps(report, indent=2, sort_keys=True)); return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS); args = parser.parse_args(); run(args.results)


if __name__ == "__main__":
    main()
