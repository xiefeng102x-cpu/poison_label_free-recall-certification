#!/usr/bin/env python3
"""Independent integrity and gate checks for P6-C bridge artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

import run_p6c_qml_noise_bridge as bridge


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    config_path = Path(manifest["config_path"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sha256(config_path) != manifest["config_sha256"]:
        raise AssertionError("config hash differs from run manifest")
    for seed, entries in manifest["source_files"].items():
        for name, record in entries.items():
            path = bridge.PROJECT_ROOT / record["path"]
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise AssertionError(f"source hash mismatch: seed={seed}, kind={name}, path={path}")
            if name == "eligibility":
                eligibility = json.loads(path.read_text(encoding="utf-8"))
                source_metrics = eligibility["aggregated_results"][f"{float(config['poison_ratio']):.1f}"]
                ca = float(source_metrics["clean_accuracy"]["mean"]) / 100.0
                asr = float(source_metrics["attack_success_rate"]["mean"]) / 100.0
                if ca < config["gates"]["victim_ca_min"] or asr < config["gates"]["victim_asr_min"]:
                    raise AssertionError(f"frozen victim gate failed for seed {seed}: CA={ca}, ASR={asr}")

    scores = read_csv(output / config["outputs"]["scores"])
    metrics = read_csv(output / config["outputs"]["metrics"])
    for row in scores:
        values = (float(row["score"]), float(row["threshold"]))
        if not all(math.isfinite(v) for v in values):
            raise AssertionError("non-finite score or threshold")
    mode = manifest["mode"]
    if mode == "pilot_seed42_nominal":
        expected_conditions, expected_seeds = ["nominal_noise_512"], [42]
    elif mode == "diagnostic_2048":
        expected_conditions, expected_seeds = ["nominal_noise_2048"], [int(x) for x in config["seeds"]]
    else:
        expected_conditions, expected_seeds = list(bridge.CONDITIONS), [int(x) for x in config["seeds"]]
    if manifest["conditions"] != expected_conditions or manifest["seeds"] != expected_seeds:
        raise AssertionError("manifest run scope does not match frozen protocol")
    expected_metrics = len(expected_seeds) * len(expected_conditions) * 2
    if len(metrics) != expected_metrics:
        raise AssertionError(f"expected {expected_metrics} metric rows; found {len(metrics)}")
    # Every condition has 72 held-out rows per seed/control; exact validation
    # rows are also retained so the frozen FPR threshold can be independently replayed.
    expected_score_rows = expected_metrics * 72 + len(expected_seeds) * 2 * 72
    if len(scores) != expected_score_rows:
        raise AssertionError(f"expected {expected_score_rows} score rows; found {len(scores)}")
    keys = set()
    for row in metrics:
        key = (int(row["seed"]), row["condition"], row["victim_kind"])
        if key in keys:
            raise AssertionError(f"duplicate metric group: {key}")
        keys.add(key)
        group = [r for r in scores if (int(r["seed"]), r["condition"], r["victim_kind"]) == key and r["split"] == "test"]
        y = np.asarray([int(r["label"]) for r in group])
        s = np.asarray([float(r["score"]) for r in group])
        threshold = float(row["validation_threshold"])
        recomputed = {
            "auc": roc_auc_score(y, s),
            "average_precision": average_precision_score(y, s),
            "balanced_accuracy": balanced_accuracy_score(y, s >= threshold),
            "tpr": np.mean(s[y == 1] >= threshold),
            "fpr": np.mean(s[y == 0] >= threshold),
        }
        for name, value in recomputed.items():
            if not np.isclose(float(row[name]), float(value), atol=1e-12):
                raise AssertionError(f"metric replay mismatch {key} {name}: {row[name]} != {value}")
        if not np.isclose(float(row["threshold"]), threshold, atol=0.0):
            raise AssertionError("threshold column inconsistency")
    for seed in expected_seeds:
        for kind in ("backdoor", "clean"):
            validation = [r for r in scores if int(r["seed"]) == seed and r["victim_kind"] == kind and r["condition"] == "exact" and r["split"] == "validation"]
            if len(validation) != 72:
                raise AssertionError(f"expected 72 exact validation rows for seed={seed} control={kind}")
            vy = np.asarray([int(r["label"]) for r in validation])
            vs = np.asarray([float(r["score"]) for r in validation])
            replay = bridge.target_threshold(vy, vs, float(config["threshold_fpr_target"]))
            thresholds = {float(r["threshold"]) for r in validation}
            if len(thresholds) != 1 or not np.isclose(replay, next(iter(thresholds)), atol=0.0):
                raise AssertionError(f"exact validation threshold replay failed for seed={seed} control={kind}")
            for condition in expected_conditions:
                selected = [r for r in scores if int(r["seed"]) == seed and r["victim_kind"] == kind and r["condition"] == condition and r["split"] == "test"]
                if not all(np.isclose(float(r["threshold"]), replay, atol=0.0) for r in selected):
                    raise AssertionError(f"threshold changed across conditions for seed={seed} condition={condition} control={kind}")
    for seed in expected_seeds:
        for condition in expected_conditions:
            bd = sorted((r["base_sample_id"], int(r["label"])) for r in scores if int(r["seed"]) == seed and r["condition"] == condition and r["victim_kind"] == "backdoor")
            cl = sorted((r["base_sample_id"], int(r["label"])) for r in scores if int(r["seed"]) == seed and r["condition"] == condition and r["victim_kind"] == "clean")
            if bd != cl:
                raise AssertionError(f"control rows are not matched: seed={seed} condition={condition}")

    if mode == "pilot_seed42_nominal":
        if manifest["status"] != "pilot_complete" or manifest["elapsed_seconds"] <= 0:
            raise AssertionError("pilot runtime evidence incomplete")
        return {"status": "pilot_valid", "elapsed_seconds": manifest["elapsed_seconds"], "score_rows": len(scores)}
    if mode == "diagnostic_2048":
        primary = Path(manifest["primary_summary_path"])
        if not primary.is_file() or sha256(primary) != manifest["primary_summary_sha256"]:
            raise AssertionError("primary 512-shot summary hash mismatch")
        primary_summary = json.loads(primary.read_text(encoding="utf-8"))
        gate_values = [v for k, v in primary_summary.get("gate", {}).items() if k.startswith("nominal_")]
        if not gate_values or all(gate_values):
            raise AssertionError("2048 diagnostic was run without a failed primary gate")
        if manifest["status"] != "diagnostic_complete" or not manifest.get("diagnostic_only"):
            raise AssertionError("2048-shot diagnostic status is invalid")
        return {"status": "diagnostic_valid", "metric_rows": len(metrics), "score_rows": len(scores), "diagnostic_only": True}
    if manifest["status"] != "complete":
        raise AssertionError(f"full run not complete: {manifest['status']}")
    summary = json.loads((output / config["outputs"]["summary"]).read_text(encoding="utf-8"))
    nominal = summary["conditions"]["nominal_noise_512"]
    for condition, aggregate in summary["conditions"].items():
        condition_metrics = [r for r in metrics if r["condition"] == condition and r["victim_kind"] == "backdoor"]
        by_seed = {str(int(r["seed"])): r for r in condition_metrics}
        expected_auc = {seed: float(r["auc"]) for seed, r in by_seed.items()}
        expected_delta = {seed: float(r["delta_auc_vs_exact"]) for seed, r in by_seed.items()}
        if aggregate["per_seed_auc"] != expected_auc or aggregate["per_seed_delta_auc"] != expected_delta:
            raise AssertionError(f"summary per-seed metrics mismatch for {condition}")
        if not np.isclose(aggregate["mean_auc"], np.mean(list(expected_auc.values())), atol=1e-12):
            raise AssertionError(f"summary mean AUC mismatch for {condition}")
        if not np.isclose(aggregate["mean_delta_auc"], np.mean(list(expected_delta.values())), atol=1e-12):
            raise AssertionError(f"summary mean delta AUC mismatch for {condition}")
        pooled = [r for r in scores if r["condition"] == condition and r["victim_kind"] == "backdoor" and r["split"] == "test"]
        labels = np.asarray([int(r["label"]) for r in pooled])
        decisions = np.asarray([float(r["score"]) >= float(r["threshold"]) for r in pooled])
        tpr = float(np.mean(decisions[labels == 1]))
        fpr = float(np.mean(decisions[labels == 0]))
        if not np.isclose(aggregate["pooled_tpr"], tpr, atol=1e-12) or not np.isclose(aggregate["pooled_fpr"], fpr, atol=1e-12):
            raise AssertionError(f"summary pooled operating rates mismatch for {condition}")
        ci = aggregate["paired_delta_auc_seed_block_ci"]
        for field in ("estimate", "low", "high"):
            if not math.isfinite(float(ci[field])):
                raise AssertionError(f"non-finite paired CI statistic for {condition}")
        if ci["low"] > ci["high"] or not np.isclose(ci["estimate"], aggregate["mean_delta_auc"], atol=1e-12):
            raise AssertionError(f"invalid paired CI for {condition}")
    gate = summary["gate"]
    expected_gate = {
        "nominal_each_seed_auc_pass": all(v >= config["gates"]["nominal_per_seed_auc_min"] for v in nominal["per_seed_auc"].values()),
        "nominal_mean_delta_auc_pass": nominal["mean_delta_auc"] >= config["gates"]["nominal_mean_delta_auc_min"],
        "nominal_pooled_tpr_pass": nominal["pooled_tpr"] >= config["gates"]["nominal_pooled_tpr_min"],
        "nominal_pooled_fpr_pass": nominal["pooled_fpr"] <= config["gates"]["nominal_pooled_fpr_max"],
    }
    for name, value in expected_gate.items():
        if gate[name] != bool(value):
            raise AssertionError(f"gate mismatch for {name}")
    return {"status": "full_run_valid", "score_rows": len(scores), "metric_rows": len(metrics), "nominal_gate": gate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=bridge.DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify(args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
