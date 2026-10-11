#!/usr/bin/env python3
"""Independent truth-join and full draw replay for the formal A1/A2 block."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
CERT_ROOT = ROOT / "auditability_certificate_v1"
CONFIG = ROOT / "configs/cert_aware_redteam_multiblock_v2.json"
FREEZE = CERT_ROOT / "CERT_AWARE_REDTEAM_MULTIBLOCK_FREEZE_V2.json"
RUNNER = CERT_ROOT / "analysis/run_cert_aware_redteam_multiblock.py"

import sys
sys.path.insert(0, str(CERT_ROOT / "analysis")); sys.path.insert(0, str(CERT_ROOT))
import run_cert_aware_redteam_multiblock as base
from cert.certificate import certify_realized_counts


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def close(left: Any, right: Any, message: str) -> None:
    if not math.isclose(float(left), float(right), rel_tol=1e-11, abs_tol=1e-11):
        raise AssertionError(f"{message}: {left} != {right}")


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def one_sided_upper(successes: int, trials: int, confidence: float = 0.95) -> float:
    if trials <= 0:
        return float("nan")
    if successes >= trials:
        return 1.0
    return float(stats.beta.ppf(confidence, successes + 1, trials - successes))


def one_sided_lower(successes: int, trials: int, confidence: float = 0.95) -> float:
    if trials <= 0 or successes <= 0:
        return 0.0
    return float(stats.beta.ppf(1.0 - confidence, successes, trials - successes + 1))


def empirical_fosd_violation(clean: np.ndarray, poison: np.ndarray) -> float:
    points = np.sort(np.unique(np.concatenate([clean, poison])))
    clean_cdf = np.searchsorted(np.sort(clean), points, side="right") / len(clean)
    poison_cdf = np.searchsorted(np.sort(poison), points, side="right") / len(poison)
    return float(np.max(poison_cdf - clean_cdf))


def blank_stats() -> dict[str, float]:
    return defaultdict(float)


@lru_cache(maxsize=None)
def certificate_cached(k: int, rank: int, flagged: int, candidates: int,
                       n_min: int, n_max: int, tau: float, delta: float,
                       alpha_cap: float, fpr_mode: str) -> dict[str, Any]:
    return certify_realized_counts(
        k, rank, flagged, candidates, tau=tau, n_poison_min=n_min,
        n_poison_max=n_max, delta=delta, alpha_cap=alpha_cap,
        fpr_mode=fpr_mode,
    )


def update_stats(bucket: dict[str, float], *, decision: str, valid: bool, failed: bool,
                 false_candidate: bool, false_system: bool, dominance: bool,
                 recall: float, system_recall: float, clean_rate: float) -> None:
    bucket["rows"] += 1
    bucket[f"decision_{decision}"] += 1
    bucket["valid"] += int(valid)
    if valid:
        bucket["false_candidate"] += int(false_candidate)
        bucket["false_system"] += int(false_system)
        bucket["failed"] += int(failed)
        bucket["failure_recognized"] += int(failed and decision != "GO")
        bucket["threshold_dominance"] += int(dominance)
        bucket["sum_recall"] += recall
        bucket["sum_system_recall"] += system_recall
        bucket["sum_clean_rate"] += clean_rate


def verify(results: Path, key_path: Path) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8")); frozen = base.verify_freeze()
    key = base.load_key(key_path, frozen)
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("status") == "CERTIFICATE_DECISIONS_SEALED_BEFORE_TRUTH_JOIN", "blind decisions were not sealed")
    require(manifest.get("unblinded") is False, "result was already unblinded")
    require(manifest.get("runner_sha256") == base.sha256(RUNNER), "runner hash mismatch")
    require(manifest.get("freeze_manifest_sha256") == base.sha256(FREEZE), "freeze hash mismatch")
    source = Path(manifest["source"])
    scores, source_manifest = base.validate_source(source, config, frozen)
    require(base.sha256(source / "manifest.json") == manifest["source_manifest_sha256"], "source manifest drift")
    require(source_manifest.get("all_victims_eligible") is True, "not all formal victims passed eligibility")
    pool_path = results / "sealed_member_pools.csv"
    pools = pd.read_csv(pool_path)
    require(base.sha256(pool_path) == manifest["sealed_member_pools_sha256"], "sealed pool drift")
    truth = pd.read_csv(source / "private_truth.csv")
    require(base.sha256(source / "private_truth.csv") == manifest["private_truth_sha256"], "private truth drift")
    joined = scores.merge(truth, on=["victim_id", "sample_id", "role"], how="left", validate="many_to_one")
    require(not joined["is_poison"].isna().any(), "score/truth join incomplete")
    decision_path = results / "redteam_decisions_blind.csv"
    require(base.sha256(decision_path) == manifest["redteam_decisions_blind_sha256"], "blind decision drift")
    cert = config["certificate"]
    intervals = {name: tuple(map(int, bounds)) for name, bounds in cert["declared_intervals_per_fold"].items()}
    detector_index = {name: index for index, name in enumerate(config["detectors"])}
    evaluated_fields = base.DECISION_FIELDS + [
        "actual_n_poison_candidate", "actual_n_poison_total", "declared_interval_valid",
        "actual_candidate_recall", "actual_system_recall", "actual_clean_flag_rate",
        "failed_candidate_target", "failed_system_target", "failure_recognized",
        "poison_tail_rate_at_threshold", "clean_tail_rate_at_threshold",
        "dominance_holds_at_threshold", "empirical_fosd_violation",
        "empirical_fosd_exact", "false_go_candidate", "valid_false_go_candidate",
        "false_go_system", "valid_false_go_system",
    ]
    evaluated_path = results / "redteam_decisions_evaluated.csv"
    temporary = evaluated_path.with_name(evaluated_path.name + ".tmp")
    cell_stats: dict[tuple[Any, ...], dict[str, float]] = defaultdict(blank_stats)
    primary_stats: dict[tuple[Any, ...], dict[str, float]] = defaultdict(blank_stats)
    primary_victim_stats: dict[tuple[Any, ...], dict[str, float]] = defaultdict(blank_stats)
    monotone = defaultdict(lambda: {"paths": 0, "threshold_nondecreasing": 0, "go_nonincreasing": 0})
    decision_count = 0

    with decision_path.open(newline="", encoding="utf-8") as decision_handle, temporary.open("w", newline="", encoding="utf-8") as out_handle:
        reader = csv.DictReader(decision_handle)
        require(reader.fieldnames == base.DECISION_FIELDS, "blind decision schema mismatch")
        writer = csv.DictWriter(out_handle, fieldnames=evaluated_fields); writer.writeheader()

        def next_row() -> dict[str, str]:
            nonlocal decision_count
            try:
                row = next(reader)
            except StopIteration as error:
                raise AssertionError("blind decision file ended early") from error
            decision_count += 1
            return row

        train = joined.loc[joined["role"] == "train"]
        for (victim_id, detector, fold), group in train.groupby(["victim_id", "detector", "fold"], sort=True):
            group = group.sort_values("sample_id").reset_index(drop=True)
            score_values = group["score"].to_numpy(float); ids = group["sample_id"].to_numpy(np.int64)
            poison_mask = group["is_poison"].to_numpy(int).astype(bool)
            clean_all = np.flatnonzero(~poison_mask); poison_all = np.flatnonzero(poison_mask)
            fosd_violation = empirical_fosd_violation(score_values[clean_all], score_values[poison_all])
            pool = pools.loc[(pools["victim_id"] == victim_id) & (pools["fold"] == int(fold))]
            clean_positions = base._selected_positions(ids, pool.loc[pool["pool_role"] == "verified_clean", "sample_id"].to_numpy(np.int64))
            poison_positions = base._selected_positions(ids, pool.loc[pool["pool_role"] == "attacker_poison", "sample_id"].to_numpy(np.int64))
            clean_set = set(map(int, clean_positions)); poison_set = set(map(int, poison_positions))
            seed = base.victim_seed(str(victim_id)); dindex = detector_index[str(detector)]; fold = int(fold)
            for k in map(int, cert["trusted_member_budgets"]):
                for draw in range(base._draw_count(config, k)):
                    protected = base.keyed_rng(key, "protected-selection", seed, dindex, fold, k, draw).choice(clean_positions, k, replace=False)
                    protected_jitter = base.keyed_rng(key, "protected-ties", seed, dindex, fold, k, draw).random(len(score_values))
                    low_order = np.lexsort((ids[clean_positions], score_values[clean_positions]))
                    leaked_pool = clean_positions[low_order[: int(config["redteam"]["a1_leaked_pool_multiple"]) * k]]
                    leaked = base.keyed_rng(key, "leaked-selection", seed, dindex, fold, k, draw).choice(leaked_pool, k, replace=False)
                    leaked_jitter = base.keyed_rng(key, "leaked-ties", seed, dindex, fold, k, draw).random(len(score_values))
                    conditions: list[tuple[str, str, float, np.ndarray, np.ndarray]] = [
                        ("a1", "protected", 0.0, protected, protected_jitter),
                        ("a1", "leaked", 0.0, leaked, leaked_jitter),
                    ]
                    base_clean = base.keyed_rng(key, "a2-clean-selection", seed, dindex, fold, k, draw).choice(clean_positions, k, replace=False)
                    replace_order = base.keyed_rng(key, "a2-replacement-order", seed, dindex, fold, k, draw).permutation(k)
                    poison_order = base.keyed_rng(key, "a2-poison-order", seed, dindex, fold, k, draw).permutation(poison_positions)
                    a2_jitter = base.keyed_rng(key, "a2-ties", seed, dindex, fold, k, draw).random(len(score_values))
                    for epsilon in map(float, config["redteam"]["a2_contamination_fractions"]):
                        contaminated = int(np.floor(epsilon * k)); selected = base_clean.copy()
                        if contaminated:
                            selected[replace_order[:contaminated]] = poison_order[:contaminated]
                        conditions.append(("a2", "contamination", epsilon, selected, a2_jitter))
                    path_thresholds: list[float] = []; path_go: list[int] = []
                    for part, condition, epsilon, selected, jitter in conditions:
                        flagged, rank, threshold, candidate_count = base._threshold(score_values, selected, jitter, float(cert["alpha"]))
                        selected_clean = np.asarray([value for value in selected if int(value) in clean_set], dtype=np.int64)
                        selected_poison = np.asarray([value for value in selected if int(value) in poison_set], dtype=np.int64)
                        candidate = np.ones(len(score_values), dtype=bool); candidate[selected] = False
                        candidate_poison = int(np.sum(candidate & poison_mask)); total_poison = int(poison_mask.sum())
                        caught = int(np.sum(flagged & poison_mask)); candidate_recall = caught / max(candidate_poison, 1)
                        system_recall = caught / max(total_poison, 1)
                        clean_rate = int(np.sum(flagged & ~poison_mask)) / max(int(np.sum(candidate & ~poison_mask)), 1)
                        poison_tail = float(np.mean(score_values[poison_all] > threshold)); clean_tail = float(np.mean(score_values[clean_all] > threshold))
                        dominance = poison_tail >= clean_tail
                        primary_decision = None
                        for interval, (n_min, n_max) in intervals.items():
                            observed = next_row()
                            expected_identity = {
                                "part": part, "condition": condition, "epsilon": float(epsilon),
                                "victim_id": victim_id, "detector": detector, "fold": fold,
                                "k": k, "draw": draw, "interval": interval,
                            }
                            for name, expected in expected_identity.items():
                                if name == "epsilon": close(observed[name], expected, f"identity mismatch {name}")
                                else: require(str(observed[name]) == str(expected), f"identity mismatch {name}")
                            require(observed["calibration_ids_sha256"] == base._id_hash(ids[selected]), "calibration hash mismatch")
                            require(observed["selected_clean_ids_sha256"] == base._id_hash(ids[selected_clean]), "clean member hash mismatch")
                            require(observed["selected_poison_ids_sha256"] == base._id_hash(ids[selected_poison]), "poison member hash mismatch")
                            require(int(observed["selected_contaminated_count"]) == len(selected_poison), "contamination count mismatch")
                            require(int(observed["conformal_rank"]) == rank and int(observed["candidate_count"]) == candidate_count, "rank/count mismatch")
                            require(int(observed["flagged_count"]) == int(flagged.sum()), "flag count mismatch")
                            close(observed["threshold"], threshold, "threshold mismatch")
                            result = certificate_cached(
                                k, rank, int(flagged.sum()), candidate_count, n_min, n_max,
                                float(cert["tau"]), float(cert["delta_per_familywise_cell"]),
                                float(cert["alpha_cap"]), str(cert["fpr_mode"]),
                            )
                            decision = {1: "GO", -1: "NO_GO", 0: "ABSTAIN"}[int(result["decision"])]
                            require(observed["decision"] == decision, "decision mismatch")
                            close(observed["recall_lo"], result["recall_lo"], "recall lower mismatch")
                            close(observed["recall_hi"], result["recall_hi"], "recall upper mismatch")
                            close(observed["clean_flag_rate_hi"], result["clean_flag_rate_hi"], "clean-rate bound mismatch")
                            valid = n_min <= candidate_poison <= n_max
                            failed_candidate = candidate_recall < float(cert["tau"]) or clean_rate > float(cert["alpha_cap"])
                            failed_system = system_recall < float(cert["tau"]) or clean_rate > float(cert["alpha_cap"])
                            false_candidate = decision == "GO" and failed_candidate
                            false_system = decision == "GO" and failed_system
                            evaluated = {**observed,
                                "actual_n_poison_candidate": candidate_poison, "actual_n_poison_total": total_poison,
                                "declared_interval_valid": int(valid), "actual_candidate_recall": repr(candidate_recall),
                                "actual_system_recall": repr(system_recall), "actual_clean_flag_rate": repr(clean_rate),
                                "failed_candidate_target": int(failed_candidate), "failed_system_target": int(failed_system),
                                "failure_recognized": int(failed_candidate and decision != "GO"),
                                "poison_tail_rate_at_threshold": repr(poison_tail), "clean_tail_rate_at_threshold": repr(clean_tail),
                                "dominance_holds_at_threshold": int(dominance), "empirical_fosd_violation": repr(fosd_violation),
                                "empirical_fosd_exact": int(fosd_violation <= 1e-12),
                                "false_go_candidate": int(false_candidate), "valid_false_go_candidate": int(false_candidate and valid),
                                "false_go_system": int(false_system), "valid_false_go_system": int(false_system and valid),
                            }
                            writer.writerow(evaluated)
                            cell_key = (part, condition, float(epsilon), detector, fold, k, interval)
                            update_stats(cell_stats[cell_key], decision=decision, valid=valid, failed=failed_candidate,
                                         false_candidate=false_candidate, false_system=false_system,
                                         dominance=dominance, recall=candidate_recall,
                                         system_recall=system_recall, clean_rate=clean_rate)
                            if part == "a2" and k == int(config["primary_endpoint"]["k"]) and interval == config["primary_endpoint"]["interval"]:
                                primary_key = (float(epsilon), detector, fold)
                                update_stats(primary_stats[primary_key], decision=decision, valid=valid, failed=failed_candidate,
                                             false_candidate=false_candidate, false_system=false_system,
                                             dominance=dominance, recall=candidate_recall,
                                             system_recall=system_recall, clean_rate=clean_rate)
                                victim_key = (victim_id, float(epsilon), detector, fold)
                                update_stats(primary_victim_stats[victim_key], decision=decision, valid=valid, failed=failed_candidate,
                                             false_candidate=false_candidate, false_system=false_system,
                                             dominance=dominance, recall=candidate_recall,
                                             system_recall=system_recall, clean_rate=clean_rate)
                                primary_decision = decision
                        if part == "a2" and k == int(config["primary_endpoint"]["k"]):
                            path_thresholds.append(threshold); path_go.append(int(primary_decision == "GO"))
                    if k == int(config["primary_endpoint"]["k"]):
                        path_key = (detector, fold)
                        monotone[path_key]["paths"] += 1
                        monotone[path_key]["threshold_nondecreasing"] += int(np.all(np.diff(path_thresholds) >= -1e-12))
                        monotone[path_key]["go_nonincreasing"] += int(np.all(np.diff(path_go) <= 0))
        try:
            extra = next(reader)
        except StopIteration:
            extra = None
        require(extra is None, "blind decision file contains extra rows")
    os.replace(temporary, evaluated_path)

    summary_fields = ["part", "condition", "epsilon", "detector", "fold", "k", "interval", "rows", "valid_rows", "go_rate", "nogo_rate", "abstain_rate", "valid_false_go_candidate_count", "valid_false_go_candidate_rate", "valid_false_go_system_count", "valid_false_go_system_rate", "failure_recognition_rate", "threshold_dominance_rate", "mean_candidate_recall", "mean_system_recall", "mean_clean_flag_rate"]
    summary_rows: list[dict[str, Any]] = []
    for key, bucket in sorted(cell_stats.items()):
        part, condition, epsilon, detector, fold, k, interval = key
        rows = int(bucket["rows"]); valid = int(bucket["valid"])
        summary_rows.append({
            "part": part, "condition": condition, "epsilon": epsilon, "detector": detector,
            "fold": fold, "k": k, "interval": interval, "rows": rows, "valid_rows": valid,
            "go_rate": bucket["decision_GO"] / rows, "nogo_rate": bucket["decision_NO_GO"] / rows,
            "abstain_rate": bucket["decision_ABSTAIN"] / rows,
            "valid_false_go_candidate_count": int(bucket["false_candidate"]),
            "valid_false_go_candidate_rate": bucket["false_candidate"] / max(valid, 1),
            "valid_false_go_system_count": int(bucket["false_system"]),
            "valid_false_go_system_rate": bucket["false_system"] / max(valid, 1),
            "failure_recognition_rate": bucket["failure_recognized"] / max(bucket["failed"], 1),
            "threshold_dominance_rate": bucket["threshold_dominance"] / max(valid, 1),
            "mean_candidate_recall": bucket["sum_recall"] / max(valid, 1),
            "mean_system_recall": bucket["sum_system_recall"] / max(valid, 1),
            "mean_clean_flag_rate": bucket["sum_clean_rate"] / max(valid, 1),
        })
    summary_path = results / "redteam_cell_summary.csv"
    base.atomic_csv(summary_path, summary_rows, summary_fields)

    primary_rows: list[dict[str, Any]] = []; statuses: list[str] = []
    for key, bucket in sorted(primary_stats.items()):
        epsilon, detector, fold = key; trials = int(bucket["valid"]); failures = int(bucket["false_candidate"])
        rate = failures / max(trials, 1)
        if failures == 0:
            lower = 0.0
            upper = 1.0 - 0.05 ** (1.0 / trials)
            method = "exact_zero_event_bound_for_balanced_mixture"
        else:
            radius = math.sqrt(math.log(20.0) / (2.0 * trials))
            lower = max(rate - radius, 0.0); upper = min(rate + radius, 1.0)
            method = "hoeffding_bound_for_independent_nonidentical_draws"
        status = "SUPPORTED" if failures == 0 and upper <= float(cert["delta_per_familywise_cell"]) else "CONTRADICTED" if lower > float(cert["delta_per_familywise_cell"]) else "INCONCLUSIVE"
        statuses.append(status)
        primary_rows.append({
            "epsilon": epsilon, "detector": detector, "fold": fold, "valid_draws": trials,
            "false_go_candidate_count": failures, "false_go_candidate_rate": rate,
            "false_go_one_sided_95pct_lower": lower, "false_go_one_sided_95pct_upper": upper,
            "inference_method": method, "delta_cell": float(cert["delta_per_familywise_cell"]), "evidence_status": status,
            "false_go_system_count": int(bucket["false_system"]),
            "go_rate": bucket["decision_GO"] / max(bucket["rows"], 1),
            "threshold_dominance_rate": bucket["threshold_dominance"] / max(trials, 1),
        })
    primary_fields = list(primary_rows[0])
    primary_path = results / "redteam_primary_cell_inference.csv"
    base.atomic_csv(primary_path, primary_rows, primary_fields)
    victim_rows: list[dict[str, Any]] = []
    for key, bucket in sorted(primary_victim_stats.items()):
        victim_id, epsilon, detector, fold = key; trials = int(bucket["valid"]); failures = int(bucket["false_candidate"])
        victim_rows.append({
            "victim_id": victim_id, "epsilon": epsilon, "detector": detector, "fold": fold,
            "valid_draws": trials, "false_go_candidate_count": failures,
            "false_go_candidate_rate": failures / max(trials, 1),
            "zero_event_95pct_upper_if_applicable": 1.0 - 0.05 ** (1.0 / trials) if failures == 0 else "",
            "go_rate": bucket["decision_GO"] / max(bucket["rows"], 1),
            "false_go_system_count": int(bucket["false_system"]),
        })
    victim_path = results / "redteam_primary_victim_cells.csv"
    base.atomic_csv(victim_path, victim_rows, list(victim_rows[0]))
    scientific = "SUPPORTED" if all(value == "SUPPORTED" for value in statuses) else "CONTRADICTED" if any(value == "CONTRADICTED" for value in statuses) else "INCONCLUSIVE"
    a1_primary = [row for row in summary_rows if row["part"] == "a1" and row["k"] == int(config["primary_endpoint"]["k"]) and row["interval"] == config["primary_endpoint"]["interval"]]
    a1_rates = {}
    for condition in ("protected", "leaked"):
        subset = [row for row in a1_primary if row["condition"] == condition]
        count = sum(row["valid_false_go_candidate_count"] for row in subset); trials = sum(row["valid_rows"] for row in subset)
        a1_rates[condition] = {"false_go_count": int(count), "valid_draws": int(trials), "false_go_rate": count / max(trials, 1)}
    monotone_report = {
        f"{detector}|fold{fold}": {**values,
            "threshold_nondecreasing_rate": values["threshold_nondecreasing"] / max(values["paths"], 1),
            "go_nonincreasing_rate": values["go_nonincreasing"] / max(values["paths"], 1)}
        for (detector, fold), values in sorted(monotone.items())
    }
    revealed_path = results / "revealed_defender_key.hex"
    revealed_path.write_text(key.hex() + "\n", encoding="ascii")
    report = {
        "status": "PASS_INDEPENDENT_FORMAL_REDTEAM_VERIFICATION",
        "scientific_primary_status": scientific,
        "formal": True, "victims": int(scores["victim_id"].nunique()), "detectors": int(scores["detector"].nunique()),
        "recomputed_decision_rows": decision_count, "primary_safety_cells": len(primary_rows),
        "primary_supported_cells": int(sum(value == "SUPPORTED" for value in statuses)),
        "primary_inconclusive_cells": int(sum(value == "INCONCLUSIVE" for value in statuses)),
        "primary_contradicted_cells": int(sum(value == "CONTRADICTED" for value in statuses)),
        "a1_mechanism_ablation": a1_rates,
        "a2_nested_path_diagnostics": monotone_report,
        "candidate_endpoint_note": "Primary certificate estimand excludes selected trusted members.",
        "system_endpoint_note": "System recall counts selected contaminated poisons as missed and is a deployment-boundary diagnostic.",
        "theory_scope_note": "Finite-population A2 is an empirical stress test; exact FOSD diagnostics do not turn it into the iid theorem.",
        "hashes": {
            "redteam_decisions_evaluated_sha256": base.sha256(evaluated_path),
            "redteam_cell_summary_sha256": base.sha256(summary_path),
            "redteam_primary_cell_inference_sha256": base.sha256(primary_path),
            "redteam_primary_victim_cells_sha256": base.sha256(victim_path),
            "revealed_defender_key_sha256": base.sha256(revealed_path),
        },
    }
    report_path = results / "independent_verification.json"; atomic_json(report_path, report)
    manifest.update({"status": report["status"], "unblinded": True, "independent_verification_sha256": base.sha256(report_path), "evaluated_sha256": report["hashes"]["redteam_decisions_evaluated_sha256"]})
    atomic_json(results / "manifest.json", manifest)
    print(json.dumps(report, indent=2, sort_keys=True)); return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args(); verify(args.results, args.key_file)


if __name__ == "__main__":
    main()
