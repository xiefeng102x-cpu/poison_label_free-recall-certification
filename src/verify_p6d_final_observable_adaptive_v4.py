#!/usr/bin/env python3
"""Independent replay verifier for the P6-D v4 ten-seed extension."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import beta

import verify_p6d_final_observable_adaptive_v3 as v3_verifier


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "p6d_final_observable_adaptive_protocol_v4_extension"
PROTOCOL_PATH = ROOT / "results/top_journal_program_v1/p6d_final_observable_adaptive_protocol_v4.json"
DEFAULT_OUTPUT = ROOT / "results/p6d_final_observable_adaptive_v4/confirmation"
CONFIRMATION_SEEDS = tuple(range(124, 134))
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_RNG_SEED = 20_261_006


def sha256_file(path: Path) -> str:
    return v3_verifier.sha256_file(path)


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_protocol() -> dict[str, Any]:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise AssertionError("P6-D v4 protocol identifier mismatch")
    if tuple(payload["seed_audit"]["confirmation_extension_seeds"]) != CONFIRMATION_SEEDS:
        raise AssertionError("P6-D v4 protocol seed block mismatch")
    audit = payload["source_audit"]
    runner = ROOT / audit["v4_runner"]
    if not runner.is_file() or sha256_file(runner) != audit["v4_runner_sha256"]:
        raise AssertionError("P6-D v4 runner source hash mismatch")
    if sha256_file(Path(__file__).resolve()) != audit["v4_verifier_sha256"]:
        raise AssertionError("P6-D v4 verifier source hash mismatch")
    return payload


def _clopper_pearson(successes: int, trials: int) -> tuple[float, float]:
    alpha = 0.05
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return low, high


def _sign_pvalue(successes: int, trials: int) -> float:
    return float(sum(math.comb(trials, k) for k in range(successes, trials + 1)) / (2**trials))


def replay_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    rng_seed: int = BOOTSTRAP_RNG_SEED,
) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    if sorted(int(row["seed"]) for row in rows) != list(CONFIRMATION_SEEDS) or len(rows) != 10:
        raise AssertionError("P6-D v4 per-seed block is incomplete")
    eligible = [row for row in rows if _boolean(row["eligible"])]
    joint = [
        row for row in rows
        if _boolean(row["eligible"])
        and float(row["tpr"]) < 0.90
        and float(row["fpr"]) <= 0.05
    ]
    viability_ci = _clopper_pearson(len(eligible), 10)
    joint_ci = _clopper_pearson(len(joint), 10)
    result: dict[str, Any] = {
        "all_seed_count": 10,
        "attack_viable_count": len(eligible),
        "attack_viability_rate": len(eligible) / 10,
        "attack_viability_ci_low": viability_ci[0],
        "attack_viability_ci_high": viability_ci[1],
        "joint_adaptive_evasion_count": len(joint),
        "joint_adaptive_evasion_rate": len(joint) / 10,
        "joint_adaptive_evasion_ci_low": joint_ci[0],
        "joint_adaptive_evasion_ci_high": joint_ci[1],
        "conditional_efficacy_reported": bool(eligible),
        "bootstrap_draws": int(draws),
        "bootstrap_rng_seed": int(rng_seed),
    }
    if not eligible:
        result.update({
            "decision": "INSUFFICIENT_ATTACK_VIABILITY",
            "sign_successes": 0,
            "sign_trials": 0,
            "sign_test_pvalue_one_sided": 1.0,
        })
        return result
    tprs = np.asarray([float(row["tpr"]) for row in eligible], dtype=np.float64)
    fprs = np.asarray([float(row["fpr"]) for row in eligible], dtype=np.float64)
    counts = np.asarray(
        [[int(row[name]) for name in ("tp", "fn", "fp", "tn")] for row in eligible],
        dtype=np.int64,
    )
    tp, fn, fp, tn = map(int, counts.sum(axis=0))
    rng = np.random.default_rng(rng_seed)
    choices = rng.integers(0, len(eligible), size=(draws, len(eligible)))
    tpr_low, tpr_high = np.quantile(tprs[choices].mean(axis=1), [0.025, 0.975])
    fpr_low, fpr_high = np.quantile(fprs[choices].mean(axis=1), [0.025, 0.975])
    sign_successes = int(np.sum(tprs < 0.90))
    sign_p = _sign_pvalue(sign_successes, len(eligible))
    if len(eligible) < 8:
        decision = "INSUFFICIENT_ATTACK_VIABILITY"
    elif sign_p <= 0.05 and float(tpr_high) < 0.90 and float(fpr_high) <= 0.05:
        decision = "ADAPTIVE_EVASION_SUPPORTED"
    elif float(tpr_low) >= 0.90 and float(fpr_high) <= 0.05:
        decision = "DETECTOR_ROBUSTNESS_SUPPORTED"
    else:
        decision = "INCONCLUSIVE_MIXED"
    result.update({
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "pooled_tpr": tp / (tp + fn),
        "pooled_fpr": fp / (fp + tn),
        "mean_seed_tpr": float(tprs.mean()),
        "mean_seed_fpr": float(fprs.mean()),
        "min_seed_tpr": float(tprs.min()),
        "max_seed_tpr": float(tprs.max()),
        "min_seed_fpr": float(fprs.min()),
        "max_seed_fpr": float(fprs.max()),
        "mean_seed_tpr_ci_low": float(tpr_low),
        "mean_seed_tpr_ci_high": float(tpr_high),
        "mean_seed_fpr_ci_low": float(fpr_low),
        "mean_seed_fpr_ci_high": float(fpr_high),
        "sign_successes": sign_successes,
        "sign_trials": len(eligible),
        "sign_test_pvalue_one_sided": sign_p,
        "decision": decision,
    })
    return result


def _compare_json(expected: Any, actual: Any, path: str = "root") -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            raise AssertionError(f"JSON key mismatch at {path}")
        for key in expected:
            _compare_json(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise AssertionError(f"JSON list length mismatch at {path}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare_json(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not np.isclose(float(expected), float(actual), atol=1e-12, rtol=0.0):
            raise AssertionError(f"JSON numeric mismatch at {path}: {expected} != {actual}")
        return
    if expected != actual:
        raise AssertionError(f"JSON mismatch at {path}: {expected!r} != {actual!r}")


def _validate_manifest_files(output: Path, manifest: Mapping[str, Any]) -> None:
    expected = {"manifest.json", *manifest.get("artifacts", {})}
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)
    if extra or missing:
        raise AssertionError(f"P6-D v4 manifest coverage mismatch: extra={extra}, missing={missing}")


def _validate_lock(output: Path, protocol: Mapping[str, Any]) -> None:
    path = output / "confirmation_lock_receipt.json"
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.read_text(encoding="utf-8").strip() != sha256_file(path):
        raise AssertionError("P6-D v4 lock sidecar mismatch")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("protocol_id") != PROTOCOL_ID:
        raise AssertionError("P6-D v4 lock protocol mismatch")
    if lock.get("protocol_sha256") != sha256_file(PROTOCOL_PATH):
        raise AssertionError("P6-D v4 lock protocol hash mismatch")
    if tuple(lock.get("confirmation_extension_seeds", [])) != CONFIRMATION_SEEDS:
        raise AssertionError("P6-D v4 lock seed mismatch")
    if lock.get("replacement_seeds") != []:
        raise AssertionError("P6-D v4 lock contains replacement seeds")
    if lock.get("decision_rule_sha256") != v3_verifier.stable_json_sha256(protocol["decision_rule"]):
        raise AssertionError("P6-D v4 decision-rule hash mismatch")
    for source, digest in lock.get("source_hashes", {}).items():
        source_path = Path(source)
        if not source_path.is_file() or sha256_file(source_path) != digest:
            raise AssertionError(f"P6-D v4 source-lock mismatch: {source}")


def _validate_receipts(output: Path) -> None:
    splits = json.loads((output / "split_receipts.json").read_text(encoding="utf-8"))
    checkpoints = json.loads((output / "checkpoint_receipts.json").read_text(encoding="utf-8"))
    expected = list(CONFIRMATION_SEEDS)
    if sorted(map(int, splits)) != expected or sorted(map(int, checkpoints)) != expected:
        raise AssertionError("P6-D v4 receipt seed block mismatch")
    for seed, receipt in splits.items():
        poison = receipt.get("trigger_receipt", {}).get("poison_indices")
        if not isinstance(poison, list) or len(poison) != 60 or not all(isinstance(value, int) for value in poison):
            raise AssertionError(f"P6-D v4 poison-index receipt mismatch: {seed}")
    for seed, kinds in checkpoints.items():
        if set(kinds) != {"clean_checkpoint", "attacked_checkpoint"}:
            raise AssertionError(f"P6-D v4 checkpoint kinds mismatch: {seed}")
        for kind, record in kinds.items():
            path = output / record["path"]
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise AssertionError(f"P6-D v4 checkpoint hash mismatch: {seed}/{kind}")


def verify_bundle(output: Path) -> dict[str, Any]:
    output = output.resolve()
    protocol = _load_protocol()
    runtime = v3_verifier._runtime
    old_id, old_path = runtime.PROTOCOL_ID, runtime.PROTOCOL_PATH
    runtime.PROTOCOL_ID, runtime.PROTOCOL_PATH = PROTOCOL_ID, PROTOCOL_PATH
    try:
        base = runtime.verify_bundle(output)
    finally:
        runtime.PROTOCOL_ID, runtime.PROTOCOL_PATH = old_id, old_path
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    _validate_manifest_files(output, manifest)
    _validate_lock(output, protocol)
    _validate_receipts(output)
    victims = _read_csv(output / "victim_metrics.csv")
    per_seed = _read_csv(output / "per_seed_metrics.csv")
    if sorted(int(row["seed"]) for row in victims) != list(CONFIRMATION_SEEDS):
        raise AssertionError("P6-D v4 victim seed block mismatch")
    eligibility = {int(row["seed"]): _boolean(row["eligible"]) for row in victims}
    for row in per_seed:
        row["eligible"] = eligibility[int(row["seed"])]
    replayed = replay_summary(per_seed)
    stored = json.loads((output / "confirmation_summary.json").read_text(encoding="utf-8"))
    _compare_json(replayed, stored, "confirmation_summary")
    expected_rows = {
        "epoch_history.csv": 130,
        "per_sample_scores.csv": 6600,
        "observable_rows.csv": 8600,
        "per_seed_metrics.csv": 10,
        "victim_metrics.csv": 10,
    }
    for filename, count in expected_rows.items():
        if len(_read_csv(output / filename)) != count:
            raise AssertionError(f"P6-D v4 row-count mismatch: {filename}")
    if base.get("observable_score_groups") != 10 or base.get("epoch_selections") != 10:
        raise AssertionError("P6-D v4 observable/epoch replay count mismatch")
    if base.get("max_abs_score_error", math.inf) > 1e-12:
        raise AssertionError("P6-D v4 score replay exceeds tolerance")
    return {
        "status": "valid",
        "protocol_id": PROTOCOL_ID,
        "victim_rows": len(victims),
        "observable_score_groups": base["observable_score_groups"],
        "epoch_selections": base["epoch_selections"],
        "max_abs_score_error": base["max_abs_score_error"],
        "confirmation_summary": replayed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify_bundle(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
