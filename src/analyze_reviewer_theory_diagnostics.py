#!/usr/bin/env python3
"""Reproduce low-cost diagnostics requested by the theory review.

The script uses only archived, sample-level artifacts.  It does not retrain a
victim, alter a detector, or consume a new confirmation seed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OBSERVABLES = (
    ROOT
    / "results/p6d_final_observable_adaptive_v4/confirmation/observable_rows.csv"
)
SANITIZATION = ROOT / "results/bloodmnist_post_sanitization_v1/per_seed_metrics.csv"
OUTPUT = ROOT / "results/reviewer_response_2026_10_06"
PI = 60.0 / 660.0
BOOTSTRAP_SEED = 20261006
BOOTSTRAP_DRAWS = 200_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def member_bias() -> tuple[list[dict], dict]:
    rows = read_rows(OBSERVABLES)
    features = [name for name in rows[0] if name.startswith("delta_xz16_")]
    by_seed: dict[int, dict[str, list[np.ndarray]]] = {}
    for row in rows:
        seed = int(row["seed"])
        population = row["population"]
        by_seed.setdefault(seed, {}).setdefault(population, []).append(
            np.asarray([float(row[name]) for name in features], dtype=np.float64)
        )
    output = []
    for seed in sorted(by_seed):
        groups = by_seed[seed]
        required = {
            "reference": 120,
            "clean_candidate": 600,
            "poison_candidate": 60,
            "calibration": 80,
        }
        counts = {name: len(groups.get(name, [])) for name in required}
        if counts != required:
            raise AssertionError(f"seed {seed}: population counts {counts} != {required}")
        reference = np.mean(groups["reference"], axis=0)
        clean = np.mean(groups["clean_candidate"], axis=0)
        poison = np.mean(groups["poison_candidate"], axis=0)
        b_norm = float(np.linalg.norm(clean - reference))
        delta_norm = float(np.linalg.norm(poison - clean))
        scaled_signal = PI * delta_norm
        output.append(
            {
                "seed": seed,
                "dimension": len(features),
                "poison_prevalence": PI,
                "b_w_norm": b_norm,
                "delta_w_norm": delta_norm,
                "pi_delta_w_norm": scaled_signal,
                "bias_to_scaled_signal_ratio": b_norm / scaled_signal,
            }
        )
    ratios = np.asarray([row["bias_to_scaled_signal_ratio"] for row in output])
    summary = {
        "seed_count": len(output),
        "poison_prevalence": PI,
        "ratio_mean": float(np.mean(ratios)),
        "ratio_median": float(np.median(ratios)),
        "ratio_min": float(np.min(ratios)),
        "ratio_max": float(np.max(ratios)),
        "seeds_with_ratio_at_least_one": int(np.sum(ratios >= 1.0)),
        "interpretation": (
            "Empirical transformed-space member/reference bias is not negligible "
            "relative to the prevalence-scaled poison displacement."
        ),
    }
    return output, summary


def sanitization_audit() -> tuple[list[dict], dict]:
    rows = read_rows(SANITIZATION)
    by_seed: dict[int, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), {})[row["condition"]] = row
    output = []
    for seed in sorted(by_seed):
        conditions = by_seed[seed]
        detected = float(conditions["detected_removal"]["qml_asr"])
        clean_oracle = float(conditions["oracle_poison_removal"]["qml_asr"])
        output.append(
            {
                "seed": seed,
                "detector_guided_qml_asr": detected,
                "clean_oracle_qml_asr": clean_oracle,
                "paired_difference_guided_minus_clean_oracle": detected - clean_oracle,
            }
        )
    differences = np.asarray(
        [row["paired_difference_guided_minus_clean_oracle"] for row in output],
        dtype=np.float64,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(differences), size=(BOOTSTRAP_DRAWS, len(differences)))
    boot = np.mean(differences[indices], axis=1)
    interval = np.quantile(boot, [0.025, 0.975])
    summary = {
        "seed_count": len(output),
        "detector_guided_mean_qml_asr": float(
            np.mean([row["detector_guided_qml_asr"] for row in output])
        ),
        "clean_oracle_mean_qml_asr": float(
            np.mean([row["clean_oracle_qml_asr"] for row in output])
        ),
        "paired_mean_difference": float(np.mean(differences)),
        "paired_seed_bootstrap_95_interval": [float(interval[0]), float(interval[1])],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "interpretation": (
            "Detector-guided retraining numerically reaches the never-poisoned/oracle-removal "
            "QML reference in this seven-seed block; the interval is not an equivalence test."
        ),
    }
    return output, summary


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bias_rows, bias_summary = member_bias()
    sanitization_rows, sanitization_summary = sanitization_audit()
    bias_path = OUTPUT / "final_observable_member_bias.csv"
    sanitization_path = OUTPUT / "sanitization_clean_oracle_comparison.csv"
    write_csv(bias_path, bias_rows)
    write_csv(sanitization_path, sanitization_rows)
    payload = {
        "schema": "paper14_reviewer_theory_diagnostics_v1",
        "inputs": {
            str(OBSERVABLES.relative_to(ROOT)): sha256(OBSERVABLES),
            str(SANITIZATION.relative_to(ROOT)): sha256(SANITIZATION),
        },
        "member_bias": bias_summary,
        "sanitization": sanitization_summary,
        "outputs": {
            str(bias_path.relative_to(ROOT)): sha256(bias_path),
            str(sanitization_path.relative_to(ROOT)): sha256(sanitization_path),
        },
    }
    json_path = OUTPUT / "diagnostics.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report = f"""# Reviewer theory diagnostics

No new victim was trained and no fresh confirmation seed was consumed.

## Final-observable member bias

- Seeds: {bias_summary['seed_count']}
- Prevalence: {bias_summary['poison_prevalence']:.6f}
- `||b_w|| / (pi ||delta_w||)`: mean {bias_summary['ratio_mean']:.3f}, median {bias_summary['ratio_median']:.3f}, range {bias_summary['ratio_min']:.3f}--{bias_summary['ratio_max']:.3f}
- Seeds with ratio at least one: {bias_summary['seeds_with_ratio_at_least_one']}/{bias_summary['seed_count']}

This is a descriptive transformed-space check, not a certificate of Proposition A.5.

## Clean/oracle sanitization reference

The oracle-removal condition removes all 60 poisons and no clean candidate, then
retrains from scratch.  It is therefore also the never-poisoned training reference
for this frozen split and trigger.

- Detector-guided mean QML ASR: {100 * sanitization_summary['detector_guided_mean_qml_asr']:.2f}%
- Clean/oracle mean QML ASR: {100 * sanitization_summary['clean_oracle_mean_qml_asr']:.2f}%
- Paired mean difference: {100 * sanitization_summary['paired_mean_difference']:.2f} percentage points
- Paired seed-bootstrap 95% interval: {100 * sanitization_summary['paired_seed_bootstrap_95_interval'][0]:.2f} to {100 * sanitization_summary['paired_seed_bootstrap_95_interval'][1]:.2f} percentage points

The comparison supports “numerically reaches the clean/oracle reference” but is
not an equivalence test and does not support adaptive-attack sanitization.
"""
    (OUTPUT / "AUDIT_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
