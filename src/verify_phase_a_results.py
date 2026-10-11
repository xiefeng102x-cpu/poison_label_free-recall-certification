#!/usr/bin/env python3
"""Independent result checks for the Paper 14 Phase A outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pairwise_auc(rows: list[dict]) -> float:
    """Mann-Whitney AUC, independent of sklearn and the experiment helper."""
    positive = [float(row["score"]) for row in rows if int(row["label"]) == 1]
    negative = [float(row["score"]) for row in rows if int(row["label"]) == 0]
    return sum(
        (pos > neg) + 0.5 * (pos == neg)
        for pos in positive
        for neg in negative
    ) / (len(positive) * len(negative))


def main(results: Path) -> None:
    metrics = read_csv(results / "heldout_metrics.csv")
    scores = read_csv(results / "heldout_scores.csv")
    splits = read_csv(results / "splits.csv")
    pair_audit = read_csv(results / "pair_audit.csv")
    invariance = read_csv(results / "unitary_invariance_checks.csv")

    grouped_scores: dict[tuple, list[dict]] = {}
    for row in scores:
        key = (
            row["seed"],
            row["victim_kind"],
            row["checkpoint"],
            row["representation"],
            row["detector"],
        )
        grouped_scores.setdefault(key, []).append(row)

    max_auc_error = 0.0
    for row in metrics:
        key = (
            row["seed"],
            row["victim_kind"],
            row["checkpoint"],
            row["representation"],
            row["detector"],
        )
        recomputed = pairwise_auc(grouped_scores[key])
        max_auc_error = max(max_auc_error, abs(recomputed - float(row["auc"])))

    split_counts = {}
    for row in splits:
        seed = row["seed"]
        split_counts.setdefault(seed, {}).setdefault(row["split"].strip(), set()).add(
            row["base_sample_id"]
        )
    leakage_free = True
    for groups in split_counts.values():
        leakage_free &= not (
            groups["train"] & groups["validation"]
            or groups["train"] & groups["test"]
            or groups["validation"] & groups["test"]
        )

    exact_score_pair_coverage = True
    for rows in grouped_scores.values():
        counts = {}
        for row in rows:
            counts[row["base_sample_id"]] = counts.get(row["base_sample_id"], 0) + 1
        exact_score_pair_coverage &= len(rows) == 72 and set(counts.values()) == {2}

    max_pair_error = max(float(row["pair_reconstruction_max_abs_error"]) for row in pair_audit)
    max_gram_relative_error = max(
        float(row["full_pauli_gram_relative_fro_delta_vs_encoding"])
        for row in invariance
    )
    checks = {
        "all_108_auc_values_independently_recomputed": max_auc_error <= 1e-12,
        "pair_grouped_split_has_no_overlap": bool(leakage_free),
        "each_score_group_has_36_complete_test_pairs": bool(exact_score_pair_coverage),
        "matched_trigger_reconstruction_exact": max_pair_error <= 1e-7,
        "full_pauli_fixed_unitary_gram_invariant": max_gram_relative_error <= 1e-6,
    }
    receipt = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metric_rows": len(metrics),
        "score_rows": len(scores),
        "score_groups": len(grouped_scores),
        "independent_auc_max_abs_error": max_auc_error,
        "max_pair_reconstruction_error": max_pair_error,
        "max_full_pauli_gram_relative_fro_error": max_gram_relative_error,
        "split_pair_counts": {
            seed: {name: len(values) for name, values in groups.items()}
            for seed, groups in split_counts.items()
        },
    }
    with (results / "verification_receipt.json").open("w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    main(args.results.resolve())
