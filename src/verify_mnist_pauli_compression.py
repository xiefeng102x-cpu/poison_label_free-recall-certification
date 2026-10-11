#!/usr/bin/env python3
"""Independent structural checks for the MNIST Pauli compression audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PAPER14_ROOT / "results/mnist_pauli_compression_v1"


def main(results: Path) -> None:
    rows = list(csv.DictReader((results / "probe_metrics.csv").open()))
    summary = json.loads((results / "pauli_compression_summary.json").read_text())
    checks: dict[str, bool] = {}
    checks["three_expected_seeds"] = sorted({int(row["seed"]) for row in rows}) == [42, 43, 44]
    checks["twenty_representations_per_seed"] = all(
        sum(int(row["seed"]) == seed for row in rows) == 20 for seed in (42, 43, 44)
    )
    expected_exact = {weight: math.comb(5, weight) * 3**weight for weight in range(1, 6)}
    expected_cumulative = {
        weight: sum(expected_exact[w] for w in range(1, weight + 1))
        for weight in range(1, 6)
    }
    checks["pauli_weight_dimensions"] = all(
        int(row["dimension"])
        == (
            expected_exact[int(row["pauli_weight"])]
            if row["path"] == "exact_weight"
            else expected_cumulative[int(row["pauli_weight"])]
        )
        for row in rows
        if row["path"] in {"exact_weight", "cumulative_weight"}
    )
    checks["auc_values_bounded"] = all(
        0.0 <= float(row[field]) <= 1.0
        for row in rows
        for field in ("validation_auc", "test_auc", "test_auc_ci_low", "test_auc_ci_high")
    )
    checks["bootstrap_intervals_contain_auc"] = all(
        float(row["test_auc_ci_low"]) - 1e-12
        <= float(row["test_auc"])
        <= float(row["test_auc_ci_high"]) + 1e-12
        for row in rows
    )
    development_full = next(
        row for row in rows if row["seed"] == "42" and row["representation"] == "Full"
    )
    threshold = float(development_full["validation_auc"]) - 0.02
    development_pca = sorted(
        (row for row in rows if row["seed"] == "42" and row["path"] == "pca"),
        key=lambda row: int(row["dimension"]),
    )
    selected = next(row for row in development_pca if float(row["validation_auc"]) >= threshold)
    checks["selection_recomputed_as_pca_2"] = selected["representation"] == "pca_2"
    checks["pca_1_fails_development_threshold"] = float(development_pca[0]["validation_auc"]) < threshold
    checks["summary_matches_recomputed_selection"] = (
        summary["selected_pca"]["representation"] == selected["representation"]
        and abs(summary["selected_pca"]["validation_threshold"] - threshold) < 1e-12
    )
    checks["all_source_caches_exist"] = all(
        Path(receipt["source_cache"]).is_file() for receipt in summary["receipts"]
    )
    checks["source_numeric_errors_small"] = all(
        receipt["state_norm_max_abs_error"] < 1e-9
        and receipt["pauli_purity_max_abs_error"] < 1e-5
        for receipt in summary["receipts"]
    )
    receipt = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "recomputed_development_threshold": threshold,
        "recomputed_selection": selected["representation"],
    }
    (results / "verification_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    main(args.results.resolve())
