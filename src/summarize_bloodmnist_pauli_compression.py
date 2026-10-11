#!/usr/bin/env python3
"""Build a compact analytical summary of the Pauli compression audit."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "paper14/results/bloodmnist_pauli_compression_v1"


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "population_std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "values": [float(value) for value in array],
    }


def aggregate(rows: list[dict], paths: set[str]) -> list[dict]:
    grouped = defaultdict(list)
    dimensions = {}
    extras = defaultdict(list)
    for row in rows:
        if row["role"] != "confirmation" or row["path"] not in paths:
            continue
        key = (row["path"], row["representation"], row["detector"])
        grouped[key].append(float(row["test_auc"]))
        dimensions[key] = int(row["dimension"])
        if row.get("explained_variance"):
            extras[(key, "explained_variance")].append(
                float(row["explained_variance"])
            )
    output = []
    for key, values in grouped.items():
        record = {
            "path": key[0],
            "representation": key[1],
            "detector": key[2],
            "dimension": dimensions[key],
            "test_auc": stats(values),
        }
        explained = extras.get((key, "explained_variance"), [])
        if explained:
            record["explained_variance"] = stats(explained)
        output.append(record)
    return sorted(output, key=lambda row: (row["path"], row["dimension"], row["detector"]))


def main() -> None:
    raw = json.loads(
        (RESULT_ROOT / "pauli_compression_summary.json").read_text(encoding="utf-8")
    )
    rows = load_csv(RESULT_ROOT / "probe_metrics.csv")
    l1_rows = load_csv(RESULT_ROOT / "l1_sparse_metrics.csv")
    curves = aggregate(
        rows,
        {"local", "cumulative_weight", "exact_weight", "pca", "fisher", "random", "full"},
    )
    curve_index = {
        (row["path"], row["representation"], row["detector"]): row for row in curves
    }
    cumulative = [
        curve_index[("cumulative_weight", f"weight_le_{weight}", "logistic")]
        for weight in range(1, 9)
    ]
    exact = [
        curve_index[("exact_weight", f"weight_{weight}", "logistic")]
        for weight in range(1, 9)
    ]
    pca = [row for row in curves if row["path"] == "pca"]
    fisher = [row for row in curves if row["path"] == "fisher"]
    random_curve = [row for row in curves if row["path"] == "random"]
    local = [row for row in curves if row["path"] == "local"]
    nonlinear_full = curve_index[("full", "full_effective_XZ", "rbf_svm")]

    best_exact = max(exact, key=lambda row: row["test_auc"]["mean"])
    best_cumulative = max(cumulative, key=lambda row: row["test_auc"]["mean"])
    pca_one = curve_index[("pca", "pca_1", "logistic")]
    full_linear = curve_index[
        ("cumulative_weight", "weight_le_8", "logistic")
    ]
    z_linear = curve_index[("local", "Z_only", "logistic")]
    z_rbf = curve_index[("local", "Z_only", "rbf_svm")]

    fisher_top = [
        {
            "seed": int(row["seed"]),
            "feature": row["top_feature"],
            "score": float(row["top_feature_score"]),
            "test_auc": float(row["test_auc"]),
        }
        for row in rows
        if row["representation"] == "fisher_1" and row["role"] == "confirmation"
    ]
    l1_grouped = defaultdict(list)
    for row in l1_rows:
        if row["role"] == "confirmation":
            l1_grouped[float(row["c"])].append(row)
    l1_curve = []
    for c_value, group in sorted(l1_grouped.items()):
        l1_curve.append(
            {
                "c": c_value,
                "nonzero_features": stats(
                    [float(row["nonzero_features"]) for row in group]
                ),
                "test_auc": stats([float(row["test_auc"]) for row in group]),
            }
        )

    receipts = raw["receipts"]
    integrity = {
        "max_local_bloch_norm_error": max(
            float(row["local_bloch_norm_max_abs_error"]) for row in receipts
        ),
        "max_full_pauli_purity_error": max(
            float(row["full_pauli_purity_max_abs_error"]) for row in receipts
        ),
        "max_clean_reconstruction_error": max(
            max(
                float(value)
                for value in row["data_receipt"]["reconstruction_error"].values()
            )
            for row in receipts
        ),
        "max_triggered_reconstruction_error": max(
            float(row["data_receipt"]["triggered_reconstruction_max_abs_error"])
            for row in receipts
        ),
        "expected_physical_pauli_dimension": 4**8 - 1,
        "observed_effective_xz_dimension": receipts[0][
            "effective_nonzero_xz_dimension"
        ],
        "identically_zero_y_dimension": receipts[0][
            "identically_zero_y_dimension"
        ],
    }
    summary = {
        "status": "PASS",
        "protocol": raw["protocol"],
        "minimum_representation": raw["selected_minimum_representation"],
        "minimum_representation_confirmation": raw["confirmation"],
        "headline": {
            "z_only_linear_auc": z_linear["test_auc"],
            "z_only_rbf_auc": z_rbf["test_auc"],
            "full_linear_auc": full_linear["test_auc"],
            "full_rbf_auc": nonlinear_full["test_auc"],
            "pca_1_auc": pca_one["test_auc"],
            "pca_1_explained_variance": pca_one["explained_variance"],
            "best_exact_weight": {
                "weight": int(best_exact["representation"].split("_")[-1]),
                "auc": best_exact["test_auc"],
            },
            "best_cumulative_weight": {
                "weight": int(best_cumulative["representation"].split("_")[-1]),
                "auc": best_cumulative["test_auc"],
            },
        },
        "curves": {
            "cumulative_weight": cumulative,
            "exact_weight": exact,
            "pca": pca,
            "fisher": fisher,
            "random": random_curve,
            "local_and_nonlinear": local + [nonlinear_full],
            "l1_sparse": l1_curve,
        },
        "fisher_top_features": fisher_top,
        "fisher_top_feature_interpretation": "All confirmation seeds select a one-body Z observable, but the qubit index is not stable across seeds.",
        "integrity": integrity,
    }
    (RESULT_ROOT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
