#!/usr/bin/env python3
"""Independent checks for the MNIST single-sample screening experiment."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PAPER14_ROOT / "results/phase_a_mnist5q_real_v1"
DEFAULT_RESULTS = PAPER14_ROOT / "results/mnist_single_sample_screening_v1"


def main(source: Path, results: Path) -> None:
    scores = list(csv.DictReader((results / "sample_scores.csv").open()))
    metrics = list(csv.DictReader((results / "screening_metrics.csv").open()))
    summary = json.loads((results / "screening_summary.json").read_text())
    parameters = json.loads((results / "screener_parameters.json").read_text())
    checks: dict[str, bool] = {}

    configurations = sorted({row["configuration"] for row in scores})
    checks["four_expected_configurations"] = configurations == [
        "universal_encoding_pca2",
        "universal_encoding_z",
        "victim_specific_output_pca2",
        "victim_specific_output_z",
    ]
    checks["balanced_216_sample_evaluation_per_configuration"] = all(
        sum(row["configuration"] == configuration for row in scores) == 216
        and sum(
            row["configuration"] == configuration and row["label"] == "0"
            for row in scores
        )
        == 108
        and sum(
            row["configuration"] == configuration and row["label"] == "1"
            for row in scores
        )
        == 108
        for configuration in configurations
    )
    checks["stored_decisions_match_threshold_rule"] = all(
        int(row["predicted_attack"])
        == int(float(row["score"]) >= float(row["threshold"]))
        for row in scores
    )

    primary = [row for row in scores if row["configuration"] == "universal_encoding_z"]
    tn = sum(row["label"] == "0" and row["predicted_attack"] == "0" for row in primary)
    fp = sum(row["label"] == "0" and row["predicted_attack"] == "1" for row in primary)
    fn = sum(row["label"] == "1" and row["predicted_attack"] == "0" for row in primary)
    tp = sum(row["label"] == "1" and row["predicted_attack"] == "1" for row in primary)
    checks["primary_confusion_recomputed"] = (tn, fp, fn, tp) == (108, 0, 2, 106)
    checks["primary_summary_matches_recomputed_counts"] = (
        summary["primary_result"]["tn"],
        summary["primary_result"]["fp"],
        summary["primary_result"]["fn"],
        summary["primary_result"]["tp"],
    ) == (tn, fp, fn, tp)
    checks["primary_all_seed_auc_is_one"] = all(
        abs(float(row["auc"]) - 1.0) < 1e-12
        for row in metrics
        if row["configuration"] == "universal_encoding_z"
    )
    checks["zero_observed_fpr_retains_nonzero_upper_bound"] = (
        summary["primary_result"]["fpr"] == 0.0
        and summary["primary_result"]["fpr_ci_high"] > 0.03
    )

    split_map = {
        (int(row["seed"]), row["base_sample_id"]): row["split"]
        for row in csv.DictReader((source / "splits.csv").open())
    }
    checks["all_scored_rows_are_test_only"] = all(
        split_map[(int(row["evaluation_seed"]), row["base_sample_id"])] == "test"
        for row in scores
    )
    pauli_labels = [
        "".join(chars)
        for chars in itertools.product("IXYZ", repeat=5)
        if any(char != "I" for char in chars)
    ]
    weights = np.asarray([sum(char != "I" for char in label) for label in pauli_labels])
    z_indices = np.asarray(
        [i for i, label in enumerate(pauli_labels) if weights[i] == 1 and "Z" in label]
    )
    checks["parameter_feature_order_matches_cache"] = parameters["feature_order"] == [
        pauli_labels[index] for index in z_indices
    ]

    expected_scores = []
    coefficient = np.asarray(parameters["coefficient"])
    center = np.asarray(parameters["center"])
    intercept = float(parameters["intercept"])
    for seed in (42, 43, 44):
        payload = np.load(source / "cache" / f"seed{seed}_shared_encoding_pauli.npz")
        pair_ids = payload["pair_ids"].astype(str)
        mask = np.asarray([split_map[(seed, pair_id)] == "test" for pair_id in pair_ids])
        full = np.concatenate([payload["clean_full"][mask], payload["attack_full"][mask]])
        expected_scores.extend(((full[:, z_indices] - center) @ coefficient + intercept).tolist())
    stored_scores = [float(row["score"]) for row in primary]
    maximum_score_error = float(np.max(np.abs(np.asarray(expected_scores) - stored_scores)))
    # The cached Pauli features are float32, while sklearn evaluates the fitted
    # pipeline in float64.  Reconstructing the affine score in NumPy can change
    # accumulation order, so use a tolerance far below any observed margin.
    checks["exported_parameters_reproduce_all_primary_scores"] = maximum_score_error < 1e-6

    receipt = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "maximum_exported_score_reconstruction_error": maximum_score_error,
        "recomputed_primary_confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    (results / "verification_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    main(args.source.resolve(), args.results.resolve())
