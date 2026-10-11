#!/usr/bin/env python3
"""Independent checks for the MNIST trigger-family compression audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PAPER14_ROOT / "results/mnist_trigger_family_compression_v1"
PCA_DIMS = (1, 2, 4, 8, 16, 32, 64)


def as_bool(value: str) -> bool:
    return value == "True"


def main(results: Path) -> None:
    metrics = list(csv.DictReader((results / "probe_metrics.csv").open()))
    eligibility = list(csv.DictReader((results / "eligibility.csv").open()))
    summaries = list(csv.DictReader((results / "trigger_summary.csv").open()))
    manifest = json.loads((results / "trigger_family_summary.json").read_text())
    checks: dict[str, bool] = {}
    triggers = sorted({row["trigger"] for row in eligibility})
    checks["nine_triggers"] = len(triggers) == 9
    checks["ninety_seed_trigger_eligibility_configurations"] = len(eligibility) == 90
    checks["twenty_seven_eligible_configurations"] = sum(
        as_bool(row["eligible"]) for row in eligibility
    ) == 27
    evaluated_pairs = {(row["trigger"], row["seed"]) for row in metrics}
    checks["forty_five_feature_evaluated_configurations"] = len(evaluated_pairs) == 45
    checks["every_eligible_configuration_has_features"] = all(
        (row["trigger"], row["seed"]) in evaluated_pairs
        for row in eligibility
        if as_bool(row["eligible"])
    )
    checks["ten_representations_per_configuration"] = all(
        sum(row["trigger"] == trigger and row["seed"] == seed for row in metrics) == 10
        for trigger, seed in evaluated_pairs
    )
    checks["all_auc_values_bounded"] = all(
        0.0 <= float(row[field]) <= 1.0
        for row in metrics
        for field in ("validation_auc", "test_auc")
    )
    checks["cache_numeric_errors_small"] = all(
        receipt["state_norm_max_abs_error"] < 1e-9
        and receipt["pauli_purity_max_abs_error"] < 1e-5
        for receipt in manifest["numeric_receipts"]
    )

    recomputed = {}
    for trigger in triggers:
        rows = [row for row in metrics if row["trigger"] == trigger]
        full_dev = next(
            row for row in rows if row["seed"] == "42" and row["representation"] == "Full"
        )
        threshold = float(full_dev["validation_auc"]) - 0.02
        selected = next(
            (
                dimension
                for dimension in PCA_DIMS
                if float(
                    next(
                        row
                        for row in rows
                        if row["seed"] == "42" and row["representation"] == f"PCA_{dimension}"
                    )["validation_auc"]
                )
                >= threshold
            ),
            None,
        )
        recomputed[trigger] = selected
    stored = {
        row["trigger"]: int(row["selected_pca_dimension"])
        for row in summaries
        if row["selected_pca_dimension"].isdigit()
    }
    checks["all_selected_dimensions_recomputed"] = stored == recomputed
    checks["t9_recomputed_as_pca64"] = recomputed["t7_vs_t9"] == 64

    full_lookup = {
        (row["trigger"], row["seed"]): float(row["test_auc"])
        for row in metrics
        if row["representation"] == "Full"
    }
    eligible_pca2_successes = []
    eligible_z_successes = []
    eligible_two_path_successes = []
    t9_detail = None
    for item in eligibility:
        if not as_bool(item["eligible"]):
            continue
        trigger, seed = item["trigger"], item["seed"]
        dimension = recomputed[trigger]
        selected_auc = float(
            next(
                row
                for row in metrics
                if row["trigger"] == trigger
                and row["seed"] == seed
                and row["representation"] == f"PCA_{dimension}"
            )["test_auc"]
        )
        within = selected_auc >= full_lookup[(trigger, seed)] - 0.02
        z_auc = float(
            next(
                row
                for row in metrics
                if row["trigger"] == trigger
                and row["seed"] == seed
                and row["representation"] == "Z"
            )["test_auc"]
        )
        z_within = z_auc >= full_lookup[(trigger, seed)] - 0.02
        if dimension <= 2 and within:
            eligible_pca2_successes.append((trigger, seed))
        if z_within:
            eligible_z_successes.append((trigger, seed))
        if (dimension <= 2 and within) or z_within:
            eligible_two_path_successes.append((trigger, seed))
        if trigger == "t7_vs_t9":
            xyz_auc = float(
                next(
                    row
                    for row in metrics
                    if row["trigger"] == trigger
                    and row["seed"] == seed
                    and row["representation"] == "XYZ"
                )["test_auc"]
            )
            t9_detail = {
                "seed": int(seed),
                "selected_dimension": dimension,
                "selected_auc": selected_auc,
                "z_auc": z_auc,
                "xyz_auc": xyz_auc,
                "full_auc": full_lookup[(trigger, seed)],
            }
    checks["twenty_four_of_twenty_seven_pass_fixed_pca2_or_less"] = (
        len(eligible_pca2_successes) == 24
    )
    checks["twenty_five_of_twenty_seven_pass_five_z"] = len(eligible_z_successes) == 25
    checks["twenty_six_of_twenty_seven_pass_prespecified_two_path_screen"] = (
        len(eligible_two_path_successes) == 26
    )
    checks["t9_is_effective_counterexample"] = bool(
        t9_detail
        and t9_detail["seed"] == 44
        and t9_detail["selected_dimension"] == 64
        and t9_detail["selected_auc"] >= t9_detail["full_auc"] - 0.02
        and t9_detail["z_auc"] < t9_detail["full_auc"] - 0.02
        and t9_detail["xyz_auc"] < t9_detail["full_auc"] - 0.02
    )

    receipt = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "eligible_pca2_or_less_success_count": len(eligible_pca2_successes),
        "eligible_five_z_success_count": len(eligible_z_successes),
        "eligible_two_path_low_dim_success_count": len(eligible_two_path_successes),
        "eligible_configuration_count": 27,
        "t9_effective_counterexample": t9_detail,
        "recomputed_selected_dimensions": recomputed,
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
