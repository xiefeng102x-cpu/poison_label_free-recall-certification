#!/usr/bin/env python3
"""Verify Pauli compression dimensions, selection logic, and conclusions."""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

import run_bloodmnist_pauli_compression as experiment


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "paper14/results/bloodmnist_pauli_compression_v1"


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check(name: str, condition: bool, detail) -> dict:
    return {"name": name, "passed": bool(condition), "detail": detail}


def explicit_three_qubit_error() -> tuple[float, float]:
    rng = np.random.default_rng(20260928)
    alpha = rng.uniform(-math.pi / 2, math.pi / 2, size=(3, 3))
    local = np.stack([np.cos(alpha), np.sin(alpha)], axis=2).astype(np.float32)
    features, labels = experiment.pauli_xz_by_weight(local)
    pauli = {
        "I": np.eye(2, dtype=np.complex128),
        "X": np.asarray([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.asarray([[1, 0], [0, -1]], dtype=np.complex128),
    }
    maximum_error = 0.0
    maximum_y = 0.0
    for sample in range(len(alpha)):
        state = np.asarray([1.0 + 0.0j])
        for value in alpha[sample]:
            theta = math.pi / 2 - value
            state = np.kron(
                state,
                np.asarray([math.cos(theta / 2), math.sin(theta / 2)]),
            )
        for weight in range(1, 4):
            for column, label in enumerate(labels[weight]):
                operator = np.asarray([1.0 + 0.0j])
                for symbol in label:
                    operator = np.kron(operator, pauli[symbol])
                explicit = float(np.real(np.vdot(state, operator @ state)))
                maximum_error = max(
                    maximum_error, abs(explicit - float(features[weight][sample, column]))
                )
        for label in ("YII", "XYI", "ZZY", "YYY"):
            operator = np.asarray([1.0 + 0.0j])
            for symbol in label:
                operator = np.kron(operator, pauli[symbol])
            maximum_y = max(maximum_y, abs(float(np.real(np.vdot(state, operator @ state)))))
    return maximum_error, maximum_y


def recompute_selection(rows: list[dict]) -> dict:
    candidates = [
        row
        for row in rows
        if int(row["seed"]) == 42
        and row["detector"] == "logistic"
        and row["path"] in {"cumulative_weight", "pca", "fisher"}
        and row["validation_auc"]
    ]
    full = next(
        row
        for row in candidates
        if row["path"] == "cumulative_weight"
        and row["representation"] == "weight_le_8"
    )
    threshold = float(full["validation_auc"]) - 0.02
    eligible = [row for row in candidates if float(row["validation_auc"]) >= threshold]
    eligible.sort(
        key=lambda row: (
            int(row["dimension"]),
            -float(row["validation_auc"]),
            {"cumulative_weight": 0, "pca": 1, "fisher": 2}[row["path"]],
        )
    )
    return {
        "path": eligible[0]["path"],
        "representation": eligible[0]["representation"],
        "dimension": int(eligible[0]["dimension"]),
        "threshold": threshold,
    }


def main() -> None:
    raw = json.loads(
        (RESULT_ROOT / "pauli_compression_summary.json").read_text(encoding="utf-8")
    )
    analysis = json.loads(
        (RESULT_ROOT / "analysis_summary.json").read_text(encoding="utf-8")
    )
    rows = load_csv(RESULT_ROOT / "probe_metrics.csv")
    selected = raw["selected_minimum_representation"]
    recomputed = recompute_selection(rows)
    explicit_error, y_error = explicit_three_qubit_error()
    exact_counts = {
        weight: math.comb(8, weight) * 2**weight for weight in range(1, 9)
    }
    receipt_counts_match = all(
        {
            int(key): int(value)
            for key, value in receipt["feature_counts_by_exact_weight"].items()
        }
        == exact_counts
        for receipt in raw["receipts"]
    )
    confirmation = raw["confirmation"]
    selected_auc = np.asarray(confirmation["selected_auc_values"], dtype=float)
    full_auc = np.asarray(confirmation["full_auc_values"], dtype=float)
    pca_one_variance = analysis["headline"]["pca_1_explained_variance"]

    checks = [
        check(
            "physical and effective Pauli dimensions are exact",
            analysis["integrity"]["expected_physical_pauli_dimension"] == 65535
            and analysis["integrity"]["observed_effective_xz_dimension"] == 6560
            and analysis["integrity"]["identically_zero_y_dimension"] == 58975,
            analysis["integrity"],
        ),
        check(
            "exact-weight feature counts follow C(8,w)2^w",
            receipt_counts_match and sum(exact_counts.values()) == 6560,
            exact_counts,
        ),
        check(
            "analytic X/Z Pauli products match explicit statevectors",
            explicit_error <= 1e-6,
            explicit_error,
        ),
        check(
            "Y-containing Pauli expectations are zero",
            y_error <= 1e-12,
            y_error,
        ),
        check(
            "Bloch norm and Pauli purity identities hold",
            analysis["integrity"]["max_local_bloch_norm_error"] <= 1e-6
            and analysis["integrity"]["max_full_pauli_purity_error"] <= 1e-3,
            analysis["integrity"],
        ),
        check(
            "development minimum is recomputed from validation AUC only",
            recomputed["path"] == selected["path"]
            and recomputed["representation"] == selected["representation"]
            and recomputed["dimension"] == selected["dimension"]
            and abs(recomputed["threshold"] - selected["threshold"]) <= 1e-12,
            {"saved": selected, "recomputed": recomputed},
        ),
        check(
            "PCA-1 is within epsilon of full on every confirmation seed",
            np.all(selected_auc >= full_auc - float(selected["epsilon"])),
            {
                "selected": selected_auc.tolist(),
                "full": full_auc.tolist(),
                "epsilon": selected["epsilon"],
            },
        ),
        check(
            "confirmation seed coverage is complete",
            confirmation["seeds"] == [43, 44, 45, 46],
            confirmation["seeds"],
        ),
        check(
            "PCA-1 captures less than one third of variance",
            pca_one_variance["max"] < 1 / 3,
            pca_one_variance,
        ),
        check(
            "source reconstructions remain within tolerance",
            analysis["integrity"]["max_clean_reconstruction_error"] <= 5e-3
            and analysis["integrity"]["max_triggered_reconstruction_error"] <= 5e-3,
            analysis["integrity"],
        ),
    ]
    receipt = {
        "status": "PASS" if all(row["passed"] for row in checks) else "FAIL",
        "checks": checks,
        "explicit_statevector_xz_max_abs_error": explicit_error,
        "explicit_statevector_y_max_abs_expectation": y_error,
    }
    (RESULT_ROOT / "verification_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
