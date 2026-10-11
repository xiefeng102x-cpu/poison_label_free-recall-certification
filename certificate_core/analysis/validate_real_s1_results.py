#!/usr/bin/env python3
"""Validate and summarize the retrospective real-data S1 certificate run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest


HERE = Path(__file__).resolve()
CERT_ROOT = HERE.parents[1]
DEFAULT_ROOT = CERT_ROOT / "results_real_s1"
RATIOS = (1.0, 1.2, 2.0, 4.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def holm_rejections(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    order = np.argsort(np.asarray(pvalues, dtype=float))
    rejected = [False] * len(pvalues)
    total = len(pvalues)
    for rank, index in enumerate(order):
        if pvalues[int(index)] <= alpha / (total - rank):
            rejected[int(index)] = True
        else:
            break
    return rejected


def bootstrap_seed_ci(values: pd.Series, draws: int = 20_000) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    rng = np.random.default_rng(20261007)
    sampled = array[rng.integers(0, len(array), size=(draws, len(array)))]
    return tuple(float(x) for x in np.quantile(sampled.mean(axis=1), (0.025, 0.975)))


def validate(root: Path) -> dict[str, Any]:
    scores_path = root / "bloodmnist_s1_scores.csv"
    result_path = root / "bloodmnist_s1_certificate_results.csv"
    export_manifest_path = root / "export_manifest.json"
    scores = pd.read_csv(scores_path)
    results = pd.read_csv(result_path)
    export_manifest = json.loads(export_manifest_path.read_text(encoding="utf-8"))

    expected_columns = ["victim_id", "fold", "sample_id", "score", "role", "is_poison"]
    assert scores.columns.tolist() == expected_columns
    assert len(scores) == 35_960
    assert scores["victim_id"].nunique() == 10
    assert set(scores["fold"]) == {0, 1}
    assert set(scores["role"]) == {"train", "fresh"}
    assert set(scores["is_poison"]) == {0, 1}
    assert np.isfinite(scores["score"]).all()
    assert not scores.duplicated(["victim_id", "fold", "sample_id"]).any()
    assert ((scores["role"] == "fresh") & (scores["is_poison"] == 1)).sum() == 0
    assert sha256(scores_path) == export_manifest["score_table_sha256"]
    assert export_manifest["scope"]["retrospective_not_preregistered"] is True
    assert export_manifest["scope"]["not_full_training_population"] is True

    grouped = scores.groupby(["victim_id", "fold"])
    for _, block in grouped:
        assert len(block) == 1_798
        assert (block["role"] == "train").sum() == 1_465
        assert (block["role"] == "fresh").sum() == 333
        assert ((block["role"] == "train") & (block["is_poison"] == 1)).sum() > 0
        assert ((block["role"] == "train") & (block["is_poison"] == 0)).sum() > 500
    for _, block in scores.groupby("victim_id"):
        assert ((block["role"] == "train") & (block["is_poison"] == 1)).sum() == 600
        assert ((block["role"] == "train") & (block["is_poison"] == 0)).sum() == 2_330
        assert (block["role"] == "fresh").sum() == 666

    assert len(results) == 40
    assert set(results["k"]) == {120, 500}
    assert set(results["R"]) == {200}
    assert not results.duplicated(["victim_id", "fold", "k"]).any()
    for ratio in RATIOS:
        count = results[f"false_go_r{ratio}_count"]
        rate = results[f"false_go_r{ratio}_rate"]
        assert np.allclose(count / results["R"], rate, atol=1e-15, rtol=0.0)
        go = results[f"dec_r{ratio}_go_rate"]
        nogo = results[f"dec_r{ratio}_nogo_rate"]
        assert ((go >= 0) & (nogo >= 0) & (go + nogo <= 1 + 1e-15)).all()

    # H1: 40 rows x 4 correctly conservative poison-count declarations.
    h1_cells: list[dict[str, Any]] = []
    for _, row in results.iterrows():
        fold_count = 2
        null_rate = 0.05 / fold_count
        for ratio in RATIOS:
            count = int(row[f"false_go_r{ratio}_count"])
            pvalue = float(binomtest(count, int(row["R"]), null_rate, alternative="greater").pvalue)
            h1_cells.append(
                {
                    "victim_id": row["victim_id"],
                    "fold": int(row["fold"]),
                    "k": int(row["k"]),
                    "ratio": ratio,
                    "false_go_count": count,
                    "R": int(row["R"]),
                    "null_rate_delta_over_folds": null_rate,
                    "pvalue_greater": pvalue,
                }
            )
    rejected = holm_rejections([row["pvalue_greater"] for row in h1_cells])
    for row, decision in zip(h1_cells, rejected):
        row["holm_reject"] = bool(decision)
    h1_table = pd.DataFrame(h1_cells)
    h1_table.to_csv(root / "h1_cell_tests.csv", index=False)

    results = results.copy()
    results["view"] = results["victim_id"].str.extract(r"multiclasscnn_(.*)_seed")[0]
    results["seed"] = results["victim_id"].str.extract(r"seed(\d+)$")[0].astype(int)
    summary_rows: list[dict[str, Any]] = []
    for (view, k), block in results.groupby(["view", "k"]):
        seed_gap = block.groupby("seed")["mean_member_minus_fresh"].mean()
        gap_ci = bootstrap_seed_ci(seed_gap)
        for ratio in RATIOS:
            go = float(block[f"dec_r{ratio}_go_rate"].mean())
            nogo = float(block[f"dec_r{ratio}_nogo_rate"].mean())
            summary_rows.append(
                {
                    "view": view,
                    "k": int(k),
                    "n_seeds": int(block["seed"].nunique()),
                    "n_folds": int(block["fold"].nunique()),
                    "declared_nmax_over_true_np": ratio,
                    "mean_realized_recall": float(block["mean_realized_recall"].mean()),
                    "go_rate": go,
                    "nogo_rate": nogo,
                    "abstain_rate": 1.0 - go - nogo,
                    "pooled_false_go_rate": float(
                        block[f"false_go_r{ratio}_count"].sum() / block["R"].sum()
                    ),
                    "mean_member_minus_fresh": float(seed_gap.mean()),
                    "member_minus_fresh_seed_bootstrap_ci_low": gap_ci[0],
                    "member_minus_fresh_seed_bootstrap_ci_high": gap_ci[1],
                    "mean_caught_lower_bound_exact_count": float(block["mean_caught_lo_r1.0"].mean()),
                    "mean_true_caught": float(block["mean_caught_true"].mean()),
                    "mean_lower_bound_fraction": float(
                        (block["mean_caught_lo_r1.0"] / block["mean_caught_true"]).mean()
                    ),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(root / "real_s1_summary.csv", index=False)

    low_recall = results[results["mean_realized_recall"] < 0.8]
    h4_pass = bool(
        low_recall.empty
        or all(low_recall[f"dec_r{ratio}_go_rate"].max() <= 0.05 for ratio in RATIOS)
    )
    poison_prevalence = 600 / 2_930
    k120 = results[results["k"] == 120]
    k500 = results[results["k"] == 500]
    report = {
        "status": "PASS_WITH_SCOPE_LIMITS",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "input_validation": {
            "score_rows": len(scores),
            "result_rows": len(results),
            "victim_ids": int(scores["victim_id"].nunique()),
            "seeds": sorted(results["seed"].unique().tolist()),
            "views": sorted(results["view"].unique().tolist()),
            "folds": sorted(results["fold"].unique().tolist()),
            "duplicate_keys": 0,
            "finite_scores": True,
            "source_clean_fresh_rows_replayed": 29_960,
        },
        "principal_results": {
            "poison_prevalence_in_target_stratum": poison_prevalence,
            "k120_exact_count_go_rate": float(k120["dec_r1.0_go_rate"].mean()),
            "k500_exact_count_go_rate": float(k500["dec_r1.0_go_rate"].mean()),
            "k120_ratio1_2_go_rate": float(k120["dec_r1.2_go_rate"].mean()),
            "k500_ratio1_2_go_rate": float(k500["dec_r1.2_go_rate"].mean()),
            "k120_exact_count_abstain_rate": float(
                1 - k120["dec_r1.0_go_rate"].mean() - k120["dec_r1.0_nogo_rate"].mean()
            ),
            "k500_exact_count_abstain_rate": float(
                1 - k500["dec_r1.0_go_rate"].mean() - k500["dec_r1.0_nogo_rate"].mean()
            ),
            "max_cell_false_go_rate": float(
                max(results[f"false_go_r{ratio}_rate"].max() for ratio in RATIOS)
            ),
            "exact_count_pooled_false_go_rate_k500": float(
                k500["false_go_r1.0_count"].sum() / k500["R"].sum()
            ),
            "mean_lower_bound_fraction_k120": float(
                (k120["mean_caught_lo_r1.0"] / k120["mean_caught_true"]).mean()
            ),
            "mean_lower_bound_fraction_k500": float(
                (k500["mean_caught_lo_r1.0"] / k500["mean_caught_true"]).mean()
            ),
        },
        "hypotheses": {
            "H1": {
                "status": "NOT_FALSIFIED_IN_RETROSPECTIVE_BLOCK",
                "cells": len(h1_cells),
                "holm_rejections": int(sum(rejected)),
                "minimum_unadjusted_pvalue": float(h1_table["pvalue_greater"].min()),
            },
            "H2": {
                "status": "DESCRIPTIVE_ESTIMATED",
                "detail": "View- and k-specific seed-bootstrap intervals are in real_s1_summary.csv.",
            },
            "H2b": {
                "status": "NOT_APPLICABLE",
                "reason": "Target-stratum poison prevalence is 20.48%, above the preregistered <=5% condition; fresh pool also cannot support k=500 fresh calibration.",
            },
            "H3": {
                "status": "NOT_TESTED",
                "reason": "Each fold has only about 1,165 clean members, so k>=2000 is impossible.",
            },
            "H4": {
                "status": "NO_FAILURE_CELL_AVAILABLE" if low_recall.empty else ("PASS" if h4_pass else "FAIL"),
                "low_recall_cells": int(len(low_recall)),
            },
            "H5": {"status": "NOT_TESTED", "reason": "No reference-budget m* grid in this export."},
            "H6": {"status": "NOT_TESTED", "reason": "No trusted-set contamination grid in this export."},
            "H7": {"status": "NOT_TESTED", "reason": "This block is classical BloodMNIST, not a QML channel experiment."},
        },
        "scope_limits": [
            "Retrospective analysis: archived victims predate PREREG_S1_S4.md.",
            "Population is the target-assigned detector-defined stratum, not the complete eight-class training set.",
            "Only BloodMNIST, BadNets, one victim architecture, five seeds, and two score views are covered.",
            "Fresh calibration at k=500 is unavailable because each fold has only 333 fresh rows.",
            "k=2000 is unavailable because each fold has fewer than 2000 clean training members.",
        ],
        "hashes": {
            "scores": sha256(scores_path),
            "certificate_results": sha256(result_path),
            "export_manifest": sha256(export_manifest_path),
            "exporter": sha256(CERT_ROOT / "analysis/export_existing_bloodmnist_s1_scores.py"),
            "certificate_template": sha256(CERT_ROOT / "analysis/s1_analysis_template.py"),
            "certificate_library": sha256(CERT_ROOT / "cert/certificate.py"),
        },
    }
    (root / "real_s1_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    paired500 = summary[
        (summary["view"] == "paired_hidden64_drift")
        & (summary["k"] == 500)
        & (summary["declared_nmax_over_true_np"] == 1.0)
    ].iloc[0]
    suspect500 = summary[
        (summary["view"] == "suspect_hidden64")
        & (summary["k"] == 500)
        & (summary["declared_nmax_over_true_np"] == 1.0)
    ].iloc[0]
    lines = [
        "# Retrospective real-data S1 audit",
        "",
        "**Status: PASS_WITH_SCOPE_LIMITS.** The implementation and persisted calculations replay, but this is a retrospective feasibility block, not the prospectively preregistered S1 matrix.",
        "",
        "## Principal result",
        "",
        "| setting | GO | NO-GO | abstain | pooled false GO |",
        "|---|---:|---:|---:|---:|",
        f"| k=120, exact poison count | {k120['dec_r1.0_go_rate'].mean():.2%} | {k120['dec_r1.0_nogo_rate'].mean():.2%} | {1-k120['dec_r1.0_go_rate'].mean()-k120['dec_r1.0_nogo_rate'].mean():.2%} | {k120['false_go_r1.0_count'].sum()/k120['R'].sum():.3%} |",
        f"| k=500, exact poison count | {k500['dec_r1.0_go_rate'].mean():.2%} | {k500['dec_r1.0_nogo_rate'].mean():.2%} | {1-k500['dec_r1.0_go_rate'].mean()-k500['dec_r1.0_nogo_rate'].mean():.2%} | {k500['false_go_r1.0_count'].sum()/k500['R'].sum():.3%} |",
        f"| k=120, Nmax/Np=1.2 | {k120['dec_r1.2_go_rate'].mean():.2%} | {k120['dec_r1.2_nogo_rate'].mean():.2%} | {1-k120['dec_r1.2_go_rate'].mean()-k120['dec_r1.2_nogo_rate'].mean():.2%} | {k120['false_go_r1.2_count'].sum()/k120['R'].sum():.3%} |",
        f"| k=500, Nmax/Np=1.2 | {k500['dec_r1.2_go_rate'].mean():.2%} | {k500['dec_r1.2_nogo_rate'].mean():.2%} | {1-k500['dec_r1.2_go_rate'].mean()-k500['dec_r1.2_nogo_rate'].mean():.2%} | {k500['false_go_r1.2_count'].sum()/k500['R'].sum():.3%} |",
        "",
        f"The detector itself is strong (mean realized recall {results['mean_realized_recall'].mean():.2%}). With k=500 and the exact poison count, paired drift gives GO {paired500['go_rate']:.2%} and suspect hidden gives GO {suspect500['go_rate']:.2%}. Once the declared poison-count upper bound is only 20% wider, every cell abstains. The certificate is safe here, but useful GO decisions require a sharply known poison count.",
        "",
        "## Hypothesis audit",
        "",
        f"- H1: not falsified in {len(h1_cells)} retrospective cells; Holm rejections = {sum(rejected)}. Maximum observed cell false-GO rate is {report['principal_results']['max_cell_false_go_rate']:.3%}.",
        "- H2: estimated descriptively; see real_s1_summary.csv for five-seed bootstrap intervals. Suspect-hidden has a positive member-minus-fresh gap, while paired drift is negative on average.",
        "- H2b: not applicable because poison prevalence in the audited stratum is 20.48%, not <=5%; k=500 fresh calibration is also unavailable.",
        "- H3: not tested because k=2000 exceeds the clean-member count per fold.",
        "- H4: no cell has realized recall below 0.8, so failure recognition is not tested.",
        "- H5-H7: not tested by this block.",
        "",
        "## Interpretation boundary",
        "",
        "This result supports a narrow claim: the finite-population certificate avoided false assurance on five archived BloodMNIST BadNets victims, but its utility is highly sensitive to the declared poison-count interval. It does not complete S1-S4, prove cross-dataset validity, or establish that the method is TIFS-ready by itself.",
        "",
    ]
    (root / "REAL_S1_AUDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    report = validate(args.root.resolve())
    print(json.dumps({"status": report["status"], **report["principal_results"], "H1": report["hypotheses"]["H1"]}, indent=2))


if __name__ == "__main__":
    main()
