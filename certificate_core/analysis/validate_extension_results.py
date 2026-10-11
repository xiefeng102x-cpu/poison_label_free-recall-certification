#!/usr/bin/env python3
"""Validate retrospective cross-dataset/cross-attack certificate outputs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1] / "results_extension_v1"
BLOCKS = (
    "adaptive_cfcmdu",
    "adaptive_global_baselines",
    "bloodmnist_cross_attack",
    "cifar10_crossfit",
    "cifar10_global_baseline",
)
RATIOS = (1.0, 1.2, 2.0, 4.0)
TAU = 0.8
DELTA = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def holm(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    order = np.argsort(pvalues)
    rejected = np.zeros(len(pvalues), dtype=bool)
    for rank, index in enumerate(order):
        if pvalues[index] <= alpha / (len(pvalues) - rank):
            rejected[index] = True
        else:
            break
    return rejected.tolist()


def parse_id(value: str) -> tuple[str, str, str, str, str]:
    parts = value.split("|")
    if len(parts) != 5:
        raise ValueError(f"unexpected victim id: {value}")
    return tuple(parts)  # type: ignore[return-value]


def main() -> None:
    manifest = json.loads((ROOT / "export_manifest.json").read_text())
    all_results: list[pd.DataFrame] = []
    mode_rows: list[dict] = []
    input_checks: dict[str, dict] = {}
    for block in BLOCKS:
        score_path = ROOT / f"{block}_scores.csv"
        loose_path = ROOT / f"{block}_certificate.csv"
        result_path = ROOT / f"{block}_tight_certificate.csv"
        if not result_path.exists() or result_path.stat().st_size == 0:
            raise RuntimeError(f"missing result: {result_path}")
        scores = pd.read_csv(score_path)
        loose = pd.read_csv(loose_path)
        results = pd.read_csv(result_path)
        expected = manifest["outputs"][block]
        if sha256(score_path) != expected["sha256"]:
            raise RuntimeError(f"score hash changed: {block}")
        if scores.duplicated(["victim_id", "fold", "sample_id"]).any():
            raise RuntimeError(f"duplicate score key: {block}")
        if not np.isfinite(scores["score"]).all():
            raise RuntimeError(f"non-finite score: {block}")
        if not np.isfinite(results.select_dtypes(include=[np.number])).all().all():
            raise RuntimeError(f"non-finite certificate output: {block}")
        keys = ["victim_id", "fold", "k"]
        paired = loose.merge(results, on=keys, suffixes=("_loose", "_tight"), validate="one_to_one")
        for invariant in ("mean_realized_recall", "mean_caught_true", "mean_caught_lo_r1.0"):
            if not np.allclose(paired[f"{invariant}_loose"], paired[f"{invariant}_tight"]):
                raise RuntimeError(f"tight mode changed an invariant: {block}/{invariant}")
        for ratio in RATIOS:
            for mode in ("loose", "tight"):
                mode_rows.append(
                    {
                        "block": block,
                        "mode": mode,
                        "ratio": ratio,
                        "go_rate": float(paired[f"dec_r{ratio}_go_rate_{mode}"].mean()),
                        "nogo_rate": float(paired[f"dec_r{ratio}_nogo_rate_{mode}"].mean()),
                        "false_go_rate": float(
                            paired[f"false_go_r{ratio}_count_{mode}"].sum()
                            / paired[f"R_{mode}"].sum()
                        ),
                    }
                )
        for ratio in RATIOS:
            count = results[f"false_go_r{ratio}_count"]
            rate = results[f"false_go_r{ratio}_rate"]
            if not np.allclose(count / results["R"], rate, atol=1e-15):
                raise RuntimeError(f"false-GO mismatch: {block}, ratio={ratio}")
        parsed = results["victim_id"].map(parse_id)
        results = results.copy()
        results["dataset"] = parsed.map(lambda x: x[0])
        results["attack"] = parsed.map(lambda x: x[1])
        results["method"] = parsed.map(lambda x: x[2])
        results["view"] = parsed.map(lambda x: x[3])
        results["seed"] = parsed.map(lambda x: int(x[4].replace("seed", "")))
        results["block"] = block
        all_results.append(results)
        input_checks[block] = {
            "score_rows": int(len(scores)),
            "result_rows": int(len(results)),
            "victims": int(scores["victim_id"].nunique()),
            "folds": int(scores.groupby("victim_id")["fold"].nunique().max()),
            "score_sha256": sha256(score_path),
            "loose_result_sha256": sha256(loose_path),
            "tight_result_sha256": sha256(result_path),
        }

    results = pd.concat(all_results, ignore_index=True)
    mode_comparison = pd.DataFrame(mode_rows)
    mode_comparison.to_csv(ROOT / "fpr_mode_comparison.csv", index=False)

    h1_rows: list[dict] = []
    for _, row in results.iterrows():
        folds = int(input_checks[row["block"]]["folds"])
        null_rate = DELTA / folds
        for ratio in RATIOS:
            count = int(row[f"false_go_r{ratio}_count"])
            repetitions = int(row["R"])
            h1_rows.append(
                {
                    "block": row["block"],
                    "victim_id": row["victim_id"],
                    "fold": int(row["fold"]),
                    "k": int(row["k"]),
                    "ratio": ratio,
                    "false_go_count": count,
                    "R": repetitions,
                    "delta_over_folds": null_rate,
                    "pvalue_greater": float(
                        binomtest(count, repetitions, null_rate, alternative="greater").pvalue
                    ),
                }
            )
    h1 = pd.DataFrame(h1_rows)
    h1["holm_reject"] = holm(h1["pvalue_greater"].tolist())
    h1.to_csv(ROOT / "h1_extension_cell_tests.csv", index=False)

    coverage_rows: list[dict] = []
    for keys, group in results.groupby(["dataset", "attack", "method"], sort=True):
        for ratio in RATIOS:
            total = int(group["R"].sum())
            go = float(np.average(group[f"dec_r{ratio}_go_rate"], weights=group["R"]))
            nogo = float(np.average(group[f"dec_r{ratio}_nogo_rate"], weights=group["R"]))
            false_go = int(group[f"false_go_r{ratio}_count"].sum()) / total
            coverage_rows.append(
                {
                    "dataset": keys[0],
                    "attack": keys[1],
                    "method": keys[2],
                    "ratio": ratio,
                    "victims": int(group["victim_id"].nunique()),
                    "seeds": int(group["seed"].nunique()),
                    "certificate_rows": int(len(group)),
                    "mean_realized_recall": float(group["mean_realized_recall"].mean()),
                    "go_rate": go,
                    "nogo_rate": nogo,
                    "abstain_rate": 1.0 - go - nogo,
                    "false_go_rate": false_go,
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(ROOT / "coverage_summary.csv", index=False)

    h4_rows: list[dict] = []
    weak = results[results["mean_realized_recall"] < TAU].copy()
    for keys, group in weak.groupby(["dataset", "attack", "method"], sort=True):
        for ratio in RATIOS:
            total = int(group["R"].sum())
            go = float(np.average(group[f"dec_r{ratio}_go_rate"], weights=group["R"]))
            nogo = float(np.average(group[f"dec_r{ratio}_nogo_rate"], weights=group["R"]))
            false_go = int(group[f"false_go_r{ratio}_count"].sum()) / total
            h4_rows.append(
                {
                    "dataset": keys[0],
                    "attack": keys[1],
                    "method": keys[2],
                    "ratio": ratio,
                    "low_recall_rows": int(len(group)),
                    "victims": int(group["victim_id"].nunique()),
                    "seeds": int(group["seed"].nunique()),
                    "mean_realized_recall": float(group["mean_realized_recall"].mean()),
                    "minimum_realized_recall": float(group["mean_realized_recall"].min()),
                    "go_rate": go,
                    "nogo_rate": nogo,
                    "abstain_rate": 1.0 - go - nogo,
                    "false_go_rate": false_go,
                    "h4_false_go_rate_le_delta": bool(false_go <= DELTA + 1e-15),
                }
            )
    h4 = pd.DataFrame(h4_rows)
    h4.to_csv(ROOT / "h4_failure_recognition.csv", index=False)

    ratio1_weak = h4[h4["ratio"] == 1.0]
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "PASS_WITH_SCOPE_LIMITS",
        "input_checks": input_checks,
        "coverage": {
            "datasets": sorted(results["dataset"].unique().tolist()),
            "attacks": sorted(results["attack"].unique().tolist()),
            "methods": sorted(results["method"].unique().tolist()),
            "seeds": sorted(results["seed"].unique().tolist()),
            "certificate_rows": int(len(results)),
            "redraws": int((results["R"]).sum()),
        },
        "H1": {
            "cells": int(len(h1)),
            "holm_rejections": int(h1["holm_reject"].sum()),
            "minimum_unadjusted_pvalue": float(h1["pvalue_greater"].min()),
            "maximum_observed_false_go_rate": float(
                max(results[f"false_go_r{ratio}_rate"].max() for ratio in RATIOS)
            ),
        },
        "H4": {
            "low_recall_certificate_rows": int(len(weak)),
            "low_recall_victims": int(weak["victim_id"].nunique()),
            "ratio1_groups": int(len(ratio1_weak)),
            "ratio1_all_false_go_rates_le_delta": bool(
                ratio1_weak["h4_false_go_rate_le_delta"].all()
            ) if len(ratio1_weak) else False,
            "ratio1_max_group_go_rate": float(ratio1_weak["go_rate"].max()) if len(ratio1_weak) else None,
            "ratio1_max_group_false_go_rate": float(ratio1_weak["false_go_rate"].max()) if len(ratio1_weak) else None,
            "ratio1_max_cell_false_go_rate": float(weak["false_go_r1.0_rate"].max()) if len(weak) else None,
            "ratio1_pooled_false_go_rate": float(
                weak["false_go_r1.0_count"].sum() / weak["R"].sum()
            ) if len(weak) else None,
        },
        "limitations": manifest["limitations"] + [
            "The extension is retrospective because all source victims and detector scores predate this certificate handoff.",
            "H4 evidence is restricted to naturally occurring weak archived detector cells, not a prospectively generated failure grid.",
            "Because every trusted-set redraw changes the threshold and realized recall, H4 is evaluated by per-redraw false-GO frequency; grouping rows only by mean recall makes raw GO rate ambiguous.",
            "Dataset coverage is BloodMNIST plus CIFAR-10; GTSRB and PathMNIST are not confirmation-complete.",
        ],
    }
    (ROOT / "extension_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
