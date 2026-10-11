#!/usr/bin/env python3
"""Independently validate the two prospective-block certificate handoffs.

This validator does not retrain a model and does not reuse the exporter helper.
It checks the frozen six-column tables, their manifest hashes, the certificate
cell layout, and all reported aggregate decision and false-GO rates.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_prospective_blocks_v1"
AMENDMENT = ROOT / "PREREG_AMENDMENT_03.md"
RATIOS = (1.0, 1.2, 2.0, 4.0)
SCORE_COLUMNS = ("victim_id", "fold", "sample_id", "score", "role", "is_poison")
EXPECTED = {
    "pathmnist": {
        "victims": 5,
        "score_rows": 99_255,
        "train_rows": 86_925,
        "fresh_rows": 12_330,
        "poison_rows": 22_500,
        "ks": (120, 500, 2000),
        "certificate_rows": 30,
        "fresh_available_ks": (120, 500),
    },
    "gtsrb": {
        "victims": 10,
        "score_rows": 41_140,
        "train_rows": 36_140,
        "fresh_rows": 5_000,
        "poison_rows": 22_640,
        "ks": (120, 500),
        "certificate_rows": 40,
        "fresh_available_ks": (120,),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def weighted_rate(frame: pd.DataFrame, column: str) -> float:
    return float(np.average(frame[column], weights=frame["R"]))


def validate_dataset(name: str, manifest: dict) -> dict:
    expected = EXPECTED[name]
    score_path = RESULTS / f"{name}_primary_scores.csv"
    result_path = RESULTS / f"{name}_primary_certificate_tight.csv"
    scores = pd.read_csv(score_path)
    results = pd.read_csv(result_path)

    require(tuple(scores.columns) == SCORE_COLUMNS, f"{name}: six-column schema mismatch")
    require(len(scores) == expected["score_rows"], f"{name}: score row count mismatch")
    require(scores["victim_id"].nunique() == expected["victims"], f"{name}: victim count mismatch")
    require(set(scores["fold"].unique()) == {0, 1}, f"{name}: fold coverage mismatch")
    require(set(scores["role"].unique()) == {"train", "fresh"}, f"{name}: role coverage mismatch")
    require(not scores.duplicated(["victim_id", "fold", "sample_id"]).any(), f"{name}: duplicate key")
    require(np.isfinite(scores["score"]).all(), f"{name}: non-finite score")
    require(scores["is_poison"].isin([0, 1]).all(), f"{name}: invalid poison label")
    require(not ((scores["role"] == "fresh") & (scores["is_poison"] == 1)).any(), f"{name}: poisoned fresh row")
    require(int((scores["role"] == "train").sum()) == expected["train_rows"], f"{name}: train rows mismatch")
    require(int((scores["role"] == "fresh").sum()) == expected["fresh_rows"], f"{name}: fresh rows mismatch")
    require(int(scores["is_poison"].sum()) == expected["poison_rows"], f"{name}: poison rows mismatch")
    require(sha256(score_path) == manifest[name]["score_table_sha256"], f"{name}: score hash mismatch")

    require(len(results) == expected["certificate_rows"], f"{name}: certificate row count mismatch")
    require(set(results["k"].unique()) == set(expected["ks"]), f"{name}: k coverage mismatch")
    require((results["R"] == 200).all(), f"{name}: R mismatch")
    require(
        not results.duplicated(["victim_id", "fold", "k"]).any(),
        f"{name}: duplicate certificate cell",
    )
    require(results["victim_id"].nunique() == expected["victims"], f"{name}: certificate victim count mismatch")
    require(set(results["fold"].unique()) == {0, 1}, f"{name}: certificate fold mismatch")

    for ratio in RATIOS:
        go = results[f"dec_r{ratio}_go_rate"]
        nogo = results[f"dec_r{ratio}_nogo_rate"]
        count = results[f"false_go_r{ratio}_count"]
        rate = results[f"false_go_r{ratio}_rate"]
        require(((go >= 0) & (nogo >= 0) & (go + nogo <= 1 + 1e-12)).all(), f"{name}: invalid decision rates r={ratio}")
        require(np.allclose(count / results["R"], rate, atol=1e-15), f"{name}: false-GO mismatch r={ratio}")

    fresh_expected = results["k"].isin(expected["fresh_available_ks"])
    for ratio in RATIOS:
        fresh_columns = [
            f"fresh_dec_r{ratio}_go_rate",
            f"fresh_dec_r{ratio}_nogo_rate",
            f"fresh_false_go_r{ratio}_count",
            f"fresh_false_go_r{ratio}_rate",
        ]
        require(results.loc[fresh_expected, fresh_columns].notna().all().all(), f"{name}: missing available fresh result r={ratio}")
        require(results.loc[~fresh_expected, fresh_columns].isna().all().all(), f"{name}: unexpected fresh result r={ratio}")
        fresh = results.loc[fresh_expected]
        require(
            np.allclose(
                fresh[f"fresh_false_go_r{ratio}_count"] / fresh["R"],
                fresh[f"fresh_false_go_r{ratio}_rate"],
                atol=1e-15,
            ),
            f"{name}: fresh false-GO mismatch r={ratio}",
        )

    by_k: dict[str, dict] = {}
    for k, group in results.groupby("k", sort=True):
        row: dict[str, object] = {
            "certificate_cells": int(len(group)),
            "redraws": int(group["R"].sum()),
            "mean_realized_recall": float(group["mean_realized_recall"].mean()),
            "mean_member_minus_fresh": float(group["mean_member_minus_fresh"].mean()),
            "member_calibration": {},
            "fresh_calibration": {},
        }
        for ratio in RATIOS:
            go = weighted_rate(group, f"dec_r{ratio}_go_rate")
            nogo = weighted_rate(group, f"dec_r{ratio}_nogo_rate")
            false_count = int(group[f"false_go_r{ratio}_count"].sum())
            row["member_calibration"][str(ratio)] = {
                "go_rate": go,
                "nogo_rate": nogo,
                "abstain_rate": float(1.0 - go - nogo),
                "false_go_count": false_count,
                "false_go_rate": float(false_count / group["R"].sum()),
            }
            fresh_column = f"fresh_dec_r{ratio}_go_rate"
            available = group[fresh_column].notna()
            if available.any():
                fresh = group.loc[available]
                fresh_go = weighted_rate(fresh, fresh_column)
                fresh_nogo = weighted_rate(fresh, f"fresh_dec_r{ratio}_nogo_rate")
                fresh_false_count = int(fresh[f"fresh_false_go_r{ratio}_count"].sum())
                row["fresh_calibration"][str(ratio)] = {
                    "go_rate": fresh_go,
                    "nogo_rate": fresh_nogo,
                    "abstain_rate": float(1.0 - fresh_go - fresh_nogo),
                    "false_go_count": fresh_false_count,
                    "false_go_rate": float(fresh_false_count / fresh["R"].sum()),
                }
            else:
                row["fresh_calibration"][str(ratio)] = None
        by_k[str(int(k))] = row

    return {
        "score_table": {
            "sha256": sha256(score_path),
            "rows": int(len(scores)),
            "victims": int(scores["victim_id"].nunique()),
            "folds": 2,
            "train_rows": int((scores["role"] == "train").sum()),
            "fresh_rows": int((scores["role"] == "fresh").sum()),
            "poison_rows": int(scores["is_poison"].sum()),
        },
        "certificate_table": {
            "sha256": sha256(result_path),
            "rows": int(len(results)),
            "fpr_mode": "tight",
            "R": 200,
        },
        "by_k": by_k,
    }


def main() -> None:
    manifest_path = RESULTS / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["status"] == "READY_FOR_LOCKED_POSTHOC_CERTIFICATE_ANALYSIS", "manifest status mismatch")
    require(manifest["amendment_sha256"] == sha256(AMENDMENT), "amendment hash mismatch")

    datasets = {name: validate_dataset(name, manifest) for name in EXPECTED}
    gtsrb = datasets["gtsrb"]["by_k"]
    pathmnist = datasets["pathmnist"]["by_k"]

    require(
        all(
            gtsrb[k]["member_calibration"][str(ratio)]["go_rate"] == 0.0
            and gtsrb[k]["member_calibration"][str(ratio)]["false_go_count"] == 0
            for k in ("120", "500")
            for ratio in RATIOS
        ),
        "GTSRB failure-recognition invariant failed",
    )
    require(pathmnist["500"]["member_calibration"]["1.0"]["go_rate"] == 1.0, "PathMNIST k=500 exact-count GO mismatch")
    require(pathmnist["2000"]["member_calibration"]["1.0"]["go_rate"] == 1.0, "PathMNIST k=2000 exact-count GO mismatch")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_WITH_SCOPE_LIMITS",
        "manifest_sha256": sha256(manifest_path),
        "amendment_sha256": sha256(AMENDMENT),
        "datasets": datasets,
        "cross_block_checks": {
            "gtsrb_zero_member_calibrated_go_all_k_and_ratios": True,
            "gtsrb_zero_member_calibrated_false_go_all_k_and_ratios": True,
            "pathmnist_correct_upper_bound_go_rate_k500": pathmnist["500"]["member_calibration"]["1.0"]["go_rate"],
            "pathmnist_correct_upper_bound_go_rate_k2000": pathmnist["2000"]["member_calibration"]["1.0"]["go_rate"],
            "pathmnist_fresh_calibration_false_go_rate_k120_correct_upper_bound": pathmnist["120"]["fresh_calibration"]["1.0"]["false_go_rate"],
            "pathmnist_fresh_calibration_false_go_rate_k500_correct_upper_bound": pathmnist["500"]["fresh_calibration"]["1.0"]["false_go_rate"],
        },
        "scope_limits": [
            "Victim/model blocks were prospective, but the six-column certificate handoff was locked after detector endpoints were known.",
            "GTSRB uses the frozen GTSRBTinyCNN-43 block, not a ResNet-18 victim.",
            "Fresh-calibrated results are auxiliary stress tests; fresh pools overlap across redraws and do not replace member calibration.",
            "GTSRB failure recognition is mainly abstention rather than decisive NO-GO.",
        ],
    }
    output = RESULTS / "prospective_certificate_validation.json"
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
