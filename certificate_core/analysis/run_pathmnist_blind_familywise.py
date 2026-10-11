#!/usr/bin/env python3
"""Pre-outcome familywise sensitivity for the five-detector blind block."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CERT_ROOT = ROOT / "auditability_certificate_v1"
import sys
sys.path.insert(0, str(CERT_ROOT))
from cert.certificate import certify_realized_counts

DEFAULT_RESULTS = ROOT / "results/pathmnist_blind_e2e_multidetector_certificate_v1"
PREREG = CERT_ROOT / "PREREG_BLIND_E2E_FAMILYWISE_V1.md"
FREEZE = CERT_ROOT / "BLIND_E2E_FAMILYWISE_FREEZE_V1.json"
DELTA = 0.005
TAU = 0.8
CAP = 0.1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("empty familywise output")
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run(results: Path) -> dict[str, Any]:
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    expected = {"prereg_sha256": sha256(PREREG), "runner_sha256": sha256(Path(__file__))}
    if frozen.get("status") != "FROZEN_BEFORE_OUTCOME_INSPECTION" or frozen.get("hashes") != expected:
        raise RuntimeError("familywise freeze mismatch")
    decisions_path = results / "certificate_decisions_blind.csv"
    evaluated_path = results / "certificate_decisions_evaluated.csv"
    decisions = pd.read_csv(decisions_path)
    evaluated = pd.read_csv(evaluated_path)
    keys = ["victim_id", "detector", "fold", "k", "draw", "interval"]
    truth_columns = keys + ["actual_recall", "actual_clean_flag_rate", "declared_interval_valid"]
    merged = decisions.merge(evaluated[truth_columns], on=keys, validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        result = certify_realized_counts(
            int(row.k), int(row.conformal_rank), int(row.flagged_count), int(row.candidate_count),
            tau=TAU, n_poison_min=int(row.n_poison_min), n_poison_max=int(row.n_poison_max),
            delta=DELTA, alpha_cap=CAP, fpr_mode="tight",
        )
        code = int(result["decision"])
        decision = {1: "GO", -1: "NO_GO", 0: "ABSTAIN"}[code]
        false_go = bool(code == 1 and (float(row.actual_recall) < TAU or float(row.actual_clean_flag_rate) > CAP))
        rows.append({
            "victim_id": row.victim_id,
            "detector": row.detector,
            "fold": int(row.fold),
            "k": int(row.k),
            "draw": int(row.draw),
            "interval": row.interval,
            "delta_family": 0.05,
            "detectors": 5,
            "folds": 2,
            "delta_per_detector_fold": DELTA,
            "decision": decision,
            "recall_lo": float(result["recall_lo"]),
            "recall_hi": float(result["recall_hi"]),
            "clean_flag_rate_hi": float(result["clean_flag_rate_hi"]),
            "declared_interval_valid": int(row.declared_interval_valid),
            "actual_recall": float(row.actual_recall),
            "actual_clean_flag_rate": float(row.actual_clean_flag_rate),
            "false_go": int(false_go),
            "valid_false_go": int(false_go and bool(row.declared_interval_valid)),
        })
    out = results / "certificate_familywise_evaluated.csv"
    atomic_csv(out, rows)
    frame = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for (k, interval), group in frame.groupby(["k", "interval"], sort=True):
        valid = group[group["declared_interval_valid"] == 1]
        summary_rows.append({
            "k": int(k),
            "interval": interval,
            "rows": int(len(group)),
            "valid_rows": int(len(valid)),
            "go_rate": float((group["decision"] == "GO").mean()),
            "nogo_rate": float((group["decision"] == "NO_GO").mean()),
            "abstain_rate": float((group["decision"] == "ABSTAIN").mean()),
            "valid_false_go_count": int(valid["valid_false_go"].sum()),
            "valid_false_go_rate": float(valid["valid_false_go"].mean()) if len(valid) else float("nan"),
        })
    summary_path = results / "certificate_familywise_summary.csv"
    atomic_csv(summary_path, summary_rows)
    report = {
        "status": "COMPLETE_FAMILYWISE_SENSITIVITY",
        "preregistered_before_outcome_inspection": True,
        "delta_family": 0.05,
        "delta_per_detector_fold": DELTA,
        "evaluated_sha256": sha256(out),
        "summary_sha256": sha256(summary_path),
        "valid_false_go_count": int(frame.loc[frame["declared_interval_valid"] == 1, "valid_false_go"].sum()),
        "valid_rows": int((frame["declared_interval_valid"] == 1).sum()),
    }
    atomic_json(results / "certificate_familywise_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(); run(args.results)


if __name__ == "__main__":
    main()
