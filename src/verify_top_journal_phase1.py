#!/usr/bin/env python3
"""Replay the frozen BloodMNIST XZ16 fresh-seed confirmation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


PAPER14_ROOT = Path(__file__).resolve().parents[1]
ROOT = PAPER14_ROOT / "results/bloodmnist_xz16_fresh_confirmation_v1"
OUTPUT = PAPER14_ROOT / "results/top_journal_program_v1/phase1_receipt.json"
PRIMARY = "coherent_qml_xz_drift16"
SECONDARY = "coherent_bonferroni_input_qml_xz"


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(left - right) <= tolerance


def gate(row: dict) -> bool:
    return float(row["pooled_tpr"]) >= 0.90 and float(row["pooled_fpr"]) <= 0.05


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol_role"] == "fresh_xz_confirmation"
    assert manifest["seeds"] == list(range(52, 62))
    assert manifest["poison_labels_used_for_fit"] is False
    assert manifest["pre_registered_deployment_rules"] == [PRIMARY, SECONDARY]

    metrics = rows(ROOT / "per_seed_metrics.csv")
    samples = rows(ROOT / "per_sample_scores.csv")
    summaries = {row["representation"]: row for row in rows(ROOT / "pooled_summary.csv")}
    eligible = {
        row["representation"]: row
        for row in rows(ROOT / "victim_eligible_summary.csv")
    }
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in samples:
        grouped[(row["seed"], row["trigger"], row["representation"])].append(row)
    for metric in metrics:
        selected = grouped[
            (metric["seed"], metric["trigger"], metric["representation"])
        ]
        clean = [row for row in selected if int(row["label"]) == 0]
        poison = [row for row in selected if int(row["label"]) == 1]
        fp = sum(int(row["flagged"]) for row in clean)
        tp = sum(int(row["flagged"]) for row in poison)
        assert (tp, len(poison) - tp, fp, len(clean) - fp) == tuple(
            int(metric[key]) for key in ("tp", "fn", "fp", "tn")
        )

    by_representation: dict[str, list[dict]] = defaultdict(list)
    for metric in metrics:
        by_representation[metric["representation"]].append(metric)
    for name, summary in summaries.items():
        selected = by_representation[name]
        counts = tuple(
            sum(int(row[key]) for row in selected) for key in ("tp", "fn", "fp", "tn")
        )
        assert counts == tuple(int(summary[key]) for key in ("tp", "fn", "fp", "tn"))
        assert close(counts[0] / (counts[0] + counts[1]), float(summary["pooled_tpr"]))
        assert close(counts[2] / (counts[2] + counts[3]), float(summary["pooled_fpr"]))

    assert len(metrics) == 70
    assert len(samples) == 46200
    assert tuple(int(summaries[PRIMARY][key]) for key in ("tp", "fn", "fp", "tn")) == (
        569,
        31,
        295,
        5705,
    )
    assert tuple(int(summaries[SECONDARY][key]) for key in ("tp", "fn", "fp", "tn")) == (
        565,
        35,
        261,
        5739,
    )
    assert gate(summaries[PRIMARY]) is True
    assert gate(summaries[SECONDARY]) is True
    assert int(eligible[PRIMARY]["eligible_seed_count"]) == 8
    assert gate(eligible[PRIMARY]) is False
    assert float(eligible[PRIMARY]["pooled_fpr"]) > 0.05
    assert gate(eligible[SECONDARY]) is True

    receipt = {
        "status": "PASS",
        "meaning": "sample-level replay and frozen decision logic passed",
        "phase1_scientific_verdict": "MIXED",
        "all_seed_primary_gate": "PASS",
        "attack_eligible_primary_gate": "FAIL_MARGINAL_FPR",
        "attack_eligible_secondary_gate": "PASS",
        "metric_rows": len(metrics),
        "sample_rows": len(samples),
        "primary_all_seed_tpr": float(summaries[PRIMARY]["pooled_tpr"]),
        "primary_all_seed_fpr": float(summaries[PRIMARY]["pooled_fpr"]),
        "primary_eligible_tpr": float(eligible[PRIMARY]["pooled_tpr"]),
        "primary_eligible_fpr": float(eligible[PRIMARY]["pooled_fpr"]),
        "secondary_eligible_tpr": float(eligible[SECONDARY]["pooled_tpr"]),
        "secondary_eligible_fpr": float(eligible[SECONDARY]["pooled_fpr"]),
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
