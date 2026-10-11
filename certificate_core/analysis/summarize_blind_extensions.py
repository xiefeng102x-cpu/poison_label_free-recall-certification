#!/usr/bin/env python3
"""Create a compact, reproducible summary of the two blind extension blocks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "auditability_certificate_v1/results_extension_v1/blind_strong_adaptive_summary.json"
BLOCKS = {
    "strong_resnet50": ROOT / "results/pathmnist_blind_strong_resnet50_certificate_v1",
    "adaptive_multiscore": ROOT / "results/pathmnist_blind_adaptive_multiscore_certificate_v1",
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def summarize_block(path: Path) -> dict[str, Any]:
    receipts = json.loads((path / "seed_receipts.json").read_text(encoding="utf-8"))
    conventional = pd.read_csv(path / "conventional_detector_metrics.csv")
    familywise = pd.read_csv(path / "certificate_familywise_summary.csv")
    family_report = json.loads((path / "certificate_familywise_report.json").read_text(encoding="utf-8"))
    verification = json.loads((path / "independent_verification.json").read_text(encoding="utf-8"))
    detector_rows = []
    for detector, group in conventional.groupby("detector", sort=True):
        detector_rows.append(
            {
                "detector": detector,
                "pooled_tpr": float(group["tp"].sum() / group["n_poison"].sum()),
                "pooled_target_clean_fpr": float(
                    group["fp_target_clean"].sum() / group["n_target_clean"].sum()
                ),
            }
        )
    bca = [float(row["victim_metrics"]["balanced_accuracy"]) for row in receipts]
    asr = [float(row["victim_metrics"]["asr"]) for row in receipts]
    primary = familywise.loc[familywise["interval"] == "primary"].sort_values("k")
    return {
        "status": "COMPLETE" if verification.get("status") == "PASS" else "INVALID",
        "seeds": [int(row["seed"]) for row in receipts],
        "eligible_victims": int(sum(bool(row["eligible"]) for row in receipts)),
        "victim_count": len(receipts),
        "balanced_accuracy": {
            "mean": sum(bca) / len(bca),
            "min": min(bca),
            "max": max(bca),
        },
        "attack_success_rate": {
            "mean": sum(asr) / len(asr),
            "min": min(asr),
            "max": max(asr),
        },
        "conventional_detectors": detector_rows,
        "familywise_primary_by_budget": primary[
            ["k", "go_rate", "nogo_rate", "abstain_rate", "valid_false_go_count", "valid_false_go_rate"]
        ].to_dict(orient="records"),
        "familywise_valid_false_go_count_all_intervals": int(
            family_report["valid_false_go_count"]
        ),
        "familywise_valid_rows_all_intervals": int(family_report["valid_rows"]),
        "independent_verification": verification,
    }


def main() -> None:
    summary = {name: summarize_block(path) for name, path in BLOCKS.items()}
    summary["interpretation_rule"] = (
        "A zero false-GO count supports safety only at the frozen operating point; "
        "GO/ABSTAIN rates quantify utility and are not interchangeable with safety."
    )
    atomic_json(OUTPUT, summary)
    markdown = OUTPUT.with_suffix(".md")
    lines = ["# Blind strong-architecture and adaptive extension summary", ""]
    for name in ("strong_resnet50", "adaptive_multiscore"):
        row = summary[name]
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Eligible victims: {row['eligible_victims']}/{row['victim_count']}",
                f"- Mean BCA: {100 * row['balanced_accuracy']['mean']:.2f}%",
                f"- Mean ASR: {100 * row['attack_success_rate']['mean']:.2f}%",
                f"- Familywise valid false GO: {row['familywise_valid_false_go_count_all_intervals']}/{row['familywise_valid_rows_all_intervals']}",
                "",
                "| k | GO | ABSTAIN | false GO |",
                "|---:|---:|---:|---:|",
            ]
        )
        for item in row["familywise_primary_by_budget"]:
            lines.append(
                f"| {int(item['k'])} | {100*item['go_rate']:.2f}% | "
                f"{100*item['abstain_rate']:.2f}% | {int(item['valid_false_go_count'])} |"
            )
        lines.append("")
    tmp = markdown.with_name(markdown.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, markdown)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
