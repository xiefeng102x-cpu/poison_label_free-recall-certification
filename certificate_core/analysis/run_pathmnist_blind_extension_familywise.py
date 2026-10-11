#!/usr/bin/env python3
"""Frozen familywise analysis for the strong/adaptive blind extensions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_pathmnist_blind_familywise as base


ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "auditability_certificate_v1/PREREG_BLIND_EXTENSIONS_FAMILYWISE_V1.md"
BLOCKS = {
    "strong": {
        "freeze": ROOT / "auditability_certificate_v1/BLIND_STRONG_RESNET50_FAMILYWISE_FREEZE_V1.json",
        "results": ROOT / "results/pathmnist_blind_strong_resnet50_certificate_v1",
    },
    "adaptive": {
        "freeze": ROOT / "auditability_certificate_v1/BLIND_ADAPTIVE_MULTISCORE_FAMILYWISE_FREEZE_V1.json",
        "results": ROOT / "results/pathmnist_blind_adaptive_multiscore_certificate_v1",
    },
}


def run(block: str, results: Path | None = None):
    selected = BLOCKS[block]
    destination = Path(results) if results is not None else selected["results"]
    base.PREREG = PREREG
    base.FREEZE = selected["freeze"]
    base.DEFAULT_RESULTS = destination
    report = base.run(destination)
    path = destination / "certificate_familywise_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["extension_block"] = block
    base.atomic_json(path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", choices=tuple(BLOCKS), required=True)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    run(args.block, args.results)


if __name__ == "__main__":
    main()
