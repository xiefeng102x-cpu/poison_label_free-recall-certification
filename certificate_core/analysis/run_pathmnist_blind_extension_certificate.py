#!/usr/bin/env python3
"""Issue and evaluate the frozen strong/adaptive blind certificates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_pathmnist_blind_certificate as base


ROOT = Path(__file__).resolve().parents[2]
BLOCKS = {
    "strong": {
        "config": ROOT / "configs/pathmnist_blind_strong_resnet50_certificate_v1.json",
        "freeze": ROOT / "auditability_certificate_v1/BLIND_STRONG_RESNET50_FREEZE_V1.json",
        "results": ROOT / "results/pathmnist_blind_strong_resnet50_certificate_v1",
        "scope": "PathMNIST, official 28x28 ResNet-50, BadNets, five unseen seeds, five score families; no cross-domain universality claim",
    },
    "adaptive": {
        "config": ROOT / "configs/pathmnist_blind_adaptive_multiscore_certificate_v1.json",
        "freeze": ROOT / "auditability_certificate_v1/BLIND_ADAPTIVE_MULTISCORE_FREEZE_V1.json",
        "results": ROOT / "results/pathmnist_blind_adaptive_multiscore_certificate_v1",
        "scope": "PathMNIST, official 28x28 ResNet-18, certificate-aware multi-score adaptive BadNets, five unseen seeds; no adaptive-attack universality claim",
    },
}


def run(block: str, results: Path | None = None):
    selected = BLOCKS[block]
    destination = Path(results) if results is not None else selected["results"]
    base.CONFIG = selected["config"]
    base.FREEZE = selected["freeze"]
    base.DEFAULT_RESULTS = destination
    report = base.run(destination)
    report_path = destination / "blind_e2e_certificate_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["extension_block"] = block
    report["scope"] = selected["scope"]
    base.atomic_json(report_path, report)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["certificate_report_sha256"] = base.sha256(report_path)
    base.atomic_json(manifest_path, manifest)
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
