#!/usr/bin/env python3
"""Independent replay verifier for the strong/adaptive blind extensions."""

from __future__ import annotations

import argparse
from pathlib import Path

import verify_pathmnist_blind_e2e as base


ROOT = Path(__file__).resolve().parents[1]
BLOCKS = {
    "strong": {
        "config": ROOT / "configs/pathmnist_blind_strong_resnet50_certificate_v1.json",
        "freeze": ROOT / "auditability_certificate_v1/BLIND_STRONG_RESNET50_FREEZE_V1.json",
        "results": ROOT / "results/pathmnist_blind_strong_resnet50_certificate_v1",
    },
    "adaptive": {
        "config": ROOT / "configs/pathmnist_blind_adaptive_multiscore_certificate_v1.json",
        "freeze": ROOT / "auditability_certificate_v1/BLIND_ADAPTIVE_MULTISCORE_FREEZE_V1.json",
        "results": ROOT / "results/pathmnist_blind_adaptive_multiscore_certificate_v1",
    },
}


def verify(block: str, results: Path | None = None):
    selected = BLOCKS[block]
    destination = Path(results) if results is not None else selected["results"]
    base.CONFIG = selected["config"]
    base.FREEZE = selected["freeze"]
    base.DEFAULT_RESULTS = destination
    return base.verify(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", choices=tuple(BLOCKS), required=True)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    verify(args.block, args.results)


if __name__ == "__main__":
    main()
