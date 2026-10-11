#!/usr/bin/env python3
"""Prospective PathMNIST victim/score stage for the protocol-correct v2 block."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "auditability_certificate_v1/analysis"))

import run_pathmnist_blind_e2e_multidetector as base
import prospective_protocol_v2_common as frozen


def configure() -> None:
    base.CONFIG = frozen.CONFIG
    base.PREREG = frozen.PREREG
    base.FREEZE = frozen.FREEZE
    base.DEFAULT_DATASET = frozen.DATASET
    base.DEFAULT_OUTPUT = frozen.SOURCE
    base.SEEDS = frozen.SEEDS
    base.WEIGHT_MAP = frozen.WEIGHT_MAP
    base.ACK = frozen.ACK
    base.engine.PROTOCOL_PATH = frozen.CONFIG
    base.engine.CONFIRMATION_SEEDS = frozen.SEEDS
    base.engine.WEIGHT_MAP = frozen.WEIGHT_MAP
    base.__file__ = __file__
    base.verify_freeze = frozen.verify_freeze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--dataset", type=Path, default=frozen.DATASET)
    parser.add_argument("--output", type=Path, default=frozen.SOURCE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ack")
    args = parser.parse_args()
    if args.dataset != frozen.DATASET or args.output != frozen.SOURCE:
        raise RuntimeError("formal dataset and output paths are frozen")
    configure()
    if args.selftest:
        frozen.verify_freeze(require_dataset=False)
        base.selftest()
    elif args.formal:
        base.run_formal(args.dataset, args.output, resume=args.resume, ack=args.ack)
    else:
        parser.error("choose --selftest or --formal")


if __name__ == "__main__":
    main()
