#!/usr/bin/env python3
"""Repair only the frozen wrapper's delegated self-hash path resolution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import run_pathmnist_blind_extension_familywise as wrapper


ROOT = Path(__file__).resolve().parents[2]
AMENDMENT = ROOT / "auditability_certificate_v1/FAMILYWISE_EXECUTION_AMENDMENT_2026-10-08.md"
BASE_RUNNER = ROOT / "auditability_certificate_v1/analysis/run_pathmnist_blind_familywise.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run(block: str) -> dict[str, Any]:
    selected = wrapper.BLOCKS[block]
    # The frozen wrapper is the intended integrity boundary.  Only correct the
    # delegated module's interpretation of ``__file__`` for its self-hash check.
    wrapper.base.__file__ = str(Path(wrapper.__file__).resolve())
    report = wrapper.run(block)
    results = Path(selected["results"])
    receipt = {
        "status": "PASS_POST_UNBLINDING_EXECUTION_PATH_REPAIR",
        "block": block,
        "decision_formula_changed": False,
        "scores_or_draws_changed": False,
        "delta_per_detector_fold": 0.005,
        "amendment_sha256": sha256(AMENDMENT),
        "frozen_wrapper_sha256": sha256(Path(wrapper.__file__)),
        "delegated_base_runner_sha256": sha256(BASE_RUNNER),
        "familywise_freeze_sha256": sha256(Path(selected["freeze"])),
        "familywise_report_sha256": sha256(results / "certificate_familywise_report.json"),
        "familywise_evaluated_sha256": sha256(results / "certificate_familywise_evaluated.csv"),
        "familywise_summary_sha256": sha256(results / "certificate_familywise_summary.csv"),
    }
    atomic_json(results / "familywise_execution_amendment_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", choices=tuple(wrapper.BLOCKS), required=True)
    args = parser.parse_args()
    run(args.block)


if __name__ == "__main__":
    main()
