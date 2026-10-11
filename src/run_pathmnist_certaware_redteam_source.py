#!/usr/bin/env python3
"""Generate the frozen, still-blinded victim score handoff for A1/A2."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import run_pathmnist_blind_adaptive_multiscore as adaptive
import run_pathmnist_blind_e2e_multidetector as base
import pathmnist_official_resnet18 as architecture


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pathmnist_certaware_redteam_source_v2.json"
PREREG = ROOT / "auditability_certificate_v1/CERT_AWARE_REDTEAM_OPERATIONAL_AMENDMENT_V2.md"
FREEZE = ROOT / "auditability_certificate_v1/CERT_AWARE_REDTEAM_MULTIBLOCK_FREEZE_V2.json"
OUTPUT = ROOT / "results/pathmnist_certaware_redteam_source_v2"
SEEDS = (5301, 5302, 5303, 5304, 5305)
WEIGHTS = {5301: "resnet18_28_1.pth", 5302: "resnet18_28_2.pth", 5303: "resnet18_28_1.pth", 5304: "resnet18_28_2.pth", 5305: "resnet18_28_1.pth"}
ACK = "OPEN_CERT_AWARE_REDTEAM_V2_SEEDS_5301_5305"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_freeze() -> dict[str, Any]:
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    hashes = frozen.get("hashes", {})
    expected = {
        "source_config_sha256": base.sha256(CONFIG),
        "source_runner_sha256": base.sha256(Path(__file__)),
        "adaptive_training_runner_sha256": base.sha256(Path(adaptive.__file__)),
        "base_score_runner_sha256": base.sha256(Path(base.__file__)),
        "architecture_sha256": base.sha256(Path(architecture.__file__)),
    }
    if frozen.get("status") != "FROZEN_BEFORE_FORMAL_VICTIM_TRAINING":
        raise RuntimeError("red-team source freeze is not active")
    for key, value in expected.items():
        if hashes.get(key) != value:
            raise RuntimeError(f"red-team source freeze mismatch: {key}")
    return frozen


def configure() -> None:
    adaptive.CONFIG = CONFIG
    adaptive.PREREG = PREREG
    adaptive.FREEZE = FREEZE
    adaptive.DEFAULT_OUTPUT = OUTPUT
    adaptive.SEEDS = SEEDS
    adaptive.WEIGHT_MAP = WEIGHTS
    adaptive.ACK = ACK
    adaptive.verify_freeze = verify_freeze
    adaptive.configure_base()


def run(dataset: Path, output: Path, *, resume: bool, ack: str | None) -> dict[str, Any]:
    verify_freeze()
    if ack != ACK:
        raise RuntimeError(f"formal execution requires --ack {ACK}")
    # Keep the start receipt beside (not inside) the output directory because the
    # inherited formal runner requires a completely empty destination on first run.
    start_path = output.with_name(output.name + "_formal_training_start.json")
    if start_path.exists():
        start = json.loads(start_path.read_text(encoding="utf-8"))
        if not resume:
            raise RuntimeError("formal source already started; pass --resume")
    else:
        start = {
            "status": "FORMAL_TRAINING_STARTED_AFTER_FREEZE",
            "started_unix": time.time(),
            "freeze_manifest_sha256": base.sha256(FREEZE),
            "source_runner_sha256": base.sha256(Path(__file__)),
        }
        if start["started_unix"] < float(verify_freeze()["created_unix"]):
            raise RuntimeError("training start predates freeze")
        atomic_json(start_path, start)
    configure()
    manifest = adaptive.run_formal(dataset, output, resume=resume, ack=ack)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "experiment": "pathmnist_certaware_redteam_source_v2",
        "redteam_freeze_manifest_sha256": base.sha256(FREEZE),
        "victim_training_started_unix": float(start["started_unix"]),
        "formal_training_start_sha256": base.sha256(start_path),
        "redteam_source_runner_sha256": base.sha256(Path(__file__)),
    })
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--dataset", type=Path, default=base.DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ack")
    args = parser.parse_args()
    if not args.formal:
        parser.error("formal execution requires --formal")
    result = run(args.dataset, args.output, resume=args.resume, ack=args.ack)
    print(json.dumps({"status": result["status"], "completed_seeds": result["completed_seeds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
