#!/usr/bin/env python3
"""Defender-only A1/A2 member-plan preparation.

This is the only execution component allowed to read ``private_truth.csv``.
The emitted plan uses opaque pool names, per-draw HMAC tokens, and a commitment
to a 32-byte defender key.  The key itself is never written to the plan.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CERT_ROOT = ROOT / "auditability_certificate_v1"
CONFIG = ROOT / "configs/cert_aware_redteam_multiblock_v1.json"
FREEZE = CERT_ROOT / "CERT_AWARE_REDTEAM_MULTIBLOCK_FREEZE_V1.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def token(key: bytes, *values: Any) -> str:
    message = "|".join(map(str, values)).encode("utf-8")
    return hmac.new(key, message, hashlib.sha256).hexdigest()[:32]


def prepare(source: Path, key_file: Path, output: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "SCORE_HANDOFF_COMPLETE" or manifest.get("unblinded") is not False:
        raise RuntimeError("member planning requires a sealed score handoff")
    if manifest.get("redteam_freeze_manifest_sha256") != sha256(FREEZE):
        raise RuntimeError("score handoff is not bound to the red-team freeze")
    if float(manifest.get("victim_training_started_unix", 0)) < float(freeze["created_unix"]):
        raise RuntimeError("victim training predates the operational freeze")
    scores_path = source / "blind_scores.csv"
    truth_path = source / "private_truth.csv"
    if sha256(scores_path) != manifest.get("blind_scores_sha256") or sha256(truth_path) != manifest.get("private_truth_sha256"):
        raise RuntimeError("source handoff hash mismatch")
    key = key_file.read_bytes()
    if len(key) != 32:
        raise RuntimeError("defender key must contain exactly 32 bytes")
    scores = pd.read_csv(scores_path, usecols=["victim_id", "detector", "fold", "sample_id", "role"])
    truth = pd.read_csv(truth_path)
    joined = scores.merge(truth, on=["victim_id", "sample_id", "role"], how="left", validate="many_to_one")
    if joined["is_poison"].isna().any():
        raise RuntimeError("score/truth join is incomplete")
    groups = []
    for (victim_id, detector, fold), group in joined.groupby(["victim_id", "detector", "fold"], sort=True):
        train = group.loc[group["role"] == "train"]
        pool_a = sorted(map(int, train.loc[train["is_poison"] == 0, "sample_id"]))
        pool_b = sorted(map(int, train.loc[train["is_poison"] == 1, "sample_id"]))
        draw_tokens = {}
        for k in config["certificate"]["trusted_member_budgets"]:
            draw_tokens[str(k)] = [
                token(key, victim_id, detector, int(fold), int(k), draw)
                for draw in range(int(config["certificate"]["draws_per_cell"]))
            ]
        groups.append({
            "victim_id": victim_id,
            "detector": detector,
            "fold": int(fold),
            "pool_a_sample_ids": pool_a,
            "pool_b_sample_ids": pool_b,
            "draw_tokens": draw_tokens,
        })
    plan = {
        "status": "DEFENDER_MEMBER_PLAN_SEALED",
        "created_unix": time.time(),
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "blind_scores_sha256": manifest["blind_scores_sha256"],
        "private_truth_commitment_sha256": manifest["private_truth_sha256"],
        "freeze_manifest_sha256": sha256(FREEZE),
        "config_sha256": sha256(CONFIG),
        "defender_key_commitment_sha256": hashlib.sha256(key).hexdigest(),
        "pool_semantics_disclosed_to_issuer": False,
        "groups": groups,
    }
    atomic_json(output, plan)
    print(json.dumps({
        "status": plan["status"],
        "groups": len(groups),
        "defender_key_commitment_sha256": plan["defender_key_commitment_sha256"],
        "plan_sha256": sha256(output),
    }, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.key_file, args.output)


if __name__ == "__main__":
    main()
