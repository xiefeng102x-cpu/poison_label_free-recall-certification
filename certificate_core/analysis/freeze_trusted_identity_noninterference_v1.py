#!/usr/bin/env python3
"""Freeze the trusted-member identity non-interference experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "auditability_certificate_v1"
CONFIG = ROOT / "configs/trusted_identity_noninterference_v1.json"
PREREG = CERT / "PREREG_TRUSTED_IDENTITY_NONINTERFERENCE_V1.md"
RUNNER = CERT / "analysis/run_trusted_identity_noninterference_v1.py"
VERIFIER = CERT / "analysis/verify_trusted_identity_noninterference_v1.py"
FREEZE = CERT / "TRUSTED_IDENTITY_NONINTERFERENCE_FREEZE_V1.json"
SOURCE = ROOT / "results/pathmnist_certaware_redteam_source_v2"
POOL = ROOT / "results/cert_aware_redteam_multiblock_v2/sealed_member_pools.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output"]
    if FREEZE.exists():
        raise RuntimeError(f"freeze already exists: {FREEZE}")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"non-empty formal output already exists: {output}")
    args.key_file.parent.mkdir(parents=True, exist_ok=True)
    if args.key_file.exists():
        raise RuntimeError(f"refusing to reuse key file: {args.key_file}")
    key = secrets.token_bytes(32)
    args.key_file.write_bytes(key)
    args.key_file.chmod(0o600)
    payload = {
        "status": "FROZEN_BEFORE_FORMAL_IDENTITY_DRAW",
        "created_unix": time.time(),
        "target_identities_opened_at_freeze": False,
        "defender_key_sha256": hashlib.sha256(key).hexdigest(),
        "hashes": {
            "config_sha256": sha256(CONFIG),
            "prereg_sha256": sha256(PREREG),
            "runner_sha256": sha256(RUNNER),
            "verifier_sha256": sha256(VERIFIER),
            "source_manifest_sha256": sha256(SOURCE / "manifest.json"),
            "blind_scores_sha256": sha256(SOURCE / "blind_scores.csv"),
            "candidate_pool_sha256": sha256(POOL),
            "freezer_sha256": sha256(Path(__file__)),
        },
    }
    FREEZE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
