#!/usr/bin/env python3
"""Truth-blind six-field issuer for the prospective protocol-correct v2 block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue_protocol_correct_certificates as base
import prospective_protocol_v2_common as frozen
from protocol_correct_common import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=frozen.OUTPUT / "public")
    parser.add_argument("--output", type=Path, default=frozen.OUTPUT / "sealed")
    args = parser.parse_args()
    if args.input != frozen.OUTPUT / "public" or args.output != frozen.OUTPUT / "sealed":
        raise RuntimeError("prospective issuer paths are frozen")
    freeze = frozen.verify_freeze()
    manifest_path = args.input / "issuer_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("freeze_manifest_sha256") != sha256(frozen.FREEZE):
        raise RuntimeError("public issuer manifest is not bound to the prospective freeze")
    if manifest.get("key_commitment_sha256") != freeze.get("key_commitment_sha256"):
        raise RuntimeError("public key commitment mismatch")
    if manifest.get("allowed_schema") != base.INPUT_FIELDS:
        raise RuntimeError("issuer schema is not the exact six-field allow-list")
    base.CONFIG = frozen.CONFIG
    base.__file__ = __file__
    base.issue(args.input, args.output)


if __name__ == "__main__":
    main()
