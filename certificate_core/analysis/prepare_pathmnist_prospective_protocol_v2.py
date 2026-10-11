#!/usr/bin/env python3
"""Truth-aware registrar for the prospective protocol-correct v2 block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_protocol_correct_observations as base
import prospective_protocol_v2_common as frozen
from protocol_correct_common import atomic_json, load_key, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=frozen.SOURCE)
    parser.add_argument("--output", type=Path, default=frozen.OUTPUT)
    parser.add_argument("--key-file", type=Path, default=frozen.KEY_FILE)
    args = parser.parse_args()
    if args.source != frozen.SOURCE or args.output != frozen.OUTPUT or args.key_file != frozen.KEY_FILE:
        raise RuntimeError("prospective registrar paths are frozen")
    key = load_key(args.key_file)
    freeze = frozen.verify_freeze(key=key)
    source_manifest = frozen.verify_source_handoff()
    base.CONFIG = frozen.CONFIG
    base.__file__ = __file__
    result = base.run("path", args.source, args.output, args.key_file)
    public_path = args.output / "public/issuer_manifest.json"
    private_path = args.output / "private/registrar_manifest.json"
    public = json.loads(public_path.read_text(encoding="utf-8"))
    private = json.loads(private_path.read_text(encoding="utf-8"))
    additions = {
        "prospective": True,
        "freeze_manifest_sha256": sha256(frozen.FREEZE),
        "pretraining_key_commitment_sha256": freeze["key_commitment_sha256"],
        "source_score_handoff_sha256": source_manifest["blind_scores_sha256"],
    }
    public.update(additions)
    private.update(additions)
    atomic_json(public_path, public)
    atomic_json(private_path, private)
    print(json.dumps({"status": result["status"], **additions}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
