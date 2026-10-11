#!/usr/bin/env python3
"""Post-seal independent verifier for the prospective protocol-correct v2 block."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_protocol_correct_rerun as base
import prospective_protocol_v2_common as frozen
from protocol_correct_common import atomic_json, load_key, sha256


def configure(source: Path, key_file: Path) -> None:
    key = load_key(key_file)
    original = frozen.verify_freeze(key=key)
    source_manifest = frozen.verify_source_handoff()

    def verify_frozen_files(_: bytes) -> dict:
        issuer_sources = [frozen.ISSUER_WRAPPER, frozen.BASE_ISSUER]
        for issuer_path in issuer_sources:
            text = issuer_path.read_text(encoding="utf-8")
            for banned in ("private_truth", "is_poison", "pool_role", "selected_clean", "selected_poison"):
                if banned in text:
                    raise RuntimeError(f"issuer contains prohibited identifier: {banned}")
        augmented = deepcopy(original)
        augmented["hashes"]["path_score_sha256"] = source_manifest["blind_scores_sha256"]
        augmented["hashes"]["path_evaluation_sha256"] = source_manifest["private_truth_sha256"]
        return augmented

    base.CONFIG = frozen.CONFIG
    base.FREEZE = frozen.FREEZE
    base.REGISTRAR = frozen.REGISTRAR_WRAPPER
    base.ISSUER = frozen.ISSUER_WRAPPER
    base.COMMON = frozen.PROTOCOL_COMMON
    base.__file__ = __file__
    base.verify_frozen_files = verify_frozen_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=frozen.SOURCE)
    parser.add_argument("--output", type=Path, default=frozen.OUTPUT)
    parser.add_argument("--key-file", type=Path, default=frozen.KEY_FILE)
    args = parser.parse_args()
    if args.source != frozen.SOURCE or args.output != frozen.OUTPUT or args.key_file != frozen.KEY_FILE:
        raise RuntimeError("prospective verifier paths are frozen")
    configure(args.source, args.key_file)
    report = base.verify("path", args.source, args.output, args.key_file)
    report["scope"] = "prospective victim training through protocol-correct post-seal verification"
    report["prospective_from_victim_training"] = True
    report["freeze_status"] = "FROZEN_BEFORE_VICTIM_TRAINING_AND_CERTIFICATE_ISSUANCE"
    report["source_manifest_sha256"] = sha256(args.source / "manifest.json")
    report_path = args.output / "verified/verification_report.json"
    atomic_json(report_path, report)
    final = {
        "status": report["status"],
        "protocol": report["protocol"],
        "prospective_from_victim_training": True,
        "issuer_seal_preserved": True,
        "verification_report_sha256": sha256(report_path),
        "verified_unix": report["verified_unix"],
    }
    atomic_json(args.output / "verification_manifest.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
