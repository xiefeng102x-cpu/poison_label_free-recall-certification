#!/usr/bin/env python3
"""Compatibility-shim verifier for the frozen P6-D v4 output bundle.

The pre-registered verifier correctly implements all replay checks but calls a
JSON-hash helper that the imported v3 compatibility module does not export.
This post-hoc shim supplies only that pure helper and then invokes the frozen
verifier unchanged.  It never writes to, repairs, or re-runs the experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import verify_p6d_final_observable_adaptive_v4 as frozen_v4


def stable_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    encoded = (encoded + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_bundle(output: Path) -> dict[str, Any]:
    # Supply the missing interface without modifying either frozen source file.
    frozen_v4.v3_verifier.stable_json_sha256 = stable_json_sha256
    result = frozen_v4.verify_bundle(output)
    result["verification_mode"] = "posthoc_compatibility_shim"
    result["frozen_verifier_modified"] = False
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=frozen_v4.DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify_bundle(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
