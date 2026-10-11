#!/usr/bin/env python3
"""Strict final artifact audit for the prospective protocol-correct v2 block."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_protocol_correct_outputs as base
import prospective_protocol_v2_common as frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=frozen.OUTPUT)
    parser.add_argument("--output", type=Path, default=frozen.CERT_ROOT / "PATHMNIST_PROSPECTIVE_PROTOCOL_CORRECT_STRICT_AUDIT_V2.json")
    args = parser.parse_args()
    frozen.verify_freeze()
    run = base.audit_one("pathmnist_prospective_protocol_correct_v2", args.results)
    report = {
        "status": run["status"],
        "scope": "prospective training-to-issuer strict schema, lineage, timing, hash, arithmetic, and leakage audit",
        "config_sha256": frozen.sha256(frozen.CONFIG),
        "freeze_sha256": frozen.sha256(frozen.FREEZE),
        "audit_wrapper_sha256": frozen.sha256(Path(__file__)),
        "base_audit_sha256": frozen.sha256(frozen.BASE_AUDIT),
        "run": run,
        "audited_unix": time.time(),
    }
    base.atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
