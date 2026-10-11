#!/usr/bin/env python3
"""Schema-restricted issuer for opaque certificate sufficient statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CERT_ROOT = ROOT / "auditability_certificate_v1"
CONFIG = ROOT / "configs/protocol_correct_certificate_rerun_v1.json"

import sys
sys.path.insert(0, str(CERT_ROOT))
from cert.certificate import certify_realized_counts

INPUT_FIELDS = ["protocol", "trial_token", "k", "conformal_rank", "candidate_count", "flagged_count"]
OUTPUT_FIELDS = INPUT_FIELDS + [
    "interval", "n_poison_min", "n_poison_max", "recall_lo", "recall_hi",
    "clean_flag_rate_hi", "caught_poison_lo", "decision",
]


@lru_cache(maxsize=None)
def cached_certificate(k: int, rank: int, flagged_count: int, candidate_count: int,
                       n_min: int, n_max: int, tau: float, delta: float,
                       alpha_cap: float, fpr_mode: str) -> dict[str, Any]:
    """Cache the pure certificate map; no private field enters the key."""
    return certify_realized_counts(
        k, rank, flagged_count, candidate_count, tau=tau,
        n_poison_min=n_min, n_poison_max=n_max, delta=delta,
        alpha_cap=alpha_cap, fpr_mode=fpr_mode,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def issue(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    input_manifest_path = input_dir / "issuer_manifest.json"
    observation_path = input_dir / "issuer_observations.csv"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if input_manifest.get("status") != "ISSUER_OBSERVATIONS_SEALED":
        raise RuntimeError("issuer input is not sealed")
    if input_manifest.get("allowed_schema") != INPUT_FIELDS:
        raise RuntimeError("issuer allow-list mismatch")
    if input_manifest.get("config_sha256") != sha256(CONFIG):
        raise RuntimeError("configuration hash mismatch")
    if input_manifest.get("issuer_observations_sha256") != sha256(observation_path):
        raise RuntimeError("observation hash mismatch")
    protocol = input_manifest["protocol"]
    section_name = "pathmnist_blind" if protocol == "pathmnist_blind" else "certificate_aware_redteam"
    section = cfg[section_name]
    common = cfg["common"]
    intervals = {name: tuple(map(int, bounds)) for name, bounds in section["declared_intervals_per_fold"].items()}
    delta = float(section["delta_per_cell"])
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_path = output_dir / "certificate_decisions_sealed.csv"
    temporary = decision_path.with_name(decision_path.name + ".tmp")
    input_rows = 0
    output_rows = 0
    with observation_path.open(newline="", encoding="utf-8") as source, temporary.open("w", newline="", encoding="utf-8") as target:
        reader = csv.DictReader(source)
        if reader.fieldnames != INPUT_FIELDS:
            raise RuntimeError("unexpected issuer input fields")
        writer = csv.DictWriter(target, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in reader:
            input_rows += 1
            k = int(row["k"])
            rank = int(row["conformal_rank"])
            candidate_count = int(row["candidate_count"])
            flagged_count = int(row["flagged_count"])
            for interval, (n_min, n_max) in intervals.items():
                result = cached_certificate(
                    k, rank, flagged_count, candidate_count, n_min, n_max,
                    float(common["tau"]), delta, float(common["alpha_cap"]),
                    str(common["fpr_mode"]),
                )
                writer.writerow({
                    **row,
                    "interval": interval,
                    "n_poison_min": n_min,
                    "n_poison_max": n_max,
                    "recall_lo": repr(float(result["recall_lo"])),
                    "recall_hi": repr(float(result["recall_hi"])),
                    "clean_flag_rate_hi": repr(float(result["clean_flag_rate_hi"])),
                    "caught_poison_lo": repr(float(result["flagged_poisons_lo"])),
                    "decision": {1: "GO", -1: "NO_GO", 0: "ABSTAIN"}[int(result["decision"])],
                })
                output_rows += 1
    os.replace(temporary, decision_path)
    if input_rows != int(input_manifest["rows"]):
        raise RuntimeError("issuer input row count mismatch")
    report = {
        "status": "CERTIFICATE_DECISIONS_HASH_SEALED",
        "protocol": protocol,
        "input_rows": input_rows,
        "decision_rows": output_rows,
        "issuer_input_manifest_sha256": sha256(input_manifest_path),
        "issuer_observations_sha256": sha256(observation_path),
        "certificate_decisions_sealed_sha256": sha256(decision_path),
        "config_sha256": sha256(CONFIG),
        "issuer_sha256": sha256(Path(__file__)),
        "key_commitment_sha256": input_manifest["key_commitment_sha256"],
        "sealed_unix": time.time(),
        "input_schema_exactly_allowlisted": True,
    }
    atomic_json(output_dir / "issuer_seal_manifest.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    issue(args.input, args.output)


if __name__ == "__main__":
    main()
