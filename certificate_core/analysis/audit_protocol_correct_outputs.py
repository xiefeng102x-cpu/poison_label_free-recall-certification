#!/usr/bin/env python3
"""Strict post-run audit for the two protocol-correct certificate replays."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CERT_ROOT = ROOT / "auditability_certificate_v1"
CONFIG = ROOT / "configs/protocol_correct_certificate_rerun_v1.json"
FREEZE = CERT_ROOT / "PROTOCOL_CORRECT_RERUN_FREEZE_V1.json"

PUBLIC_FIELDS = ["protocol", "trial_token", "k", "conformal_rank", "candidate_count", "flagged_count"]
SEALED_FIELDS = PUBLIC_FIELDS + [
    "interval", "n_poison_min", "n_poison_max", "recall_lo", "recall_hi",
    "clean_flag_rate_hi", "caught_poison_lo", "decision",
]
PROHIBITED_PUBLIC_FIELDS = {
    "victim_id", "detector", "fold", "sample_id", "role", "is_poison",
    "condition", "epsilon", "pool_role", "selected_contaminated_count",
}


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


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def audit_one(name: str, output: Path) -> dict[str, Any]:
    failures: list[str] = []
    public_manifest_path = output / "public/issuer_manifest.json"
    registrar_manifest_path = output / "private/registrar_manifest.json"
    observations_path = output / "public/issuer_observations.csv"
    map_path = output / "private/trial_map.csv"
    seal_path = output / "sealed/issuer_seal_manifest.json"
    decisions_path = output / "sealed/certificate_decisions_sealed.csv"
    verification_path = output / "verified/verification_report.json"
    evaluated_path = output / "verified/certificate_decisions_evaluated.csv"
    final_manifest_path = output / "verification_manifest.json"
    reveal_path = output / "private/revealed_after_seal_key.hex"
    required = [public_manifest_path, registrar_manifest_path, observations_path, map_path, seal_path, decisions_path, verification_path, evaluated_path, final_manifest_path, reveal_path]
    for path in required:
        check(path.is_file(), f"missing required artifact: {path.name}", failures)
    if failures:
        return {"name": name, "status": "FAIL", "failures": failures}

    public_manifest = json.loads(public_manifest_path.read_text(encoding="utf-8"))
    registrar_manifest = json.loads(registrar_manifest_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    observations = pd.read_csv(observations_path)
    private_map = pd.read_csv(map_path)
    decisions = pd.read_csv(decisions_path)
    evaluated = pd.read_csv(evaluated_path)

    check(list(observations.columns) == PUBLIC_FIELDS, "public observation schema differs from allow-list", failures)
    check(list(decisions.columns) == SEALED_FIELDS, "sealed decision schema differs from allow-list", failures)
    check(not PROHIBITED_PUBLIC_FIELDS.intersection(observations.columns), "public observations expose prohibited fields", failures)
    check(not PROHIBITED_PUBLIC_FIELDS.intersection(decisions.columns), "sealed decisions expose prohibited fields", failures)
    check(not observations.isna().any().any(), "public observations contain nulls", failures)
    check(not decisions.isna().any().any(), "sealed decisions contain nulls", failures)
    check(not observations["trial_token"].duplicated().any(), "public trial tokens are not unique", failures)
    check(not private_map["trial_token"].duplicated().any(), "private trial tokens are not unique", failures)
    check(set(observations["trial_token"]) == set(private_map["trial_token"]), "public/private token sets differ", failures)
    check(set(observations["trial_token"]) == set(decisions["trial_token"]), "public/sealed token sets differ", failures)
    check((decisions.groupby("trial_token").size() == 2).all(), "each trial must have exactly two interval decisions", failures)
    check(len(observations) == int(public_manifest["rows"]), "public row count disagrees with manifest", failures)
    check(len(decisions) == int(seal["decision_rows"]) == 2 * len(observations), "sealed row count mismatch", failures)
    check(len(evaluated) == len(decisions), "evaluated row count mismatch", failures)
    check(int(verification["checked_trials"]) == len(observations), "verification trial count mismatch", failures)
    check(int(verification["checked_decisions"]) == len(decisions), "verification decision count mismatch", failures)
    check(int(verification["unique_trial_tokens"]) == len(observations), "verified token count mismatch", failures)
    check(sha256(observations_path) == public_manifest["issuer_observations_sha256"], "public observation hash mismatch", failures)
    check(sha256(map_path) == registrar_manifest["trial_map_sha256"], "private map hash mismatch", failures)
    check(sha256(decisions_path) == seal["certificate_decisions_sealed_sha256"], "sealed decision hash mismatch", failures)
    check(sha256(decisions_path) == verification["hashes"]["certificate_decisions_sealed_sha256"], "verification references different sealed decisions", failures)
    check(sha256(evaluated_path) == verification["hashes"]["certificate_decisions_evaluated_sha256"], "evaluated result hash mismatch", failures)
    check(float(public_manifest["created_unix"]) <= float(seal["sealed_unix"]) <= float(verification["verified_unix"]), "stage timestamps are not monotone", failures)
    check(final_manifest["status"] == verification["status"] == "PASS_PROTOCOL_CORRECT_INDEPENDENT_REPLAY", "verification status is not PASS", failures)
    check(verification["issuer_input_schema_allowlist"] == "PASS", "issuer input allow-list audit failed", failures)
    check(verification["issuer_static_prohibited_identifier_scan"] == "PASS", "issuer static isolation audit failed", failures)
    check(verification["decision_sealed_before_verification"] is True, "decision sealing order not established", failures)
    check(verification["non_exhaustive_trusted_pool"] is True, "trusted pool non-exhaustiveness not established", failures)

    numeric = observations[["k", "conformal_rank", "candidate_count", "flagged_count"]].to_numpy(float)
    check(np.isfinite(numeric).all(), "public numeric fields contain non-finite values", failures)
    check((observations["k"] > 0).all(), "non-positive calibration budget", failures)
    check(((observations["conformal_rank"] >= 1) & (observations["conformal_rank"] <= observations["k"])).all(), "invalid conformal rank", failures)
    check(((observations["flagged_count"] >= 0) & (observations["flagged_count"] <= observations["candidate_count"])).all(), "invalid flagged count", failures)
    for column in ("recall_lo", "recall_hi", "clean_flag_rate_hi"):
        values = decisions[column].to_numpy(float)
        check(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), f"invalid bound values: {column}", failures)
    check((decisions["recall_lo"] <= decisions["recall_hi"] + 1e-12).all(), "recall interval is inverted", failures)
    check(set(decisions["decision"].unique()).issubset({"GO", "NO_GO", "ABSTAIN"}), "invalid decision label", failures)

    valid = evaluated["declared_interval_valid"].astype(int) == 1
    false_go_recomputed = ((evaluated["decision"] == "GO") & ((evaluated["actual_candidate_recall"] < 0.8) | (evaluated["actual_clean_flag_rate"] > 0.1)) & valid).astype(int)
    check(np.array_equal(false_go_recomputed.to_numpy(), evaluated["valid_false_go_candidate"].astype(int).to_numpy()), "false-GO field does not recompute", failures)
    check(int(false_go_recomputed.sum()) == int(verification["valid_false_go_candidate_count_all_cells"]), "false-GO total disagrees with verification report", failures)

    key_hex = reveal_path.read_text(encoding="ascii").strip()
    key_bytes = bytes.fromhex(key_hex)
    check(len(key_bytes) == 32, "revealed key length mismatch", failures)
    commitment = hashlib.sha256(key_bytes).hexdigest()
    check(commitment == public_manifest["key_commitment_sha256"] == seal["key_commitment_sha256"], "revealed key does not open commitment", failures)

    return {
        "name": name,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "public_trials": int(len(observations)),
        "sealed_decisions": int(len(decisions)),
        "valid_decisions": int(valid.sum()),
        "valid_false_go_count": int(false_go_recomputed.sum()),
        "go_rate": float((decisions["decision"] == "GO").mean()),
        "nogo_rate": float((decisions["decision"] == "NO_GO").mean()),
        "abstain_rate": float((decisions["decision"] == "ABSTAIN").mean()),
        "public_observations_sha256": sha256(observations_path),
        "sealed_decisions_sha256": sha256(decisions_path),
        "verification_report_sha256": sha256(verification_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-results", type=Path, required=True)
    parser.add_argument("--redteam-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [audit_one("pathmnist_blind", args.path_results), audit_one("certificate_aware_redteam", args.redteam_results)]
    status = "PASS" if all(report["status"] == "PASS" for report in reports) else "FAIL"
    payload = {
        "status": status,
        "scope": "strict artifact, schema, lineage, timing, hash, arithmetic, and leakage audit",
        "config_sha256": sha256(CONFIG),
        "freeze_sha256": sha256(FREEZE),
        "audit_script_sha256": sha256(Path(__file__)),
        "runs": reports,
        "audited_unix": time.time(),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
