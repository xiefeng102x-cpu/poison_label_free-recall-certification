#!/usr/bin/env python3
"""Post-unblinding descriptive audit for the prospective protocol-correct v2 block.

This script is not part of the frozen decision rule.  It independently recomputes
the headline victim, detector, and certificate summaries from the sealed handoff.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/pathmnist_prospective_protocol_correct_v2_source"
OUTPUT = ROOT / "results/pathmnist_prospective_protocol_correct_v2"
REPORT = OUTPUT / "verified/post_unblinding_descriptive_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rates(frame: pd.DataFrame) -> dict[str, float | int]:
    total = len(frame)
    decisions = frame["decision"].value_counts()
    return {
        "rows": int(total),
        "go_rate": float(decisions.get("GO", 0) / total),
        "nogo_rate": float(decisions.get("NO_GO", 0) / total),
        "abstain_rate": float(decisions.get("ABSTAIN", 0) / total),
        "false_go_count": int(frame["valid_false_go_candidate"].sum()),
    }


def main() -> None:
    source_manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    receipts = json.loads((SOURCE / "seed_receipts.json").read_text(encoding="utf-8"))
    verification = json.loads(
        (OUTPUT / "verified/verification_report.json").read_text(encoding="utf-8")
    )
    issuer = json.loads((OUTPUT / "public/issuer_manifest.json").read_text(encoding="utf-8"))
    seal = json.loads((OUTPUT / "sealed/issuer_seal_manifest.json").read_text(encoding="utf-8"))

    score_path = SOURCE / "blind_scores.csv"
    truth_path = SOURCE / "private_truth.csv"
    decision_path = OUTPUT / "verified/certificate_decisions_evaluated.csv"
    scores = pd.read_csv(score_path)
    truth = pd.read_csv(truth_path)
    decisions = pd.read_csv(decision_path)

    hash_checks = {
        "blind_scores": sha256(score_path) == source_manifest["blind_scores_sha256"],
        "private_truth": sha256(truth_path) == source_manifest["private_truth_sha256"],
        "sealed_decisions": sha256(OUTPUT / "sealed/certificate_decisions_sealed.csv")
        == seal["certificate_decisions_sealed_sha256"],
        "evaluated_decisions": sha256(decision_path)
        == verification["hashes"]["certificate_decisions_evaluated_sha256"],
    }
    if not all(hash_checks.values()):
        raise RuntimeError(f"hash check failed: {hash_checks}")

    train_scores = scores.loc[scores["role"].eq("train")].merge(
        truth.loc[truth["role"].eq("train"), ["victim_id", "sample_id", "is_poison"]],
        on=["victim_id", "sample_id"],
        how="left",
        validate="many_to_one",
    )
    if train_scores["is_poison"].isna().any():
        raise RuntimeError("score-to-truth join has unmatched rows")

    detector_rows = []
    for detector, frame in train_scores.groupby("detector", sort=True):
        poison = frame.loc[frame["is_poison"].eq(1)]
        clean = frame.loc[frame["is_poison"].eq(0)]
        detector_rows.append(
            {
                "detector": detector,
                "poison_rows": int(len(poison)),
                "clean_rows": int(len(clean)),
                "realized_recall": float(poison["conventional_flag"].mean()),
                "target_clean_flag_rate": float(clean["conventional_flag"].mean()),
            }
        )

    victim_rows = [
        {
            "seed": int(receipt["seed"]),
            "eligible": bool(receipt["eligible"]),
            "balanced_accuracy": float(receipt["victim_metrics"]["balanced_accuracy"]),
            "attack_success_rate": float(receipt["victim_metrics"]["asr"]),
            "elapsed_seconds": float(receipt["elapsed_seconds"]),
            "model_sha256": receipt["model_sha256"],
        }
        for receipt in receipts
    ]

    interval_summary = {}
    for interval, frame in decisions.groupby("interval", sort=True):
        interval_summary[str(interval)] = rates(frame)

    primary = decisions.loc[decisions["interval"].eq("primary")]
    primary_by_k = {
        str(int(k)): rates(frame) for k, frame in primary.groupby("k", sort=True)
    }
    primary_by_detector = {
        str(detector): {
            **rates(frame),
            "mean_actual_candidate_recall": float(frame["actual_candidate_recall"].mean()),
            "mean_actual_clean_flag_rate": float(frame["actual_clean_flag_rate"].mean()),
        }
        for detector, frame in primary.groupby("detector", sort=True)
    }

    report = {
        "status": "PASS_POST_UNBLINDING_DESCRIPTIVE_AUDIT",
        "scope": "descriptive recomputation only; frozen decision rule unchanged",
        "hash_checks": hash_checks,
        "source": {
            "status": source_manifest["status"],
            "device": source_manifest["device"],
            "all_victims_eligible": bool(source_manifest["all_victims_eligible"]),
            "score_rows": int(len(scores)),
            "truth_rows": int(len(truth)),
            "completed_seeds": source_manifest["completed_seeds"],
            "blind_scores_sha256": source_manifest["blind_scores_sha256"],
        },
        "victims": victim_rows,
        "victim_aggregate": {
            "count": len(victim_rows),
            "balanced_accuracy_mean": float(
                sum(row["balanced_accuracy"] for row in victim_rows) / len(victim_rows)
            ),
            "balanced_accuracy_min": float(min(row["balanced_accuracy"] for row in victim_rows)),
            "balanced_accuracy_max": float(max(row["balanced_accuracy"] for row in victim_rows)),
            "attack_success_rate_mean": float(
                sum(row["attack_success_rate"] for row in victim_rows) / len(victim_rows)
            ),
            "attack_success_rate_min": float(min(row["attack_success_rate"] for row in victim_rows)),
            "attack_success_rate_max": float(max(row["attack_success_rate"] for row in victim_rows)),
        },
        "detectors": detector_rows,
        "certificate": {
            "issuer_input_schema_exactly_allowlisted": bool(
                seal["input_schema_exactly_allowlisted"]
            ),
            "issuer_input_rows": int(seal["input_rows"]),
            "sealed_decision_rows": int(seal["decision_rows"]),
            "sealed_decisions_sha256": seal["certificate_decisions_sealed_sha256"],
            "verification_status": verification["status"],
            "prospective_from_victim_training": bool(
                verification["prospective_from_victim_training"]
            ),
            "decision_sealed_before_verification": bool(
                verification["decision_sealed_before_verification"]
            ),
            "primary_false_go_count": int(verification["primary_false_go_count"]),
            "all_valid_false_go_count": int(
                verification["valid_false_go_candidate_count_all_cells"]
            ),
            "intervals": interval_summary,
            "primary_by_k": primary_by_k,
            "primary_by_detector": primary_by_detector,
            "key_commitment_sha256": issuer["key_commitment_sha256"],
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
