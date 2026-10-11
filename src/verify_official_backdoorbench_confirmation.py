#!/usr/bin/env python3
"""Independent replay of the official BackdoorBench feature confirmation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/bloodmnist_official_backdoorbench_confirmation_v1"
PILOT_V1 = ROOT / "results/bloodmnist_official_backdoorbench_pilot_v1/manifest.json"
PILOT_V2 = ROOT / "results/bloodmnist_official_backdoorbench_pilot_v2/manifest.json"
RECEIPT = ROOT / "results/top_journal_program_v1/official_backdoorbench_confirmation_receipt.json"
EXPECTED_COMMIT = "f02e3534645f0ee63d6848653062cd6c0d6c400d"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(left - right) <= tolerance


def main() -> None:
    manifest = json.loads((RESULT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["gate"]["confirmation_complete"] is True
    assert manifest["upstream_commit"] == EXPECTED_COMMIT
    protocol = Path(manifest["protocol"])
    assert sha256(protocol) == manifest["protocol_sha256"]
    assert sha256(ROOT / "src/backdoorbench_feature_adapter.py") == manifest[
        "adapter_sha256"
    ]
    assert manifest["seeds"] == list(range(52, 62))
    assert len(manifest["scenarios"]) == 3
    assert len(manifest["methods"]) == 4

    metrics = rows(RESULT / "per_config_metrics.csv")
    samples = rows(RESULT / "per_sample_flags.csv")
    victims = rows(RESULT / "victim_metrics.csv")
    counts = rows(RESULT / "pool_count_checks.csv")
    summaries = rows(RESULT / "eligible_summary.csv")
    assert len(metrics) == 120
    assert len(samples) == 133320
    assert len(victims) == 30
    assert len(counts) == 10
    assert len(summaries) == 12
    for row in counts:
        assert tuple(
            int(row[key])
            for key in (
                "target_clean",
                "source_clean",
                "poison",
                "reference_target",
                "reference_source",
            )
        ) == (600, 451, 60, 200, 170)

    victim_map = {
        (row["seed"], row["scenario"]): int(row["eligible"]) for row in victims
    }
    grouped_samples: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in samples:
        grouped_samples[(row["seed"], row["scenario"], row["method"])].append(row)

    replayed: list[dict] = []
    for metric in metrics:
        key = (metric["seed"], metric["scenario"], metric["method"])
        selected = grouped_samples[key]
        assert len(selected) == 1111
        assert int(metric["eligible"]) == victim_map[key[:2]]
        poison = [row for row in selected if int(row["is_poison"])]
        clean = [row for row in selected if not int(row["is_poison"])]
        target = [row for row in clean if row["sample_kind"] == "target_clean"]
        source = [row for row in clean if row["sample_kind"] == "source_clean"]
        tp = sum(int(row["flagged"]) for row in poison)
        fp = sum(int(row["flagged"]) for row in clean)
        target_fp = sum(int(row["flagged"]) for row in target)
        source_fp = sum(int(row["flagged"]) for row in source)
        assert (tp, len(poison) - tp, fp, len(clean) - fp) == tuple(
            int(metric[name]) for name in ("tp", "fn", "fp", "tn")
        )
        assert close(tp / len(poison), float(metric["tpr"]))
        assert close(fp / len(clean), float(metric["fpr_all_clean"]))
        assert close(target_fp / len(target), float(metric["target_clean_fpr"]))
        assert close(source_fp / len(source), float(metric["source_clean_fpr"]))
        receipt = json.loads(metric["receipt_json"])
        assert receipt["commit"] == EXPECTED_COMMIT
        assert receipt["upstream_symbol"]
        if "spectre" in metric["method"]:
            assert receipt["scipy_solve_compatibility"] in {
                "native_sym_pos",
                "sym_pos_to_assume_a_pos",
            }
        replayed.append(
            {
                "scenario": metric["scenario"],
                "method": metric["method"],
                "eligible": int(metric["eligible"]),
                "tp": tp,
                "fn": len(poison) - tp,
                "fp": fp,
                "tn": len(clean) - fp,
                "target_clean_fpr": target_fp / len(target),
                "source_clean_fpr": source_fp / len(source),
            }
        )

    expected_summary: dict[tuple[str, str], dict] = {}
    for row in replayed:
        if not row["eligible"]:
            continue
        key = (row["scenario"], row["method"])
        aggregate = expected_summary.setdefault(
            key,
            {"n": 0, "tp": 0, "fn": 0, "fp": 0, "tn": 0, "tfpr": [], "sfpr": []},
        )
        aggregate["n"] += 1
        for name in ("tp", "fn", "fp", "tn"):
            aggregate[name] += row[name]
        aggregate["tfpr"].append(row["target_clean_fpr"])
        aggregate["sfpr"].append(row["source_clean_fpr"])

    summary_digest = {}
    for row in summaries:
        key = (row["scenario"], row["method"])
        aggregate = expected_summary[key]
        pooled_tpr = aggregate["tp"] / (aggregate["tp"] + aggregate["fn"])
        pooled_fpr = aggregate["fp"] / (aggregate["fp"] + aggregate["tn"])
        mean_target = sum(aggregate["tfpr"]) / aggregate["n"]
        mean_source = sum(aggregate["sfpr"]) / aggregate["n"]
        assert int(row["eligible_seed_count"]) == aggregate["n"]
        assert close(float(row["pooled_tpr"]), pooled_tpr)
        assert close(float(row["pooled_fpr_all_clean"]), pooled_fpr)
        assert close(float(row["mean_target_clean_fpr"]), mean_target)
        assert close(float(row["mean_source_clean_fpr"]), mean_source)
        summary_digest["|".join(key)] = {
            "eligible_seeds": aggregate["n"],
            "pooled_tpr": pooled_tpr,
            "pooled_fpr_all_clean": pooled_fpr,
            "mean_target_clean_fpr": mean_target,
        }

    pilot_v1 = json.loads(PILOT_V1.read_text(encoding="utf-8"))
    pilot_v2 = json.loads(PILOT_V2.read_text(encoding="utf-8"))
    assert pilot_v1["gate"]["expand_to_confirmation"] is False
    assert pilot_v1["gate"]["pool_counts_match"] is False
    assert pilot_v2["gate"]["expand_to_confirmation"] is True
    assert pilot_v2["gate"]["pool_counts_match"] is True

    receipt = {
        "status": "PASS",
        "experiment": manifest["experiment"],
        "meaning": "Independent replay of sample flags, confusion matrices, eligibility filtering, pooled summaries, provenance receipts, hashes, and the preserved V1 STOP/V2 PASS audit trail.",
        "rows_verified": {
            "per_config": len(metrics),
            "per_sample": len(samples),
            "victims": len(victims),
            "eligible_summaries": len(summaries),
        },
        "upstream_commit": EXPECTED_COMMIT,
        "file_sha256": {
            name: sha256(RESULT / name)
            for name in (
                "per_config_metrics.csv",
                "per_sample_flags.csv",
                "victim_metrics.csv",
                "pool_count_checks.csv",
                "eligible_summary.csv",
                "manifest.json",
            )
        },
        "eligible_summary": summary_digest,
        "audit_trail": {
            "pilot_v1_stopped_on_wrong_count_assumption": True,
            "pilot_v2_corrected_counts_without_method_changes": True,
        },
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "receipt": str(RECEIPT)}, indent=2))


if __name__ == "__main__":
    main()
