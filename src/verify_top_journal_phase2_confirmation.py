#!/usr/bin/env python3
"""Replay and statistically verify the P2 Blended/SIG confirmation block."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import verify_top_journal_phase2_pilot as common


PAPER14_ROOT = Path(__file__).resolve().parents[1]
ROOT = PAPER14_ROOT / "results/bloodmnist_cross_attack_confirmation_v1"
PROGRAM_ROOT = PAPER14_ROOT / "results/top_journal_program_v1"
PROTOCOL = PROGRAM_ROOT / "phase2_confirmation_protocol.json"
OUTPUT = PROGRAM_ROOT / "phase2_confirmation_receipt.json"
SEEDS = tuple(range(52, 62))
ATTACKS = ("blended", "sig")
ENDPOINTS = {
    "coherent_bonferroni_input_classical_margin": "classical",
    "coherent_qml_xz_drift16": "qml",
    "coherent_bonferroni_input_qml_xz": "qml",
}


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, values: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(left - right) <= tolerance


def gate(row: dict) -> bool:
    return float(row["pooled_tpr"]) >= 0.90 and float(row["pooled_fpr"]) <= 0.05


def bootstrap(selected: list[dict], rng: np.random.Generator) -> dict:
    counts = np.asarray(
        [[int(row[key]) for key in ("tp", "fn", "fp", "tn")] for row in selected],
        dtype=np.int64,
    )
    draws = rng.integers(0, len(selected), size=(10000, len(selected)))
    sampled = counts[draws].sum(axis=1)
    tpr = sampled[:, 0] / (sampled[:, 0] + sampled[:, 1])
    fpr = sampled[:, 2] / (sampled[:, 2] + sampled[:, 3])
    pooled = counts.sum(axis=0)
    return {
        "n_seed_blocks": len(selected),
        "tp": int(pooled[0]),
        "fn": int(pooled[1]),
        "fp": int(pooled[2]),
        "tn": int(pooled[3]),
        "pooled_tpr": float(pooled[0] / (pooled[0] + pooled[1])),
        "tpr_seed_bootstrap_ci_low": float(np.quantile(tpr, 0.025)),
        "tpr_seed_bootstrap_ci_high": float(np.quantile(tpr, 0.975)),
        "pooled_fpr": float(pooled[2] / (pooled[2] + pooled[3])),
        "fpr_seed_bootstrap_ci_low": float(np.quantile(fpr, 0.025)),
        "fpr_seed_bootstrap_ci_high": float(np.quantile(fpr, 0.975)),
    }


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert protocol["frozen_before_execution"] is True
    assert protocol["seeds"] == list(SEEDS)
    assert protocol["attacks"] == list(ATTACKS)
    assert manifest["role"] == "selective_confirmation"
    assert manifest["seeds"] == list(SEEDS)
    assert manifest["attacks"] == list(ATTACKS)
    assert manifest["alpha"] == 0.05
    assert manifest["poison_count"] == 60
    assert manifest["trusted_reference_split"] == "validation_target"

    metrics = rows(ROOT / "per_config_metrics.csv")
    samples = rows(ROOT / "per_sample_scores.csv")
    victims = rows(ROOT / "victim_metrics.csv")
    pooled = rows(ROOT / "pooled_summary.csv")
    eligible = rows(ROOT / "eligible_summary.csv")
    assert len(metrics) == 10 * 2 * 6
    assert len(samples) == 75600
    assert len(victims) == 20
    assert len(pooled) == 12
    assert len(eligible) == 6

    grouped_samples: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in samples:
        assert int(row["flagged"]) == (float(row["p_value"]) <= float(row["alpha"]))
        grouped_samples[(row["seed"], row["trigger"], row["representation"])].append(row)
    for metric in metrics:
        selected = grouped_samples[
            (metric["seed"], metric["trigger"], metric["representation"])
        ]
        clean = [row for row in selected if int(row["label"]) == 0]
        poison = [row for row in selected if int(row["label"]) == 1]
        fp = sum(int(row["flagged"]) for row in clean)
        tp = sum(int(row["flagged"]) for row in poison)
        counts = (tp, len(poison) - tp, fp, len(clean) - fp)
        assert counts == tuple(int(metric[key]) for key in ("tp", "fn", "fp", "tn"))
        assert close(tp / len(poison), float(metric["tpr"]))
        assert close(fp / len(clean), float(metric["fpr"]))

    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in metrics:
        by_group[(row["trigger"], row["representation"])].append(row)
    for summary in pooled:
        selected = by_group[(summary["trigger"], summary["representation"])]
        counts = tuple(
            sum(int(row[key]) for row in selected) for key in ("tp", "fn", "fp", "tn")
        )
        assert counts == tuple(int(summary[key]) for key in ("tp", "fn", "fp", "tn"))
        assert close(counts[0] / (counts[0] + counts[1]), float(summary["pooled_tpr"]))
        assert close(counts[2] / (counts[2] + counts[3]), float(summary["pooled_fpr"]))

    victim_map = {(row["attack"], row["seed"]): row for row in victims}
    expected_eligible = {
        ("blended", "classical"): 10,
        ("blended", "qml"): 10,
        ("sig", "classical"): 10,
        ("sig", "qml"): 3,
    }
    eligible_map = {}
    for row in eligible:
        attack = row["trigger"]
        victim_type = row["victim_type"]
        valid_seeds = []
        for seed in map(str, SEEDS):
            victim = victim_map[(attack, seed)]
            if (
                float(victim[f"{victim_type}_balanced_ca"]) >= 0.90
                and float(victim[f"{victim_type}_asr"]) >= 0.80
            ):
                valid_seeds.append(seed)
        assert int(row["eligible_seed_count"]) == len(valid_seeds)
        assert len(valid_seeds) == expected_eligible[(attack, victim_type)]
        assert row["status"] == "CONFIRMATORY"
        assert gate(row)
        eligible_map[(attack, row["representation"])] = row

    fingerprints_verified = common.verify_attack_fingerprints(manifest, ROOT)

    rng = np.random.default_rng(20260928)
    bootstrap_rows = []
    for (attack, representation), summary in sorted(eligible_map.items()):
        victim_type = ENDPOINTS[representation]
        selected = []
        for row in by_group[(attack, representation)]:
            victim = victim_map[(attack, row["seed"])]
            if (
                float(victim[f"{victim_type}_balanced_ca"]) >= 0.90
                and float(victim[f"{victim_type}_asr"]) >= 0.80
            ):
                selected.append(row)
        result = bootstrap(selected, rng)
        assert close(result["pooled_tpr"], float(summary["pooled_tpr"]))
        assert close(result["pooled_fpr"], float(summary["pooled_fpr"]))
        bootstrap_rows.append(
            {
                "attack": attack,
                "representation": representation,
                "victim_type": victim_type,
                **result,
            }
        )
    write_csv(ROOT / "eligible_seed_bootstrap_summary.csv", bootstrap_rows)

    worst_tpr = min(row["pooled_tpr"] for row in bootstrap_rows)
    worst_fpr = max(row["pooled_fpr"] for row in bootstrap_rows)
    receipt = {
        "status": "PASS",
        "meaning": (
            "frozen protocol, 75,600 sample decisions, pooled calculations, "
            "eligibility, attack/data fingerprints, checkpoint hashes, and seed-block "
            "bootstrap statistics replayed"
        ),
        "phase2_confirmation_verdict": "DETECTOR_PASS_WITH_SIG_QML_ELIGIBILITY_LIMIT",
        "metric_rows": len(metrics),
        "sample_rows": len(samples),
        "victim_rows": len(victims),
        "attack_seed_fingerprints_verified": fingerprints_verified,
        "all_six_eligible_endpoints_pass_primary_gate": True,
        "worst_eligible_endpoint_tpr": worst_tpr,
        "worst_eligible_endpoint_fpr": worst_fpr,
        "eligible_victim_counts": {
            f"{attack}_{kind}": count
            for (attack, kind), count in sorted(expected_eligible.items())
        },
        "material_limit": (
            "SIG QML is attack-eligible on only 3/10 seeds; its detector result is "
            "valid for those victims but does not establish robust SIG attack transfer to QML."
        ),
    }
    OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
