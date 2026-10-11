#!/usr/bin/env python3
"""Independently recompute the frozen TIFS gap-closure experiment results.

This verifier consumes only persisted protocols, sample-level outputs, victim
metrics, and manifests.  It never imports an experiment runner or a model.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "tifs_gap_closure_v1"
TOL = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, label: str) -> None:
    if abs(actual - expected) > TOL:
        raise AssertionError(f"{label}: {actual} != {expected}")


def verify_adaptive() -> dict:
    root = RESULTS / "tifs_adaptive_blend_patch_v1"
    confirm = root / "confirmation"
    protocol = root / "configs/confirmation_protocol.json"
    expected_hash = (root / "configs/confirmation_protocol.sha256").read_text().split()[0]
    assert sha256(protocol) == expected_hash

    victims = read_csv(confirm / "victim_metrics.csv")
    assert len(victims) == 20
    assert {int(row["seed"]) for row in victims} == set(range(73, 83))
    victim_map = {
        (row["attack"], int(row["seed"])): row
        for row in victims
    }
    assert len(victim_map) == 20

    persisted = read_csv(confirm / "per_seed_detector_metrics.csv")
    persisted_map = {
        (row["attack"], row["view"], row["method"], int(row["seed"])): row
        for row in persisted
    }
    assert len(persisted) == len(persisted_map) == 200

    counts: dict[tuple[str, str, str, int], list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    with (confirm / "per_sample_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["attack"], row["view"], row["method"], int(row["seed"]))
            label = int(row["label"])
            flag = int(row["flagged"])
            bucket = counts[key]
            bucket[0] += int(label == 1 and flag == 1)  # tp
            bucket[1] += int(label == 1 and flag == 0)  # fn
            bucket[2] += int(label == 0 and flag == 1)  # fp
            bucket[3] += int(label == 0 and flag == 0)  # tn
            bucket[4] += 1
    assert set(counts) == set(persisted_map)
    for key, (tp, fn, fp, tn, rows) in counts.items():
        assert rows == 660
        saved = persisted_map[key]
        assert (tp, fn, fp, tn) == tuple(int(saved[name]) for name in ("tp", "fn", "fp", "tn"))
        close(tp / (tp + fn), float(saved["tpr"]), f"adaptive {key} tpr")
        close(fp / (fp + tn), float(saved["fpr"]), f"adaptive {key} fpr")

    summary = read_csv(confirm / "summary.csv")
    summary_map = {
        (row["subset"], row["attack"], row["view"], row["method"]): row
        for row in summary
    }
    for subset in ("all_seeds", "eligible_only"):
        for attack in ("adaptive_blend", "adaptive_patch"):
            for view in ("qml_checkpoint_drift_xz16", "classical_checkpoint_drift_margin1"):
                eligibility_name = "qml_eligible" if view.startswith("qml_") else "classical_eligible"
                for method in ("cf_cmd_u", "spectral_signatures", "spectre_que", "activation_clustering", "scan_style"):
                    selected = []
                    for seed in range(73, 83):
                        if subset == "eligible_only" and int(victim_map[(attack, seed)][eligibility_name]) != 1:
                            continue
                        selected.append(counts[(attack, view, method, seed)])
                    saved = summary_map[(subset, attack, view, method)]
                    total = [sum(row[index] for row in selected) for index in range(4)]
                    assert total == [int(saved[name]) for name in ("tp", "fn", "fp", "tn")]
                    assert len(selected) == int(saved["seed_count"])
                    close(total[0] / (total[0] + total[1]), float(saved["pooled_tpr"]), "adaptive pooled tpr")
                    close(total[2] / (total[2] + total[3]), float(saved["pooled_fpr"]), "adaptive pooled fpr")

    eligible = {
        attack: {
            "classical": sum(int(row["classical_eligible"]) for row in victims if row["attack"] == attack),
            "qml": sum(int(row["qml_eligible"]) for row in victims if row["attack"] == attack),
        }
        for attack in ("adaptive_blend", "adaptive_patch")
    }
    primary = {
        attack: {
            view: {
                "seed_count": int(summary_map[("eligible_only", attack, view, "cf_cmd_u")]["seed_count"]),
                "tpr": float(summary_map[("eligible_only", attack, view, "cf_cmd_u")]["pooled_tpr"]),
                "fpr": float(summary_map[("eligible_only", attack, view, "cf_cmd_u")]["pooled_fpr"]),
                "decision": summary_map[("eligible_only", attack, view, "cf_cmd_u")]["decision"],
            }
            for view in ("qml_checkpoint_drift_xz16", "classical_checkpoint_drift_margin1")
        }
        for attack in ("adaptive_blend", "adaptive_patch")
    }
    return {
        "status": "PASS",
        "confirmation_protocol_sha256": expected_hash,
        "sample_rows": sum(row[4] for row in counts.values()),
        "group_count": len(counts),
        "eligible_victims": eligible,
        "primary_cf_cmd_u": primary,
    }


def verify_cifar() -> dict:
    root = RESULTS / "tifs_native_cifar_gtsrb_v1/cifar10"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, receipt in manifest["output_files"].items():
        path = root / name
        assert path.stat().st_size == int(receipt["bytes"])
        assert sha256(path) == receipt["sha256"]
    victims = read_csv(root / "victim_metrics.csv")
    assert len(victims) == 10
    assert {int(row["seed"]) for row in victims} == set(range(42, 52))
    assert all(int(row["eligible"]) == 1 for row in victims)

    saved_rows = read_csv(root / "detector_metrics.csv")
    saved = {(int(row["seed"]), row["method"]): row for row in saved_rows}
    assert len(saved_rows) == len(saved) == 30
    counts: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    unique: dict[tuple[int, str], set[str]] = defaultdict(set)
    with (root / "per_sample_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["seed"]), row["method"])
            assert int(row["defined"]) == 1
            truth, flag = int(row["poison_truth"]), int(row["flag"])
            counts[key][0] += int(truth == 1 and flag == 1)
            counts[key][1] += int(truth == 1 and flag == 0)
            counts[key][2] += int(truth == 0 and flag == 1)
            counts[key][3] += int(truth == 0 and flag == 0)
            counts[key][4] += 1
            unique[key].add(row["record_id"])
    assert set(counts) == set(saved)
    for key, (tp, fn, fp, tn, rows) in counts.items():
        assert rows == len(unique[key]) == 11_750
        row = saved[key]
        assert (tp, fn, fp, tn) == (int(row["tp"]), int(row["fn"]), int(row["fp"]), int(row["tn_target"]))
        close(tp / (tp + fn), float(row["tpr"]), f"cifar {key} tpr")
        close(fp / (fp + tn), float(row["target_clean_fpr"]), f"cifar {key} target fpr")

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    for method, block in summary["detector_summaries"].items():
        selected = [counts[(seed, method)] for seed in range(42, 52)]
        tp, fn, fp, tn = (sum(row[index] for row in selected) for index in range(4))
        close(tp / (tp + fn), float(block["pooled_tpr"]), f"cifar {method} pooled tpr")
        close(fp / (fp + tn), float(block["pooled_target_clean_fpr"]), f"cifar {method} pooled target fpr")
        close(fp / (10 * 50_000), float(block["pooled_all_clean_fpr"]), f"cifar {method} pooled all fpr")
    return {
        "status": "PASS",
        "protocol_sha256": sha256(root / "frozen_protocol.json"),
        "sample_rows": sum(row[4] for row in counts.values()),
        "group_count": len(counts),
        "eligible_victims": sum(int(row["eligible"]) for row in victims),
        "decision": summary["decision"],
        "detector_summaries": summary["detector_summaries"],
    }


def verify_flare() -> dict:
    root = RESULTS / "tifs_flare_matched_v1"
    manifest = json.loads((root / "manifest_matched.json").read_text(encoding="utf-8"))
    assert sha256(root / "protocol.json") == manifest["protocol_sha256_at_execution"]
    assert sha256(root / "ACCESS_MATRIX.md") == manifest["access_matrix_sha256_at_execution"]
    assert sha256(root / "upstream_FLARE.py") == manifest["upstream_flare_sha256"]
    saved_rows = read_csv(root / "per_seed_metrics_matched.csv")
    saved = {int(row["seed"]): row for row in saved_rows}
    assert set(saved) == {103, 104, 105, 106, 107}
    counts: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    unique: dict[int, set[int]] = defaultdict(set)
    with gzip.open(root / "per_sample_flags.csv.gz", "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            assert row["method"] == "flare_hidden512_matched_adapter"
            seed = int(row["seed"])
            truth, flag = int(row["is_poison"]), int(row["flag"])
            counts[seed][0] += int(truth == 1 and flag == 1)
            counts[seed][1] += int(truth == 1 and flag == 0)
            counts[seed][2] += int(truth == 0 and flag == 1)
            counts[seed][3] += int(truth == 0 and flag == 0)
            counts[seed][4] += 1
            unique[seed].add(int(row["original_index"]))
    assert set(counts) == set(saved)
    for seed, (tp, fn, fp, tn, rows) in counts.items():
        assert rows == len(unique[seed]) == 89_996
        row = saved[seed]
        assert tp == int(row["tp"])
        assert tp + fn == int(row["n_poison"]) == 4_500
        assert fp == int(row["fp_all_clean"])
        assert fp + tn == int(row["n_all_clean"]) == 85_496
    summary = json.loads((root / "summary_matched.json").read_text(encoding="utf-8"))
    flare = summary["flare_hidden512_matched_adapter"]
    total = [sum(row[index] for row in counts.values()) for index in range(4)]
    close(total[0] / (total[0] + total[1]), float(flare["pooled_tpr"]), "flare pooled tpr")
    close(total[2] / (total[2] + total[3]), float(flare["pooled_all_clean_fpr"]), "flare pooled fpr")
    return {
        "status": "PASS",
        "protocol_sha256": sha256(root / "protocol.json"),
        "sample_rows": sum(row[4] for row in counts.values()),
        "seed_count": len(counts),
        "flare_hidden512_matched_adapter": flare,
        "archived_primary_cf_cmd_u": summary["archived_primary_cf_cmd_u"],
        "boundary": "Clean-room hidden512 adapter under matched candidate scope; not paper-primary FLARE-C and not a bit-identical official pipeline.",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "PASS",
        "adaptive_blend_patch": verify_adaptive(),
        "native_cifar10": verify_cifar(),
        "flare_matched": verify_flare(),
        "gtsrb": {"status": "PENDING"},
    }
    (OUT / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# TIFS gap-closure independent validation",
        "",
        "Overall status: **PASS for completed blocks; native GTSRB is pending.**",
        "",
        f"- Adaptive sample rows independently recomputed: {report['adaptive_blend_patch']['sample_rows']:,}.",
        f"- Native CIFAR-10 sample rows independently recomputed: {report['native_cifar10']['sample_rows']:,}; decision: {report['native_cifar10']['decision']}.",
        f"- Matched FLARE sample rows independently recomputed: {report['flare_matched']['sample_rows']:,}.",
        "- Adaptive-Patch remains inconclusive at attack level because fewer than 8/10 victims satisfy the frozen CA/ASR gate.",
        "- FLARE is a clean-room matched hidden-512 adapter; it is not a reproduction of paper-primary FLARE-C.",
        "",
    ]
    (OUT / "VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
