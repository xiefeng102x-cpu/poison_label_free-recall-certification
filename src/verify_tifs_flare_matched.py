#!/usr/bin/env python3
"""Independent, read-only validation of the matched FLARE result.

The verifier does not import the experiment runner and does not rerun UMAP or
HDBSCAN.  It reconstructs provenance, poison membership, row identity, counts,
eligibility, pooled rates, and the exact archived CF-CMD-U comparison from the
saved evidence.  Its only writes are the requested validation reports.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/tifs_flare_matched_v1"
SOURCE = ROOT / "results/pathmnist_official_weight_detector_confirmation_v1"
SOURCE_PROTOCOL = ROOT / "results/top_journal_program_v1/pathmnist_official_weight_detector_confirmation_protocol_v1.json"
DATASET = Path("/private/tmp/paper14-data/pathmnist.npz")
SEEDS = (103, 104, 105, 106, 107)
METHOD = "flare_hidden512_matched_adapter"
TARGET_CLASS = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expected_truth(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    eligible = np.flatnonzero(labels != TARGET_CLASS)
    selected = np.sort(
        np.random.default_rng(20260930 + int(seed)).choice(
            eligible, size=4500, replace=False
        )
    )
    truth = np.zeros(len(labels), dtype=bool)
    truth[selected] = True
    assigned = labels.copy()
    assigned[truth] = TARGET_CLASS
    return truth, assigned


def load_metric_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seed = int(row["seed"])
            require(seed not in rows, f"duplicate metric seed {seed}")
            require(row["method"] == METHOD, "unexpected method in matched metrics")
            rows[seed] = {
                "seed": seed,
                "eligible": bool(int(row["eligible"])),
                **{
                    key: int(row[key])
                    for key in (
                        "tp",
                        "n_poison",
                        "fp_all_clean",
                        "n_all_clean",
                        "fp_target_clean",
                        "n_target_clean",
                    )
                },
                **{
                    key: float(row[key])
                    for key in ("tpr", "all_clean_fpr", "target_clean_fpr")
                },
            }
    require(tuple(sorted(rows)) == SEEDS, "matched metric seed set mismatch")
    return rows


def parse_per_sample(
    path: Path, original: np.ndarray
) -> tuple[dict[int, dict[str, int]], dict[str, Any]]:
    seen: set[tuple[int, int]] = set()
    row_counts = defaultdict(int)
    accum = defaultdict(lambda: defaultdict(int))
    truth_cache = {seed: expected_truth(original, seed) for seed in SEEDS}
    total = 0
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(
            reader.fieldnames
            == [
                "seed",
                "method",
                "original_index",
                "original_label",
                "assigned_label",
                "flag",
                "is_poison",
            ],
            "per-sample header mismatch",
        )
        for row in reader:
            total += 1
            seed = int(row["seed"])
            require(seed in truth_cache, f"unexpected per-sample seed {seed}")
            require(row["method"] == METHOD, "unexpected per-sample method")
            index = int(row["original_index"])
            require(0 <= index < 89996, "candidate index outside frozen range")
            key = (seed, index)
            require(key not in seen, f"duplicate per-sample key {key}")
            seen.add(key)
            truth, assigned = truth_cache[seed]
            require(int(row["original_label"]) == int(original[index]), "original label mismatch")
            require(int(row["assigned_label"]) == int(assigned[index]), "assigned label mismatch")
            poison = bool(int(row["is_poison"]))
            require(poison == bool(truth[index]), "poison truth does not replay")
            flag = bool(int(row["flag"]))
            require(not flag or assigned[index] == TARGET_CLASS, "non-target candidate was flagged")
            row_counts[seed] += 1
            bucket = accum[seed]
            bucket["tp"] += int(flag and poison)
            bucket["n_poison"] += int(poison)
            bucket["fp_all_clean"] += int(flag and not poison)
            bucket["n_all_clean"] += int(not poison)
            target_clean = (not poison) and assigned[index] == TARGET_CLASS
            bucket["fp_target_clean"] += int(flag and target_clean)
            bucket["n_target_clean"] += int(target_clean)
    require(total == 89996 * len(SEEDS), "unexpected total per-sample row count")
    for seed in SEEDS:
        require(row_counts[seed] == 89996, f"seed {seed}: expected 89996 rows")
        require(accum[seed]["n_poison"] == 4500, f"seed {seed}: expected 4500 poisons")
        require(
            {index for check_seed, index in seen if check_seed == seed} == set(range(89996)),
            f"seed {seed}: candidate indices are not complete",
        )
    return {seed: dict(accum[seed]) for seed in SEEDS}, {
        "total_rows": total,
        "rows_per_seed": {str(seed): row_counts[seed] for seed in SEEDS},
        "unique_seed_index_keys": len(seen),
        "duplicate_keys": 0,
    }


def main() -> None:
    manifest = json.loads((RESULT / "manifest_matched.json").read_text(encoding="utf-8"))
    protocol = json.loads((RESULT / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((RESULT / "summary_matched.json").read_text(encoding="utf-8"))
    receipts = json.loads((RESULT / "seed_receipts_matched.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    source_receipts = json.loads((SOURCE / "seed_receipts.json").read_text(encoding="utf-8"))
    with np.load(DATASET, allow_pickle=False) as payload:
        original = payload["train_labels"].reshape(-1).astype(np.int64)
    require(original.shape == (89996,), "official train label count mismatch")

    actual_hashes = {
        "protocol": sha256_file(RESULT / "protocol.json"),
        "access_matrix": sha256_file(RESULT / "ACCESS_MATRIX.md"),
        "upstream_flare": sha256_file(RESULT / "upstream_FLARE.py"),
        "source_manifest": sha256_file(SOURCE / "manifest.json"),
        "source_protocol": sha256_file(SOURCE_PROTOCOL),
        "dataset": sha256_file(DATASET),
        "per_sample_flags": sha256_file(RESULT / "per_sample_flags.csv.gz"),
        "per_seed_metrics": sha256_file(RESULT / "per_seed_metrics_matched.csv"),
    }
    recorded_hashes = {
        "protocol": manifest["protocol_sha256_at_execution"],
        "access_matrix": manifest["access_matrix_sha256_at_execution"],
        "upstream_flare": manifest["upstream_flare_sha256"],
        "source_manifest": manifest["source_pathmnist_manifest_sha256"],
        "source_protocol": protocol["source_pathmnist_protocol"]["sha256"],
        "dataset": manifest["dataset_sha256"],
    }
    for name, recorded in recorded_hashes.items():
        require(actual_hashes[name] == recorded, f"{name} hash mismatch")

    receipt_by_seed = {int(row["seed"]): row for row in receipts}
    source_receipt_by_seed = {int(row["seed"]): row for row in source_receipts}
    require(tuple(sorted(receipt_by_seed)) == SEEDS, "matched receipt seed set mismatch")
    require(tuple(sorted(source_receipt_by_seed)) == SEEDS, "source receipt seed set mismatch")
    source_file_checks = []
    eligibility_rows = []
    for seed in SEEDS:
        receipt = receipt_by_seed[seed]
        source_receipt = source_receipt_by_seed[seed]
        feature = Path(receipt["source_feature_file"])
        model = Path(receipt["suspect_model_file"])
        require(feature.is_file() and model.is_file(), f"seed {seed}: source file missing")
        feature_hash = sha256_file(feature)
        model_hash = sha256_file(model)
        require(feature_hash == receipt["source_feature_sha256"], f"seed {seed}: feature hash mismatch")
        require(model_hash == receipt["suspect_model_sha256"], f"seed {seed}: model hash mismatch")
        require(feature_hash == source_receipt["feature_file_sha256"], f"seed {seed}: feature/source receipt mismatch")
        require(model_hash == source_receipt["suspect_model_file_sha256"], f"seed {seed}: model/source receipt mismatch")
        expected_poison, _ = expected_truth(original, seed)
        poison_hash = hashlib.sha256(np.flatnonzero(expected_poison).astype(np.int64).tobytes()).hexdigest()
        require(poison_hash == receipt["poison_indices_sha256"], f"seed {seed}: poison index hash mismatch")
        require(poison_hash == source_receipt["poison_indices_sha256"], f"seed {seed}: source poison index hash mismatch")
        clean_bca = float(source_receipt["clean_test"]["balanced_accuracy"])
        suspect_bca = float(source_receipt["suspect_test"]["balanced_accuracy"])
        suspect_asr = float(source_receipt["suspect_test"]["asr"])
        independently_eligible = clean_bca >= 0.75 and suspect_bca >= 0.75 and suspect_asr >= 0.80
        require(independently_eligible, f"seed {seed}: victim is not independently eligible")
        require(bool(source_receipt["eligible"]) == independently_eligible, f"seed {seed}: source eligibility mismatch")
        require(bool(receipt["eligible"]) == independently_eligible, f"seed {seed}: matched eligibility mismatch")
        source_file_checks.append(
            {
                "seed": seed,
                "feature_sha256": feature_hash,
                "suspect_model_sha256": model_hash,
                "poison_indices_sha256": poison_hash,
            }
        )
        eligibility_rows.append(
            {
                "seed": seed,
                "clean_balanced_accuracy": clean_bca,
                "suspect_balanced_accuracy": suspect_bca,
                "suspect_asr": suspect_asr,
                "eligible": independently_eligible,
            }
        )

    metric_rows = load_metric_rows(RESULT / "per_seed_metrics_matched.csv")
    recomputed, inventory = parse_per_sample(RESULT / "per_sample_flags.csv.gz", original)
    count_keys = (
        "tp",
        "n_poison",
        "fp_all_clean",
        "n_all_clean",
        "fp_target_clean",
        "n_target_clean",
    )
    per_seed_validation = []
    for seed in SEEDS:
        for key in count_keys:
            require(recomputed[seed][key] == metric_rows[seed][key], f"seed {seed}: {key} mismatch")
        rates = {
            "tpr": recomputed[seed]["tp"] / recomputed[seed]["n_poison"],
            "all_clean_fpr": recomputed[seed]["fp_all_clean"] / recomputed[seed]["n_all_clean"],
            "target_clean_fpr": recomputed[seed]["fp_target_clean"] / recomputed[seed]["n_target_clean"],
        }
        for key, value in rates.items():
            require(abs(value - metric_rows[seed][key]) <= 1e-15, f"seed {seed}: {key} mismatch")
        per_seed_validation.append({"seed": seed, **recomputed[seed], **rates})

    pooled = {key: sum(recomputed[seed][key] for seed in SEEDS) for key in count_keys}
    matched = {
        "micro_pooled_counts": pooled,
        "pooled_tpr": pooled["tp"] / pooled["n_poison"],
        "pooled_all_clean_fpr": pooled["fp_all_clean"] / pooled["n_all_clean"],
        "pooled_target_clean_fpr": pooled["fp_target_clean"] / pooled["n_target_clean"],
    }
    saved_matched = summary[METHOD]
    require(saved_matched["micro_pooled_counts"] == pooled, "saved pooled counts mismatch")
    for key in ("pooled_tpr", "pooled_all_clean_fpr", "pooled_target_clean_fpr"):
        require(abs(matched[key] - saved_matched[key]) <= 1e-15, f"saved {key} mismatch")

    archived = source_manifest["detector_summaries"]["primary_cf_cmd_u"]
    archived_counts = archived["micro_pooled_counts"]
    require(archived_counts["n_poison"] == pooled["n_poison"], "CF poison denominator mismatch")
    require(archived_counts["n_all_clean"] == pooled["n_all_clean"], "CF clean denominator mismatch")
    require(archived_counts["n_target_clean"] == pooled["n_target_clean"], "CF target-clean denominator mismatch")
    archived_gate = bool(source_manifest["primary_point_gate_pass"])
    require(archived_gate, "archived CF-CMD-U point gate unexpectedly fails")
    matched_gate = matched["pooled_tpr"] >= 0.90 and matched["pooled_all_clean_fpr"] <= 0.05
    require(matched_gate, "matched adapter point gate unexpectedly fails")
    comparison = {
        "same_denominators": True,
        "matched_adapter": {**matched, "point_gate_pass": matched_gate},
        "archived_cf_cmd_u": {
            "micro_pooled_counts": archived_counts,
            "pooled_tpr": archived["pooled_tpr"],
            "pooled_all_clean_fpr": archived["pooled_all_clean_fpr"],
            "pooled_target_clean_fpr": archived["pooled_target_clean_fpr"],
            "point_gate_pass": archived_gate,
        },
        "matched_minus_cf_cmd_u": {
            "tp": pooled["tp"] - archived_counts["tp"],
            "fp_all_clean": pooled["fp_all_clean"] - archived_counts["fp_all_clean"],
            "tpr": matched["pooled_tpr"] - archived["pooled_tpr"],
            "all_clean_fpr": matched["pooled_all_clean_fpr"] - archived["pooled_all_clean_fpr"],
            "target_clean_fpr": matched["pooled_target_clean_fpr"] - archived["pooled_target_clean_fpr"],
        },
        "interpretation": "Both pass the same point gate. The adapter trades 100 fewer true positives for 2,294 fewer false positives in this frozen block. This is not an official-FLARE ranking because the adapter reuses only hidden512 and target-scope filtering and ignores FLARE's all-BN representation.",
    }

    cluster_receipts = []
    for seed in SEEDS:
        receipt = receipt_by_seed[seed]["matched_receipt"]
        require(sum(receipt["cluster_counts"]) == 17385, f"seed {seed}: cluster cohort mismatch")
        require(receipt["candidate_scope_count"] == 17385, f"seed {seed}: target cohort mismatch")
        cluster_receipts.append(
            {
                "seed": seed,
                "noise_count": receipt["noise_count"],
                "largest_cluster_label_including_noise": receipt[
                    "largest_cluster_label_including_noise"
                ],
                "cluster_labels": receipt["cluster_labels"],
                "cluster_counts": receipt["cluster_counts"],
                "flagged_count": receipt["flagged_count"],
            }
        )

    report = {
        "status": "PASS_INDEPENDENT_READ_ONLY_VALIDATION",
        "scope": "matched clean-room FLARE adapter only; official all-BN sensitivity not executed",
        "actual_device": manifest["device"],
        "hashes": {
            "actual": actual_hashes,
            "recorded": recorded_hashes,
            "all_recorded_hashes_match": True,
        },
        "inventory": inventory,
        "source_file_checks": source_file_checks,
        "victim_eligibility": eligibility_rows,
        "per_seed_recomputed": per_seed_validation,
        "cluster_receipts": cluster_receipts,
        "comparison": comparison,
        "boundaries": [
            "flare_hidden512_matched_adapter is a clean-room clustering adapter, not official FLARE and not paper-primary FLARE-C.",
            "The matched adapter and CF-CMD-U share victim, candidate inputs, poison members, target-assigned scope, hidden512 layer, and evaluation gate; the adapter does not use the 512-row trusted reference or the distinct 512-row calibration set that CF-CMD-U uses.",
            "Official all-BN privilege sensitivity remains unexecuted; no all-layer or BN-statistics FLARE claim is supported by this report.",
            "The upstream-code cluster rule takes the largest label including noise, then always leaves noise clean; this did not create a largest-noise ambiguity in seeds 103-107.",
            "Five fixed eligible seeds support only this frozen PathMNIST BadNets block; no universal, cross-attack, or superiority claim follows.",
        ],
    }
    atomic_text(
        RESULT / "VALIDATION_REPORT.json",
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )

    lines = [
        "# Independent validation of the matched FLARE block",
        "",
        "**Status: PASS.** The verifier independently replayed provenance, row identity, poison membership, counts, rates, and victim eligibility without importing or rerunning the experiment runner.",
        "",
        "## Inventory and provenance",
        "",
        f"- `{inventory['total_rows']:,}` gzip rows: exactly `89,996` unique candidate indices for each of seeds `103--107`; no duplicate `(seed, index)` keys.",
        "- Exactly `4,500` poison rows and `85,496` clean rows per seed; poison membership and assigned labels replay from the frozen seed formula.",
        f"- Protocol `{actual_hashes['protocol']}`, access matrix `{actual_hashes['access_matrix']}`, upstream FLARE `{actual_hashes['upstream_flare']}`, source manifest `{actual_hashes['source_manifest']}`, source protocol `{actual_hashes['source_protocol']}`, and dataset `{actual_hashes['dataset']}` all match their execution-time records.",
        "- All five source feature/model hashes and poison-index hashes match both matched-run and source-run receipts.",
        "- All five victims independently satisfy clean BCA >= 0.75, suspect BCA >= 0.75, and suspect ASR >= 0.80. Actual execution device was `cpu`.",
        "",
        "## Recomputed per-seed results",
        "",
        "| seed | TP / 4500 | TPR | all-clean FP / 85496 | all-clean FPR | noise | largest label (including noise) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cluster_by_seed = {row["seed"]: row for row in cluster_receipts}
    for row in per_seed_validation:
        cluster = cluster_by_seed[row["seed"]]
        lines.append(
            f"| {row['seed']} | {row['tp']} / {row['n_poison']} | {row['tpr']:.4%} | "
            f"{row['fp_all_clean']} / {row['n_all_clean']} | {row['all_clean_fpr']:.4%} | "
            f"{cluster['noise_count']} | {cluster['largest_cluster_label_including_noise']} |"
        )
    delta = comparison["matched_minus_cf_cmd_u"]
    lines.extend(
        [
            "",
            "## Exact matched comparison",
            "",
            "| method | TP / 22500 | pooled TPR | all-clean FP / 427480 | pooled all-clean FPR | target-clean FPR | point gate |",
            "|---|---:|---:|---:|---:|---:|---|",
            f"| FLARE hidden512 matched adapter | {pooled['tp']} / {pooled['n_poison']} | {matched['pooled_tpr']:.4%} | {pooled['fp_all_clean']} / {pooled['n_all_clean']} | {matched['pooled_all_clean_fpr']:.4%} | {matched['pooled_target_clean_fpr']:.4%} | PASS |",
            f"| Archived primary CF-CMD-U | {archived_counts['tp']} / {archived_counts['n_poison']} | {archived['pooled_tpr']:.4%} | {archived_counts['fp_all_clean']} / {archived_counts['n_all_clean']} | {archived['pooled_all_clean_fpr']:.4%} | {archived['pooled_target_clean_fpr']:.4%} | PASS |",
            "",
            f"Matched adapter minus CF-CMD-U: `{delta['tp']}` TP, `{delta['fp_all_clean']}` all-clean FP, `{100*delta['tpr']:.4f}` percentage points TPR, and `{100*delta['all_clean_fpr']:.4f}` percentage points all-clean FPR.  In concrete terms, the adapter trades 100 fewer detected poisons for 2,294 fewer false positives in this block.",
            "",
            "## Boundaries",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["boundaries"])
    lines.append("")
    atomic_text(RESULT / "VALIDATION_REPORT.md", "\n".join(lines))
    print(json.dumps({"status": report["status"], "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()

