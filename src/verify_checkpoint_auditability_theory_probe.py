#!/usr/bin/env python3
"""Replay the frozen checkpoint-auditability mechanism probe artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import analyze_checkpoint_auditability_theory_probe as probe


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(first: float | str, second: float | str, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(float(first), float(second), rtol=0.0, atol=tolerance))


def verify(result_root: Path, source_root: Path) -> dict:
    manifest = json.loads((result_root / "manifest.json").read_text(encoding="utf-8"))
    decision = json.loads((result_root / "decision.json").read_text(encoding="utf-8"))
    protocol = json.loads(probe.PROTOCOL.read_text(encoding="utf-8"))
    assert manifest["protocol_sha256"] == probe.sha256_file(probe.PROTOCOL)
    assert manifest["source_manifest_sha256"] == probe.sha256_file(
        source_root / "manifest.json"
    )
    assert manifest["source_splits_sha256"] == probe.sha256_file(source_root / "splits.csv")
    for row in manifest["source_cache_hashes"]:
        assert probe.sha256_file(Path(row["path"])) == row["sha256"]

    reported = read_csv(result_root / "per_view_metrics.csv")
    comparisons = read_csv(result_root / "paired_view_comparisons.csv")
    aggregate = read_csv(result_root / "aggregate_auc_deltas.csv")
    assert len(reported) == 36
    assert len(comparisons) == 18
    assert len(aggregate) == 6
    lookup = {
        (int(row["seed"]), row["checkpoint"], row["representation"], row["view"]): row
        for row in reported
    }
    assert len(lookup) == len(reported)

    split_map = probe.load_split_map(source_root / "splits.csv")
    indices = probe.pauli_indices(5)
    replayed_count = 0
    for seed in probe.SEEDS:
        for checkpoint in probe.CHECKPOINTS:
            cache = probe.load_pair_cache(source_root, seed, checkpoint)
            masks = probe.split_masks(seed, cache["pair_ids"], split_map)
            for representation in probe.REPRESENTATIONS:
                columns = indices[representation]
                values = {
                    "suspect_absolute": (
                        cache["suspect_clean"][:, columns],
                        cache["suspect_attack"][:, columns],
                    ),
                    "paired_checkpoint_drift": (
                        cache["suspect_clean"][:, columns] - cache["clean_clean"][:, columns],
                        cache["suspect_attack"][:, columns] - cache["clean_attack"][:, columns],
                    ),
                }
                for view, (clean, attack) in values.items():
                    metrics, _ = probe.analyze_view(
                        clean,
                        attack,
                        masks,
                        alpha=float(protocol["outputs"]["rank_alpha"]),
                        ridge_fraction=float(
                            protocol["estimation"]["ridge_fraction_of_mean_variance"]
                        ),
                    )
                    row = lookup[(seed, checkpoint, representation, view)]
                    for field, value in metrics.items():
                        assert close(row[field], value), (seed, checkpoint, representation, view, field)
                    replayed_count += 1

    recomputed_aggregate = {}
    for checkpoint in probe.CHECKPOINTS:
        for representation in probe.REPRESENTATIONS:
            values = [
                float(row["auc_delta"])
                for row in comparisons
                if row["checkpoint"] == checkpoint and row["representation"] == representation
            ]
            recomputed_aggregate[(checkpoint, representation)] = {
                "mean_auc_delta": float(np.mean(values)),
                "minimum_auc_delta": float(np.min(values)),
                "maximum_auc_delta": float(np.max(values)),
                "all_seeds_nonnegative": all(value >= 0.0 for value in values),
            }
    for row in aggregate:
        values = recomputed_aggregate[(row["checkpoint"], row["representation"])]
        for field in ("mean_auc_delta", "minimum_auc_delta", "maximum_auc_delta"):
            assert close(row[field], values[field])
        assert (row["all_seeds_nonnegative"] == "True") is values["all_seeds_nonnegative"]

    positive_both = [
        representation
        for representation in probe.REPRESENTATIONS
        if all(
            recomputed_aggregate[(checkpoint, representation)]["mean_auc_delta"] > 0.0
            for checkpoint in probe.CHECKPOINTS
        )
    ]
    sign_consistent = [
        representation
        for representation in probe.REPRESENTATIONS
        if all(
            recomputed_aggregate[(checkpoint, representation)]["all_seeds_nonnegative"]
            for checkpoint in probe.CHECKPOINTS
        )
    ]
    assert decision["positive_mean_delta_both_checkpoints"] == positive_both
    assert decision["nonnegative_delta_every_seed_both_checkpoints"] == sign_consistent
    assert decision["supports_conditional_representation_hypothesis"] is False
    assert decision["proceed_to_fresh_warm_start_pilot"] is False

    return {
        "status": "PASS",
        "replayed_view_configurations": replayed_count,
        "replayed_comparisons": len(comparisons),
        "protocol_sha256": manifest["protocol_sha256"],
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=probe.DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=probe.DEFAULT_SOURCE)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=probe.PAPER14_ROOT
        / "results/top_journal_program_v1/checkpoint_auditability_theory_receipt.json",
    )
    args = parser.parse_args()
    receipt = verify(args.results, args.source)
    probe.write_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
