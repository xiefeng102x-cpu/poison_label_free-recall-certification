#!/usr/bin/env python3
"""Verify PathMNIST full-data ResNet-18 victim-pilot artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import pathmnist_external_validation as core
import run_pathmnist_resnet18_victim_pilot as pilot


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify(
    output: Path,
    dataset: Path,
    *,
    allow_dry_run: bool = False,
) -> dict[str, Any]:
    failures: list[str] = []
    try:
        _, source_hashes = pilot.validate_frozen_inputs(dataset)
    except Exception as error:
        return {"status": "FAIL", "formal_evidence": False, "failures": [str(error)]}
    manifest_path = output / "manifest.json"
    metrics_path = output / "victim_metrics.csv"
    receipts_path = output / "seed_receipts.json"
    for path in (manifest_path, metrics_path, receipts_path):
        if not path.exists():
            failures.append(f"missing artifact: {path}")
    if failures:
        return {"status": "FAIL", "formal_evidence": False, "failures": failures}

    manifest = pilot.read_json(manifest_path)
    rows = read_csv(metrics_path)
    receipts = pilot.read_json(receipts_path)
    formal = manifest.get("formal") is True and manifest.get("dry_run") is False
    if not formal and not allow_dry_run:
        failures.append("artifacts are a non-formal dry run")
    if manifest.get("experiment") != "pathmnist_full_data_official_resnet18_victim_pilot_v1":
        failures.append("experiment identifier mismatch")
    if manifest.get("protocol_sha256") != core.sha256_file(pilot.PROTOCOL):
        failures.append("protocol hash mismatch")
    if manifest.get("source_hashes") != source_hashes:
        failures.append("manifest source hashes mismatch")
    if manifest.get("test_data_used_during_training_or_selection") is not False:
        failures.append("manifest does not exclude test-based selection")
    if int(manifest.get("poison_count", -1)) != pilot.poison_count_for_training(89996):
        failures.append("poison count mismatch")
    if formal:
        if tuple(manifest.get("seeds", [])) != pilot.FORMAL_SEEDS:
            failures.append("formal seed block mismatch")
        if int(manifest.get("target_epochs", -1)) != pilot.FORMAL_EPOCHS:
            failures.append("formal epoch count mismatch")
        if manifest.get("status") != "FORMAL_COMPLETE":
            failures.append("formal completion status mismatch")
    elif manifest.get("status") != "DRY_RUN_NONFORMAL":
        failures.append("dry-run status mismatch")

    data = core.load_pathmnist(dataset)
    receipt_by_seed = {int(row["seed"]): row for row in receipts}
    metric_by_seed = {int(row["seed"]): row for row in rows}
    if set(receipt_by_seed) != set(metric_by_seed):
        failures.append("metric/receipt seed sets differ")
    for seed, receipt in receipt_by_seed.items():
        pool = core.build_poisoned_training(
            data["train"]["images"],
            data["train"]["labels"],
            seed=seed,
            poison_count=pilot.poison_count_for_training(len(data["train"]["labels"])),
            target_class=pilot.TARGET_CLASS,
            patch_size=pilot.PATCH_SIZE,
        )
        if receipt.get("pool", {}).get("selected_indices_sha256") != pool["receipt"][
            "selected_indices_sha256"
        ]:
            failures.append(f"seed {seed}: poison index hash mismatch")
        training = receipt.get("training", {})
        if training.get("selection_metric") != pilot.SELECTION_METRIC:
            failures.append(f"seed {seed}: selection metric mismatch")
        if training.get("selection_data_split") != "clean_validation_only":
            failures.append(f"seed {seed}: selection split mismatch")
        if training.get("test_data_used_during_training_or_selection") is not False:
            failures.append(f"seed {seed}: test-based selection was not excluded")
        for artifact_name in ("primary_checkpoint", "secondary_checkpoint"):
            artifact = receipt.get(artifact_name, {})
            path = Path(artifact.get("path", ""))
            if not path.exists():
                failures.append(f"seed {seed}: missing {artifact_name}")
                continue
            if core.sha256_file(path) != artifact.get("file_sha256"):
                failures.append(f"seed {seed}: {artifact_name} file hash mismatch")
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("protocol_sha256") != manifest.get("protocol_sha256"):
                failures.append(f"seed {seed}: {artifact_name} protocol mismatch")
            if payload.get("runner_source_sha256") != source_hashes["runner_source_sha256"]:
                failures.append(f"seed {seed}: {artifact_name} runner mismatch")
            if core.state_fingerprint(payload["state_dict"]) != artifact.get(
                "state_sha256"
            ):
                failures.append(f"seed {seed}: {artifact_name} state mismatch")
        latest = output / "resume" / f"seed_{seed}_latest.pt"
        if not latest.exists():
            failures.append(f"seed {seed}: missing exact-resume checkpoint")
        else:
            payload = torch.load(latest, map_location="cpu", weights_only=False)
            expected_contract = pilot.checkpoint_contract(
                protocol_sha256=manifest["protocol_sha256"],
                source_hashes=source_hashes,
                seed=seed,
                mode=pilot.ExecutionMode(
                    formal=formal,
                    target_epochs=int(manifest["target_epochs"]),
                    max_batches_per_epoch=manifest["max_batches_per_epoch"],
                ),
            )
            if payload.get("contract") != expected_contract:
                failures.append(f"seed {seed}: resume contract mismatch")
            if int(payload.get("epoch", -1)) != int(manifest["target_epochs"]):
                failures.append(f"seed {seed}: resume epoch mismatch")

    parsed_for_gate = [
        {
            "seed": int(row["seed"]),
            "eligible": row["eligible"].lower() == "true",
        }
        for row in rows
    ]
    expected_gate = pilot.compute_gate(parsed_for_gate, formal=formal)
    if manifest.get("gate") != expected_gate:
        failures.append("gate does not replay from victim metrics")
    if not formal and manifest.get("gate", {}).get("authorize_followup") is not False:
        failures.append("dry run incorrectly authorizes follow-up")

    status = "FAIL"
    if not failures:
        status = "PASS" if formal else "DRY_RUN_VALID_NONFORMAL"
    return {
        "status": status,
        "formal_evidence": bool(formal and not failures),
        "failures": failures,
        "protocol_sha256": core.sha256_file(pilot.PROTOCOL),
        "source_hashes": source_hashes,
        "seeds": sorted(receipt_by_seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=pilot.DEFAULT_OUTPUT)
    parser.add_argument("--dataset", type=Path, default=pilot.DEFAULT_DATASET)
    parser.add_argument("--allow-dry-run", action="store_true")
    args = parser.parse_args()
    result = verify(args.output, args.dataset, allow_dry_run=args.allow_dry_run)
    destination = args.output / "verification.json"
    pilot.atomic_write_json(destination, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

