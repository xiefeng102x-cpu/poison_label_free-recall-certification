#!/usr/bin/env python3
"""Verify official-weight warm-start PathMNIST victim-feasibility artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import pathmnist_external_validation as core
import pathmnist_official_resnet18 as architecture
import run_pathmnist_official_weight_warmstart as pilot


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=tolerance))


def verify(
    output: Path,
    dataset: Path,
    *,
    allow_dry_run: bool = False,
) -> dict[str, Any]:
    failures: list[str] = []
    try:
        _, receipt, frozen_hashes = pilot.validate_frozen_inputs(dataset)
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
    if manifest.get("experiment") != (
        "pathmnist_official_weight_warmstart_victim_feasibility_v1"
    ):
        failures.append("experiment identifier mismatch")
    if manifest.get("protocol_sha256") != core.sha256_file(pilot.PROTOCOL):
        failures.append("protocol hash mismatch")
    if manifest.get("frozen_hashes") != frozen_hashes:
        failures.append("manifest frozen hashes mismatch")
    if manifest.get("primary_checkpoint") != pilot.PRIMARY_CHECKPOINT:
        failures.append("primary checkpoint is not fixed final epoch")
    if manifest.get("validation_role") != pilot.VALIDATION_ROLE:
        failures.append("validation role mismatch")
    if manifest.get("test_timing") != "after_fixed_target_epoch_only":
        failures.append("test timing mismatch")
    if manifest.get("test_data_used_during_training_or_selection") is not False:
        failures.append("manifest does not exclude test-based training/selection")
    if manifest.get("trains_clean_pair") is not False:
        failures.append("manifest unexpectedly trains a clean pair")
    if manifest.get("runs_detectors") is not False:
        failures.append("manifest unexpectedly runs detectors")
    if int(manifest.get("poison_count", -1)) != pilot.POISON_COUNT:
        failures.append("poison count mismatch")
    if manifest.get("scheduler", "missing") is not None:
        failures.append("scheduler must be absent")

    if formal:
        if tuple(manifest.get("requested_seeds", [])) != pilot.FORMAL_SEEDS:
            failures.append("formal requested seed order mismatch")
        if int(manifest.get("target_epochs", -1)) != pilot.FORMAL_EPOCHS:
            failures.append("formal epoch count mismatch")
        if manifest.get("max_batches_per_epoch") is not None:
            failures.append("formal run truncated batches")
    elif manifest.get("status") != "DRY_RUN_NONFORMAL":
        failures.append("dry-run status mismatch")

    data = core.load_pathmnist(dataset)
    if formal:
        validation_images = data["val"]["images"]
        validation_labels = data["val"]["labels"]
        test_images = data["test"]["images"]
        test_labels = data["test"]["labels"]
    else:
        val_indices = pilot.balanced_dry_indices(data["val"]["labels"])
        test_indices = pilot.balanced_dry_indices(data["test"]["labels"])
        validation_images = data["val"]["images"][val_indices]
        validation_labels = data["val"]["labels"][val_indices]
        test_images = data["test"]["images"][test_indices]
        test_labels = data["test"]["labels"][test_indices]

    receipt_by_seed = {int(row["seed"]): row for row in receipts}
    metric_by_seed = {int(row["seed"]): row for row in rows}
    if len(receipt_by_seed) != len(receipts) or len(metric_by_seed) != len(rows):
        failures.append("duplicate seed artifact")
    if set(receipt_by_seed) != set(metric_by_seed):
        failures.append("metric/receipt seed sets differ")
    if list(metric_by_seed) != list(manifest.get("completed_seeds", [])):
        failures.append("completed seed order mismatch")

    for seed, seed_receipt in receipt_by_seed.items():
        if seed not in pilot.FORMAL_SEEDS:
            failures.append(f"seed {seed}: outside frozen seed block")
            continue
        pool = core.build_poisoned_training(
            data["train"]["images"],
            data["train"]["labels"],
            seed=seed,
            poison_count=pilot.POISON_COUNT,
            target_class=pilot.TARGET_CLASS,
            patch_size=pilot.PATCH_SIZE,
        )
        if seed_receipt.get("pool") != pool["receipt"]:
            failures.append(f"seed {seed}: poison receipt mismatch")

        expected_model, official_state, official_path = pilot.load_official_initial_state(
            seed, receipt
        )
        del expected_model
        official_state_sha256 = core.state_fingerprint(official_state)
        training = seed_receipt.get("training", {})
        if training.get("official_weight_path") != str(official_path):
            failures.append(f"seed {seed}: official weight path mismatch")
        if training.get("initial_official_state_sha256") != official_state_sha256:
            failures.append(f"seed {seed}: initial official state hash mismatch")
        if training.get("initial_state_bit_identical_to_official") is not True:
            failures.append(f"seed {seed}: initial official equality not attested")
        if training.get("primary_checkpoint") != pilot.PRIMARY_CHECKPOINT:
            failures.append(f"seed {seed}: training selected a non-fixed checkpoint")
        if training.get("validation_role") != pilot.VALIDATION_ROLE:
            failures.append(f"seed {seed}: validation used beyond diagnosis")
        if training.get("test_data_used_during_training_or_selection") is not False:
            failures.append(f"seed {seed}: test-based selection not excluded")
        history = training.get("history", [])
        if len(history) != int(manifest.get("target_epochs", -1)):
            failures.append(f"seed {seed}: history length mismatch")
        elif [int(item["epoch"]) for item in history] != list(
            range(1, int(manifest["target_epochs"]) + 1)
        ):
            failures.append(f"seed {seed}: history epochs are not contiguous")
        elif any(item.get("validation_role") != pilot.VALIDATION_ROLE for item in history):
            failures.append(f"seed {seed}: history validation role mismatch")

        artifact = seed_receipt.get("primary_checkpoint", {})
        artifact_path = Path(artifact.get("path", ""))
        if not artifact_path.exists():
            failures.append(f"seed {seed}: missing primary checkpoint")
            continue
        if core.sha256_file(artifact_path) != artifact.get("file_sha256"):
            failures.append(f"seed {seed}: primary checkpoint file hash mismatch")
            continue
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
        if payload.get("protocol_sha256") != manifest.get("protocol_sha256"):
            failures.append(f"seed {seed}: artifact protocol mismatch")
        if payload.get("frozen_hashes") != frozen_hashes:
            failures.append(f"seed {seed}: artifact frozen hashes mismatch")
        if payload.get("epoch") != int(manifest["target_epochs"]):
            failures.append(f"seed {seed}: artifact epoch mismatch")
        if payload.get("validation_selection") is not False:
            failures.append(f"seed {seed}: artifact claims validation selection")
        state = payload.get("state_dict", {})
        if core.state_fingerprint(state) != artifact.get("state_sha256"):
            failures.append(f"seed {seed}: artifact state hash mismatch")
        model = architecture.PathMNISTResNet18()
        model.load_state_dict(state, strict=True)

        row = metric_by_seed[seed]
        replayed_test = pilot.test_metrics(model, test_images, test_labels)
        for field in ("test_balanced_accuracy", "test_macro_auc", "attack_success_rate"):
            if not close(row[field], replayed_test[field]):
                failures.append(f"seed {seed}: {field} does not replay")
        if (row["eligible"].lower() == "true") != replayed_test["eligible"]:
            failures.append(f"seed {seed}: eligibility does not replay")
        if row.get("validation_selection_used", "").lower() != "false":
            failures.append(f"seed {seed}: metric row implies validation selection")
        if int(row.get("checkpoint_epoch", -1)) != int(manifest["target_epochs"]):
            failures.append(f"seed {seed}: metric checkpoint epoch mismatch")
        if history:
            replayed_validation = pilot.validation_diagnostics(
                model, validation_images, validation_labels
            )
            for field in ("validation_macro_auc", "validation_balanced_accuracy"):
                if not close(history[-1][field], replayed_validation[field]):
                    failures.append(f"seed {seed}: final {field} does not replay")

        latest = output / "resume" / f"seed_{seed}_latest.pt"
        if not latest.exists():
            failures.append(f"seed {seed}: missing exact-resume checkpoint")
        else:
            resume_payload = torch.load(latest, map_location="cpu", weights_only=False)
            expected_contract = pilot.checkpoint_contract(
                protocol_sha256=manifest["protocol_sha256"],
                frozen_hashes=frozen_hashes,
                seed=seed,
                official_state_sha256=official_state_sha256,
                mode=pilot.ExecutionMode(
                    formal=formal,
                    target_epochs=int(manifest["target_epochs"]),
                    max_batches_per_epoch=manifest["max_batches_per_epoch"],
                ),
            )
            if resume_payload.get("contract") != expected_contract:
                failures.append(f"seed {seed}: resume contract mismatch")
            if int(resume_payload.get("epoch", -1)) != int(manifest["target_epochs"]):
                failures.append(f"seed {seed}: resume epoch mismatch")
            saved_initial = resume_payload.get("initial_official_state", {})
            if not pilot.states_bit_identical(saved_initial, official_state):
                failures.append(f"seed {seed}: resume initial state not bit-identical")
            if resume_payload.get("initial_official_state_sha256") != (
                official_state_sha256
            ):
                failures.append(f"seed {seed}: resume initial state hash mismatch")
            if core.state_fingerprint(resume_payload.get("model_state", {})) != (
                artifact.get("state_sha256")
            ):
                failures.append(f"seed {seed}: resume/final model state mismatch")
            resume_model = architecture.PathMNISTResNet18()
            resume_model.load_state_dict(official_state, strict=True)
            resume_optimizer = torch.optim.Adam(
                resume_model.parameters(),
                lr=pilot.LEARNING_RATE,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
            )
            resume_generator = torch.Generator().manual_seed(20261001 + seed)
            try:
                restored = pilot.load_resume_checkpoint(
                    latest,
                    expected_contract=expected_contract,
                    expected_official_state=official_state,
                    model=resume_model,
                    optimizer=resume_optimizer,
                    shuffle_generator=resume_generator,
                )
                if int(restored["epoch"]) != int(manifest["target_epochs"]):
                    failures.append(f"seed {seed}: loaded resume epoch mismatch")
                if core.state_fingerprint(resume_model.state_dict()) != artifact.get(
                    "state_sha256"
                ):
                    failures.append(f"seed {seed}: loaded resume state mismatch")
            except Exception as error:
                failures.append(f"seed {seed}: exact resume replay failed: {error}")

    parsed_rows = [
        {
            "seed": int(row["seed"]),
            "eligible": row["eligible"].lower() == "true",
        }
        for row in rows
    ]
    expected_gate = pilot.compute_gate(parsed_rows, formal=formal)
    if manifest.get("gate") != expected_gate:
        failures.append("gate does not replay from victim metrics")
    expected_status = pilot.status_from_gate(expected_gate, formal=formal)
    if manifest.get("status") != expected_status:
        failures.append("manifest status does not replay from gate")
    if formal and not expected_gate["formal_conclusion_complete"]:
        failures.append("formal artifacts do not reach a permitted terminal state")
    if not formal and expected_gate["authorize_followup"]:
        failures.append("dry run incorrectly authorizes follow-up")

    status = "FAIL"
    if not failures:
        status = "PASS" if formal else "DRY_RUN_VALID_NONFORMAL"
    return {
        "status": status,
        "formal_evidence": bool(formal and not failures),
        "failures": failures,
        "protocol_sha256": core.sha256_file(pilot.PROTOCOL),
        "frozen_hashes": frozen_hashes,
        "completed_seeds": list(metric_by_seed),
        "gate": expected_gate,
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
