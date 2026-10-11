#!/usr/bin/env python3
"""Independent replay verifier for the P3 adaptive confirmation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_group_coherent_10seeds as baseline
import run_bloodmnist_whitebox_adaptive_confirmation as confirmation
import run_bloodmnist_whitebox_adaptive_pilot as pilot
import train_bloodmnist_clean_encoder as clean_encoder


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = PAPER14_ROOT / "results/bloodmnist_whitebox_adaptive_confirmation_v1"
DEFAULT_RECEIPT = PAPER14_ROOT / "results/top_journal_program_v1/phase3_adaptive_receipt.json"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def reconstructed_joint(seed_data: dict, checkpoint: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    prepared = seed_data["prepared"]
    base = prepared["pca_splits"][split][0]
    if split == "train":
        base = base[seed_data["poison_indices"]]
    # Replay the training path exactly in float32.  sklearn's stored scaler
    # arrays are float64, so a NumPy expression would preserve decisions but
    # not the byte-exact artifact fingerprints.
    base_t = torch.as_tensor(base, dtype=torch.float32)
    initial_t = torch.as_tensor(checkpoint["initial_pca_trigger"], dtype=torch.float32)
    diagonal_t, bias_t = [
        torch.as_tensor(value, dtype=torch.float32)
        for value in checkpoint["trigger_corrector_state"]
    ]
    mean_t = torch.as_tensor(prepared["post_scaler"].mean_, dtype=torch.float32)
    scale_t = torch.as_tensor(prepared["post_scaler"].scale_, dtype=torch.float32)
    standardized = (base_t - mean_t) / scale_t
    correction = 0.2 * torch.tanh(standardized * diagonal_t + bias_t)
    trigger_t = torch.clamp(initial_t[None, :] + correction, -0.8, 0.8)
    triggered_t = torch.clamp(base_t + trigger_t, -3.0, 3.0)
    processed_t = (triggered_t - mean_t) / scale_t
    return processed_t.numpy(), trigger_t.numpy()


def verify(args: argparse.Namespace) -> dict:
    result = args.result.resolve()
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    protocol = Path(manifest["protocol"])
    errors: list[str] = []
    if pilot.sha256_file(protocol) != manifest["protocol_sha256"]:
        errors.append("protocol hash mismatch")
    dataset = Path(manifest["dataset_root"]) / "bloodmnist.npz"
    if pilot.sha256_file(dataset) != manifest["dataset_npz_sha256"]:
        errors.append("dataset hash mismatch")
    if manifest["seeds"] != list(confirmation.SEEDS):
        errors.append("fresh seed block mismatch")
    if manifest["modes"] != list(pilot.MODES):
        errors.append("adaptive mode list mismatch")

    metric_rows = read_csv(result / "per_config_metrics.csv")
    sample_rows = read_csv(result / "per_sample_scores.csv")
    victim_rows = read_csv(result / "victim_metrics.csv")
    summary_rows = read_csv(result / "eligible_summary.csv")
    if len(metric_rows) != 90:
        errors.append(f"expected 90 metric rows, found {len(metric_rows)}")
    if len(sample_rows) != 59400:
        errors.append(f"expected 59400 sample rows, found {len(sample_rows)}")
    if len(victim_rows) != 30:
        errors.append(f"expected 30 victim rows, found {len(victim_rows)}")

    metrics = {
        (row["trigger"], int(row["seed"]), row["representation"]): row
        for row in metric_rows
    }
    samples: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in sample_rows:
        samples[(row["trigger"], int(row["seed"]), row["representation"])].append(row)
    victims = {(row["mode"], int(row["seed"])): row for row in victim_rows}
    receipts = {
        (row["mode"], int(row["seed"])): row for row in manifest["receipts"]
    }

    class_data = heldout.collect_classes(Path(manifest["dataset_root"]))
    rgb_trigger, _ = baseline.fixed_rgb_trigger(class_data)
    replayed_samples = 0
    replayed_checkpoints = 0
    replayed_poison_fingerprints = 0

    for seed in confirmation.SEEDS:
        seed_data = pilot.prepare_seed(
            class_data,
            rgb_trigger,
            seed,
            clean_root=Path(manifest["clean_qml_root"]),
        )
        for mode in pilot.MODES:
            key = (mode, seed)
            victim = victims[key]
            receipt = receipts[key]
            checkpoint_path = result / victim["checkpoint"]
            if pilot.sha256_file(checkpoint_path) != victim["checkpoint_sha256"]:
                errors.append(f"checkpoint hash mismatch: {mode} seed {seed}")
                continue
            if receipt["checkpoint_sha256"] != victim["checkpoint_sha256"]:
                errors.append(f"receipt checkpoint mismatch: {mode} seed {seed}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = clean_encoder.CleanQNN("local", 0)
            model.load_state_dict(checkpoint["state_dict"])
            replayed_checkpoints += 1

            if mode == "joint_input_internal_detector_evasion":
                poison, final_trigger = reconstructed_joint(seed_data, checkpoint, "train")
                test_triggered, _ = reconstructed_joint(seed_data, checkpoint, "test")
            else:
                poison = seed_data["splits"]["archived_poison"]
                final_trigger = np.asarray(checkpoint["pca_trigger"], dtype=np.float32)
                test_triggered = seed_data["splits"]["test"]["triggered_non_target"]
            if pilot.sha256_array(poison) != receipt["poison_sha256_float32"]:
                errors.append(f"poison fingerprint mismatch: {mode} seed {seed}")
            if pilot.sha256_array(final_trigger) != receipt["final_trigger_sha256_float32"]:
                errors.append(f"trigger fingerprint mismatch: {mode} seed {seed}")
            replayed_poison_fingerprints += 1

            test_ca, test_asr = pilot.balanced_ca_and_asr(
                model,
                seed_data["splits"]["test"]["target"],
                seed_data["splits"]["test"]["non_target"],
                test_triggered,
            )
            if not close(test_ca, victim["test_balanced_ca"]):
                errors.append(f"test CA mismatch: {mode} seed {seed}")
            if not close(test_asr, victim["test_asr"]):
                errors.append(f"test ASR mismatch: {mode} seed {seed}")
            eligible = int(test_ca >= 0.90 and test_asr >= 0.80)
            if eligible != int(victim["test_eligible"]):
                errors.append(f"eligibility mismatch: {mode} seed {seed}")

            replay, features = pilot.detector_evaluation(
                mode=mode,
                seed=seed,
                clean_model=seed_data["clean_model"],
                attacked_model=model,
                trusted=seed_data["splits"]["val"]["target"],
                candidate_clean=seed_data["splits"]["train"]["target"],
                poison=poison,
                alpha=float(manifest["alpha"]),
            )
            if pilot.sha256_array(features["xz_drift"]["poison"]) != receipt["xz_poison_sha256_float32"]:
                errors.append(f"XZ poison fingerprint mismatch: {mode} seed {seed}")
            for representation in (pilot.INPUT_NAME, pilot.XZ_NAME, pilot.PRIMARY_NAME):
                stored = metrics[(mode, seed, representation)]
                replay_row, replay_samples = replay[representation]
                for field in ("tp", "fn", "fp", "tn"):
                    if int(stored[field]) != int(replay_row[field]):
                        errors.append(
                            f"{field} mismatch: {mode} seed {seed} {representation}"
                        )
                for field in ("tpr", "fpr", "auc_crossfit_score", "auc_conformal_rank"):
                    if not close(stored[field], replay_row[field], 1e-8):
                        errors.append(
                            f"{field} mismatch: {mode} seed {seed} {representation}"
                        )
                stored_samples = sorted(
                    samples[(mode, seed, representation)],
                    key=lambda row: int(row["sample_index"]),
                )
                if len(stored_samples) != len(replay_samples):
                    errors.append(
                        f"sample count mismatch: {mode} seed {seed} {representation}"
                    )
                    continue
                for stored_sample, replay_sample in zip(stored_samples, replay_samples):
                    if int(stored_sample["label"]) != int(replay_sample["label"]):
                        errors.append("sample label mismatch")
                        break
                    if int(stored_sample["flagged"]) != int(replay_sample["flagged"]):
                        errors.append("sample decision mismatch")
                        break
                    if not close(stored_sample["p_value"], replay_sample["p_value"], 1e-8):
                        errors.append("sample p-value mismatch")
                        break
                replayed_samples += len(replay_samples)

    replay_summary = confirmation.eligible_summaries(
        [
            {
                **row,
                **{
                    field: int(row[field])
                    for field in ("seed", "tp", "fn", "fp", "tn", "config_pass")
                },
                **{
                    field: float(row[field])
                    for field in (
                        "tpr",
                        "fpr",
                        "auc_crossfit_score",
                        "auc_conformal_rank",
                    )
                },
            }
            for row in metric_rows
        ],
        [
            {
                **row,
                "seed": int(row["seed"]),
                "test_eligible": int(row["test_eligible"]),
            }
            for row in victim_rows
        ],
        int(manifest["bootstrap_draws"]),
    )
    stored_summary = {(row["mode"], row["representation"]): row for row in summary_rows}
    for row in replay_summary:
        stored = stored_summary[(row["mode"], row["representation"])]
        if row["status"] != stored["status"]:
            errors.append(f"summary status mismatch: {row['mode']} {row['representation']}")
        if int(row["eligible_seed_count"]) != int(stored["eligible_seed_count"]):
            errors.append(f"summary eligibility mismatch: {row['mode']} {row['representation']}")
        for field in (
            "pooled_tpr",
            "pooled_fpr",
            "tpr_seed_bootstrap_ci_low",
            "tpr_seed_bootstrap_ci_high",
            "fpr_seed_bootstrap_ci_low",
            "fpr_seed_bootstrap_ci_high",
        ):
            if field in row and not close(row[field], stored[field], 1e-8):
                errors.append(f"summary {field} mismatch: {row['mode']} {row['representation']}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "experiment": manifest["experiment"],
        "replayed_metric_rows": len(metric_rows),
        "replayed_sample_rows": replayed_samples,
        "replayed_checkpoints": replayed_checkpoints,
        "replayed_poison_and_trigger_fingerprints": replayed_poison_fingerprints,
        "replayed_summary_rows": len(replay_summary),
        "errors": errors,
    }


def main(args: argparse.Namespace) -> None:
    receipt = verify(args)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    main(parser.parse_args())
