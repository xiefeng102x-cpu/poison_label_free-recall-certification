#!/usr/bin/env python3
"""Verify adaptive replacement-encoder artifacts and replay held-out metrics."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

import run_bloodmnist_new_encoder_adaptive as adaptive
import train_bloodmnist_clean_encoder as clean_encoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "paper14/results/bloodmnist_new_encoder_adaptive_v1"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check(name: str, condition: bool, detail) -> dict:
    return {"name": name, "passed": bool(condition), "detail": detail}


def max_state_error(left: dict, right: dict, prefix: str) -> float:
    errors = []
    for name, tensor in left.items():
        if name.startswith(prefix):
            errors.append(float(torch.max(torch.abs(tensor - right[name])).item()))
    return max(errors, default=0.0)


def replay_confirmation(dataset_root: Path) -> tuple[list[dict], float, float]:
    records = []
    classifier_error = 0.0
    projector_serialization_error = 0.0
    for filename in sorted(glob.glob(str(RESULT_ROOT / "confirmation/*.json"))):
        artifact_path = Path(filename)
        artifact = load(artifact_path)
        checkpoint_path = artifact_path.with_suffix(".pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        clean_checkpoint = torch.load(
            artifact["clean_checkpoint"], map_location="cpu", weights_only=False
        )
        model = clean_encoder.CleanQNN("local", 0)
        model.load_state_dict(checkpoint["state_dict"])
        splits, _ = adaptive.load_data(
            int(artifact["seed"]), dataset_root, checkpoint
        )
        detector_reference = adaptive.numpy_xyz(
            model, splits["train"]["target"], int(artifact["configuration"]["batch_size"])
        )
        replayed = adaptive.evaluate_split(
            model,
            splits["test"],
            detector_reference,
            int(artifact["configuration"]["batch_size"]),
        )
        saved = artifact["selected"]["test"]
        records.append(
            {
                "seed": int(artifact["seed"]),
                "clean_accuracy_saved": float(saved["clean"]["accuracy"]),
                "clean_accuracy_replayed": float(replayed["clean"]["accuracy"]),
                "heldout_asr_saved": float(saved["attack"]["asr"]),
                "heldout_asr_replayed": float(replayed["attack"]["asr"]),
                "separability_saved": float(
                    saved["detector"]["separability_xyz_independent_reference"]
                ),
                "separability_replayed": float(
                    replayed["detector"][
                        "separability_xyz_independent_reference"
                    ]
                ),
            }
        )
        classifier_error = max(
            classifier_error,
            max_state_error(
                checkpoint["state_dict"],
                clean_checkpoint["state_dict"],
                "classifier.",
            ),
        )
        selected_state = artifact["selected_projector_state"]
        projector_serialization_error = max(
            projector_serialization_error,
            float(
                np.max(
                    np.abs(
                        checkpoint["state_dict"]["encoder.projector.weight"].numpy()
                        - np.asarray(selected_state["weight"], dtype=np.float32)
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        checkpoint["state_dict"]["encoder.projector.bias"].numpy()
                        - np.asarray(selected_state["bias"], dtype=np.float32)
                    )
                )
            ),
        )
    return records, classifier_error, projector_serialization_error


def main(args: argparse.Namespace) -> None:
    summary = load(RESULT_ROOT / "adaptive_summary.json")
    development = [
        load(Path(filename))
        for filename in sorted(glob.glob(str(RESULT_ROOT / "development/*.json")))
    ]
    delayed = load(
        next(Path(filename) for filename in glob.glob(str(RESULT_ROOT / "delayed_test/*.json")))
    )
    confirmation = [
        load(Path(filename))
        for filename in sorted(glob.glob(str(RESULT_ROOT / "confirmation/*.json")))
    ]
    replayed, classifier_error, projector_serialization_error = replay_confirmation(
        args.dataset_root
    )
    config_keys = (
        "epochs",
        "learning_rate",
        "backdoor_weight",
        "stealth_weight",
        "distill_weight",
        "anchor_weight",
        "gradient_clip",
        "batch_size",
    )
    configurations = {
        tuple(row["configuration"][key] for key in config_keys) for row in confirmation
    }
    replay_ca_match = all(
        row["clean_accuracy_saved"] == row["clean_accuracy_replayed"]
        for row in replayed
    )
    replay_asr_match = all(
        row["heldout_asr_saved"] == row["heldout_asr_replayed"] for row in replayed
    )
    # PCA/SVD can differ slightly with BLAS thread topology across hosts.  AUC
    # is rank-based and may move by a few pairs while CA/ASR remain identical.
    replay_separation_max_error = max(
        abs(row["separability_saved"] - row["separability_replayed"])
        for row in replayed
    )
    delayed_validation = delayed["selected"]["validation"]
    selected_development = next(
        row
        for row in development
        if float(row["configuration"]["learning_rate"]) == 0.01
    )["selected"]["validation"]

    checks = [
        check(
            "development sweep kept test unread",
            all(not row["selected"]["test_evaluated"] for row in development),
            [row["selected"]["test_evaluated"] for row in development],
        ),
        check(
            "delayed run reproduces selected validation CA and ASR",
            delayed_validation["clean"]["accuracy"]
            == selected_development["clean"]["accuracy"]
            and delayed_validation["attack"]["asr"]
            == selected_development["attack"]["asr"],
            {
                "development_ca": selected_development["clean"]["accuracy"],
                "delayed_ca": delayed_validation["clean"]["accuracy"],
                "development_asr": selected_development["attack"]["asr"],
                "delayed_asr": delayed_validation["attack"]["asr"],
            },
        ),
        check(
            "confirmation seeds are 43-46",
            sorted(int(row["seed"]) for row in confirmation) == [43, 44, 45, 46],
            sorted(int(row["seed"]) for row in confirmation),
        ),
        check(
            "confirmation uses one frozen configuration",
            len(configurations) == 1,
            [list(item) for item in configurations],
        ),
        check(
            "all four outcome gates pass",
            all(
                [
                    summary["confirmation"]["all_test_ca_at_least_90_percent"],
                    summary["confirmation"][
                        "all_test_ca_drops_at_most_5_points"
                    ],
                    summary["confirmation"][
                        "all_heldout_asr_at_least_80_percent"
                    ],
                    summary["confirmation"][
                        "all_independent_separability_at_most_065"
                    ],
                ]
            ),
            summary["confirmation"],
        ),
        check(
            "test CA replays from saved checkpoints",
            replay_ca_match,
            replayed,
        ),
        check(
            "held-out ASR replays from saved checkpoints",
            replay_asr_match,
            replayed,
        ),
        check(
            "detector separation replay is stable across BLAS paths",
            replay_separation_max_error <= 0.01,
            replay_separation_max_error,
        ),
        check(
            "classifier is bitwise frozen",
            classifier_error == 0.0,
            classifier_error,
        ),
        check(
            "serialized projector matches selected state",
            projector_serialization_error == 0.0,
            projector_serialization_error,
        ),
        check(
            "clean and triggered reconstruction errors pass threshold",
            summary["integrity"]["max_clean_reconstruction_error"]
            <= summary["integrity"]["reconstruction_threshold"]
            and summary["integrity"]["max_triggered_reconstruction_error"]
            <= summary["integrity"]["reconstruction_threshold"],
            summary["integrity"],
        ),
    ]
    receipt = {
        "status": "PASS" if all(row["passed"] for row in checks) else "FAIL",
        "checks": checks,
        "checkpoint_replay": replayed,
        "classifier_max_abs_error": classifier_error,
        "projector_serialization_max_abs_error": projector_serialization_error,
        "detector_separation_replay_max_abs_error": replay_separation_max_error,
    }
    output = RESULT_ROOT / "verification_receipt.json"
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parsed = parser.parse_args()
    parsed.dataset_root = parsed.dataset_root.resolve()
    main(parsed)
