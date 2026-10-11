#!/usr/bin/env python3
"""Evaluate a frozen replacement clean-encoder checkpoint.

This is separated from candidate training so a development test split is not
read until after the encoder choice has been frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_bloodmnist_checkpoints as heldout
import train_bloodmnist_clean_encoder as clean_encoder


def close_metric(left: dict, right: dict, tolerance: float = 1e-10) -> bool:
    names = ("accuracy", "balanced_accuracy", "roc_auc")
    return all(abs(float(left[name]) - float(right[name])) <= tolerance for name in names)


def main(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    seed = int(checkpoint["seed"])
    encoder_name = str(checkpoint["encoder"])
    layers = int(checkpoint["layers"])
    model = clean_encoder.CleanQNN(encoder_name, layers)
    model.load_state_dict(checkpoint["state_dict"])

    class_data = heldout.collect_classes(args.dataset_root)
    arrays, data_receipt, scaler_state = clean_encoder.load_seed_data(class_data, seed)
    metrics = {
        split: clean_encoder.evaluate(model, *arrays[split], batch_size=args.batch_size)
        for split in ("train", "val", "test")
    }
    scaler_error = None
    if "input_scaler_mean" in checkpoint and "input_scaler_scale" in checkpoint:
        scaler_error = {
            "mean_max_abs_error": float(
                np.max(np.abs(checkpoint["input_scaler_mean"] - scaler_state["mean"]))
            ),
            "scale_max_abs_error": float(
                np.max(np.abs(checkpoint["input_scaler_scale"] - scaler_state["scale"]))
            ),
        }

    training_record_match = None
    if args.training_record is not None:
        training_record = json.loads(args.training_record.read_text(encoding="utf-8"))
        training_record_match = {
            "train": close_metric(metrics["train"], training_record["train"]),
            "val": close_metric(metrics["val"], training_record["val"]),
        }
        if not all(training_record_match.values()):
            raise RuntimeError(
                f"Frozen checkpoint does not match its training record: {training_record_match}"
            )

    receipt = {
        "checkpoint": str(args.checkpoint),
        "seed": seed,
        "encoder": encoder_name,
        "layers": layers,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "data_receipt": data_receipt,
        "scaler_checkpoint_match": scaler_error,
        "training_record_match": training_record_match,
        "metrics": metrics,
        "test_access_note": "Frozen checkpoint evaluated after development selection.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parsed = parser.parse_args()
    parsed.dataset_root = parsed.dataset_root.resolve()
    parsed.checkpoint = parsed.checkpoint.resolve()
    if parsed.training_record is not None:
        parsed.training_record = parsed.training_record.resolve()
    parsed.output = parsed.output.resolve()
    main(parsed)
