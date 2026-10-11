#!/usr/bin/env python3
"""Mechanically verify the BloodMNIST clean-encoder experiment artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

import train_bloodmnist_clean_encoder as clean_encoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "paper14/results/bloodmnist_clean_encoder_v1"


def check(name: str, condition: bool, detail) -> dict:
    return {"name": name, "passed": bool(condition), "detail": detail}


def explicit_local_z_error(checkpoint_path: Path) -> tuple[float, bool]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = clean_encoder.CleanQNN(checkpoint["encoder"], int(checkpoint["layers"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    generator = torch.Generator().manual_seed(20260928)
    x = torch.randn(7, 256, generator=generator)
    with torch.no_grad():
        analytic = model.encoder(x)
        latent = model.encoder.projected_latent(x)
        angles = (math.pi / 2) - (math.pi / 2) * latent
        psi = torch.zeros((len(x), 2**8), dtype=torch.complex64)
        psi[:, 0] = 1.0
        for qubit in range(8):
            psi = clean_encoder.apply_batch_ry(psi, angles[:, qubit], qubit)
        statevector_z = clean_encoder.z_expectations(psi, 8)
        error = float(torch.max(torch.abs(analytic - statevector_z)).item())

    model.train()
    logits = model(x)
    labels = torch.arange(len(x), dtype=torch.long) % 2
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    finite_gradients = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    return error, finite_gradients


def main() -> None:
    summary = json.loads(
        (RESULT_ROOT / "clean_encoder_summary.json").read_text(encoding="utf-8")
    )
    development = json.loads(
        (RESULT_ROOT / "development/run_summary.json").read_text(encoding="utf-8")
    )
    confirmation = json.loads(
        (RESULT_ROOT / "confirmation/run_summary.json").read_text(encoding="utf-8")
    )
    explicit_error, finite_gradients = explicit_local_z_error(
        RESULT_ROOT / "development/seed_42_local_l0.pt"
    )

    confirmation_seeds = sorted(int(row["seed"]) for row in confirmation)
    confirmation_configs = {
        (str(row["encoder"]), int(row["layers"])) for row in confirmation
    }
    test_access_flags = [bool(row["test_evaluated"]) for row in development]
    test_accuracies = [float(row["test"]["accuracy"]) for row in confirmation]
    checks = [
        check(
            "development candidates did not evaluate test",
            not any(test_access_flags),
            test_access_flags,
        ),
        check(
            "confirmation seeds are independent 43-46",
            confirmation_seeds == [43, 44, 45, 46],
            confirmation_seeds,
        ),
        check(
            "confirmation uses one frozen encoder configuration",
            confirmation_configs == {("local", 0)},
            sorted(confirmation_configs),
        ),
        check(
            "all confirmation held-out accuracies meet 90%",
            all(value >= 0.90 for value in test_accuracies),
            test_accuracies,
        ),
        check(
            "confirmation mean matches summary",
            abs(
                sum(test_accuracies) / len(test_accuracies)
                - float(summary["confirmation"]["clean_accuracy"]["mean"])
            )
            < 1e-12,
            summary["confirmation"]["clean_accuracy"]["mean"],
        ),
        check(
            "paired improvement exceeds ten percentage points",
            float(summary["confirmation"]["paired_improvement"]["mean"]) >= 0.10,
            summary["confirmation"]["paired_improvement"]["mean"],
        ),
        check(
            "historical reconstruction errors are accepted",
            float(summary["integrity"]["max_reconstruction_error"])
            <= float(summary["integrity"]["reconstruction_threshold"]),
            {
                "observed": summary["integrity"]["max_reconstruction_error"],
                "threshold": summary["integrity"]["reconstruction_threshold"],
            },
        ),
        check(
            "delayed development checkpoint reproduces training metrics",
            all(summary["integrity"]["delayed_checkpoint_matches_training_record"].values()),
            summary["integrity"]["delayed_checkpoint_matches_training_record"],
        ),
        check(
            "checkpoint scaler reproduces reconstructed train scaler",
            max(summary["integrity"]["scaler_checkpoint_match"].values()) == 0.0,
            summary["integrity"]["scaler_checkpoint_match"],
        ),
        check(
            "analytic local Z readout matches explicit statevector",
            explicit_error <= 1e-6,
            explicit_error,
        ),
        check("all model gradients are finite", finite_gradients, finite_gradients),
        check(
            "selected model parameter count is stable",
            int(summary["selected_encoder"]["parameters"]) == 2410,
            summary["selected_encoder"]["parameters"],
        ),
    ]
    receipt = {
        "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "checks": checks,
        "explicit_statevector_z_max_abs_error": explicit_error,
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
    main()
