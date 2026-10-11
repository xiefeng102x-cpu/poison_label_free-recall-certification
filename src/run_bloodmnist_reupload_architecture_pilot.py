#!/usr/bin/env python3
"""Cross-architecture gate using an exact three-layer data-reupload QNN."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_cross_attack_pilot as cross
import run_bloodmnist_group_coherent_10seeds as baseline
import run_bloodmnist_new_encoder_adaptive as adaptive
import run_bloodmnist_whitebox_adaptive_pilot as whitebox
import run_mnist_group_coherent_drift as coherent
import run_unified_oneclass_poison_screening as unified
import train_bloodmnist_clean_encoder as clean_encoder


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PAPER14_ROOT / "results/bloodmnist_reupload_architecture_pilot_v1"
PROTOCOL = PAPER14_ROOT / "results/top_journal_program_v1/phase4_reupload_architecture_protocol.json"
SEEDS = (42, 46, 51)
INPUT_NAME = "coherent_input256"
XZ_NAME = "coherent_reupload_qml_xz_drift16"
PRIMARY_NAME = "coherent_bonferroni_input_reupload_qml_xz"


def quantum_state(encoder: clean_encoder.SupervisedAngleEncoder, x: torch.Tensor) -> torch.Tensor:
    if encoder.encoder != "reupload" or encoder.qweights is None:
        raise ValueError("Exact cross-architecture view requires a reupload encoder")
    latent = encoder.projected_latent(x)
    angles = (math.pi / 2.0) - (math.pi / 2.0) * latent
    psi = torch.zeros((len(x), 2**encoder.n_qubits), dtype=torch.complex64, device=x.device)
    psi[:, 0] = 1.0
    for layer in range(encoder.n_layers):
        for qubit in range(encoder.n_qubits):
            psi = clean_encoder.apply_batch_ry(psi, angles[:, qubit], qubit)
        for qubit in range(encoder.n_qubits):
            theta, phi = encoder.qweights[layer, qubit]
            cosine, sine = torch.cos(theta / 2), torch.sin(theta / 2)
            ry = torch.stack((cosine, -sine, sine, cosine)).reshape(2, 2)
            psi = clean_encoder.apply_one_qubit(psi, ry.to(torch.complex64), qubit)
            phase_minus = torch.exp(torch.complex(torch.zeros_like(phi), -phi / 2))
            phase_plus = torch.exp(torch.complex(torch.zeros_like(phi), phi / 2))
            zero = torch.zeros((), dtype=torch.complex64, device=x.device)
            rz = torch.stack((phase_minus, zero, zero, phase_plus)).reshape(2, 2)
            psi = clean_encoder.apply_one_qubit(psi, rz, qubit)
        for qubit in range(encoder.n_qubits):
            source = clean_encoder.cnot_source(
                encoder.n_qubits, qubit, (qubit + 1) % encoder.n_qubits, x.device
            )
            psi = psi[:, source]
    return psi


def x_expectations(psi: torch.Tensor, n_qubits: int) -> torch.Tensor:
    values = []
    dimension = psi.shape[1]
    for qubit in range(n_qubits):
        step = dimension >> (qubit + 1)
        groups = dimension // (2 * step)
        pair = psi.reshape(-1, groups, 2, step)
        amp0, amp1 = pair[:, :, 0, :], pair[:, :, 1, :]
        values.append(2.0 * (amp0.conj() * amp1).real.sum(dim=(1, 2)))
    return torch.stack(values, dim=1)


def xz_features(model: clean_encoder.CleanQNN, x: torch.Tensor) -> torch.Tensor:
    psi = quantum_state(model.encoder, x)
    x_values = x_expectations(psi, model.encoder.n_qubits)
    z_values = clean_encoder.z_expectations(psi, model.encoder.n_qubits)
    return torch.cat([x_values, z_values], dim=1)


def numpy_xz(model: clean_encoder.CleanQNN, values: np.ndarray) -> np.ndarray:
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 128):
            batch = torch.as_tensor(values[start : start + 128], dtype=torch.float32)
            output.append(xz_features(model, batch).cpu().numpy())
    return np.concatenate(output, axis=0)


def load_or_train_clean(output: Path, splits: dict, seed: int) -> tuple[clean_encoder.CleanQNN, dict]:
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / f"seed_{seed}_clean.pt"
    record_path = model_dir / f"seed_{seed}_clean.json"
    if checkpoint.exists() and record_path.exists():
        model = clean_encoder.CleanQNN("reupload", 3)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"])
        return model, json.loads(record_path.read_text(encoding="utf-8"))
    model, record = clean_encoder.train_one(
        baseline.clean_arrays(splits), seed, "reupload", 3, 60, 1e-3, 1e-4, 64, 15
    )
    torch.save({"state_dict": model.state_dict(), "seed": seed, "encoder": "reupload", "layers": 3}, checkpoint)
    compact = {
        "seed": seed,
        "best_epoch": record["best_epoch"],
        "epochs_completed": record["epochs_completed"],
        "train": record["train"],
        "validation": record["val"],
        "checkpoint_sha256": whitebox.sha256_file(checkpoint),
    }
    record_path.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return model, compact


def validation_metrics(model: clean_encoder.CleanQNN, splits: dict) -> tuple[float, float]:
    return whitebox.balanced_ca_and_asr(
        model,
        splits["val"]["target"],
        splits["val"]["non_target"],
        splits["val"]["triggered_non_target"],
    )


def selection_key(point: dict) -> tuple:
    eligible = point["balanced_ca"] >= 0.90 and point["asr"] >= 0.80
    if eligible:
        return (1, point["balanced_ca"], point["asr"], -point["epoch"])
    margin = min(point["balanced_ca"] / 0.90, point["asr"] / 0.80)
    return (0, margin, point["balanced_ca"], point["asr"], -point["epoch"])


def load_or_adapt(
    output: Path, clean_model: clean_encoder.CleanQNN, splits: dict, seed: int, args: argparse.Namespace
) -> tuple[clean_encoder.CleanQNN, dict]:
    model_dir = output / "models"
    checkpoint = model_dir / f"seed_{seed}_adaptive.pt"
    record_path = model_dir / f"seed_{seed}_adaptive.json"
    if checkpoint.exists() and record_path.exists():
        model = clean_encoder.CleanQNN("reupload", 3)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"])
        return model, json.loads(record_path.read_text(encoding="utf-8"))

    torch.manual_seed(seed)
    model = copy.deepcopy(clean_model)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(False)
    parameters = list(model.encoder.parameters())
    initial = [parameter.detach().clone() for parameter in parameters]
    x_clean, y_clean = splits["train"]["clean_x"], splits["train"]["clean_y"]
    poison = splits["archived_poison"]
    with torch.no_grad():
        teacher = clean_model(torch.as_tensor(x_clean, dtype=torch.float32)).detach()
    rng = np.random.default_rng(seed + 7301)
    history, states = [], []

    def record(epoch: int) -> None:
        ca, asr = validation_metrics(model, splits)
        history.append({"epoch": epoch, "balanced_ca": ca, "asr": asr})
        states.append(copy.deepcopy(model.encoder.state_dict()))

    record(0)
    for epoch in range(1, args.adaptive_epochs + 1):
        clean_order = rng.permutation(len(x_clean))
        poison_order = rng.permutation(len(poison))
        cursor = 0
        for start in range(0, len(x_clean), 64):
            clean_indices = clean_order[start : start + 64]
            size = len(clean_indices)
            if cursor + size > len(poison_order):
                poison_order = rng.permutation(len(poison))
                cursor = 0
            poison_indices = poison_order[cursor : cursor + size]
            cursor += size
            if len(poison_indices) < size:
                poison_indices = np.concatenate(
                    [poison_indices, rng.choice(len(poison), size - len(poison_indices), replace=True)]
                )
            clean_batch = torch.as_tensor(x_clean[clean_indices], dtype=torch.float32)
            labels = torch.as_tensor(y_clean[clean_indices], dtype=torch.long)
            poison_batch = torch.as_tensor(poison[poison_indices], dtype=torch.float32)
            clean_logits = model(clean_batch)
            clean_loss = F.cross_entropy(clean_logits, labels)
            backdoor_loss = F.cross_entropy(model(poison_batch), torch.ones(size, dtype=torch.long))
            temperature = 2.0
            distill = F.kl_div(
                F.log_softmax(clean_logits / temperature, dim=1),
                F.softmax(teacher[clean_indices] / temperature, dim=1),
                reduction="batchmean",
            ) * temperature**2
            anchor = torch.stack(
                [torch.mean((parameter - start_value) ** 2) for parameter, start_value in zip(parameters, initial)]
            ).mean()
            objective = clean_loss + 10.0 * backdoor_loss + 20.0 * distill + 0.001 * anchor
            gradients = torch.autograd.grad(objective, parameters)
            norm = torch.sqrt(sum(torch.sum(gradient**2) for gradient in gradients))
            scale = min(1.0, 1.0 / max(float(norm), 1e-12))
            with torch.no_grad():
                for parameter, gradient in zip(parameters, gradients):
                    parameter -= 0.01 * scale * gradient
        record(epoch)

    selected_index = max(range(len(history)), key=lambda index: selection_key(history[index]))
    model.encoder.load_state_dict(states[selected_index])
    selected = {**history[selected_index], "selection_key": list(selection_key(history[selected_index]))}
    torch.save(
        {"state_dict": model.state_dict(), "seed": seed, "encoder": "reupload", "layers": 3},
        checkpoint,
    )
    record = {
        "seed": seed,
        "selected_epoch": selected["epoch"],
        "validation": selected,
        "history": history,
        "checkpoint_sha256": whitebox.sha256_file(checkpoint),
    }
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return model, record


def detector(
    seed: int,
    clean_model: clean_encoder.CleanQNN,
    attacked_model: clean_encoder.CleanQNN,
    splits: dict,
    alpha: float,
) -> tuple[list[dict], list[dict]]:
    input_groups = {
        "trusted": splits["val"]["target"],
        "clean": splits["train"]["target"],
        "poison": splits["archived_poison"],
    }
    drift_groups = {
        name: numpy_xz(attacked_model, values) - numpy_xz(clean_model, values)
        for name, values in input_groups.items()
    }
    config = {"seed": seed, "trigger": "fixed_pca_additive_reupload", "archive_ca": 0.0, "archive_asr": 0.0}
    results = {}
    for name, groups in ((INPUT_NAME, input_groups), (XZ_NAME, drift_groups)):
        results[name] = cross.evaluate_external_reference(
            config, name, groups["trusted"], groups["clean"], groups["poison"], alpha
        )
    results[PRIMARY_NAME] = coherent.combine_margin_and_correlator(
        config,
        results[INPUT_NAME][0],
        results[INPUT_NAME][1],
        results[XZ_NAME][1],
        alpha,
        combined_name=PRIMARY_NAME,
        second_dimension=16,
    )
    rows, samples = [], []
    for name in (INPUT_NAME, XZ_NAME, PRIMARY_NAME):
        rows.append(results[name][0])
        samples.extend(results[name][1])
    return rows, samples


def main(args: argparse.Namespace) -> None:
    if tuple(args.seeds) != SEEDS:
        raise ValueError("Cross-architecture pilot is frozen to seeds 42 46 51")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    class_data = heldout.collect_classes(args.dataset_root)
    rgb_trigger, rgb_receipt = baseline.fixed_rgb_trigger(class_data)
    rows, samples, victims, receipts = [], [], [], []
    for seed in args.seeds:
        print(f"seed={seed}: reupload clean training", flush=True)
        prepared = baseline.preprocess_seed(class_data, seed)
        splits, trigger_receipt = baseline.seed_splits(prepared, rgb_trigger, seed)
        clean_model, clean_record = load_or_train_clean(args.output, splits, seed)
        probe = torch.as_tensor(splits["val"]["target"][:8], dtype=torch.float32)
        with torch.no_grad():
            state = quantum_state(clean_model.encoder, probe)
            state_norm_error = float(torch.max(torch.abs((state.conj() * state).real.sum(1) - 1)))
            z_error = float(torch.max(torch.abs(xz_features(clean_model, probe)[:, 8:] - clean_model.encoder(probe))))
        if state_norm_error > 1e-5 or z_error > 1e-5:
            raise RuntimeError("Exact reupload observable self-check failed")
        attacked_model, attack_record = load_or_adapt(args.output, clean_model, splits, seed, args)
        test_ca, test_asr = whitebox.balanced_ca_and_asr(
            attacked_model,
            splits["test"]["target"],
            splits["test"]["non_target"],
            splits["test"]["triggered_non_target"],
        )
        selected_rows, selected_samples = detector(seed, clean_model, attacked_model, splits, args.alpha)
        for row in selected_rows:
            row["archive_ca"] = test_ca
            row["archive_asr"] = test_asr
        rows.extend(selected_rows)
        samples.extend(selected_samples)
        victims.append(
            {
                "seed": seed,
                "clean_validation_balanced_ca": clean_record["validation"]["balanced_accuracy"],
                "adaptive_selected_epoch": attack_record["selected_epoch"],
                "test_balanced_ca": test_ca,
                "test_asr": test_asr,
                "test_eligible": int(test_ca >= 0.90 and test_asr >= 0.80),
            }
        )
        receipts.append(
            {
                "seed": seed,
                "clean_checkpoint_sha256": clean_record["checkpoint_sha256"],
                "adaptive_checkpoint_sha256": attack_record["checkpoint_sha256"],
                "poison_sha256_float32": whitebox.sha256_array(splits["archived_poison"]),
                "state_norm_max_abs_error": state_norm_error,
                "z_forward_max_abs_error": z_error,
                "trigger_receipt": {
                    **trigger_receipt,
                    "poison_indices": np.asarray(
                        trigger_receipt["poison_indices"], dtype=int
                    ).tolist(),
                },
            }
        )
        print(
            f"seed={seed}: test CA/ASR={test_ca:.3f}/{test_asr:.3f}; "
            f"primary={selected_rows[-1]['tpr']:.3f}/{selected_rows[-1]['fpr']:.3f}",
            flush=True,
        )

    eligible_seeds = {int(row["seed"]) for row in victims if row["test_eligible"]}
    summaries = []
    for representation in (INPUT_NAME, XZ_NAME, PRIMARY_NAME):
        selected = [row for row in rows if row["representation"] == representation and int(row["seed"]) in eligible_seeds]
        if selected:
            summaries.append(
                {
                    "representation": representation,
                    "eligible_seed_count": len(selected),
                    "eligible_seeds": ",".join(map(str, sorted(eligible_seeds))),
                    **{key: value for key, value in coherent.aggregate(selected, "representation")[0].items() if key != "representation"},
                }
            )
        else:
            summaries.append({"representation": representation, "eligible_seed_count": 0, "eligible_seeds": ""})
    unified.write_csv(args.output / "per_config_metrics.csv", rows)
    unified.write_csv(args.output / "per_sample_scores.csv", samples)
    unified.write_csv(args.output / "victim_metrics.csv", victims)
    unified.write_csv(args.output / "eligible_summary.csv", summaries)
    unified.json_dump(
        args.output / "manifest.json",
        {
            "experiment": "bloodmnist_reupload_architecture_pilot_v1",
            "protocol": str(PROTOCOL),
            "protocol_sha256": whitebox.sha256_file(PROTOCOL),
            "dataset_npz_sha256": whitebox.sha256_file(args.dataset_root / "bloodmnist.npz"),
            "seeds": list(args.seeds),
            "architecture": {"encoder": "reupload", "layers": 3},
            "actual_xz_statevector_observables": True,
            "adaptive_epochs": args.adaptive_epochs,
            "alpha": args.alpha,
            "rgb_trigger_receipt": rgb_receipt,
            "receipts": receipts,
            "elapsed_seconds": time.time() - started,
        },
    )
    report = [
        "# BloodMNIST entangling data-reupload cross-architecture pilot",
        "",
        "The XZ16 detector view uses actual statevector X/Z expectations from the three-layer entangling circuit. It does not reuse the local-angle latent proxy.",
        "",
        "| seed | clean val balanced CA | test CA | test ASR | eligible |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in victims:
        report.append(
            f"| {row['seed']} | {row['clean_validation_balanced_ca']:.3f} | {row['test_balanced_ca']:.3f} | {row['test_asr']:.3f} | {row['test_eligible']} |"
        )
    report.extend(["", "| endpoint | eligible seeds | pooled TPR | pooled FPR |", "|---|---:|---:|---:|"])
    for row in summaries:
        report.append(
            f"| {row['representation']} | {row['eligible_seed_count']} | "
            + (f"{row['pooled_tpr']:.3f} | {row['pooled_fpr']:.3f} |" if row['eligible_seed_count'] else "— | — |")
        )
    report.extend(["", f"Runtime: {time.time() - started:.1f} seconds.", ""])
    (args.output / "EXPERIMENT_REPORT.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--adaptive-epochs", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.05)
    main(parser.parse_args())
