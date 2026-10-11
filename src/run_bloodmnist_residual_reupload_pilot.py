#!/usr/bin/env python3
"""Residual entangling-reupload QML cross-architecture pilot."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import evaluate_bloodmnist_checkpoints as heldout
import run_bloodmnist_cross_attack_pilot as cross
import run_bloodmnist_group_coherent_10seeds as baseline
import run_bloodmnist_reupload_architecture_pilot as pure
import run_bloodmnist_whitebox_adaptive_pilot as whitebox
import run_mnist_group_coherent_drift as coherent
import run_unified_oneclass_poison_screening as unified
import train_bloodmnist_clean_encoder as clean_encoder


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PAPER14_ROOT / "results/bloodmnist_residual_reupload_pilot_v1"
PROTOCOL = PAPER14_ROOT / "results/top_journal_program_v1/phase4_residual_reupload_protocol.json"
LOCAL_ROOT = PAPER14_ROOT / "results/bloodmnist_group_coherent_10seeds_v1"
SEEDS = (42, 46, 51)
INPUT_NAME = "coherent_input256"
XZ_NAME = "coherent_residual_reupload_xz_drift16"
PRIMARY_NAME = "coherent_bonferroni_input_residual_reupload_xz"


class ResidualReuploadQNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = clean_encoder.SupervisedAngleEncoder(
            encoder="reupload", n_layers=3
        )
        self.classifier = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))

    def branch_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder.projected_latent(x)
        local_z = torch.sin((np.pi / 2.0) * latent)
        quantum_z = self.encoder(x)
        return local_z, quantum_z

    def logits_with_quantum(self, x: torch.Tensor, enabled: bool = True) -> torch.Tensor:
        local_z, quantum_z = self.branch_features(x)
        if not enabled:
            quantum_z = torch.zeros_like(quantum_z)
        return self.classifier(torch.cat([local_z, quantum_z], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits_with_quantum(x, True)


def warm_start(local: clean_encoder.CleanQNN) -> ResidualReuploadQNN:
    model = ResidualReuploadQNN()
    model.encoder.projector.load_state_dict(local.encoder.projector.state_dict())
    with torch.no_grad():
        local_first = local.classifier[0]
        model.classifier[0].weight[:, :8].copy_(0.9 * local_first.weight)
        model.classifier[0].weight[:, 8:].copy_(0.1 * local_first.weight)
        model.classifier[0].bias.copy_(local_first.bias)
        model.classifier[2].load_state_dict(local.classifier[2].state_dict())
    return model


def clean_ca(model: nn.Module, split: dict) -> float:
    target, _ = whitebox.adaptive.predict(model, split["target"], 256)
    source, _ = whitebox.adaptive.predict(model, split["non_target"], 256)
    return 0.5 * (float(np.mean(target == 1)) + float(np.mean(source == 0)))


def load_or_train_clean(
    output: Path,
    local: clean_encoder.CleanQNN,
    splits: dict,
    seed: int,
) -> tuple[ResidualReuploadQNN, dict]:
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / f"seed_{seed}_clean.pt"
    record_path = model_dir / f"seed_{seed}_clean.json"
    if checkpoint.exists() and record_path.exists():
        model = ResidualReuploadQNN()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"])
        return model, json.loads(record_path.read_text(encoding="utf-8"))

    torch.manual_seed(seed)
    model = warm_start(local)
    parameters = list(model.parameters())
    initial = [parameter.detach().clone() for parameter in parameters]
    x, y = splits["train"]["clean_x"], splits["train"]["clean_y"]
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.long)
    with torch.no_grad():
        teacher = local(x_tensor).detach()
    generator = torch.Generator().manual_seed(seed + 22103)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tensor, y_tensor),
        batch_size=64,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-4)
    history, states = [], []

    def record(epoch: int) -> None:
        model.eval()
        history.append({"epoch": epoch, "validation_balanced_ca": clean_ca(model, splits["val"])})
        states.append(copy.deepcopy(model.state_dict()))
        model.train()

    record(0)
    for epoch in range(1, 11):
        for batch, labels in loader:
            optimizer.zero_grad()
            indices = None
            # Recompute teacher logits for this batch to avoid depending on shuffled indices.
            with torch.no_grad():
                teacher_batch = local(batch)
            logits = model(batch)
            ce = F.cross_entropy(logits, labels)
            temperature = 2.0
            distill = F.kl_div(
                F.log_softmax(logits / temperature, dim=1),
                F.softmax(teacher_batch / temperature, dim=1),
                reduction="batchmean",
            ) * temperature**2
            anchor = torch.stack(
                [torch.mean((parameter - start) ** 2) for parameter, start in zip(parameters, initial)]
            ).mean()
            objective = ce + 10.0 * distill + 0.001 * anchor
            objective.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        record(epoch)
    selected_index = max(
        range(len(history)),
        key=lambda index: (history[index]["validation_balanced_ca"], -history[index]["epoch"]),
    )
    model.load_state_dict(states[selected_index])
    first = model.classifier[0].weight.detach()
    quantum_fraction = float(
        torch.linalg.vector_norm(first[:, 8:]) / torch.linalg.vector_norm(first)
    )
    probe = torch.as_tensor(splits["val"]["clean_x"], dtype=torch.float32)
    with torch.no_grad():
        branch_logit_effect = float(
            torch.mean(torch.abs(model(probe) - model.logits_with_quantum(probe, False)))
        )
    torch.save({"state_dict": model.state_dict(), "seed": seed}, checkpoint)
    record = {
        "seed": seed,
        "selected_epoch": history[selected_index]["epoch"],
        "validation_balanced_ca": history[selected_index]["validation_balanced_ca"],
        "history": history,
        "quantum_classifier_weight_fraction": quantum_fraction,
        "mean_abs_quantum_branch_logit_effect": branch_logit_effect,
        "checkpoint_sha256": whitebox.sha256_file(checkpoint),
    }
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return model, record


def selection_key(point: dict) -> tuple:
    eligible = point["balanced_ca"] >= 0.90 and point["asr"] >= 0.80
    if eligible:
        return (1, point["balanced_ca"], point["asr"], -point["epoch"])
    return (
        0,
        min(point["balanced_ca"] / 0.90, point["asr"] / 0.80),
        point["balanced_ca"],
        point["asr"],
        -point["epoch"],
    )


def load_or_adapt(
    output: Path,
    clean_model: ResidualReuploadQNN,
    splits: dict,
    seed: int,
) -> tuple[ResidualReuploadQNN, dict]:
    model_dir = output / "models"
    checkpoint = model_dir / f"seed_{seed}_adaptive.pt"
    record_path = model_dir / f"seed_{seed}_adaptive.json"
    if checkpoint.exists() and record_path.exists():
        model = ResidualReuploadQNN()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"])
        return model, json.loads(record_path.read_text(encoding="utf-8"))
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
        ca, asr = whitebox.balanced_ca_and_asr(
            model,
            splits["val"]["target"],
            splits["val"]["non_target"],
            splits["val"]["triggered_non_target"],
        )
        history.append({"epoch": epoch, "balanced_ca": ca, "asr": asr})
        states.append(copy.deepcopy(model.encoder.state_dict()))

    record(0)
    for epoch in range(1, 9):
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
            backdoor = F.cross_entropy(model(poison_batch), torch.ones(size, dtype=torch.long))
            temperature = 2.0
            distill = F.kl_div(
                F.log_softmax(clean_logits / temperature, dim=1),
                F.softmax(teacher[clean_indices] / temperature, dim=1),
                reduction="batchmean",
            ) * temperature**2
            anchor = torch.stack(
                [torch.mean((parameter - start_value) ** 2) for parameter, start_value in zip(parameters, initial)]
            ).mean()
            objective = clean_loss + 10.0 * backdoor + 20.0 * distill + 0.001 * anchor
            gradients = torch.autograd.grad(objective, parameters)
            norm = torch.sqrt(sum(torch.sum(gradient**2) for gradient in gradients))
            scale = min(1.0, 1.0 / max(float(norm), 1e-12))
            with torch.no_grad():
                for parameter, gradient in zip(parameters, gradients):
                    parameter -= 0.01 * scale * gradient
        record(epoch)
    selected_index = max(range(len(history)), key=lambda index: selection_key(history[index]))
    model.encoder.load_state_dict(states[selected_index])
    torch.save({"state_dict": model.state_dict(), "seed": seed}, checkpoint)
    record = {
        "seed": seed,
        "selected_epoch": history[selected_index]["epoch"],
        "validation": history[selected_index],
        "history": history,
        "checkpoint_sha256": whitebox.sha256_file(checkpoint),
    }
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return model, record


def numpy_xz(model: ResidualReuploadQNN, values: np.ndarray) -> np.ndarray:
    return pure.numpy_xz(model, values)


def detector(
    seed: int,
    clean_model: ResidualReuploadQNN,
    attacked_model: ResidualReuploadQNN,
    splits: dict,
    alpha: float,
) -> tuple[list[dict], list[dict]]:
    inputs = {
        "trusted": splits["val"]["target"],
        "clean": splits["train"]["target"],
        "poison": splits["archived_poison"],
    }
    drift = {
        name: numpy_xz(attacked_model, values) - numpy_xz(clean_model, values)
        for name, values in inputs.items()
    }
    config = {"seed": seed, "trigger": "fixed_pca_additive_residual_reupload", "archive_ca": 0.0, "archive_asr": 0.0}
    result = {}
    for name, groups in ((INPUT_NAME, inputs), (XZ_NAME, drift)):
        result[name] = cross.evaluate_external_reference(
            config, name, groups["trusted"], groups["clean"], groups["poison"], alpha
        )
    result[PRIMARY_NAME] = coherent.combine_margin_and_correlator(
        config,
        result[INPUT_NAME][0],
        result[INPUT_NAME][1],
        result[XZ_NAME][1],
        alpha,
        combined_name=PRIMARY_NAME,
        second_dimension=16,
    )
    rows, samples = [], []
    for name in (INPUT_NAME, XZ_NAME, PRIMARY_NAME):
        rows.append(result[name][0])
        samples.extend(result[name][1])
    return rows, samples


def main(args: argparse.Namespace) -> None:
    if tuple(args.seeds) != SEEDS:
        raise ValueError("Residual reupload pilot is frozen to seeds 42 46 51")
    args.output.mkdir(parents=True, exist_ok=True)
    class_data = heldout.collect_classes(args.dataset_root)
    rgb_trigger, rgb_receipt = baseline.fixed_rgb_trigger(class_data)
    rows, samples, victims, receipts = [], [], [], []
    for seed in args.seeds:
        print(f"seed={seed}: residual reupload", flush=True)
        prepared = baseline.preprocess_seed(class_data, seed)
        splits, trigger_receipt = baseline.seed_splits(prepared, rgb_trigger, seed)
        local_model, _ = cross.load_reference_clean_model(seed, LOCAL_ROOT)
        clean_model, clean_record = load_or_train_clean(args.output, local_model, splits, seed)
        attacked_model, attack_record = load_or_adapt(args.output, clean_model, splits, seed)
        test_ca, test_asr = whitebox.balanced_ca_and_asr(
            attacked_model,
            splits["test"]["target"],
            splits["test"]["non_target"],
            splits["test"]["triggered_non_target"],
        )
        selected_rows, selected_samples = detector(seed, clean_model, attacked_model, splits, args.alpha)
        for row in selected_rows:
            row["archive_ca"], row["archive_asr"] = test_ca, test_asr
        rows.extend(selected_rows)
        samples.extend(selected_samples)
        victims.append(
            {
                "seed": seed,
                "clean_validation_balanced_ca": clean_record["validation_balanced_ca"],
                "quantum_classifier_weight_fraction": clean_record["quantum_classifier_weight_fraction"],
                "mean_abs_quantum_branch_logit_effect": clean_record["mean_abs_quantum_branch_logit_effect"],
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
                "trigger_receipt": {
                    **trigger_receipt,
                    "poison_indices": np.asarray(trigger_receipt["poison_indices"], dtype=int).tolist(),
                },
            }
        )
        print(
            f"seed={seed}: test CA/ASR={test_ca:.3f}/{test_asr:.3f}; primary={selected_rows[-1]['tpr']:.3f}/{selected_rows[-1]['fpr']:.3f}",
            flush=True,
        )

    eligible = {int(row["seed"]) for row in victims if row["test_eligible"]}
    summaries = []
    for representation in (INPUT_NAME, XZ_NAME, PRIMARY_NAME):
        selected = [row for row in rows if row["representation"] == representation and int(row["seed"]) in eligible]
        summaries.append(
            {
                "representation": representation,
                "eligible_seed_count": len(selected),
                "eligible_seeds": ",".join(map(str, sorted(eligible))),
                **(
                    {key: value for key, value in coherent.aggregate(selected, "representation")[0].items() if key != "representation"}
                    if selected
                    else {}
                ),
            }
        )
    unified.write_csv(args.output / "per_config_metrics.csv", rows)
    unified.write_csv(args.output / "per_sample_scores.csv", samples)
    unified.write_csv(args.output / "victim_metrics.csv", victims)
    unified.write_csv(args.output / "eligible_summary.csv", summaries)
    unified.json_dump(
        args.output / "manifest.json",
        {
            "experiment": "bloodmnist_residual_reupload_pilot_v1",
            "protocol": str(PROTOCOL),
            "protocol_sha256": whitebox.sha256_file(PROTOCOL),
            "dataset_npz_sha256": whitebox.sha256_file(args.dataset_root / "bloodmnist.npz"),
            "seeds": list(args.seeds),
            "architecture": "local-angle plus three-layer entangling data-reupload residual",
            "actual_entangling_branch_xz_observables": True,
            "alpha": args.alpha,
            "rgb_trigger_receipt": rgb_receipt,
            "receipts": receipts,
        },
    )
    lines = [
        "# BloodMNIST residual entangling-reupload pilot",
        "",
        "| seed | clean val CA | quantum weight fraction | quantum logit effect | test CA | test ASR | eligible |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in victims:
        lines.append(
            f"| {row['seed']} | {row['clean_validation_balanced_ca']:.3f} | {row['quantum_classifier_weight_fraction']:.3f} | {row['mean_abs_quantum_branch_logit_effect']:.4f} | {row['test_balanced_ca']:.3f} | {row['test_asr']:.3f} | {row['test_eligible']} |"
        )
    lines.extend(["", "| endpoint | eligible seeds | pooled TPR | pooled FPR |", "|---|---:|---:|---:|"])
    for row in summaries:
        lines.append(
            f"| {row['representation']} | {row['eligible_seed_count']} | "
            + (f"{row['pooled_tpr']:.3f} | {row['pooled_fpr']:.3f} |" if row['eligible_seed_count'] else "— | — |")
        )
    lines.append("")
    (args.output / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--alpha", type=float, default=0.05)
    main(parser.parse_args())
