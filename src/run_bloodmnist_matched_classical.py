#!/usr/bin/env python3
"""Protocol-matched classical control for the BloodMNIST CA investigation.

The published Classical Bloch control first trains its encoder and classifier
jointly on an already poisoned training pool.  The adaptive QNN instead starts
from a clean checkpoint and freezes its classifier while planting the backdoor.
This intervention removes that protocol mismatch:

1. pretrain the Classical Bloch encoder and classifier on clean target and
   clean non-target samples only;
2. freeze the classifier; and
3. adapt only the encoder on clean + poisoned samples with the same CE plus
   Mahalanobis objective used by the original classical control.

The experiment intentionally keeps the large Classical Bloch encoder.  It
isolates the clean-versus-poisoned pretraining protocol, not parameter count.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import evaluate_bloodmnist_checkpoints as heldout


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_ROOT / "Blackdoor_1"
SOURCE_ROOT = PROJECT_ROOT / "data/MedMnist/single_detect_20260411/epsilon_0.8"
CONTROL_CODE = (
    PROJECT_ROOT
    / "qml_adaptive_nc_project/paper_nc_adaptive/code/classical_bloch_control.py"
)


def load_control_module():
    sys.path.insert(0, str(CONTROL_CODE.parent))
    spec = importlib.util.spec_from_file_location("blood_classical_control_impl", CONTROL_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {CONTROL_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_data(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = SOURCE_ROOT / f"seed_{seed}/layer_8_grid/t6_vs_t0/pr_0.1"
    meta = torch.load(
        base / f"poisoned_samples_meta_seed_{seed}_pr_0.1.pt",
        map_location="cpu",
        weights_only=False,
    )
    return tuple(
        meta[key].numpy().astype(np.float32)
        for key in ("target_clean_data", "non_target_clean_data", "poisoned_non_target_data")
    )


def clean_pretrain(
    control,
    head,
    classifier,
    x_target: np.ndarray,
    x_nontarget: np.ndarray,
    epochs: int,
    lr: float,
    seed: int,
) -> None:
    x_train = np.vstack([x_target, x_nontarget])
    y_train = np.concatenate(
        [
            np.ones(len(x_target), dtype=np.int64),
            np.zeros(len(x_nontarget), dtype=np.int64),
        ]
    )
    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(seed + 9137)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(
        list(head.parameters()) + list(classifier.parameters()), lr=lr
    )
    head.train()
    classifier.train()
    for _ in range(epochs):
        for x_batch, y_batch in loader:
            features = head(x_batch)
            z_features = features[:, 2 * head.n_qubits :]
            loss = classifier.ce_loss(z_features, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def evaluate_without_detector(control, head, classifier, x_target, x_nontarget, x_poison):
    head.eval()
    classifier.eval()
    with torch.no_grad():
        xyz_target = head(torch.tensor(x_target, dtype=torch.float32))
        xyz_nontarget = head(torch.tensor(x_nontarget, dtype=torch.float32))
        xyz_poison = head(torch.tensor(x_poison, dtype=torch.float32))
        n_qubits = head.n_qubits
        pred_target = classifier(xyz_target[:, 2 * n_qubits :]).argmax(dim=1)
        pred_nontarget = classifier(xyz_nontarget[:, 2 * n_qubits :]).argmax(dim=1)
        pred_poison = classifier(xyz_poison[:, 2 * n_qubits :]).argmax(dim=1)
    clean_correct = int((pred_target == 1).sum()) + int((pred_nontarget == 0).sum())
    return {
        "ca": clean_correct / (len(x_target) + len(x_nontarget)),
        "asr": float((pred_poison == 1).float().mean()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(
    output: Path,
    seeds: list[int],
    lambdas: list[float],
    epochs: int,
    dataset_root: Path | None,
) -> None:
    control = load_control_module()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    run_records: list[dict] = []
    class_data = heldout.collect_classes(dataset_root) if dataset_root is not None else None

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        x_target, x_nontarget, x_poison = load_data(seed)
        head = control.BlochHead(x_target.shape[1], 8, hidden=64, constrained=True)
        classifier = control.ClassicalClassifier(8)
        start = time.time()
        clean_pretrain(
            control,
            head,
            classifier,
            x_target,
            x_nontarget,
            epochs=150,
            lr=1e-3,
            seed=seed,
        )
        clean_metrics = evaluate_without_detector(
            control, head, classifier, x_target, x_nontarget, x_poison
        )
        clean_heldout_metrics = None
        heldout_target = heldout_nontarget = None
        if class_data is not None:
            reconstruction = heldout.reconstruct_split(class_data, seed)
            reconstruction_errors = heldout.reconstruction_error(
                reconstruction["processed"]["train"], seed
            )
            if max(
                reconstruction_errors["target_max_abs_error"],
                reconstruction_errors["non_target_max_abs_error"],
            ) > 5e-3:
                raise RuntimeError(
                    f"Seed {seed} held-out reconstruction mismatch: {reconstruction_errors}"
                )
            heldout_target = reconstruction["processed"]["test"][6]
            heldout_nontarget = reconstruction["processed"]["test"][0]
            clean_heldout_metrics = evaluate_without_detector(
                control,
                head,
                classifier,
                heldout_target,
                heldout_nontarget,
                x_poison,
            )
        clean_head_state = copy.deepcopy(head.state_dict())
        clean_classifier_state = copy.deepcopy(classifier.state_dict())

        for lambda_r in lambdas:
            adapted_head = control.BlochHead(
                x_target.shape[1], 8, hidden=64, constrained=True
            )
            adapted_classifier = control.ClassicalClassifier(8)
            adapted_head.load_state_dict(clean_head_state)
            adapted_classifier.load_state_dict(clean_classifier_state)
            history, adapted_head = control.train_adaptive(
                adapted_head,
                adapted_classifier,
                x_target,
                x_nontarget,
                x_poison,
                epochs=epochs,
                lr=0.01,
                lambda_r=lambda_r,
                eval_every=20,
            )
            final_heldout_metrics = None
            if heldout_target is not None and heldout_nontarget is not None:
                final_heldout_metrics = evaluate_without_detector(
                    control,
                    adapted_head,
                    adapted_classifier,
                    heldout_target,
                    heldout_nontarget,
                    x_poison,
                )
            record = {
                "seed": seed,
                "lambda_r": lambda_r,
                "clean_pretrain": clean_metrics,
                "clean_pretrain_heldout": clean_heldout_metrics,
                "adaptive_initial": history[0],
                "adaptive_final": history[-1],
                "adaptive_final_heldout": final_heldout_metrics,
                "history": history,
                "clean_pretrain_epochs": 150,
                "adaptive_epochs": epochs,
                "adaptive_parameters": sum(
                    parameter.numel() for parameter in adapted_head.parameters()
                ),
                "frozen_classifier_parameters": sum(
                    parameter.numel() for parameter in adapted_classifier.parameters()
                ),
                "elapsed_seconds_since_seed_start": time.time() - start,
            }
            run_records.append(record)
            rows.append(
                {
                    "seed": seed,
                    "lambda_r": lambda_r,
                    "clean_pretrain_ca": clean_metrics["ca"],
                    "clean_pretrain_asr": clean_metrics["asr"],
                    "final_ca": history[-1]["ca"],
                    "final_asr": history[-1]["asr"],
                    "final_auc_xyz": history[-1]["auc_xyz"],
                    "final_jser_xyz_5fpr": history[-1]["jser_xyz_5fpr"],
                    "clean_pretrain_heldout_ca": (
                        clean_heldout_metrics["ca"]
                        if clean_heldout_metrics is not None
                        else None
                    ),
                    "final_heldout_ca": (
                        final_heldout_metrics["ca"]
                        if final_heldout_metrics is not None
                        else None
                    ),
                }
            )

    write_csv(output / "matched_classical_per_run.csv", rows)
    summary_rows: list[dict] = []
    for lambda_r in lambdas:
        selected = [row for row in rows if row["lambda_r"] == lambda_r]
        summary_rows.append(
            {
                "lambda_r": lambda_r,
                "n_seeds": len(selected),
                "clean_pretrain_ca_mean": float(
                    np.mean([row["clean_pretrain_ca"] for row in selected])
                ),
                "clean_pretrain_asr_mean": float(
                    np.mean([row["clean_pretrain_asr"] for row in selected])
                ),
                "final_ca_mean": float(np.mean([row["final_ca"] for row in selected])),
                "final_ca_population_sd": float(
                    np.std([row["final_ca"] for row in selected], ddof=0)
                ),
                "final_asr_mean": float(np.mean([row["final_asr"] for row in selected])),
                "final_auc_xyz_mean": float(
                    np.mean([row["final_auc_xyz"] for row in selected])
                ),
                "clean_pretrain_heldout_ca_mean": (
                    float(
                        np.mean(
                            [
                                row["clean_pretrain_heldout_ca"]
                                for row in selected
                                if row["clean_pretrain_heldout_ca"] is not None
                            ]
                        )
                    )
                    if any(row["clean_pretrain_heldout_ca"] is not None for row in selected)
                    else None
                ),
                "final_heldout_ca_mean": (
                    float(
                        np.mean(
                            [
                                row["final_heldout_ca"]
                                for row in selected
                                if row["final_heldout_ca"] is not None
                            ]
                        )
                    )
                    if any(row["final_heldout_ca"] is not None for row in selected)
                    else None
                ),
            }
        )
    write_csv(output / "matched_classical_summary.csv", summary_rows)
    (output / "matched_classical_runs.json").write_text(
        json.dumps(run_records, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary_rows, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "paper14/results/bloodmnist_ca_diagnosis_v1",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 1.0])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Optional official BloodMNIST root for untouched held-out evaluation",
    )
    args = parser.parse_args()
    main(
        args.output.resolve(),
        args.seeds,
        args.lambdas,
        args.epochs,
        args.dataset_root.resolve() if args.dataset_root is not None else None,
    )
