#!/usr/bin/env python3
"""White-box adaptive attacks against the frozen BloodMNIST CF-CMD rule.

This is a development resource gate.  Seed 42 selects one pre-registered
evasion weight per attack from {0.1, 1, 10}; seeds 46 and 51 reuse it without
retuning.  Test CA/ASR are read exactly once after checkpoint selection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
import run_bloodmnist_unified_oneclass as blood
import run_mnist_group_coherent_drift as coherent
import run_unified_oneclass_poison_screening as unified
import train_bloodmnist_clean_encoder as clean_encoder


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PAPER14_ROOT / "results/bloodmnist_whitebox_adaptive_pilot_v1_1"
PROTOCOL = (
    PAPER14_ROOT / "results/top_journal_program_v1/phase3_adaptive_protocol_v1_1.json"
)
REFERENCE_CLEAN_ROOT = PAPER14_ROOT / "results/bloodmnist_group_coherent_10seeds_v1"
MODES = (
    "checkpoint_drift_representation_matching",
    "multi_direction_coherence_breaking",
    "joint_input_internal_detector_evasion",
)
PILOT_SEEDS = (42, 46, 51)
EVASION_WEIGHTS = (0.1, 1.0, 10.0)
INPUT_NAME = "coherent_input256"
XZ_NAME = "coherent_qml_xz_drift16"
PRIMARY_NAME = "coherent_bonferroni_input_qml_xz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray | torch.Tensor) -> str:
    array = (
        values.detach().cpu().numpy()
        if isinstance(values, torch.Tensor)
        else np.asarray(values)
    )
    canonical = np.ascontiguousarray(array.astype("<f4", copy=False))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def xz_features(model: clean_encoder.CleanQNN, x: torch.Tensor) -> torch.Tensor:
    latent = model.encoder.projected_latent(x)
    alpha = (math.pi / 2.0) * latent
    return torch.cat([torch.cos(alpha), torch.sin(alpha)], dim=1)


def numpy_xz(
    model: clean_encoder.CleanQNN, x: np.ndarray, batch_size: int = 256
) -> np.ndarray:
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32)
            rows.append(xz_features(model, batch).cpu().numpy())
    return np.concatenate(rows, axis=0)


def checkpoint_drift(
    clean_model: clean_encoder.CleanQNN,
    attacked_model: clean_encoder.CleanQNN,
    x: np.ndarray,
) -> np.ndarray:
    return numpy_xz(attacked_model, x) - numpy_xz(clean_model, x)


def distribution_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Scale-stable first/two-moment matching used by the adaptive proxy."""

    mean_term = F.mse_loss(left.mean(dim=0), right.mean(dim=0))
    variance_term = F.mse_loss(
        left.var(dim=0, unbiased=False), right.var(dim=0, unbiased=False)
    )
    return mean_term + 0.25 * variance_term


def balanced_ca_and_asr(
    model: clean_encoder.CleanQNN,
    target: np.ndarray,
    source: np.ndarray,
    triggered_source: np.ndarray,
) -> tuple[float, float]:
    target_prediction, _ = adaptive.predict(model, target, 256)
    source_prediction, _ = adaptive.predict(model, source, 256)
    trigger_prediction, _ = adaptive.predict(model, triggered_source, 256)
    ca = 0.5 * (
        float(np.mean(target_prediction == 1))
        + float(np.mean(source_prediction == 0))
    )
    return ca, float(np.mean(trigger_prediction == 1))


def initial_pca_trigger(prepared: dict, rgb_trigger: np.ndarray) -> np.ndarray:
    trigger = (
        rgb_trigger / prepared["raw_scaler"].scale_
    ) @ prepared["pca"].components_.T
    maximum = float(np.max(np.abs(trigger)))
    if maximum <= 0:
        raise RuntimeError("Initial trigger is identically zero")
    return (0.8 / maximum * trigger).astype(np.float32)


def apply_pca_trigger_numpy(
    pca_values: np.ndarray, trigger: np.ndarray, prepared: dict
) -> np.ndarray:
    triggered = np.clip(pca_values + trigger, -3.0, 3.0)
    return prepared["post_scaler"].transform(triggered).astype(np.float32)


def fixed_directions(seed: int) -> torch.Tensor:
    rng = np.random.default_rng(20260928 + 4099 * seed)
    matrix = rng.normal(size=(16, 2))
    q, _ = np.linalg.qr(matrix)
    directions = np.stack([q[:, 0], -q[:, 0], q[:, 1], -q[:, 1]])
    return torch.as_tensor(directions.astype(np.float32))


def detector_evaluation(
    *,
    mode: str,
    seed: int,
    clean_model: clean_encoder.CleanQNN,
    attacked_model: clean_encoder.CleanQNN,
    trusted: np.ndarray,
    candidate_clean: np.ndarray,
    poison: np.ndarray,
    alpha: float,
) -> tuple[dict[str, tuple[dict, list[dict]]], dict[str, dict[str, np.ndarray]]]:
    groups = {
        "trusted": trusted.astype(np.float64),
        "clean": candidate_clean.astype(np.float64),
        "poison": poison.astype(np.float64),
    }
    xz_groups = {
        name: checkpoint_drift(clean_model, attacked_model, values.astype(np.float32))
        for name, values in groups.items()
    }
    config = {
        "seed": seed,
        "trigger": mode,
        "archive_ca": 0.0,
        "archive_asr": 0.0,
    }
    output: dict[str, tuple[dict, list[dict]]] = {}
    for name, features in ((INPUT_NAME, groups), (XZ_NAME, xz_groups)):
        row, samples = cross.evaluate_external_reference(
            config,
            name,
            features["trusted"],
            features["clean"],
            features["poison"],
            alpha,
        )
        output[name] = (row, samples)
    combined_row, combined_samples = coherent.combine_margin_and_correlator(
        config,
        output[INPUT_NAME][0],
        output[INPUT_NAME][1],
        output[XZ_NAME][1],
        alpha,
        combined_name=PRIMARY_NAME,
        second_dimension=16,
    )
    combined_row["trusted_reference_split"] = "validation_target"
    output[PRIMARY_NAME] = (combined_row, combined_samples)
    return output, {"input": groups, "xz_drift": xz_groups}


def covariance_free_metrics(
    seed: int,
    trusted: np.ndarray,
    clean: np.ndarray,
    poison: np.ndarray,
    alpha: float,
) -> dict:
    """External-reference cross-fit score without covariance whitening."""

    trusted = np.asarray(trusted, dtype=np.float64)
    features = np.concatenate([clean, poison], axis=0).astype(np.float64)
    labels = np.concatenate(
        [np.zeros(len(clean), dtype=int), np.ones(len(poison), dtype=int)]
    )
    rng = np.random.default_rng(20260928 + 1009 * seed)
    order = rng.permutation(len(trusted))
    reference, calibration = trusted[order[:120]], trusted[order[120:]]
    mean = reference.mean(axis=0, keepdims=True)
    reference_centered = reference - mean
    calibration_centered = calibration - mean
    scores = np.empty(len(features), dtype=np.float64)
    flags = np.zeros(len(features), dtype=bool)
    folds = np.array_split(rng.permutation(len(features)), 4)
    for query_indices in folds:
        halves = np.array_split(rng.permutation(query_indices), 2)
        for score_half, direction_half in (
            (halves[0], halves[1]),
            (halves[1], halves[0]),
        ):
            direction = (features[direction_half] - mean).mean(axis=0) - reference_centered.mean(axis=0)
            norm = float(np.linalg.norm(direction))
            if norm < 1e-12:
                direction = np.ones(features.shape[1], dtype=np.float64)
                norm = float(np.linalg.norm(direction))
            direction /= norm
            calibration_scores = calibration_centered @ direction
            query_scores = (features[score_half] - mean) @ direction
            scores[score_half] = query_scores
            flags[score_half] = (
                coherent.one_sided_p_values(calibration_scores, query_scores) <= alpha
            )
    clean_flags, poison_flags = flags[labels == 0], flags[labels == 1]
    return {
        "tpr": float(poison_flags.mean()),
        "fpr": float(clean_flags.mean()),
        "tp": int(poison_flags.sum()),
        "fn": int((~poison_flags).sum()),
        "fp": int(clean_flags.sum()),
        "tn": int((~clean_flags).sum()),
    }


def point_key(point: dict, selection_detector: str = "frozen_dual_view") -> tuple:
    eligible = point["validation_balanced_ca"] >= 0.90 and point["validation_asr"] >= 0.80
    if selection_detector == "covariance_free_xz16":
        primary_tpr = point["covariance_free_tpr"]
        secondary_tpr = point["covariance_free_tpr"]
    elif selection_detector == "frozen_dual_view":
        primary_tpr = point["primary_tpr"]
        secondary_tpr = point["xz_tpr"]
    else:
        raise ValueError(f"Unknown selection detector: {selection_detector}")
    if eligible:
        return (
            1,
            -primary_tpr,
            -secondary_tpr,
            point["validation_balanced_ca"],
            point["validation_asr"],
            -point["epoch"],
        )
    gate_margin = min(
        point["validation_balanced_ca"] / 0.90,
        point["validation_asr"] / 0.80,
    )
    return (
        0,
        gate_margin,
        point["validation_balanced_ca"],
        point["validation_asr"],
        -primary_tpr,
        -point["epoch"],
    )


def prepare_seed(
    class_data: dict[int, np.ndarray],
    rgb_trigger: np.ndarray,
    seed: int,
    clean_root: Path = REFERENCE_CLEAN_ROOT,
) -> dict:
    prepared = baseline.preprocess_seed(class_data, seed)
    splits, trigger_receipt = baseline.seed_splits(prepared, rgb_trigger, seed)
    clean_model, clean_record = cross.load_reference_clean_model(
        seed, clean_root
    )
    trigger = initial_pca_trigger(prepared, rgb_trigger)
    poison_indices = np.asarray(trigger_receipt["poison_indices"], dtype=int)
    rebuilt = apply_pca_trigger_numpy(
        prepared["pca_splits"]["train"][0][poison_indices], trigger, prepared
    )
    error = float(np.max(np.abs(rebuilt - splits["archived_poison"])))
    if error > 1e-5:
        raise RuntimeError(f"Seed {seed} trigger reconstruction mismatch: {error}")
    return {
        "prepared": prepared,
        "splits": splits,
        "trigger_receipt": trigger_receipt,
        "trigger": trigger,
        "poison_indices": poison_indices,
        "clean_model": clean_model,
        "clean_record": clean_record,
        "trigger_reconstruction_max_abs_error": error,
    }


def run_configuration(
    seed_data: dict,
    *,
    seed: int,
    mode: str,
    evasion_weight: float,
    args: argparse.Namespace,
) -> dict:
    torch.manual_seed(seed + int(100 * evasion_weight) + 97 * MODES.index(mode))
    np.random.seed(seed + int(100 * evasion_weight) + 97 * MODES.index(mode))
    clean_model = seed_data["clean_model"]
    model = copy.deepcopy(clean_model)
    # ``clean_model`` is reused across the frozen seed-42 weight candidates.
    # A prior candidate may have marked its parameters non-trainable; restore
    # the copied attacker's projector explicitly before building gradients.
    for parameter in model.encoder.projector.parameters():
        parameter.requires_grad_(True)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(False)
    for parameter in clean_model.parameters():
        parameter.requires_grad_(False)
    model.train()

    splits = seed_data["splits"]
    prepared = seed_data["prepared"]
    x_clean = splits["train"]["clean_x"]
    y_clean = splits["train"]["clean_y"]
    target_train = splits["train"]["target"]
    fixed_poison = splits["archived_poison"]
    poison_pca_base = prepared["pca_splits"]["train"][0][seed_data["poison_indices"]]
    clean_tensor = torch.as_tensor(x_clean, dtype=torch.float32)
    with torch.no_grad():
        teacher_logits = clean_model(clean_tensor).detach()

    projector_parameters = list(model.encoder.projector.parameters())
    initial_parameters = [parameter.detach().clone() for parameter in projector_parameters]
    correction_parameters: list[torch.nn.Parameter] = []
    if mode == "joint_input_internal_detector_evasion":
        correction_parameters = [
            torch.nn.Parameter(torch.zeros(256, dtype=torch.float32)),
            torch.nn.Parameter(torch.zeros(256, dtype=torch.float32)),
        ]
    directions = fixed_directions(seed)
    initial_trigger = torch.as_tensor(seed_data["trigger"], dtype=torch.float32)
    pca_mean = torch.as_tensor(prepared["post_scaler"].mean_, dtype=torch.float32)
    pca_scale = torch.as_tensor(prepared["post_scaler"].scale_, dtype=torch.float32)
    poison_pca_tensor = torch.as_tensor(poison_pca_base, dtype=torch.float32)
    rng = np.random.default_rng(seed + 7301)

    def current_trigger(base: torch.Tensor | None = None) -> torch.Tensor:
        if not correction_parameters:
            return initial_trigger
        if base is None:
            base = poison_pca_tensor
        diagonal, bias = correction_parameters
        standardized = (base - pca_mean) / pca_scale
        correction = args.trigger_correction_linf * torch.tanh(
            standardized * diagonal + bias
        )
        return torch.clamp(
            initial_trigger[None, :] + correction,
            -args.trigger_linf,
            args.trigger_linf,
        )

    def joint_poison(indices: np.ndarray | None = None) -> torch.Tensor:
        base = poison_pca_tensor if indices is None else poison_pca_tensor[indices]
        triggered = torch.clamp(base + current_trigger(base), -3.0, 3.0)
        return (triggered - pca_mean) / pca_scale

    def poison_numpy() -> np.ndarray:
        if not correction_parameters:
            return fixed_poison
        with torch.no_grad():
            return joint_poison().cpu().numpy().astype(np.float32)

    def triggered_split(split: str) -> np.ndarray:
        if not correction_parameters:
            return splits[split]["triggered_non_target"]
        with torch.no_grad():
            base = torch.as_tensor(
                prepared["pca_splits"][split][0], dtype=torch.float32
            )
            triggered = torch.clamp(base + current_trigger(base), -3.0, 3.0)
            return ((triggered - pca_mean) / pca_scale).cpu().numpy().astype(np.float32)

    histories: list[dict] = []
    states: list[dict] = []
    started = time.time()

    def record(epoch: int, objective: dict | None) -> None:
        model.eval()
        poison = poison_numpy()
        val_ca, val_asr = balanced_ca_and_asr(
            model,
            splits["val"]["target"],
            splits["val"]["non_target"],
            triggered_split("val"),
        )
        evaluated, _ = detector_evaluation(
            mode=mode,
            seed=seed,
            clean_model=clean_model,
            attacked_model=model,
            trusted=splits["val"]["target"],
            candidate_clean=target_train,
            poison=poison,
            alpha=args.alpha,
        )
        xz_trusted = checkpoint_drift(
            clean_model, model, splits["val"]["target"]
        )
        xz_clean = checkpoint_drift(clean_model, model, target_train)
        xz_poison = checkpoint_drift(clean_model, model, poison)
        covariance_free = covariance_free_metrics(
            seed, xz_trusted, xz_clean, xz_poison, args.alpha
        )
        histories.append(
            {
                "epoch": epoch,
                "objective": objective,
                "validation_balanced_ca": val_ca,
                "validation_asr": val_asr,
                "input_tpr": evaluated[INPUT_NAME][0]["tpr"],
                "input_fpr": evaluated[INPUT_NAME][0]["fpr"],
                "xz_tpr": evaluated[XZ_NAME][0]["tpr"],
                "xz_fpr": evaluated[XZ_NAME][0]["fpr"],
                "primary_tpr": evaluated[PRIMARY_NAME][0]["tpr"],
                "primary_fpr": evaluated[PRIMARY_NAME][0]["fpr"],
                "covariance_free_tpr": covariance_free["tpr"],
                "covariance_free_fpr": covariance_free["fpr"],
                "elapsed_seconds": time.time() - started,
            }
        )
        states.append(
            {
                "projector": copy.deepcopy(model.encoder.projector.state_dict()),
                "correction": [
                    parameter.detach().clone() for parameter in correction_parameters
                ],
            }
        )
        model.train()

    record(0, None)
    for epoch in range(1, args.epochs + 1):
        clean_order = rng.permutation(len(x_clean))
        poison_order = rng.permutation(len(fixed_poison))
        target_order = rng.permutation(len(target_train))
        poison_cursor = 0
        target_cursor = 0
        sums = {
            "clean_ce": 0.0,
            "backdoor_ce": 0.0,
            "evasion": 0.0,
            "internal_match": 0.0,
            "input_match": 0.0,
            "distill": 0.0,
            "anchor": 0.0,
            "trigger_change": 0.0,
            "total": 0.0,
            "gradient_norm_max": 0.0,
            "updates": 0,
        }
        for start in range(0, len(x_clean), args.batch_size):
            clean_indices = clean_order[start : start + args.batch_size]
            size = len(clean_indices)
            if poison_cursor + size > len(poison_order):
                poison_order = rng.permutation(len(fixed_poison))
                poison_cursor = 0
            poison_indices = poison_order[poison_cursor : poison_cursor + size]
            poison_cursor += size
            if len(poison_indices) < size:
                poison_indices = np.concatenate(
                    [
                        poison_indices,
                        rng.choice(len(fixed_poison), size - len(poison_indices), replace=True),
                    ]
                )
            if target_cursor + size > len(target_order):
                target_order = rng.permutation(len(target_train))
                target_cursor = 0
            target_indices = target_order[target_cursor : target_cursor + size]
            target_cursor += size
            if len(target_indices) < size:
                target_indices = np.concatenate(
                    [
                        target_indices,
                        rng.choice(len(target_train), size - len(target_indices), replace=True),
                    ]
                )

            clean_batch = torch.as_tensor(x_clean[clean_indices], dtype=torch.float32)
            clean_labels = torch.as_tensor(y_clean[clean_indices], dtype=torch.long)
            target_batch = torch.as_tensor(target_train[target_indices], dtype=torch.float32)
            if not correction_parameters:
                poison_batch = torch.as_tensor(fixed_poison[poison_indices], dtype=torch.float32)
            else:
                poison_batch = joint_poison(poison_indices)

            clean_logits = model(clean_batch)
            poison_logits = model(poison_batch)
            clean_loss = F.cross_entropy(clean_logits, clean_labels)
            backdoor_loss = F.cross_entropy(
                poison_logits, torch.ones(len(poison_batch), dtype=torch.long)
            )
            with torch.no_grad():
                clean_checkpoint_poison = xz_features(clean_model, poison_batch).detach()
                clean_checkpoint_target = xz_features(clean_model, target_batch).detach()
            poison_drift = xz_features(model, poison_batch) - clean_checkpoint_poison
            target_drift = xz_features(model, target_batch) - clean_checkpoint_target
            internal_match = distribution_loss(poison_drift, target_drift)
            input_match = torch.zeros((), dtype=torch.float32)
            trigger_change = torch.zeros((), dtype=torch.float32)

            if mode == "checkpoint_drift_representation_matching":
                evasion = internal_match
            elif mode == "multi_direction_coherence_breaking":
                clean_centroid = target_drift.mean(dim=0, keepdim=True)
                assigned = directions[
                    torch.as_tensor(poison_indices % 4, dtype=torch.long)
                ]
                desired = clean_centroid + args.direction_amplitude * assigned
                directional = F.mse_loss(poison_drift, desired)
                evasion = directional + 0.25 * internal_match
            elif mode == "joint_input_internal_detector_evasion":
                input_match = distribution_loss(poison_batch, target_batch)
                trigger_change = torch.mean(
                    (current_trigger(poison_pca_tensor) - initial_trigger[None, :]) ** 2
                )
                evasion = internal_match + input_match + 0.1 * trigger_change
            else:
                raise ValueError(mode)

            temperature = 2.0
            distill = F.kl_div(
                F.log_softmax(clean_logits / temperature, dim=1),
                F.softmax(teacher_logits[clean_indices] / temperature, dim=1),
                reduction="batchmean",
            ) * temperature**2
            anchor = torch.stack(
                [
                    torch.mean((parameter - initial) ** 2)
                    for parameter, initial in zip(projector_parameters, initial_parameters)
                ]
            ).mean()
            objective = (
                clean_loss
                + args.backdoor_weight * backdoor_loss
                + evasion_weight * evasion
                + args.distill_weight * distill
                + args.anchor_weight * anchor
            )
            parameters = projector_parameters + correction_parameters
            gradients = torch.autograd.grad(objective, parameters)
            projector_gradients = gradients[: len(projector_parameters)]
            correction_gradients = gradients[len(projector_parameters) :]
            projector_norm = torch.sqrt(
                sum(torch.sum(g**2) for g in projector_gradients)
            )
            correction_norm = (
                torch.sqrt(sum(torch.sum(g**2) for g in correction_gradients))
                if correction_gradients
                else torch.zeros((), dtype=torch.float32)
            )
            gradient_norm = torch.sqrt(projector_norm**2 + correction_norm**2)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Non-finite adaptive gradient")
            projector_scale = min(
                1.0, args.gradient_clip / max(float(projector_norm), 1e-12)
            )
            correction_scale = min(
                1.0, args.gradient_clip / max(float(correction_norm), 1e-12)
            )
            with torch.no_grad():
                for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
                    if index < len(projector_parameters):
                        parameter -= args.learning_rate * projector_scale * gradient
                    else:
                        parameter -= args.trigger_learning_rate * correction_scale * gradient

            for name, value in (
                ("clean_ce", clean_loss),
                ("backdoor_ce", backdoor_loss),
                ("evasion", evasion),
                ("internal_match", internal_match),
                ("input_match", input_match),
                ("distill", distill),
                ("anchor", anchor),
                ("trigger_change", trigger_change),
                ("total", objective),
            ):
                sums[name] += float(value.detach())
            sums["gradient_norm_max"] = max(sums["gradient_norm_max"], float(gradient_norm))
            sums["updates"] += 1

        objective_summary = {
            name: value / sums["updates"]
            for name, value in sums.items()
            if name not in {"updates", "gradient_norm_max"}
        }
        objective_summary["gradient_norm_max"] = sums["gradient_norm_max"]
        objective_summary["updates"] = sums["updates"]
        record(epoch, objective_summary)

    selection_detector = getattr(args, "selection_detector", "frozen_dual_view")
    selected_index = max(
        range(len(histories)),
        key=lambda index: point_key(histories[index], selection_detector),
    )
    model.encoder.projector.load_state_dict(states[selected_index]["projector"])
    if correction_parameters:
        with torch.no_grad():
            for parameter, selected_value in zip(
                correction_parameters, states[selected_index]["correction"]
            ):
                parameter.copy_(selected_value)
    selected = copy.deepcopy(histories[selected_index])
    selected["eligible_on_validation"] = bool(
        selected["validation_balanced_ca"] >= 0.90
        and selected["validation_asr"] >= 0.80
    )
    selected["selection_key"] = list(point_key(selected, selection_detector))
    selected["selection_detector"] = selection_detector
    final_trigger = current_trigger(poison_pca_tensor).detach().cpu().numpy().astype(np.float32)
    return {
        "mode": mode,
        "seed": seed,
        "evasion_weight": evasion_weight,
        "history": histories,
        "selected": selected,
        "model": model,
        "poison": poison_numpy(),
        "final_trigger": final_trigger,
        "triggered_split": triggered_split,
        "projector_state": copy.deepcopy(model.encoder.projector.state_dict()),
        "trigger_corrector_state": [
            parameter.detach().cpu().numpy().astype(np.float32)
            for parameter in correction_parameters
        ],
        "trigger_linf": float(np.max(np.abs(final_trigger))),
        "trigger_l2": float(np.linalg.norm(final_trigger)),
        "trigger_correction_linf": float(
            np.max(np.abs(final_trigger - seed_data["trigger"][None, :]))
        ),
    }


def attempt_key(result: dict) -> tuple:
    point = result["selected"]
    return (*point_key(point), -abs(math.log10(result["evasion_weight"])))


def save_selected(
    output: Path,
    result: dict,
    seed_data: dict,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], dict, dict]:
    mode, seed = result["mode"], result["seed"]
    model: clean_encoder.CleanQNN = result["model"]
    splits = seed_data["splits"]
    test_ca, test_asr = balanced_ca_and_asr(
        model,
        splits["test"]["target"],
        splits["test"]["non_target"],
        result["triggered_split"]("test"),
    )
    detector, features = detector_evaluation(
        mode=mode,
        seed=seed,
        clean_model=seed_data["clean_model"],
        attacked_model=model,
        trusted=splits["val"]["target"],
        candidate_clean=splits["train"]["target"],
        poison=result["poison"],
        alpha=args.alpha,
    )
    model_dir = output / "models" / mode
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / f"seed_{seed}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "seed": seed,
            "mode": mode,
            "evasion_weight": result["evasion_weight"],
            "selected_epoch": result["selected"]["epoch"],
            "pca_trigger": result["final_trigger"],
            "initial_pca_trigger": seed_data["trigger"],
            "trigger_corrector_state": result["trigger_corrector_state"],
        },
        checkpoint,
    )
    rows, samples = [], []
    for name in (INPUT_NAME, XZ_NAME, PRIMARY_NAME):
        row, sample_rows = detector[name]
        row["archive_ca"] = test_ca
        row["archive_asr"] = test_asr
        row["selected_epoch"] = result["selected"]["epoch"]
        row["evasion_weight"] = result["evasion_weight"]
        rows.append(row)
        samples.extend(sample_rows)
    victim = {
        "mode": mode,
        "seed": seed,
        "evasion_weight": result["evasion_weight"],
        "selected_epoch": result["selected"]["epoch"],
        "validation_balanced_ca": result["selected"]["validation_balanced_ca"],
        "validation_asr": result["selected"]["validation_asr"],
        "test_balanced_ca": test_ca,
        "test_asr": test_asr,
        "test_eligible": int(test_ca >= 0.90 and test_asr >= 0.80),
        "trigger_linf": result["trigger_linf"],
        "trigger_l2": result["trigger_l2"],
        "trigger_correction_linf": result["trigger_correction_linf"],
        "checkpoint": str(checkpoint.relative_to(output)),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    receipt = {
        "mode": mode,
        "seed": seed,
        "evasion_weight": result["evasion_weight"],
        "selected_epoch": result["selected"]["epoch"],
        "poison_sha256_float32": sha256_array(result["poison"]),
        "input_trusted_sha256_float32": sha256_array(features["input"]["trusted"]),
        "xz_poison_sha256_float32": sha256_array(features["xz_drift"]["poison"]),
        "initial_trigger_sha256_float32": sha256_array(seed_data["trigger"]),
        "final_trigger_sha256_float32": sha256_array(result["final_trigger"]),
        "checkpoint_sha256": victim["checkpoint_sha256"],
        "trigger_reconstruction_max_abs_error": seed_data[
            "trigger_reconstruction_max_abs_error"
        ],
    }
    return rows, samples, victim, receipt


def eligible_summaries(rows: list[dict], victims: list[dict]) -> list[dict]:
    eligible = {(row["mode"], int(row["seed"])) for row in victims if row["test_eligible"]}
    output = []
    for mode in MODES:
        for representation in (INPUT_NAME, XZ_NAME, PRIMARY_NAME):
            selected = [
                row
                for row in rows
                if row["trigger"] == mode
                and row["representation"] == representation
                and (mode, int(row["seed"])) in eligible
            ]
            seeds = sorted(int(row["seed"]) for row in selected)
            if not selected:
                output.append(
                    {
                        "mode": mode,
                        "representation": representation,
                        "eligible_seed_count": 0,
                        "eligible_seeds": "",
                        "status": "NO_ELIGIBLE_VICTIM",
                    }
                )
                continue
            summary = coherent.aggregate(selected, "representation")[0]
            promoted = (
                representation == PRIMARY_NAME
                and len(seeds) >= 2
                and (summary["pooled_tpr"] < 0.90 or summary["pooled_fpr"] > 0.05)
            )
            output.append(
                {
                    "mode": mode,
                    "representation": representation,
                    "eligible_seed_count": len(seeds),
                    "eligible_seeds": ",".join(map(str, seeds)),
                    "status": "PROMOTE_TO_FRESH_CONFIRMATION" if promoted else "PILOT_ONLY",
                    **{key: value for key, value in summary.items() if key != "representation"},
                }
            )
    return output


def write_report(
    output: Path,
    attempts: list[dict],
    victims: list[dict],
    summaries: list[dict],
    elapsed: float,
) -> None:
    lines = [
        "# BloodMNIST white-box adaptive attack pilot",
        "",
        "This is a pre-registered three-seed development gate, not confirmatory evidence. Seed 42 selects among exactly three frozen evasion weights; seeds 46 and 51 reuse the selected weight. The test split is read only after model selection for one-shot CA/ASR eligibility.",
        "",
        "## Seed-42 weight search",
        "",
        "| attack | weight | epoch | val CA | val ASR | primary TPR | XZ16 TPR | selected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in attempts:
        lines.append(
            f"| {row['mode']} | {row['evasion_weight']:.1f} | {row['selected_epoch']} | "
            f"{row['validation_balanced_ca']:.3f} | {row['validation_asr']:.3f} | "
            f"{row['primary_tpr']:.3f} | {row['xz_tpr']:.3f} | {row['selected_for_mode']} |"
        )
    lines.extend(
        [
            "",
            "## Selected victim feasibility",
            "",
            "| attack | seed | weight | epoch | test CA | test ASR | eligible |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in victims:
        lines.append(
            f"| {row['mode']} | {row['seed']} | {row['evasion_weight']:.1f} | "
            f"{row['selected_epoch']} | {row['test_balanced_ca']:.3f} | "
            f"{row['test_asr']:.3f} | {row['test_eligible']} |"
        )
    lines.extend(
        [
            "",
            "## Attack-eligible detector results",
            "",
            "| attack | endpoint | eligible seeds | pooled TPR | pooled FPR | decision |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in summaries:
        if row["eligible_seed_count"] == 0:
            lines.append(
                f"| {row['mode']} | {row['representation']} | 0 | — | — | {row['status']} |"
            )
        else:
            lines.append(
                f"| {row['mode']} | {row['representation']} | {row['eligible_seed_count']} | "
                f"{row['pooled_tpr']:.3f} | {row['pooled_fpr']:.3f} | {row['status']} |"
            )
    promoted = [
        row["mode"]
        for row in summaries
        if row["representation"] == PRIMARY_NAME
        and row["status"] == "PROMOTE_TO_FRESH_CONFIRMATION"
    ]
    lines.extend(
        [
            "",
            "## Pilot decision",
            "",
            (
                "Fresh ten-seed confirmation is required for: " + ", ".join(promoted) + "."
                if promoted
                else "No adaptive attack satisfied the frozen promotion rule; feasibility failures remain reported rather than interpreted as detector robustness."
            ),
            "",
            f"Runtime: {elapsed:.1f} seconds.",
            "",
        ]
    )
    (output / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    if tuple(args.seeds) != PILOT_SEEDS:
        raise ValueError("This pilot is frozen to seeds 42 46 51")
    if tuple(args.evasion_weights) != EVASION_WEIGHTS:
        raise ValueError("The frozen evasion weights are 0.1 1.0 10.0")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["pilot_seeds"] != list(PILOT_SEEDS):
        raise RuntimeError("Protocol seed mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    class_data = heldout.collect_classes(args.dataset_root)
    rgb_trigger, rgb_receipt = baseline.fixed_rgb_trigger(class_data)
    seed_cache = {
        seed: prepare_seed(class_data, rgb_trigger, seed) for seed in args.seeds
    }

    all_rows: list[dict] = []
    all_samples: list[dict] = []
    victims: list[dict] = []
    receipts: list[dict] = []
    attempts: list[dict] = []
    attempt_histories: list[dict] = []
    selected_weights: dict[str, float] = {}

    for mode in MODES:
        print(f"mode={mode}: seed-42 frozen weight search", flush=True)
        tuning_results = []
        for weight in args.evasion_weights:
            result = run_configuration(
                seed_cache[42],
                seed=42,
                mode=mode,
                evasion_weight=weight,
                args=args,
            )
            tuning_results.append(result)
            attempt_histories.append(
                {
                    "mode": mode,
                    "seed": 42,
                    "evasion_weight": weight,
                    "history": result["history"],
                }
            )
        selected_tuning = max(tuning_results, key=attempt_key)
        selected_weights[mode] = float(selected_tuning["evasion_weight"])
        for result in tuning_results:
            point = result["selected"]
            attempts.append(
                {
                    "mode": mode,
                    "evasion_weight": result["evasion_weight"],
                    "selected_epoch": point["epoch"],
                    "validation_balanced_ca": point["validation_balanced_ca"],
                    "validation_asr": point["validation_asr"],
                    "primary_tpr": point["primary_tpr"],
                    "primary_fpr": point["primary_fpr"],
                    "xz_tpr": point["xz_tpr"],
                    "xz_fpr": point["xz_fpr"],
                    "eligible_on_validation": int(point["eligible_on_validation"]),
                    "selected_for_mode": int(result is selected_tuning),
                }
            )

        selected_runs = [selected_tuning]
        for seed in (46, 51):
            print(
                f"mode={mode}: seed={seed} frozen weight={selected_weights[mode]}",
                flush=True,
            )
            result = run_configuration(
                seed_cache[seed],
                seed=seed,
                mode=mode,
                evasion_weight=selected_weights[mode],
                args=args,
            )
            selected_runs.append(result)
            attempt_histories.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "evasion_weight": selected_weights[mode],
                    "history": result["history"],
                }
            )

        for result in selected_runs:
            rows, samples, victim, receipt = save_selected(
                args.output,
                result,
                seed_cache[result["seed"]],
                args,
            )
            all_rows.extend(rows)
            all_samples.extend(samples)
            victims.append(victim)
            receipts.append(receipt)
            print(
                f"mode={mode} seed={result['seed']}: test CA/ASR="
                f"{victim['test_balanced_ca']:.3f}/{victim['test_asr']:.3f}; "
                f"primary TPR/FPR={rows[-1]['tpr']:.3f}/{rows[-1]['fpr']:.3f}",
                flush=True,
            )

    summaries = eligible_summaries(all_rows, victims)
    unified.write_csv(args.output / "hyperparameter_attempts.csv", attempts)
    unified.write_csv(args.output / "per_config_metrics.csv", all_rows)
    unified.write_csv(args.output / "per_sample_scores.csv", all_samples)
    unified.write_csv(args.output / "victim_metrics.csv", victims)
    unified.write_csv(args.output / "eligible_summary.csv", summaries)
    unified.json_dump(args.output / "training_histories.json", attempt_histories)
    unified.json_dump(
        args.output / "manifest.json",
        {
            "experiment": "bloodmnist_whitebox_adaptive_pilot_v1_1",
            "role": "development_resource_gate_not_confirmation",
            "protocol": str(PROTOCOL),
            "protocol_sha256": sha256_file(PROTOCOL),
            "dataset_root": str(args.dataset_root),
            "dataset_npz_sha256": sha256_file(args.dataset_root / "bloodmnist.npz"),
            "seeds": list(args.seeds),
            "modes": list(MODES),
            "evasion_weights_searched_on_seed_42": list(args.evasion_weights),
            "selected_weights": selected_weights,
            "epochs": args.epochs,
            "alpha": args.alpha,
            "victim_gate": {"balanced_ca_min": 0.90, "asr_min": 0.80},
            "primary_gate": {"pooled_tpr_min": 0.90, "pooled_fpr_max": 0.05},
            "trusted_reference_split": "validation_target",
            "trusted_reference_count": 120,
            "trusted_calibration_count": 80,
            "candidate_clean_count": 600,
            "poison_count": 60,
            "test_use": "one-shot CA/ASR eligibility after validation-only selection",
            "rgb_trigger_receipt": rgb_receipt,
            "receipts": receipts,
            "elapsed_seconds": time.time() - started,
        },
    )
    write_report(args.output, attempts, victims, summaries, time.time() - started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PILOT_SEEDS))
    parser.add_argument(
        "--evasion-weights", nargs="+", type=float, default=list(EVASION_WEIGHTS)
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--trigger-learning-rate", type=float, default=0.002)
    parser.add_argument("--backdoor-weight", type=float, default=10.0)
    parser.add_argument("--distill-weight", type=float, default=20.0)
    parser.add_argument("--anchor-weight", type=float, default=0.001)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--direction-amplitude", type=float, default=0.2)
    parser.add_argument("--trigger-linf", type=float, default=0.8)
    parser.add_argument("--trigger-correction-linf", type=float, default=0.2)
    main(parser.parse_args())
