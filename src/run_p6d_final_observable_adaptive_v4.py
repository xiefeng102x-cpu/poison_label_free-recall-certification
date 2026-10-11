#!/usr/bin/env python3
"""P6-D v4: locked ten-seed extension of the final-observable adaptive attack."""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import beta

import run_p6d_final_observable_adaptive_v3 as v3


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "p6d_final_observable_adaptive_protocol_v4_extension"
PROTOCOL_PATH = ROOT / "results/top_journal_program_v1/p6d_final_observable_adaptive_protocol_v4.json"
DEFAULT_OUTPUT = ROOT / "results/p6d_final_observable_adaptive_v4"
V3_PROTOCOL = ROOT / "results/top_journal_program_v1/p6d_final_observable_adaptive_protocol_v3.json"
V3_RUNNER = ROOT / "src/run_p6d_final_observable_adaptive_v3.py"
V3_VERIFIER = ROOT / "src/verify_p6d_final_observable_adaptive_v3.py"
V3_ROOT = ROOT / "results/p6d_final_observable_adaptive_v3"
CONFIRMATION_SEEDS = tuple(range(124, 134))
FORMAL_ACKNOWLEDGEMENT = "CONSUME_P6D_V4_EXTENSION_SEEDS_124_133"
SELECTED_CONFIGURATION_ID = "w0.5"
SELECTED_EVASION_WEIGHT = 0.5
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_RNG_SEED = 20_261_006


def sha256_file(path: Path) -> str:
    return v3.sha256_file(path)


def load_protocol() -> dict[str, Any]:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("P6-D v4 protocol identifier mismatch")
    if tuple(payload["seed_audit"]["confirmation_extension_seeds"]) != CONFIRMATION_SEEDS:
        raise RuntimeError("P6-D v4 seed lock mismatch")
    audit = payload["source_audit"]
    if audit.get("v4_runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("P6-D v4 runner hash mismatch")
    verifier = ROOT / audit["v4_verifier"]
    if not verifier.is_file() or audit.get("v4_verifier_sha256") != sha256_file(verifier):
        raise RuntimeError("P6-D v4 verifier hash mismatch")
    upstream = payload["upstream_lock"]
    checks = {
        V3_PROTOCOL: "v3_protocol_sha256",
        V3_RUNNER: "v3_runner_sha256",
        V3_VERIFIER: "v3_verifier_sha256",
        V3_ROOT / "development/manifest.json": "v3_development_manifest_sha256",
        V3_ROOT / "development/development_selection_receipt.json": "v3_development_selection_sha256",
        V3_ROOT / "confirmation_lock_receipt.json": "v3_confirmation_lock_sha256",
        V3_ROOT / "confirmation/manifest.json": "v3_confirmation_manifest_sha256",
        V3_ROOT / "confirmation/confirmation_summary.json": "v3_confirmation_summary_sha256",
    }
    for path, key in checks.items():
        if not path.is_file() or sha256_file(path) != upstream[key]:
            raise RuntimeError(f"P6-D v4 upstream hash mismatch: {key}")
    return payload


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("Invalid exact-binomial counts")
    alpha = 1.0 - confidence
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return low, high


def one_sided_sign_pvalue(successes: int, trials: int) -> float:
    """Exact upper tail under the least-favourable null p=0.5."""
    if not 0 <= successes <= trials:
        raise ValueError("Invalid sign-test counts")
    return float(sum(math.comb(trials, k) for k in range(successes, trials + 1)) / (2**trials))


def summarize_outcomes(
    per_seed: Sequence[Mapping[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    rng_seed: int = BOOTSTRAP_RNG_SEED,
) -> dict[str, Any]:
    rows = [dict(row) for row in per_seed]
    seeds = sorted(int(row["seed"]) for row in rows)
    if seeds != list(CONFIRMATION_SEEDS) or len(rows) != len(CONFIRMATION_SEEDS):
        raise ValueError("P6-D v4 summary requires exactly the frozen ten-seed block")
    eligible = [row for row in rows if _boolean(row["eligible"])]
    n_all = len(rows)
    n_eligible = len(eligible)
    joint = [
        row for row in rows
        if _boolean(row["eligible"])
        and float(row["tpr"]) < 0.90
        and float(row["fpr"]) <= 0.05
    ]
    viability_ci = clopper_pearson(n_eligible, n_all)
    joint_ci = clopper_pearson(len(joint), n_all)
    summary: dict[str, Any] = {
        "all_seed_count": n_all,
        "attack_viable_count": n_eligible,
        "attack_viability_rate": n_eligible / n_all,
        "attack_viability_ci_low": viability_ci[0],
        "attack_viability_ci_high": viability_ci[1],
        "joint_adaptive_evasion_count": len(joint),
        "joint_adaptive_evasion_rate": len(joint) / n_all,
        "joint_adaptive_evasion_ci_low": joint_ci[0],
        "joint_adaptive_evasion_ci_high": joint_ci[1],
        "conditional_efficacy_reported": n_eligible > 0,
        "bootstrap_draws": int(draws),
        "bootstrap_rng_seed": int(rng_seed),
    }
    if n_eligible == 0:
        summary.update({
            "decision": "INSUFFICIENT_ATTACK_VIABILITY",
            "sign_successes": 0,
            "sign_trials": 0,
            "sign_test_pvalue_one_sided": 1.0,
        })
        return summary

    tprs = np.asarray([float(row["tpr"]) for row in eligible], dtype=np.float64)
    fprs = np.asarray([float(row["fpr"]) for row in eligible], dtype=np.float64)
    counts = np.asarray(
        [[int(row[name]) for name in ("tp", "fn", "fp", "tn")] for row in eligible],
        dtype=np.int64,
    )
    total = counts.sum(axis=0)
    tp, fn, fp, tn = map(int, total)
    rng = np.random.default_rng(rng_seed)
    chosen = rng.integers(0, n_eligible, size=(draws, n_eligible))
    mean_tpr_draws = tprs[chosen].mean(axis=1)
    mean_fpr_draws = fprs[chosen].mean(axis=1)
    tpr_low, tpr_high = np.quantile(mean_tpr_draws, [0.025, 0.975])
    fpr_low, fpr_high = np.quantile(mean_fpr_draws, [0.025, 0.975])
    sign_successes = int(np.sum(tprs < 0.90))
    sign_p = one_sided_sign_pvalue(sign_successes, n_eligible)
    evasion = (
        n_eligible >= 8
        and sign_p <= 0.05
        and float(tpr_high) < 0.90
        and float(fpr_high) <= 0.05
    )
    robustness = (
        n_eligible >= 8
        and float(tpr_low) >= 0.90
        and float(fpr_high) <= 0.05
    )
    if n_eligible < 8:
        decision = "INSUFFICIENT_ATTACK_VIABILITY"
    elif evasion:
        decision = "ADAPTIVE_EVASION_SUPPORTED"
    elif robustness:
        decision = "DETECTOR_ROBUSTNESS_SUPPORTED"
    else:
        decision = "INCONCLUSIVE_MIXED"
    summary.update({
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "pooled_tpr": tp / (tp + fn),
        "pooled_fpr": fp / (fp + tn),
        "mean_seed_tpr": float(tprs.mean()),
        "mean_seed_fpr": float(fprs.mean()),
        "min_seed_tpr": float(tprs.min()),
        "max_seed_tpr": float(tprs.max()),
        "min_seed_fpr": float(fprs.min()),
        "max_seed_fpr": float(fprs.max()),
        "mean_seed_tpr_ci_low": float(tpr_low),
        "mean_seed_tpr_ci_high": float(tpr_high),
        "mean_seed_fpr_ci_low": float(fpr_low),
        "mean_seed_fpr_ci_high": float(fpr_high),
        "sign_successes": sign_successes,
        "sign_trials": n_eligible,
        "sign_test_pvalue_one_sided": sign_p,
        "decision": decision,
    })
    return summary


def _environment() -> dict[str, Any]:
    import sklearn
    import scipy
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "device": "cpu",
        "deterministic_algorithms": True,
    }


def _stable_json_sha256(payload: Any) -> str:
    return v3.stable_json_sha256(payload)


def _run_signature(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "stage": "confirmation_extension",
        "seeds": list(CONFIRMATION_SEEDS),
        "selected_configuration_id": SELECTED_CONFIGURATION_ID,
        "selected_evasion_weight": SELECTED_EVASION_WEIGHT,
        "decision_rule_sha256": _stable_json_sha256(protocol["decision_rule"]),
        "counts": {
            "trusted": v3.TRUSTED_COUNT,
            "reference": v3.REFERENCE_COUNT,
            "calibration": v3.CALIBRATION_COUNT,
            "clean_candidate": v3.CLEAN_CANDIDATE_COUNT,
            "poison_candidate": v3.POISON_CANDIDATE_COUNT,
        },
        "attack_epochs": 12,
        "batch_size": 64,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }


def _initialize_manifest(
    output: Path, signature: Mapping[str, Any], immutable: Mapping[str, str]
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    signature_hash = _stable_json_sha256(signature)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("run_signature_sha256") != signature_hash:
            raise RuntimeError("Resume refused: run signature hash mismatch")
        if manifest.get("run_signature") != dict(signature):
            raise RuntimeError("Resume refused: run signature differs")
        if manifest.get("immutable_hashes") != dict(immutable):
            raise RuntimeError("Resume refused: immutable hashes differ")
        for relative, record in manifest.get("artifacts", {}).items():
            artifact = output / relative
            if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
                raise RuntimeError(f"Resume refused: corrupt artifact {relative}")
        expected = {"manifest.json", *manifest.get("artifacts", {})}
        actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
        extra = sorted(actual - expected)
        if extra:
            raise RuntimeError(f"Resume refused: untracked artifact {extra[0]}")
        return manifest
    if any(output.iterdir()):
        raise RuntimeError("Resume refused: non-empty stage has no manifest")
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "status": "initialized",
        "run_signature": dict(signature),
        "run_signature_sha256": signature_hash,
        "immutable_hashes": dict(immutable),
        "artifacts": {},
    }
    v3.atomic_json(path, manifest)
    return manifest


def _record_artifact(
    output: Path,
    manifest: Mapping[str, Any],
    artifact: Path,
    *,
    row_count: int | None = None,
) -> dict[str, Any]:
    artifact = artifact.resolve()
    relative = artifact.relative_to(output.resolve()).as_posix()
    updated = copy.deepcopy(dict(manifest))
    record: dict[str, Any] = {
        "sha256": sha256_file(artifact),
        "bytes": artifact.stat().st_size,
    }
    if row_count is not None:
        record["row_count"] = int(row_count)
    updated.setdefault("artifacts", {})[relative] = record
    v3.atomic_json(output / "manifest.json", updated)
    return updated


def issue_extension_lock(output: Path) -> dict[str, Any]:
    protocol = load_protocol()
    output.mkdir(parents=True, exist_ok=True)
    target = output / "confirmation_lock_receipt.json"
    sidecar = target.with_suffix(target.suffix + ".sha256")
    if target.exists() or sidecar.exists() or any(output.iterdir()):
        raise RuntimeError("Extension-lock output must be empty and is never overwritten")

    import verify_p6d_final_observable_adaptive_v3 as v3_verifier
    replay = v3_verifier.verify_bundle(V3_ROOT / "development")
    if replay.get("status") != "valid" or replay.get("development_selected_configuration_id") != SELECTED_CONFIGURATION_ID:
        raise RuntimeError("Frozen v3 development selection did not independently replay")
    selection_path = V3_ROOT / "development/development_selection_receipt.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("selected_configuration_id") != SELECTED_CONFIGURATION_ID:
        raise RuntimeError("Selected configuration differs from the frozen v3 choice")
    if float(selection.get("selected_evasion_weight")) != SELECTED_EVASION_WEIGHT:
        raise RuntimeError("Selected evasion weight differs from the frozen v3 choice")
    upstream_lock_path = V3_ROOT / "confirmation_lock_receipt.json"
    upstream_lock = json.loads(upstream_lock_path.read_text(encoding="utf-8"))
    current = _environment()
    for key in ("python", "platform", "numpy", "torch", "scikit_learn"):
        if upstream_lock["environment"].get(key) != current.get(key):
            raise RuntimeError(f"Execution environment differs from frozen v3 lock: {key}")
    lock = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "confirmation_extension_seeds": list(CONFIRMATION_SEEDS),
        "replacement_seeds": [],
        "selected_configuration_id": SELECTED_CONFIGURATION_ID,
        "selected_evasion_weight": SELECTED_EVASION_WEIGHT,
        "dataset_sha256": protocol["upstream_lock"]["dataset_sha256"],
        "upstream_hashes": copy.deepcopy(protocol["upstream_lock"]),
        "development_replay": replay,
        "decision_rule_sha256": _stable_json_sha256(protocol["decision_rule"]),
        "source_hashes": {
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
            str((ROOT / protocol["source_audit"]["v4_verifier"]).resolve()): protocol["source_audit"]["v4_verifier_sha256"],
            str(V3_RUNNER.resolve()): sha256_file(V3_RUNNER),
            str(V3_VERIFIER.resolve()): sha256_file(V3_VERIFIER),
        },
        "environment": current,
    }
    v3.atomic_json(target, lock)
    v3._runtime._atomic_bytes(sidecar, (sha256_file(target) + "\n").encode("utf-8"))
    return lock


def validate_extension_lock(output: Path) -> dict[str, Any]:
    protocol = load_protocol()
    path = output / "confirmation_lock_receipt.json"
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError("P6-D v4 confirmation lock and sidecar are required")
    if sidecar.read_text(encoding="utf-8").strip() != sha256_file(path):
        raise RuntimeError("P6-D v4 confirmation lock hash mismatch")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("P6-D v4 confirmation lock protocol mismatch")
    if lock.get("protocol_sha256") != sha256_file(PROTOCOL_PATH):
        raise RuntimeError("P6-D v4 protocol changed after locking")
    if tuple(lock.get("confirmation_extension_seeds", [])) != CONFIRMATION_SEEDS:
        raise RuntimeError("P6-D v4 confirmation seed lock mismatch")
    if lock.get("replacement_seeds") != []:
        raise RuntimeError("P6-D v4 replacement seeds are forbidden")
    if lock.get("decision_rule_sha256") != _stable_json_sha256(protocol["decision_rule"]):
        raise RuntimeError("P6-D v4 decision rule changed after locking")
    for source, digest in lock.get("source_hashes", {}).items():
        source_path = Path(source)
        if not source_path.is_file() or sha256_file(source_path) != digest:
            raise RuntimeError(f"P6-D v4 locked source changed: {source}")
    return lock


def run_confirmation_extension(args: argparse.Namespace) -> dict[str, Any]:
    if args.acknowledge_formal_seed_consumption != FORMAL_ACKNOWLEDGEMENT:
        raise PermissionError(
            "P6-D v4 formal execution requires --acknowledge-formal-seed-consumption "
            + FORMAL_ACKNOWLEDGEMENT
        )
    protocol = load_protocol()
    lock = validate_extension_lock(args.output)
    dataset_path = args.dataset_root / "bloodmnist.npz"
    if sha256_file(dataset_path) != lock["dataset_sha256"]:
        raise RuntimeError("P6-D v4 BloodMNIST hash mismatch")

    import evaluate_bloodmnist_checkpoints as heldout
    import run_bloodmnist_group_coherent_10seeds as baseline
    import run_bloodmnist_whitebox_adaptive_pilot as whitebox
    import train_bloodmnist_clean_encoder as clean_encoder
    import verify_p6d_final_observable_adaptive_v4 as verifier

    current = _environment()
    for key in ("python", "platform", "numpy", "torch", "scikit_learn"):
        if lock["environment"].get(key) != current.get(key):
            raise RuntimeError(f"P6-D v4 environment differs from lock: {key}")
    stage_root = args.output / "confirmation"
    signature = _run_signature(protocol)
    immutable = {
        "protocol": sha256_file(PROTOCOL_PATH),
        "dataset": sha256_file(dataset_path),
        "confirmation_lock": sha256_file(args.output / "confirmation_lock_receipt.json"),
        "runner": sha256_file(Path(__file__).resolve()),
        "verifier": sha256_file(Path(verifier.__file__).resolve()),
        "v3_runner": sha256_file(V3_RUNNER),
        "v3_verifier": sha256_file(V3_VERIFIER),
        "bloodmnist_preprocessor": sha256_file(Path(baseline.__file__).resolve()),
        "whitebox_metrics": sha256_file(Path(whitebox.__file__).resolve()),
        "clean_model": sha256_file(Path(clean_encoder.__file__).resolve()),
        "heldout_loader": sha256_file(Path(heldout.__file__).resolve()),
        "whitebox_model": sha256_file(Path(whitebox.adaptive.__file__).resolve()),
    }
    manifest = _initialize_manifest(stage_root, signature, immutable)
    if manifest.get("status") == "complete":
        return verifier.verify_bundle(stage_root)
    if manifest.get("status") != "initialized" or manifest.get("artifacts"):
        raise RuntimeError("Partial formal extension is retained but never recomputed")
    manifest["status"] = "in_progress"
    v3.atomic_json(stage_root / "manifest.json", manifest)

    root_lock = args.output / "confirmation_lock_receipt.json"
    for name, source in (
        ("confirmation_lock_receipt.json", root_lock),
        ("confirmation_lock_receipt.json.sha256", root_lock.with_suffix(root_lock.suffix + ".sha256")),
    ):
        destination = stage_root / name
        v3._runtime._atomic_bytes(destination, source.read_bytes())
        manifest = _record_artifact(stage_root, manifest, destination)
    env_path = stage_root / "environment.json"
    v3.atomic_json(env_path, current)
    manifest = _record_artifact(stage_root, manifest, env_path)

    classes = heldout.collect_classes(args.dataset_root)
    rgb_trigger, _ = baseline.fixed_rgb_trigger(classes)
    with np.load(dataset_path, allow_pickle=False) as payload:
        official_indices = {
            class_id: np.flatnonzero(payload["train_labels"].reshape(-1) == class_id)[: len(classes[class_id])]
            for class_id in (6, 0)
        }
    histories: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    observables: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    victims: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    split_receipts: dict[str, Any] = {}
    checkpoint_receipts: dict[str, Any] = {}
    v3._runtime.FORMAL_SEEDS = CONFIRMATION_SEEDS

    for seed in CONFIRMATION_SEEDS:
        splits, trigger_receipt, build_test_once, trigger = v3._p6d_prepare_seed(classes, rgb_trigger, seed, baseline)
        class_order: dict[int, dict[str, np.ndarray]] = {}
        for class_id in (6, 0):
            n_train = int(len(classes[class_id]) * 0.6)
            n_val = int(len(classes[class_id]) * 0.2)
            order = torch.randperm(
                len(classes[class_id]), generator=torch.Generator().manual_seed(seed)
            ).numpy()
            class_order[class_id] = {"train": order[:n_train], "val": order[n_train:n_train + n_val]}
        splits["train"]["target_original_indices"] = official_indices[6][class_order[6]["train"]]
        splits["val"]["target_original_indices"] = official_indices[6][class_order[6]["val"]]
        splits["train"]["non_target_original_indices"] = official_indices[0][class_order[0]["train"]]
        poison_ids = np.asarray(trigger_receipt["poison_indices"], dtype=np.int64)
        splits["poison_original_indices"] = splits["train"]["non_target_original_indices"][poison_ids]
        if (
            len(splits["val"]["target"]) != v3.TRUSTED_COUNT
            or len(splits["train"]["target"]) != v3.CLEAN_CANDIDATE_COUNT
            or len(splits["archived_poison"]) != v3.POISON_CANDIDATE_COUNT
        ):
            raise RuntimeError(f"P6-D v4 seed {seed} population count mismatch")
        split_receipts[str(seed)] = {
            "seed": seed,
            "trigger_sha256_float32": v3.sha256_array(trigger),
            "poison_indices_sha256": _stable_json_sha256(poison_ids.astype(int).tolist()),
            "poison_original_indices_sha256": _stable_json_sha256(splits["poison_original_indices"].astype(int).tolist()),
            "trusted_validation_target_sha256_float32": v3.sha256_array(splits["val"]["target"]),
            "clean_train_target_sha256_float32": v3.sha256_array(splits["train"]["target"]),
            "archived_poison_sha256_float32": v3.sha256_array(splits["archived_poison"]),
            "trigger_receipt": trigger_receipt,
        }
        clean_dir = stage_root / "clean_models/models"
        clean_dir.mkdir(parents=True, exist_ok=True)
        clean_path = clean_dir / f"seed_{seed}_clean.pt"
        clean_record_path = clean_dir / f"seed_{seed}_clean.json"
        clean_model, full_record = clean_encoder.train_one(
            {
                "train": (splits["train"]["clean_x"], splits["train"]["clean_y"]),
                "val": (splits["val"]["clean_x"], splits["val"]["clean_y"]),
            },
            seed, "local", 0, 40, 3e-3, 1e-4, 64, 10,
        )
        clean_record = {
            "seed": seed,
            "best_epoch": full_record["best_epoch"],
            "train": full_record["train"],
            "validation": full_record["val"],
            "epochs_completed": full_record["epochs_completed"],
        }
        v3.atomic_torch_save(clean_path, {"state_dict": clean_model.state_dict(), "seed": seed})
        v3.atomic_json(clean_record_path, clean_record)
        manifest = _record_artifact(stage_root, manifest, clean_path)
        manifest = _record_artifact(stage_root, manifest, clean_record_path)

        def validation_metric(model: torch.nn.Module) -> tuple[float, float]:
            return whitebox.balanced_ca_and_asr(
                model,
                splits["val"]["target"],
                splits["val"]["non_target"],
                splits["val"]["triggered_non_target"],
            )

        result = v3.train_attack_configuration(
            seed=seed,
            evasion_weight=SELECTED_EVASION_WEIGHT,
            clean_model=clean_model,
            splits={"train": splits["train"], "val": splits["val"], "archived_poison": splits["archived_poison"]},
            validation_metric=validation_metric,
        )
        exact = v3.numpy_cfcmdu_score(
            seed,
            result["selected_observables"]["trusted"],
            result["selected_observables"]["clean"],
            result["selected_observables"]["poison"],
            partition=result["partition"],
        )
        for row in result["history"]:
            histories.append({
                **row,
                "stage": "confirmation_extension",
                "seed": seed,
                "configuration_id": SELECTED_CONFIGURATION_ID,
                "evasion_weight": SELECTED_EVASION_WEIGHT,
                "selected_epoch": int(row["epoch"]) == int(result["selected"]["epoch"]),
            })
        score_input = {**result["selected_observables"], "partition": result["partition"]}
        scores.extend(v3._lossless_score_rows(
            "confirmation_extension", seed, SELECTED_CONFIGURATION_ID,
            score_input, splits, int(result["selected"]["epoch"]),
        ))
        observables.extend(v3._lossless_observable_rows(
            "confirmation_extension", seed, SELECTED_CONFIGURATION_ID,
            splits, result["selected_observables"], result["partition"],
            int(result["selected"]["epoch"]),
        ))
        folds.append({
            **result["partition"].to_json(),
            "stage": "confirmation_extension",
            "configuration_id": SELECTED_CONFIGURATION_ID,
        })
        checkpoint = stage_root / "models" / f"seed_{seed}_attacked.pt"
        v3.atomic_torch_save(checkpoint, {
            "seed": seed,
            "configuration_id": SELECTED_CONFIGURATION_ID,
            "selected_epoch": result["selected"]["epoch"],
            "model_state": result["model"].state_dict(),
        })
        manifest = _record_artifact(stage_root, manifest, checkpoint)
        checkpoint_receipts[str(seed)] = {
            "clean_checkpoint": {
                "path": str(clean_path.relative_to(stage_root)),
                "sha256": sha256_file(clean_path),
            },
            "attacked_checkpoint": {
                "path": str(checkpoint.relative_to(stage_root)),
                "sha256": sha256_file(checkpoint),
            },
        }
        test = build_test_once()
        ca, asr = whitebox.balanced_ca_and_asr(
            result["model"], test["target"], test["non_target"], test["triggered_non_target"]
        )
        eligible = bool(ca >= v3.TEST_CA_MIN and asr >= v3.TEST_ASR_MIN)
        victims.append({
            "stage": "confirmation_extension",
            "seed": seed,
            "configuration_id": SELECTED_CONFIGURATION_ID,
            "selected_epoch": int(result["selected"]["epoch"]),
            "test_balanced_ca": ca,
            "test_asr": asr,
            "eligible": eligible,
        })
        per_seed.append({
            "stage": "confirmation_extension",
            "seed": seed,
            "configuration_id": SELECTED_CONFIGURATION_ID,
            "selected_epoch": int(result["selected"]["epoch"]),
            "validation_balanced_ca": result["selected"]["validation_balanced_ca"],
            "validation_asr": result["selected"]["validation_asr"],
            "eligible": eligible,
            "tpr": exact["tpr"],
            "fpr": exact["fpr"],
            **{name: int(exact[name]) for name in ("tp", "fn", "fp", "tn")},
        })

    outputs = {
        "epoch_history.csv": histories,
        "per_sample_scores.csv": scores,
        "observable_rows.csv": observables,
        "per_seed_metrics.csv": per_seed,
        "victim_metrics.csv": victims,
    }
    for filename, rows in outputs.items():
        path = stage_root / filename
        v3.atomic_csv(path, rows)
        manifest = _record_artifact(stage_root, manifest, path, row_count=len(rows))
    for filename, payload in (
        ("fold_receipts.json", folds),
        ("split_receipts.json", split_receipts),
        ("checkpoint_receipts.json", checkpoint_receipts),
    ):
        path = stage_root / filename
        v3.atomic_json(path, payload)
        manifest = _record_artifact(stage_root, manifest, path)
    summary = summarize_outcomes(per_seed)
    summary_path = stage_root / "confirmation_summary.json"
    v3.atomic_json(summary_path, summary)
    manifest = _record_artifact(stage_root, manifest, summary_path)
    report_lines = [
        "# P6-D v4 ten-seed confirmation extension",
        "",
        f"Locked configuration: {SELECTED_CONFIGURATION_ID}.",
        f"Attack-viable victims: {summary['attack_viable_count']} / 10.",
        f"Joint viable-and-evasive seeds: {summary['joint_adaptive_evasion_count']} / 10.",
        f"Pre-registered decision: {summary['decision']}.",
        "All ten frozen seeds were run once; no seed was replaced and no efficacy stopping was used.",
        "",
    ]
    report_path = stage_root / "EXPERIMENT_REPORT.md"
    v3._runtime._atomic_bytes(report_path, "\n".join(report_lines).encode("utf-8"))
    manifest = _record_artifact(stage_root, manifest, report_path)
    manifest["status"] = "complete"
    v3.atomic_json(stage_root / "manifest.json", manifest)
    return verifier.verify_bundle(stage_root)


def self_check() -> dict[str, Any]:
    protocol = load_protocol()
    rows = []
    for index, seed in enumerate(CONFIRMATION_SEEDS):
        eligible = index < 8
        tpr = 0.2 + 0.05 * (index % 4)
        fpr = 0.01
        rows.append({
            "seed": seed,
            "eligible": eligible,
            "tpr": tpr,
            "fpr": fpr,
            "tp": int(round(60 * tpr)),
            "fn": 60 - int(round(60 * tpr)),
            "fp": 6,
            "tn": 594,
        })
    summary = summarize_outcomes(rows, draws=1_000, rng_seed=12_400)
    if summary["decision"] != "ADAPTIVE_EVASION_SUPPORTED":
        raise AssertionError("P6-D v4 synthetic decision replay failed")
    return {
        "status": "PASS",
        "protocol_id": protocol["protocol_id"],
        "seeds": list(CONFIRMATION_SEEDS),
        "synthetic_decision": summary["decision"],
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    sub = output.add_subparsers(dest="command", required=True)
    lock = sub.add_parser("lock")
    lock.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run = sub.add_parser("run")
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--dataset-root", type=Path, default=Path("/private/tmp/blood-ca-data"))
    run.add_argument("--acknowledge-formal-seed-consumption")
    sub.add_parser("self-check")
    return output


def main() -> None:
    args = parser().parse_args()
    if args.command == "self-check":
        result = self_check()
    elif args.command == "lock":
        result = issue_extension_lock(args.output.resolve())
    else:
        args.output = args.output.resolve()
        args.dataset_root = args.dataset_root.resolve()
        result = run_confirmation_extension(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
