#!/usr/bin/env python3
"""Independent verifier for PathMNIST detector-confirmation receipts.

Verification replays detector scores from saved feature matrices, re-derives
the poison index set from the frozen algorithm, recomputes metrics, and checks
source/input/artifact hashes. It never trains a victim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

import backdoorbench_feature_adapter as adapter
import pathmnist_official_resnet18 as architecture


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "results/top_journal_program_v1"
PROTOCOL = PROGRAM / "pathmnist_official_weight_detector_confirmation_protocol_v1.json"
WEIGHT_RECEIPT = PROGRAM / "pathmnist_official_weights_acquisition_receipt.json"
DEFAULT_OUTPUT = ROOT / "results/pathmnist_official_weight_detector_confirmation_v1"
DEFAULT_DATASET = Path("/private/tmp/paper14-data/pathmnist.npz")
WEIGHT_RECEIPT = PROGRAM / "pathmnist_official_weights_acquisition_receipt.json"
TARGET = 8
SEEDS = (103, 104, 105, 106, 107)
WEIGHTS = {103: "resnet18_28_1.pth", 104: "resnet18_28_2.pth", 105: "resnet18_28_1.pth", 106: "resnet18_28_2.pth", 107: "resnet18_28_1.pth"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        values = tensor.detach().cpu().contiguous()
        h.update(name.encode("utf-8")); h.update(str(values.dtype).encode("ascii"))
        h.update(np.asarray(values.shape, dtype=np.int64).tobytes()); h.update(values.numpy().tobytes())
    return h.hexdigest()


def smoke_binding_failures(receipt: dict[str, Any], expected: dict[str, str | None]) -> list[str]:
    """Pure binding check, separately testable without opening any real inputs."""
    return [f"smoke {key} mismatch" for key, observed in expected.items() if receipt.get(key) != observed]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def matrix(values: Any) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("invalid saved feature matrix")
    return x


def rank_p(cal: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (1.0 + (cal[:, None] >= q[None, :]).sum(axis=0)) / (len(cal) + 1.0)


def replay_cf(reference: np.ndarray, calibration: np.ndarray, candidates: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    ref, cal, cand = matrix(reference), matrix(calibration), matrix(candidates)
    if ref.shape[1] != 512 or cal.shape[1] != 512 or cand.shape[1] != 512 or len(ref) != 512 or len(cal) != 512:
        raise ValueError("frozen CF-CMD-U requires 512 reference/calibration rows and 512-D features")
    target = np.flatnonzero(labels == TARGET)
    order = np.random.default_rng(20271001 + seed).permutation(target)
    halves = np.array_split(order, 2)
    center = ref.mean(axis=0)
    scores = np.full(len(labels), np.nan)
    p = np.ones(len(labels))
    flags = np.zeros(len(labels), dtype=bool)
    for scoring, fitting in ((halves[0], halves[1]), (halves[1], halves[0])):
        direction = cand[fitting].mean(axis=0) - center
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            raise ValueError("cross-fit direction collapsed")
        direction /= norm
        c = (cal - center) @ direction
        q = (cand[scoring] - center) @ direction
        pv = rank_p(c, q)
        scores[scoring], p[scoring] = q, pv
        flags[scoring] = pv <= .05
    return {"scores": scores, "p_values": p, "flags": flags}


def replay_centroid(reference: np.ndarray, calibration: np.ndarray, candidates: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    center = matrix(reference).mean(axis=0)
    cs = np.linalg.norm(matrix(calibration) - center, axis=1)
    scores = np.linalg.norm(matrix(candidates) - center, axis=1)
    p = np.ones(len(labels))
    target = labels == TARGET
    p[target] = rank_p(cs, scores[target])
    return {"scores": scores, "p_values": p, "flags": target & (p <= .05)}


def poison_indices(labels: np.ndarray, seed: int) -> np.ndarray:
    eligible = np.flatnonzero(labels != TARGET)
    return np.sort(np.random.default_rng(20260930 + seed).choice(eligible, 4500, replace=False))


def replay_victim_metrics(state: dict[str, torch.Tensor], images: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    model = architecture.PathMNISTResNet18()
    model.load_state_dict(state, strict=True)
    model.eval()
    def predict(values: np.ndarray) -> np.ndarray:
        output = []
        with torch.no_grad():
            for start in range(0, len(values), 512):
                batch = torch.from_numpy(np.ascontiguousarray(values[start:start + 512].transpose(0, 3, 1, 2))).float().div_(255.0)
                output.append(model(batch).argmax(1).numpy())
        return np.concatenate(output)
    pred = predict(images)
    bca = float(np.mean([np.mean(pred[labels == cls] == cls) for cls in np.unique(labels)]))
    triggered = images[labels != TARGET].copy()
    triggered[:, -4:, -4:, :] = 255
    asr = float(np.mean(predict(triggered) == TARGET))
    return {"balanced_accuracy": bca, "asr": asr}


def _hidden_features_mps(state: dict[str, torch.Tensor], images: np.ndarray) -> np.ndarray:
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable for hidden-feature provenance replay")
    model = architecture.PathMNISTResNet18(); model.load_state_dict(state, strict=True)
    model.to("mps"); model.eval(); chunks = []
    with torch.no_grad():
        for start in range(0, len(images), 512):
            x = torch.from_numpy(np.ascontiguousarray(images[start:start+512].transpose(0,3,1,2))).to("mps", dtype=torch.float32).div_(255.0)
            chunks.append(model.hidden(x).cpu().numpy())
    return np.concatenate(chunks)


def _artifact_path(receipt_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else receipt_path.parent / path


def _mps_predict_and_hidden(state_dict: dict[str, torch.Tensor], images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable for frozen smoke replay")
    model = architecture.PathMNISTResNet18()
    model.load_state_dict(state_dict, strict=True)
    model.to(torch.device("mps")); model.eval()
    logits, hidden = [], []
    with torch.no_grad():
        for start in range(0, len(images), 16):
            values = torch.from_numpy(np.ascontiguousarray(images[start:start+16].transpose(0,3,1,2))).to("mps", dtype=torch.float32).div_(255.0)
            h = model.hidden(values)
            logits.append(model.linear(h).cpu().numpy())
            hidden.append(h.cpu().numpy())
    return np.concatenate(logits), np.concatenate(hidden)


def verify_smoke_receipt(receipt_path: Path, dataset: Path) -> dict[str, Any]:
    """Independently verify MPS smoke hashes, exact resume, reload and metric replay."""
    failures: list[str] = []
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        protocol_path = PROTOCOL
        weights_receipt = json.loads(WEIGHT_RECEIPT.read_text(encoding="utf-8"))
        matches = [m for m in weights_receipt["extraction"]["members"] if m.get("name") == "resnet18_28_1.pth"]
        if len(matches) != 1:
            raise ValueError("nonformal initializer missing from acquisition receipt")
        weight_path = ROOT.parent / matches[0]["local_path"]
    except Exception as exc:
        return {"status": "FAIL", "failures": [f"smoke provenance missing: {exc}"]}
    if (receipt.get("formal_seed_opened") is not False
            or receipt.get("confirmation_seed_files_opened") is not False
            or int(receipt.get("nonformal_seed", -1)) in SEEDS):
        failures.append("smoke used or claims a formal seed")
    if int(receipt.get("nonformal_seed", -1)) != 20261002:
        failures.append("nonformal smoke seed differs from the dedicated frozen seed")
    if receipt.get("device") != "mps" or receipt.get("torch", "").split("+")[0] != "2.7.1":
        failures.append("smoke device/version mismatch")
    expected_hashes = {
        "dataset_sha256": sha256_file(dataset) if dataset.is_file() else None,
        "protocol_sha256": sha256_file(protocol_path),
        "runner_sha256": sha256_file(ROOT / "src/run_pathmnist_official_weight_detector_confirmation.py"),
        "core_sha256": sha256_file(ROOT / "src/pathmnist_external_validation.py"),
        "architecture_sha256": sha256_file(ROOT / "src/pathmnist_official_resnet18.py"),
        "weight_receipt_sha256": sha256_file(WEIGHT_RECEIPT),
        "weight_sha256": sha256_file(weight_path),
    }
    failures.extend(smoke_binding_failures(receipt, expected_hashes))
    if Path(receipt.get("weight_path", "")).name != "resnet18_28_1.pth":
        failures.append("smoke did not use the frozen nonformal initializer")
    try:
        with np.load(dataset, allow_pickle=False) as archive:
            train_x = archive["train_images"][:64]
            train_y = archive["train_labels"].reshape(-1)[:64].astype(np.int64)
            expected_val_x = archive["val_images"][:64]
            expected_val_y = archive["val_labels"].reshape(-1)[:64].astype(np.int64)
        actual_training_hash = hashlib.sha256(train_x.tobytes() + train_y.tobytes()).hexdigest()
        if actual_training_hash != receipt.get("training_data_sha256"):
            failures.append("smoke training image/label hash mismatch")
        if hashlib.sha256(np.arange(64, dtype=np.int64).tobytes()).hexdigest() != receipt.get("train_index_sha256"):
            failures.append("smoke fixed training index hash mismatch")
        if hashlib.sha256(np.arange(64, dtype=np.int64).tobytes()).hexdigest() != receipt.get("validation_index_sha256"):
            failures.append("smoke fixed validation index hash mismatch")
        weight_payload = torch.load(weight_path, map_location="cpu", weights_only=False)
        official = architecture.PathMNISTResNet18(); official.load_state_dict(weight_payload["net"], strict=True)
        initial_hash = _state_sha256(official.state_dict())
        if receipt.get("initial_state_sha256") != initial_hash:
            failures.append("smoke official initializer state hash mismatch")
    except Exception as exc:
        failures.append(f"smoke training/weight inputs could not be replayed: {exc}")
        initial_hash = ""
    files = receipt.get("checkpoint_files", {})
    checkpoint_hashes = receipt.get("checkpoint_sha256", {})
    checkpoint_payloads: dict[str, dict[str, Any]] = {}
    for name in ("continuous", "epoch2", "resumed"):
        path = _artifact_path(receipt_path, files.get(name, ""))
        if not path.is_file() or sha256_file(path) != checkpoint_hashes.get(name):
            failures.append(f"smoke {name} checkpoint hash mismatch")
            continue
        try:
            checkpoint_payloads[name] = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            failures.append(f"smoke {name} checkpoint unreadable: {exc}")
    if set(checkpoint_payloads) == {"continuous", "epoch2", "resumed"}:
        cp2 = checkpoint_payloads["epoch2"]
        if int(cp2.get("epoch", -1)) != 2 or len(cp2.get("history", [])) != 2:
            failures.append("smoke interruption checkpoint is not epoch 2")
        for name in ("continuous", "resumed"):
            payload = checkpoint_payloads[name]
            if int(payload.get("epoch", -1)) != 5:
                failures.append(f"smoke {name} checkpoint is not epoch 5")
            if [int(x.get("epoch", -1)) for x in payload.get("history", [])] != [1, 2, 3, 4, 5]:
                failures.append(f"smoke {name} training history is not continuous through epoch 5")
            if any(int(x.get("rows", -1)) != 64 for x in payload.get("history", [])):
                failures.append(f"smoke {name} epoch omitted fixed training rows")
        if [int(x.get("epoch", -1)) for x in cp2.get("history", [])] != [1, 2] or any(int(x.get("rows", -1)) != 64 for x in cp2.get("history", [])):
            failures.append("smoke epoch-2 history does not cover the fixed training rows")
        left, right = checkpoint_payloads["continuous"].get("state_dict", {}), checkpoint_payloads["resumed"].get("state_dict", {})
        if set(left) != set(right) or any(not torch.equal(left[k], right[k]) for k in left):
            failures.append("continuous and resumed smoke states are not bit-identical")
        if receipt.get("bit_identical_resume") is not True:
            failures.append("smoke resume equality attestation missing")
        for name, payload in checkpoint_payloads.items():
            expected_contract = {"seed": 20261002, "protocol_sha256": expected_hashes["protocol_sha256"],
                                 "dataset_sha256": actual_training_hash if "actual_training_hash" in locals() else "",
                                 "training_data_sha256": actual_training_hash if "actual_training_hash" in locals() else "",
                                 "weight_sha256": expected_hashes["weight_sha256"],
                                 "initial_state_sha256": initial_hash, "device": "mps", "epochs": 5}
            if payload.get("contract") != expected_contract:
                failures.append(f"smoke {name} checkpoint contract mismatch")
    feature_path = _artifact_path(receipt_path, receipt.get("feature_file", ""))
    if not feature_path.is_file() or sha256_file(feature_path) != receipt.get("feature_file_sha256"):
        failures.append("smoke feature artifact hash mismatch")
    elif "resumed" in checkpoint_payloads:
        try:
            with np.load(feature_path, allow_pickle=False) as f:
                images = f["validation_images"]; labels = f["validation_labels"].astype(np.int64)
                saved_features = f["features"]; saved_logits = f["logits"]
            if images.shape != (64, 28, 28, 3) or saved_features.shape != (64, 512):
                failures.append("smoke feature shape mismatch")
            logits, features = _mps_predict_and_hidden(checkpoint_payloads["resumed"]["state_dict"], images)
            if not np.allclose(features, saved_features, rtol=0, atol=1e-6):
                failures.append("smoke checkpoint reload feature replay mismatch")
            if not np.allclose(logits, saved_logits, rtol=0, atol=1e-6):
                failures.append("smoke checkpoint reload logits replay mismatch")
            accuracy = float(np.mean(logits.argmax(1) == labels))
            if not np.isclose(accuracy, float(receipt.get("validation_accuracy", np.nan)), rtol=0, atol=1e-12):
                failures.append("smoke validation metric replay mismatch")
            if not np.array_equal(images, expected_val_x) or not np.array_equal(labels, expected_val_y):
                failures.append("smoke validation rows differ from frozen fixed subset")
        except Exception as exc:
            failures.append(f"smoke feature/metric replay failed: {exc}")
    return {"status": "PASS_NONFORMAL_REAL_INPUT_MPS" if not failures else "FAIL",
            "formal_evidence": False, "failures": failures,
            "verified_nonformal_seed": receipt.get("nonformal_seed"),
            "checkpoint_epochs": {k: v.get("epoch") for k, v in checkpoint_payloads.items()}}


def verify(output: Path, dataset: Path = DEFAULT_DATASET, *, allow_nonformal: bool = False) -> dict[str, Any]:
    failures: list[str] = []
    try:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        seed_receipts = json.loads((output / "seed_receipts.json").read_text(encoding="utf-8"))
        metrics = rows(output / "detector_metrics.csv")
        sample_rows = rows(output / "scores.csv")
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "FAIL", "formal_evidence": False, "failures": [f"missing/unreadable artifact: {exc}"]}
    formal = manifest.get("formal") is True and manifest.get("status") == "COMPLETE_FORMAL"
    if not formal and not allow_nonformal:
        failures.append("artifact is not a completed formal run")
    if manifest.get("experiment") != "pathmnist_official_weight_detector_confirmation_v1":
        failures.append("experiment identifier mismatch")
    if manifest.get("protocol_sha256") != sha256_file(PROTOCOL):
        failures.append("protocol hash mismatch")
    if manifest.get("completed_seeds") != list(SEEDS):
        failures.append("completed seed list/order mismatch")
    if manifest.get("device") != "mps" or manifest.get("torch", "").split("+")[0] != "2.7.1":
        failures.append("execution device or torch version mismatch")
    if manifest.get("dtype") != "float32" or manifest.get("epochs") != 5:
        failures.append("dtype or epoch count mismatch")
    frozen = manifest.get("frozen_hashes", {})
    if not dataset.is_file() or sha256_file(dataset) != frozen.get("dataset_sha256"):
        failures.append("dataset hash mismatch")
    try:
        receipt = json.loads(WEIGHT_RECEIPT.read_text(encoding="utf-8"))
        observed_weights = {}
        for seed, filename in WEIGHTS.items():
            candidates = [m for m in receipt["extraction"]["members"] if m.get("name") == filename]
            if len(candidates) != 1:
                raise ValueError(f"missing unique weight {filename}")
            weight_path = ROOT.parent / candidates[0]["local_path"]
            observed_weights[str(seed)] = sha256_file(weight_path)
        if observed_weights != frozen.get("official_weight_sha256_by_seed"):
            failures.append("official weight hashes mismatch")
        if sha256_file(WEIGHT_RECEIPT) != frozen.get("weight_receipt_sha256"):
            failures.append("official weight receipt hash mismatch")
    except Exception as exc:
        failures.append(f"official weight provenance could not be verified: {exc}")
    expected_sources = {
        "runner_sha256": ROOT / "src/run_pathmnist_official_weight_detector_confirmation.py",
        "core_sha256": ROOT / "src/pathmnist_external_validation.py",
        "architecture_sha256": ROOT / "src/pathmnist_official_resnet18.py",
        "feature_adapter_sha256": ROOT / "src/backdoorbench_feature_adapter.py",
    }
    for key, source in expected_sources.items():
        if not source.is_file() or sha256_file(source) != frozen.get(key):
            failures.append(f"source hash mismatch: {key}")
    if protocol.get("seed_plan", {}).get("formal_confirmation_seeds") != list(SEEDS):
        failures.append("protocol seed plan mismatch")
    observed_map = {int(k): v for k, v in protocol.get("seed_plan", {}).get("official_weight_mapping", {}).items()}
    if observed_map != WEIGHTS:
        failures.append("protocol weight map mismatch")
    if len(seed_receipts) != 5 or [int(x.get("seed", -1)) for x in seed_receipts] != list(SEEDS):
        failures.append("seed receipt block order mismatch")
    try:
        with np.load(dataset, allow_pickle=False) as archive:
            train_images = archive["train_images"]
            train_labels = archive["train_labels"].reshape(-1).astype(np.int64)
            test_images = archive["test_images"]
            test_labels = archive["test_labels"].reshape(-1).astype(np.int64)
        if train_images.shape != (89996, 28, 28, 3) or test_images.shape != (7180, 28, 28, 3):
            raise ValueError("train/test split shape mismatch")
    except Exception as exc:
        failures.append(f"test split could not be loaded for metric replay: {exc}")
        train_images = np.empty((0, 28, 28, 3), dtype=np.uint8)
        train_labels = np.empty(0, dtype=np.int64)
        test_images = np.empty((0, 28, 28, 3), dtype=np.uint8)
        test_labels = np.empty(0, dtype=np.int64)
    metric_lookup = {(int(r["seed"]), r["method"]): r for r in metrics}
    sample_lookup: dict[tuple[int, str], list[dict[str, str]]] = {}
    for row in sample_rows:
        sample_lookup.setdefault((int(row["seed"]), row["method"]), []).append(row)
    required_methods = ("primary_cf_cmd_u", "paired_drift_cf_cmd_u", "centroid", "scan", "spectral")
    for receipt in seed_receipts:
        seed = int(receipt["seed"])
        features_path = Path(receipt.get("feature_file", ""))
        if not features_path.is_absolute():
            features_path = output / features_path
        if not features_path.is_file() or sha256_file(features_path) != receipt.get("feature_file_sha256"):
            failures.append(f"seed {seed}: feature artifact hash mismatch")
            continue
        with np.load(features_path, allow_pickle=False) as f:
            clean = matrix(f["clean_candidate"]); suspect = matrix(f["suspect_candidate"])
            cref, sref = matrix(f["clean_reference"]), matrix(f["suspect_reference"])
            ccal, scal = matrix(f["clean_calibration"]), matrix(f["suspect_calibration"])
            all_val, val_y = matrix(f["all_clean_validation"]), f["validation_labels"].astype(np.int64)
            assigned, original = f["assigned_labels"].astype(np.int64), f["original_labels"].astype(np.int64)
        if len(train_images) != 89996 or len(train_labels) != 89996:
            failures.append(f"seed {seed}: dataset training split is unavailable for candidate-pool verification")
            continue
        if any(x.shape != (89996, 512) for x in (clean, suspect)) or not np.array_equal(assigned, np.where(np.isin(np.arange(len(original)), poison_indices(original, seed)), TARGET, original)):
            failures.append(f"seed {seed}: feature row/assigned-label contract mismatch")
            continue
        chosen = poison_indices(original, seed)
        if hashlib.sha256(chosen.astype(np.int64).tobytes()).hexdigest() != receipt.get("poison_indices_sha256"):
            failures.append(f"seed {seed}: poison index hash mismatch")
        rebuilt_pool = train_images.copy()
        rebuilt_pool[chosen, -4:, -4:, :] = 255
        rebuilt_assigned = train_labels.copy(); rebuilt_assigned[chosen] = TARGET
        rebuilt_pool_hash = hashlib.sha256(rebuilt_pool.tobytes()).hexdigest()
        if rebuilt_pool_hash != receipt.get("candidate_image_sha256"):
            failures.append(f"seed {seed}: candidate pool image hash mismatch")
        if receipt.get("candidate_feature_source") != "same_realized_suspect_poison_pool_for_clean_and_suspect_models":
            failures.append(f"seed {seed}: candidate feature source mismatch")
        if not np.array_equal(original, train_labels) or not np.array_equal(assigned, rebuilt_assigned):
            failures.append(f"seed {seed}: candidate labels do not match rebuilt poison pool")
        truth = np.zeros(89996, dtype=bool); truth[chosen] = True
        if len(cref) != 512 or len(ccal) != 512 or len(sref) != 512 or len(scal) != 512:
            failures.append(f"seed {seed}: validation feature count mismatch")
        replay = {
            "primary_cf_cmd_u": replay_cf(sref, scal, suspect, assigned, seed),
            "paired_drift_cf_cmd_u": replay_cf(sref - cref, scal - ccal, suspect - clean, assigned, seed),
            "centroid": replay_centroid(sref, scal, suspect, assigned),
        }
        scan = adapter.scan(all_val, val_y, suspect, assigned, n_classes=9, seed=seed)
        spec = adapter.spectral_signatures(suspect, assigned, percentile=85.0)
        replay["scan"] = {"scores": np.asarray(scan.scores), "flags": np.asarray(scan.suspicious_mask, dtype=bool)}
        replay["spectral"] = {"scores": np.asarray(spec.scores), "flags": np.asarray(spec.suspicious_mask, dtype=bool)}
        for method in required_methods:
            key = (seed, method)
            if key not in metric_lookup or key not in sample_lookup:
                failures.append(f"seed {seed}: missing method artifact {method}")
                continue
            saved = sample_lookup[key]
            if len(saved) != 89996:
                failures.append(f"seed {seed}: {method} candidate row count mismatch")
                continue
            saved.sort(key=lambda r: int(r["original_index"]))
            if any(int(r["original_label"]) != int(original[i]) or int(r["assigned_label"]) != int(assigned[i]) or
                   (r["is_poison"].lower() == "true") != bool(truth[i]) for i, r in enumerate(saved)):
                failures.append(f"seed {seed}: {method} post-score truth/label join mismatch")
            got = replay[method]
            score_values = np.array([float(r["score"]) if r["score"] else np.nan for r in saved])
            p_values = np.array([float(r["p_value"]) if r["p_value"] else np.nan for r in saved])
            flags = np.array([r["flag"].lower() == "true" for r in saved])
            comparable = np.isfinite(got["scores"])
            if not np.allclose(score_values[comparable], got["scores"][comparable], rtol=0, atol=2e-6, equal_nan=True):
                failures.append(f"seed {seed}: {method} score replay mismatch")
            if "p_values" in got and not np.allclose(p_values, got["p_values"], rtol=0, atol=1e-12):
                failures.append(f"seed {seed}: {method} p-value replay mismatch")
            if not np.array_equal(flags, got["flags"]):
                failures.append(f"seed {seed}: {method} flag replay mismatch")
            tp = int(np.sum(flags & truth)); fp = int(np.sum(flags & ~truth)); nclean = int((~truth).sum())
            target_clean = (~truth) & (assigned == TARGET)
            counts = {"tp": tp, "n_poison": int(truth.sum()), "fp_all_clean": fp, "n_all_clean": nclean,
                      "fp_target_clean": int(np.sum(flags & target_clean)), "n_target_clean": int(target_clean.sum())}
            row = metric_lookup[key]
            if any(int(row[k]) != v for k, v in counts.items()):
                failures.append(f"seed {seed}: {method} confusion counts mismatch")
        for condition in ("clean", "suspect"):
            fpath = Path(receipt.get(f"{condition}_model_file", ""))
            if not fpath.is_file() or sha256_file(fpath) != receipt.get(f"{condition}_model_file_sha256"):
                failures.append(f"seed {seed}: {condition} model checkpoint hash mismatch")
            elif len(test_labels):
                try:
                    state = torch.load(fpath, map_location="cpu", weights_only=True)
                    replayed = replay_victim_metrics(state, test_images, test_labels)
                    recorded = receipt.get(f"{condition}_test", {})
                    if not np.isclose(replayed["balanced_accuracy"], float(recorded.get("balanced_accuracy", np.nan)), rtol=0, atol=1e-12):
                        failures.append(f"seed {seed}: {condition} test balanced accuracy does not replay")
                    if condition == "suspect" and not np.isclose(replayed["asr"], float(recorded.get("asr", np.nan)), rtol=0, atol=1e-12):
                        failures.append(f"seed {seed}: suspect test ASR does not replay")
                    if condition in ("clean", "suspect"):
                        candidate_replay = _hidden_features_mps(state, rebuilt_pool)
                        saved_candidate = clean if condition == "clean" else suspect
                        if candidate_replay.shape != saved_candidate.shape or not np.allclose(candidate_replay, saved_candidate, rtol=0, atol=2e-6):
                            failures.append(f"seed {seed}: {condition} features do not replay from shared poisoned pool")
                except Exception as exc:
                    failures.append(f"seed {seed}: {condition} test metric replay failed: {exc}")
        for condition in ("clean_training", "suspect_training"):
            training = receipt.get(condition, {})
            history = training.get("history", [])
            if [int(x.get("epoch", -1)) for x in history] != [1, 2, 3, 4, 5]:
                failures.append(f"seed {seed}: {condition} is not five fixed epochs")
            if any(int(x.get("rows", -1)) != 89996 for x in history):
                failures.append(f"seed {seed}: {condition} omitted training rows")
        if not isinstance(receipt.get("eligible"), bool):
            failures.append(f"seed {seed}: eligibility is missing")
    status = "PASS" if formal and not failures else ("DRY_RUN_VALID_NONFORMAL" if not formal and not failures else "FAIL")
    primary_summary = manifest.get("detector_summaries", {}).get("primary_cf_cmd_u", {})
    expected_primary_pass = bool(manifest.get("all_pairs_eligible") is True and
                                 primary_summary.get("pooled_tpr", -1) >= .90 and
                                 primary_summary.get("pooled_all_clean_fpr", 2) <= .05)
    if manifest.get("primary_point_gate_pass") is not expected_primary_pass:
        failures.append("primary efficacy gate mismatch")
        status = "FAIL"
    return {"status": status, "formal_evidence": bool(formal and not failures), "failures": failures,
            "verified_seeds": [int(x["seed"]) for x in seed_receipts], "verified_method_rows": len(metrics)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--allow-nonformal", action="store_true")
    parser.add_argument("--smoke-receipt", type=Path)
    args = parser.parse_args()
    result = (verify_smoke_receipt(args.smoke_receipt, args.dataset) if args.smoke_receipt
              else verify(args.output, args.dataset, allow_nonformal=args.allow_nonformal))
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] != "FAIL" else 1)


if __name__ == "__main__":
    main()
