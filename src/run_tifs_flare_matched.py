#!/usr/bin/env python3
"""Matched-access FLARE comparison for the frozen PathMNIST victims.

The pre-result protocol lives in ``results/tifs_flare_matched_v1``.  This
runner deliberately separates (1) a hidden512 clean-room adapter matched to
the archived CF-CMD-U layer/scope and (2) an all-BN, label-blind sensitivity
following the representation actually computed by BackdoorBox's FLARE.py.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import HDBSCAN
from sklearn.cluster._hdbscan._tree import _condense_tree, labelling_at_cut

try:
    import umap
except ImportError as exc:  # pragma: no cover - dependency error is user-facing.
    raise RuntimeError(
        "umap-learn is required; see results/tifs_flare_matched_v1/REPRODUCE.md"
    ) from exc

import pathmnist_official_resnet18 as architecture
import run_pathmnist_official_weight_detector_confirmation as source


PAPER14_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PAPER14_ROOT / "results/tifs_flare_matched_v1"
PROTOCOL = OUTPUT / "protocol.json"
SOURCE_RESULT = PAPER14_ROOT / "results/pathmnist_official_weight_detector_confirmation_v1"
DATASET = Path("/private/tmp/paper14-data/pathmnist.npz")
SEEDS = (103, 104, 105, 106, 107)
TARGET_CLASS = 8
MIN_CLUSTER_SIZE = 100
XI = 0.02
MAX_DEPTH = 3
BOOTSTRAP_REPLICATES = 20_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_source_data(dataset: Path) -> dict[str, dict[str, np.ndarray]]:
    with np.load(dataset, allow_pickle=False) as payload:
        result = {
            split: {
                "images": payload[f"{split}_images"],
                "labels": payload[f"{split}_labels"].reshape(-1).astype(np.int64),
            }
            for split in ("train", "val", "test")
        }
    expected = {"train": 89996, "val": 10004, "test": 7180}
    for split, count in expected.items():
        images, labels = result[split]["images"], result[split]["labels"]
        if images.shape != (count, 28, 28, 3) or images.dtype != np.uint8:
            raise ValueError(f"unexpected PathMNIST {split} images")
        if labels.shape != (count,):
            raise ValueError(f"unexpected PathMNIST {split} labels")
    return result


def poison_truth(labels: np.ndarray, seed: int) -> np.ndarray:
    eligible = np.flatnonzero(np.asarray(labels, dtype=np.int64) != TARGET_CLASS)
    selected = np.sort(
        np.random.default_rng(20260930 + int(seed)).choice(
            eligible, size=4500, replace=False
        )
    )
    truth = np.zeros(len(labels), dtype=bool)
    truth[selected] = True
    return truth


def make_candidate_pool(
    images: np.ndarray, labels: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = poison_truth(labels, seed)
    candidate = np.asarray(images).copy()
    candidate[truth, -4:, -4:, :] = 255
    assigned = np.asarray(labels, dtype=np.int64).copy()
    assigned[truth] = TARGET_CLASS
    return candidate, assigned, truth


def load_model(seed: int, device: torch.device) -> architecture.PathMNISTResNet18:
    path = SOURCE_RESULT / "models" / f"seed_{seed}_suspect_epoch5.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = architecture.PathMNISTResNet18()
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def bn_layer_names(model: torch.nn.Module) -> list[str]:
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.BatchNorm2d)
    ]


def extract_bn_layerwise_log_likelihood(
    model: torch.nn.Module,
    images: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 512,
) -> tuple[np.ndarray, list[str]]:
    """Compute the 20-D representation implemented by upstream FLARE.py.

    The hook observes each BN input.  For each sample/layer it computes the
    minimum Gaussian log probability across all channels and spatial sites,
    exactly matching ``BatchNorm2d_ent.calculate_likelihood`` without copying
    or mutating the victim.
    """

    names = bn_layer_names(model)
    if len(names) != 20:
        raise RuntimeError(f"expected 20 BN layers, found {len(names)}")
    modules = dict(model.named_modules())
    current: dict[str, torch.Tensor] = {}
    handles = []

    def hook(name: str):
        def collect(module: torch.nn.BatchNorm2d, inputs: tuple[torch.Tensor, ...]) -> None:
            values = inputs[0]
            mean = module.running_mean.reshape(1, -1, 1, 1)
            variance = module.running_var.reshape(1, -1, 1, 1)
            std = torch.sqrt(variance + 1e-5)
            log_probability = (
                -0.5 * ((values - mean) / std).square()
                - torch.log(std)
                - 0.5 * np.log(2.0 * np.pi)
            )
            current[name] = log_probability.flatten(1).amin(dim=1)

        return collect

    for name in names:
        handles.append(modules[name].register_forward_pre_hook(hook(name)))
    pieces = []
    try:
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                current.clear()
                batch = source._tensor_batch(images[start : start + batch_size], device)
                model(batch)
                missing = [name for name in names if name not in current]
                if missing:
                    raise RuntimeError(f"BN hooks did not fire: {missing}")
                matrix = torch.stack([current[name] for name in names], dim=1)
                pieces.append(matrix.cpu().numpy().astype(np.float32, copy=False))
    finally:
        for handle in handles:
            handle.remove()
    return np.concatenate(pieces, axis=0), names


def reduce_umap(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or len(values) < MIN_CLUSTER_SIZE * 2:
        raise ValueError("FLARE requires a two-dimensional, sufficiently large cohort")
    reducer = umap.UMAP(
        n_neighbors=40,
        min_dist=0.0,
        n_components=2,
        random_state=42,
    )
    return reducer.fit_transform(values).astype(np.float32, copy=False)


def code_rule_flags(embedding: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    clusterer = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE).fit(embedding)
    labels = np.asarray(clusterer.labels_, dtype=np.int64)
    unique, counts = np.unique(labels, return_counts=True)
    largest = int(unique[np.argmax(counts)])
    flags = np.zeros(len(labels), dtype=bool)
    for label in unique:
        if int(label) == -1 or int(label) == largest:
            continue
        flags[labels == label] = True
    return flags, {
        "cluster_labels": [int(x) for x in unique],
        "cluster_counts": [int(x) for x in counts],
        "largest_cluster_label_including_noise": largest,
        "noise_count": int(np.sum(labels == -1)),
        "flagged_count": int(flags.sum()),
        "backend": "sklearn.cluster.HDBSCAN 1.6.1 compatibility path",
    }


def matched_hidden_adapter(
    hidden: np.ndarray, assigned: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    target = np.flatnonzero(np.asarray(assigned, dtype=np.int64) == TARGET_CLASS)
    embedding = reduce_umap(hidden[target])
    local_flags, receipt = code_rule_flags(embedding)
    flags = np.zeros(len(hidden), dtype=bool)
    flags[target] = local_flags
    receipt.update(
        {
            "candidate_scope": "assigned_target_class_8",
            "candidate_scope_count": int(len(target)),
            "representation": "suspect_hidden512",
            "non_target_rule": "forced_unflagged",
        }
    )
    return flags, embedding, receipt


def first_stable_split(
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any], list[dict[str, Any]]]:
    """Replay the upstream stable-subspace traversal with sklearn internals."""

    search: list[dict[str, Any]] = []
    dimensions = features.shape[1]
    for excluded in range(dimensions):
        kept = dimensions - excluded
        embedding = reduce_umap(features[:, :kept])
        clusterer = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE).fit(embedding)
        condensed = _condense_tree(clusterer._single_linkage_tree_, MIN_CLUSTER_SIZE)
        stable_child = None
        stable_difference = None
        stable_depth = None
        reason = "NO_STABLE_SPLIT"
        if len(condensed):
            largest_row = condensed[np.argmax(condensed["cluster_size"])]
            node_id = int(largest_row["child"])
            for depth in range(MAX_DEPTH):
                current_rows = condensed[condensed["child"] == node_id]
                children = condensed[condensed["parent"] == node_id]
                if not len(current_rows) or not len(children):
                    reason = "TRAVERSAL_EXHAUSTED"
                    break
                current = current_rows[0]
                child = children[np.argmax(children["cluster_size"])]
                difference = float(child["value"] - current["value"])
                if difference > XI:
                    stable_child = child
                    stable_difference = difference
                    stable_depth = depth
                    reason = "STABLE"
                    break
                node_id = int(child["child"])
        row = {
            "excluded_deepest_bn_layers": excluded,
            "kept_bn_layers": kept,
            "stable": int(stable_child is not None),
            "stability_difference": stable_difference if stable_difference is not None else "",
            "stability_depth": stable_depth if stable_depth is not None else "",
            "status": reason,
        }
        search.append(row)
        if stable_child is None:
            continue
        lambda_threshold = float(stable_child["value"]) - 0.0001
        if lambda_threshold <= 0:
            search[-1]["status"] = "NONPOSITIVE_LAMBDA_THRESHOLD"
            continue
        labels = labelling_at_cut(
            clusterer._single_linkage_tree_,
            1.0 / lambda_threshold,
            MIN_CLUSTER_SIZE,
        )
        labels = np.asarray(labels, dtype=np.int64)
        unique, counts = np.unique(labels, return_counts=True)
        largest = int(unique[np.argmax(counts)])
        flags = np.zeros(len(labels), dtype=bool)
        for label in unique:
            if int(label) == -1 or int(label) == largest:
                continue
            flags[labels == label] = True
        receipt = {
            "status": "STABLE_SUBSPACE_FOUND",
            "excluded_deepest_bn_layers": excluded,
            "kept_bn_layers": kept,
            "stability_difference": stable_difference,
            "stability_depth_zero_based": stable_depth,
            "lambda_threshold": lambda_threshold,
            "distance_cut": 1.0 / lambda_threshold,
            "cluster_labels": [int(x) for x in unique],
            "cluster_counts": [int(x) for x in counts],
            "largest_cluster_label_including_noise": largest,
            "flagged_count": int(flags.sum()),
            "backend": "sklearn HDBSCAN private condensed-tree compatibility path",
        }
        return flags, embedding, receipt, search
    return (
        np.zeros(len(features), dtype=bool),
        None,
        {
            "status": "NO_STABLE_SUBSPACE",
            "excluded_deepest_bn_layers": None,
            "kept_bn_layers": None,
            "flagged_count": 0,
        },
        search,
    )


def counts(flags: np.ndarray, truth: np.ndarray, assigned: np.ndarray) -> dict[str, Any]:
    flags = np.asarray(flags, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    assigned = np.asarray(assigned, dtype=np.int64)
    clean = ~truth
    target_clean = clean & (assigned == TARGET_CLASS)
    result = {
        "tp": int(np.sum(flags & truth)),
        "n_poison": int(truth.sum()),
        "fp_all_clean": int(np.sum(flags & clean)),
        "n_all_clean": int(clean.sum()),
        "fp_target_clean": int(np.sum(flags & target_clean)),
        "n_target_clean": int(target_clean.sum()),
    }
    result["tpr"] = result["tp"] / result["n_poison"]
    result["all_clean_fpr"] = result["fp_all_clean"] / result["n_all_clean"]
    result["target_clean_fpr"] = (
        result["fp_target_clean"] / result["n_target_clean"]
    )
    return result


def summarize(rows: list[dict[str, Any]], eligibility: dict[int, bool]) -> dict[str, Any]:
    if tuple(int(row["seed"]) for row in rows) != SEEDS:
        raise ValueError("summary requires the five frozen seeds in order")
    keys = (
        "tp",
        "n_poison",
        "fp_all_clean",
        "n_all_clean",
        "fp_target_clean",
        "n_target_clean",
    )
    pooled = {key: int(sum(int(row[key]) for row in rows)) for key in keys}
    point = {
        "pooled_tpr": pooled["tp"] / pooled["n_poison"],
        "pooled_all_clean_fpr": pooled["fp_all_clean"] / pooled["n_all_clean"],
        "pooled_target_clean_fpr": pooled["fp_target_clean"] / pooled["n_target_clean"],
    }
    rng = np.random.default_rng(20261006)
    sampled = rng.integers(0, len(rows), size=(BOOTSTRAP_REPLICATES, len(rows)))
    boot = np.empty((BOOTSTRAP_REPLICATES, 3), dtype=np.float64)
    matrix = np.asarray([[row[key] for key in keys] for row in rows], dtype=np.int64)
    for i, index in enumerate(sampled):
        total = matrix[index].sum(axis=0)
        boot[i] = (
            total[0] / total[1],
            total[2] / total[3],
            total[4] / total[5],
        )
    intervals = {
        name: [float(x) for x in np.quantile(boot[:, column], (0.025, 0.975))]
        for column, name in enumerate(
            ("tpr_95pct", "all_clean_fpr_95pct", "target_clean_fpr_95pct")
        )
    }
    eligible = all(eligibility.get(seed, False) for seed in SEEDS)
    gate = eligible and point["pooled_tpr"] >= 0.90 and point["pooled_all_clean_fpr"] <= 0.05
    return {
        "all_five_victims_eligible": eligible,
        "micro_pooled_counts": pooled,
        **point,
        **intervals,
        "point_gate_pass": gate,
        "decision": "PASS" if gate else "FAIL",
    }


def append_per_sample(
    path: Path,
    *,
    seed: int,
    method: str,
    flags: np.ndarray,
    truth: np.ndarray,
    assigned: np.ndarray,
    original: np.ndarray,
) -> None:
    exists = path.exists()
    with gzip.open(path, "at", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(
                [
                    "seed",
                    "method",
                    "original_index",
                    "original_label",
                    "assigned_label",
                    "flag",
                    "is_poison",
                ]
            )
        for index in range(len(flags)):
            writer.writerow(
                [
                    seed,
                    method,
                    index,
                    int(original[index]),
                    int(assigned[index]),
                    int(flags[index]),
                    int(truth[index]),
                ]
            )


def source_eligibility() -> dict[int, bool]:
    receipts = json.loads((SOURCE_RESULT / "seed_receipts.json").read_text(encoding="utf-8"))
    return {int(row["seed"]): bool(row["eligible"]) for row in receipts}


def run_smoke(args: argparse.Namespace) -> None:
    data = load_source_data(args.dataset)
    seed = SEEDS[0]
    candidate, assigned, truth = make_candidate_pool(
        data["train"]["images"], data["train"]["labels"], seed
    )
    subset = np.arange(min(args.smoke_rows, len(candidate)))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(seed, device)
    started = time.time()
    bn, names = extract_bn_layerwise_log_likelihood(
        model, candidate[subset], device, batch_size=args.batch_size
    )
    feature_file = SOURCE_RESULT / "features" / f"seed_{seed}.npz"
    with np.load(feature_file, allow_pickle=False) as payload:
        hidden = payload["suspect_candidate"][subset]
    # Mechanics only: UMAP/HDBSCAN uses a deterministic synthetic enlargement
    # if the subset has fewer than two hundred target-assigned rows.
    target = np.flatnonzero(assigned[subset] == TARGET_CLASS)
    if len(target) < 200:
        target = np.arange(min(len(subset), 512))
    embedding = reduce_umap(hidden[target])
    clean_flags, cluster = code_rule_flags(embedding)
    receipt = {
        "status": "PASS_SMOKE",
        "seed": seed,
        "rows": int(len(subset)),
        "device": str(device),
        "bn_shape": list(bn.shape),
        "bn_layer_names": names,
        "bn_finite": bool(np.isfinite(bn).all()),
        "hidden_subset_shape": list(hidden.shape),
        "embedding_shape": list(embedding.shape),
        "cluster": cluster,
        "smoke_flagged": int(clean_flags.sum()),
        "truth_not_used_for_detector": True,
        "elapsed_seconds": time.time() - started,
    }
    write_json(OUTPUT / "smoke_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


def run_formal(args: argparse.Namespace) -> None:
    run_matched = args.component in {"matched", "both"}
    run_official = args.component in {"official", "both"}
    component_summary = OUTPUT / f"summary_{args.component}.json"
    if component_summary.exists() and not args.overwrite:
        raise RuntimeError(
            f"formal {args.component} result already exists; pass --overwrite to replay"
        )
    smoke = json.loads((OUTPUT / "smoke_receipt.json").read_text(encoding="utf-8"))
    if smoke.get("status") != "PASS_SMOKE":
        raise RuntimeError("passing smoke receipt is required")
    protocol_sha = sha256_file(PROTOCOL)
    access_sha = sha256_file(OUTPUT / "ACCESS_MATRIX.md")
    data = load_source_data(args.dataset)
    eligibility = source_eligibility()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if str(device) != "mps" and not args.allow_cpu:
        raise RuntimeError("formal feature extraction is frozen to MPS; pass --allow-cpu only with a deviation")
    per_sample_path = OUTPUT / "per_sample_flags.csv.gz"
    if run_matched and per_sample_path.exists():
        per_sample_path.unlink()
    metrics: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    seed_receipts: list[dict[str, Any]] = []
    for seed in SEEDS:
        print(f"seed={seed}: prepare frozen candidate binding", flush=True)
        started = time.time()
        candidate, assigned, truth = make_candidate_pool(
            data["train"]["images"], data["train"]["labels"], seed
        )
        feature_file = SOURCE_RESULT / "features" / f"seed_{seed}.npz"
        with np.load(feature_file, allow_pickle=False) as payload:
            hidden = payload["suspect_candidate"].astype(np.float32, copy=False)
            saved_assigned = payload["assigned_labels"].astype(np.int64, copy=False)
            saved_original = payload["original_labels"].astype(np.int64, copy=False)
            saved_image_sha = str(payload["candidate_image_sha256"])
        if not np.array_equal(saved_assigned, assigned):
            raise RuntimeError(f"seed {seed}: assigned labels do not replay")
        if not np.array_equal(saved_original, data["train"]["labels"]):
            raise RuntimeError(f"seed {seed}: original labels do not replay")
        if source.candidate_pool_image_hash(candidate) != saved_image_sha:
            raise RuntimeError(f"seed {seed}: candidate image hash does not replay")
        matched_receipt = None
        matched_counts = None
        if run_matched:
            print(f"seed={seed}: matched hidden512 adapter", flush=True)
            matched_flags, matched_embedding, matched_receipt = matched_hidden_adapter(hidden, assigned)
            np.savez_compressed(
                OUTPUT / f"embedding_matched_seed_{seed}.npz",
                target_indices=np.flatnonzero(assigned == TARGET_CLASS),
                embedding=matched_embedding,
            )
            matched_counts = counts(matched_flags, truth, assigned)
            metrics.append(
                {
                    "seed": seed,
                    "column": "matched_clean_room",
                    "method": "flare_hidden512_matched_adapter",
                    "eligible": int(eligibility[seed]),
                    **matched_counts,
                }
            )
            append_per_sample(
                per_sample_path,
                seed=seed,
                method="flare_hidden512_matched_adapter",
                flags=matched_flags,
                truth=truth,
                assigned=assigned,
                original=data["train"]["labels"],
            )

        official_receipt = None
        official_counts = None
        bn_path = None
        names = None
        if run_official:
            print(f"seed={seed}: official-code all-BN privilege sensitivity", flush=True)
            model = load_model(seed, device)
            bn_features, names = extract_bn_layerwise_log_likelihood(
                model, candidate, device, batch_size=args.batch_size
            )
            bn_path = OUTPUT / f"bn_layerwise_seed_{seed}.npy"
            np.save(bn_path, bn_features, allow_pickle=False)
            official_flags, official_embedding, official_receipt, search = first_stable_split(
                bn_features
            )
            if official_embedding is not None:
                np.savez_compressed(
                    OUTPUT / f"embedding_official_privilege_seed_{seed}.npz",
                    embedding=official_embedding,
                )
            for row in search:
                subspace_rows.append({"seed": seed, **row})
            official_counts = counts(official_flags, truth, assigned)
            metrics.append(
                {
                    "seed": seed,
                    "column": "official_privilege_sensitivity",
                    "method": "flare_all_bn_layerwise_official_code_sensitivity",
                    "eligible": int(eligibility[seed]),
                    **official_counts,
                }
            )
            append_per_sample(
                per_sample_path,
                seed=seed,
                method="flare_all_bn_layerwise_official_code_sensitivity",
                flags=official_flags,
                truth=truth,
                assigned=assigned,
                original=data["train"]["labels"],
            )
        receipt = {
            "seed": seed,
            "eligible": eligibility[seed],
            "actual_device": str(device),
            "source_feature_file": str(feature_file),
            "source_feature_sha256": sha256_file(feature_file),
            "suspect_model_file": str(
                SOURCE_RESULT / "models" / f"seed_{seed}_suspect_epoch5.pt"
            ),
            "suspect_model_sha256": sha256_file(
                SOURCE_RESULT / "models" / f"seed_{seed}_suspect_epoch5.pt"
            ),
            "candidate_image_sha256": saved_image_sha,
            "poison_indices_sha256": hashlib.sha256(
                np.flatnonzero(truth).astype(np.int64).tobytes()
            ).hexdigest(),
            "matched_receipt": matched_receipt,
            "official_privilege_receipt": official_receipt,
            "elapsed_seconds": time.time() - started,
        }
        if bn_path is not None and names is not None:
            receipt.update(
                {
                    "bn_feature_file": str(bn_path),
                    "bn_feature_sha256": sha256_file(bn_path),
                    "bn_layer_names": names,
                }
            )
        seed_receipts.append(receipt)
        write_csv(OUTPUT / f"per_seed_metrics_{args.component}.partial.csv", metrics)
        write_csv(OUTPUT / f"subspace_search_{args.component}.partial.csv", subspace_rows)
        write_json(OUTPUT / f"seed_receipts_{args.component}.partial.json", seed_receipts)
        messages = []
        if matched_counts is not None:
            messages.append(
                f"matched={matched_counts['tpr']:.4f}/{matched_counts['all_clean_fpr']:.4f}"
            )
        if official_counts is not None:
            messages.append(
                f"official-privilege={official_counts['tpr']:.4f}/{official_counts['all_clean_fpr']:.4f}"
            )
        print(f"seed={seed}: " + "; ".join(messages), flush=True)

    summaries = {}
    for method in sorted({str(row["method"]) for row in metrics}):
        block = [row for row in metrics if row["method"] == method]
        summaries[method] = summarize(block, eligibility)
    source_manifest = json.loads((SOURCE_RESULT / "manifest.json").read_text(encoding="utf-8"))
    source_summary = source_manifest["detector_summaries"]["primary_cf_cmd_u"]
    summaries["archived_primary_cf_cmd_u"] = {
        "source": str(SOURCE_RESULT / "manifest.json"),
        "source_sha256": sha256_file(SOURCE_RESULT / "manifest.json"),
        "pooled_tpr": source_summary["pooled_tpr"],
        "pooled_all_clean_fpr": source_summary["pooled_all_clean_fpr"],
        "pooled_target_clean_fpr": source_summary["pooled_target_clean_fpr"],
        "point_gate_pass": bool(source_manifest["primary_point_gate_pass"]),
        "comparison_scope": "matched only to flare_hidden512_matched_adapter",
    }
    manifest = {
        "experiment": "tifs_flare_matched_v1",
        "status": "COMPLETE" if run_official else "MATCHED_COMPLETE_OFFICIAL_PENDING",
        "component": args.component,
        "protocol_sha256_at_execution": protocol_sha,
        "access_matrix_sha256_at_execution": access_sha,
        "upstream_flare_sha256": sha256_file(OUTPUT / "upstream_FLARE.py"),
        "source_pathmnist_manifest_sha256": sha256_file(SOURCE_RESULT / "manifest.json"),
        "dataset_path": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "seeds": list(SEEDS),
        "device": str(device),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "sklearn": __import__("sklearn").__version__,
            "umap": umap.__version__,
            "platform": platform.platform(),
        },
        "summaries": summaries,
        "separate_reporting_enforced": True,
        "paper_flare_c_reproduced": False,
        "paper_code_granularity_mismatch": True,
        "official_source_directly_executable": False,
        "official_source_blockers": [
            "hard-coded CUDA calls",
            "hdbscan HDBSCAN import overwritten by sklearn HDBSCAN while external-only constructor arguments are still passed",
            "paper-primary channel-wise representation is flattened to one value per BN layer in code",
        ],
        "command_line": sys.argv,
    }
    write_csv(OUTPUT / f"per_seed_metrics_{args.component}.csv", metrics)
    write_csv(OUTPUT / f"subspace_search_{args.component}.csv", subspace_rows)
    write_json(OUTPUT / f"seed_receipts_{args.component}.json", seed_receipts)
    write_json(component_summary, summaries)
    write_json(OUTPUT / f"manifest_{args.component}.json", manifest)
    if run_official:
        combined = {}
        matched_summary_path = OUTPUT / "summary_matched.json"
        if matched_summary_path.is_file():
            combined.update(json.loads(matched_summary_path.read_text(encoding="utf-8")))
        combined.update(summaries)
        write_json(OUTPUT / "summary.json", combined)
        manifest["status"] = "COMPLETE"
        manifest["summaries"] = combined
        write_json(OUTPUT / "manifest.json", manifest)
    for partial in (
        OUTPUT / f"per_seed_metrics_{args.component}.partial.csv",
        OUTPUT / f"subspace_search_{args.component}.partial.csv",
        OUTPUT / f"seed_receipts_{args.component}.partial.json",
    ):
        if partial.exists():
            partial.unlink()
    print(json.dumps(summaries, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--smoke-rows", type=int, default=2048)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--component", choices=("matched", "official", "both"), default="both"
    )
    args = parser.parse_args()
    if args.mode == "smoke":
        run_smoke(args)
    else:
        run_formal(args)


if __name__ == "__main__":
    main()
