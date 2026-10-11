#!/usr/bin/env python3
"""Frozen simulator-to-noise bridge for the archived 5-qubit MNIST victims."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pennylane as qml
import sklearn
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase_a_full_observability as phase


PAPER14_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PAPER14_ROOT.parent
DEFAULT_CONFIG = PAPER14_ROOT / "configs/p6c_qml_noise_bridge_v1.json"
DEFAULT_OUTPUT = PAPER14_ROOT / "results/p6c_qml_noise_bridge_v1"
CONDITIONS = ("exact", "finite_shot_512", "nominal_noise_512", "light_noise_512", "heavy_noise_512")


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_paths(config: dict, seed: int) -> dict[str, Path]:
    base = (PROJECT_ROOT / config["data_root"] / f"seed_{seed}" /
            f"layer_{config['n_layers']}_grid" / config["pair"] /
            f"pr_{float(config['poison_ratio']):.1f}")
    return {
        "meta": base / f"poisoned_samples_meta_seed_{seed}_pr_{float(config['poison_ratio']):.1f}.pt",
        "backdoor": base / "models" / f"backdoor_model_seed_{seed}_pr_{float(config['poison_ratio']):.1f}.pt",
        "clean": base / "models" / f"clean_model_seed_{seed}_epoch_200.pt",
        "eligibility": base / "enhanced_scb_comprehensive_results_fixed.json",
    }


def load_cache(seed: int, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = PAPER14_ROOT / "results/phase_a_mnist5q_real_v1/cache" / f"seed{seed}_{kind}_output_pauli.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Required Phase A cache missing: {path}")
    data = np.load(path, allow_pickle=False)
    return data["clean_full"], data["attack_full"], data["pair_ids"]


def feature_indices() -> np.ndarray:
    actions = phase.build_pauli_actions(5)
    idx = phase.local_feature_indices(actions.labels)["XYZ"]
    if len(idx) != 15:
        raise AssertionError("XYZ feature index must have 15 entries")
    return idx


def target_threshold(y: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    negative = np.sort(np.asarray(scores)[np.asarray(y) == 0])
    if not len(negative):
        raise ValueError("No clean validation scores available for threshold calibration")
    # Largest number of observed clean validation points allowed by the target.
    allowed = int(math.floor(target_fpr * len(negative) + 1e-12))
    return float(np.nextafter(negative[len(negative) - allowed - 1], np.inf))


def shot_seed(seed: int, condition: str, axis: str, config_seed: int) -> int:
    token = f"{config_seed}|{seed}|{condition}|{axis}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "little")


def build_sample_qnode(weights: np.ndarray, basis: str, shots: int, p1: float = 0.0, p2: float = 0.0, device_seed: int = 0):
    noisy = (p1 > 0.0 or p2 > 0.0)
    if noisy:
        # PennyLane 0.37 default.mixed samples from NumPy's global RNG and has
        # no device seed option; measure_xyz resets it per sample below.
        np.random.seed(device_seed)
        device = qml.device("default.mixed", wires=5, shots=shots)
    else:
        device = qml.device("default.qubit", wires=5, shots=shots, seed=device_seed)
    weights = np.asarray(weights, dtype=float).reshape(12, 5, 2)

    @qml.qnode(device, interface="numpy")
    def circuit(sample):
        vector = np.asarray(sample, dtype=float).reshape(-1)
        vector = vector[:32] if vector.size >= 32 else np.pad(vector, (0, 32 - vector.size))
        vector = vector / (np.linalg.norm(vector) + 1e-12)
        qml.AmplitudeEmbedding(vector, wires=range(5), normalize=False)
        for layer in range(12):
            for qubit in range(5):
                qml.RY(weights[layer, qubit, 0], wires=qubit)
                if p1:
                    qml.DepolarizingChannel(p1, wires=qubit)
                qml.RZ(weights[layer, qubit, 1], wires=qubit)
                if p1:
                    qml.DepolarizingChannel(p1, wires=qubit)
            for qubit in range(5):
                other = (qubit + 1) % 5
                qml.CNOT(wires=[qubit, other])
                if p2:
                    qml.DepolarizingChannel(p2, wires=qubit)
                    qml.DepolarizingChannel(p2, wires=other)
        if basis == "X":
            for qubit in range(5):
                qml.Hadamard(wires=qubit)
        elif basis == "Y":
            for qubit in range(5):
                qml.adjoint(qml.S)(wires=qubit)
                qml.Hadamard(wires=qubit)
        elif basis != "Z":
            raise ValueError(f"unknown measurement basis {basis}")
        return tuple(qml.sample(qml.PauliZ(qubit)) for qubit in range(5))

    return circuit


def measure_xyz(samples: np.ndarray, weights: np.ndarray, condition: str, seed: int, config: dict, shots: int = 512) -> np.ndarray:
    if condition.startswith("finite_shot_"):
        p1 = p2 = 0.0
    else:
        noise_name = condition.rsplit("_", 1)[0].removesuffix("_noise")
        noise_key = {"nominal": "nominal", "light": "light_sensitivity", "heavy": "heavy_sensitivity"}[noise_name]
        noise = config["noise_conditions"][noise_key]
        p1, p2 = float(noise["p1"]), float(noise["p2"])
    result = np.empty((len(samples), 15), dtype=np.float64)
    for axis_i, axis in enumerate(("X", "Y", "Z")):
        circuit = build_sample_qnode(weights, axis, shots, p1, p2,
                                     shot_seed(seed, condition, axis, int(config["random_seed"])))
        measured = []
        for sample_i, sample in enumerate(samples):
            if p1 or p2:
                # Distinct reproducible shot stream per sample and basis,
                # shared across the two victim controls for paired noise.
                np.random.seed((shot_seed(seed, condition, axis, int(config["random_seed"])) + sample_i) % (2**32 - 1))
            raw = circuit(sample)
            raw = np.stack([np.asarray(v, dtype=float).reshape(-1) for v in raw], axis=1)
            measured.append(raw.mean(axis=0))
        values = np.asarray(measured)
        # Phase A's XYZ cache order is q4, q3, ..., q0, with X/Y/Z
        # within each wire. Preserve that order for the frozen classifier.
        for wire in range(5):
            result[:, 3 * (4 - wire) + axis_i] = values[:, wire]
    return result


def row_data(features: np.ndarray, pair_ids: np.ndarray, split_map: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.concatenate([features[0], features[1]], axis=0)
    pairs = np.concatenate([pair_ids, pair_ids])
    y = np.concatenate([np.zeros(len(pair_ids), dtype=int), np.ones(len(pair_ids), dtype=int)])
    split = np.asarray([split_map[str(pid)] for pid in pairs])
    return x, y, pairs, split


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = score >= threshold
    return {
        "auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "threshold": float(threshold),
        "tpr": float(np.mean(pred[y == 1])),
        "fpr": float(np.mean(pred[y == 0])),
    }


def pair_bootstrap_delta(seed_rows: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]], reps: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Hierarchical paired bootstrap: resample seed blocks, then matched pairs."""
    seeds = sorted(seed_rows)
    deltas = []
    for _ in range(reps):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_deltas = []
        for s in sampled_seeds:
            y, a, b = seed_rows[int(s)]
            # y is supplied as a two-column [label, pair-index] array by caller.
            labels = y[:, 0].astype(int)
            ids = y[:, 1]
            group_ids = np.unique(ids)
            chosen = rng.choice(group_ids, size=len(group_ids), replace=True)
            row_ix = np.concatenate([np.flatnonzero(ids == pid) for pid in chosen])
            seed_deltas.append(roc_auc_score(labels[row_ix], b[row_ix]) - roc_auc_score(labels[row_ix], a[row_ix]))
        deltas.append(float(np.mean(seed_deltas)))
    point = float(np.mean([roc_auc_score(v[0][:, 0].astype(int), v[2]) - roc_auc_score(v[0][:, 0].astype(int), v[1]) for v in seed_rows.values()]))
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return point, float(lo), float(hi)


def validate_sources(config: dict) -> dict[int, dict[str, Path]]:
    paths = {}
    for seed in config["seeds"]:
        row = source_paths(config, int(seed))
        for kind in ("backdoor", "clean"):
            row[f"cache_{kind}"] = PAPER14_ROOT / "results/phase_a_mnist5q_real_v1/cache" / f"seed{seed}_{kind}_output_pauli.npz"
        missing = [str(path) for path in row.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Required source files missing:\n" + "\n".join(missing))
        eligibility = phase.eligibility_record(config, int(seed))
        if eligibility["clean_accuracy_percent"] < 100.0 * float(config["gates"]["victim_ca_min"]):
            raise RuntimeError(f"Frozen victim seed={seed} fails CA gate: {eligibility['clean_accuracy_percent']}")
        if eligibility["attack_success_rate_percent"] < 100.0 * float(config["gates"]["victim_asr_min"]):
            raise RuntimeError(f"Frozen victim seed={seed} fails ASR gate: {eligibility['attack_success_rate_percent']}")
        paths[int(seed)] = row
    return paths


def run(config_path: Path, output: Path, *, dry_run: bool = False, pilot_seed42_nominal: bool = False, diagnostic_2048: bool = False, resume: bool = True) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_map = validate_sources(config)
    if dry_run:
        print(json.dumps({"experiment_id": config["experiment_id"], "seeds": config["seeds"], "conditions": CONDITIONS, "source_files": {str(s): {k: str(v) for k, v in row.items()} for s, row in source_map.items()}, "dry_run": True}, indent=2))
        return

    if pilot_seed42_nominal and diagnostic_2048:
        raise ValueError("pilot and 2048 diagnostic modes are mutually exclusive")
    primary_summary_path = DEFAULT_OUTPUT / "paired_summary.json"
    if diagnostic_2048:
        if not primary_summary_path.is_file():
            raise RuntimeError("2048-shot diagnostic requires a completed primary 512-shot summary")
        primary_summary = json.loads(primary_summary_path.read_text(encoding="utf-8"))
        performance_gates = [value for name, value in primary_summary.get("gate", {}).items() if name.startswith("nominal_")]
        if not performance_gates or all(performance_gates):
            raise RuntimeError("2048-shot diagnostic is allowed only after a nominal 512-shot performance gate fails")
        if output == DEFAULT_OUTPUT:
            output = output / "diagnostic_2048"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if resume and manifest_path.is_file() else {}
    config_hash = sha256(config_path)
    if manifest and manifest.get("config_sha256") != config_hash:
        raise RuntimeError("Resume refused: config hash differs from existing manifest")
    started = time.time()
    run_seeds = [42] if pilot_seed42_nominal else [int(s) for s in config["seeds"]]
    run_conditions = ("nominal_noise_512",) if pilot_seed42_nominal else (("nominal_noise_2048",) if diagnostic_2048 else CONDITIONS)
    manifest.update({
        "experiment_id": config["experiment_id"], "config_path": str(config_path.resolve()),
        "config_sha256": config_hash, "status": "running", "mode": "pilot_seed42_nominal" if pilot_seed42_nominal else ("diagnostic_2048" if diagnostic_2048 else "full"),
        "seeds": run_seeds, "conditions": list(run_conditions), "started_unix": time.time(),
        "python": sys.version, "platform": platform.platform(),
        "versions": {"numpy": np.__version__, "torch": torch.__version__, "pennylane": qml.__version__, "sklearn": sklearn.__version__},
        "source_files": {str(s): {k: {"path": str(p.relative_to(PROJECT_ROOT)), "sha256": sha256(p)} for k,p in paths.items()} for s,paths in source_map.items()},
    })
    dump_json(manifest_path, manifest)
    idx = feature_indices()
    score_path = output / "per_sample_scores.csv"
    metrics_path = output / "per_seed_metrics.csv"
    score_rows = []
    metric_rows = []
    done_keys = set()
    if resume and score_path.is_file() and metrics_path.is_file():
        with score_path.open(newline="", encoding="utf-8") as f:
            score_rows = list(csv.DictReader(f))
        with metrics_path.open(newline="", encoding="utf-8") as f:
            metric_rows = list(csv.DictReader(f))
        done_keys = {(int(r["seed"]), r["condition"], r["victim_kind"]) for r in metric_rows}

    paired_bootstrap_rows: dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for seed in run_seeds:
        clean, attack, pair_ids, audit = phase.load_matched_pairs(config, seed)
        split_map = phase.split_pairs(pair_ids, config, seed)
        for kind in ("backdoor", "clean"):
            clean_full, attack_full, cached_ids = load_cache(seed, kind)
            if not np.array_equal(pair_ids, cached_ids):
                raise RuntimeError(f"Pair ids differ from frozen cache for seed={seed} kind={kind}")
            exact_feat = np.stack([clean_full[:, idx], attack_full[:, idx]])
            x_exact, y, pair_rows, splits = row_data(exact_feat, pair_ids, split_map)
            train, val, test = splits == "train", splits == "validation", splits == "test"
            estimator, selected_c, val_auc = phase.fit_with_validation(
                "logistic", x_exact[train], y[train], x_exact[val], y[val],
                [float(v) for v in config["c_grid"]], int(config["random_seed"]) + seed)
            val_scores = phase.decision_scores(estimator, x_exact[val])
            threshold = target_threshold(y[val], val_scores, float(config["threshold_fpr_target"]))
            has_validation = any(int(r["seed"]) == seed and r["condition"] == "exact" and r["victim_kind"] == kind and r["split"] == "validation" for r in score_rows)
            if not has_validation:
                val_indices = np.flatnonzero(val)
                for val_i, ix in enumerate(val_indices):
                    score_rows.append({"seed": seed, "condition": "exact", "victim_kind": kind, "base_sample_id": str(pair_rows[ix]), "label": int(y[ix]), "split": "validation", "score": float(val_scores[val_i]), "threshold": threshold, "shots": "exact"})
                write_csv(score_path, score_rows)
            checkpoint = torch.load(source_map[seed][kind], map_location="cpu", weights_only=False)
            weights = checkpoint["weights"].detach().cpu().numpy()
            for condition in run_conditions:
                key = (seed, condition, kind)
                if key in done_keys:
                    continue
                if condition == "exact":
                    feature = exact_feat
                else:
                    sample_matrix = np.concatenate([clean, attack])
                    all_features = measure_xyz(sample_matrix, weights, condition, seed, config, int(condition.rsplit("_", 1)[1]))
                    feature = np.stack([all_features[:len(pair_ids)], all_features[len(pair_ids):]])
                x, y0, pairs, split0 = row_data(feature, pair_ids, split_map)
                test_scores = phase.decision_scores(estimator, x[test])
                m = metrics(y[test], test_scores, threshold)
                exact_test = phase.decision_scores(estimator, x_exact[test])
                m.update({"delta_auc_vs_exact": float(m["auc"] - roc_auc_score(y[test], exact_test)), "selected_c": selected_c, "exact_validation_auc": val_auc, "validation_threshold": threshold, "test_pairs": int(len(np.unique(pairs[test]))), "pair_reconstruction_max_abs_error": float(audit["pair_reconstruction_max_abs_error"])})
                metric_rows.append({"seed": seed, "condition": condition, "victim_kind": kind, **m})
                test_indices = np.flatnonzero(test)
                for ix, score in zip(test_indices, test_scores):
                    score_rows.append({"seed": seed, "condition": condition, "victim_kind": kind, "base_sample_id": str(pairs[ix]), "label": int(y[ix]), "split": "test", "score": float(score), "threshold": threshold, "shots": int(condition.rsplit("_", 1)[1]) if condition != "exact" else "exact"})
                write_csv(score_path, score_rows)
                write_csv(metrics_path, metric_rows)
                done_keys.add(key)
                dump_json(manifest_path, {**manifest, "progress": {"completed_seed_condition_controls": len(done_keys), "last_completed": list(map(str, key)), "elapsed_seconds": time.time() - started}})
                print(f"completed seed={seed} condition={condition} victim={kind} elapsed={time.time()-started:.1f}s", flush=True)

    if pilot_seed42_nominal:
        elapsed = time.time() - started
        manifest.update({"status": "pilot_complete", "elapsed_seconds": elapsed, "pilot_circuit_count": 2 * 360 * 3})
        dump_json(manifest_path, manifest)
        return
    if diagnostic_2048:
        elapsed = time.time() - started
        manifest.update({"status": "diagnostic_complete", "elapsed_seconds": elapsed, "diagnostic_only": True, "primary_summary_path": str(primary_summary_path.resolve()), "primary_summary_sha256": sha256(primary_summary_path)})
        dump_json(manifest_path, manifest)
        return

    # Paired AUC differences with seed-block and within-seed matched-pair resampling.
    summary = {"conditions": {}, "bootstrap_unit": "seed block, then matched base_sample_id within seed", "gate": {}}
    for condition in CONDITIONS:
        selected = [r for r in metric_rows if r["condition"] == condition and r["victim_kind"] == "backdoor"]
        if not selected:
            continue
        rows_by_seed = {int(r["seed"]): r for r in selected}
        aucs = {str(s): float(rows_by_seed[s]["auc"]) for s in rows_by_seed}
        deltas = {str(s): float(rows_by_seed[s]["delta_auc_vs_exact"]) for s in rows_by_seed}
        pooled = [r for r in score_rows if r["condition"] == condition and r["victim_kind"] == "backdoor" and r["split"] == "test"]
        tp = sum(int(float(r["label"]) == 1 and float(r["score"]) >= float(r["threshold"])) for r in pooled)
        fn = sum(int(float(r["label"]) == 1 and float(r["score"]) < float(r["threshold"])) for r in pooled)
        fp = sum(int(float(r["label"]) == 0 and float(r["score"]) >= float(r["threshold"])) for r in pooled)
        tn = sum(int(float(r["label"]) == 0 and float(r["score"]) < float(r["threshold"])) for r in pooled)
        summary["conditions"][condition] = {"per_seed_auc": aucs, "per_seed_delta_auc": deltas, "mean_auc": float(np.mean(list(aucs.values()))), "mean_delta_auc": float(np.mean(list(deltas.values()))), "pooled_tpr": tp / max(1, tp + fn), "pooled_fpr": fp / max(1, fp + tn)}
        # Paired seed-block bootstrap of condition minus exact AUC. Pair
        # resampling stays within each seed, preserving clean/trigger links.
        exact_condition_rows = [r for r in score_rows if r["condition"] == "exact" and r["victim_kind"] == "backdoor" and r["split"] == "test"]
        condition_rows = [r for r in score_rows if r["condition"] == condition and r["victim_kind"] == "backdoor" and r["split"] == "test"]
        boot_data = {}
        for seed in sorted({int(r["seed"]) for r in selected}):
            er = sorted([r for r in exact_condition_rows if int(r["seed"]) == seed], key=lambda r: (r["base_sample_id"], int(r["label"])))
            cr = sorted([r for r in condition_rows if int(r["seed"]) == seed], key=lambda r: (r["base_sample_id"], int(r["label"])))
            if [(r["base_sample_id"], int(r["label"])) for r in er] != [(r["base_sample_id"], int(r["label"])) for r in cr]:
                raise AssertionError("paired condition score rows do not align with exact rows")
            pair_index = {pid: i for i, pid in enumerate(dict.fromkeys(r["base_sample_id"] for r in er))}
            yy = np.asarray([[int(r["label"]), pair_index[r["base_sample_id"]]] for r in er], dtype=int)
            boot_data[seed] = (yy, np.asarray([float(r["score"]) for r in er]), np.asarray([float(r["score"]) for r in cr]))
        lo_rng = np.random.default_rng(int(config["random_seed"]) + 10000 + CONDITIONS.index(condition))
        delta_point, delta_low, delta_high = pair_bootstrap_delta(boot_data, int(config["bootstrap_repetitions"]), lo_rng)
        summary["conditions"][condition]["paired_delta_auc_seed_block_ci"] = {"estimate": delta_point, "low": delta_low, "high": delta_high}
    nominal = summary["conditions"].get("nominal_noise_512", {})
    summary["gate"] = {
        "victim_ca_asr_pass": True,
        "nominal_each_seed_auc_pass": bool(nominal and all(v >= config["gates"]["nominal_per_seed_auc_min"] for v in nominal["per_seed_auc"].values())),
        "nominal_mean_delta_auc_pass": bool(nominal and nominal["mean_delta_auc"] >= config["gates"]["nominal_mean_delta_auc_min"]),
        "nominal_pooled_tpr_pass": bool(nominal and nominal["pooled_tpr"] >= config["gates"]["nominal_pooled_tpr_min"]),
        "nominal_pooled_fpr_pass": bool(nominal and nominal["pooled_fpr"] <= config["gates"]["nominal_pooled_fpr_max"]),
        "clean_control_reported": all(any(r["condition"] == c and r["victim_kind"] == "clean" for r in metric_rows) for c in CONDITIONS),
    }
    dump_json(output / "paired_summary.json", summary)
    elapsed = time.time() - started
    manifest.update({"status": "complete", "elapsed_seconds": elapsed, "completed_unix": time.time()})
    dump_json(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pilot-seed42-nominal", action="store_true")
    parser.add_argument("--diagnostic-2048", action="store_true", help="Run nominal 2048-shot diagnostic only after primary 512 gate failure")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve(), dry_run=args.dry_run, pilot_seed42_nominal=args.pilot_seed42_nominal, diagnostic_2048=args.diagnostic_2048, resume=not args.no_resume)


if __name__ == "__main__":
    main()
