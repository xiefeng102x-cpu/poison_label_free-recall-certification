#!/usr/bin/env python3
"""Export archived paper14 score tables into the certificate six-column schema.

This is a retrospective handoff only.  It does not retrain victims or alter any
detector.  Only victim seeds that satisfy the archived eligibility gate are
included.  Fold membership is reconstructed from the frozen RNG rules used by
the original detector implementations.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
CERT_ROOT = HERE.parents[1]
PAPER14 = HERE.parents[2]
OUT = CERT_ROOT / "results_extension_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def blood_outer_folds(n: int, seed: int, trusted_count: int = 200) -> np.ndarray:
    """Replay the four outer folds used by BloodMNIST CF-CMD-U."""
    rng = np.random.default_rng(20260928 + 1009 * int(seed))
    rng.permutation(trusted_count)  # reference/calibration split consumes RNG state
    partitions = np.array_split(rng.permutation(n), 4)
    result = np.empty(n, dtype=np.int64)
    for fold, indices in enumerate(partitions):
        result[indices] = fold
    return result


def cifar_halves(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(20271001 + int(seed))
    partitions = np.array_split(rng.permutation(n), 2)
    result = np.empty(n, dtype=np.int64)
    for fold, indices in enumerate(partitions):
        result[indices] = fold
    return result


def standard_rows(
    frame: pd.DataFrame,
    victim_id: str,
    folds: np.ndarray,
    sample_col: str,
    score_col: str,
    poison_col: str,
) -> pd.DataFrame:
    if len(frame) != len(folds):
        raise ValueError("fold vector length mismatch")
    return pd.DataFrame(
        {
            "victim_id": victim_id,
            "fold": folds.astype(int),
            "sample_id": frame[sample_col].astype(str).to_numpy(),
            "score": frame[score_col].astype(float).to_numpy(),
            "role": "train",
            "is_poison": frame[poison_col].astype(int).to_numpy(),
        }
    )


def validate_and_write(name: str, pieces: list[pd.DataFrame]) -> dict:
    output = pd.concat(pieces, ignore_index=True)
    keys = ["victim_id", "fold", "sample_id"]
    if output.duplicated(keys).any():
        raise RuntimeError(f"{name}: duplicate sample keys")
    if not np.isfinite(output["score"]).all():
        raise RuntimeError(f"{name}: non-finite scores")
    if set(output["role"]) != {"train"}:
        raise RuntimeError(f"{name}: unexpected role")
    if not set(output["is_poison"]).issubset({0, 1}):
        raise RuntimeError(f"{name}: invalid poison labels")
    grouped = output.groupby(["victim_id", "fold"])
    counts = grouped["is_poison"].agg(rows="size", poisons="sum")
    counts["clean"] = counts["rows"] - counts["poisons"]
    if (counts[["clean", "poisons"]] <= 0).any().any():
        raise RuntimeError(f"{name}: a group lacks clean or poison samples")
    path = OUT / f"{name}_scores.csv"
    output.to_csv(path, index=False)
    counts.reset_index().to_csv(OUT / f"{name}_counts.csv", index=False)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows": int(len(output)),
        "victims": int(output["victim_id"].nunique()),
        "folds_per_victim": sorted(output.groupby("victim_id")["fold"].nunique().unique().tolist()),
        "minimum_clean_per_fold": int(counts["clean"].min()),
        "minimum_poison_per_fold": int(counts["poisons"].min()),
    }


def export_adaptive() -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[Path]]:
    root = PAPER14 / "results/tifs_adaptive_blend_patch_v1/confirmation"
    score_path = root / "per_sample_scores.csv"
    victim_path = root / "victim_metrics.csv"
    scores = pd.read_csv(score_path, low_memory=False)
    victims = pd.read_csv(victim_path)
    cf_pieces: list[pd.DataFrame] = []
    global_pieces: list[pd.DataFrame] = []
    for (attack, seed, view, method), group in scores.groupby(
        ["attack", "seed", "view", "method"], sort=True
    ):
        victim = victims[(victims["attack"] == attack) & (victims["seed"] == seed)]
        if len(victim) != 1:
            raise RuntimeError("adaptive victim lookup failed")
        eligibility = "qml_eligible" if str(view).startswith("qml_") else "classical_eligible"
        if int(victim.iloc[0][eligibility]) != 1:
            continue
        group = group.sort_values("sample_index").reset_index(drop=True)
        vid = f"bloodmnist|{attack}|{method}|{view}|seed{int(seed)}"
        if method == "cf_cmd_u":
            folds = blood_outer_folds(len(group), int(seed))
            cf_pieces.append(standard_rows(group, vid, folds, "sample_index", "score", "label"))
        else:
            global_pieces.append(
                standard_rows(group, vid, np.zeros(len(group), dtype=int), "sample_index", "score", "label")
            )
    return cf_pieces, global_pieces, [score_path, victim_path, root / "manifest.json"]


def export_blood_cross_attack() -> tuple[list[pd.DataFrame], list[Path]]:
    root = PAPER14 / "results/bloodmnist_cross_attack_confirmation_v1"
    score_path = root / "per_sample_scores.csv"
    victim_path = root / "victim_metrics.csv"
    scores = pd.read_csv(score_path)
    victims = pd.read_csv(victim_path)
    pieces: list[pd.DataFrame] = []
    for (attack, seed, view), group in scores.groupby(
        ["trigger", "seed", "representation"], sort=True
    ):
        victim = victims[(victims["attack"] == attack) & (victims["seed"] == seed)]
        if len(victim) != 1:
            raise RuntimeError("cross-attack victim lookup failed")
        row = victim.iloc[0]
        if "qml" in str(view):
            eligible = row["qml_balanced_ca"] >= 0.90 and row["qml_asr"] >= 0.80
        else:
            eligible = row["classical_balanced_ca"] >= 0.90 and row["classical_asr"] >= 0.80
        if not eligible:
            continue
        group = group.sort_values("sample_index").reset_index(drop=True)
        vid = f"bloodmnist|{attack}|cf_cmd|{view}|seed{int(seed)}"
        folds = blood_outer_folds(len(group), int(seed))
        pieces.append(standard_rows(group, vid, folds, "sample_index", "score", "label"))
    return pieces, [score_path, victim_path, root / "manifest.json"]


def export_cifar() -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[Path]]:
    root = PAPER14 / "results/tifs_native_cifar_gtsrb_v1/cifar10"
    score_path = root / "per_sample_scores.csv"
    victim_path = root / "victim_metrics.csv"
    scores = pd.read_csv(score_path, low_memory=False)
    victims = pd.read_csv(victim_path)
    eligible = set(victims.loc[victims["eligible"] == 1, "seed"].astype(int))
    crossfit_pieces: list[pd.DataFrame] = []
    global_pieces: list[pd.DataFrame] = []
    for (seed, method), group in scores.groupby(["seed", "method"], sort=True):
        if int(seed) not in eligible:
            continue
        group = group[group["defined"] == 1].sort_values("candidate_position").reset_index(drop=True)
        vid = f"cifar10|badnets|{method}|target_assigned|seed{int(seed)}"
        if method in {"cf_cmd_u", "crossfit_spectral"}:
            folds = cifar_halves(len(group), int(seed))
            crossfit_pieces.append(
                standard_rows(group, vid, folds, "record_id", "score", "poison_truth")
            )
        elif method == "target_centroid_distance":
            global_pieces.append(
                standard_rows(group, vid, np.zeros(len(group), dtype=int), "record_id", "score", "poison_truth")
            )
    return crossfit_pieces, global_pieces, [score_path, victim_path, root / "manifest.json"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    adaptive_cf, adaptive_global, adaptive_sources = export_adaptive()
    blood_cross, blood_sources = export_blood_cross_attack()
    cifar_cross, cifar_global, cifar_sources = export_cifar()
    outputs = {
        "adaptive_cfcmdu": validate_and_write("adaptive_cfcmdu", adaptive_cf),
        "adaptive_global_baselines": validate_and_write("adaptive_global_baselines", adaptive_global),
        "bloodmnist_cross_attack": validate_and_write("bloodmnist_cross_attack", blood_cross),
        "cifar10_crossfit": validate_and_write("cifar10_crossfit", cifar_cross),
        "cifar10_global_baseline": validate_and_write("cifar10_global_baseline", cifar_global),
    }
    source_paths = sorted(set(adaptive_sources + blood_sources + cifar_sources))
    manifest = {
        "artifact": "retrospective_cross_dataset_cross_attack_certificate_handoff_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "READY_FOR_RETROSPECTIVE_CERTIFICATE_ANALYSIS",
        "outputs": outputs,
        "source_sha256": {str(path): sha256(path) for path in source_paths},
        "scope": {
            "datasets": ["BloodMNIST", "CIFAR-10"],
            "attacks": ["BadNets", "Blended", "SIG", "Adaptive-Blend", "Adaptive-Patch"],
            "eligible_victims_only": True,
            "fresh_rows_included": False,
            "retrospective_not_preregistered": True,
            "population": "detector-defined target-assigned candidate strata",
        },
        "limitations": [
            "No new victim was trained; all score maps and victims are archived paper14 artifacts.",
            "Fresh clean rows are unavailable in these archived tables, so H2/H2b are not tested.",
            "GTSRB is excluded because only one development victim exists and its detector output is empty.",
            "PathMNIST v2 has only two ineligible development victims and no per-sample score table.",
        ],
    }
    (OUT / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": manifest["status"], "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
