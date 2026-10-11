#!/usr/bin/env python3
"""Replay structural and metric checks for the unified poison-detection study."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


PAPER14_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PAPER14_ROOT / "results"
PRIMARY = RESULTS / "unified_oneclass_poison_screening_v1"
QML_DIAGNOSTIC = RESULTS / "unified_qml_checkpoint_diagnostic_v1"
BLOOD = RESULTS / "bloodmnist_unified_oneclass_v1"
MNIST_FAMILY = RESULTS / "mnist_trigger_family_drift_v1"
MNIST_COHERENT = RESULTS / "mnist_group_coherent_drift_v1"
BLOOD_COHERENT = RESULTS / "bloodmnist_group_coherent_input_v1"
MNIST_COHERENT_10 = RESULTS / "mnist_group_coherent_drift_10seeds_v1"
BLOOD_COHERENT_10 = RESULTS / "bloodmnist_group_coherent_10seeds_v1"
OUTPUT = RESULTS / "unified_poison_detection_v1"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(left - right) <= tolerance


def verify_metric_replay(root: Path) -> dict:
    metrics = read_csv(root / "per_seed_metrics.csv")
    samples = read_csv(root / "per_sample_scores.csv")
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in samples:
        grouped[(row["seed"], row["representation"], row["group"])].append(row)
    for row in metrics:
        key = (row["seed"], row["representation"])
        clean = grouped[(*key, "clean_target")]
        poison = grouped[(*key, "poison_source")]
        calibration = grouped[(*key, "calibration")]
        fp = sum(int(value["flagged"]) for value in clean)
        tp = sum(int(value["flagged"]) for value in poison)
        cal_fp = sum(int(value["flagged"]) for value in calibration)
        assert fp == int(row["fp"])
        assert tp == int(row["tp"])
        assert len(clean) - fp == int(row["tn"])
        assert len(poison) - tp == int(row["fn"])
        assert close(fp / len(clean), float(row["fpr"]))
        assert close(tp / len(poison), float(row["tpr"]))
        assert close(cal_fp / len(calibration), float(row["calibration_fpr"]))

    summaries = read_csv(root / "pooled_summary.csv")
    by_representation: dict[str, list[dict]] = defaultdict(list)
    for row in metrics:
        by_representation[row["representation"]].append(row)
    for summary in summaries:
        selected = by_representation[summary["representation"]]
        tp = sum(int(row["tp"]) for row in selected)
        fn = sum(int(row["fn"]) for row in selected)
        fp = sum(int(row["fp"]) for row in selected)
        tn = sum(int(row["tn"]) for row in selected)
        assert (tp, fn, fp, tn) == tuple(
            int(summary[key]) for key in ("tp", "fn", "fp", "tn")
        )
        assert close(tp / (tp + fn), float(summary["pooled_tpr"]))
        assert close(fp / (fp + tn), float(summary["pooled_fpr"]))
    return {
        "per_seed_rows": len(metrics),
        "sample_rows": len(samples),
        "summary_rows": len(summaries),
    }


def verify_primary() -> dict:
    manifest = json.loads((PRIMARY / "manifest.json").read_text(encoding="utf-8"))
    assert "no attack" in manifest["detector_training_labels"]
    decision = json.loads((PRIMARY / "stage1_decision.json").read_text(encoding="utf-8"))
    assert decision["victim_gate_passed"] is True
    assert decision["primary_detector_gate_passed"] is False
    assert decision["passed"] is False
    receipts = json.loads((PRIMARY / "split_receipts.json").read_text(encoding="utf-8"))
    for receipt in receipts:
        target_groups = [
            set(receipt[key])
            for key in (
                "target_classifier_train",
                "target_reference",
                "target_calibration",
                "target_test",
            )
        ]
        for index, group in enumerate(target_groups):
            for other in target_groups[index + 1 :]:
                assert group.isdisjoint(other)
        poison_groups = [
            set(receipt[key])
            for key in ("poison_train", "poison_validation", "poison_test")
        ]
        for index, group in enumerate(poison_groups):
            for other in poison_groups[index + 1 :]:
                assert group.isdisjoint(other)
    return {**verify_metric_replay(PRIMARY), "split_receipts": len(receipts)}


def verify_blood() -> dict:
    manifest = json.loads((BLOOD / "manifest.json").read_text(encoding="utf-8"))
    assert "no triggered" in manifest["detector_training_labels"]
    victims = read_csv(BLOOD / "victim_metrics.csv")
    assert len(victims) == 4
    for row in victims:
        assert float(row["classical_balanced_ca"]) >= 0.90
        assert float(row["classical_asr"]) >= 0.80
        assert float(row["qml_balanced_ca"]) >= 0.90
        assert float(row["qml_asr"]) >= 0.80
        assert float(row["trigger_reconstruction_max_abs_error"]) <= 0.005

    summary = {
        row["representation"]: row for row in read_csv(BLOOD / "pooled_summary.csv")
    }
    required = {
        "classical_margin_drift1": (0.95, 0.10),
        "qml_z_drift8": (0.90, 0.10),
        "qml_xz_drift16": (0.95, 0.10),
    }
    for name, (minimum_tpr, maximum_fpr) in required.items():
        assert float(summary[name]["pooled_tpr"]) >= minimum_tpr
        assert float(summary[name]["pooled_fpr"]) <= maximum_fpr
    assert float(summary["classical_margin_drift1"]["mean_trigger_target_flag_rate"]) >= 0.95
    assert float(summary["classical_margin_drift1"]["mean_clean_source_flag_rate"]) <= 0.15
    assert float(summary["qml_xz_drift16"]["mean_trigger_target_flag_rate"]) >= 0.95
    assert float(summary["qml_xz_drift16"]["mean_clean_source_flag_rate"]) <= 0.20
    return {**verify_metric_replay(BLOOD), "victim_rows": len(victims)}


def verify_summary_from_metric_rows(root: Path, metric_filename: str) -> dict:
    """Replay pooled confusion counts from already-replayed config/seed rows."""
    metrics = read_csv(root / metric_filename)
    summaries = read_csv(root / "pooled_summary.csv")
    by_representation: dict[str, list[dict]] = defaultdict(list)
    for row in metrics:
        by_representation[row["representation"]].append(row)
    for summary in summaries:
        selected = by_representation[summary["representation"]]
        tp = sum(int(row["tp"]) for row in selected)
        fn = sum(int(row["fn"]) for row in selected)
        fp = sum(int(row["fp"]) for row in selected)
        tn = sum(int(row["tn"]) for row in selected)
        assert len(selected) == int(summary["n_configs"])
        assert (tp, fn, fp, tn) == tuple(
            int(summary[key]) for key in ("tp", "fn", "fp", "tn")
        )
        assert close(tp / (tp + fn), float(summary["pooled_tpr"]))
        assert close(fp / (fp + tn), float(summary["pooled_fpr"]))
    return {"metric_rows": len(metrics), "summary_rows": len(summaries)}


def verify_coherent_replay(root: Path, metric_filename: str) -> dict:
    """Replay each coherent detector row directly from sample-level flags."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["poison_labels_used_for_fit"] is False
    assert manifest["trusted_clean_reference_required"] is True
    metrics = read_csv(root / metric_filename)
    samples = read_csv(root / "per_sample_scores.csv")
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in samples:
        grouped[(row["seed"], row["trigger"], row["representation"])].append(row)
    for row in metrics:
        selected = grouped[(row["seed"], row["trigger"], row["representation"])]
        clean = [value for value in selected if int(value["label"]) == 0]
        poison = [value for value in selected if int(value["label"]) == 1]
        fp = sum(int(value["flagged"]) for value in clean)
        tp = sum(int(value["flagged"]) for value in poison)
        assert len(clean) == int(row["n_clean"])
        assert len(poison) == int(row["n_poison"])
        assert (tp, len(poison) - tp, fp, len(clean) - fp) == tuple(
            int(row[key]) for key in ("tp", "fn", "fp", "tn")
        )
        assert close(tp / len(poison), float(row["tpr"]))
        assert close(fp / len(clean), float(row["fpr"]))
    return {
        **verify_summary_from_metric_rows(root, metric_filename),
        "sample_rows": len(samples),
    }


def verify_mnist_family() -> dict:
    manifest = json.loads((MNIST_FAMILY / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["eligible_config_count"] == 27
    assert "clean target" in manifest["detector_fit"]
    summary = {
        row["representation"]: row
        for row in read_csv(MNIST_FAMILY / "pooled_summary.csv")
    }
    # Individual low-dimensional drift is useful but not universal over triggers.
    assert close(float(summary["drift_margin1"]["pooled_tpr"]), 3407 / 4860)
    assert float(summary["drift_margin1"]["config_pass_rate"]) < 0.50
    return verify_summary_from_metric_rows(MNIST_FAMILY, "per_config_metrics.csv")


def verify_mnist_coherent() -> dict:
    replay = verify_coherent_replay(MNIST_COHERENT, "per_config_metrics.csv")
    manifest = json.loads((MNIST_COHERENT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["eligible_configs"] == 27
    assert manifest["deployment_combination"] == "coherent_bonferroni_input_z_zz"
    summary = {
        row["representation"]: row
        for row in read_csv(MNIST_COHERENT / "pooled_summary.csv")
    }
    deployed = summary[manifest["deployment_combination"]]
    assert tuple(int(deployed[key]) for key in ("tp", "fn", "fp", "tn")) == (
        4857,
        3,
        427,
        10913,
    )
    assert float(deployed["pooled_tpr"]) >= 0.99
    assert float(deployed["pooled_fpr"]) <= 0.05
    assert close(float(deployed["config_pass_rate"]), 1.0)
    return replay


def verify_blood_coherent() -> dict:
    replay = verify_coherent_replay(BLOOD_COHERENT, "per_seed_metrics.csv")
    summary = {
        row["representation"]: row
        for row in read_csv(BLOOD_COHERENT / "pooled_summary.csv")
    }
    required = {
        "coherent_bonferroni_input_classical_margin": (240, 0, 94, 2306, 0.99),
        "coherent_bonferroni_input_qml_z": (233, 7, 99, 2301, 0.95),
    }
    for name, (tp, fn, fp, tn, minimum_tpr) in required.items():
        row = summary[name]
        assert tuple(int(row[key]) for key in ("tp", "fn", "fp", "tn")) == (
            tp,
            fn,
            fp,
            tn,
        )
        assert float(row["pooled_tpr"]) >= minimum_tpr
        assert float(row["pooled_fpr"]) <= 0.05
        assert close(float(row["config_pass_rate"]), 1.0)
    return replay


def verify_mnist_coherent_10seeds() -> dict:
    replay = verify_coherent_replay(MNIST_COHERENT_10, "per_config_metrics.csv")
    manifest = json.loads(
        (MNIST_COHERENT_10 / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["configuration_scope"] == "all_9_triggers_x_10_seeds"
    assert manifest["configuration_count"] == 90
    assert manifest["eligible_configs"] == 27
    assert manifest["seeds"] == list(range(42, 52))
    assert manifest["deployment_combination"] == "coherent_bonferroni_input_z_zz"

    summary = {
        row["representation"]: row
        for row in read_csv(MNIST_COHERENT_10 / "pooled_summary.csv")
    }
    deployed = summary[manifest["deployment_combination"]]
    assert tuple(int(deployed[key]) for key in ("tp", "fn", "fp", "tn")) == (
        15473,
        727,
        1365,
        36435,
    )
    assert float(deployed["pooled_tpr"]) >= 0.90
    assert float(deployed["pooled_fpr"]) <= 0.05
    assert close(float(deployed["config_pass_rate"]), 82 / 90)

    metrics = [
        row
        for row in read_csv(MNIST_COHERENT_10 / "per_config_metrics.csv")
        if row["representation"] == manifest["deployment_combination"]
    ]
    eligible = [
        row
        for row in metrics
        if float(row["archive_ca"]) >= 0.90 and float(row["archive_asr"]) >= 0.80
    ]
    assert len(eligible) == 27
    assert all(int(row["config_pass"]) == 1 for row in eligible)
    assert (
        sum(int(row["tp"]) for row in eligible),
        sum(int(row["fn"]) for row in eligible),
        sum(int(row["fp"]) for row in eligible),
        sum(int(row["tn"]) for row in eligible),
    ) == (4857, 3, 427, 10913)
    seed_stats = {
        row["representation"]: row
        for row in read_csv(MNIST_COHERENT_10 / "seed_bootstrap_summary.csv")
    }
    assert int(seed_stats[manifest["deployment_combination"]]["n_seeds"]) == 10
    return {**replay, "eligible_configs": len(eligible), "all_configs": len(metrics)}


def verify_blood_coherent_10seeds() -> dict:
    replay = verify_coherent_replay(BLOOD_COHERENT_10, "per_seed_metrics.csv")
    manifest = json.loads(
        (BLOOD_COHERENT_10 / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["seeds"] == list(range(42, 52))
    assert manifest["pre_registered_deployment_rules"] == [
        "coherent_bonferroni_input_classical_margin",
        "coherent_bonferroni_input_qml_z",
    ]
    assert manifest["post_hoc_replacement_candidate"] == "coherent_qml_xz_drift16"
    summary = {
        row["representation"]: row
        for row in read_csv(BLOOD_COHERENT_10 / "pooled_summary.csv")
    }
    exact = {
        "coherent_bonferroni_input_classical_margin": (600, 0, 256, 5744),
        "coherent_bonferroni_input_qml_z": (476, 124, 250, 5750),
        "coherent_qml_xz_drift16": (564, 36, 254, 5746),
    }
    for name, expected in exact.items():
        assert tuple(int(summary[name][key]) for key in ("tp", "fn", "fp", "tn")) == expected
        assert float(summary[name]["pooled_fpr"]) <= 0.05
    assert float(summary["coherent_bonferroni_input_classical_margin"]["pooled_tpr"]) == 1.0
    assert close(
        float(summary["coherent_bonferroni_input_classical_margin"]["config_pass_rate"]),
        1.0,
    )
    # The ten-seed extension must preserve the negative result rather than hide it.
    assert float(summary["coherent_bonferroni_input_qml_z"]["pooled_tpr"]) < 0.90
    assert close(
        float(summary["coherent_bonferroni_input_qml_z"]["config_pass_rate"]),
        0.40,
    )
    assert float(summary["coherent_qml_xz_drift16"]["pooled_tpr"]) >= 0.90
    assert close(float(summary["coherent_qml_xz_drift16"]["config_pass_rate"]), 1.0)

    victims = read_csv(BLOOD_COHERENT_10 / "victim_metrics.csv")
    assert len(victims) == 10
    assert all(float(row["classical_balanced_ca"]) >= 0.90 for row in victims)
    assert all(float(row["qml_balanced_ca"]) >= 0.90 for row in victims)
    classical_eligible = [
        row for row in victims if float(row["classical_asr"]) >= 0.80
    ]
    qml_eligible = [row for row in victims if float(row["qml_asr"]) >= 0.80]
    assert len(classical_eligible) == 7
    assert len(qml_eligible) == 9
    eligible = {
        row["representation"]: row
        for row in read_csv(BLOOD_COHERENT_10 / "victim_eligible_summary.csv")
    }
    assert close(
        float(eligible["coherent_bonferroni_input_qml_z"]["pooled_tpr"]),
        434 / 540,
    )
    assert close(
        float(eligible["coherent_qml_xz_drift16"]["pooled_tpr"]),
        509 / 540,
    )
    return {
        **replay,
        "victim_rows": len(victims),
        "classical_attack_eligible_seeds": len(classical_eligible),
        "qml_attack_eligible_seeds": len(qml_eligible),
    }


def main() -> None:
    required = [
        PRIMARY / "EXPERIMENT_REPORT.md",
        QML_DIAGNOSTIC / "EXPERIMENT_REPORT.md",
        BLOOD / "EXPERIMENT_REPORT.md",
        MNIST_FAMILY / "EXPERIMENT_REPORT.md",
        MNIST_COHERENT / "EXPERIMENT_REPORT.md",
        BLOOD_COHERENT / "EXPERIMENT_REPORT.md",
        MNIST_COHERENT_10 / "EXPERIMENT_REPORT.md",
        BLOOD_COHERENT_10 / "EXPERIMENT_REPORT.md",
    ]
    for path in required:
        assert path.exists() and path.stat().st_size > 0
    receipt = {
        "status": "PASS",
        "primary_mnist": verify_primary(),
        "qml_checkpoint_diagnostic": verify_metric_replay(QML_DIAGNOSTIC),
        "bloodmnist_confirmation": verify_blood(),
        "mnist_trigger_family_drift": verify_mnist_family(),
        "mnist_crossfit_coherent": verify_mnist_coherent(),
        "bloodmnist_crossfit_coherent": verify_blood_coherent(),
        "mnist_crossfit_coherent_10seeds": verify_mnist_coherent_10seeds(),
        "bloodmnist_crossfit_coherent_10seeds": verify_blood_coherent_10seeds(),
        "verified_claims": [
            "MNIST static clean-only Stage 1 remains a recorded failure on the QML Z5 track",
            "all tabulated counts and pooled TPR/FPR replay from per-sample flags",
            "the original four-seed BloodMNIST classical and QML victims pass CA/ASR eligibility gates",
            "low-dimensional checkpoint drift passes the prespecified detection gates",
            "drift control rates support a trigger-response rather than source-class-only interpretation",
            "individual checkpoint drift alone is not universal over 27 eligible MNIST configurations",
            "cross-fitted input plus Z/ZZ drift passes all 27 eligible MNIST configurations",
            "fixed Bonferroni input plus classical-margin/QML-Z rules pass all four BloodMNIST seeds",
            "over all 90 archived MNIST trigger-seed configurations, the fixed rule passes the pooled gate but t7_vs_t9 remains a clear weakness",
            "the ten-seed BloodMNIST classical rule passes, while the pre-registered QML-Z rule fails the pooled TPR gate",
            "QML XZ16 is a promising post-hoc replacement candidate, not a confirmed deployment rule",
            "coherent directions exclude each scored sample and never use poison labels for fitting",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "verification_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
