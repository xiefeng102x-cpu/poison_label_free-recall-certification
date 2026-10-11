from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_bloodmnist_multiclass_backdoorbench_pilot as pilot


def synthetic_training(per_class: int = 4) -> tuple[np.ndarray, np.ndarray]:
    labels = np.repeat(np.arange(8, dtype=np.int64), per_class)
    images = np.zeros((len(labels), 28, 28, 3), dtype=np.float32)
    for index in range(len(images)):
        images[index].fill(index / (len(images) + 1))
    return images, labels


def test_in_place_poison_pool_preserves_indices_and_exact_truth():
    images, labels = synthetic_training()
    pool = pilot.build_poisoned_training(images, labels, seed=72, poison_count=5)
    selected = pool["selected_indices"]

    assert len(pool["images"]) == len(images)
    assert np.array_equal(pool["original_index"], np.arange(len(images)))
    assert pool["is_poison"].sum() == 5
    assert np.array_equal(np.flatnonzero(pool["is_poison"]), selected)
    assert np.all(labels[selected] != pilot.TARGET_CLASS)
    assert np.all(pool["assigned_labels"][selected] == pilot.TARGET_CLASS)
    assert np.array_equal(
        pool["assigned_labels"][~pool["is_poison"]], labels[~pool["is_poison"]]
    )
    assert np.array_equal(
        pool["images"][~pool["is_poison"]], images[~pool["is_poison"]]
    )
    assert np.all(pool["images"][selected, -3:, -3:, :] == 1.0)
    assert pool["receipt"]["construction"] == "in_place_replacement"


def test_poison_selection_is_seeded_and_without_replacement():
    images, labels = synthetic_training(per_class=20)
    first = pilot.build_poisoned_training(images, labels, seed=72, poison_count=12)
    repeated = pilot.build_poisoned_training(images, labels, seed=72, poison_count=12)
    other = pilot.build_poisoned_training(images, labels, seed=73, poison_count=12)
    assert np.array_equal(first["selected_indices"], repeated["selected_indices"])
    assert not np.array_equal(first["selected_indices"], other["selected_indices"])
    assert len(np.unique(first["selected_indices"])) == 12


def test_paired_models_have_identical_initial_state():
    mean = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    std = np.asarray([0.8, 0.9, 1.0], dtype=np.float32)
    clean_hash = pilot.initial_state_fingerprint(72, mean, std)
    suspect_hash = pilot.initial_state_fingerprint(72, mean, std)
    other_hash = pilot.initial_state_fingerprint(73, mean, std)
    assert clean_hash == suspect_hash
    assert clean_hash != other_hash


def test_balanced_accuracy_weights_classes_equally():
    labels = np.asarray([0, 0, 0, 1])
    predictions = np.asarray([0, 0, 0, 0])
    assert pilot.balanced_accuracy(labels, predictions) == pytest.approx(0.5)


def test_detector_budgets_are_explicitly_oracle_and_nonoracle():
    labels = np.repeat(np.arange(8, dtype=np.int64), 1000)
    budgets = pilot.detector_budgets(labels)
    assert budgets["oracle_target"][6] == 600
    assert sum(budgets["oracle_target"].values()) == 600
    assert set(budgets["uniform_5pct"].values()) == {50}


def test_flag_confusion_counts_and_target_clean_fpr():
    truth = np.asarray([False, True, False, True])
    flags = np.asarray([True, True, False, False])
    assigned = np.asarray([6, 6, 0, 6])
    original = np.asarray([6, 0, 0, 1])
    metrics = pilot.evaluate_flags(flags, truth, assigned, original)
    assert (metrics["tp"], metrics["fn"], metrics["fp"], metrics["tn"]) == (
        1,
        1,
        1,
        1,
    )
    assert metrics["tpr"] == pytest.approx(0.5)
    assert metrics["fpr_all_clean"] == pytest.approx(0.5)
    assert metrics["target_clean_fpr"] == pytest.approx(1.0)


def test_frozen_stage_seeds_reject_partial_or_reordered_runs():
    assert pilot.validate_stage_seeds("development", [72, 73]) == (72, 73)
    with pytest.raises(ValueError):
        pilot.validate_stage_seeds("development", [72])
    with pytest.raises(ValueError):
        pilot.validate_stage_seeds("confirmation", [78, 77, 76, 75, 74])


def test_sample_mapping_and_confusion_gate_replays_per_sample_rows():
    samples = []
    truth = [1] * pilot.POISON_COUNT + [0, 0]
    flags = [1] + [0] * (pilot.POISON_COUNT - 1) + [1, 0]
    for index, (is_poison, flagged) in enumerate(zip(truth, flags)):
        samples.append(
            {
                "seed": 72,
                "view": "view",
                "method": "method",
                "sample_index": index,
                "original_index": index,
                "original_label": 0,
                "assigned_label": 6 if is_poison else 0,
                "is_poison": is_poison,
                "flagged": flagged,
            }
        )
    configs = [
        {
            "seed": 72,
            "view": "view",
            "method": "method",
            "tp": 1,
            "fn": pilot.POISON_COUNT - 1,
            "fp": 1,
            "tn": 1,
        }
    ]
    assert pilot.replay_sample_mappings_and_confusions(
        samples, configs, pilot.POISON_COUNT + 2
    )
    samples[-1]["original_index"] = 0
    assert not pilot.replay_sample_mappings_and_confusions(
        samples, configs, pilot.POISON_COUNT + 2
    )


def test_confirmation_gate_requires_matching_development_manifest(tmp_path):
    gate = {
        "all_development_victims_eligible": True,
        "all_classes_have_required_samples": True,
        "scan_finite_and_nondegenerate_both_views": True,
        "sample_mappings_and_confusions_replay": True,
        "at_least_one_seed_within_runtime_budget": True,
        "expand_to_confirmation": True,
    }
    payload = {
        "experiment": "bloodmnist_multiclass_backdoorbench_pilot_v1",
        "stage": "development",
        "protocol_sha256": pilot.sha256_file(pilot.PROTOCOL),
        "seeds": list(pilot.DEVELOPMENT_SEEDS),
        "gate": gate,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    assert pilot.confirmation_is_authorized(tmp_path)
    payload["stage"] = "confirmation"
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    assert not pilot.confirmation_is_authorized(tmp_path)
