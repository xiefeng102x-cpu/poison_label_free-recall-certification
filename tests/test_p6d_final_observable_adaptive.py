from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_p6d_final_observable_adaptive as p6d
import verify_p6d_final_observable_adaptive as verifier


SYNTHETIC_SEEDS = (10000, 10001, 10002)


def synthetic_observables(seed: int = SYNTHETIC_SEEDS[0], *, requires_grad=False):
    p6d.require_synthetic_seed(seed)
    rng = np.random.default_rng(seed)

    def make(rows):
        return torch.tensor(
            rng.normal(size=(rows, p6d.OBSERVABLE_DIMENSION)),
            dtype=torch.float64,
            requires_grad=requires_grad,
        )

    return make(p6d.TRUSTED_COUNT), make(p6d.CLEAN_CANDIDATE_COUNT), make(
        p6d.POISON_CANDIDATE_COUNT
    )


def epoch(seed, number, ca, asr, tpr, mean_p):
    p6d.require_synthetic_seed(seed)
    return {
        "seed": seed,
        "epoch": number,
        "validation_balanced_ca": ca,
        "validation_asr": asr,
        "tpr": tpr,
        "mean_poison_p_value": mean_p,
    }


def test_partition_and_exact_replay_match_independent_verifier():
    seed = SYNTHETIC_SEEDS[0]
    trusted, clean, poison = synthetic_observables(seed)
    partition = p6d.build_crossfit_partition(seed)
    exact = p6d.numpy_cfcmdu_score(
        seed, trusted.numpy(), clean.numpy(), poison.numpy(), partition=partition
    )
    replay = verifier.replay_numpy_cfcmdu(
        seed,
        trusted.numpy(),
        clean.numpy(),
        poison.numpy(),
        partition_receipt=partition.to_json(),
    )
    np.testing.assert_array_equal(exact["scores"], replay["scores"])
    np.testing.assert_array_equal(exact["p_values"], replay["p_values"])
    np.testing.assert_array_equal(exact["flags"], replay["flags"])
    assert (exact["tp"], exact["fn"], exact["fp"], exact["tn"]) == (
        replay["tp"],
        replay["fn"],
        replay["fp"],
        replay["tn"],
    )
    scored = np.concatenate([group.score_indices for group in partition.groups])
    assert np.array_equal(np.sort(scored), np.arange(660))
    assert all(
        np.intersect1d(group.score_indices, group.direction_indices).size == 0
        for group in partition.groups
    )


def test_tie_behavior_uses_greater_equal_calibration_rank():
    calibration = np.concatenate([np.ones(4), np.zeros(76)])
    p_equal_fourth = p6d._numpy_one_sided_p_values(calibration, np.asarray([1.0]))
    p_strictly_above = p6d._numpy_one_sided_p_values(
        calibration, np.asarray([np.nextafter(1.0, np.inf)])
    )
    assert p_equal_fourth[0] == 5 / 81
    assert p_equal_fourth[0] > p6d.ALPHA
    assert p_strictly_above[0] == 1 / 81
    assert p_strictly_above[0] <= p6d.ALPHA


def test_torch_channel_has_exact_parity_and_full_channel_gradients():
    seed = SYNTHETIC_SEEDS[1]
    trusted, clean, poison = synthetic_observables(seed, requires_grad=True)
    result = p6d.assert_torch_numpy_parity(seed, trusted, clean, poison)
    assert result["max_abs_score_error"] <= 1e-12
    assert result["max_abs_threshold_error"] <= 1e-12
    channel = result["torch"]
    assert channel["groups"][0]["tau"].requires_grad
    assert channel["groups"][0]["direction"].requires_grad
    assert channel["groups"][0]["calibration_scores"].requires_grad
    assert not channel["groups"][0]["sigma_detached"].requires_grad
    channel["soft_flag_loss"].backward()
    assert trusted.grad is not None and torch.count_nonzero(trusted.grad) > 0
    assert clean.grad is not None and torch.count_nonzero(clean.grad) > 0
    assert poison.grad is not None and torch.count_nonzero(poison.grad) > 0
    calibration = p6d.build_crossfit_partition(seed).calibration_indices
    assert torch.count_nonzero(trusted.grad[calibration]) > 0


def test_epoch_selection_is_validation_only_and_uses_frozen_ties():
    seed = SYNTHETIC_SEEDS[0]
    selected = p6d.select_epoch(
        [
            epoch(seed, 0, 0.93, 0.86, 0.20, 0.30),
            epoch(seed, 1, 0.94, 0.87, 0.10, 0.20),
            epoch(seed, 2, 0.95, 0.88, 0.10, 0.40),
            epoch(seed, 3, 0.96, 0.89, 0.10, 0.40),
        ]
    )
    assert selected["epoch"] == 3  # larger normalized guard margin
    assert selected["validation_eligible"] is True
    with pytest.raises(RuntimeError, match="test information"):
        p6d.select_epoch(
            [
                {
                    **epoch(seed, 0, 0.95, 0.90, 0.1, 0.4),
                    "test_asr": 0.99,
                }
            ]
        )


def test_epoch_fallback_maximizes_guard_margin_then_tpr_then_earlier():
    seed = SYNTHETIC_SEEDS[1]
    selected = p6d.select_epoch(
        [
            epoch(seed, 0, 0.91, 0.84, 0.80, 0.1),
            epoch(seed, 1, 0.91, 0.84, 0.20, 0.9),
            epoch(seed, 2, 0.91, 0.84, 0.20, 0.1),
        ]
    )
    assert selected["epoch"] == 1
    assert selected["validation_eligible"] is False


def configuration_summary(configuration_id, weight, eligible, tpr, asr, ca):
    return {
        "configuration_id": configuration_id,
        "evasion_weight": weight,
        "validation_eligible_count": eligible,
        "mean_tpr_eligible": tpr,
        "worst_validation_asr": asr,
        "worst_validation_balanced_ca": ca,
    }


def test_development_ranking_and_test_firewall(tmp_path):
    summaries = [
        configuration_summary("w0.5", 0.5, 2, 0.10, 0.90, 0.94),
        configuration_summary("w2", 2.0, 3, 0.40, 0.88, 0.93),
        configuration_summary("w8", 8.0, 3, 0.30, 0.86, 0.92),
    ]
    receipt_path = tmp_path / "development_selection_receipt.json"
    receipt = p6d.create_development_selection_receipt(receipt_path, summaries)
    assert receipt["selected_configuration_id"] == "w8"
    assert p6d.authorize_development_test_read(receipt_path, "w8") == receipt
    with pytest.raises(PermissionError, match="unselected"):
        p6d.authorize_development_test_read(receipt_path, "w2")
    receipt_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        p6d.authorize_development_test_read(receipt_path, "w8")


def test_atomic_resume_refuses_signature_and_artifact_hash_mismatch(tmp_path):
    signature = {"stage": "synthetic", "seeds": list(SYNTHETIC_SEEDS)}
    output = tmp_path / "run"
    manifest = p6d.initialize_or_resume(
        output, signature, immutable_hashes={"fixture": "abc"}
    )
    assert p6d.initialize_or_resume(
        output, signature, immutable_hashes={"fixture": "abc"}
    ) == manifest
    with pytest.raises(RuntimeError, match="signature"):
        p6d.initialize_or_resume(
            output,
            {**signature, "seeds": [10003]},
            immutable_hashes={"fixture": "abc"},
        )
    artifact = output / "synthetic.json"
    p6d.atomic_json(artifact, {"seed": SYNTHETIC_SEEDS[0]})
    manifest = p6d.update_manifest_artifact(output, manifest, artifact)
    artifact.write_text(json.dumps({"seed": SYNTHETIC_SEEDS[1]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="corrupt artifact"):
        p6d.initialize_or_resume(
            output, signature, immutable_hashes={"fixture": "abc"}
        )


def test_atomic_resume_refuses_untracked_files(tmp_path):
    output = tmp_path / "run"
    signature = {"stage": "synthetic", "seeds": list(SYNTHETIC_SEEDS)}
    p6d.initialize_or_resume(output, signature, immutable_hashes={})
    p6d.atomic_json(output / "orphan.json", {"unexpected": True})
    with pytest.raises(RuntimeError, match="untracked artifact"):
        p6d.initialize_or_resume(output, signature, immutable_hashes={})


def test_confirmation_lock_issuer_requires_complete_development_bundle(tmp_path):
    with pytest.raises(RuntimeError, match="development manifest is missing"):
        p6d.issue_confirmation_lock_from_development(tmp_path)
    assert not (tmp_path / "confirmation_lock_receipt.json").exists()


def test_confirmation_resume_refuses_partial_runs():
    assert p6d.confirmation_resume_action({"status": "initialized", "artifacts": {}}) == "start"
    assert p6d.confirmation_resume_action({"status": "complete", "artifacts": {}}) == "verify_complete"
    with pytest.raises(RuntimeError, match="incomplete confirmation stage"):
        p6d.confirmation_resume_action({"status": "in_progress", "artifacts": {"seed.json": {}}})


def test_lossless_observable_writer_records_frozen_reference_population():
    seed = SYNTHETIC_SEEDS[2]
    trusted, clean, poison = synthetic_observables(seed)
    partition = p6d.build_crossfit_partition(seed)
    trusted_orig = np.arange(p6d.TRUSTED_COUNT) + 100
    clean_orig = np.arange(p6d.CLEAN_CANDIDATE_COUNT) + 1000
    poison_orig = np.arange(p6d.POISON_CANDIDATE_COUNT) + 5000
    splits = {
        "val": {"target_original_indices": trusted_orig},
        "train": {"target_original_indices": clean_orig},
        "poison_original_indices": poison_orig,
    }
    rows = p6d._lossless_observable_rows(
        "synthetic", seed, "w2", splits,
        {"trusted": trusted.numpy(), "clean": clean.numpy(), "poison": poison.numpy()},
        partition,
    )
    assert len(rows) == p6d.TRUSTED_COUNT + p6d.CLEAN_CANDIDATE_COUNT + p6d.POISON_CANDIDATE_COUNT
    refs = {row["trusted_sample_index"] for row in rows if row["population"] == "reference"}
    assert refs == set(map(int, partition.reference_indices))
    first = rows[0]
    trusted_index = first["trusted_sample_index"]
    assert first["original_dataset_index"] == int(trusted_orig[trusted_index])
    expected = trusted.numpy()[trusted_index].astype("<f4").tolist()
    assert [first[f"delta_xz16_{i:02d}"] for i in range(16)] == expected


def test_partition_receipt_tampering_is_refused():
    seed = SYNTHETIC_SEEDS[2]
    receipt = p6d.build_crossfit_partition(seed).to_json()
    receipt["groups"][0]["score_indices"][0] = 659
    with pytest.raises(AssertionError, match="frozen RNG"):
        verifier.validate_partition_receipt(seed, receipt)


def test_bootstrap_requires_all_five_eligible_victims():
    rows = [
        {
            "seed": 10010 + index,
            "eligible": index < 4,
            "tp": 55,
            "fn": 5,
            "fp": 20,
            "tn": 580,
        }
        for index in range(5)
    ]
    summary = p6d.bootstrap_detector_outcomes(rows)
    assert summary == {
        "eligible_victim_count": 4,
        "decision": "INSUFFICIENT_ELIGIBLE_VICTIMS",
        "efficacy_pooled": False,
    }


def test_formal_guard_does_not_initialize_without_exact_acknowledgement(tmp_path):
    args = type(
        "Args",
        (),
        {
            "acknowledge_formal_seed_consumption": None,
            "stage": "development",
            "output": tmp_path,
        },
    )()
    with pytest.raises(PermissionError, match="requires"):
        p6d.guarded_formal_stage(args)
    assert not any(tmp_path.iterdir())
