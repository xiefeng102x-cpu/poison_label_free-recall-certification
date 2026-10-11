from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import run_pathmnist_official_weight_detector_confirmation as confirmation  # noqa: E402
import verify_pathmnist_official_weight_detector_confirmation as verifier  # noqa: E402


def test_protocol_freezes_confirmation_map_and_scope() -> None:
    protocol_path = (
        PAPER14
        / "results/top_journal_program_v1/"
        "pathmnist_official_weight_detector_confirmation_protocol_v1.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    seeds = tuple(protocol["seed_plan"]["formal_confirmation_seeds"])
    mapping = {
        int(seed): name
        for seed, name in protocol["seed_plan"]["official_weight_mapping"].items()
    }
    confirmation.validate_seed_map(seeds, mapping)
    assert protocol["victims"]["training"]["epochs"] == 5
    assert protocol["victims"]["suspect_attack"]["poison_count"] == 4500
    assert protocol["detectors"]["primary"]["feature_transform"].startswith("identity")
    assert protocol["execution_firewall"]["formal_runner_status"].startswith("implemented")
    assert "20271000" in protocol["dataset"]["validation_index_selection"]
    assert "both clean and suspect" in protocol["victim_eligibility"]["population_rule"]


def test_seed_guard_rejects_feasibility_and_partial_blocks() -> None:
    with pytest.raises(ValueError, match="fixed"):
        confirmation.validate_seed_map((93, 94), confirmation.WEIGHT_MAP)
    wrong_map = dict(confirmation.WEIGHT_MAP)
    wrong_map[107] = "resnet18_28_2.pth"
    with pytest.raises(ValueError, match="map"):
        confirmation.validate_seed_map(confirmation.CONFIRMATION_SEEDS, wrong_map)


def test_primary_is_raw_space_and_scores_only_target_assigned_rows() -> None:
    result = confirmation.synthetic_dry_run()
    assert result["status"] == "synthetic_dry_run_passed_nonformal"
    assert result["formal_data_or_weights_loaded"] is False
    assert result["training_performed"] is False
    assert result["confirmation_seed_files_opened"] is False
    receipt = result["primary_receipt"]
    assert receipt["whitening"] is False
    assert receipt["covariance_estimation"] is False
    assert receipt["projection"] == "identity/raw hidden-feature coordinates"


def test_metrics_report_both_all_clean_and_target_clean_fpr() -> None:
    flags = np.array([True, True, True, False, False])
    poison = np.array([True, False, False, False, False])
    assigned = np.array([8, 8, 3, 8, 2])
    metrics = confirmation.confusion_counts(flags, poison, assigned)
    assert metrics["tpr"] == 1.0
    assert metrics["all_clean_fpr"] == pytest.approx(2 / 4)
    assert metrics["target_clean_fpr"] == pytest.approx(1 / 2)


def test_paired_drift_uses_aligned_same_input_differences() -> None:
    clean_validation = np.array([[1.0, 3.0], [2.0, 2.0]])
    suspect_validation = np.array([[2.0, 4.0], [1.0, 4.0]])
    clean_candidates = np.array([[0.0, 1.0]])
    suspect_candidates = np.array([[1.0, -1.0]])
    validation_drift, candidate_drift = confirmation.paired_drift_view(
        clean_validation,
        suspect_validation,
        clean_candidates,
        suspect_candidates,
    )
    np.testing.assert_array_equal(validation_drift, [[1.0, 1.0], [-1.0, 2.0]])
    np.testing.assert_array_equal(candidate_drift, [[1.0, -2.0]])
    with pytest.raises(ValueError, match="matching rows"):
        confirmation.paired_drift_view(
            clean_validation, suspect_validation[:1], clean_candidates, suspect_candidates
        )


def test_seed_block_bootstrap_requires_all_five_eligible_blocks() -> None:
    rows = [
        {
            "seed": seed,
            "eligible": True,
            "tp": 90 + (seed - 103),
            "n_poison": 100,
            "fp_all_clean": 2,
            "n_all_clean": 100,
            "fp_target_clean": 1,
            "n_target_clean": 30,
        }
        for seed in confirmation.CONFIRMATION_SEEDS
    ]
    summary = confirmation.summarize_seed_blocks(rows, bootstrap_replicates=200)
    replay = confirmation.summarize_seed_blocks(rows, bootstrap_replicates=200)
    assert summary == replay
    assert summary["pooled_tpr"] == pytest.approx(0.92)
    assert summary["pooled_all_clean_fpr"] == pytest.approx(0.02)
    assert summary["primary_confirmation_pass"] is True
    low, high = summary["seed_block_bootstrap_95pct_ci"]["pooled_tpr"]
    assert 0.0 <= low <= high <= 1.0
    rows[-1]["eligible"] = False
    with pytest.raises(ValueError, match="five eligible"):
        confirmation.summarize_seed_blocks(rows, bootstrap_replicates=20)


def test_formal_flag_is_disabled_before_real_inputs_are_opened() -> None:
    import subprocess

    runner = PAPER14 / "src/run_pathmnist_official_weight_detector_confirmation.py"
    completed = subprocess.run(
        [sys.executable, str(runner), "--formal", "--smoke-receipt", "/nonexistent/nonformal-smoke-receipt.json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "explicit --ack" in completed.stderr


def test_mps_device_selection_can_be_exercised_with_a_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(confirmation.torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(confirmation.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(confirmation.torch, "__version__", "2.7.1")
    assert str(confirmation._mps_device()) == "mps"


def test_real_executor_is_gated_by_independent_smoke_verification(tmp_path, monkeypatch) -> None:
    bad_receipt = tmp_path / "bad.json"
    bad_receipt.write_text('{"status":"PASS_NONFORMAL_SYNTHETIC_ONLY"}', encoding="utf-8")
    monkeypatch.setattr(verifier, "verify_smoke_receipt", lambda *_: {"status": "FAIL", "failures": ["tampered"]})
    with pytest.raises(RuntimeError, match="smoke receipt"):
        confirmation.execute_formal(tmp_path / "no-data.npz", tmp_path / "out", bad_receipt,
                                    ack=confirmation.FORMAL_ACK)
    assert not (tmp_path / "out").exists()


def test_formal_ack_is_required_before_any_input_hash_or_output_creation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(confirmation, "sha256_file", lambda path: pytest.fail("input hashing happened before ACK"))
    output = tmp_path / "new-output"
    with pytest.raises(RuntimeError, match="explicit --ack"):
        confirmation.execute_formal(tmp_path / "unopened.npz", output, tmp_path / "unopened-receipt.json")
    assert not output.exists()


def test_completed_formal_output_cannot_be_resumed(tmp_path) -> None:
    output = tmp_path / "complete"
    output.mkdir()
    (output / "manifest.json").write_text('{"status":"COMPLETE_FORMAL"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="completed formal output cannot be resumed"):
        confirmation.execute_formal(tmp_path / "unopened.npz", output, tmp_path / "unopened-smoke.json",
                                    resume=True, ack=confirmation.FORMAL_ACK)


def test_candidate_features_are_tied_to_realized_suspect_pool() -> None:
    rng = np.random.default_rng(9)
    images = rng.integers(0, 255, size=(40, 28, 28, 3), dtype=np.uint8)
    labels = np.repeat(np.arange(8), 5).astype(np.int64)
    pool = confirmation.build_suspect_candidate_pool(images, labels, seed=77, poison_count=8)
    clean_input, suspect_input = confirmation.shared_candidate_feature_inputs(pool["images"])
    assert clean_input is suspect_input
    assert confirmation.candidate_pool_image_hash(clean_input) == confirmation.candidate_pool_image_hash(suspect_input)
    assert np.all(pool["assigned_labels"][pool["selected_indices"]] == confirmation.TARGET_CLASS)
    assert np.all(pool["images"][pool["selected_indices"], -4:, -4:, :] == 255)
    assert len(pool["selected_indices"]) == 8
    assert np.all(images[np.setdiff1d(np.arange(40), pool["selected_indices"])] == pool["images"][np.setdiff1d(np.arange(40), pool["selected_indices"])])


def test_smoke_receipt_tampering_is_detected_without_real_input_access() -> None:
    receipt = {"dataset_sha256": "dataset-a", "runner_sha256": "runner-a"}
    expected = {"dataset_sha256": "dataset-a", "runner_sha256": "runner-a"}
    assert verifier.smoke_binding_failures(receipt, expected) == []
    receipt["dataset_sha256"] = "tampered"
    assert verifier.smoke_binding_failures(receipt, expected) == ["smoke dataset_sha256 mismatch"]
