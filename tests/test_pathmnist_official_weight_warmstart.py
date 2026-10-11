from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pathmnist_external_validation as core  # noqa: E402
import pathmnist_official_resnet18 as architecture  # noqa: E402
import run_pathmnist_official_weight_warmstart as pilot  # noqa: E402


def require_redistributable_official_weights() -> None:
    receipt = pilot.read_json(pilot.RECEIPT)
    missing = [
        pilot.weight_path_from_receipt(receipt, seed)
        for seed in pilot.FORMAL_SEEDS
        if not pilot.weight_path_from_receipt(receipt, seed).is_file()
    ]
    if missing:
        pytest.skip(
            "official PathMNIST weights are intentionally excluded from the "
            "blind-review profile pending explicit redistribution terms"
        )


def test_protocol_locks_all_sources_receipt_weights_and_mapping() -> None:
    require_redistributable_official_weights()
    protocol, receipt, observed = pilot.validate_frozen_inputs(pilot.DEFAULT_DATASET)
    assert observed == protocol["frozen_inputs"]
    assert receipt["selection"]["intended_attack_seed_mapping"] == {
        "93": "resnet18_28_1.pth",
        "94": "resnet18_28_2.pth",
    }
    assert protocol["official_weight_provenance"]["selection_prohibition"].startswith(
        "The sanity metrics must not be used"
    )


@pytest.mark.parametrize("seed", pilot.FORMAL_SEEDS)
def test_official_initial_state_strict_loads_bit_identically(seed: int) -> None:
    require_redistributable_official_weights()
    receipt = pilot.read_json(pilot.RECEIPT)
    model, official_state, path = pilot.load_official_initial_state(seed, receipt)
    assert path.name == pilot.WEIGHT_FILENAMES[seed]
    assert pilot.states_bit_identical(model.state_dict(), official_state)
    assert core.state_fingerprint(model.state_dict()) == core.state_fingerprint(
        official_state
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 11_173_449


def test_full_training_poison_count_and_in_place_mapping() -> None:
    assert pilot.poison_count_for_training(89_996) == 4_500
    with pytest.raises(ValueError, match="89,996"):
        pilot.poison_count_for_training(89_995)
    labels = np.tile(np.arange(9, dtype=np.int64), 20)
    images = np.zeros((len(labels), 28, 28, 3), dtype=np.uint8)
    first = core.build_poisoned_training(
        images, labels, seed=93, poison_count=9, target_class=8, patch_size=4
    )
    second = core.build_poisoned_training(
        images, labels, seed=93, poison_count=9, target_class=8, patch_size=4
    )
    assert np.array_equal(first["selected_indices"], second["selected_indices"])
    assert int(first["is_poison"].sum()) == 9
    assert np.array_equal(first["original_index"], np.arange(len(labels)))
    assert np.all(first["original_labels"][first["is_poison"]] != 8)
    assert np.all(first["assigned_labels"][first["is_poison"]] == 8)
    assert np.all(first["images"][first["is_poison"], -4:, -4:, :] == 255)


def test_fixed_epoch_training_has_no_test_input_or_selection_hook() -> None:
    parameters = set(inspect.signature(pilot.train_seed).parameters)
    assert "test_images" not in parameters
    assert "test_labels" not in parameters
    assert {"validation_images", "validation_labels"}.issubset(parameters)
    assert pilot.PRIMARY_CHECKPOINT == "fixed_final_epoch_5"
    assert pilot.VALIDATION_ROLE == "diagnostic_only_no_model_selection"
    assert pilot.FORMAL_EPOCHS == 5
    assert pilot.LEARNING_RATE == 3e-4


def test_optimizer_is_full_parameter_adam_without_scheduler() -> None:
    model = architecture.PathMNISTResNet18()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=pilot.LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimized == {id(parameter) for parameter in model.parameters()}
    assert optimizer.defaults["lr"] == 3e-4
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["eps"] == 1e-8
    assert optimizer.defaults["weight_decay"] == 0.0
    assert pilot.read_json(pilot.PROTOCOL)["training"]["scheduler"] is None


def test_resume_strictly_rejects_contract_hash_mismatch(tmp_path: Path) -> None:
    model = architecture.PathMNISTResNet18()
    optimizer = torch.optim.Adam(model.parameters(), lr=pilot.LEARNING_RATE)
    checkpoint = tmp_path / "latest.pt"
    torch.save({"contract": {"runner_source_sha256": "wrong"}}, checkpoint)
    with pytest.raises(RuntimeError, match="contract mismatch"):
        pilot.load_resume_checkpoint(
            checkpoint,
            expected_contract={"runner_source_sha256": "expected"},
            expected_official_state=pilot.cpu_state(model.state_dict()),
            model=model,
            optimizer=optimizer,
            shuffle_generator=torch.Generator(),
        )


def test_resume_rejects_nonidentical_initial_official_state(tmp_path: Path) -> None:
    model = architecture.PathMNISTResNet18()
    official = pilot.cpu_state(model.state_dict())
    corrupted = pilot.cpu_state(official)
    corrupted["linear.bias"][0] += 1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=pilot.LEARNING_RATE)
    contract = {"contract": "exact"}
    checkpoint = tmp_path / "latest.pt"
    torch.save(
        {
            "contract": contract,
            "model_state": pilot.cpu_state(model.state_dict()),
            "optimizer_state": optimizer.state_dict(),
            "epoch": 1,
            "rng_state": pilot.capture_rng_state(torch.Generator()),
            "history": [],
            "initial_official_state": corrupted,
            "initial_official_state_sha256": core.state_fingerprint(official),
        },
        checkpoint,
    )
    with pytest.raises(RuntimeError, match="not bit-identical"):
        pilot.load_resume_checkpoint(
            checkpoint,
            expected_contract=contract,
            expected_official_state=official,
            model=model,
            optimizer=optimizer,
            shuffle_generator=torch.Generator(),
        )


def test_dry_run_is_permanently_nonformal() -> None:
    mode = pilot.validate_execution_mode(
        dry_run=True,
        max_epochs=1,
        dry_run_batches=1,
        seeds=(93,),
    )
    assert mode == pilot.ExecutionMode(False, 1, 1)
    gate = pilot.compute_gate([{"seed": 93, "eligible": True}], formal=False)
    assert not gate["formal_execution"]
    assert not gate["authorize_followup"]
    assert pilot.status_from_gate(gate, formal=False) == "DRY_RUN_NONFORMAL"
    with pytest.raises(ValueError, match="only with --dry-run"):
        pilot.validate_execution_mode(
            dry_run=False,
            max_epochs=1,
            dry_run_batches=1,
            seeds=pilot.FORMAL_SEEDS,
        )


def test_prespecified_seed93_stop_and_two_seed_gate() -> None:
    seed93_failure = [{"seed": 93, "eligible": False}]
    assert pilot.should_stop_after_seed93(seed93_failure[0], formal=True)
    failed = pilot.compute_gate(seed93_failure, formal=True)
    assert failed["valid_prespecified_early_stop_after_seed93"]
    assert failed["formal_conclusion_complete"]
    assert failed["definitive_gate_failure"]
    assert not failed["seed94_required"]
    assert not failed["authorize_followup"]
    assert pilot.status_from_gate(failed, formal=True) == (
        "FORMAL_STOPPED_AFTER_SEED93_GATE_FAILED"
    )

    seed93_pass_only = pilot.compute_gate(
        [{"seed": 93, "eligible": True}], formal=True
    )
    assert seed93_pass_only["seed94_required"]
    assert not seed93_pass_only["formal_conclusion_complete"]
    assert not seed93_pass_only["authorize_followup"]

    passed = pilot.compute_gate(
        [{"seed": 93, "eligible": True}, {"seed": 94, "eligible": True}],
        formal=True,
    )
    assert passed["both_fixed_epoch5_victims_eligible"]
    assert passed["formal_conclusion_complete"]
    assert passed["authorize_followup"]

    seed94_failure = pilot.compute_gate(
        [{"seed": 93, "eligible": True}, {"seed": 94, "eligible": False}],
        formal=True,
    )
    assert seed94_failure["formal_conclusion_complete"]
    assert seed94_failure["definitive_gate_failure"]
    assert not seed94_failure["authorize_followup"]
