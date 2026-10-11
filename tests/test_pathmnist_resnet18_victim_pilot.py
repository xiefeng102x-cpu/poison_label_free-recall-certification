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
import pathmnist_official_resnet18 as model_source  # noqa: E402
import run_pathmnist_resnet18_victim_pilot as pilot  # noqa: E402


def test_official_style_resnet18_architecture_and_parameter_count() -> None:
    model = model_source.make_model(93)
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert [len(model.layer1), len(model.layer2), len(model.layer3), len(model.layer4)] == [
        2,
        2,
        2,
        2,
    ]
    assert [model.layer1[0].conv1.out_channels, model.layer2[0].conv1.out_channels,
            model.layer3[0].conv1.out_channels, model.layer4[0].conv1.out_channels] == [
        64,
        128,
        256,
        512,
    ]
    assert not hasattr(model, "maxpool")
    assert model.linear.out_features == 9
    assert "channel_mean" not in model.state_dict()
    assert "channel_std" not in model.state_dict()
    values = torch.rand(2, 3, 28, 28)
    assert model.hidden(values).shape == (2, 512)
    assert model(values).shape == (2, 9)
    assert sum(parameter.numel() for parameter in model.parameters()) == 11_173_449


def test_full_training_poison_count_and_in_place_mapping() -> None:
    assert pilot.poison_count_for_training(89_996) == 4_500
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


def test_training_signature_cannot_receive_test_data() -> None:
    parameters = set(inspect.signature(pilot.train_seed).parameters)
    assert "test_images" not in parameters
    assert "test_labels" not in parameters
    assert {"validation_images", "validation_labels"}.issubset(parameters)
    assert pilot.SELECTION_METRIC == "clean_validation_macro_ovr_auc"


def test_resume_strictly_rejects_contract_hash_mismatch(tmp_path: Path) -> None:
    model = model_source.make_model(93)
    optimizer = torch.optim.Adam(model.parameters(), lr=pilot.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(pilot.MILESTONES), gamma=pilot.SCHEDULER_GAMMA
    )
    checkpoint = tmp_path / "latest.pt"
    torch.save({"contract": {"runner_source_sha256": "wrong"}}, checkpoint)
    with pytest.raises(RuntimeError, match="contract mismatch"):
        pilot.load_resume_checkpoint(
            checkpoint,
            expected_contract={"runner_source_sha256": "expected"},
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            shuffle_generator=torch.Generator(),
        )


def test_dry_run_is_nonformal_and_cannot_authorize_followup() -> None:
    mode = pilot.validate_execution_mode(
        dry_run=True,
        max_epochs=1,
        dry_run_batches=1,
        seeds=(93,),
    )
    assert mode == pilot.ExecutionMode(False, 1, 1)
    gate = pilot.compute_gate(
        [{"seed": 93, "eligible": True}, {"seed": 94, "eligible": True}],
        formal=mode.formal,
    )
    assert gate["both_frozen_seeds_complete"]
    assert gate["both_best_validation_victims_eligible"]
    assert not gate["formal_execution"]
    assert not gate["authorize_followup"]
    with pytest.raises(ValueError, match="only with --dry-run"):
        pilot.validate_execution_mode(
            dry_run=False,
            max_epochs=2,
            dry_run_batches=1,
            seeds=pilot.FORMAL_SEEDS,
        )


def test_formal_gate_requires_both_frozen_eligible_seeds() -> None:
    passing = pilot.compute_gate(
        [{"seed": 93, "eligible": True}, {"seed": 94, "eligible": True}],
        formal=True,
    )
    assert passing["authorize_followup"]
    failed = pilot.compute_gate(
        [{"seed": 93, "eligible": True}, {"seed": 94, "eligible": False}],
        formal=True,
    )
    assert not failed["both_best_validation_victims_eligible"]
    assert not failed["authorize_followup"]
    duplicate = pilot.compute_gate(
        [
            {"seed": 93, "eligible": True},
            {"seed": 94, "eligible": True},
            {"seed": 94, "eligible": True},
        ],
        formal=True,
    )
    assert not duplicate["both_frozen_seeds_complete"]
    assert not duplicate["authorize_followup"]
