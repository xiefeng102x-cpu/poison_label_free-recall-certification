from __future__ import annotations

import numpy as np

import run_tifs_flare_matched as flare


def test_poison_truth_is_fixed_and_has_4500_rows() -> None:
    labels = np.tile(np.arange(9, dtype=np.int64), 10_000)[:89_996]
    truth_a = flare.poison_truth(labels, 103)
    truth_b = flare.poison_truth(labels, 103)
    truth_c = flare.poison_truth(labels, 104)
    assert truth_a.sum() == 4_500
    assert np.array_equal(truth_a, truth_b)
    assert not np.array_equal(truth_a, truth_c)
    assert not np.any(truth_a[labels == flare.TARGET_CLASS])


def test_counts_use_full_clean_denominator() -> None:
    truth = np.asarray([1, 1, 0, 0, 0, 0], dtype=bool)
    flags = np.asarray([1, 0, 1, 0, 0, 0], dtype=bool)
    assigned = np.asarray([8, 8, 8, 8, 2, 3], dtype=np.int64)
    result = flare.counts(flags, truth, assigned)
    assert result["tp"] == 1
    assert result["n_poison"] == 2
    assert result["fp_all_clean"] == 1
    assert result["n_all_clean"] == 4
    assert result["fp_target_clean"] == 1
    assert result["n_target_clean"] == 2
    assert result["tpr"] == 0.5
    assert result["all_clean_fpr"] == 0.25


def test_bn_inventory_is_exact() -> None:
    model = flare.architecture.PathMNISTResNet18()
    names = flare.bn_layer_names(model)
    assert len(names) == 20
    assert names[0] == "bn1"
    assert names[-1] == "layer4.1.bn2"
    assert sum(model.get_submodule(name).num_features for name in names) == 4_800

