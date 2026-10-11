from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import backdoorbench_feature_adapter as adapter


def test_spectral_signatures_matches_upstream_block():
    features = np.asarray([[1.0, 0.0], [2.0, 0.0], [9.0, 0.0]])
    result = adapter.spectral_signatures(features, np.asarray([0, 0, 0]), 50)
    assert np.allclose(result.scores, [1.0, 2.0, 9.0])
    assert result.suspicious_mask.tolist() == [False, False, True]
    assert result.receipt["threshold_rule"] == "strictly_greater_per_class"


class _FakeSCAN:
    def build_global_model(self, features, labels, n_classes):
        assert n_classes == 2
        return {"ok": True}

    def build_local_model(self, features, labels, global_model, n_classes):
        np.random.random(7)
        return {
            "subg": [
                np.asarray([0, 1, 1, 1]),
                np.asarray([0, 0, 0, 0]),
            ]
        }

    def calc_final_score(self, local_model):
        return np.asarray([8.0, 0.0])


def test_scan_calls_upstream_and_restores_rng(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "_upstream_module",
        lambda name: SimpleNamespace(SCAn=_FakeSCAN),
    )
    reference = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    labels = np.asarray([0, 0, 1, 1])
    candidate = reference + 0.1
    before = np.random.get_state()
    result = adapter.scan(reference, labels, candidate, labels, 2, seed=17)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert result.suspicious_mask.tolist() == [True, False, False, False]
    assert result.scores.tolist() == [8.0, 8.0, 0.0, 0.0]


def test_spectre_inverts_upstream_retained_mask(monkeypatch):
    calls = []

    def fake_filter(features, epsilon, dimension):
        calls.append((features.shape, epsilon, dimension))
        mask = np.ones(len(features), dtype=bool)
        mask[-1] = False
        return mask

    monkeypatch.setattr(
        adapter,
        "_upstream_module",
        lambda name: SimpleNamespace(rcov_quantum_filter=fake_filter),
    )
    monkeypatch.setattr(
        adapter, "_spectre_scipy_compatibility", lambda: "test_stub"
    )
    x = np.arange(32, dtype=float).reshape(8, 4)
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    result = adapter.spectre(x, y, {0: 1, 1: 1}, pca_dimension=2)
    assert result.suspicious_mask.tolist() == [False, False, False, True] * 2
    assert np.isnan(result.scores).all()
    assert calls == [((4, 4), 1, 2), ((4, 4), 1, 2)]


@pytest.mark.parametrize(
    "call",
    [
        lambda: adapter.spectral_signatures([[1, 2]], [0.0]),
        lambda: adapter.spectral_signatures([[1, np.nan]], np.asarray([0])),
        lambda: adapter.spectral_signatures([[1, 2]], np.asarray([0]), 100),
        lambda: adapter.scan(
            np.ones((2, 2)), np.asarray([0, 0]), np.ones((2, 3)), np.asarray([0, 0]), 1
        ),
        lambda: adapter.spectre(
            np.ones((4, 2)), np.asarray([0, 0, 1, 1]), {0: 0}, 1
        ),
    ],
)
def test_validation_rejects_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()


def test_real_upstream_spectre_adapter_smoke():
    try:
        adapter._upstream_module("detection_pretrain.spectre")
    except Exception as error:
        pytest.skip(f"BackdoorBench optional dependencies unavailable: {error}")
    rng = np.random.default_rng(4)
    result = adapter.spectre(
        rng.normal(size=(30, 4)), np.zeros(30, dtype=np.int64), {0: 2}, 2
    )
    assert result.suspicious_mask.shape == (30,)
    assert result.suspicious_mask.dtype == np.bool_
    assert result.receipt["scipy_solve_compatibility"] in {
        "native_sym_pos",
        "sym_pos_to_assume_a_pos",
    }
