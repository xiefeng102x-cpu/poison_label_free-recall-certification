from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


PAPER14 = Path(__file__).resolve().parents[1]
RUNNER_PATH = PAPER14 / "src" / "run_p6c_qml_noise_bridge.py"
SPEC = importlib.util.spec_from_file_location("p6c_bridge_test", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def test_frozen_config_has_three_conditions_and_requested_noise_levels():
    config = json.loads((PAPER14 / "configs/p6c_qml_noise_bridge_v1.json").read_text())
    assert config["seeds"] == [42, 43, 44]
    assert config["representation"] == "XYZ" and config["n_features"] == 15
    assert config["shot_budgets"] == {"primary": 512, "diagnostic_only": 2048}
    assert config["noise_conditions"] == {
        "nominal": {"p1": 0.001, "p2": 0.01},
        "light_sensitivity": {"p1": 0.0005, "p2": 0.005},
        "heavy_sensitivity": {"p1": 0.002, "p2": 0.02},
    }
    assert RUNNER.CONDITIONS == ("exact", "finite_shot_512", "nominal_noise_512", "light_noise_512", "heavy_noise_512")


def test_validation_threshold_respects_empirical_fpr_limit():
    y = np.asarray([0] * 36 + [1] * 36)
    scores = np.arange(72, dtype=float)
    threshold = RUNNER.target_threshold(y, scores, 0.05)
    assert np.mean(scores[y == 0] >= threshold) <= 0.05
    assert np.mean(scores[y == 0] >= np.nextafter(threshold, -np.inf)) > 0.05


def test_shot_seeds_are_reproducible_and_axis_independent():
    one = RUNNER.shot_seed(42, "nominal_noise_512", "X", 20260929)
    assert one == RUNNER.shot_seed(42, "nominal_noise_512", "X", 20260929)
    assert one != RUNNER.shot_seed(42, "nominal_noise_512", "Y", 20260929)


def test_xyz_output_uses_phase_a_reverse_wire_order(monkeypatch):
    def fake_build(weights, basis, shots, p1, p2, device_seed):
        value = {"X": 1.0, "Y": 2.0, "Z": 3.0}[basis]
        return lambda sample: tuple(np.full(shots, value * 10 + q, dtype=float) for q in range(5))

    monkeypatch.setattr(RUNNER, "build_sample_qnode", fake_build)
    config = {"noise_conditions": {"nominal": {"p1": 0.001, "p2": 0.01}}, "random_seed": 5}
    output = RUNNER.measure_xyz(np.zeros((1, 32)), np.zeros((120,)), "nominal_noise_512", 42, config)
    expected = [14, 24, 34, 13, 23, 33, 12, 22, 32, 11, 21, 31, 10, 20, 30]
    np.testing.assert_allclose(output[0], expected)


def test_noise_definition_clearly_freezes_local_channel_placement():
    config = json.loads((PAPER14 / "configs/p6c_qml_noise_bridge_v1.json").read_text())
    definition = config["noise_definition"].lower()
    assert "after every ry and rz" in definition
    assert "after each cnot" in definition
    assert "no correlated two-qubit pauli term" in definition


def test_nominal_2048_diagnostic_uses_fixed_nominal_noise_and_budget(monkeypatch):
    captured = []

    def fake_build(weights, basis, shots, p1, p2, device_seed):
        captured.append((basis, shots, p1, p2))
        return lambda sample: tuple(np.full(shots, 1.0) for _ in range(5))

    monkeypatch.setattr(RUNNER, "build_sample_qnode", fake_build)
    config = {"noise_conditions": {"nominal": {"p1": 0.001, "p2": 0.01}}, "random_seed": 5}
    output = RUNNER.measure_xyz(np.zeros((1, 32)), np.zeros((120,)), "nominal_noise_2048", 42, config, 2048)
    assert output.shape == (1, 15)
    assert captured == [(axis, 2048, 0.001, 0.01) for axis in ("X", "Y", "Z")]
