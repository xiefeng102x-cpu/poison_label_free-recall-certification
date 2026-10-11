from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_p6d_final_observable_adaptive_v2 as v2
import run_p6d_final_observable_adaptive_v3 as v3
import verify_p6d_final_observable_adaptive_v3 as verifier_v3


def test_v3_plain_json_is_recursive_exact_and_strict(tmp_path):
    payload = {"array": np.asarray([[1, 2], [3, 4]], dtype=np.int64),
               "scalar": np.float32(0.25), "nested": [np.bool_(True)]}
    path = tmp_path / "plain.json"
    v3.atomic_json(path, payload)
    assert json.loads(path.read_text()) == {
        "array": [[1, 2], [3, 4]], "scalar": 0.25, "nested": [True]}
    with pytest.raises(ValueError):
        v3.atomic_json(tmp_path / "nan.json", {"bad": np.float64(math.nan)})


def test_v3_full_path_shadow_writes_reads_and_independently_replays_every_family():
    result = v3.full_path_synthetic_shadow_self_check(10_116)
    assert result["status"] == "PASS"
    assert result["json_numpy_roundtrip"] is True
    assert result["test_firewall_denied_unselected"] is True
    assert result["authorized_test_calls"] == {"116": 1, "117": 1, "118": 1}
    assert result["rows"] == {
        "history": 9, "scores": 5940, "observable": 7740,
        "per_seed": 9, "victim": 3, "folds": 9}
    assert result["manifest_artifacts"] == 18
    assert result["independent_replay"] == "valid"


def test_v3_changes_only_delivery_and_fresh_seed_identity():
    old = v2._protocol_payload()
    new = v3._protocol_payload()
    for key in ("dataset_and_split_lock", "threat_model", "final_observable_and_decision",
                "adaptive_attack", "victim_eligibility", "development_stage",
                "confirmation_stage", "detector_outcomes", "stopping_rules",
                "compute_budget", "implementation_lock_policy"):
        assert new[key] == old[key]
    assert v3.DEVELOPMENT_SEEDS == (116, 117, 118)
    assert v3.CONFIRMATION_SEEDS == (119, 120, 121, 122, 123)
    assert set(v3.FORMAL_SEEDS).isdisjoint(set(range(95, 116)))
    protocol = json.loads(v3.PROTOCOL_PATH.read_text())
    assert v3.sha256_file(Path(v3.__file__)) == protocol["source_audit"]["v3_runner_sha256"]
    assert verifier_v3.sha256_file(Path(verifier_v3.__file__)) == protocol["source_audit"]["v3_verifier_sha256"]


def test_v3_formal_entry_refuses_wrong_ack_after_preflight_without_opening_seed(tmp_path):
    args = argparse.Namespace(stage="development", output=tmp_path / "formal",
                              dataset_root=tmp_path / "data",
                              acknowledge_formal_seed_consumption="WRONG")
    with pytest.raises(PermissionError, match="Formal execution requires"):
        v3._runtime.run_development_launcher(args)
    assert not (tmp_path / "formal").exists()
