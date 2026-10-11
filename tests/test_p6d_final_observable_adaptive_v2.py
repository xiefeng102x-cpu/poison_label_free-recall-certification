from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_p6d_final_observable_adaptive as v1
import run_p6d_final_observable_adaptive_v2 as v2
import verify_p6d_final_observable_adaptive_v2 as verifier_v2


def _fixture(seed=10_108):
    v2.require_synthetic_seed(seed)
    rng = np.random.default_rng(seed)
    return ({"trusted": rng.normal(size=(200, 16)).astype(np.float32),
             "clean": rng.normal(size=(600, 16)).astype(np.float32),
             "poison": rng.normal(size=(60, 16)).astype(np.float32)},
            {"val": {"target_original_indices": np.arange(200) + 2000},
             "train": {"target_original_indices": np.arange(600) + 3000},
             "poison_original_indices": np.arange(60) + 4000})


def test_v1_bug_fixture_is_rejected_by_shared_ordered_csv_writer(tmp_path):
    seed = 10_108
    observables, splits = _fixture(seed)
    partition = v1.build_crossfit_partition(seed)
    legacy_rows = v1._lossless_observable_rows(
        "synthetic", seed, "fixture", splits, observables, partition
    )
    with pytest.raises(ValueError, match="ordered schema"):
        v1.atomic_csv(tmp_path / "v1_bug.csv", legacy_rows)

    fixed_rows = v2._lossless_observable_rows(
        "synthetic", seed, "fixture", splits, observables,
        v2.build_crossfit_partition(seed)
    )
    assert len({tuple(row) for row in fixed_rows}) == 1
    v2.atomic_csv(tmp_path / "v2_fixed.csv", fixed_rows)
    fields, rows = v2._read_csv(tmp_path / "v2_fixed.csv")
    assert "trusted_sample_index" in fields
    assert len(rows) == 860
    assert all(row["trusted_sample_index"] for row in rows
               if row["population"] in ("reference", "calibration"))
    assert all(row["trusted_sample_index"] == "" for row in rows
               if row["population"] in ("clean_candidate", "poison_candidate"))


def test_all_lossless_csv_families_roundtrip_before_formal_writes():
    report = v2.lossless_csv_roundtrip_self_check(10_109)
    assert report["status"] == "PASS"
    assert report["rows"] == {"observable": 860, "scores": 660, "history": 1}


def test_v2_locks_fresh_seed_blocks_and_v1_hashes_without_touching_formal_seeds():
    protocol = v2._protocol_payload()
    assert v2.DEVELOPMENT_SEEDS == (108, 109, 110)
    assert v2.CONFIRMATION_SEEDS == (111, 112, 113, 114, 115)
    assert protocol["seed_audit"]["v1_consumed_seeds"] == [95, 96, 97]
    assert protocol["seed_audit"]["v1_confirmation_seeds_permanently_closed"] == [98, 99, 100, 101, 102]
    assert protocol["seed_audit"]["reserved_for_pathmnist"] == [103, 104, 105, 106, 107]
    assert v2.sha256_file(v2.V1_RUNNER) == protocol["source_audit"]["v1_runner_sha256"]
    assert verifier_v2.sha256_file(verifier_v2.V1_VERIFIER) == protocol["source_audit"]["v1_verifier_sha256"]
    assert v2.sha256_file(Path(v2.__file__)) == protocol["source_audit"]["v2_runner_sha256"]
    assert verifier_v2.sha256_file(Path(verifier_v2.__file__)) == protocol["source_audit"]["v2_verifier_sha256"]
    assert protocol["adaptive_attack"]["training"]["evasion_weight_grid_development_only"] == [0.5, 2.0, 8.0]
    assert set(v2.FORMAL_SEEDS) == set(range(108, 116))


def test_v2_verifier_enforces_single_schema_and_population_index_semantics(tmp_path):
    observables, splits = _fixture()
    rows = v2._lossless_observable_rows(
        "synthetic", 10_108, "fixture", splits, observables,
        v2.build_crossfit_partition(10_108)
    )
    path = tmp_path / "observable_rows.csv"
    v2.atomic_csv(path, rows)
    reread = verifier_v2._runtime.read_csv(path)
    verifier_v2.validate_observable_schema(reread)
    assert len(reread) == 860
    assert all(row["trusted_sample_index"] != "" for row in reread
               if row["population"] in ("reference", "calibration"))
    malformed = [dict(row) for row in reread]
    malformed[-1].pop("trusted_sample_index")
    with pytest.raises(AssertionError, match="ordered schema"):
        verifier_v2.validate_observable_schema(malformed)
