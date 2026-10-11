from __future__ import annotations

import csv
from collections import Counter
import hashlib
from itertools import combinations
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parents[1]
CERT = PAPER / "auditability_certificate_v1"
sys.path.insert(0, str(CERT / "analysis"))

import issue_protocol_correct_certificates as issuer
import prepare_protocol_correct_observations as registrar
from protocol_correct_common import opaque_token


def test_issuer_source_has_no_private_identifiers() -> None:
    text = Path(issuer.__file__).read_text(encoding="utf-8")
    for prohibited in ("private_truth", "is_poison", "pool_role", "selected_clean", "selected_poison"):
        assert prohibited not in text


def test_issuer_schema_is_minimal() -> None:
    assert issuer.INPUT_FIELDS == [
        "protocol", "trial_token", "k", "conformal_rank", "candidate_count", "flagged_count"
    ]
    assert not {"sample_id", "victim_id", "detector", "fold"}.intersection(issuer.INPUT_FIELDS)


def test_issuer_accepts_only_hash_sealed_allowlisted_input(tmp_path: Path) -> None:
    input_dir = tmp_path / "public"
    output_dir = tmp_path / "sealed"
    input_dir.mkdir()
    observation = input_dir / "issuer_observations.csv"
    with observation.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=issuer.INPUT_FIELDS)
        writer.writeheader()
        writer.writerow({
            "protocol": "pathmnist_blind",
            "trial_token": "0" * 64,
            "k": 120,
            "conformal_rank": 115,
            "candidate_count": 8000,
            "flagged_count": 1000,
        })
    manifest = {
        "status": "ISSUER_OBSERVATIONS_SEALED",
        "protocol": "pathmnist_blind",
        "rows": 1,
        "key_commitment_sha256": "1" * 64,
        "config_sha256": issuer.sha256(issuer.CONFIG),
        "issuer_observations_sha256": issuer.sha256(observation),
        "allowed_schema": issuer.INPUT_FIELDS,
    }
    (input_dir / "issuer_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = issuer.issue(input_dir, output_dir)
    assert report["status"] == "CERTIFICATE_DECISIONS_HASH_SEALED"
    assert report["input_rows"] == 1
    assert report["decision_rows"] == 2


def test_trusted_pool_is_reproducible_and_non_exhaustive() -> None:
    frame = pd.DataFrame({
        "sample_id": list(range(8)),
        "is_poison": [0, 0, 0, 0, 0, 1, 1, 1],
    })
    key = bytes(range(32))
    first = registrar.clean_pool_for_fold(key, "test", frame, 3, "victim_seed7", 0)
    second = registrar.clean_pool_for_fold(key, "test", frame, 3, "victim_seed7", 0)
    assert first.tolist() == second.tolist()
    assert len(first) == 3
    assert set(first).issubset({0, 1, 2, 3, 4})
    assert set(range(8)) - set(first)


def test_two_stage_draw_is_uniform_over_complete_audit_universe() -> None:
    population_size, pool_size, calibration_size = 7, 5, 3
    counts: Counter[tuple[int, ...]] = Counter()
    for pool in combinations(range(population_size), pool_size):
        for selected in combinations(pool, calibration_size):
            counts[selected] += 1

    assert set(counts) == set(combinations(range(population_size), calibration_size))
    assert set(counts.values()) == {
        math.comb(population_size - calibration_size, pool_size - calibration_size)
    }
    assert (
        math.comb(population_size - calibration_size, pool_size - calibration_size)
        * math.comb(population_size, calibration_size)
        == math.comb(population_size, pool_size) * math.comb(pool_size, calibration_size)
    )

    scores = np.arange(8, dtype=float)
    selected = np.array([0, 2, 4], dtype=np.int64)
    jitter = np.linspace(0.0, 0.7, 8)
    _, candidate_count, _, _ = registrar.threshold_statistics(
        scores, selected, jitter, alpha=0.5
    )
    assert candidate_count == len(scores) - len(selected)


def test_opaque_tokens_are_domain_separated() -> None:
    key = hashlib.sha256(b"unit-test").digest()
    assert opaque_token(key, "path", 1, 2) != opaque_token(key, "redteam", 1, 2)
