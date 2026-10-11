from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "auditability_certificate_v1/analysis"
sys.path.insert(0, str(ANALYSIS))

import prospective_protocol_v2_common as frozen  # noqa: E402


def test_frozen_protocol_surface() -> None:
    cfg = json.loads(frozen.CONFIG.read_text(encoding="utf-8"))
    assert tuple(cfg["formal_seeds"]) == frozen.SEEDS
    assert {int(k): value for k, value in cfg["weight_map"].items()} == frozen.WEIGHT_MAP
    assert cfg["dataset_sha256"] == frozen.EXPECTED_DATASET_SHA256
    assert cfg["pathmnist_blind"]["delta_per_cell"] == 0.005
    assert cfg["issuer_public_fields"] == [
        "protocol", "trial_token", "k", "conformal_rank", "candidate_count", "flagged_count"
    ]


def test_issuer_has_no_private_identifiers() -> None:
    banned = ("private_truth", "is_poison", "pool_role", "selected_clean", "selected_poison")
    for path in (frozen.ISSUER_WRAPPER, frozen.BASE_ISSUER):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text


def test_prospective_freeze_is_complete() -> None:
    freeze = frozen.verify_freeze(require_dataset=False)
    assert freeze["outcomes_opened_at_freeze"] is False
    assert freeze["key_revealed_at_freeze"] is False
    assert freeze["formal_seeds"] == list(frozen.SEEDS)
