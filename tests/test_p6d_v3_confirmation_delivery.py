from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import verify_p6d_v3_confirmation_delivery as forensic


def test_stage_aware_seed_blocks() -> None:
    assert forensic._expected_seeds(
        {"run_signature": {"stage": "development", "seeds": [116, 117, 118]}}
    ) == (116, 117, 118)
    assert forensic._expected_seeds(
        {"run_signature": {"stage": "confirmation", "seeds": [119, 120, 121, 122, 123]}}
    ) == (119, 120, 121, 122, 123)


def test_stage_seed_mismatch_is_rejected() -> None:
    with pytest.raises(AssertionError, match="seed block"):
        forensic._expected_seeds(
            {"run_signature": {"stage": "confirmation", "seeds": [116, 117, 118]}}
        )


def test_completed_formal_confirmation_forensic_replay() -> None:
    output = ROOT / "results/p6d_final_observable_adaptive_v3/confirmation"
    manifest = output / "manifest.json"
    if not manifest.is_file() or json.loads(manifest.read_text()).get("status") != "complete":
        pytest.skip("Completed formal P6-D v3 confirmation bundle is not present")
    result = forensic.verify(output)
    assert result["status"] == "PASS_STAGE_AWARE_FORENSIC_REPLAY"
    assert result["formal_bundle_unchanged"] is True
    assert result["delivery_replay"]["seeds"] == [119, 120, 121, 122, 123]
