from __future__ import annotations

import json
from pathlib import Path
import sys


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import verify_tifs_pathmnist_terminal_template as verifier


def test_preoutcome_tifs_terminal_template_is_preserved_after_route_selection() -> None:
    template = json.loads(verifier.TEMPLATE.read_text(encoding="utf-8"))
    assert template["status"] == "FROZEN_BEFORE_CORRECTED_TERMINAL_REPLAY"
    assert template["base_contract_sha256"] == verifier.sha256(
        verifier.BASE_CONTRACT
    )
    assert [item["route"] for item in template["ordered_routes"]] == [
        "R0_INVALID_OR_INCOMPLETE",
        "R1_INSUFFICIENT_ELIGIBLE_VICTIMS",
        "R2_ELIGIBLE_PRIMARY_GATE_FAIL",
        "R3_ELIGIBLE_PRIMARY_GATE_PASS",
    ]
    assert [item["pooling_allowed"] for item in template["ordered_routes"]] == [
        False,
        False,
        True,
        True,
    ]
    assert len(template["source_targets"]) == 4
    assert len(template["global_prohibitions"]) == 6
    assert len(template["mandatory_post_selection_actions"]) == 7
    assert template["performance_endpoints_inspected"] is False

    terminal = json.loads(
        (
            PAPER14
            / "results/top_journal_program_v1/tifs_main_terminal_receipt_2026-10-04.json"
        ).read_text(encoding="utf-8")
    )
    assert terminal["status"] == "PASS_TERMINAL"
    assert terminal["selected_route"] == "R3_ELIGIBLE_PRIMARY_GATE_PASS"
