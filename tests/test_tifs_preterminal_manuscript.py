from __future__ import annotations

import json
import sys
from pathlib import Path


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import verify_tifs_preterminal_manuscript as audit  # noqa: E402


def test_citation_key_parser_handles_lists() -> None:
    assert audit.citation_keys(
        r"x \cite{alpha,beta} y \citep{gamma} z \citet{delta}"
    ) == {
        "alpha",
        "beta",
        "gamma",
        "delta",
    }


def test_preterminal_tifs_receipt_is_preserved_as_history() -> None:
    receipt_path = (
        PAPER14
        / "results/top_journal_program_v1/tifs_main_preterminal_receipt_2026-10-03.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS_PRETERMINAL"
    assert receipt["errors"] == []
    assert receipt["page_count"] <= 13
    assert receipt["supplement_page_count"] >= 1
    assert receipt["theory_proposition_heading_count"] == 8
    assert receipt["configuration_ledger_protocol_count"] == 9
    assert receipt["configuration_ledger_assertion_count"] == 35
    assert receipt["supplement_overfull_box_count"] == 0
    assert receipt["pathmnist_partial_artifacts_read"] is False
    assert receipt["preterminal_visual_qa_status"] == "PASS_PRETERMINAL_VISUAL_QA"
    assert receipt["submission_ready"] is False
    assert audit.sha256(receipt_path) == (
        "831165a7c37ac2fbc31c3a1890b0eb0e68aca5f4c27a027eb0531d2ebf736ccc"
    )


def test_configuration_ledger_text_drift_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    source = audit.SUPPLEMENT_SOURCE.read_text(encoding="utf-8")
    tampered = tmp_path / "supplement_tifs.tex"
    tampered.write_text(
        source.replace("4,500 uniform non-target replacements", "4,501 replacements"),
        encoding="utf-8",
    )
    monkeypatch.setattr(audit, "SUPPLEMENT_SOURCE", tampered)
    receipt = audit.run(write_receipt=False)
    assert receipt["status"] == "FAIL"
    assert any(
        "configuration ledger missing protocol-bound token for PathMNIST poison count"
        in error
        for error in receipt["errors"]
    )


def test_configuration_protocol_drift_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    protocol = json.loads(
        audit.CONFIGURATION_PROTOCOLS["phase2"].read_text(encoding="utf-8")
    )
    protocol["poison_count"] = 61
    tampered = tmp_path / "phase2_confirmation_protocol.json"
    tampered.write_text(json.dumps(protocol), encoding="utf-8")
    protocols = dict(audit.CONFIGURATION_PROTOCOLS)
    protocols["phase2"] = tampered
    monkeypatch.setattr(audit, "CONFIGURATION_PROTOCOLS", protocols)
    receipt = audit.run(write_receipt=False)
    assert receipt["status"] == "FAIL"
    assert any(
        "configuration protocol drift for phase2 poison count" in error
        for error in receipt["errors"]
    )
