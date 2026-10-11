from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import re
import shutil
import sys

import pytest


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import integrate_pathmnist_terminal_surfaces_v2 as integrator  # noqa: E402


def _preterminal_text(relative: str) -> str:
    """Project the current terminal surface back to a synthetic clean baseline."""

    text = (PAPER14 / relative).read_text(encoding="utf-8")
    if relative.endswith(".md") or relative.endswith(".tex"):
        text = re.sub(
            r"(?ms)^[^\n]*PATHMNIST_TERMINAL_BLOCK_V2_START[^\n]*\n.*?"
            r"^[^\n]*PATHMNIST_TERMINAL_BLOCK_V2_END[^\n]*\n?",
            "",
            text,
        )
    if relative.endswith("bilingual_manuscript_receipt.json"):
        payload = json.loads(text)
        payload.pop("pathmnist_terminal_integration", None)
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return text


def _prepare_terminal_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str
) -> tuple[dict, dict, dict[str, bytes], Path]:
    contract = json.loads(integrator.CONTRACT.read_text(encoding="utf-8"))
    originals: dict[str, bytes] = {}
    for relative in (
        contract["mutable_integration_targets"] + contract["protected_history"]
    ):
        source = PAPER14 / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative in contract["mutable_integration_targets"]:
            destination.write_text(_preterminal_text(relative), encoding="utf-8")
        else:
            shutil.copy2(source, destination)
        originals[relative] = destination.read_bytes()

    contract["preterminal_mutable_hashes"] = {
        relative: integrator.sha256(tmp_path / relative)
        for relative in contract["mutable_integration_targets"]
    }

    contract_path = tmp_path / integrator.CONTRACT.relative_to(PAPER14)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    capsule = integrator.synthetic_capsule(route)
    capsule_path = tmp_path / integrator.CAPSULE.relative_to(PAPER14)
    capsule_path.parent.mkdir(parents=True, exist_ok=True)
    capsule_path.write_text(
        json.dumps(capsule, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    verification_path = tmp_path / integrator.CAPSULE_VERIFICATION.relative_to(PAPER14)
    verification_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "capsule_sha256": integrator.sha256(capsule_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    receipt_path = tmp_path / "results/integration-receipt.json"
    monkeypatch.setattr(integrator, "ROOT", tmp_path)
    monkeypatch.setattr(integrator, "CONTRACT", contract_path)
    monkeypatch.setattr(integrator, "CAPSULE", capsule_path)
    monkeypatch.setattr(integrator, "CAPSULE_VERIFICATION", verification_path)
    monkeypatch.setattr(
        integrator.final_v2,
        "verify_terminal_control",
        lambda *_args, **_kwargs: {
            "status": "PASS_TERMINAL_CONTROL",
            "errors": [],
            "selected_route": route,
        },
    )
    return contract, capsule, originals, receipt_path


@pytest.mark.parametrize("route", sorted(integrator.TERMINAL_ROUTES))
def test_all_routes_render_all_surfaces_without_crossing_information_budget(
    route: str,
) -> None:
    contract = json.loads(integrator.CONTRACT.read_text(encoding="utf-8"))
    capsule = integrator.synthetic_capsule(route)
    capsule_sha = "a" * 64
    rendered = {}
    for relative in contract["mutable_integration_targets"]:
        source = _preterminal_text(relative)
        rendered[relative] = integrator.render_target(
            relative, source, capsule, capsule_sha
        )
        assert rendered[relative] != source
        assert capsule["route_marker"] in rendered[relative]
        assert f"PATHMNIST_CLAIM_CAPSULE_SHA256={capsule_sha}" in rendered[relative]

    for relative in contract["terminal_claim_binding_surfaces"]["english_sentence_targets"]:
        assert capsule["main_text_en"] in rendered[relative]
    for relative in contract["terminal_claim_binding_surfaces"]["chinese_sentence_targets"]:
        assert capsule["main_text_zh"] in rendered[relative]
    for relative in contract["terminal_claim_binding_surfaces"]["metric_targets"]:
        assert ("PATHMNIST_METRIC=" in rendered[relative]) is (
            capsule["detector_summaries"] is not None
        )

    matrix = list(
        csv.DictReader(io.StringIO(rendered["results/top_journal_program_v1/FINAL_CLAIM_MATRIX.csv"]))
    )
    c27 = [row for row in matrix if row["claim_id"] == "C27"]
    assert len(c27) == 1
    assert c27[0]["decision"] == capsule["claim_decision"]


def test_preterminal_self_check_is_three_route_and_write_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_terminal_sandbox(
        tmp_path, monkeypatch, "R1_INSUFFICIENT_ELIGIBLE_VICTIMS"
    )
    result = integrator.preterminal_self_check()
    assert result["status"] == "PASS_PRETERMINAL_SURFACE_INTEGRATOR_PREP"
    assert result["routes_simulated"] == 3
    assert result["targets_per_route"] == 19
    assert result["synthetic_renders_checked"] == 57
    assert result["terminal_receipts_opened"] is False
    assert result["pathmnist_performance_artifacts_parsed"] is False
    assert result["files_modified"] is False


def test_integrator_rejects_duplicate_terminal_block() -> None:
    capsule = integrator.synthetic_capsule("R1_INSUFFICIENT_ELIGIBLE_VICTIMS")
    with pytest.raises(RuntimeError, match="already exists"):
        integrator.render_target(
            "results/top_journal_program_v1/STATUS.md",
            f"<!-- {integrator.BLOCK_START} -->\n",
            capsule,
            "b" * 64,
        )


def test_terminal_run_atomically_integrates_all_surfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    route = "R3_ELIGIBLE_PRIMARY_GATE_PASS"
    contract, capsule, originals, receipt_path = _prepare_terminal_sandbox(
        tmp_path, monkeypatch, route
    )

    receipt = integrator.terminal_run(receipt_path=receipt_path)

    assert receipt["status"] == "PASS_TERMINAL_SURFACE_INTEGRATION"
    assert receipt["selected_route"] == route
    assert receipt["changed_target_count"] == 19
    assert set(receipt["changed_targets"]) == set(
        contract["mutable_integration_targets"]
    )
    assert receipt_path.is_file()
    for relative in contract["mutable_integration_targets"]:
        target = tmp_path / relative
        assert target.read_bytes() != originals[relative]
        text = target.read_text(encoding="utf-8")
        assert capsule["route_marker"] in text
    for relative in contract["protected_history"]:
        assert (tmp_path / relative).read_bytes() == originals[relative]


def test_terminal_run_rolls_back_every_replaced_surface_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, _capsule, originals, receipt_path = _prepare_terminal_sandbox(
        tmp_path, monkeypatch, "R2_ELIGIBLE_PRIMARY_GATE_FAIL"
    )
    real_replace = integrator.os.replace
    calls = 0

    def fail_during_replacement(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("synthetic replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(integrator.os, "replace", fail_during_replacement)
    with pytest.raises(OSError, match="synthetic replacement failure"):
        integrator.terminal_run(receipt_path=receipt_path)

    assert not receipt_path.exists()
    for relative in (
        contract["mutable_integration_targets"] + contract["protected_history"]
    ):
        assert (tmp_path / relative).read_bytes() == originals[relative]


def test_terminal_run_rolls_back_surfaces_when_receipt_publication_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, _capsule, originals, receipt_path = _prepare_terminal_sandbox(
        tmp_path, monkeypatch, "R3_ELIGIBLE_PRIMARY_GATE_PASS"
    )

    def fail_receipt_publication(path: Path, _payload: dict) -> None:
        path.with_name(path.name + ".tmp").write_text("partial", encoding="utf-8")
        raise OSError("synthetic receipt publication failure")

    monkeypatch.setattr(
        integrator.final_v2, "atomic_json", fail_receipt_publication
    )
    with pytest.raises(OSError, match="synthetic receipt publication failure"):
        integrator.terminal_run(receipt_path=receipt_path)

    assert not receipt_path.exists()
    assert not receipt_path.with_name(receipt_path.name + ".tmp").exists()
    for relative in (
        contract["mutable_integration_targets"] + contract["protected_history"]
    ):
        assert (tmp_path / relative).read_bytes() == originals[relative]


def test_terminal_run_never_overwrites_an_existing_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, _capsule, originals, receipt_path = _prepare_terminal_sandbox(
        tmp_path, monkeypatch, "R1_INSUFFICIENT_ELIGIBLE_VICTIMS"
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text("historical receipt\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="receipt already exists"):
        integrator.terminal_run(receipt_path=receipt_path)

    assert receipt_path.read_text(encoding="utf-8") == "historical receipt\n"
    for relative in (
        contract["mutable_integration_targets"] + contract["protected_history"]
    ):
        assert (tmp_path / relative).read_bytes() == originals[relative]
