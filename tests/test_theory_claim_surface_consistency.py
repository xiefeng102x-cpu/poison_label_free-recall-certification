from __future__ import annotations

from pathlib import Path
import sys


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import verify_theory_claim_surface_consistency as verifier  # noqa: E402


def test_theory_claim_surfaces_are_synchronized() -> None:
    result = verifier.run(write_receipt=False)
    assert result["status"] == "PASS"
    assert result["errors"] == []
    assert result["propositions_per_surface"] == 8
    assert result["proposition_surfaces_checked"] == 4
    assert result["assumption_and_nonimplication_tokens_checked"] == 68
    assert result["hostile_audit_interfaces_checked"] == 5
    assert result["forbidden_positive_forms_checked"] == 37
    assert result["pathmnist_artifacts_read"] is False


def test_theory_interface_drift_is_rejected(tmp_path: Path, monkeypatch) -> None:
    text = verifier.EN_APPENDIX.read_text(encoding="utf-8")
    tampered = tmp_path / "theory_appendix.tex"
    tampered.write_text(
        text.replace(
            "Warm-starting and coordinate alignment imply neither",
            "Warm-starting is sufficient and coordinate alignment implies",
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(verifier, "EN_APPENDIX", tampered)
    try:
        verifier.run(write_receipt=False)
    except RuntimeError as exc:
        assert "Warm-starting and coordinate alignment imply neither" in str(exc)
    else:
        raise AssertionError("tampered theory boundary was accepted")
