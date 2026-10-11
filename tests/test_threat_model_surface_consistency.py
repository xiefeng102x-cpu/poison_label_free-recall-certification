from __future__ import annotations

from pathlib import Path
import sys


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import verify_threat_model_surface_consistency as verifier  # noqa: E402


def test_threat_model_surfaces_are_synchronized() -> None:
    result = verifier.run(write_receipt=False)
    assert result["status"] == "PASS"
    assert result["errors"] == []
    assert result["surfaces_checked"] == 5
    assert result["threat_model_and_boundary_tokens_checked"] == 50
    assert result["privilege_ledger_rows_checked"] == 5
    assert result["forbidden_positive_forms_checked"] == 25
    assert len(result["dimensions_guarded"]) == 7
    assert result["pathmnist_artifacts_read"] is False


def test_trusted_root_drift_is_rejected(tmp_path: Path, monkeypatch) -> None:
    text = verifier.EN_MAIN.read_text(encoding="utf-8")
    tampered = tmp_path / "main_en.tex"
    tampered.write_text(
        text.replace(
            "cannot modify $M_0$, the reference, or calibration",
            "may modify $M_0$, the reference, or calibration",
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(verifier, "EN_MAIN", tampered)
    try:
        verifier.run(write_receipt=False)
    except RuntimeError as exc:
        assert "cannot modify $M_0$, the reference, or calibration" in str(exc)
    else:
        raise AssertionError("tampered trusted-root boundary was accepted")
