from pathlib import Path


PAPER14 = Path(__file__).resolve().parents[1]
DOSSIER = (
    PAPER14
    / "results/top_journal_program_v1/TARGET_JOURNAL_SELECTION_DOSSIER_2026-10-03.md"
)


def test_target_dossier_applies_r3_without_authorizing_submission() -> None:
    text = DOSSIER.read_text(encoding="utf-8")
    assert "TIFS_RECOMMENDED_NOT_SUBMISSION_AUTHORIZED" in text
    assert "NOT_SUBMISSION_READY" in text
    assert "does not expand the frozen claim contract" in text
    assert "R3 ROUTE SELECTED" in text
    assert "R0 — invalid or incomplete" in text
    assert "R1 — insufficient eligible victims" in text
    assert "R2 — eligible primary gate fails" in text
    assert "R3 — eligible primary gate passes" in text


def test_target_dossier_uses_official_sources_and_preserves_claim_bounds() -> None:
    text = DOSSIER.read_text(encoding="utf-8")
    required_hosts = {
        "signalprocessingsociety.org": 3,
        "open.ieee.org": 1,
        "www.computer.org": 3,
        "www.nature.com": 5,
        "support.nature.com": 1,
    }
    for host, minimum in required_hosts.items():
        assert text.count(f"https://{host}/") >= minimum
    assert "TIFS is the recommended main target" in text
    assert "universal detector" in text
    assert "do not alter C27" in text or "does not alter C27" in text
    assert "Do not state that Claude wrote scientific text" in text
