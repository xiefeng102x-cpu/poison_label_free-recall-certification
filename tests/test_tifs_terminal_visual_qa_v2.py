from __future__ import annotations

from pathlib import Path
import sys

import pytest


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import prepare_tifs_terminal_visual_qa_v2 as visual  # noqa: E402


def test_preterminal_self_check_claims_no_visual_inspection() -> None:
    result = visual.preterminal_self_check()
    assert result["status"] == "PASS_PRETERMINAL_TIFS_VISUAL_QA_PREP"
    assert result["terminal_receipts_opened"] is False
    assert result["pathmnist_performance_artifacts_parsed"] is False
    assert result["visual_inspection_claimed"] is False
    assert result["submission_ready"] is False


def test_terminal_mode_requires_explicit_postinspection_token(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="confirmation token"):
        visual.terminal_run(
            render_directory=tmp_path / "rendered",
            confirmation_token="",
            receipt_path=tmp_path / "receipt.json",
        )


def test_contact_sheet_covers_every_rendered_page(tmp_path: Path) -> None:
    pages = []
    for index in range(1, 6):
        path = tmp_path / f"page-{index}.png"
        from PIL import Image

        Image.new("RGB", (120, 180), color=(240, 240, 240)).save(path)
        pages.append(path)
    output = tmp_path / "contact.png"
    visual.contact_sheet(pages, output, columns=4)
    assert output.is_file()
    with __import__("PIL").Image.open(output) as sheet:
        assert sheet.width > 4 * 120
        assert sheet.height > 180
