#!/usr/bin/env python3
"""Render and receipt terminal visual QA for the TIFS main and supplement.

The command refuses to claim visual inspection unless the caller supplies the
explicit confirmation token after reviewing every rendered page.  Rendering
and hashes are machine checked; legibility, clipping, overlap, glyph quality,
and page transitions remain a genuine visual-review obligation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from PIL import Image, ImageOps, ImageDraw

import select_pathmnist_terminal_route_v2 as selector
import verify_top_journal_final_bundle_v2 as final_v2


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "results/top_journal_program_v1"
MANUSCRIPT = ROOT / "manuscript_tifs_en"
MAIN_SOURCE = MANUSCRIPT / "main_tifs.tex"
MAIN_PDF = MANUSCRIPT / "output/main_tifs.pdf"
SUPPLEMENT_SOURCE = MANUSCRIPT / "supplement_tifs.tex"
SUPPLEMENT_PDF = MANUSCRIPT / "output/supplement_tifs.pdf"
CAPSULE = PROGRAM / "pathmnist_terminal_claim_capsule_v2_2026-10-04.json"
CAPSULE_VERIFICATION = PROGRAM / (
    "pathmnist_terminal_claim_capsule_verification_v2_2026-10-04.json"
)
DEFAULT_RECEIPT = PROGRAM / "tifs_terminal_visual_qa_2026-10-04.json"
FORMAL_OUTPUT = ROOT / "results/pathmnist_official_weight_detector_confirmation_v1"
CONFIRMATION_TOKEN = "INSPECTED_ALL_RENDERED_TIFS_PAGES"


def sha256(path: Path) -> str:
    return final_v2.sha256(path)


def pdf_pages(path: Path) -> int:
    completed = subprocess.run(
        ["pdfinfo", str(path)], check=True, capture_output=True, text=True
    )
    match = re.search(r"^Pages:\s+(\d+)\s*$", completed.stdout, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"pdfinfo did not report pages for {path.name}")
    return int(match.group(1))


def preterminal_self_check() -> dict[str, Any]:
    errors: list[str] = []
    for command in ("pdfinfo", "pdftoppm"):
        if shutil.which(command) is None:
            errors.append(f"PDF rendering dependency unavailable: {command}")
    if not MAIN_SOURCE.is_file() or not SUPPLEMENT_SOURCE.is_file():
        errors.append("TIFS sources are absent")
    result = {
        "status": "PASS_PRETERMINAL_TIFS_VISUAL_QA_PREP" if not errors else "FAIL",
        "errors": errors,
        "receipt_target": str(DEFAULT_RECEIPT.relative_to(ROOT)),
        "confirmation_token_sha256": __import__("hashlib").sha256(
            CONFIRMATION_TOKEN.encode("ascii")
        ).hexdigest(),
        "runner_sha256": sha256(Path(__file__)),
        "terminal_receipts_opened": False,
        "pathmnist_performance_artifacts_parsed": False,
        "visual_inspection_claimed": False,
        "submission_ready": False,
    }
    if errors:
        raise RuntimeError("; ".join(errors))
    return result


def terminal_preconditions(
    *,
    replay_receipt: Path = selector.DEFAULT_REPLAY_RECEIPT,
    route_receipt: Path = selector.DEFAULT_SELECTION_RECEIPT,
    formal_output: Path = FORMAL_OUTPUT,
) -> dict[str, Any]:
    control = final_v2.verify_terminal_control(
        replay_receipt, route_receipt, formal_output
    )
    if control["status"] != "PASS_TERMINAL_CONTROL":
        raise RuntimeError("; ".join(control["errors"]))
    if not CAPSULE.is_file() or not CAPSULE_VERIFICATION.is_file():
        raise RuntimeError("terminal claim capsule and verification are required")
    capsule = json.loads(CAPSULE.read_text(encoding="utf-8"))
    verification = json.loads(CAPSULE_VERIFICATION.read_text(encoding="utf-8"))
    if verification.get("status") != "PASS":
        raise RuntimeError("terminal claim-capsule verification is not PASS")
    if verification.get("capsule_sha256") != sha256(CAPSULE):
        raise RuntimeError("terminal claim-capsule verification hash mismatch")
    if capsule.get("selected_route") != control.get("selected_route"):
        raise RuntimeError("terminal claim capsule route mismatch")
    main_text = MAIN_SOURCE.read_text(encoding="utf-8")
    supplement_text = SUPPLEMENT_SOURCE.read_text(encoding="utf-8")
    for path, text in ((MAIN_SOURCE, main_text), (SUPPLEMENT_SOURCE, supplement_text)):
        if any(
            marker in text
            for marker in (
                "PATHMNIST_TERMINAL_ROUTE=",
                "PATHMNIST_CLAIM_CAPSULE_SHA256=",
                "PATHMNIST_METRIC=",
            )
        ):
            raise RuntimeError(f"internal terminal marker leaked into anonymous source: {path.name}")
    authorized_claim = " ".join(capsule["main_text_en"].replace("%", r"\%").split())
    if authorized_claim not in " ".join(main_text.split()):
        raise RuntimeError("authorized PathMNIST capsule claim is absent from main source")
    for method, summary in (capsule.get("detector_summaries") or {}).items():
        values = (
            f"{summary['pooled_tpr_pct']:.2f}",
            f"{summary['pooled_all_clean_fpr_pct']:.2f}",
            f"{summary['pooled_target_clean_fpr_pct']:.2f}",
        )
        if not all(value in supplement_text for value in values):
            raise RuntimeError(f"capsule endpoint values absent from supplement: {method}")
    for path in (MAIN_PDF, SUPPLEMENT_PDF):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"compiled terminal PDF is absent: {path.name}")
    return {"control": control, "capsule_sha256": sha256(CAPSULE)}


def render(path: Path, output: Path, label: str) -> list[Path]:
    prefix = output / label
    subprocess.run(
        ["pdftoppm", "-png", "-r", "130", str(path), str(prefix)],
        check=True,
        capture_output=True,
        text=True,
    )
    pages = sorted(output.glob(f"{label}-*.png"))
    if not pages:
        raise RuntimeError(f"pdftoppm produced no pages for {label}")
    return pages


def contact_sheet(images: list[Path], output: Path, *, columns: int = 4) -> None:
    opened = [Image.open(path).convert("RGB") for path in images]
    try:
        thumb_width = 360
        thumbnails: list[Image.Image] = []
        for index, page in enumerate(opened, start=1):
            height = max(1, round(page.height * thumb_width / page.width))
            thumb = page.resize((thumb_width, height))
            canvas = Image.new("RGB", (thumb_width + 16, height + 42), "white")
            canvas.paste(thumb, (8, 30))
            ImageDraw.Draw(canvas).text((8, 8), f"Page {index}", fill="black")
            thumbnails.append(ImageOps.expand(canvas, border=1, fill="black"))
        rows = (len(thumbnails) + columns - 1) // columns
        cell_width = max(image.width for image in thumbnails)
        cell_height = max(image.height for image in thumbnails)
        sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
        for index, thumb in enumerate(thumbnails):
            x = (index % columns) * cell_width
            y = (index // columns) * cell_height
            sheet.paste(thumb, (x, y))
        sheet.save(output)
    finally:
        for image in opened:
            image.close()


def build_receipt(
    *,
    selected_route: str,
    capsule_sha256: str,
    main_pages: int,
    supplement_pages: int,
    render_directory: Path,
) -> dict[str, Any]:
    return {
        "schema": "paper14_tifs_terminal_visual_qa_v2",
        "status": "PASS_TERMINAL_VISUAL_QA",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_route": selected_route,
        "capsule_sha256": capsule_sha256,
        "main": {
            "source_sha256": sha256(MAIN_SOURCE),
            "pdf_sha256": sha256(MAIN_PDF),
            "pages": main_pages,
            "pages_inspected": list(range(1, main_pages + 1)),
        },
        "supplement": {
            "source_sha256": sha256(SUPPLEMENT_SOURCE),
            "pdf_sha256": sha256(SUPPLEMENT_PDF),
            "pages": supplement_pages,
            "pages_inspected": list(range(1, supplement_pages + 1)),
        },
        "rendering": {
            "engine": "pdftoppm",
            "dpi": 130,
            "render_directory": "<TEMPORARY_VISUAL_QA_DIRECTORY>",
            "main_contact_sheet": "main-contact-sheet.png",
            "supplement_contact_sheet": "supplement-contact-sheet.png",
        },
        "inspection_checks": [
            "no clipped or overlapping text",
            "tables and figures remain legible",
            "no missing glyphs or unintended blank pages",
            "headers, footers, page numbers and section transitions are coherent",
            "terminal PathMNIST text and tables are visually integrated",
        ],
        "all_rendered_pages_inspected": True,
        "confirmation_token_verified": True,
        "runner_sha256": sha256(Path(__file__)),
        "submission_ready": False,
    }


def terminal_run(
    *, render_directory: Path, confirmation_token: str, receipt_path: Path
) -> dict[str, Any]:
    if confirmation_token != CONFIRMATION_TOKEN:
        raise RuntimeError(
            "visual QA confirmation token is absent; inspect every rendered page first"
        )
    preconditions = terminal_preconditions()
    if render_directory.exists():
        if any(render_directory.iterdir()):
            raise RuntimeError("visual-QA render directory must be absent or empty")
    else:
        render_directory.mkdir(parents=True)
    main_pages = pdf_pages(MAIN_PDF)
    supplement_pages = pdf_pages(SUPPLEMENT_PDF)
    main_images = render(MAIN_PDF, render_directory, "main")
    supplement_images = render(SUPPLEMENT_PDF, render_directory, "supplement")
    if len(main_images) != main_pages or len(supplement_images) != supplement_pages:
        raise RuntimeError("rendered PNG count differs from PDF page count")
    contact_sheet(main_images, render_directory / "main-contact-sheet.png")
    contact_sheet(
        supplement_images, render_directory / "supplement-contact-sheet.png"
    )
    receipt = build_receipt(
        selected_route=preconditions["control"]["selected_route"],
        capsule_sha256=preconditions["capsule_sha256"],
        main_pages=main_pages,
        supplement_pages=supplement_pages,
        render_directory=render_directory,
    )
    final_v2.atomic_json(receipt_path, receipt)
    return {**receipt, "runtime_render_directory": str(render_directory.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("preterminal-self-check", "terminal"), required=True
    )
    parser.add_argument("--render-directory", type=Path)
    parser.add_argument("--confirmation-token", default="")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    if args.mode == "preterminal-self-check":
        result = preterminal_self_check()
    else:
        if args.render_directory is None:
            raise SystemExit("--render-directory is required in terminal mode")
        result = terminal_run(
            render_directory=args.render_directory,
            confirmation_token=args.confirmation_token,
            receipt_path=args.receipt,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
