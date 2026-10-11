#!/usr/bin/env python3
"""Verify the post-route TIFS main paper and supplement without rewriting history."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import select_pathmnist_terminal_route_v2 as selector
import verify_top_journal_final_bundle_v2 as final_v2


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "results/top_journal_program_v1"
MANUSCRIPT = ROOT / "manuscript_tifs_en"
SOURCE = MANUSCRIPT / "main_tifs.tex"
PDF = MANUSCRIPT / "output/main_tifs.pdf"
LOG = MANUSCRIPT / "output/main_tifs.log"
BBL = MANUSCRIPT / "output/main_tifs.bbl"
SUPPLEMENT_SOURCE = MANUSCRIPT / "supplement_tifs.tex"
SUPPLEMENT_PDF = MANUSCRIPT / "output/supplement_tifs.pdf"
SUPPLEMENT_LOG = MANUSCRIPT / "output/supplement_tifs.log"
SUPPLEMENT_BBL = MANUSCRIPT / "output/supplement_tifs.bbl"
THEORY_APPENDIX = ROOT / "manuscript_en/theory_appendix.tex"
PRETERMINAL_RECEIPT = PROGRAM / "tifs_main_preterminal_receipt_2026-10-03.json"
CAPSULE = PROGRAM / "pathmnist_terminal_claim_capsule_v2_2026-10-04.json"
CAPSULE_VERIFICATION = PROGRAM / (
    "pathmnist_terminal_claim_capsule_verification_v2_2026-10-04.json"
)
TERMINAL_VISUAL_QA = PROGRAM / "tifs_terminal_visual_qa_2026-10-04.json"
TERMINAL_RECEIPT = PROGRAM / "tifs_main_terminal_receipt_2026-10-04.json"
PREPARATION_RECEIPT = PROGRAM / "tifs_terminal_verifier_preparation_2026-10-04.json"
FORMAL_OUTPUT = ROOT / "results/pathmnist_official_weight_detector_confirmation_v1"

CONFIGURATION_TOKENS = [
    "dev. 52--61; conf. 62--71",
    "120 / 80 validation-target rows",
    "C2 Blended and SIG & conf. 52--61",
    "60 poisons; Blended",
    r"Blended $\alpha=0.2$",
    "frequency 6",
    "Classical MLP, 80 epochs",
    "applies five adaptive epochs",
    "BadNets: 60 rows",
    "CNN classifier, 30 epochs",
    "0.1 poison ratio per batch",
    "mask density 0.032",
    "C3 white-box QML & conf. 52--61",
    "evasion weight 10",
    "eight attack epochs",
    "learning rate 0.01",
    "pilot 42,46,51; conf. 52--61",
    "pilot 42,46,51; conf. 52--61",
    "freezes ten clean fine-tuning",
    "eight poisoned-adaptation epochs",
    "dev. 72--73; conf. 74--78",
    "dev. 72--73; conf. 74--78",
    "600 in-place non-target replacements",
    "target class 6",
    "at most ten epochs",
    "Retrain MLP from scratch for 80 epochs",
    "103--107",
    "89,996 official training rows",
    "4,500 uniform non-target replacements",
    "target class 8",
    "five full-data epochs",
    "batch 128",
    r"Adam $3\times10^{-4}$",
    "512 / 512 target-class validation rows",
    "fixed-seed independent replay",
]


def sha256(path: Path) -> str:
    return final_v2.sha256(path)


def normalized(text: str) -> str:
    return " ".join(text.split())


def pdf_page_count(path: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(path)], check=True, capture_output=True, text=True
    )
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("pdfinfo did not report a page count")
    return int(match.group(1))


def citation_keys(source: str) -> set[str]:
    keys: set[str] = set()
    for payload in re.findall(r"\\cite(?:p|t)?\{([^}]+)\}", source):
        keys.update(key.strip() for key in payload.split(",") if key.strip())
    return keys


def frozen_protocol_hashes() -> dict[str, str]:
    receipt = json.loads(PRETERMINAL_RECEIPT.read_text(encoding="utf-8"))
    return {
        relative: digest
        for relative, digest in receipt["hashes"].items()
        if relative.startswith("results/top_journal_program_v1/")
        and "protocol" in Path(relative).name
        and relative.endswith(".json")
    }


def preterminal_self_check(*, write_receipt: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    supplement = SUPPLEMENT_SOURCE.read_text(encoding="utf-8")
    for token in CONFIGURATION_TOKENS:
        if token not in supplement:
            errors.append(f"current configuration ledger lacks token: {token}")
    protocols = frozen_protocol_hashes()
    for relative, expected in protocols.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"frozen TIFS protocol hash drift: {relative}")
    result = {
        "status": "PASS_PRETERMINAL_TIFS_TERMINAL_VERIFIER_PREP"
        if not errors
        else "FAIL",
        "errors": errors,
        "configuration_tokens_checked": len(CONFIGURATION_TOKENS),
        "frozen_protocol_hashes_checked": len(protocols),
        "verifier_sha256": sha256(Path(__file__)),
        "pathmnist_terminal_receipts_opened": False,
        "pathmnist_performance_artifacts_parsed": False,
        "submission_ready": False,
    }
    if write_receipt:
        final_v2.atomic_json(PREPARATION_RECEIPT, result)
    if errors:
        raise RuntimeError("; ".join(errors))
    return result


def terminal_route_surface_errors(
    main: str,
    supplement: str,
    capsule: dict[str, Any],
    capsule_sha: str,
) -> list[str]:
    errors: list[str] = []
    # The route and capsule hashes remain in machine-readable receipts.  They
    # must not leak into the anonymous submission source as internal workflow
    # labels.  The manuscript is instead checked against the authorized prose
    # and numerical endpoint values carried by the verified capsule.
    for label, text in (("main", main), ("supplement", supplement)):
        for marker in (
            "PATHMNIST_TERMINAL_ROUTE=",
            "PATHMNIST_CLAIM_CAPSULE_SHA256=",
            "PATHMNIST_METRIC=",
        ):
            if marker in text:
                errors.append(f"internal terminal marker leaked into TIFS {label}: {marker}")
    authorized_claim = capsule["main_text_en"].replace("%", r"\%")
    if normalized(authorized_claim) not in normalized(main):
        errors.append("capsule English claim is absent from TIFS main")
    summaries = capsule.get("detector_summaries")
    if summaries is None:
        if "Frozen PathMNIST detector endpoints" in supplement:
            errors.append("ineligible route exposes pooled PathMNIST endpoints")
    else:
        for method, summary in summaries.items():
            values = (
                f"{summary['pooled_tpr_pct']:.2f}",
                f"{summary['pooled_all_clean_fpr_pct']:.2f}",
                f"{summary['pooled_target_clean_fpr_pct']:.2f}",
            )
            if not all(value in supplement for value in values):
                errors.append(f"TIFS supplement lacks capsule endpoint values: {method}")
    for token in (
        "% PRETERMINAL DRAFT: PathMNIST 103--107 endpoints remain sealed.",
        "% PRETERMINAL SUPPLEMENT: PathMNIST 103--107 endpoints remain sealed.",
    ):
        if token in main or token in supplement:
            errors.append(f"stale preterminal TIFS marker remains: {token}")
    return errors


def terminal_run(
    *,
    replay_receipt: Path = selector.DEFAULT_REPLAY_RECEIPT,
    route_receipt: Path = selector.DEFAULT_SELECTION_RECEIPT,
    formal_output: Path = FORMAL_OUTPUT,
    receipt_path: Path = TERMINAL_RECEIPT,
) -> dict[str, Any]:
    control = final_v2.verify_terminal_control(
        replay_receipt, route_receipt, formal_output
    )
    if control["status"] != "PASS_TERMINAL_CONTROL":
        raise RuntimeError("; ".join(control["errors"]))
    required = [
        SOURCE,
        PDF,
        LOG,
        BBL,
        SUPPLEMENT_SOURCE,
        SUPPLEMENT_PDF,
        SUPPLEMENT_LOG,
        SUPPLEMENT_BBL,
        THEORY_APPENDIX,
        CAPSULE,
        CAPSULE_VERIFICATION,
        TERMINAL_VISUAL_QA,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"terminal TIFS files missing: {missing}")

    main = SOURCE.read_text(encoding="utf-8")
    supplement = SUPPLEMENT_SOURCE.read_text(encoding="utf-8")
    capsule = json.loads(CAPSULE.read_text(encoding="utf-8"))
    capsule_verification = json.loads(CAPSULE_VERIFICATION.read_text(encoding="utf-8"))
    visual = json.loads(TERMINAL_VISUAL_QA.read_text(encoding="utf-8"))
    errors = terminal_route_surface_errors(main, supplement, capsule, sha256(CAPSULE))
    if capsule_verification.get("status") != "PASS":
        errors.append("independent terminal capsule verification is not PASS")
    if capsule_verification.get("capsule_sha256") != sha256(CAPSULE):
        errors.append("terminal capsule verification hash mismatch")

    for token in CONFIGURATION_TOKENS:
        if token not in supplement:
            errors.append(f"terminal configuration ledger lacks token: {token}")
    protocols = frozen_protocol_hashes()
    for relative, expected in protocols.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"frozen TIFS protocol hash drift: {relative}")

    general_tokens = [
        r"\documentclass[journal]{IEEEtran}",
        "end-to-end guarantee",
        "candidate-batch screen",
        "42.67--48.33\\%",
        "44.17\\%",
        "four of five",
        "99,000 decisions",
        "52.67--60.71\\%",
        "71.3\\%",
    ]
    for token in general_tokens:
        if token not in main:
            errors.append(f"terminal TIFS main lacks bounded-claim token: {token}")
    for token in (
        "universal poison detector",
        "architecture-invariant quantum signature",
        "state-of-the-art detector performance",
    ):
        if token.lower() in main.lower():
            errors.append(f"forbidden terminal TIFS overclaim: {token}")

    main_pages = pdf_page_count(PDF)
    supplement_pages = pdf_page_count(SUPPLEMENT_PDF)
    if main_pages > 13:
        errors.append(f"terminal TIFS page limit exceeded: {main_pages}")
    if visual.get("status") != "PASS_TERMINAL_VISUAL_QA":
        errors.append("terminal TIFS visual QA is not PASS")
    for label, source_path, pdf_path, pages in (
        ("main", SOURCE, PDF, main_pages),
        ("supplement", SUPPLEMENT_SOURCE, SUPPLEMENT_PDF, supplement_pages),
    ):
        record = visual.get(label, {})
        if record.get("source_sha256") != sha256(source_path):
            errors.append(f"terminal visual source hash drift: {label}")
        if record.get("pdf_sha256") != sha256(pdf_path):
            errors.append(f"terminal visual PDF hash drift: {label}")
        if record.get("pages") != pages:
            errors.append(f"terminal visual page count drift: {label}")
        if record.get("pages_inspected") != list(range(1, pages + 1)):
            errors.append(f"terminal visual page coverage incomplete: {label}")

    unresolved_patterns = [
        r"Citation [`'][^`']+['`] on page .* undefined",
        r"Reference [`'][^`']+['`] on page .* undefined",
        r"There were undefined references",
        r"There were undefined citations",
    ]
    log_counts: dict[str, int] = {}
    for label, path in (("main", LOG), ("supplement", SUPPLEMENT_LOG)):
        text = path.read_text(encoding="utf-8", errors="replace")
        overfull = len(re.findall(r"Overfull \\[hv]box", text))
        unresolved = sum(
            len(re.findall(pattern, text, flags=re.IGNORECASE))
            for pattern in unresolved_patterns
        )
        log_counts[f"{label}_overfull"] = overfull
        log_counts[f"{label}_unresolved"] = unresolved
        if overfull or unresolved:
            errors.append(f"terminal {label} log has layout/reference errors")

    for label, source_text, bbl_path in (
        ("main", main, BBL),
        ("supplement", supplement + "\n" + THEORY_APPENDIX.read_text(encoding="utf-8"), SUPPLEMENT_BBL),
    ):
        cited = citation_keys(source_text)
        emitted = set(
            re.findall(
                r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}",
                bbl_path.read_text(encoding="utf-8", errors="replace"),
            )
        )
        if cited - emitted:
            errors.append(f"terminal {label} BBL misses cited keys: {sorted(cited-emitted)}")

    receipt = {
        "status": "PASS_TERMINAL" if not errors else "FAIL",
        "errors": errors,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_route": control["selected_route"],
        "claim_decision": control["claim_decision"],
        "capsule_sha256": sha256(CAPSULE),
        "capsule_verification_sha256": sha256(CAPSULE_VERIFICATION),
        "page_count": main_pages,
        "supplement_page_count": supplement_pages,
        "working_target_met": 9 <= main_pages <= 11,
        "configuration_tokens_checked": len(CONFIGURATION_TOKENS),
        "frozen_protocol_hashes_checked": len(protocols),
        "log_counts": log_counts,
        "source_sha256": sha256(SOURCE),
        "pdf_sha256": sha256(PDF),
        "supplement_source_sha256": sha256(SUPPLEMENT_SOURCE),
        "supplement_pdf_sha256": sha256(SUPPLEMENT_PDF),
        "visual_qa_sha256": sha256(TERMINAL_VISUAL_QA),
        "verifier_sha256": sha256(Path(__file__)),
        "submission_ready": False,
        "submission_ready_reason": "terminal bundle, release profiles and author-controlled declarations remain separate gates",
    }
    final_v2.atomic_json(receipt_path, receipt)
    if errors:
        raise RuntimeError("; ".join(errors))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preterminal-self-check", "terminal"), required=True)
    args = parser.parse_args()
    result = preterminal_self_check() if args.mode == "preterminal-self-check" else terminal_run()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
