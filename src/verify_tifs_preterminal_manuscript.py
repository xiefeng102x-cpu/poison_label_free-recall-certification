#!/usr/bin/env python3
"""Verify the preterminal IEEE TIFS main-paper surface without reading Path outputs.

This verifier proves build integrity, page-limit compliance, and a small set of
claim-surface invariants.  It deliberately does not inspect the running/sealed
PathMNIST 103--107 experiment and cannot promote the paper to submission-ready.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
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
BIBS = [MANUSCRIPT / "references_en.bib", MANUSCRIPT / "references_tifs.bib"]
CONTRACT = (
    ROOT
    / "results/top_journal_program_v1"
    / "TIFS_13_PAGE_COMPRESSION_CONTRACT_2026-10-03.md"
)
VISUAL_QA = (
    ROOT
    / "results/top_journal_program_v1"
    / "tifs_preterminal_visual_qa_2026-10-03.json"
)
RECEIPT = (
    ROOT
    / "results/top_journal_program_v1"
    / "tifs_main_preterminal_receipt_2026-10-03.json"
)
PROTOCOL_ROOT = ROOT / "results/top_journal_program_v1"
CONFIGURATION_PROTOCOLS = {
    "covariance_free": PROTOCOL_ROOT / "covariance_free_confirmation_protocol.json",
    "phase2": PROTOCOL_ROOT / "phase2_confirmation_protocol.json",
    "phase2_counterexample": PROTOCOL_ROOT
    / "phase2_counterexample_confirmation_protocol.json",
    "phase3_adaptive": PROTOCOL_ROOT / "phase3_adaptive_confirmation_protocol.json",
    "residual_pilot": PROTOCOL_ROOT / "phase4_residual_reupload_protocol.json",
    "residual_confirmation": PROTOCOL_ROOT
    / "phase4_residual_reupload_confirmation_protocol.json",
    "multiclass": PROTOCOL_ROOT / "multiclass_backdoorbench_pilot_protocol.json",
    "sanitization": PROTOCOL_ROOT / "phase5_post_sanitization_protocol.json",
    "pathmnist": PROTOCOL_ROOT
    / "pathmnist_official_weight_detector_confirmation_protocol_v1.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt_path(path: Path) -> str:
    """Use project-relative paths in production and stable absolute paths in tests."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def pdf_page_count(path: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, flags=re.MULTILINE)
    if not match:
        raise AssertionError("pdfinfo did not report a page count")
    return int(match.group(1))


def citation_keys(source: str) -> set[str]:
    keys: set[str] = set()
    for payload in re.findall(r"\\cite(?:p|t)?\{([^}]+)\}", source):
        keys.update(key.strip() for key in payload.split(",") if key.strip())
    return keys


def run(*, write_receipt: bool = True) -> dict[str, Any]:
    required_files = [
        SOURCE,
        PDF,
        LOG,
        BBL,
        SUPPLEMENT_SOURCE,
        SUPPLEMENT_PDF,
        SUPPLEMENT_LOG,
        SUPPLEMENT_BBL,
        THEORY_APPENDIX,
        CONTRACT,
        VISUAL_QA,
        *CONFIGURATION_PROTOCOLS.values(),
        *BIBS,
    ]
    missing = [receipt_path(path) for path in required_files if not path.is_file()]
    if missing:
        raise AssertionError(f"missing TIFS manuscript files: {missing}")

    source = SOURCE.read_text(encoding="utf-8")
    log = LOG.read_text(encoding="utf-8", errors="replace")
    bbl = BBL.read_text(encoding="utf-8", errors="replace")
    supplement_source = SUPPLEMENT_SOURCE.read_text(encoding="utf-8")
    supplement_log = SUPPLEMENT_LOG.read_text(encoding="utf-8", errors="replace")
    supplement_bbl = SUPPLEMENT_BBL.read_text(encoding="utf-8", errors="replace")
    theory_source = THEORY_APPENDIX.read_text(encoding="utf-8")
    visual_qa = json.loads(VISUAL_QA.read_text(encoding="utf-8"))
    protocols = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in CONFIGURATION_PROTOCOLS.items()
    }
    errors: list[str] = []

    required_tokens = [
        r"\documentclass[journal]{IEEEtran}",
        "PRETERMINAL DRAFT",
        "end-to-end guarantee",
        "candidate-batch screen",
        "42.67--48.33\\%",
        "44.17\\%",
        "four of five",
        "70.18\\% ASR",
        "\\emph{inconclusive}",
        "99,000 decisions",
        "52.67--60.71\\%",
        "71.3\\%",
        "584.2~MiB",
        "endpoints remain sealed",
        "contribute no positive or negative efficacy conclusion",
    ]
    for token in required_tokens:
        if token not in source:
            errors.append(f"missing required source token: {token}")

    forbidden_tokens = [
        "universal poison detector",
        "architecture-invariant quantum signature",
        "confirmed third-dataset detector efficacy",
        "state-of-the-art detector performance",
        "PathMNIST confirms",
        "PathMNIST detector achieves",
    ]
    for token in forbidden_tokens:
        if token.lower() in source.lower():
            errors.append(f"forbidden positive claim: {token}")

    # The preterminal source may discuss victim feasibility, but it must not
    # contain a PathMNIST detector TPR/FPR/AUC endpoint.  Treat LaTeX table
    # rows as separate units so an acronym in one claim-ledger row cannot be
    # falsely paired with ``PathMNIST`` in the following row.
    for surface_label, claim_surface in [
        ("main", source),
        ("supplement", supplement_source),
    ]:
        for text_unit in re.split(r"\\\\|\n\s*\n", claim_surface):
            if re.search(
                r"PathMNIST", text_unit, flags=re.IGNORECASE
            ) and re.search(r"\b(?:TPR|FPR|AUC)\b", text_unit):
                errors.append(
                    f"PathMNIST detector metric appears in preterminal {surface_label}"
                )
                break

    page_count = pdf_page_count(PDF)
    supplement_page_count = pdf_page_count(SUPPLEMENT_PDF)
    if page_count > 13:
        errors.append(f"TIFS page limit exceeded: {page_count} > 13")
    visual_expectations = {
        "main": (SOURCE, PDF, page_count),
        "supplement": (
            SUPPLEMENT_SOURCE,
            SUPPLEMENT_PDF,
            supplement_page_count,
        ),
    }
    if visual_qa.get("status") != "PASS_PRETERMINAL_VISUAL_QA":
        errors.append("preterminal visual-QA status is not PASS")
    for label, (source_path, pdf_path, pages) in visual_expectations.items():
        record = visual_qa.get(label, {})
        if record.get("source_sha256") != sha256(source_path):
            errors.append(f"visual-QA source hash drift: {label}")
        if record.get("pdf_sha256") != sha256(pdf_path):
            errors.append(f"visual-QA PDF hash drift: {label}")
        if record.get("pages") != pages:
            errors.append(f"visual-QA page-count drift: {label}")
        if record.get("pages_inspected") != list(range(1, pages + 1)):
            errors.append(f"visual-QA page coverage incomplete: {label}")

    supplement_tokens = [
        "PRETERMINAL SUPPLEMENT",
        "PathMNIST 103--107 endpoints remain sealed",
        "no positive or negative",
        "99,000 decisions",
        "inconclusive",
        r"TIFS\_13\_PAGE\_COMPRESSION\_CONTRACT",
    ]
    for token in supplement_tokens:
        if token not in supplement_source:
            errors.append(f"missing required supplement token: {token}")

    # Bind the human-readable configuration ledger to the frozen protocol
    # control plane.  These checks read protocol metadata only; no PathMNIST
    # result directory or performance endpoint is opened.
    ledger_assertions: list[tuple[str, Any, Any, str]] = [
        (
            "covariance-free fresh seeds",
            protocols["covariance_free"]["fresh_seeds"],
            list(range(62, 72)),
            "dev. 52--61; conf. 62--71",
        ),
        (
            "covariance-free reference count",
            protocols["covariance_free"]["method"]["reference_count"],
            120,
            "120 / 80 validation-target rows",
        ),
        (
            "phase2 fresh seeds",
            protocols["phase2"]["seeds"],
            list(range(52, 62)),
            "C2 Blended and SIG & conf. 52--61",
        ),
        (
            "phase2 poison count",
            protocols["phase2"]["poison_count"],
            60,
            "60 poisons; Blended",
        ),
        (
            "phase2 blend alpha",
            protocols["phase2"]["attack_parameters"]["blended"]["alpha"],
            0.2,
            r"Blended $\alpha=0.2$",
        ),
        (
            "phase2 SIG frequency",
            protocols["phase2"]["attack_parameters"]["sig"]["frequency"],
            6.0,
            "frequency 6",
        ),
        (
            "phase2 classical epochs",
            protocols["phase2"]["fixed_training"]["classical_epochs"],
            80,
            "Classical MLP, 80 epochs",
        ),
        (
            "phase2 QML adaptation epochs",
            protocols["phase2"]["fixed_training"]["qml_adaptive_epochs"],
            5,
            "applies five adaptive epochs",
        ),
        (
            "counterexample BadNets poison count",
            protocols["phase2_counterexample"]["badnets_cnn"]["poison_count"],
            60,
            "BadNets: 60 rows",
        ),
        (
            "counterexample BadNets epochs",
            protocols["phase2_counterexample"]["badnets_cnn"]["cnn_epochs"],
            30,
            "CNN classifier, 30 epochs",
        ),
        (
            "counterexample Input-Aware poison ratio",
            protocols["phase2_counterexample"]["input_aware_style_cnn"][
                "poison_ratio_per_batch"
            ],
            0.1,
            "0.1 poison ratio per batch",
        ),
        (
            "counterexample Input-Aware mask density",
            protocols["phase2_counterexample"]["input_aware_style_cnn"][
                "mask_density"
            ],
            0.032,
            "mask density 0.032",
        ),
        (
            "adaptive fresh seeds",
            protocols["phase3_adaptive"]["fresh_seeds"],
            list(range(52, 62)),
            "C3 white-box QML & conf. 52--61",
        ),
        (
            "adaptive evasion weight",
            protocols["phase3_adaptive"]["frozen_hyperparameters"][
                "evasion_weight"
            ],
            10.0,
            "evasion weight 10",
        ),
        (
            "adaptive epochs",
            protocols["phase3_adaptive"]["frozen_hyperparameters"]["epochs"],
            8,
            "eight attack epochs",
        ),
        (
            "adaptive learning rate",
            protocols["phase3_adaptive"]["frozen_hyperparameters"][
                "learning_rate"
            ],
            0.01,
            "learning rate 0.01",
        ),
        (
            "residual pilot seeds",
            protocols["residual_pilot"]["pilot_seeds"],
            [42, 46, 51],
            "pilot 42,46,51; conf. 52--61",
        ),
        (
            "residual confirmation seeds",
            protocols["residual_confirmation"]["fresh_seeds"],
            list(range(52, 62)),
            "pilot 42,46,51; conf. 52--61",
        ),
        (
            "residual clean fine-tuning epochs",
            protocols["residual_pilot"]["architecture"][
                "clean_finetuning_epochs"
            ],
            10,
            "freezes ten clean fine-tuning",
        ),
        (
            "residual poisoned-adaptation epochs",
            protocols["residual_pilot"]["poisoned_adaptation"]["epochs"],
            8,
            "eight poisoned-adaptation epochs",
        ),
        (
            "multiclass development seeds",
            protocols["multiclass"]["stages"]["development"]["seeds"],
            [72, 73],
            "dev. 72--73; conf. 74--78",
        ),
        (
            "multiclass confirmation seeds",
            protocols["multiclass"]["stages"]["confirmation"]["seeds"],
            list(range(74, 79)),
            "dev. 72--73; conf. 74--78",
        ),
        (
            "multiclass poison count",
            protocols["multiclass"]["attack"]["poison_count"],
            600,
            "600 in-place non-target replacements",
        ),
        (
            "multiclass target class",
            protocols["multiclass"]["attack"]["target_class"],
            6,
            "target class 6",
        ),
        (
            "multiclass maximum epochs",
            protocols["multiclass"]["victim"]["maximum_epochs"],
            10,
            "at most ten epochs",
        ),
        (
            "sanitization classical training",
            protocols["sanitization"]["retraining"]["classical"],
            "MLP from scratch, 80 epochs",
            "Retrain MLP from scratch for 80 epochs",
        ),
        (
            "PathMNIST formal seeds",
            protocols["pathmnist"]["seed_plan"]["formal_confirmation_seeds"],
            list(range(103, 108)),
            "sealed 103--107",
        ),
        (
            "PathMNIST training rows",
            protocols["pathmnist"]["dataset"]["training_rows"],
            89996,
            "89,996 official training rows",
        ),
        (
            "PathMNIST poison count",
            protocols["pathmnist"]["victims"]["suspect_attack"]["poison_count"],
            4500,
            "4,500 uniform non-target replacements",
        ),
        (
            "PathMNIST target class",
            protocols["pathmnist"]["victims"]["suspect_attack"]["target_class"],
            8,
            "target class 8",
        ),
        (
            "PathMNIST training epochs",
            protocols["pathmnist"]["victims"]["training"]["epochs"],
            5,
            "five full-data epochs",
        ),
        (
            "PathMNIST training batch size",
            protocols["pathmnist"]["victims"]["training"]["batch_size"],
            128,
            "batch 128",
        ),
        (
            "PathMNIST learning rate",
            protocols["pathmnist"]["victims"]["training"]["learning_rate"],
            0.0003,
            r"Adam $3\times10^{-4}$",
        ),
        (
            "PathMNIST reference role",
            protocols["pathmnist"]["dataset"]["validation_roles"]["reference"],
            "512 target-class clean validation examples",
            "512 / 512 target-class validation rows",
        ),
        (
            "PathMNIST bootstrap draws",
            protocols["pathmnist"]["primary_efficacy_gate"]["uncertainty"],
            "report two-sided 95% percentile bootstrap intervals with 20,000 deterministic replicates, resampling five seed blocks with replacement and retaining every row/count within each sampled block; no sample-level bootstrap",
            "20,000 seed-block bootstrap draws",
        ),
    ]
    for label, actual, expected, manuscript_token in ledger_assertions:
        if actual != expected:
            errors.append(
                f"configuration protocol drift for {label}: {actual!r} != {expected!r}"
            )
        if manuscript_token not in supplement_source:
            errors.append(
                f"configuration ledger missing protocol-bound token for {label}: "
                f"{manuscript_token}"
            )
    proposition_count = len(re.findall(r"Proposition A\.[1-8]", theory_source))
    # Each proposition name occurs once in a paragraph heading; internal proof
    # cross-references add a few more matches, so verify the eight headings.
    proposition_heading_count = len(
        re.findall(r"\\paragraph\{Proposition A\.[1-8]", theory_source)
    )
    if proposition_heading_count != 8:
        errors.append(
            f"theory supplement has {proposition_heading_count} proposition headings, expected 8"
        )

    overfull_count = len(re.findall(r"Overfull \\[hv]box", log))
    underfull_count = len(re.findall(r"Underfull \\[hv]box", log))
    if overfull_count:
        errors.append(f"LaTeX log contains {overfull_count} overfull boxes")
    unresolved_patterns = [
        r"Citation [`'][^`']+['`] on page .* undefined",
        r"Reference [`'][^`']+['`] on page .* undefined",
        r"There were undefined references",
        r"There were undefined citations",
    ]
    unresolved_count = sum(
        len(re.findall(pattern, log, flags=re.IGNORECASE))
        for pattern in unresolved_patterns
    )
    if unresolved_count:
        errors.append(f"LaTeX log contains {unresolved_count} unresolved-reference warnings")

    supplement_overfull_count = len(
        re.findall(r"Overfull \\[hv]box", supplement_log)
    )
    supplement_underfull_count = len(
        re.findall(r"Underfull \\[hv]box", supplement_log)
    )
    if supplement_overfull_count:
        errors.append(
            f"supplement log contains {supplement_overfull_count} overfull boxes"
        )
    supplement_unresolved_count = sum(
        len(re.findall(pattern, supplement_log, flags=re.IGNORECASE))
        for pattern in unresolved_patterns
    )
    if supplement_unresolved_count:
        errors.append(
            "supplement log contains "
            f"{supplement_unresolved_count} unresolved-reference warnings"
        )

    cited = citation_keys(source)
    emitted = set(re.findall(r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}", bbl))
    missing_bibitems = sorted(cited - emitted)
    if missing_bibitems:
        errors.append(f"cited keys missing from BBL: {missing_bibitems}")
    supplement_cited = citation_keys(supplement_source + "\n" + theory_source)
    supplement_emitted = set(
        re.findall(
            r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}", supplement_bbl
        )
    )
    missing_supplement_bibitems = sorted(supplement_cited - supplement_emitted)
    if missing_supplement_bibitems:
        errors.append(
            f"supplement cited keys missing from BBL: {missing_supplement_bibitems}"
        )

    hashes = {
        receipt_path(path): sha256(path)
        for path in [
            SOURCE,
            PDF,
            BBL,
            SUPPLEMENT_SOURCE,
            SUPPLEMENT_PDF,
            SUPPLEMENT_BBL,
            THEORY_APPENDIX,
            CONTRACT,
            VISUAL_QA,
            *CONFIGURATION_PROTOCOLS.values(),
            *BIBS,
        ]
    }
    status = "PASS_PRETERMINAL" if not errors else "FAIL"
    receipt: dict[str, Any] = {
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "preterminal_build_and_claim_surface_only",
        "submission_ready": False,
        "submission_blockers": [
            "PathMNIST 103--107 terminal endpoint has not been independently integrated",
            "terminal full regression, staged replay, final-bundle verification, and renewed visual QA remain required",
        ],
        "page_count": page_count,
        "supplement_page_count": supplement_page_count,
        "hard_page_limit": 13,
        "working_target_pages": [9, 11],
        "working_target_met": 9 <= page_count <= 11,
        "overfull_box_count": overfull_count,
        "underfull_box_count": underfull_count,
        "supplement_overfull_box_count": supplement_overfull_count,
        "supplement_underfull_box_count": supplement_underfull_count,
        "citation_key_count": len(cited),
        "bibitem_count": len(emitted),
        "supplement_citation_key_count": len(supplement_cited),
        "supplement_bibitem_count": len(supplement_emitted),
        "theory_proposition_heading_count": proposition_heading_count,
        "theory_proposition_reference_count": proposition_count,
        "configuration_ledger_protocol_count": len(CONFIGURATION_PROTOCOLS),
        "configuration_ledger_assertion_count": len(ledger_assertions),
        "required_claim_token_count": len(required_tokens),
        "forbidden_claim_token_count": len(forbidden_tokens),
        "pathmnist_partial_artifacts_read": False,
        "preterminal_visual_qa_status": visual_qa["status"],
        "hashes": hashes,
        "errors": errors,
    }
    if write_receipt:
        RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        RECEIPT.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return receipt


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if result["status"] == "PASS_PRETERMINAL" else 1)
