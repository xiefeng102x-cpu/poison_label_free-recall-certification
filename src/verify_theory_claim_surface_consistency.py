#!/usr/bin/env python3
"""Verify proposition assumptions and non-implications across paper surfaces.

This is a textual contract, not a theorem prover.  It guards the repaired
interfaces identified by the hostile theory audit: full-decision versus
pointwise observation, joint-law calibration transport, fixed-transform
direction estimation, and the narrow covariance premises for paired drift and
Fisher whitening.  It reads no PathMNIST result artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "results/top_journal_program_v1"
THEORY = ROOT / "theory/CHECKPOINT_AUDITABILITY_THEORY.md"
EN_APPENDIX = ROOT / "manuscript_en/theory_appendix.tex"
ZH_APPENDIX = ROOT / "manuscript_zh/theory_appendix_zh.tex"
TIFS_MAIN = ROOT / "manuscript_tifs_en/main_tifs.tex"
TIFS_ZH_SUPPLEMENT = ROOT / "manuscript_tifs_zh/supplement_tifs_zh.tex"
REFEREE_AUDIT = PROGRAM / "THEORY_REFEREE_AUDIT_2026-09-30.md"
RECEIPT = PROGRAM / "theory_claim_surface_consistency_receipt_2026-10-04.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.split())


def check_tokens(
    text: str, tokens: list[str], *, surface: str, errors: list[str]
) -> int:
    compact = normalized(text)
    for token in tokens:
        if token not in compact:
            errors.append(f"missing {surface} theory token: {token}")
    return len(tokens)


def run(*, write_receipt: bool = True) -> dict[str, Any]:
    texts = {
        "theory": THEORY.read_text(encoding="utf-8"),
        "english_appendix": EN_APPENDIX.read_text(encoding="utf-8"),
        "chinese_appendix": ZH_APPENDIX.read_text(encoding="utf-8"),
        "tifs_main": TIFS_MAIN.read_text(encoding="utf-8"),
        "tifs_zh_supplement": TIFS_ZH_SUPPLEMENT.read_text(encoding="utf-8"),
        "referee_audit": REFEREE_AUDIT.read_text(encoding="utf-8"),
    }
    errors: list[str] = []

    english_contract = [
        "include every other candidate observable",
        "equality of pointwise marginals alone does not",
        "same hypothesis-independent Markov kernel",
        "uncontrolled drift that produces different kernels",
        "conditionally exchangeable benchmark law",
        "Only under the stronger product-law assumption",
        "fixed unit vector $u$ is independent of the query",
        "not a power formula for a direction estimated from the query batch",
        "Fix a deterministic linear transform",
        "does not cover whitening estimated from the same reference observations",
        "Condition on the complete indicator vector",
        "it contains no $\\|\\delta_w\\|_2$ term",
        "there is no dimension-dependent, signal-independent prevalence floor",
        "exchangeable poison assignment among eligible candidates",
        "This is only an expectation identity",
        "does not cover a different contamination law",
        "same poison mean displacement $\\delta$",
        "Warm-starting and coordinate alignment imply neither",
        "For known positive-definite $\\Sigma$",
        "cannot be promoted to a population theorem",
    ]
    chinese_contract = [
        "实际使用的其他候选观测",
        "只有逐点边缘分布相同则不足",
        "同一个、与假设无关的 Markov 核",
        "产生不同核的未控设备漂移",
        "条件可交换基准联合律",
        "更强的条件乘积律",
        "固定单位向量 $u$ 与查询独立",
        "并不是从查询批次估计方向时的功效公式",
        "确定性线性变换",
        "不覆盖使用下述同一 reference 样本估计白化矩阵",
        "条件于完整指示向量",
        "且不含 $\\|\\delta_w\\|_2$",
        "不存在与维度相关、与信号无关的 prevalence 下限",
        "毒样本身份于合格候选索引上可交换",
        "这只是期望恒等式",
        "不覆盖不同的污染分布",
        "相同 poison 均值位移",
        "warm-start 与坐标对齐均不蕴含",
        "对已知正定 $\\Sigma$",
        "不能提升为总体定理",
    ]
    theory_contract = [
        "full per-decision law",
        "Equality of the pointwise marginal laws alone does not imply",
        "one common, hypothesis-independent Markov kernel",
        "uncontrolled device drift is not represented by one common kernel",
        "joint rank experiment",
        "marginal score-law TV is not a valid substitute",
        "fixed unit direction \\(u\\) is independent of a query",
        "does **not** cover a whitening transform estimated from the same reference",
        "Condition on the complete indicator vector",
        "it contains no \\(\\|\\delta_w\\|_2\\) term",
        "there is no dimension-dependent, signal-independent prevalence floor",
        "only an expectation identity",
        "same poison mean displacement",
        "Warm-starting and coordinate alignment imply neither premise",
        "cannot be advertised as a population theorem",
    ]
    main_contract = [
        "complete per-decision experiment",
        "marginal sample representation",
        "conditional joint-law comparison",
        "without using $i$ to estimate $u_B$",
        "fixed oracle transform",
        "no dimension-dependent, signal-independent prevalence floor",
        "poison membership is exchangeable over the eligible candidate indices",
        "10\\% trusted contamination lowers TPR",
        "equal poison displacement plus an ordered noise covariance",
        "Known population covariance would favor a Fisher direction",
    ]

    token_checks = 0
    token_checks += check_tokens(
        texts["english_appendix"],
        english_contract,
        surface="English appendix",
        errors=errors,
    )
    token_checks += check_tokens(
        texts["chinese_appendix"],
        chinese_contract,
        surface="Chinese appendix",
        errors=errors,
    )
    token_checks += check_tokens(
        texts["theory"], theory_contract, surface="theory note", errors=errors
    )
    token_checks += check_tokens(
        texts["tifs_main"], main_contract, surface="TIFS main", errors=errors
    )
    token_checks += check_tokens(
        texts["tifs_zh_supplement"],
        [
            "条件于完整指示向量",
            "不存在与维度相关、与信号无关的 prevalence 下限",
            "毒样本身份于合格候选索引上可交换",
        ],
        surface="TIFS Chinese supplement",
        errors=errors,
    )

    english_headings = re.findall(r"\\paragraph\{Proposition A\.(\d)", texts["english_appendix"])
    chinese_headings = re.findall(r"\\paragraph\{命题 A\.(\d)", texts["chinese_appendix"])
    theory_headings = re.findall(r"^### Proposition (\d) ", texts["theory"], flags=re.MULTILINE)
    expected_headings = [str(index) for index in range(1, 9)]
    if english_headings != expected_headings:
        errors.append("English proposition heading order/count mismatch")
    if chinese_headings != expected_headings:
        errors.append("Chinese proposition heading order/count mismatch")
    if theory_headings != expected_headings:
        errors.append("theory-note proposition heading order/count mismatch")
    tifs_zh_headings = re.findall(
        r"\\paragraph\{命题 A\.(\d)", texts["tifs_zh_supplement"]
    )
    if tifs_zh_headings != expected_headings:
        errors.append("TIFS Chinese supplement proposition heading order/count mismatch")

    repaired_issues = ["P0.1", "P0.2", "P1.1", "P1.2", "P1.3"]
    for issue in repaired_issues:
        if issue not in texts["referee_audit"]:
            errors.append(f"missing hostile-audit issue: {issue}")

    forbidden_patterns = [
        r"we (?:prove|establish) an end-to-end (?:guarantee|theorem)",
        r"warm-start(?:ing)? (?:guarantees|implies) (?:the|a) covariance",
        r"exact conformal validity (?:holds|is guaranteed) for training members",
        r"the propositions are independently novel",
    ]
    forbidden_checks = 0
    for surface in ("theory", "english_appendix", "tifs_main"):
        for pattern in forbidden_patterns:
            forbidden_checks += 1
            if re.search(pattern, texts[surface], flags=re.IGNORECASE):
                errors.append(f"forbidden theory overclaim in {surface}: {pattern}")

    obsolete_floor_tokens = ["K_B", "23{,}550", "23,550", "1{,}472", "1,472"]
    for surface in (
        "theory",
        "english_appendix",
        "chinese_appendix",
        "tifs_main",
        "tifs_zh_supplement",
    ):
        for token in obsolete_floor_tokens:
            forbidden_checks += 1
            if token in texts[surface]:
                errors.append(f"obsolete signal-dependent noise floor in {surface}: {token}")

    receipt = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "propositions_per_surface": 8,
        "proposition_surfaces_checked": 4,
        "assumption_and_nonimplication_tokens_checked": token_checks,
        "hostile_audit_interfaces_checked": len(repaired_issues),
        "forbidden_positive_forms_checked": forbidden_checks,
        "source_hashes": {
            "theory": sha256(THEORY),
            "english_appendix": sha256(EN_APPENDIX),
            "chinese_appendix": sha256(ZH_APPENDIX),
            "tifs_main": sha256(TIFS_MAIN),
            "tifs_zh_supplement": sha256(TIFS_ZH_SUPPLEMENT),
            "referee_audit": sha256(REFEREE_AUDIT),
            "verifier": sha256(Path(__file__)),
        },
        "pathmnist_artifacts_read": False,
    }
    if write_receipt:
        RECEIPT.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if errors:
        raise RuntimeError("; ".join(errors))
    return receipt


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
