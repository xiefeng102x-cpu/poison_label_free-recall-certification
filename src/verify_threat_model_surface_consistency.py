#!/usr/bin/env python3
"""Verify threat-model and information-budget boundaries across paper surfaces.

This is a textual consistency contract, not a security proof.  It prevents the
TIFS paper, bilingual long papers, resource audit, and claim map from silently
changing attacker control/knowledge, trusted-state integrity, clean-checkpoint
privilege, the batch-dependent decision unit, or the context-free boundary.
It reads no PathMNIST result artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "results/top_journal_program_v1"
TIFS_MAIN = ROOT / "manuscript_tifs_en/main_tifs.tex"
EN_MAIN = ROOT / "manuscript_en/main_en.tex"
ZH_MAIN = ROOT / "manuscript_zh/main_zh.tex"
INFO_AUDIT = PROGRAM / "INFORMATION_BUDGET_AND_RESOURCE_AUDIT_2026-10-03.md"
CLAIM_MAP = PROGRAM / "TIFS_CLAIM_EVIDENCE_MAP_2026-10-03.md"
RECEIPT = PROGRAM / "threat_model_surface_consistency_receipt_2026-10-04.json"


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
            errors.append(f"missing {surface} threat-model token: {token}")
    return len(tokens)


def run(*, write_receipt: bool = True) -> dict[str, Any]:
    paths = {
        "tifs_main": TIFS_MAIN,
        "english_main": EN_MAIN,
        "chinese_main": ZH_MAIN,
        "information_budget_audit": INFO_AUDIT,
        "claim_evidence_map": CLAIM_MAP,
    }
    texts = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}
    errors: list[str] = []

    contracts = {
        "tifs_main": [
            "The defender acts after training on a candidate batch",
            "trusted reference $\\mathcal R$, disjoint clean calibration $\\mathcal K$",
            "Poison labels and attack family are unavailable",
            "paired view is an optional, higher-access setting",
            "controls a training- data subset whose poisoning budget is fixed",
            "defender is not given candidate-pool prevalence",
            "cannot modify the trusted reference/calibration state or the archived clean checkpoint",
            "knows the observation map, reference construction, cross-fitting rule, calibration procedure, and threshold",
            "cannot replace the defender's trusted root",
            "trusted-state contamination is an assumption-violation stress test",
            "Method/view & Trusted ref./cal. & Clean ckpt. & Poison info & Access",
            "Suspect hidden64 & 96/96 & no & none & white-box",
            "SPECTRE oracle & no & no & target/count & features",
            "ACBD~\\cite{wu2026acbd} & $\\leq10$ clean & n/a & labeled attacks & trained detector",
            "current method remains a candidate-batch screen",
            "not an impossibility theorem for every pointwise detector",
        ],
        "english_main": [
            "a candidate batch with one claimed label",
            "a trusted clean reference set",
            "an independent clean calibration set",
            "access to comparable representations of $M_0$ and $M_1$",
            "candidate-pool prevalence is not revealed to the defender",
            "white-box attacker knows the representation, reference statistics, cross-fitting procedure, and fusion rule",
            "cannot modify $M_0$, the reference, or calibration",
            "assumption-violation stress test",
            "not a context-free isolated-sample detector",
            "not a reliable context-free single-sample detector",
        ],
        "chinese_main": [
            "同声称标签候选批次",
            "可信干净 reference 集",
            "独立干净 calibration 集",
            "对 $M_0$ 和 $M_1$ 的可比较表示访问",
            "候选池 prevalence 不提供给防御者",
            "白盒攻击者知道表示、reference 统计、交叉拟合方式和融合规则",
            "不能修改 $M_0$、reference 或 calibration",
            "违反基础假设的压力测试",
            "不是可靠的无上下文单样本检测器",
        ],
        "information_budget_audit": [
            "仍需要每个 victim 的可信 clean reference",
            "paired-drift 版本还需要匹配的 clean checkpoint",
            "这不等于“零先验、零参考、纯黑盒”",
            "已知 poison 标签/数量",
            "不需要 clean checkpoint；方向由候选池交叉拟合",
            "不是 context-free 单样本方法",
            "需要一个 poisoned class 的带标签毒样本",
            "label-holder/top-model 权限是 VFL 特权",
            "不使用其他候选查询；25,800 次查询未过 80% TPR gate",
        ],
        "claim_evidence_map": [
            "攻击者不得改写 trusted root；paired view 另需坐标对齐 clean checkpoint",
            "Adaptive attacker 知道 observation、score、reference construction、cross-fitting、calibration 与 threshold",
            "但不能改写 trusted root",
            "固定 trusted split、candidate-independent query",
            "不得声称 context-free、candidate-independent 或 exact-threshold transport 普遍可靠",
            "跨方法比较按信息预算",
        ],
    }

    token_checks = 0
    for surface, tokens in contracts.items():
        token_checks += check_tokens(
            texts[surface], tokens, surface=surface, errors=errors
        )

    privilege_rows = {
        "paired_view_requires_clean_checkpoint": (
            "tifs_main", "\\CFU XZ16 drift & 120/80 & yes & none & white-box"
        ),
        "suspect_view_requires_no_clean_checkpoint": (
            "tifs_main", "Suspect hidden64 & 96/96 & no & none & white-box"
        ),
        "spectre_has_target_count_oracle": (
            "tifs_main", "SPECTRE oracle & no & no & target/count & features"
        ),
        "acbd_has_labeled_attack_supervision": (
            "tifs_main",
            "ACBD~\\cite{wu2026acbd} & $\\leq10$ clean & n/a & labeled attacks & trained detector",
        ),
        "confirm_has_vfl_label_holder_privilege": (
            "tifs_main",
            "ConFirm~\\cite{cheng2026confirm} & task-specific & n/a & no claim & vertical federated label holder",
        ),
    }
    for label, (surface, row) in privilege_rows.items():
        if row not in normalized(texts[surface]):
            errors.append(f"missing privilege-ledger row: {label}")

    forbidden_patterns = [
        r"CF-CMD(?:-U)? is (?:a )?zero-reference",
        r"the attacker (?:can|may) (?:modify|replace) the (?:trusted root|trusted reference)",
        r"paired[- ]drift (?:does not require|requires no) (?:a )?(?:matched )?clean checkpoint",
        r"(?:context-free|candidate-independent) single-sample detector is reliable",
        r"攻击者(?:可以|能够)修改(?:可信根|可信 clean reference)",
    ]
    forbidden_checks = 0
    for surface, text in texts.items():
        for pattern in forbidden_patterns:
            forbidden_checks += 1
            if re.search(pattern, text, flags=re.IGNORECASE):
                errors.append(f"forbidden threat-model overclaim in {surface}: {pattern}")

    receipt = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "surfaces_checked": len(texts),
        "threat_model_and_boundary_tokens_checked": token_checks,
        "privilege_ledger_rows_checked": len(privilege_rows),
        "forbidden_positive_forms_checked": forbidden_checks,
        "dimensions_guarded": [
            "attacker_control",
            "attacker_knowledge",
            "trusted_state_integrity",
            "clean_checkpoint_privilege",
            "candidate_batch_dependence",
            "context_free_single_sample_boundary",
            "baseline_information_budget",
        ],
        "source_hashes": {
            **{name: sha256(path) for name, path in paths.items()},
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
