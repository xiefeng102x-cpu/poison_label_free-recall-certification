#!/usr/bin/env python3
"""Hash-bound terminal bilingual manuscript claim/evidence consistency audit.

The audit reads completed authoritative artifacts plus the independently
verified, route-authorized PathMNIST claim capsule. It does not read partial or
raw PathMNIST performance artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "results/top_journal_program_v1"
RESULTS = ROOT / "results"
DEFAULT_RECEIPT = PROGRAM / "manuscript_claim_consistency_terminal_v2_2026-10-05.json"
PATH_CAPSULE = PROGRAM / "pathmnist_terminal_claim_capsule_v2_2026-10-04.json"
PATH_CAPSULE_VERIFICATION = PROGRAM / (
    "pathmnist_terminal_claim_capsule_verification_v2_2026-10-04.json"
)
DOCUMENTATION_SURFACES = {
    "status": PROGRAM / "STATUS.md",
    "reproduce": PROGRAM / "REPRODUCE.md",
    "artifact_index": PROGRAM / "ARTIFACT_INDEX.md",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f}"


def require_tokens(
    text: str, tokens: list[str], *, surface: str, errors: list[str]
) -> int:
    checked = 0
    for token in tokens:
        checked += 1
        if token not in text:
            errors.append(f"missing {surface} token: {token}")
    return checked


def require_any(
    text: str, alternatives: list[str], *, surface: str, errors: list[str]
) -> int:
    if not any(token in text for token in alternatives):
        errors.append(
            f"missing {surface} alternatives: " + " | ".join(alternatives)
        )
    return 1


def require_normalized_tokens(
    text: str, tokens: list[str], *, surface: str, errors: list[str]
) -> int:
    """Check prose contracts while ignoring Markdown line wrapping."""
    return require_tokens(
        " ".join(text.split()),
        tokens,
        surface=surface,
        errors=errors,
    )


def primary_single_sample_summary() -> dict[str, float]:
    decision = json.loads(
        (RESULTS / "single_sample_reference_audit_v1/decision.json").read_text(
            encoding="utf-8"
        )
    )
    summary = read_csv(
        RESULTS / "single_sample_reference_audit_v1/population_summary.csv"
    )
    primary = [
        row
        for row in summary
        if row["population"] == "stored_test_eligible_victims_per_attack_mode"
    ]
    all_fpr = [
        float(endpoint[field])
        for endpoint in decision["mode_endpoints"].values()
        for field in ("pooled_member_clean_fpr", "pooled_fresh_clean_fpr")
    ]
    all_auc = [
        float(row[field])
        for row in primary
        for field in (
            "mean_poison_auc_against_member_clean",
            "mean_poison_auc_against_fresh_clean",
        )
    ]
    tpr = [
        float(endpoint["pooled_poison_tpr"])
        for endpoint in decision["mode_endpoints"].values()
    ]
    return {
        "tpr_min": min(tpr),
        "tpr_max": max(tpr),
        "fpr_min": min(all_fpr),
        "fpr_max": max(all_fpr),
        "auc_min": min(all_auc),
        "auc_max": max(all_auc),
    }


def run(*, receipt_path: Optional[Path] = DEFAULT_RECEIPT) -> dict[str, Any]:
    en_path = ROOT / "manuscript_en/main_en.tex"
    zh_path = ROOT / "manuscript_zh/main_zh.tex"
    en_pdf = ROOT / "manuscript_en/output/pdf/main_en.pdf"
    zh_pdf = ROOT / "manuscript_zh/output/pdf/main_zh.pdf"
    en_text, zh_text = (
        en_path.read_text(encoding="utf-8"),
        zh_path.read_text(encoding="utf-8"),
    )
    errors: list[str] = []
    token_checks = 0

    bilingual = json.loads(
        (PROGRAM / "bilingual_manuscript_receipt.json").read_text(encoding="utf-8")
    )
    capsule = json.loads(PATH_CAPSULE.read_text(encoding="utf-8"))
    capsule_verification = json.loads(
        PATH_CAPSULE_VERIFICATION.read_text(encoding="utf-8")
    )
    if capsule.get("status") != "PASS":
        errors.append("PathMNIST terminal claim capsule is not PASS")
    if capsule_verification.get("status") != "PASS":
        errors.append("PathMNIST terminal claim-capsule verification is not PASS")
    if capsule_verification.get("capsule_sha256") != sha256(PATH_CAPSULE):
        errors.append("PathMNIST terminal claim-capsule hash mismatch")
    hash_checks = {
        "english_source": (en_path, bilingual["english"]["source_sha256"]),
        "english_pdf": (en_pdf, bilingual["english"]["pdf_sha256"]),
        "chinese_source": (zh_path, bilingual["chinese"]["source_sha256"]),
        "chinese_pdf": (zh_pdf, bilingual["chinese"]["pdf_sha256"]),
        "english_appendix": (
            ROOT / "manuscript_en/theory_appendix.tex",
            bilingual["english"]["theory_appendix_sha256"],
        ),
        "chinese_appendix": (
            ROOT / "manuscript_zh/theory_appendix_zh.tex",
            bilingual["chinese"]["theory_appendix_sha256"],
        ),
    }
    for label, (path, expected) in hash_checks.items():
        if sha256(path) != expected:
            errors.append(f"hash mismatch: {label}")
    if bilingual["english"]["pages"] != 30 or bilingual["chinese"]["pages"] != 28:
        errors.append("bilingual page-count receipt mismatch")

    # Keep the three user-facing package guides synchronized with the
    # hash-bound manuscript receipt.  This closes a recurrent failure mode in
    # which the PDFs and scientific claims were current but the hand-off
    # documentation still advertised an older page or verification count.
    documentation_text = {
        name: path.read_text(encoding="utf-8")
        for name, path in DOCUMENTATION_SURFACES.items()
    }
    documentation_token_checks = 0
    documentation_token_checks += require_normalized_tokens(
        documentation_text["status"],
        [
            "Updated 28-page Chinese manuscript",
            "Updated 30-page anonymous English manuscript",
            "PathMNIST five-seed detector confirmation on 103--107",
            "99.79%/0.54%/3.56%",
            "Terminal TIFS main/supplement compile",
        ],
        surface="STATUS documentation contract",
        errors=errors,
    )
    documentation_token_checks += require_normalized_tokens(
        documentation_text["reproduce"],
        [
            "Current terminal PathMNIST/TIFS state",
            "R3_ELIGIBLE_PRIMARY_GATE_PASS",
            "99.79% TPR / 0.54% all-clean FPR",
        ],
        surface="REPRODUCE documentation contract",
        errors=errors,
    )
    documentation_token_checks += require_normalized_tokens(
        documentation_text["artifact_index"],
        [
            "tifs_main_terminal_receipt_2026-10-04.json",
            "pathmnist_terminal_claim_capsule_v2_2026-10-04.json",
            "formal replay selects R3 with 5/5 eligible victims",
        ],
        surface="ARTIFACT_INDEX documentation contract",
        errors=errors,
    )

    # Adaptive white-box endpoint, recomputed from its completed summary.
    p3_rows = [
        row
        for row in read_csv(
            RESULTS / "bloodmnist_whitebox_adaptive_confirmation_v1/eligible_summary.csv"
        )
        if row["representation"] == "coherent_bonferroni_input_qml_xz"
    ]
    p3_tpr = [float(row["pooled_tpr"]) for row in p3_rows]
    p3_range = f"{pct(min(p3_tpr))}\\%--{pct(max(p3_tpr))}\\%"
    token_checks += require_tokens(en_text, [p3_range], surface="English P3", errors=errors)
    token_checks += require_tokens(zh_text, [p3_range], surface="Chinese P3", errors=errors)

    # Candidate-independent single-sample endpoint and grid explanation.
    single = primary_single_sample_summary()
    single_tpr = f"{pct(single['tpr_min'])}\\%--{pct(single['tpr_max'])}\\%"
    single_fpr = f"{pct(single['fpr_min'])}\\%--{pct(single['fpr_max'])}\\%"
    single_auc = f"{single['auc_min']:.3f}--{single['auc_max']:.3f}"
    single_tokens = [single_tpr, single_fpr, single_auc, "1/101", "6/101"]
    token_checks += require_tokens(en_text, single_tokens, surface="English single-sample", errors=errors)
    token_checks += require_tokens(zh_text, single_tokens, surface="Chinese single-sample", errors=errors)

    # Realized-geometry boundary.
    geometry = json.loads(
        (RESULTS / "auditability_geometry_diagnostic_v1/decision.json").read_text(
            encoding="utf-8"
        )
    )
    geometry_tokens = [
        f"{geometry['multi_to_representation_mean_shift_ratio']:.3f}",
        f"{geometry['multi_minus_representation_mean_resultant_length']:.3f}",
        "0.9965",
        "0.9947",
        "0.9985",
    ]
    token_checks += require_tokens(en_text, geometry_tokens, surface="English geometry", errors=errors)
    token_checks += require_tokens(zh_text, geometry_tokens, surface="Chinese geometry", errors=errors)

    # Multiclass SCAn applicability.
    multiclass = json.loads(
        (PROGRAM / "multiclass_backdoorbench_program_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    scan = next(
        row
        for row in multiclass["confirmation"]["eligible_summary"]
        if row["method"] == "backdoorbench_scan" and row["view"] == "suspect_hidden64"
    )
    scan_tokens = [
        f"{pct(scan['pooled_tpr'])}\\%",
        f"{pct(scan['pooled_fpr_all_clean'], 4)}\\%",
        f"{pct(scan['minimum_seed_tpr'], 0)}\\%",
    ]
    token_checks += require_tokens(en_text, scan_tokens, surface="English SCAn", errors=errors)
    token_checks += require_tokens(zh_text, scan_tokens, surface="Chinese SCAn", errors=errors)

    # Fresh-query calibration confirmation.
    fresh = json.loads(
        (PROGRAM / "fresh_query_calibration_verification.json").read_text(
            encoding="utf-8"
        )
    )
    confirmation = next(stage for stage in fresh["stages"] if stage["stage"] == "confirmation")
    suspect_gap = 100.0 * confirmation["pooled"]["suspect_hidden64"]["member_minus_fresh_gap"]
    drift_gap = 100.0 * confirmation["pooled"]["paired_hidden64_drift"]["member_minus_fresh_gap"]
    fresh_tokens = [f"+{suspect_gap:.2f}", f"{drift_gap:.2f}", "41,944"]
    token_checks += require_tokens(en_text, fresh_tokens, surface="English calibration", errors=errors)
    token_checks += require_tokens(zh_text, fresh_tokens, surface="Chinese calibration", errors=errors)

    # QML noise operating-point boundary.
    noise = json.loads(
        (RESULTS / "p6c_qml_noise_bridge_v1/paired_summary.json").read_text(
            encoding="utf-8"
        )
    )["conditions"]["nominal_noise_512"]
    noise_tokens = [
        f"{noise['mean_auc']:.3f}",
        f"{noise['pooled_tpr']:.3f}/{noise['pooled_fpr']:.3f}",
    ]
    token_checks += require_tokens(en_text, noise_tokens, surface="English noise", errors=errors)
    token_checks += require_tokens(zh_text, noise_tokens, surface="Chinese noise", errors=errors)

    # Final-observable attack eligibility boundary.
    p6d = json.loads(
        (PROGRAM / "p6d_v3_formal_confirmation_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    p6d_ratio = f"{p6d['confirmation']['eligible_victim_count']}/{p6d['confirmation']['victims_total']}"
    p6d_asr = f"{100 * float(p6d['confirmation']['diagnostic_per_seed'][0]['test_asr']):.2f}\\%"
    token_checks += require_any(
        en_text,
        [p6d_ratio, "four of five"],
        surface="English P6-D eligibility",
        errors=errors,
    )
    token_checks += require_tokens(en_text, [p6d_asr], surface="English P6-D", errors=errors)
    token_checks += require_tokens(zh_text, [p6d_ratio, p6d_asr], surface="Chinese P6-D", errors=errors)

    # External-domain language is checked only against the verified terminal
    # capsule, never against partial formal output.
    token_checks += require_tokens(
        en_text,
        [
            capsule["main_text_en"],
            "PATHMNIST_TERMINAL_ROUTE=R3_ELIGIBLE_PRIMARY_GATE_PASS",
            f"PATHMNIST_CLAIM_CAPSULE_SHA256={sha256(PATH_CAPSULE)}",
            "99.50--99.96\\%",
            "0.48--0.60\\%",
        ],
        surface="English PathMNIST boundary",
        errors=errors,
    )
    token_checks += require_tokens(
        zh_text,
        [
            capsule["main_text_zh"],
            "PATHMNIST_TERMINAL_ROUTE=R3_ELIGIBLE_PRIMARY_GATE_PASS",
            f"PATHMNIST_CLAIM_CAPSULE_SHA256={sha256(PATH_CAPSULE)}",
            "99.50--99.96\\%",
            "0.48--0.60\\%",
        ],
        surface="Chinese PathMNIST boundary",
        errors=errors,
    )

    # The long-form manuscripts must carry the frozen core configuration in
    # both languages rather than delegating every setting to JSON files.  The
    # TIFS verifier separately binds the same facts to nine protocol files.
    configuration_token_checks = 0
    configuration_token_checks += require_tokens(
        en_text,
        [
            "Frozen core configurations and data roles",
            "60 poisoned training rows",
            r"Blended uses $\alpha=0.2$ with pattern seed 14014",
            r"SIG uses amplitude $40/255$ and frequency 6",
            "evasion weight 10, eight epochs, and learning rate 0.01",
            "development seeds 72--73 and confirmation seeds 74--78",
            "target class 6 and 600 in-place non-target",
            "full 89,996-row official PathMNIST training split",
            "4,500 uniformly selected non-target replacements",
            "512 target-class reference and 512 disjoint calibration rows",
            "20,000 seed-block bootstrap replicates",
            "These protocol facts were frozen before the atomic replay and mechanical R3 route selection",
        ],
        surface="English frozen configuration",
        errors=errors,
    )
    configuration_token_checks += require_tokens(
        zh_text,
        [
            "冻结的核心配置与数据角色",
            "60 个训练毒样本",
            r"Blended 固定 $\alpha=0.2$ 与 pattern seed 14014",
            r"SIG 固定振幅 $40/255$、频率 6",
            "evasion weight 10、八个 epochs 与 0.01 learning rate",
            "seeds 72--73 开发、74--78 确认",
            "类别 6 为 target，对 600 个非目标训练样本",
            "完整的 89,996 行 PathMNIST 官方训练集",
            "均匀选择 4,500 行",
            "512 个 target-class reference 与 512 个互不重叠的 calibration",
            "20,000 次 seed-block bootstrap",
            "上述协议在原子重放与机械 R3 路线选择前冻结",
        ],
        surface="Chinese frozen configuration",
        errors=errors,
    )
    token_checks += configuration_token_checks

    # Central claim decisions must agree with the current claim matrix.
    claims = {
        row["claim_id"]: row["decision"]
        for row in read_csv(PROGRAM / "FINAL_CLAIM_MATRIX.csv")
    }
    expected_decisions = {
        "C17": "REJECTED",
        "C18": "REJECTED",
        "C21": "REJECTED",
        "C26": "INCONCLUSIVE",
        "C27": "SUPPORTED_CONDITIONALLY",
        "C28": "REJECTED",
        "C29": "REJECTED",
    }
    for claim_id, expected in expected_decisions.items():
        if claims.get(claim_id) != expected:
            errors.append(f"claim decision mismatch: {claim_id}")

    # Catch positive forms of the most damaging overclaims.  These exact forms
    # do not match the manuscript's explicitly negated boundary statements.
    forbidden = {
        "English": (
            en_text,
            [
                r"CF-CMD is robust to detector-aware white-box attacks",
                r"checkpoint differences are universally superior",
                r"PathMNIST establishes universal detector efficacy",
                r"the reference-only XZ16 rule is a reliable context-free",
            ],
        ),
        "Chinese": (
            zh_text,
            [
                r"CF-CMD 对检测器感知白盒攻击具有鲁棒性",
                r"检查点差分具有普遍优势",
                r"PathMNIST 已确认普适检测器效力",
                r"reference-only XZ16 规则是可靠的无上下文",
            ],
        ),
    }
    forbidden_checks = 0
    for surface, (text, patterns) in forbidden.items():
        for pattern in patterns:
            forbidden_checks += 1
            if re.search(pattern, text):
                errors.append(f"forbidden positive claim in {surface}: {pattern}")

    receipt = {
        "status": "PASS" if not errors else "FAIL",
        "scope": "completed_artifacts_plus_verified_terminal_capsule_no_partial_pathmnist_endpoint_access",
        "verifier": str(Path(__file__).relative_to(ROOT)),
        "verifier_sha256": sha256(Path(__file__)),
        "manuscript_hashes": {
            "english_source": sha256(en_path),
            "english_pdf": sha256(en_pdf),
            "chinese_source": sha256(zh_path),
            "chinese_pdf": sha256(zh_pdf),
        },
        "authoritative_claim_families_recomputed": 7,
        "required_bilingual_tokens_checked": token_checks,
        "frozen_configuration_tokens_checked": configuration_token_checks,
        "documentation_consistency_tokens_checked": documentation_token_checks,
        "documentation_hashes": {
            name: sha256(path) for name, path in DOCUMENTATION_SURFACES.items()
        },
        "forbidden_positive_forms_checked": forbidden_checks,
        "claim_matrix_decisions_checked": len(expected_decisions),
        "single_sample_operating_grid": {
            "calibration_count": 100,
            "scorer_count": 3,
            "smallest_fused_rank": 3.0 / 101.0,
            "next_fused_rank": 6.0 / 101.0,
            "alpha": 0.05,
        },
        "pathmnist_terminal_capsule_read": True,
        "pathmnist_terminal_capsule_sha256": sha256(PATH_CAPSULE),
        "pathmnist_raw_or_partial_performance_artifacts_read": False,
        "errors": errors,
    }
    # Tests and staged-review replays use ``receipt_path=None`` so the audit is
    # read-only.  The CLI retains the explicit default receipt for release use.
    if receipt_path is not None:
        write_json(receipt_path, receipt)
    if errors:
        raise RuntimeError("; ".join(errors))
    return receipt


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
