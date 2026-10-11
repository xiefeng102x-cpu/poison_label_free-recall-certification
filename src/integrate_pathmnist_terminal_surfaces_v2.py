#!/usr/bin/env python3
"""Mechanically integrate one authorized PathMNIST route into all 19 surfaces.

The integrator accepts only R1--R3 after the V2 replay, route receipt, claim
capsule, and independent capsule replay pass.  It enforces the preterminal
hash baseline, renders every target in a same-filesystem staging directory,
validates route/capsule/numeric bindings there, and then replaces all targets
with rollback on failure.  It never edits the protected V1 history.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import select_pathmnist_terminal_route_v2 as selector
import verify_top_journal_final_bundle_v2 as final_v2


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "results/top_journal_program_v1"
CONTRACT = PROGRAM / "pathmnist_terminal_integration_execution_contract_v2.json"
CAPSULE = PROGRAM / "pathmnist_terminal_claim_capsule_v2_2026-10-04.json"
CAPSULE_VERIFICATION = PROGRAM / (
    "pathmnist_terminal_claim_capsule_verification_v2_2026-10-04.json"
)
FORMAL_OUTPUT = ROOT / "results/pathmnist_official_weight_detector_confirmation_v1"
DEFAULT_RECEIPT = PROGRAM / "pathmnist_terminal_surface_integration_v2_2026-10-04.json"
BLOCK_START = "PATHMNIST_TERMINAL_BLOCK_V2_START"
BLOCK_END = "PATHMNIST_TERMINAL_BLOCK_V2_END"
TERMINAL_ROUTES = {
    "R1_INSUFFICIENT_ELIGIBLE_VICTIMS",
    "R2_ELIGIBLE_PRIMARY_GATE_FAIL",
    "R3_ELIGIBLE_PRIMARY_GATE_PASS",
}


def sha256(path: Path) -> str:
    return final_v2.sha256(path)


def tex_escape(text: str) -> str:
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("$", r"\$"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    return text


def metric_markers(capsule: dict[str, Any]) -> list[str]:
    summaries = capsule.get("detector_summaries")
    if summaries is None:
        return []
    return [
        final_v2.metric_marker(method, summary)
        for method, summary in summaries.items()
    ]


def victim_markdown(capsule: dict[str, Any]) -> str:
    lines = [
        "| Seed | Eligible | Clean BCA (%) | Suspect BCA (%) | Suspect ASR (%) |",
        "|---:|:---:|---:|---:|---:|",
    ]
    for row in capsule["victim_rows"]:
        lines.append(
            f"| {row['seed']} | {'yes' if row['eligible'] else 'no'} | "
            f"{row['clean_bca_pct']:.2f} | {row['suspect_bca_pct']:.2f} | "
            f"{row['suspect_asr_pct']:.2f} |"
        )
    return "\n".join(lines)


def detector_markdown(capsule: dict[str, Any]) -> str:
    summaries = capsule.get("detector_summaries")
    if summaries is None:
        return (
            "Pooled detector efficacy is prohibited on this route; only the five "
            "victim-feasibility rows above are reported."
        )
    lines = [
        "| Method | Pooled TPR (%) | All-clean FPR (%) | Target-clean FPR (%) | Seed-block TPR 95% CI (%) | All-clean FPR 95% CI (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, summary in summaries.items():
        intervals = summary["seed_block_bootstrap_95pct_ci_pct"]
        lines.append(
            f"| `{method}` | {summary['pooled_tpr_pct']:.2f} | "
            f"{summary['pooled_all_clean_fpr_pct']:.2f} | "
            f"{summary['pooled_target_clean_fpr_pct']:.2f} | "
            f"[{intervals['pooled_tpr'][0]:.2f}, {intervals['pooled_tpr'][1]:.2f}] | "
            f"[{intervals['pooled_all_clean_fpr'][0]:.2f}, "
            f"{intervals['pooled_all_clean_fpr'][1]:.2f}] |"
        )
    lines.extend(
        [
            "",
            "Per-seed detector rows (lossless capsule projection):",
            "",
            "```json",
            json.dumps(
                {method: summary["per_seed"] for method, summary in summaries.items()},
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            "```",
        ]
    )
    return "\n".join(lines)


def markdown_block(capsule: dict[str, Any], capsule_sha: str) -> str:
    markers = "\n".join(f"<!-- {value} -->" for value in metric_markers(capsule))
    return (
        f"<!-- {BLOCK_START} -->\n"
        f"<!-- {capsule['route_marker']} -->\n"
        f"<!-- PATHMNIST_CLAIM_CAPSULE_SHA256={capsule_sha} -->\n"
        f"{markers}\n"
        "## Terminal PathMNIST route integration\n\n"
        f"- Route: `{capsule['selected_route']}`\n"
        f"- Claim decision: `{capsule['claim_decision']}`\n"
        f"- Eligible victim pairs: `{capsule['eligible_seed_count']}/5`\n"
        f"- Pooling allowed: `{str(capsule['pooling_allowed']).lower()}`\n\n"
        f"{capsule['main_text_en']}\n\n"
        f"{capsule['main_text_zh']}\n\n"
        f"{victim_markdown(capsule)}\n\n"
        f"{detector_markdown(capsule)}\n"
        f"<!-- {BLOCK_END} -->\n"
    )


def tex_detector_block(capsule: dict[str, Any]) -> str:
    summaries = capsule.get("detector_summaries")
    if summaries is None:
        return (
            "Pooled detector efficacy is prohibited because fewer than five "
            "victim pairs passed the frozen feasibility gate."
        )
    lines = [
        r"\begin{itemize}",
    ]
    for method, summary in summaries.items():
        intervals = summary["seed_block_bootstrap_95pct_ci_pct"]
        lines.append(
            r"\item \texttt{" + tex_escape(method) + "}: "
            f"TPR {summary['pooled_tpr_pct']:.2f}\\%, all-clean FPR "
            f"{summary['pooled_all_clean_fpr_pct']:.2f}\\%, target-clean FPR "
            f"{summary['pooled_target_clean_fpr_pct']:.2f}\\%; seed-block TPR "
            f"95\\% CI [{intervals['pooled_tpr'][0]:.2f}, "
            f"{intervals['pooled_tpr'][1]:.2f}]\\%."
        )
    lines.append(r"\end{itemize}")
    lines.append(
        "Complete per-seed rows remain losslessly bound in the independently "
        "verified terminal claim capsule."
    )
    return "\n".join(lines)


def tex_block(
    capsule: dict[str, Any], capsule_sha: str, *, language: str, detailed: bool
) -> str:
    comments = [
        f"% {BLOCK_START}",
        f"% {capsule['route_marker']}",
        f"% PATHMNIST_CLAIM_CAPSULE_SHA256={capsule_sha}",
    ]
    comments.extend(f"% {value}" for value in metric_markers(capsule))
    if language == "zh":
        comments.append(f"% {capsule['main_text_zh']}")
        heading = r"\paragraph{PathMNIST 终端结果。}"
        claim = tex_escape(capsule["main_text_zh"])
    else:
        comments.append(f"% {capsule['main_text_en']}")
        heading = r"\paragraph{Terminal PathMNIST result.}"
        claim = tex_escape(capsule["main_text_en"])
    body = [*comments, heading, claim]
    if detailed:
        body.append(tex_detector_block(capsule))
    body.append(f"% {BLOCK_END}")
    return "\n".join(body) + "\n"


def replace_tifs_preterminal_markers(text: str) -> str:
    return text.replace(
        "% PRETERMINAL DRAFT: PathMNIST 103--107 endpoints remain sealed.\n", ""
    ).replace(
        "% PRETERMINAL SUPPLEMENT: PathMNIST 103--107 endpoints remain sealed.\n",
        "",
    )


def render_csv(text: str, capsule: dict[str, Any], capsule_sha: str) -> str:
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if reader.fieldnames is None:
        raise RuntimeError("claim matrix header is absent")
    matches = [row for row in rows if row.get("claim_id") == "C27"]
    if len(matches) != 1:
        raise RuntimeError("claim matrix must contain exactly one C27")
    row = matches[0]
    row["decision"] = capsule["claim_decision"]
    row["evidence"] = (
        f"{capsule['main_text_en']} {capsule['route_marker']} "
        f"PATHMNIST_CLAIM_CAPSULE_SHA256={capsule_sha}"
    )
    qualification_field = "required_qualification"
    if qualification_field not in reader.fieldnames:
        raise RuntimeError(
            "claim matrix is missing the required_qualification column"
        )
    row[qualification_field] = (
        "The route remains dataset-, architecture-, attack-, reference-budget- "
        "and calibration-law-specific; no route authorizes invariant detection."
    )
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=reader.fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def render_json(text: str, capsule: dict[str, Any], capsule_sha: str) -> str:
    payload = json.loads(text)
    payload["pathmnist_terminal_integration"] = {
        "state": "ROUTE_RENDERED_RECOMPILATION_AND_HASH_REFRESH_REQUIRED",
        "selected_route": capsule["selected_route"],
        "claim_decision": capsule["claim_decision"],
        "route_marker": capsule["route_marker"],
        "capsule_marker": f"PATHMNIST_CLAIM_CAPSULE_SHA256={capsule_sha}",
        "capsule_sha256": capsule_sha,
        "submission_ready": False,
    }
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def render_target(
    relative: str, text: str, capsule: dict[str, Any], capsule_sha: str
) -> str:
    if BLOCK_START in text:
        raise RuntimeError(f"terminal block already exists: {relative}")
    if relative.endswith("FINAL_CLAIM_MATRIX.csv"):
        return render_csv(text, capsule, capsule_sha)
    if relative.endswith("bilingual_manuscript_receipt.json"):
        return render_json(text, capsule, capsule_sha)
    if relative.endswith(".md"):
        return text.rstrip() + "\n\n" + markdown_block(capsule, capsule_sha)
    if relative.endswith(".tex"):
        text = replace_tifs_preterminal_markers(text)
        language = "zh" if relative.startswith("manuscript_zh/") else "en"
        detailed = relative.endswith("supplement_tifs.tex")
        block = tex_block(
            capsule, capsule_sha, language=language, detailed=detailed
        )
        anchor = r"\end{document}"
        if anchor not in text:
            raise RuntimeError(f"TeX terminal anchor is absent: {relative}")
        return text.replace(anchor, block + "\n" + anchor, 1)
    raise RuntimeError(f"unsupported terminal target format: {relative}")


def validate_rendered(
    *, root: Path, contract: dict[str, Any], capsule: dict[str, Any], capsule_sha: str
) -> list[str]:
    errors: list[str] = []
    route = capsule["selected_route"]
    markers = final_v2.verify_route_markers(contract, route, root=root)
    bindings = final_v2.verify_capsule_surface_binding(
        contract, route, root / str(CAPSULE.relative_to(ROOT)), root=root
    )
    # The staged root carries a byte-identical capsule solely for independent
    # binding checks; it is not a mutable target.
    errors.extend(markers["errors"])
    errors.extend(bindings["errors"])
    matrix = root / "results/top_journal_program_v1/FINAL_CLAIM_MATRIX.csv"
    with matrix.open(newline="", encoding="utf-8") as handle:
        matches = [
            row for row in csv.DictReader(handle) if row.get("claim_id") == "C27"
        ]
    if len(matches) != 1 or matches[0].get("decision") != capsule["claim_decision"]:
        errors.append("staged C27 decision differs from claim capsule")
    for relative, baseline in contract["preterminal_mutable_hashes"].items():
        path = root / relative
        if path.is_file() and sha256(path) == baseline:
            errors.append(f"staged terminal target did not change: {relative}")
    for relative in (
        "manuscript_tifs_en/main_tifs.tex",
        "manuscript_tifs_en/supplement_tifs.tex",
    ):
        text = (root / relative).read_text(encoding="utf-8")
        if "% PRETERMINAL" in text and "PathMNIST 103--107 endpoints remain sealed" in text:
            errors.append(f"stale TIFS preterminal marker remains: {relative}")
    return errors


def synthetic_capsule(route: str) -> dict[str, Any]:
    eligible = 4 if route.startswith("R1_") else 5
    summaries = None
    if eligible == 5:
        summaries = {}
        for index, method in enumerate(
            ("primary_cf_cmd_u", "paired_drift_cf_cmd_u", "centroid", "scan", "spectral")
        ):
            summaries[method] = {
                "pooled_tpr_pct": 92.0 - index,
                "pooled_all_clean_fpr_pct": 2.0 + index,
                "pooled_target_clean_fpr_pct": 3.0 + index,
                "seed_block_bootstrap_95pct_ci_pct": {
                    "pooled_tpr": [88.0, 96.0],
                    "pooled_all_clean_fpr": [1.0, 4.0],
                    "pooled_target_clean_fpr": [2.0, 5.0],
                },
                "per_seed": [{"seed": seed, "tpr": 0.9} for seed in range(103, 108)],
            }
    decisions = {
        "R1_INSUFFICIENT_ELIGIBLE_VICTIMS": "INCONCLUSIVE",
        "R2_ELIGIBLE_PRIMARY_GATE_FAIL": "REJECTED",
        "R3_ELIGIBLE_PRIMARY_GATE_PASS": "SUPPORTED_CONDITIONALLY",
    }
    return {
        "status": "PASS",
        "selected_route": route,
        "route_marker": f"PATHMNIST_TERMINAL_ROUTE={route}",
        "claim_decision": decisions[route],
        "pooling_allowed": eligible == 5,
        "eligible_seed_count": eligible,
        "victim_rows": [
            {
                "seed": seed,
                "eligible": index < eligible,
                "clean_bca_pct": 86.0,
                "suspect_bca_pct": 84.0,
                "suspect_asr_pct": 95.0,
            }
            for index, seed in enumerate(range(103, 108))
        ],
        "detector_summaries": summaries,
        "main_text_en": f"Synthetic bounded terminal claim for {route}.",
        "main_text_zh": f"{route} 的合成有界终端主张。",
    }


def preterminal_self_check() -> dict[str, Any]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    errors: list[str] = []
    renders = 0
    for route in sorted(TERMINAL_ROUTES):
        capsule = synthetic_capsule(route)
        capsule_sha = hashlib.sha256(route.encode("ascii")).hexdigest()
        rendered: dict[str, str] = {}
        for relative in contract["mutable_integration_targets"]:
            source = (ROOT / relative).read_text(encoding="utf-8")
            output = render_target(relative, source, capsule, capsule_sha)
            rendered[relative] = output
            renders += 1
            if output == source:
                errors.append(f"synthetic route did not change target: {route}:{relative}")
            if capsule["route_marker"] not in output:
                errors.append(f"synthetic route marker missing: {route}:{relative}")
            if f"PATHMNIST_CLAIM_CAPSULE_SHA256={capsule_sha}" not in output:
                errors.append(f"synthetic capsule marker missing: {route}:{relative}")
        for relative in contract["terminal_claim_binding_surfaces"]["english_sentence_targets"]:
            if capsule["main_text_en"] not in rendered[relative]:
                errors.append(f"synthetic English claim missing: {route}:{relative}")
        for relative in contract["terminal_claim_binding_surfaces"]["chinese_sentence_targets"]:
            if capsule["main_text_zh"] not in rendered[relative]:
                errors.append(f"synthetic Chinese claim missing: {route}:{relative}")
        for relative in contract["terminal_claim_binding_surfaces"]["metric_targets"]:
            has_metric = "PATHMNIST_METRIC=" in rendered[relative]
            if has_metric is not (capsule["detector_summaries"] is not None):
                errors.append(f"synthetic metric budget mismatch: {route}:{relative}")
    result = {
        "status": "PASS_PRETERMINAL_SURFACE_INTEGRATOR_PREP" if not errors else "FAIL",
        "errors": errors,
        "routes_simulated": len(TERMINAL_ROUTES),
        "targets_per_route": len(contract["mutable_integration_targets"]),
        "synthetic_renders_checked": renders,
        "runner_sha256": sha256(Path(__file__)),
        "contract_sha256": sha256(CONTRACT),
        "terminal_receipts_opened": False,
        "pathmnist_performance_artifacts_parsed": False,
        "files_modified": False,
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
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    control = final_v2.verify_terminal_control(
        replay_receipt, route_receipt, formal_output
    )
    if control["status"] != "PASS_TERMINAL_CONTROL":
        raise RuntimeError("; ".join(control["errors"]))
    capsule = json.loads(CAPSULE.read_text(encoding="utf-8"))
    verification = json.loads(CAPSULE_VERIFICATION.read_text(encoding="utf-8"))
    if verification.get("status") != "PASS":
        raise RuntimeError("independent claim-capsule verification is not PASS")
    if verification.get("capsule_sha256") != sha256(CAPSULE):
        raise RuntimeError("claim-capsule verification hash mismatch")
    if capsule.get("selected_route") not in TERMINAL_ROUTES:
        raise RuntimeError("only terminal R1--R3 may be integrated")
    if capsule.get("selected_route") != control.get("selected_route"):
        raise RuntimeError("claim-capsule route differs from terminal control")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for relative, expected in contract["preterminal_mutable_hashes"].items():
        path = ROOT / relative
        if sha256(path) != expected:
            raise RuntimeError(f"preterminal integration baseline drift: {relative}")
    for relative, expected in contract["protected_history_hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"protected history drift: {relative}")
    return control, capsule, contract


def terminal_run(*, receipt_path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    control, capsule, contract = terminal_preconditions()
    receipt_temporary = receipt_path.with_name(receipt_path.name + ".tmp")
    if receipt_path.exists() or receipt_temporary.exists():
        raise RuntimeError("terminal surface-integration receipt already exists")
    capsule_sha = sha256(CAPSULE)
    with tempfile.TemporaryDirectory(prefix=".pathmnist-terminal-", dir=ROOT) as temp:
        work = Path(temp)
        staged = work / "staged"
        backup = work / "backup"
        for relative in contract["mutable_integration_targets"]:
            source = ROOT / relative
            staged_path = staged / relative
            backup_path = backup / relative
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            backup_path.write_bytes(source.read_bytes())
            staged_path.write_text(
                render_target(
                    relative,
                    source.read_text(encoding="utf-8"),
                    capsule,
                    capsule_sha,
                ),
                encoding="utf-8",
            )
        staged_capsule = staged / str(CAPSULE.relative_to(ROOT))
        staged_capsule.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CAPSULE, staged_capsule)
        errors = validate_rendered(
            root=staged,
            contract=contract,
            capsule=capsule,
            capsule_sha=capsule_sha,
        )
        if errors:
            raise RuntimeError("; ".join(errors))
        replaced: list[str] = []
        try:
            for relative in contract["mutable_integration_targets"]:
                os.replace(staged / relative, ROOT / relative)
                replaced.append(relative)
            protected = {
                relative: sha256(ROOT / relative)
                for relative in contract["protected_history"]
            }
            receipt = {
                "schema": "pathmnist_terminal_surface_integration_v2",
                "status": "PASS_TERMINAL_SURFACE_INTEGRATION",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "selected_route": control["selected_route"],
                "claim_decision": capsule["claim_decision"],
                "capsule_sha256": capsule_sha,
                "changed_targets": {
                    relative: sha256(ROOT / relative)
                    for relative in contract["mutable_integration_targets"]
                },
                "changed_target_count": len(contract["mutable_integration_targets"]),
                "protected_history_hashes": protected,
                "protected_history_count": len(protected),
                "runner_sha256": sha256(Path(__file__)),
                "recompilation_required": True,
                "semantic_editorial_pass_required": True,
                "submission_ready": False,
            }
            final_v2.atomic_json(receipt_path, receipt)
            return receipt
        except Exception as original:
            rollback_errors: list[str] = []
            for relative in replaced:
                try:
                    shutil.copy2(backup / relative, ROOT / relative)
                except Exception as rollback_error:  # pragma: no cover - OS failure
                    rollback_errors.append(f"{relative}: {rollback_error}")
            for artifact in (receipt_path, receipt_temporary):
                try:
                    artifact.unlink(missing_ok=True)
                except Exception as cleanup_error:  # pragma: no cover - OS failure
                    rollback_errors.append(f"{artifact}: {cleanup_error}")
            if rollback_errors:
                raise RuntimeError(
                    "terminal integration failed and rollback was incomplete: "
                    + "; ".join(rollback_errors)
                ) from original
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("preterminal-self-check", "terminal"), required=True
    )
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    result = (
        preterminal_self_check()
        if args.mode == "preterminal-self-check"
        else terminal_run(receipt_path=args.receipt)
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
