#!/usr/bin/env python3
"""Validate the author-controlled submission-action checklist.

The checked-in file is a deliberately unresolved template.  This verifier
prevents a missing, empty, or placeholder author fact from being mistaken for
submission readiness.  It does not select a licence, infer conflicts, assign
CRediT roles, or make any other author-controlled decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "results" / "top_journal_program_v1"
DEFAULT_INPUT = PROGRAM / "submission_author_actions_template_v1.json"

SCHEMA = "paper14_submission_author_actions_v1"
REQUIRED_ACTIONS = {
    "A01_TARGET_JOURNAL",
    "A02_ETHICS",
    "A03_COMPETING_INTERESTS",
    "A04_FUNDING",
    "A05_CREDIT_ROLES",
    "A06_AI_DISCLOSURE",
    "A07_PROJECT_CODE_LICENSE",
    "A08_GENERATED_ARTIFACT_LICENSE",
    "A09_REVIEWER_ACCESSION",
}
ALLOWED_STATES = {"UNRESOLVED", "CONFIRMED", "NOT_APPLICABLE"}
NOT_APPLICABLE_FORBIDDEN = {
    "A01_TARGET_JOURNAL",
    "A05_CREDIT_ROLES",
    "A06_AI_DISCLOSURE",
    "A07_PROJECT_CODE_LICENSE",
    "A08_GENERATED_ARTIFACT_LICENSE",
    "A09_REVIEWER_ACCESSION",
}
PLACEHOLDER_TOKENS = (
    "TODO",
    "TBD",
    "REQUIRED",
    "PLACEHOLDER",
    "UNRESOLVED",
    "待定",
    "待确认",
    "需要作者",
)


def _meaningful(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        return bool(stripped) and not any(
            token.casefold() in stripped.casefold() for token in PLACEHOLDER_TOKENS
        )
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_meaningful(item) for item in value)
    if isinstance(value, dict):
        return bool(value) and all(
            isinstance(key, str)
            and bool(key.strip())
            and _meaningful(item)
            for key, item in value.items()
        )
    return True


def _validate_confirmed_payload(action_id: str, value: Any) -> list[str]:
    errors: list[str] = []
    if not _meaningful(value):
        return [f"{action_id}: confirmed value is empty or placeholder-like"]

    if action_id == "A01_TARGET_JOURNAL":
        if not isinstance(value, dict) or not _meaningful(value.get("journal_name")):
            errors.append("A01_TARGET_JOURNAL: journal_name is required")
        if not isinstance(value, dict) or not _meaningful(value.get("policy_urls")):
            errors.append("A01_TARGET_JOURNAL: at least one official policy URL is required")
    elif action_id == "A05_CREDIT_ROLES":
        if not isinstance(value, list) or not value:
            errors.append("A05_CREDIT_ROLES: a non-empty author/role list is required")
        elif any(
            not isinstance(item, dict)
            or not _meaningful(item.get("author"))
            or not _meaningful(item.get("roles"))
            for item in value
        ):
            errors.append("A05_CREDIT_ROLES: every author needs one or more CRediT roles")
    elif action_id in {"A07_PROJECT_CODE_LICENSE", "A08_GENERATED_ARTIFACT_LICENSE"}:
        if not isinstance(value, dict) or not _meaningful(value.get("license_identifier")):
            errors.append(f"{action_id}: license_identifier is required")
        if not isinstance(value, dict) or value.get("rights_holder_confirmed") is not True:
            errors.append(f"{action_id}: rights_holder_confirmed must be true")
        if not isinstance(value, dict) or value.get("third_party_terms_preserved") is not True:
            errors.append(f"{action_id}: third_party_terms_preserved must be true")
    elif action_id == "A09_REVIEWER_ACCESSION":
        if not isinstance(value, dict) or not _meaningful(value.get("reviewer_url_or_doi")):
            errors.append("A09_REVIEWER_ACCESSION: reviewer_url_or_doi is required")
    return errors


def audit_submission_actions(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append(f"schema must equal {SCHEMA}")
    if not isinstance(payload.get("blind_package_safe"), bool):
        errors.append("blind_package_safe must be a boolean")

    actions = payload.get("actions")
    if not isinstance(actions, list):
        actions = []
        errors.append("actions must be a list")

    by_id: dict[str, dict[str, Any]] = {}
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            errors.append(f"actions[{index}] must be an object")
            continue
        action_id = action.get("id")
        if not isinstance(action_id, str) or action_id not in REQUIRED_ACTIONS:
            errors.append(f"actions[{index}] has an unknown id: {action_id!r}")
            continue
        if action_id in by_id:
            errors.append(f"duplicate action id: {action_id}")
            continue
        by_id[action_id] = action

    missing = sorted(REQUIRED_ACTIONS - set(by_id))
    if missing:
        errors.append(f"missing required actions: {missing}")

    unresolved: list[str] = []
    for action_id in sorted(REQUIRED_ACTIONS & set(by_id)):
        action = by_id[action_id]
        state = action.get("state")
        if state not in ALLOWED_STATES:
            errors.append(f"{action_id}: invalid state {state!r}")
            continue
        if state == "UNRESOLVED":
            unresolved.append(action_id)
            if action.get("value") is not None or action.get("evidence") is not None:
                errors.append(
                    f"{action_id}: unresolved actions must keep value/evidence null"
                )
        elif state == "NOT_APPLICABLE":
            if action_id in NOT_APPLICABLE_FORBIDDEN:
                errors.append(f"{action_id}: NOT_APPLICABLE is not permitted")
            if not _meaningful(action.get("evidence")):
                errors.append(f"{action_id}: NOT_APPLICABLE requires evidence")
        else:
            errors.extend(_validate_confirmed_payload(action_id, action.get("value")))
            if not _meaningful(action.get("evidence")):
                errors.append(f"{action_id}: CONFIRMED requires author evidence")

    if any(
        action.get("state") in {"CONFIRMED", "NOT_APPLICABLE"}
        for action in by_id.values()
    ) and payload.get("blind_package_safe") is not False:
        errors.append(
            "a populated author record must set blind_package_safe to false"
        )

    if errors:
        status = "INVALID_AUTHOR_ACTION_RECORD"
    elif unresolved:
        status = "PRETERMINAL_AUTHOR_INPUT_REQUIRED"
    else:
        status = "AUTHOR_ACTIONS_CONFIRMED"

    return {
        "schema": SCHEMA,
        "status": status,
        "required_action_count": len(REQUIRED_ACTIONS),
        "resolved_action_count": len(REQUIRED_ACTIONS) - len(unresolved),
        "unresolved_action_ids": unresolved,
        "error_count": len(errors),
        "errors": errors,
        "submission_ready_from_author_actions": status == "AUTHOR_ACTIONS_CONFIRMED",
        "blind_package_safe": payload.get("blind_package_safe"),
        "interpretation": (
            "This result covers author-controlled declarations and release choices only; "
            "it does not close scientific evidence gates or constitute legal advice."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = audit_submission_actions(payload)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if result["status"] == "INVALID_AUTHOR_ACTION_RECORD":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
