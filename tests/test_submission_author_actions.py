from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


PAPER14 = Path(__file__).resolve().parents[1]
MODULE_PATH = PAPER14 / "src" / "verify_submission_author_actions.py"
SPEC = importlib.util.spec_from_file_location("verify_submission_author_actions", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def load_template() -> dict[str, object]:
    return json.loads(MODULE.DEFAULT_INPUT.read_text(encoding="utf-8"))


def test_checked_in_template_is_explicitly_unresolved_and_valid() -> None:
    result = MODULE.audit_submission_actions(load_template())
    assert result["status"] == "PRETERMINAL_AUTHOR_INPUT_REQUIRED"
    assert result["required_action_count"] == 9
    assert result["resolved_action_count"] == 0
    assert result["unresolved_action_ids"] == sorted(MODULE.REQUIRED_ACTIONS)
    assert result["error_count"] == 0
    assert result["submission_ready_from_author_actions"] is False


def test_placeholder_cannot_be_marked_confirmed() -> None:
    payload = load_template()
    payload["actions"][0].update(
        {"state": "CONFIRMED", "value": "TBD", "evidence": "TODO"}
    )
    result = MODULE.audit_submission_actions(payload)
    assert result["status"] == "INVALID_AUTHOR_ACTION_RECORD"
    assert result["error_count"] >= 2


def test_duplicate_or_missing_action_is_invalid() -> None:
    payload = load_template()
    payload["actions"] = payload["actions"][:-1] + [copy.deepcopy(payload["actions"][0])]
    result = MODULE.audit_submission_actions(payload)
    assert result["status"] == "INVALID_AUTHOR_ACTION_RECORD"
    assert any("duplicate action id" in item for item in result["errors"])
    assert any("missing required actions" in item for item in result["errors"])


def test_complete_author_record_requires_structured_release_facts() -> None:
    payload = load_template()
    values = {
        "A01_TARGET_JOURNAL": {
            "journal_name": "Example Journal",
            "policy_urls": ["https://example.org/policy"],
        },
        "A02_ETHICS": "Confirmed against the journal and source-data policies.",
        "A03_COMPETING_INTERESTS": "The authors declare no competing interests.",
        "A04_FUNDING": "No external funding was received.",
        "A05_CREDIT_ROLES": [
            {"author": "Example Author", "roles": ["Conceptualization"]}
        ],
        "A06_AI_DISCLOSURE": "Confirmed journal-compliant disclosure text.",
        "A07_PROJECT_CODE_LICENSE": {
            "license_identifier": "Example-License",
            "rights_holder_confirmed": True,
            "third_party_terms_preserved": True,
        },
        "A08_GENERATED_ARTIFACT_LICENSE": {
            "license_identifier": "Example-Artifact-License",
            "rights_holder_confirmed": True,
            "third_party_terms_preserved": True,
        },
        "A09_REVIEWER_ACCESSION": {
            "reviewer_url_or_doi": "https://example.org/private-review"
        },
    }
    for action in payload["actions"]:
        action.update(
            {
                "state": "CONFIRMED",
                "value": values[action["id"]],
                "evidence": "Confirmed by the responsible author in the private record.",
            }
        )
    payload["blind_package_safe"] = False
    result = MODULE.audit_submission_actions(payload)
    assert result["status"] == "AUTHOR_ACTIONS_CONFIRMED"
    assert result["resolved_action_count"] == 9
    assert result["error_count"] == 0
    assert result["submission_ready_from_author_actions"] is True


def test_mandatory_release_action_cannot_be_not_applicable() -> None:
    payload = load_template()
    payload["blind_package_safe"] = False
    for action in payload["actions"]:
        if action["id"] == "A07_PROJECT_CODE_LICENSE":
            action.update(
                {
                    "state": "NOT_APPLICABLE",
                    "value": None,
                    "evidence": "Claimed not applicable.",
                }
            )
    result = MODULE.audit_submission_actions(payload)
    assert result["status"] == "INVALID_AUTHOR_ACTION_RECORD"
    assert any("NOT_APPLICABLE is not permitted" in item for item in result["errors"])
