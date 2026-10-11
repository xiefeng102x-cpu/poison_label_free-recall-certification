"""Resolve original evidence hashes inside a path-sanitized staged package."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path


LEDGER_NAME = "SANITIZATION_LEDGER.json"


def raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _ledger_index(root: Path) -> dict[str, dict[str, object]]:
    ledger_path = root / LEDGER_NAME
    if not ledger_path.is_file():
        return {}
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("schema") != "paper14_blind_review_sanitization_audit_v1":
        raise AssertionError("unexpected staged-package ledger schema")
    return {item["path"]: item for item in ledger.get("files", [])}


def effective_sha256(path: Path, root: Path) -> str:
    """Return the pre-sanitization hash when the ledger binds the staged file."""
    path = path.resolve()
    root = root.resolve()
    observed = raw_sha256(path)
    try:
        relative = str(path.relative_to(root))
    except ValueError:
        return observed
    detail = _ledger_index(root).get(relative)
    if detail is None:
        return observed
    if detail.get("profile_action") == "EXCLUDED_WEIGHT_BINARY":
        raise AssertionError(f"excluded binary unexpectedly present: {relative}")
    if detail.get("staged_sha256") != observed:
        raise AssertionError(f"staged file differs from sanitization ledger: {relative}")
    source = detail.get("source_sha256")
    if not isinstance(source, str) or len(source) != 64:
        raise AssertionError(f"invalid source hash in sanitization ledger: {relative}")
    return source


def effective_size(path: Path, root: Path) -> int:
    """Return the pre-sanitization size when the staged ledger binds the file."""
    path = path.resolve()
    root = root.resolve()
    observed = path.stat().st_size
    try:
        relative = str(path.relative_to(root))
    except ValueError:
        return observed
    detail = _ledger_index(root).get(relative)
    if detail is None:
        return observed
    if int(detail.get("staged_size_bytes", -1)) != observed:
        raise AssertionError(f"staged size differs from sanitization ledger: {relative}")
    return int(detail["source_size_bytes"])
