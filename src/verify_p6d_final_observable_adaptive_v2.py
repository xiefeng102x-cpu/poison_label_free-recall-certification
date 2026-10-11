#!/usr/bin/env python3
"""Independent P6-D v2 replay verifier derived from the frozen v1 verifier."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

PAPER14_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parent
V1_VERIFIER = SRC_ROOT / "verify_p6d_final_observable_adaptive.py"
V1_PROTOCOL = PAPER14_ROOT / "results/top_journal_program_v1/p6d_final_observable_adaptive_protocol_v1.json"
PROTOCOL_PATH = PAPER14_ROOT / "results/top_journal_program_v1/p6d_final_observable_adaptive_protocol_v2.json"
DEFAULT_OUTPUT = PAPER14_ROOT / "results/p6d_final_observable_adaptive_v2"
DEVELOPMENT_SEEDS = (108, 109, 110)
CONFIRMATION_SEEDS = (111, 112, 113, 114, 115)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol_payload() -> dict[str, Any]:
    overlay = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if overlay.get("base_protocol_sha256") != sha256_file(V1_PROTOCOL):
        raise RuntimeError("P6-D v2 base protocol hash mismatch")
    base = json.loads(V1_PROTOCOL.read_text(encoding="utf-8"))
    base.update({key: value for key, value in overlay.items() if key != "versioned_changes"})
    base.update(overlay["versioned_changes"])
    base["protocol_id"] = overlay["protocol_id"]
    return base


class _VersionLockTransformer(ast.NodeTransformer):
    values = {
        "PROTOCOL_ID": "p6d_final_observable_adaptive_protocol_v2",
        "PROTOCOL_PATH": f"Path({str(PROTOCOL_PATH)!r})",
        "DEFAULT_OUTPUT": f"Path({str(DEFAULT_OUTPUT)!r})",
        "DEVELOPMENT_SEEDS": DEVELOPMENT_SEEDS,
        "CONFIRMATION_SEEDS": CONFIRMATION_SEEDS,
    }

    def visit_Assign(self, node: ast.Assign):
        name = node.targets[0].id if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) else None
        if name in self.values:
            value = self.values[name]
            node.value = ast.parse(value if isinstance(value, str) and value.startswith("Path(")
                                  else repr(value), mode="eval").body
        return self.generic_visit(node)


def _load_runtime() -> ModuleType:
    source_audit = _protocol_payload()["source_audit"]
    if sha256_file(Path(__file__).resolve()) != source_audit["v2_verifier_sha256"]:
        raise RuntimeError("P6-D v2 verifier hash mismatch")
    expected = source_audit["v1_verifier_sha256"]
    if sha256_file(V1_VERIFIER) != expected:
        raise RuntimeError("P6-D v1 verifier source changed; v2 replay refused")
    tree = ast.parse(V1_VERIFIER.read_text(encoding="utf-8"), filename=str(V1_VERIFIER))
    tree = ast.fix_missing_locations(_VersionLockTransformer().visit(tree))
    module = ModuleType("_p6d_v2_independent_replay_runtime")
    module.__file__ = str(Path(__file__).resolve())
    module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(tree, str(V1_VERIFIER), "exec"), module.__dict__)
    return module


_runtime = _load_runtime()
for _name, _value in vars(_runtime).items():
    if not _name.startswith("__"):
        globals()[_name] = _value
PROTOCOL_ID = "p6d_final_observable_adaptive_protocol_v2"
PROTOCOL_PATH = PAPER14_ROOT / "results/top_journal_program_v1/p6d_final_observable_adaptive_protocol_v2.json"
DEFAULT_OUTPUT = PAPER14_ROOT / "results/p6d_final_observable_adaptive_v2"
DEVELOPMENT_SEEDS = (108, 109, 110)
CONFIRMATION_SEEDS = (111, 112, 113, 114, 115)


def verify_bundle(output: Path) -> dict[str, Any]:
    result = _runtime.verify_bundle(output)
    observable_path = output / "observable_rows.csv"
    if observable_path.exists():
        rows = _runtime.read_csv(observable_path)
        validate_observable_schema(rows)
    return result


def validate_observable_schema(rows: list[dict[str, str]]) -> None:
    if not rows or "trusted_sample_index" not in rows[0]:
        raise AssertionError("V2 observable schema lacks trusted_sample_index")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise AssertionError("V2 observable rows do not share one ordered schema")
    for row in rows:
        is_trusted = row.get("population") in ("reference", "calibration")
        if is_trusted and row["trusted_sample_index"] == "":
            raise AssertionError("V2 trusted row is missing trusted_sample_index")
        if not is_trusted and row["trusted_sample_index"] != "":
            raise AssertionError("V2 non-trusted row must have blank trusted_sample_index")


if __name__ == "__main__":
    _runtime.main()
