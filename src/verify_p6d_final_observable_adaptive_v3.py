#!/usr/bin/env python3
"""Independent P6-D v3 replay verifier derived from the frozen v1 verifier."""
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
V2_PROTOCOL = PAPER14_ROOT / "results/top_journal_program_v1/p6d_final_observable_adaptive_protocol_v2.json"
PROTOCOL_PATH = PAPER14_ROOT / "results/top_journal_program_v1/p6d_final_observable_adaptive_protocol_v3.json"
DEFAULT_OUTPUT = PAPER14_ROOT / "results/p6d_final_observable_adaptive_v3"
DEVELOPMENT_SEEDS = (116, 117, 118)
CONFIRMATION_SEEDS = (119, 120, 121, 122, 123)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol_payload() -> dict[str, Any]:
    overlay = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if overlay.get("base_protocol_sha256") != sha256_file(V1_PROTOCOL):
        raise RuntimeError("P6-D v3 base protocol hash mismatch")
    if overlay.get("v2_protocol_sha256") != sha256_file(V2_PROTOCOL):
        raise RuntimeError("P6-D v3 v2 protocol hash mismatch")
    base = json.loads(V1_PROTOCOL.read_text(encoding="utf-8"))
    base.update({key: value for key, value in overlay.items() if key != "versioned_changes"})
    base.update(overlay["versioned_changes"])
    base["protocol_id"] = overlay["protocol_id"]
    return base


class _VersionLockTransformer(ast.NodeTransformer):
    values = {
        "PROTOCOL_ID": "p6d_final_observable_adaptive_protocol_v3",
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
        elif name == "fold_rows":
            # The immutable writer emits a bare list.  The v1 verifier intended
            # to support both a list and {"receipts": [...]}, but called .get
            # unconditionally.  Accept both encodings without changing replay.
            node.value = ast.parse(
                'fold_payload.get("receipts", fold_payload) '
                'if isinstance(fold_payload, dict) else fold_payload',
                mode="eval",
            ).body
        return self.generic_visit(node)


def _load_runtime() -> ModuleType:
    audit = _protocol_payload()["source_audit"]
    if sha256_file(Path(__file__).resolve()) != audit["v3_verifier_sha256"]:
        raise RuntimeError("P6-D v3 verifier hash mismatch")
    if sha256_file(V1_VERIFIER) != audit["v1_verifier_sha256"]:
        raise RuntimeError("P6-D v1 verifier source changed; v3 replay refused")
    tree = ast.parse(V1_VERIFIER.read_text(encoding="utf-8"), filename=str(V1_VERIFIER))
    tree = ast.fix_missing_locations(_VersionLockTransformer().visit(tree))
    module = ModuleType("_p6d_v3_independent_replay_runtime")
    module.__file__ = str(Path(__file__).resolve())
    module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(tree, str(V1_VERIFIER), "exec"), module.__dict__)
    return module


_runtime = _load_runtime()
for _name, _value in vars(_runtime).items():
    if not _name.startswith("__"):
        globals()[_name] = _value
PROTOCOL_ID = "p6d_final_observable_adaptive_protocol_v3"
PROTOCOL_PATH = PAPER14_ROOT / "results/top_journal_program_v1/p6d_final_observable_adaptive_protocol_v3.json"
DEFAULT_OUTPUT = PAPER14_ROOT / "results/p6d_final_observable_adaptive_v3"
DEVELOPMENT_SEEDS = (116, 117, 118)
CONFIRMATION_SEEDS = (119, 120, 121, 122, 123)


def validate_observable_schema(rows: list[dict[str, str]]) -> None:
    if not rows or "trusted_sample_index" not in rows[0]:
        raise AssertionError("V3 observable schema lacks trusted_sample_index")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise AssertionError("V3 observable rows do not share one ordered schema")
    for row in rows:
        trusted = row.get("population") in ("reference", "calibration")
        if trusted != (row["trusted_sample_index"] != ""):
            raise AssertionError("V3 trusted_sample_index population semantics failed")


def _validate_delivery_receipts(output: Path) -> None:
    splits = json.loads((output / "split_receipts.json").read_text(encoding="utf-8"))
    if sorted(int(seed) for seed in splits) != list(DEVELOPMENT_SEEDS):
        raise AssertionError("V3 split receipt seed block mismatch")
    for seed, receipt in splits.items():
        poison = receipt.get("trigger_receipt", {}).get("poison_indices")
        if not isinstance(poison, list) or not all(isinstance(value, int) for value in poison):
            raise AssertionError(f"V3 split receipt is not plain lossless JSON: {seed}")
    checkpoints = json.loads((output / "checkpoint_receipts.json").read_text(encoding="utf-8"))
    if sorted(int(seed) for seed in checkpoints) != list(DEVELOPMENT_SEEDS):
        raise AssertionError("V3 checkpoint receipt seed block mismatch")
    for seed, kinds in checkpoints.items():
        for kind, record in kinds.items():
            path = output / record["path"]
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise AssertionError(f"V3 checkpoint receipt mismatch: {seed}/{kind}")


def verify_bundle(output: Path) -> dict[str, Any]:
    result = _runtime.verify_bundle(output)
    validate_observable_schema(_runtime.read_csv(output / "observable_rows.csv"))
    _validate_delivery_receipts(output)
    return result


if __name__ == "__main__":
    _runtime.main()
