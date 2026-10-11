#!/usr/bin/env python3
"""Versioned P6-D recovery runner.

The v1 implementation is an immutable, hash-pinned scientific base. This
module derives a private runtime namespace from that source with AST edits to
the version-specific protocol, seed locks, output root, and verifier import.
No globals in the imported v1 module are mutated. The derivation is explicit
and guarded by the v1 source hash in the v2 protocol.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np

PAPER14_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parent
V1_RUNNER = SRC_ROOT / "run_p6d_final_observable_adaptive.py"
V1_PROTOCOL = PAPER14_ROOT / "results/top_journal_program_v1/p6d_final_observable_adaptive_protocol_v1.json"
PROTOCOL_PATH = PAPER14_ROOT / "results/top_journal_program_v1/p6d_final_observable_adaptive_protocol_v2.json"
DEFAULT_OUTPUT = PAPER14_ROOT / "results/p6d_final_observable_adaptive_v2"
DEVELOPMENT_SEEDS = (108, 109, 110)
CONFIRMATION_SEEDS = (111, 112, 113, 114, 115)
FORMAL_ACKNOWLEDGEMENT = "CONSUME_P6D_V2_FORMAL_SEEDS_108_115"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol_payload() -> dict[str, Any]:
    """Resolve the compact v2 lock over the complete frozen v1 science spec."""
    overlay = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if overlay.get("base_protocol_sha256") != sha256_file(V1_PROTOCOL):
        raise RuntimeError("P6-D v2 base protocol hash mismatch")
    base = json.loads(V1_PROTOCOL.read_text(encoding="utf-8"))
    base.update({key: value for key, value in overlay.items() if key != "versioned_changes"})
    base.update(overlay["versioned_changes"])
    base["protocol_id"] = overlay["protocol_id"]
    return base


class _VersionLockTransformer(ast.NodeTransformer):
    """Derive the frozen v1 source with only declared v2 identity changes."""
    constants = {
        "PROTOCOL_ID": "p6d_final_observable_adaptive_protocol_v2",
        "PROTOCOL_PATH": f"Path({str(PROTOCOL_PATH)!r})",
        "DEFAULT_OUTPUT": f"Path({str(DEFAULT_OUTPUT)!r})",
        "DEVELOPMENT_SEEDS": DEVELOPMENT_SEEDS,
        "CONFIRMATION_SEEDS": CONFIRMATION_SEEDS,
        "FORMAL_ACKNOWLEDGEMENT": FORMAL_ACKNOWLEDGEMENT,
    }

    def visit_Assign(self, node: ast.Assign):
        name = node.targets[0].id if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) else None
        if name in self.constants:
            value = self.constants[name]
            node.value = ast.parse(value if isinstance(value, str) and value.startswith("Path(")
                                  else repr(value), mode="eval").body
        return self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if alias.name == "verify_p6d_final_observable_adaptive":
                alias.name = "verify_p6d_final_observable_adaptive_v2"
        return node


def _load_runtime() -> ModuleType:
    source_audit = _protocol_payload()["source_audit"]
    if sha256_file(Path(__file__).resolve()) != source_audit["v2_runner_sha256"]:
        raise RuntimeError("P6-D v2 runner hash mismatch")
    expected = source_audit["v1_runner_sha256"]
    actual = sha256_file(V1_RUNNER)
    if actual != expected:
        raise RuntimeError("P6-D v1 source changed; v2 derivation refused")
    tree = ast.parse(V1_RUNNER.read_text(encoding="utf-8"), filename=str(V1_RUNNER))
    transformed = ast.fix_missing_locations(_VersionLockTransformer().visit(tree))
    module = ModuleType("_p6d_v2_frozen_science_runtime")
    module.__file__ = str(Path(__file__).resolve())
    module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(transformed, str(V1_RUNNER), "exec"), module.__dict__)
    # v1 reads the base JSON directly. Keep the inherited science constants,
    # while requiring the full v2 lock before the runtime accepts execution.
    original_load_protocol = module.load_protocol
    def load_v2_protocol(path: Path = PROTOCOL_PATH):
        payload = _protocol_payload()
        if payload.get("protocol_id") != module.PROTOCOL_ID:
            raise RuntimeError("P6-D v2 protocol identifier mismatch")
        if tuple(payload["seed_audit"]["development_seeds"]) != DEVELOPMENT_SEEDS:
            raise RuntimeError("P6-D v2 development seed lock mismatch")
        if tuple(payload["seed_audit"]["confirmation_seeds"]) != CONFIRMATION_SEEDS:
            raise RuntimeError("P6-D v2 confirmation seed lock mismatch")
        if tuple(float(value) for value in payload["adaptive_attack"]["training"][
                "evasion_weight_grid_development_only"]) != tuple(module.EVASION_WEIGHTS):
            raise RuntimeError("P6-D v2 evasion-weight lock mismatch")
        return payload
    module.load_protocol = load_v2_protocol
    module._v1_load_protocol = original_load_protocol
    return module


_runtime = _load_runtime()
for _name, _value in vars(_runtime).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

# Own versioned identity and paths are surfaced by this wrapper; all science
# functions come from the audited, AST-derived runtime above.
PROTOCOL_ID = "p6d_final_observable_adaptive_protocol_v2"
PROTOCOL_PATH = PAPER14_ROOT / "results/top_journal_program_v1/p6d_final_observable_adaptive_protocol_v2.json"
DEFAULT_OUTPUT = PAPER14_ROOT / "results/p6d_final_observable_adaptive_v2"
DEVELOPMENT_SEEDS = (108, 109, 110)
CONFIRMATION_SEEDS = (111, 112, 113, 114, 115)
FORMAL_SEEDS = DEVELOPMENT_SEEDS + CONFIRMATION_SEEDS
FORMAL_ACKNOWLEDGEMENT = "CONSUME_P6D_V2_FORMAL_SEEDS_108_115"


def _lossless_observable_rows(stage: str, seed: int, configuration_id: str,
                              splits: Mapping[str, Any], observables: Mapping[str, np.ndarray],
                              partition: Any, epoch: int = 0) -> list[dict[str, Any]]:
    """V2 schema: every population carries trusted_sample_index (blank if N/A)."""
    rows = _base_lossless_observable_rows(
        stage, seed, configuration_id, splits, observables, partition, epoch
    )
    fields = list(rows[0]) if rows else []
    trusted_field = "trusted_sample_index"
    if trusted_field not in fields:
        # Insert next to dataset provenance, preserving the ordered schema.
        position = fields.index("original_dataset_index") + 1
        fields.insert(position, trusted_field)
    for row in rows:
        if trusted_field not in row:
            row[trusted_field] = ""
    return [{name: row.get(name, "") for name in fields} for row in rows]


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def lossless_csv_roundtrip_self_check(seed: int = 10_108) -> dict[str, Any]:
    """Exercise observable, score, and history CSV writer/reader paths offline."""
    _runtime.require_synthetic_seed(seed)
    rng = np.random.default_rng(seed)
    trusted = rng.normal(size=(200, 16)).astype(np.float32)
    clean = rng.normal(size=(600, 16)).astype(np.float32)
    poison = rng.normal(size=(60, 16)).astype(np.float32)
    partition = _runtime.build_crossfit_partition(seed)
    splits = {"val": {"target_original_indices": np.arange(200) + 2000},
              "train": {"target_original_indices": np.arange(600) + 3000},
              "poison_original_indices": np.arange(60) + 4000}
    observables = {"trusted": trusted, "clean": clean, "poison": poison}
    observable_rows = _lossless_observable_rows("synthetic", seed, "fixture", splits,
                                                observables, partition)
    assert len({tuple(row) for row in observable_rows}) == 1
    assert all("trusted_sample_index" in row for row in observable_rows)
    assert all(row["trusted_sample_index"] == "" for row in observable_rows
               if row["population"] in ("clean_candidate", "poison_candidate"))
    score_input = {**observables, "partition": partition}
    score_rows = _runtime._lossless_score_rows("synthetic", seed, "fixture", score_input,
                                                splits)
    history_rows = [{"seed": seed, "epoch": 0, "fixture": "roundtrip"}]
    with __import__("tempfile").TemporaryDirectory(prefix="p6d-v2-csv-") as temp:
        root = Path(temp)
        for name, rows in (("observable", observable_rows), ("scores", score_rows),
                            ("history", history_rows)):
            path = root / f"{name}.csv"
            _runtime.atomic_csv(path, rows)
            fields, reread = _read_csv(path)
            if len(reread) != len(rows) or fields != list(rows[0]):
                raise AssertionError(f"{name} CSV roundtrip shape/schema mismatch")
            if any(list(row) != list(rows[0]) for row in reread):
                raise AssertionError(f"{name} CSV roundtrip ordered schema mismatch")
            for original, loaded in zip(rows, reread):
                expected = {key: "" if value is None else str(value)
                            for key, value in original.items()}
                if loaded != expected:
                    raise AssertionError(f"{name} CSV roundtrip value mismatch")
            if name == "observable":
                if any(row["trusted_sample_index"] == "" for row in reread
                       if row["population"] in ("reference", "calibration")):
                    raise AssertionError("trusted observable index was lost")
                if any(row["trusted_sample_index"] != "" for row in reread
                       if row["population"] in ("clean_candidate", "poison_candidate")):
                    raise AssertionError("non-trusted observable index must be blank")
    return {"status": "PASS", "synthetic_seed": seed,
            "rows": {"observable": len(observable_rows), "scores": len(score_rows),
                     "history": len(history_rows)},
            "observable_schema": list(observable_rows[0])}


_base_lossless_observable_rows = _runtime._lossless_observable_rows
_base_write_selected_artifacts = _runtime.write_selected_artifacts
_base_run_development_launcher = _runtime.run_development_launcher
_base_run_confirmation_launcher = _runtime.run_confirmation_launcher
_base_guarded_formal_stage = _runtime.guarded_formal_stage


def _preflight_formal(func):
    def guarded(*args, **kwargs):
        lossless_csv_roundtrip_self_check()
        return func(*args, **kwargs)
    return guarded


def write_selected_artifacts(*args, **kwargs):
    lossless_csv_roundtrip_self_check()
    return _base_write_selected_artifacts(*args, **kwargs)


def main() -> None:
    lossless_csv_roundtrip_self_check()
    _runtime.main()


# Explicit dependency injection into the private source-derived runtime keeps
# every formal artifact path on the v2 ordered-schema writer.
_runtime._lossless_observable_rows = _lossless_observable_rows
_runtime.write_selected_artifacts = write_selected_artifacts
_runtime.run_development_launcher = _preflight_formal(_base_run_development_launcher)
_runtime.run_confirmation_launcher = _preflight_formal(_base_run_confirmation_launcher)
_runtime.guarded_formal_stage = _preflight_formal(_base_guarded_formal_stage)


if __name__ == "__main__":
    main()
