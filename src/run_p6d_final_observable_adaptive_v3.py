#!/usr/bin/env python3
"""P6-D v3: hash-pinned v1 science plus lossless JSON delivery recovery."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np
import torch

PAPER14_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parent
V1_RUNNER = SRC_ROOT / "run_p6d_final_observable_adaptive.py"
V1_PROTOCOL = PAPER14_ROOT / "results/top_journal_program_v1/p6d_final_observable_adaptive_protocol_v1.json"
V2_PROTOCOL = PAPER14_ROOT / "results/top_journal_program_v1/p6d_final_observable_adaptive_protocol_v2.json"
PROTOCOL_PATH = PAPER14_ROOT / "results/top_journal_program_v1/p6d_final_observable_adaptive_protocol_v3.json"
DEFAULT_OUTPUT = PAPER14_ROOT / "results/p6d_final_observable_adaptive_v3"
DEVELOPMENT_SEEDS = (116, 117, 118)
CONFIRMATION_SEEDS = (119, 120, 121, 122, 123)
FORMAL_ACKNOWLEDGEMENT = "CONSUME_P6D_V3_FORMAL_SEEDS_116_123"


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
    constants = {
        "PROTOCOL_ID": "p6d_final_observable_adaptive_protocol_v3",
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
                alias.name = "verify_p6d_final_observable_adaptive_v3"
        return node


def _load_runtime() -> ModuleType:
    audit = _protocol_payload()["source_audit"]
    if sha256_file(Path(__file__).resolve()) != audit["v3_runner_sha256"]:
        raise RuntimeError("P6-D v3 runner hash mismatch")
    if sha256_file(V1_RUNNER) != audit["v1_runner_sha256"]:
        raise RuntimeError("P6-D v1 source changed; v3 derivation refused")
    tree = ast.parse(V1_RUNNER.read_text(encoding="utf-8"), filename=str(V1_RUNNER))
    tree = ast.fix_missing_locations(_VersionLockTransformer().visit(tree))
    module = ModuleType("_p6d_v3_frozen_science_runtime")
    module.__file__ = str(Path(__file__).resolve())
    module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(tree, str(V1_RUNNER), "exec"), module.__dict__)

    def load_v3_protocol(path: Path = PROTOCOL_PATH):
        payload = _protocol_payload()
        if payload.get("protocol_id") != module.PROTOCOL_ID:
            raise RuntimeError("P6-D v3 protocol identifier mismatch")
        if tuple(payload["seed_audit"]["development_seeds"]) != DEVELOPMENT_SEEDS:
            raise RuntimeError("P6-D v3 development seed lock mismatch")
        if tuple(payload["seed_audit"]["confirmation_seeds"]) != CONFIRMATION_SEEDS:
            raise RuntimeError("P6-D v3 confirmation seed lock mismatch")
        if tuple(float(x) for x in payload["adaptive_attack"]["training"]
                 ["evasion_weight_grid_development_only"]) != tuple(module.EVASION_WEIGHTS):
            raise RuntimeError("P6-D v3 evasion-weight lock mismatch")
        return payload

    module.load_protocol = load_v3_protocol
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
FORMAL_SEEDS = DEVELOPMENT_SEEDS + CONFIRMATION_SEEDS
FORMAL_ACKNOWLEDGEMENT = "CONSUME_P6D_V3_FORMAL_SEEDS_116_123"


def plain_json(value: Any) -> Any:
    """Recursively map NumPy containers/scalars to strict plain-JSON values."""
    if isinstance(value, np.ndarray):
        return plain_json(value.tolist())
    if isinstance(value, np.generic):
        return plain_json(value.item())
    if isinstance(value, Mapping):
        return {str(key): plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_json(item) for item in value]
    return value


def stable_json_bytes(payload: Any) -> bytes:
    return (json.dumps(plain_json(payload), indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def stable_json_sha256(payload: Any) -> str:
    return hashlib.sha256(stable_json_bytes(payload)).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    _runtime._atomic_bytes(path, stable_json_bytes(payload))


_base_lossless_observable_rows = _runtime._lossless_observable_rows


def _lossless_observable_rows(stage: str, seed: int, configuration_id: str,
                              splits: Mapping[str, Any], observables: Mapping[str, np.ndarray],
                              partition: Any, epoch: int = 0) -> list[dict[str, Any]]:
    """Retain the exact v2 ordered observable schema repair."""
    rows = _base_lossless_observable_rows(stage, seed, configuration_id, splits,
                                          observables, partition, epoch)
    fields = list(rows[0]) if rows else []
    if "trusted_sample_index" not in fields:
        fields.insert(fields.index("original_dataset_index") + 1, "trusted_sample_index")
    for row in rows:
        row.setdefault("trusted_sample_index", "")
    return [{name: row.get(name, "") for name in fields} for row in rows]


_runtime.atomic_json = atomic_json
_runtime.stable_json_bytes = stable_json_bytes
_runtime.stable_json_sha256 = stable_json_sha256
_runtime._lossless_observable_rows = _lossless_observable_rows


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def full_path_synthetic_shadow_self_check(seed: int = 10_116) -> dict[str, Any]:
    """Exercise every development delivery artifact and the test firewall."""
    require_synthetic_seed(seed)
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory(prefix="p6d-v3-shadow-") as temp:
        stage_root = Path(temp) / "development"
        signature = run_signature(stage="development", seeds=DEVELOPMENT_SEEDS,
                                  evasion_weights=EVASION_WEIGHTS)
        immutable = {"protocol": sha256_file(PROTOCOL_PATH),
                     "runner": sha256_file(Path(__file__).resolve()),
                     "fixture": stable_json_sha256({"seed": seed})}
        manifest = initialize_or_resume(stage_root, signature, immutable_hashes=immutable)
        environment = {"mode": "synthetic_shadow", "seed": seed, "test_data": False}
        atomic_json(stage_root / "environment.json", environment)
        manifest = update_manifest_artifact(stage_root, manifest, stage_root / "environment.json")

        history_rows: list[dict[str, Any]] = []
        score_rows: list[dict[str, Any]] = []
        observable_rows: list[dict[str, Any]] = []
        fold_receipts: list[dict[str, Any]] = []
        split_receipts: dict[str, Any] = {}
        selected_rows: dict[float, list[dict[str, Any]]] = {float(w): [] for w in EVASION_WEIGHTS}
        exact_by_key: dict[tuple[int, float], dict[str, Any]] = {}

        for formal_seed in DEVELOPMENT_SEEDS:
            splits = {
                "val": {"target_original_indices": np.arange(TRUSTED_COUNT) + formal_seed * 10_000},
                "train": {"target_original_indices": np.arange(CLEAN_CANDIDATE_COUNT) + formal_seed * 20_000},
                "poison_original_indices": np.arange(POISON_CANDIDATE_COUNT) + formal_seed * 30_000,
            }
            poison_ids = np.arange(POISON_CANDIDATE_COUNT, dtype=np.int64)
            split_receipts[str(formal_seed)] = {
                "seed": np.int64(formal_seed),
                "trigger_receipt": {
                    "poison_indices": poison_ids,
                    "mapped_trigger_linf": np.float32(0.25),
                    "nested": {"shape": np.asarray([60, 16], dtype=np.int32)},
                },
                "poison_indices_sha256": stable_json_sha256(poison_ids.astype(int).tolist()),
            }
            for weight in EVASION_WEIGHTS:
                config = f"w{weight:g}"
                observables = {
                    "trusted": rng.normal(size=(TRUSTED_COUNT, OBSERVABLE_DIMENSION)).astype(np.float32),
                    "clean": rng.normal(size=(CLEAN_CANDIDATE_COUNT, OBSERVABLE_DIMENSION)).astype(np.float32),
                    "poison": rng.normal(loc=0.2, size=(POISON_CANDIDATE_COUNT, OBSERVABLE_DIMENSION)).astype(np.float32),
                }
                partition = build_crossfit_partition(formal_seed)
                exact = numpy_cfcmdu_score(formal_seed, observables["trusted"],
                                            observables["clean"], observables["poison"],
                                            partition=partition)
                exact_by_key[(formal_seed, float(weight))] = exact
                base_row = {
                    "epoch": 1, "validation_balanced_ca": 0.93,
                    "validation_asr": 0.86, "tpr": float(exact["tpr"]),
                    "fpr": float(exact["fpr"]),
                    "mean_poison_p_value": float(exact["mean_poison_p_value"]),
                }
                chosen = select_epoch([base_row])
                selected_rows[float(weight)].append({**chosen, "seed": formal_seed})
                history_rows.append({
                    **base_row, "stage": "development", "seed": formal_seed,
                    "evasion_weight": float(weight), "configuration_id": config,
                    "selected_epoch": True,
                })
                score_rows.extend(_runtime._lossless_score_rows(
                    "development", formal_seed, config,
                    {**observables, "partition": partition}, splits, 1))
                observable_rows.extend(_lossless_observable_rows(
                    "development", formal_seed, config, splits, observables, partition, 1))
                fold_receipts.append({**partition.to_json(), "stage": "development",
                                      "configuration_id": config})

        summaries = [summarize_development_configuration(
            f"w{weight:g}", weight, selected_rows[float(weight)]) for weight in EVASION_WEIGHTS]
        receipt_path = stage_root / "development_selection_receipt.json"
        receipt = create_development_selection_receipt(receipt_path, summaries)
        for artifact in (receipt_path, receipt_path.with_suffix(".json.sha256")):
            manifest = update_manifest_artifact(stage_root, manifest, artifact)

        atomic_csv(stage_root / "epoch_history.csv", history_rows)
        atomic_csv(stage_root / "per_sample_scores.csv", score_rows)
        atomic_csv(stage_root / "observable_rows.csv", observable_rows)
        atomic_json(stage_root / "fold_receipts.json", fold_receipts)
        atomic_json(stage_root / "split_receipts.json", split_receipts)
        for artifact, count in (
            (stage_root / "epoch_history.csv", len(history_rows)),
            (stage_root / "per_sample_scores.csv", len(score_rows)),
            (stage_root / "observable_rows.csv", len(observable_rows)),
            (stage_root / "fold_receipts.json", None),
            (stage_root / "split_receipts.json", None),
        ):
            manifest = update_manifest_artifact(stage_root, manifest, artifact, row_count=count)

        selected_config = str(receipt["selected_configuration_id"])
        unselected = next(f"w{weight:g}" for weight in EVASION_WEIGHTS
                          if f"w{weight:g}" != selected_config)
        denied = False
        try:
            authorize_development_test_read(receipt_path, unselected)
        except PermissionError:
            denied = True
        if not denied:
            raise AssertionError("Synthetic test firewall admitted an unselected configuration")
        authorize_development_test_read(receipt_path, selected_config)
        selected_weight = float(receipt["selected_evasion_weight"])

        victims: list[dict[str, Any]] = []
        per_seed: list[dict[str, Any]] = []
        checkpoint_receipts: dict[str, Any] = {}
        test_calls: dict[str, int] = {str(s): 0 for s in DEVELOPMENT_SEEDS}
        for formal_seed in DEVELOPMENT_SEEDS:
            clean_path = stage_root / "clean_models/models" / f"seed_{formal_seed}_clean.pt"
            attacked_path = stage_root / f"seed_{formal_seed}" / selected_config / "attacked_projector.pt"
            atomic_torch_save(clean_path, {"seed": formal_seed, "state_dict": {"w": torch.tensor([1.0])}})
            atomic_torch_save(attacked_path, {"seed": formal_seed,
                                             "configuration_id": selected_config,
                                             "projector_state": {"w": torch.tensor([2.0])}})
            manifest = update_manifest_artifact(stage_root, manifest, clean_path)
            manifest = update_manifest_artifact(stage_root, manifest, attacked_path)
            checkpoint_receipts[str(formal_seed)] = {
                "clean_checkpoint": {"path": str(clean_path.relative_to(stage_root)),
                                     "sha256": sha256_file(clean_path)},
                "attacked_projector": {"path": str(attacked_path.relative_to(stage_root)),
                                       "sha256": sha256_file(attacked_path)},
            }
            test_calls[str(formal_seed)] += 1
            victims.append({
                "stage": "development", "seed": formal_seed,
                "configuration_id": selected_config, "selected_epoch": 1,
                "test_balanced_ca": 0.91, "test_asr": 0.81, "eligible": True,
            })
        for weight in EVASION_WEIGHTS:
            for row in selected_rows[float(weight)]:
                exact = exact_by_key[(int(row["seed"]), float(weight))]
                per_seed.append({
                    "stage": "development", "seed": int(row["seed"]),
                    "configuration_id": f"w{weight:g}", "selected_epoch": int(row["epoch"]),
                    "validation_balanced_ca": float(row["validation_balanced_ca"]),
                    "validation_asr": float(row["validation_asr"]),
                    "eligible": True, "tpr": float(exact["tpr"]), "fpr": float(exact["fpr"]),
                    **{name: int(exact[name]) for name in ("tp", "fn", "fp", "tn")},
                })
        atomic_csv(stage_root / "victim_metrics.csv", victims)
        atomic_csv(stage_root / "per_seed_metrics.csv", per_seed)
        atomic_json(stage_root / "checkpoint_receipts.json", checkpoint_receipts)
        _runtime._atomic_bytes(stage_root / "EXPERIMENT_REPORT.md",
                               b"# Synthetic P6-D v3 delivery shadow\n\nNo efficacy evidence.\n")
        for artifact, count in (
            (stage_root / "victim_metrics.csv", len(victims)),
            (stage_root / "per_seed_metrics.csv", len(per_seed)),
            (stage_root / "checkpoint_receipts.json", None),
            (stage_root / "EXPERIMENT_REPORT.md", None),
        ):
            manifest = update_manifest_artifact(stage_root, manifest, artifact, row_count=count)
        manifest["status"] = "synthetic_complete"
        atomic_json(stage_root / "manifest.json", manifest)

        reread_splits = json.loads((stage_root / "split_receipts.json").read_text())
        if not isinstance(reread_splits[str(DEVELOPMENT_SEEDS[0])]["trigger_receipt"]
                          ["poison_indices"], list):
            raise AssertionError("Nested NumPy array did not roundtrip as plain JSON")
        resumed = initialize_or_resume(stage_root, signature, immutable_hashes=immutable)
        if resumed["status"] != "synthetic_complete":
            raise AssertionError("Completed synthetic manifest did not resume exactly")
        import verify_p6d_final_observable_adaptive_v3 as independent
        replay = independent.verify_bundle(stage_root)
        required = {
            "environment.json", "development_selection_receipt.json",
            "development_selection_receipt.json.sha256", "epoch_history.csv",
            "per_sample_scores.csv", "observable_rows.csv", "fold_receipts.json",
            "split_receipts.json", "per_seed_metrics.csv", "victim_metrics.csv",
            "checkpoint_receipts.json", "EXPERIMENT_REPORT.md",
        }
        if not required.issubset(resumed["artifacts"]):
            raise AssertionError("Synthetic shadow manifest is incomplete")
        return {
            "status": "PASS", "synthetic_seed": seed,
            "json_numpy_roundtrip": True, "test_firewall_denied_unselected": denied,
            "authorized_test_calls": test_calls,
            "rows": {"history": len(history_rows), "scores": len(score_rows),
                     "observable": len(observable_rows), "per_seed": len(per_seed),
                     "victim": len(victims), "folds": len(fold_receipts)},
            "manifest_artifacts": len(resumed["artifacts"]),
            "independent_replay": replay["status"],
        }


_base_run_development_launcher = _runtime.run_development_launcher
_base_run_confirmation_launcher = _runtime.run_confirmation_launcher
_base_guarded_formal_stage = _runtime.guarded_formal_stage


def _preflight_formal(func):
    def guarded(*args, **kwargs):
        report = full_path_synthetic_shadow_self_check()
        if report["status"] != "PASS":
            raise RuntimeError("P6-D v3 full-path synthetic preflight failed")
        return func(*args, **kwargs)
    return guarded


_runtime.run_development_launcher = _preflight_formal(_base_run_development_launcher)
_runtime.run_confirmation_launcher = _preflight_formal(_base_run_confirmation_launcher)
_runtime.guarded_formal_stage = _preflight_formal(_base_guarded_formal_stage)


def main() -> None:
    # Formal entry points are wrapped above, so the full shadow runs exactly
    # once per formal invocation rather than once here and once in the wrapper.
    _runtime.main()


if __name__ == "__main__":
    main()
