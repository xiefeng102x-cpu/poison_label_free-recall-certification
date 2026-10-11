#!/usr/bin/env python3
"""Forensic, stage-aware replay of the completed P6-D v3 confirmation bundle.

The hash-pinned v3 verifier correctly replays the frozen numerical endpoint,
epoch selections, victim gate, confirmation lock and bootstrap.  Its additional
delivery check, however, hard-codes the development seed block for both stages.
This verifier leaves the formal bundle and pinned sources untouched.  It calls
the pinned independent numerical replay and replaces only that post-replay
receipt check with a manifest-derived, stage-aware seed assertion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from staged_provenance import effective_sha256, effective_size


PAPER14_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PAPER14_ROOT / "results/p6d_final_observable_adaptive_v3/confirmation"
DEVELOPMENT_SEEDS = (116, 117, 118)
CONFIRMATION_SEEDS = (119, 120, 121, 122, 123)


def _expected_seeds(manifest: Mapping[str, Any]) -> tuple[int, ...]:
    signature = manifest.get("run_signature", {})
    stage = signature.get("stage")
    expected = {
        "development": DEVELOPMENT_SEEDS,
        "confirmation": CONFIRMATION_SEEDS,
    }.get(stage)
    if expected is None:
        raise AssertionError(f"Unknown P6-D stage: {stage!r}")
    if tuple(int(seed) for seed in signature.get("seeds", ())) != expected:
        raise AssertionError("Manifest seed block differs from the frozen stage")
    return expected


def _validate_stage_delivery(output: Path, manifest: Mapping[str, Any], pinned: Any) -> dict[str, Any]:
    expected = _expected_seeds(manifest)
    splits = json.loads((output / "split_receipts.json").read_text(encoding="utf-8"))
    if sorted(int(seed) for seed in splits) != list(expected):
        raise AssertionError("Stage-aware split receipt seed block mismatch")
    for seed, receipt in splits.items():
        poison = receipt.get("trigger_receipt", {}).get("poison_indices")
        if not isinstance(poison, list) or not all(isinstance(value, int) for value in poison):
            raise AssertionError(f"Split receipt is not plain lossless JSON: {seed}")

    checkpoints = json.loads((output / "checkpoint_receipts.json").read_text(encoding="utf-8"))
    if sorted(int(seed) for seed in checkpoints) != list(expected):
        raise AssertionError("Stage-aware checkpoint receipt seed block mismatch")
    verified = 0
    for seed, kinds in checkpoints.items():
        for kind, record in kinds.items():
            path = output / record["path"]
            if not path.is_file() or pinned.sha256_file(path) != record["sha256"]:
                raise AssertionError(f"Checkpoint receipt mismatch: {seed}/{kind}")
            verified += 1
    return {
        "stage": manifest["run_signature"]["stage"],
        "seeds": list(expected),
        "split_receipts": len(splits),
        "checkpoint_receipts": verified,
        "plain_json": True,
    }


def _compare_summary(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    if set(left) != set(right):
        raise AssertionError("Confirmation summary fields differ from independent replay")
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if abs(float(a) - float(b)) > 1e-12:
                raise AssertionError(f"Confirmation summary mismatch: {key}")
        elif a != b:
            raise AssertionError(f"Confirmation summary mismatch: {key}")


def verify(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = output.resolve()
    import verify_p6d_final_observable_adaptive_v3 as pinned

    def ledger_aware_sha256(path: Path) -> str:
        return effective_sha256(path, PAPER14_ROOT)

    def ledger_aware_manifest(stage_output: Path) -> dict[str, Any]:
        manifest = json.loads(
            (stage_output / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("protocol_id") != pinned._runtime.PROTOCOL_ID:
            raise AssertionError("Manifest protocol identifier mismatch")
        signature = manifest.get("run_signature", {})
        if signature and signature.get("protocol_sha256") != ledger_aware_sha256(
            pinned._runtime.PROTOCOL_PATH
        ):
            raise AssertionError("Manifest protocol hash mismatch")
        for relative, record in manifest.get("artifacts", {}).items():
            artifact = stage_output / relative
            if not artifact.is_file():
                raise AssertionError(f"Manifest artifact is missing: {relative}")
            if effective_size(artifact, PAPER14_ROOT) != int(record["bytes"]):
                raise AssertionError(f"Manifest artifact size mismatch: {relative}")
            if ledger_aware_sha256(artifact) != record["sha256"]:
                raise AssertionError(f"Manifest artifact hash mismatch: {relative}")
        return manifest

    pinned.sha256_file = ledger_aware_sha256
    pinned._runtime.sha256_file = ledger_aware_sha256
    pinned._runtime._verify_manifest = ledger_aware_manifest

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise AssertionError("Confirmation manifest is not complete")
    if manifest.get("protocol_id") != pinned.PROTOCOL_ID:
        raise AssertionError("Confirmation protocol identifier mismatch")

    # This is the original independent numerical replay beneath the faulty
    # v3 post-replay delivery wrapper; it does not import training code.
    replay = pinned._runtime.verify_bundle(output)
    pinned.validate_observable_schema(
        pinned._runtime.read_csv(output / "observable_rows.csv")
    )
    delivery = _validate_stage_delivery(output, manifest, pinned)
    if replay.get("status") != "valid" or "confirmation" not in replay:
        raise AssertionError("Pinned independent confirmation replay did not complete")
    recorded = json.loads((output / "confirmation_summary.json").read_text(encoding="utf-8"))
    _compare_summary(replay["confirmation"], recorded)
    return {
        "status": "PASS_STAGE_AWARE_FORENSIC_REPLAY",
        "formal_bundle_unchanged": True,
        "pinned_numerical_replay": replay,
        "delivery_replay": delivery,
        "recorded_summary_matches_replay": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
