#!/usr/bin/env python3
"""Freeze the GTSRB v1.1 execution-only MPS layout compatibility revision."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCIENCE_KEYS = (
    "dataset",
    "seeds",
    "split",
    "victim",
    "trusted_split",
    "detectors",
    "execution",
    "victim_eligibility",
    "minimum_eligible_victims",
    "primary_detector_gate",
    "no_seed_replacement",
    "stopping_rule",
)
ATTACK_SCIENCE_KEYS = (
    "name",
    "target_class",
    "poison_fraction_per_non_target_training_class",
    "poison_selection_rng",
    "patch_size",
    "patch_alpha",
    "patch_rgb_uint8",
    "decision_unit",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scientific_projection(protocol: dict[str, Any]) -> dict[str, Any]:
    projection = {key: protocol[key] for key in SCIENCE_KEYS}
    projection["attack"] = {
        key: protocol["attack"][key] for key in ATTACK_SCIENCE_KEYS
    }
    return projection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--previous-protocol",
        type=Path,
        default=ROOT / "configs/tifs_native_gtsrb_confirmation_v1.json",
    )
    parser.add_argument(
        "--failure-record",
        type=Path,
        default=ROOT
        / "results/tifs_native_cifar_gtsrb_v1/gtsrb/EXECUTION_FAILURE_MPS_LAYOUT.json",
    )
    parser.add_argument(
        "--failed-run-log",
        type=Path,
        default=ROOT
        / "results/tifs_native_cifar_gtsrb_v1/gtsrb/run_v1_failed_mps_layout.log",
    )
    parser.add_argument(
        "--mps-self-test",
        type=Path,
        default=ROOT
        / "results/tifs_native_cifar_gtsrb_v1/gtsrb/MPS_FORWARD_BACKWARD_SELF_TEST.json",
    )
    parser.add_argument(
        "--runner", type=Path, default=ROOT / "src/run_tifs_native_gtsrb.py"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "configs/tifs_native_gtsrb_confirmation_v1_1.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite frozen protocol: {args.output}")

    previous = read_json(args.previous_protocol)
    failure = read_json(args.failure_record)
    self_test = read_json(args.mps_self_test)
    if previous["protocol_id"] != "tifs_native_gtsrb_confirmation_v1":
        raise RuntimeError("unexpected previous protocol")
    if failure.get("completed_seed_artifacts_present") is not False:
        raise RuntimeError("failure record does not certify absence of completed seeds")
    result_root = args.failed_run_log.parent
    forbidden = [
        result_root / "victim_metrics.csv",
        result_root / "detector_metrics.csv",
        result_root / "models/seed_101_suspect.pt",
        result_root / "receipts/seed_101.json",
        result_root / "per_sample/seed_101_scores.csv",
    ]
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise RuntimeError(f"refusing execution revision after seed result creation: {present}")
    runner_sha256 = sha256_file(args.runner)
    if self_test.get("status") != "pass" or self_test.get("actual_device") != "mps":
        raise RuntimeError("final MPS forward/backward self-test did not pass")
    if self_test.get("runner_sha256") != runner_sha256:
        raise RuntimeError("MPS self-test was not run with the final runner")

    revised = copy.deepcopy(previous)
    revised["protocol_id"] = "tifs_native_gtsrb_confirmation_v1_1"
    revised["frozen_at"] = datetime.now(timezone.utc).astimezone().isoformat()
    revised["freeze_method"] = (
        "programmatic datetime.now(timezone.utc).astimezone().isoformat(); "
        "execution-only compatibility revision; refuse overwrite"
    )
    revised["runner"] = {
        "path": str(args.runner.resolve()),
        "sha256": runner_sha256,
    }
    revised["attack"].pop("candidate_cohort", None)
    revised["attack"].update(
        {
            "conceptual_candidate_population": (
                "all 23,976 clean records in the frozen 90% training split plus "
                "2,264 selected poison records; total 26,240"
            ),
            "detector_defined_target_scope": (
                "1,350 legitimate target-class clean training records plus all 2,264 "
                "poison records assigned to the target; total 3,614 scored rows"
            ),
            "non_target_rule": (
                "the 22,626 legitimate non-target clean training records are outside "
                "the target-assigned detector score domain and are forced unflagged by "
                "the frozen label-conditional audit rule"
            ),
            "per_sample_storage": (
                "per-sample score files contain only the 3,614 detector-defined rows; "
                "they do not claim scores for the forced-unflagged non-target clean rows"
            ),
            "count_invariants_per_seed": {
                "clean_training_rows": 23976,
                "poison_rows": 2264,
                "conceptual_candidate_rows": 26240,
                "defined_target_clean_rows": 1350,
                "defined_poison_rows": 2264,
                "detector_defined_rows": 3614,
                "forced_unflagged_non_target_clean_rows": 22626,
                "target_clean_fpr_denominator": 1350,
                "all_clean_fpr_denominator": 23976,
                "identity": "3614 + 22626 = 26240 and 1350 + 22626 = 23976",
            },
        }
    )
    old_science = scientific_projection(previous)
    new_science = scientific_projection(revised)
    if old_science != new_science:
        raise RuntimeError("scientific protocol changed in an execution-only revision")
    science_sha256 = canonical_sha256(old_science)
    revised["compatibility_revision"] = {
        "revision": "v1.1",
        "scope": "execution-only MPS tensor-layout compatibility",
        "previous_protocol_path": str(args.previous_protocol.resolve()),
        "previous_protocol_sha256": sha256_file(args.previous_protocol),
        "failure_record_path": str(args.failure_record.resolve()),
        "failure_record_sha256": sha256_file(args.failure_record),
        "failed_run_log_path": str(args.failed_run_log.resolve()),
        "failed_run_log_sha256": sha256_file(args.failed_run_log),
        "mps_forward_backward_self_test_path": str(args.mps_self_test.resolve()),
        "mps_forward_backward_self_test_sha256": sha256_file(args.mps_self_test),
        "old_runner_sha256": previous["runner"]["sha256"],
        "new_runner_sha256": runner_sha256,
        "code_change": (
            "to_tensor now calls contiguous() immediately after NHWC-to-NCHW permute; "
            "runner also accepts this v1.1 protocol id and exposes an audited one-step "
            "MPS forward/backward self-test"
        ),
        "scientific_parameters_and_seeds_unchanged": True,
        "scientific_projection_sha256": science_sha256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(revised, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "frozen",
                "path": str(args.output.resolve()),
                "sha256": sha256_file(args.output),
                "frozen_at": revised["frozen_at"],
                "runner_sha256": runner_sha256,
                "scientific_projection_sha256": science_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
