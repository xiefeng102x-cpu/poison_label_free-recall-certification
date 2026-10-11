#!/usr/bin/env python3
"""Frozen paths and integrity checks for the prospective protocol-correct block."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CERT_ROOT = ROOT / "auditability_certificate_v1"
ANALYSIS = CERT_ROOT / "analysis"
CONFIG = ROOT / "configs/pathmnist_prospective_protocol_correct_v2.json"
PREREG = CERT_ROOT / "PREREG_PATHMNIST_PROSPECTIVE_PROTOCOL_CORRECT_V2.md"
FREEZE = CERT_ROOT / "PATHMNIST_PROSPECTIVE_PROTOCOL_CORRECT_FREEZE_V2.json"
DATASET = Path("/private/tmp/paper14-data/pathmnist.npz")
SOURCE = ROOT / "results/pathmnist_prospective_protocol_correct_v2_source"
OUTPUT = ROOT / "results/pathmnist_prospective_protocol_correct_v2"
KEY_FILE = Path("/private/tmp/paper14-prospective-protocol-v2.key")

TRAIN_WRAPPER = ROOT / "src/run_pathmnist_prospective_protocol_correct_v2.py"
REGISTRAR_WRAPPER = ANALYSIS / "prepare_pathmnist_prospective_protocol_v2.py"
ISSUER_WRAPPER = ANALYSIS / "issue_pathmnist_prospective_protocol_v2.py"
VERIFIER_WRAPPER = ANALYSIS / "verify_pathmnist_prospective_protocol_v2.py"
AUDIT_WRAPPER = ANALYSIS / "audit_pathmnist_prospective_protocol_v2.py"
BASE_TRAINER = ROOT / "src/run_pathmnist_blind_e2e_multidetector.py"
BASE_REGISTRAR = ANALYSIS / "prepare_protocol_correct_observations.py"
BASE_ISSUER = ANALYSIS / "issue_protocol_correct_certificates.py"
BASE_VERIFIER = ANALYSIS / "verify_protocol_correct_rerun.py"
BASE_AUDIT = ANALYSIS / "audit_protocol_correct_outputs.py"
PROTOCOL_COMMON = ANALYSIS / "protocol_correct_common.py"
CERTIFICATE_LIBRARY = CERT_ROOT / "cert/certificate.py"
TRAINING_ENGINE = ROOT / "src/run_pathmnist_official_weight_detector_confirmation.py"
ARCHITECTURE = ROOT / "src/pathmnist_official_resnet18.py"
TRAINING_CORE = ROOT / "src/pathmnist_external_validation.py"
SMOKE_RECEIPT = ROOT / "results/pathmnist_real_input_mps_smoke_v1/smoke_receipt.json"
SMOKE_VERIFICATION = ROOT / "results/pathmnist_real_input_mps_smoke_v1/smoke_verification.json"
WEIGHT_RECEIPT = ROOT / "results/top_journal_program_v1/pathmnist_official_weights_acquisition_receipt.json"
TEST_FILE = ROOT / "tests/test_pathmnist_prospective_protocol_v2.py"

SEEDS = (8301, 8302, 8303, 8304, 8305)
WEIGHT_MAP = {
    8301: "resnet18_28_1.pth",
    8302: "resnet18_28_2.pth",
    8303: "resnet18_28_1.pth",
    8304: "resnet18_28_2.pth",
    8305: "resnet18_28_1.pth",
}
ACK = "OPEN_PATHMNIST_PROSPECTIVE_PROTOCOL_CORRECT_V2_SEEDS_8301_8305"
EXPECTED_DATASET_SHA256 = "81823f52dc622e69db2db4c72f8e8e617938dd6864d3c1f23d4e49724a28ea72"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_code_hashes() -> dict[str, str]:
    return {
        "config_sha256": sha256(CONFIG),
        "prereg_sha256": sha256(PREREG),
        "prospective_common_sha256": sha256(Path(__file__)),
        "training_wrapper_sha256": sha256(TRAIN_WRAPPER),
        "registrar_wrapper_sha256": sha256(REGISTRAR_WRAPPER),
        "issuer_wrapper_sha256": sha256(ISSUER_WRAPPER),
        "verifier_wrapper_sha256": sha256(VERIFIER_WRAPPER),
        "audit_wrapper_sha256": sha256(AUDIT_WRAPPER),
        "base_trainer_sha256": sha256(BASE_TRAINER),
        "base_registrar_sha256": sha256(BASE_REGISTRAR),
        "base_issuer_sha256": sha256(BASE_ISSUER),
        "base_verifier_sha256": sha256(BASE_VERIFIER),
        "base_audit_sha256": sha256(BASE_AUDIT),
        "protocol_common_sha256": sha256(PROTOCOL_COMMON),
        "certificate_library_sha256": sha256(CERTIFICATE_LIBRARY),
        "training_engine_sha256": sha256(TRAINING_ENGINE),
        "architecture_sha256": sha256(ARCHITECTURE),
        "training_core_sha256": sha256(TRAINING_CORE),
        "smoke_receipt_sha256": sha256(SMOKE_RECEIPT),
        "smoke_verification_sha256": sha256(SMOKE_VERIFICATION),
        "weight_receipt_sha256": sha256(WEIGHT_RECEIPT),
        "test_file_sha256": sha256(TEST_FILE),
    }


def verify_freeze(*, key: bytes | None = None, require_dataset: bool = True) -> dict[str, Any]:
    frozen = load_json(FREEZE)
    if frozen.get("status") != "FROZEN_BEFORE_VICTIM_TRAINING_AND_CERTIFICATE_ISSUANCE":
        raise RuntimeError("prospective freeze status mismatch")
    if tuple(frozen.get("formal_seeds", [])) != SEEDS:
        raise RuntimeError("formal seed freeze mismatch")
    for name, value in expected_code_hashes().items():
        if frozen.get("hashes", {}).get(name) != value:
            raise RuntimeError(f"frozen hash mismatch: {name}")
    if frozen.get("dataset_sha256") != EXPECTED_DATASET_SHA256:
        raise RuntimeError("frozen dataset digest mismatch")
    if require_dataset:
        if not DATASET.is_file() or sha256(DATASET) != EXPECTED_DATASET_SHA256:
            raise RuntimeError("formal dataset is missing or has the wrong SHA-256")
    if key is not None:
        commitment = hashlib.sha256(key).hexdigest()
        if commitment != frozen.get("key_commitment_sha256"):
            raise RuntimeError("fresh key does not open the pre-training commitment")
    return frozen


def verify_source_handoff() -> dict[str, Any]:
    manifest = load_json(SOURCE / "manifest.json")
    if manifest.get("status") != "SCORE_HANDOFF_COMPLETE":
        raise RuntimeError("prospective score handoff is incomplete")
    if tuple(manifest.get("seeds", [])) != SEEDS:
        raise RuntimeError("source seed list mismatch")
    if not manifest.get("all_victims_eligible"):
        raise RuntimeError("one or more prospective victims failed eligibility")
    if manifest.get("blind_scores_sha256") != sha256(SOURCE / "blind_scores.csv"):
        raise RuntimeError("sealed score handoff drift")
    if manifest.get("private_truth_sha256") != sha256(SOURCE / "private_truth.csv"):
        raise RuntimeError("private truth handoff drift")
    return manifest
