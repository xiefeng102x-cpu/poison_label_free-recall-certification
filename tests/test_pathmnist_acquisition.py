from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import verify_pathmnist_acquisition as acquisition  # noqa: E402


def test_frozen_protocol_has_bounded_official_contract() -> None:
    protocol = json.loads(acquisition.PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["official_source"]["expected_md5"] == "a8b06965200029087d5bd730944a56c1"
    assert protocol["official_source"]["maximum_accepted_bytes"] == 230000000
    assert protocol["dataset_contract"]["expected_split_counts"] == {
        "train": 89996,
        "val": 10004,
        "test": 7180,
    }
    assert protocol["dataset_contract"]["expected_label_set"] == list(range(9))


def test_file_digest_matches_known_bytes(tmp_path: Path) -> None:
    path = tmp_path / "bytes.bin"
    path.write_bytes(b"pathmnist-contract")
    assert acquisition.file_digest(path, "md5") == "355c3438e3a4c745a3ad951f936206f4"
    assert acquisition.sha256_file(path) == "c7976121eaadd82da49324e8efab677065a427487007295ef91b86b5755ce92e"


def test_verify_rejects_wrong_file_before_npz_load(tmp_path: Path) -> None:
    path = tmp_path / "pathmnist.npz"
    np.savez(
        path,
        train_images=np.zeros((1, 28, 28, 3), dtype=np.uint8),
        train_labels=np.zeros((1, 1), dtype=np.uint8),
        val_images=np.zeros((1, 28, 28, 3), dtype=np.uint8),
        val_labels=np.zeros((1, 1), dtype=np.uint8),
        test_images=np.zeros((1, 28, 28, 3), dtype=np.uint8),
        test_labels=np.zeros((1, 1), dtype=np.uint8),
    )
    receipt = acquisition.verify(path)
    assert receipt["status"] == "FAIL"
    assert "md5_mismatch" in receipt["failures"]
    assert receipt["training_authorized"] is False
