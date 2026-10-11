from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import run_pathmnist_terminal_staged_replay_v2 as staged_v2  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fake_output(tmp_path: Path, dataset_payload: bytes) -> Path:
    output = tmp_path / "formal"
    output.mkdir()
    for name in staged_v2.predecessor.TOP_LEVEL_FILES:
        (output / name).write_bytes((name + "\n").encode("utf-8"))
    (output / "features").mkdir()
    manifest = {
        "frozen_hashes": {"dataset_sha256": digest(dataset_payload)}
    }
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return output


def test_dataset_binding_matches_acquisition_and_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    payload = b"frozen-path-dataset"
    dataset = tmp_path / "pathmnist.npz"
    dataset.write_bytes(payload)
    output = fake_output(tmp_path, payload)
    acquisition = tmp_path / "acquisition.json"
    acquisition.write_text(
        json.dumps({"sha256": digest(payload), "byte_count": len(payload)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(staged_v2, "ACQUISITION_RECEIPT", acquisition)
    binding = staged_v2.bind_dataset(dataset, output)
    assert binding["sha256"] == digest(payload)
    assert binding["byte_count"] == len(payload)


def test_receipt_survives_dataset_unlink_after_frozen_verifier_returns(
    tmp_path: Path, monkeypatch
) -> None:
    payload = b"frozen-path-dataset"
    dataset = tmp_path / "pathmnist.npz"
    dataset.write_bytes(payload)
    output = fake_output(tmp_path, payload)
    acquisition = tmp_path / "acquisition.json"
    acquisition.write_text(
        json.dumps({"sha256": digest(payload), "byte_count": len(payload)}),
        encoding="utf-8",
    )
    amendment = tmp_path / "amendment.json"
    amendment.write_text("{}", encoding="utf-8")
    receipt = tmp_path / "receipt.json"

    monkeypatch.setattr(staged_v2, "ACQUISITION_RECEIPT", acquisition)
    monkeypatch.setattr(
        staged_v2,
        "validate_amendment",
        lambda _: {
            "base_contract_sha256": "base",
            "predecessor_amendment_sha256": "predecessor",
        },
    )

    def verify(stage: Path, supplied: Path, *, allow_nonformal: bool):
        assert supplied == dataset
        assert allow_nonformal is False
        supplied.unlink()
        return {
            "status": "PASS",
            "formal_evidence": True,
            "failures": [],
            "verified_seeds": [103, 104, 105, 106, 107],
            "verified_method_rows": 25,
        }

    monkeypatch.setattr(staged_v2.frozen, "verify", verify)
    result = staged_v2.replay(output, dataset, amendment, receipt)
    assert result["status"] == "PASS"
    assert result["dataset_present_after_replay"] is False
    assert result["dataset_reopened_after_replay"] is False
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "PASS"
