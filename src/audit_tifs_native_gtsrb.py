#!/usr/bin/env python3
"""Create the frozen native-GTSRB feasibility receipt without downloading data."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/tifs_native_gtsrb_feasibility_v1.json"
OUTPUT = ROOT / "results/tifs_native_cifar_gtsrb_v1/gtsrb"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    project = ROOT.parent
    home = Path.home()
    configured_roots = [
        Path(value).expanduser()
        for value in (
            os.environ.get("GTSRB_DATA_ROOT"),
            os.environ.get("TORCH_HOME"),
        )
        if value
    ]
    candidate_data_roots = [
        *configured_roots,
        ROOT / "data/gtsrb_official",
        ROOT / "data",
        project / "paper5/datasets",
        home / ".cache/torchvision",
        home / "Downloads",
        home / "datasets",
        home / "data",
    ]
    expected_train = Path("gtsrb/GTSRB/Training")
    expected_test = Path("gtsrb/GTSRB/Final_Test/Images")
    usable_data = [
        str(path)
        for path in candidate_data_roots
        if (path / expected_train).is_dir() and (path / expected_test).is_dir()
    ]
    dataset_receipt: dict[str, Any] | None = None
    if usable_data:
        from torchvision.datasets import GTSRB

        selected_root = Path(usable_data[0])
        train = GTSRB(selected_root, split="train", download=False)
        test = GTSRB(selected_root, split="test", download=False)
        if len(train) != 26640 or len(test) != 12630:
            raise RuntimeError("GTSRB row-count mismatch")
        archive_rows = []
        for source in config["torchvision_0_22_1_sources"]:
            path = selected_root / "gtsrb" / source["archive"]
            actual_md5 = hashlib.md5(path.read_bytes()).hexdigest()
            actual_sha256 = sha256_file(path)
            if actual_md5 != source["md5"] or actual_sha256 != source["sha256"]:
                raise RuntimeError(f"GTSRB archive hash mismatch: {path}")
            archive_rows.append(
                {
                    "path": str(path),
                    "md5": actual_md5,
                    "sha256": actual_sha256,
                    "bytes": path.stat().st_size,
                }
            )
        dataset_receipt = {
            "root": str(selected_root),
            "train_rows": len(train),
            "test_rows": len(test),
            "train_class_count": len({int(label) for _, label in train._samples}),
            "test_class_count": len({int(label) for _, label in test._samples}),
            "archives": archive_rows,
        }
    checkpoint_roots = [ROOT / "results", project / "paper5"]
    checkpoint_candidates: list[str] = []
    for base in checkpoint_roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".pt", ".pth", ".ckpt"}:
                continue
            local_names = [
                path.name.lower(),
                path.parent.name.lower(),
                path.parent.parent.name.lower(),
            ]
            if any("gtsrb" in name or "gtrsb" in name for name in local_names):
                checkpoint_candidates.append(str(path))
    gate_pass = bool(usable_data and len(checkpoint_candidates) >= 3)
    if gate_pass:
        status = "READY_FOR_CONFIRMATION"
    elif usable_data:
        status = "DATA_READY_VICTIM_TRAINING_REQUIRED"
    else:
        status = "NOT_RUN_BLOCKED_NO_NATIVE_DATA"
    receipt = {
        "protocol_id": config["protocol_id"],
        "status": status,
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(CONFIG),
        "dataset_roots_checked": [str(path) for path in candidate_data_roots],
        "usable_native_dataset_roots": usable_data,
        "dataset_receipt": dataset_receipt,
        "checkpoint_roots_checked": [str(path) for path in checkpoint_roots],
        "native_checkpoint_candidates": checkpoint_candidates,
        "backdoorbench_support_files_present": len(
            list((ROOT / "external/BackdoorBench/config").rglob("*gtsrb*.yaml"))
        ),
        "interpretation": (
            "Official native GTSRB data are present and torchvision-readable, but no audited native "
            "victim checkpoint existed at audit time. A prespecified one-seed smoke and victim training "
            "are therefore required before confirmation."
            if usable_data and not gate_pass
            else (
                "BackdoorBench configuration support is code/configuration only and does not establish "
                "a cached native dataset or an audited trained victim."
                if not usable_data
                else "The frozen feasibility gate passed; a separate prespecified confirmation is permitted."
            )
        ),
    }
    write_json(OUTPUT / "feasibility_receipt.json", receipt)
    lines = [
        "# Native GTSRB feasibility audit",
        "",
        f"Status: `{status}`.",
        "",
        f"Usable native dataset roots: {len(usable_data)}.",
        f"Audited native victim checkpoints: {len(checkpoint_candidates)}.",
        f"BackdoorBench GTSRB configuration files: {receipt['backdoorbench_support_files_present']}.",
        "",
        receipt["interpretation"],
        "",
        "No MedMNIST, CIFAR relabeling, synthetic image, or literature-number substitution was used.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUTPUT / "audit.log").write_text(
        f"{receipt['audited_at']} {status} data_roots={len(usable_data)} checkpoints={len(checkpoint_candidates)}\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
