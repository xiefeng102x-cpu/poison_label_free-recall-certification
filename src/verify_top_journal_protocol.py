#!/usr/bin/env python3
"""Static integrity checks for the frozen top-journal evidence protocol."""

from __future__ import annotations

import json
from pathlib import Path


PAPER14_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PAPER14_ROOT / "configs/top_journal_evidence_matrix_v1.json"
OUTPUT = PAPER14_ROOT / "results/top_journal_program_v1/protocol_receipt.json"


def main() -> None:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    development = set(protocol["development_seeds"])
    confirmation = set(protocol["fresh_confirmation_seeds"])
    pilot = set(protocol["pilot_seeds"])
    assert development.isdisjoint(confirmation)
    assert pilot <= development
    assert len(confirmation) == 10
    assert len(protocol["attack_families"]) >= 6
    assert len(protocol["adaptive_attacks"]) >= 3
    assert all(len(values) >= 2 for values in protocol["architectures"].values())
    assert len(protocol["required_baselines"]) >= 4
    assert len(protocol["required_ablations"]) >= 7
    assert sum(protocol["token_allocation_percent"].values()) == 100
    assert protocol["fixed_qml_confirmation_view"] == "checkpoint_drift_local_xz16"
    assert protocol["failed_ablation_view"] == "checkpoint_drift_local_z8"
    assert protocol["primary_detector_gate"] == {
        "minimum_pooled_tpr": 0.9,
        "maximum_pooled_fpr": 0.05,
    }
    receipt = {
        "status": "PASS",
        "protocol_id": protocol["protocol_id"],
        "development_seed_count": len(development),
        "fresh_confirmation_seed_count": len(confirmation),
        "attack_family_count": len(protocol["attack_families"]),
        "adaptive_attack_count": len(protocol["adaptive_attacks"]),
        "architecture_count": sum(len(values) for values in protocol["architectures"].values()),
        "baseline_count": len(protocol["required_baselines"]),
        "token_allocation_total": sum(protocol["token_allocation_percent"].values()),
        "confirmation_is_disjoint": development.isdisjoint(confirmation),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
