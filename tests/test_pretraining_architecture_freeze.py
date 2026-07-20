from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pretraining_architecture_freeze as freeze


def test_freeze_accepts_banked_kerc_mechanics_without_claiming_behavior() -> None:
    config = freeze.load_config()
    manifest = freeze.artifact_manifest(config)
    dispositions = freeze.architecture_dispositions(config)
    kerc = dispositions["rows"][
        "planned.kernel_english_hierarchical_residual_compiler_v1"
    ]

    assert dispositions["ready_count"] == dispositions["required_count"]
    assert kerc["status"] == "pretraining_wired_behavior_qualification_pending"
    assert kerc["evidence"] is None
    assert kerc["negative_disposition"] is None
    assert len(manifest) >= 50
    assert "scripts/standard_causal_transformer_model.py" in manifest
    assert "configs/onecell_rwm_pretraining_disposition.json" in manifest


def test_freeze_binds_generated_effect_and_governance_receipts() -> None:
    receipts = freeze.receipt_manifest(freeze.load_config())

    assert set(receipts) == {
        "reports/governance_rights_receipt_suite.json",
        "reports/theseus_assistant_effect_complete_canary.json",
    }
    assert all(len(row["sha256"]) == 64 and row["bytes"] > 0 for row in receipts.values())


def test_freeze_refuses_to_claim_replay_without_executing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        freeze,
        "architecture_dispositions",
        lambda _config: {"required_count": 1, "ready_count": 1, "rows": {}},
    )
    with pytest.raises(freeze.ArchitectureFreezeFault, match="independent_replay_required"):
        freeze.build_report(freeze.load_config(), execute_replays=False)
