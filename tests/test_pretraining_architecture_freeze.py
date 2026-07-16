from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pretraining_architecture_freeze as freeze


def test_freeze_manifest_covers_finite_docket_and_architecture_sources() -> None:
    config = freeze.load_config()
    dispositions = freeze.architecture_dispositions(config)
    manifest = freeze.artifact_manifest(config)
    assert dispositions["required_count"] == dispositions["ready_count"] == 13
    assert len(manifest) >= 45
    assert "scripts/standard_causal_transformer_model.py" in manifest
    assert "configs/onecell_rwm_pretraining_disposition.json" in manifest


def test_freeze_refuses_to_claim_replay_without_executing_it() -> None:
    with pytest.raises(freeze.ArchitectureFreezeFault, match="independent_replay_required"):
        freeze.build_report(freeze.load_config(), execute_replays=False)
