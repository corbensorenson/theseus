from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pretraining_architecture_freeze as freeze


def test_freeze_is_invalidated_until_faithful_kerc_closes() -> None:
    config = freeze.load_config()
    manifest = freeze.artifact_manifest(config)
    with pytest.raises(
        freeze.ArchitectureFreezeFault,
        match="planned.kernel_english_hierarchical_residual_compiler_v1",
    ):
        freeze.architecture_dispositions(config)
    assert len(manifest) >= 45
    assert "scripts/standard_causal_transformer_model.py" in manifest
    assert "configs/onecell_rwm_pretraining_disposition.json" in manifest


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
