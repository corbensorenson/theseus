from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_task7_dependency_canary as owner  # noqa: E402


def test_preflight_seals_task7_dependency_only_authority() -> None:
    path = ROOT / "configs" / "theseus_vcm_task7_dependency_canary.json"
    report = owner.preflight(json.loads(path.read_text()), path)
    assert report["trigger_state"] == "PAUSED"
    assert report["state"] == "READY_FOR_TASK_7_EXACT_PNPM_DEPENDENCY_CANARY"
    assert report["observed_parent_target_dependency_identities"]["parent"] == report["observed_parent_target_dependency_identities"]["target"]
    assert report["dependency_installations"] == 0
    assert report["repository_runner_executions"] == 0
    assert report["candidate_or_control_calls"] == 0
    assert report["external_reference_calls"] == 0


def test_commands_disable_scripts_and_offline_replay_adds_only_offline() -> None:
    config = json.loads((ROOT / "configs" / "theseus_vcm_task7_dependency_canary.json").read_text())
    online = config["commands"]["online_acquisition_args"]
    assert "--ignore-scripts" in online
    assert config["commands"]["offline_replay_args"] == [*online, "--offline"]
    assert config["authority"]["repository_runner_execution_authorized"] is False
