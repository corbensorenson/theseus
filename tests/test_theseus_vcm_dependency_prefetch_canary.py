from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_dependency_prefetch_canary as owner  # noqa: E402


def test_preflight_seals_only_task_three_dependency_work() -> None:
    path = ROOT / "configs" / "theseus_vcm_dependency_prefetch_canary.json"
    report = owner.preflight(json.loads(path.read_text()), path)
    assert report["trigger_state"] == "PAUSED"
    assert report["dependency_installations"] == 0
    assert report["repository_runner_executions"] == 0
    assert report["candidate_or_control_calls"] == 0
    assert report["observed_parent_target_dependency_identities"]["parent"] == report["observed_parent_target_dependency_identities"]["target"]


def test_lock_integrities_bind_the_two_exact_packages() -> None:
    config = json.loads((ROOT / "configs" / "theseus_vcm_dependency_prefetch_canary.json").read_text())
    integrities = owner.expected_lock_integrities(config)
    assert set(integrities) == {"node_modules/@vscode/tree-sitter-wasm", "node_modules/esbuild-wasm"}
    assert all(len(value) == 128 for value in integrities.values())


def test_online_and_offline_commands_disable_lifecycle_scripts() -> None:
    config = json.loads((ROOT / "configs" / "theseus_vcm_dependency_prefetch_canary.json").read_text())
    online = config["commands"]["online_acquisition_args"]
    offline = config["commands"]["offline_replay_args"]
    assert "--ignore-scripts" in online
    assert offline == [*online, "--offline"]
