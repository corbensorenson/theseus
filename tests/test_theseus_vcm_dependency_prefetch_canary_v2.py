from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_dependency_prefetch_canary_v2 as owner  # noqa: E402


def test_v2_preflight_binds_v1_red_and_only_repairs_file_limit_coupling() -> None:
    path = ROOT / "configs" / "theseus_vcm_dependency_prefetch_canary_v2.json"
    report = owner.preflight(json.loads(path.read_text()), path)
    assert report["trigger_state"] == "PAUSED"
    assert report["predecessor_report"]["sha256"] == "39b892cede02b13e0115984d960ffb4c95844fd7a44abb60f6f5a945bd0f2a42"
    assert report["boundary_repair"] == {
        "captured_output_mib": 8,
        "maximum_single_written_file_mib": 4096,
        "maximum_retained_bytes": 4294967296,
        "captured_output_monitored_independently": True,
    }
    assert report["dependency_installations"] == 0
    assert report["repository_runner_executions"] == 0


def test_v2_keeps_scientific_task_and_command_contract_identical_to_v1() -> None:
    v1 = json.loads((ROOT / "configs" / "theseus_vcm_dependency_prefetch_canary.json").read_text())
    v2 = json.loads((ROOT / "configs" / "theseus_vcm_dependency_prefetch_canary_v2.json").read_text())
    for key in ("reports", "task", "archives", "tools", "commands", "retained_store", "authority"):
        assert v2[key] == v1[key]
