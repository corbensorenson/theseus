from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p4v2r2_autonomous_launch as launcher  # noqa: E402


CONFIG_PATH = ROOT / "configs/theseus_p4v2r2_autonomous_launch.json"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def green_campaign() -> dict:
    return {
        "trigger_state": "GREEN",
        "complete_tasks": 0,
        "pending_tasks": 10,
        "model_calls_retained": 0,
    }


def test_config_has_no_user_or_quality_token_gate() -> None:
    value = config()
    assert launcher.validate_config(value) == []
    authority = value["authority"]
    assert authority["user_or_operator_approval_required"] is False
    assert authority["project_selected_quality_token_cap_allowed"] is False
    assert authority["physical_boundary_is_negative_evidence"] is False


def test_exact_sealed_bindings_pass_before_candidate_generation() -> None:
    report = launcher.audit_bindings(config())
    assert report["passed"] is True
    assert report["faults"] == []


def test_machine_predicates_authorize_without_a_user_gate() -> None:
    report = launcher.preflight(
        config(),
        config_path=CONFIG_PATH,
        power_override={"external_connected": True, "discharging": False},
        memory_override={"passed": True},
        disk_override={"passed": True},
        jobs_override=[],
        campaign_override=green_campaign(),
        lease_exists_override=False,
    )
    assert report["trigger_state"] == "GREEN"
    assert report["launch_authorized"] is True
    assert report["failed_gates"] == []


def test_battery_or_competing_accelerator_pauses_without_consumption() -> None:
    report = launcher.preflight(
        config(),
        config_path=CONFIG_PATH,
        power_override={"external_connected": False, "discharging": True},
        memory_override={"passed": True},
        disk_override={"passed": True},
        jobs_override=[{"pid": 7, "command": "training"}],
        campaign_override=green_campaign(),
        lease_exists_override=False,
    )
    assert report["trigger_state"] == "PAUSED"
    assert report["launch_authorized"] is False
    assert "external_power_physically_connected" in report["failed_gates"]
    assert "battery_not_discharging" in report["failed_gates"]
    assert "no_competing_accelerator_job" in report["failed_gates"]


def test_disk_envelope_is_derived_from_bound_sources_and_context_not_a_quality_cap() -> None:
    report = launcher.disk_status(config())
    assert report["passed"] is True
    assert report["source_fixture_bytes"] > 0
    assert report["maximum_retained_output_bytes"] == 60 * 262_144 * 8
    assert report["required_bytes"] > report["maximum_retained_output_bytes"]


def test_waiter_pauses_without_deadline_then_executes_when_machine_turns_green(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paused = {
        "trigger_state": "PAUSED",
        "launch_authorized": False,
        "campaign_audit": green_campaign(),
        "failed_gates": ["external_power_physically_connected"],
        "faults": [],
    }
    green = {
        "trigger_state": "GREEN",
        "launch_authorized": True,
        "campaign_audit": green_campaign(),
        "failed_gates": [],
        "faults": [],
    }
    states = iter((paused, green))
    monkeypatch.setattr(launcher, "preflight", lambda *_args, **_kwargs: next(states))
    monkeypatch.setattr(launcher.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        launcher,
        "execute_once",
        lambda *_args, **_kwargs: {
            "trigger_state": "GREEN",
            "launch_authorized": True,
            "final_campaign_audit": {
                "complete_tasks": 10,
                "pending_tasks": 0,
            },
        },
    )
    result = launcher.wait_and_execute(
        config(),
        config_path=CONFIG_PATH,
        out=tmp_path / "wait.json",
        poll_seconds=30.0,
    )
    assert result["trigger_state"] == "GREEN"
    assert json.loads((tmp_path / "wait.json").read_text())["trigger_state"] == "GREEN"
