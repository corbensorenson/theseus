from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p4v2r2r2_autonomous_launch as launcher  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "theseus_p4v2r2r2_autonomous_launch.json"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_launch_is_exact_and_has_no_human_or_token_gate() -> None:
    value = config()

    assert launcher.validate_config(value) == []
    assert launcher.audit_bindings(value)["passed"] is True
    assert value["authority"]["user_or_operator_approval_required"] is False
    assert value["authority"]["project_selected_quality_token_cap_allowed"] is False


def test_machine_predicates_authorize_the_unconsumed_campaign() -> None:
    report = launcher.preflight(
        config(),
        config_path=CONFIG_PATH,
        overrides={
            "power": {"external_connected": True, "discharging": False},
            "memory": {"passed": True},
            "disk": {"passed": True},
            "runtime": {"passed": True},
            "metal": {"passed": True},
            "jobs": [],
            "lease_exists": False,
            "campaign": {"trigger_state": "GREEN", "pending_tasks": 10},
        },
    )
    assert report["trigger_state"] == "GREEN"
    assert report["launch_authorized"] is True
    assert report["failed_gates"] == []


def test_physical_resource_failure_pauses_without_user_gate() -> None:
    report = launcher.preflight(
        config(),
        config_path=CONFIG_PATH,
        overrides={
            "power": {"external_connected": False, "discharging": True},
            "memory": {"passed": True},
            "disk": {"passed": True},
            "runtime": {"passed": True},
            "metal": {"passed": True},
            "jobs": [{"pid": 7}],
            "lease_exists": False,
            "campaign": {"trigger_state": "GREEN", "pending_tasks": 10},
        },
    )
    assert report["trigger_state"] == "PAUSED"
    assert "external_power_physically_connected" in report["failed_gates"]
    assert "no_competing_accelerator_job" in report["failed_gates"]


def test_wait_mode_retries_machine_pause_without_a_human_gate(monkeypatch) -> None:
    paused = {
        "created_utc": "2026-08-02T00:00:00Z",
        "trigger_state": "PAUSED",
        "failed_gates": ["measured_runtime_memory_available"],
        "faults": [],
    }
    green = {
        "created_utc": "2026-08-02T00:00:20Z",
        "trigger_state": "GREEN",
        "failed_gates": [],
        "faults": [],
    }
    checks = iter((paused, green))
    monkeypatch.setattr(launcher, "preflight", lambda *args, **kwargs: next(checks))
    monkeypatch.setattr(launcher.p2a, "write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        launcher,
        "execute_once",
        lambda *args, **kwargs: {
            "trigger_state": "GREEN",
            "campaign_complete": True,
        },
    )

    result = launcher.wait_and_execute(
        config(), config_path=CONFIG_PATH, poll_seconds=1.0
    )

    assert result["campaign_complete"] is True
