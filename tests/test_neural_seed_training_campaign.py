from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import neural_seed_training_campaign as campaign


def policy() -> dict:
    return json.loads(
        (
            ROOT / "configs" / "neural_seed_training_availability.json"
        ).read_text(encoding="utf-8")
    )


def green_snapshot() -> dict:
    return {
        "on_ac_power": True,
        "low_power_mode": False,
        "disk_free_gib": 20.0,
        "reclaimable_available_mib": 8000.0,
        "active_accelerator_jobs": [],
        "yield_requested": False,
    }


def test_availability_policy_forbids_clock_based_launch_windows() -> None:
    config = policy()
    assert "launch_windows" not in config
    config["launch_windows"] = [
        {"start_local": "21:00", "end_local": "08:00"}
    ]
    try:
        campaign.validate_availability_policy(config)
    except ValueError as exc:
        assert "clock-based launch windows are forbidden" in str(exc)
    else:
        raise AssertionError("clock-based launch restriction was accepted")


def test_availability_gate_requires_every_laptop_safety_condition() -> None:
    config = policy()
    campaign.validate_availability_policy(config)
    report = campaign.evaluate_availability(config, green_snapshot())
    assert report["trigger_state"] == "GREEN"
    assert all(report["gates"].values())

    for key, value, failed_gate in (
        ("on_ac_power", False, "ac_power"),
        ("low_power_mode", True, "low_power_mode_off"),
        ("disk_free_gib", 1.0, "disk_reserve"),
        (
            "active_accelerator_jobs",
            [{"pid": 7}],
            "no_interactive_accelerator_job",
        ),
        ("yield_requested", True, "yield_control_absent"),
    ):
        snapshot = green_snapshot()
        snapshot[key] = value
        report = campaign.evaluate_availability(config, snapshot)
        assert report["trigger_state"] == "PAUSED"
        assert failed_gate in report["failed_gates"]


def test_availability_policy_never_suspends_inflight_graph() -> None:
    config = policy()
    behavior = config["segment_behavior"]
    assert behavior == {
        "never_suspend_in_flight_metal_graph": True,
        "reevaluate_after_every_transactional_segment": True,
        "stop_launching_when_gate_closes": True,
        "atomic_checkpoint_before_yield": True,
    }


def test_process_inventory_failure_pauses_instead_of_authorizing(
    monkeypatch,
) -> None:
    def deny(*_args, **_kwargs):
        raise PermissionError("inventory denied")

    monkeypatch.setattr(subprocess, "run", deny)
    jobs = campaign.active_accelerator_jobs(["mlx"])
    assert jobs == [
        {
            "telemetry_fault": "process_inventory_unavailable",
            "error_type": "PermissionError",
        }
    ]
    snapshot = green_snapshot()
    snapshot["active_accelerator_jobs"] = jobs
    report = campaign.evaluate_availability(policy(), snapshot)
    assert report["trigger_state"] == "PAUSED"
    assert "no_interactive_accelerator_job" in report["failed_gates"]
