from __future__ import annotations

import json
import copy
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
        "disk_free_bytes": 20 * 1024**3,
        "disk_required_bytes": 2 * 1024**3,
        "checkpoint_transaction_requirement_available": True,
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
        ("disk_free_bytes", 1, "disk_reserve"),
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


def test_disk_reserve_is_derived_from_live_checkpoint_transaction() -> None:
    config = policy()
    requirement = campaign.checkpoint_transaction_requirement(config)
    assert requirement["available"] is True
    assert requirement["complete_transactions_required"] == 2
    assert requirement["required_bytes"] == (
        2 * requirement["transaction_bytes"]
    )
    assert requirement["transaction_bytes"] == sum(
        item["bytes"] for item in requirement["files"]
    )
    assert "minimum_disk_free_gib" not in config


def test_availability_policy_never_suspends_inflight_graph() -> None:
    config = policy()
    behavior = config["segment_behavior"]
    assert behavior == {
        "never_suspend_in_flight_metal_graph": True,
        "reevaluate_after_every_transactional_segment": True,
        "stop_launching_when_gate_closes": True,
        "atomic_checkpoint_before_yield": True,
    }


def test_availability_policy_requires_append_only_lineage_custody() -> None:
    config = policy()
    campaign.validate_availability_policy(config)
    assert config["lineage_custody"]["policy"] == campaign.LINEAGE_POLICY
    for field in (
        "archive_before_and_after_receipts",
        "archive_child_and_host_guard_receipts",
        "require_contiguous_identity_before_launch",
        "manifest_written_last",
    ):
        broken = copy.deepcopy(config)
        broken["lineage_custody"][field] = False
        try:
            campaign.validate_availability_policy(broken)
        except ValueError as exc:
            assert "append-only segment lineage custody" in str(exc)
        else:
            raise AssertionError(f"lineage custody accepted {field}=false")


def test_current_t1_receipt_matches_prospective_lineage_anchor() -> None:
    config = policy()
    _training, _plan, _target, receipt = campaign.campaign_state(
        ROOT / "configs/moecot_language_arm_training.json"
    )
    state = campaign.lineage_state(config, receipt)
    assert state["trigger_state"] == "GREEN"
    assert state["manifest_count"] == 0
    assert state["head_identity"]["optimizer_steps"] == 9048
    assert state["pre_anchor_full_chain_available"] is False


def test_lineage_rejects_live_identity_drift_without_manifest() -> None:
    config = policy()
    _training, _plan, _target, receipt = campaign.campaign_state(
        ROOT / "configs/moecot_language_arm_training.json"
    )
    changed = copy.deepcopy(receipt)
    changed["optimizer_steps"] = int(changed["optimizer_steps"]) + 1
    try:
        campaign.lineage_state(config, changed)
    except ValueError as exc:
        assert "does not match the append-only lineage head" in str(exc)
    else:
        raise AssertionError("unledgered T1 identity drift was accepted")


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


def test_process_inventory_excludes_the_current_launcher_ancestry(
    monkeypatch,
) -> None:
    class Result:
        returncode = 0
        stdout = (
            "10 1 zsh python3 scripts/selected_route_sustained_qualification.py --execute\n"
            "20 10 python3 scripts/selected_route_sustained_qualification.py --execute\n"
            "30 1 python3 scripts/optimizer_update_efficiency_qualification.py --execute\n"
        )
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: Result())
    monkeypatch.setattr(campaign.os, "getpid", lambda: 20)
    jobs = campaign.active_accelerator_jobs(
        [
            "selected_route_sustained_qualification.py --execute",
            "optimizer_update_efficiency_qualification.py --execute",
        ]
    )
    assert [row["pid"] for row in jobs] == [30]
