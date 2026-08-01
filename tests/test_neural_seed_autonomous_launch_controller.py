from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import neural_seed_autonomous_launch_controller as controller  # noqa: E402


CONFIG_PATH = ROOT / "configs/neural_seed_autonomous_launch_controller.json"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def green_package() -> dict:
    return {
        "trigger_state": "GREEN",
        "failed_gates": [],
        "package_identity": "sha256:test",
        "source_binding": {
            "commit": "a" * 40,
            "clean_at_generation": True,
        },
        "functional_surface": {"consumed_case_count": 0},
    }


def green_availability() -> dict:
    return {
        "trigger_state": "PAUSED",
        "snapshot": {
            "on_ac_power": True,
            "low_power_mode": False,
            "disk_free_bytes": 10_000,
            "disk_required_bytes": 1_000,
            "checkpoint_transaction_requirement_available": True,
            "active_accelerator_jobs": [],
            "yield_requested": True,
        },
    }


def test_config_replaces_user_gate_with_one_machine_predicate_segment() -> None:
    value = config()
    controller.validate_config(value)
    assert value["authority"]["user_or_operator_approval_required"] is False
    assert value["authority"]["removes_hold_permanently"] is False
    assert value["one_shot_command"]["max_segments"] == 1
    assert value["one_shot_command"]["expected_maximum_optimizer_steps"] == 64
    assert value["authority"]["D2_evaluation_authorized"] is False


def test_machine_preflight_authorizes_without_user_when_every_predicate_passes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(controller.replacement_freeze, "verify_package_identity", lambda _: True)
    report = controller.preflight(
        config(),
        config_path=CONFIG_PATH,
        source_state_override={
            "commit": "a" * 40,
            "branch": "main",
            "clean_at_generation": True,
            "dirty_path_count": 0,
            "dirty_paths": [],
        },
        process_jobs_override=[],
        package_override=green_package(),
        independent_override={"trigger_state": "GREEN", "failed_audits": []},
        scale_override={
            "trigger_state": "GREEN",
            "contract_state": "GREEN",
            "proposal_state": "AUTHORIZED_FOR_FROZEN_TRAINING_PLAN",
            "training_authorized": True,
            "boundaries": {"D2_cases_consumed": 0},
        },
        availability_override=green_availability(),
        review_override={"trigger_state": "READY"},
    )
    assert report["trigger_state"] == "GREEN"
    assert report["launch_authorized"] is True
    assert report["failed_gates"] == []


def test_competing_accelerator_or_dirty_source_pauses_automatically(monkeypatch) -> None:
    monkeypatch.setattr(controller.replacement_freeze, "verify_package_identity", lambda _: True)
    report = controller.preflight(
        config(),
        config_path=CONFIG_PATH,
        source_state_override={
            "commit": "a" * 40,
            "branch": "main",
            "clean_at_generation": False,
            "dirty_path_count": 1,
            "dirty_paths": [" M source.py"],
        },
        process_jobs_override=[{"pid": 7, "command": "theseus_p4s_campaign.py"}],
        package_override=green_package(),
        independent_override={"trigger_state": "GREEN", "failed_audits": []},
        scale_override={
            "trigger_state": "GREEN",
            "contract_state": "GREEN",
            "proposal_state": "AUTHORIZED_FOR_FROZEN_TRAINING_PLAN",
            "training_authorized": True,
            "boundaries": {"D2_cases_consumed": 0},
        },
        availability_override=green_availability(),
        review_override={"trigger_state": "READY"},
    )
    assert report["trigger_state"] == "PAUSED"
    assert report["launch_authorized"] is False
    assert "source_clean_and_exactly_bound" in report["failed_gates"]
    assert "no_competing_accelerator_job" in report["failed_gates"]


def test_transactional_hold_lease_restores_hold_after_failure(tmp_path: Path) -> None:
    hold = tmp_path / "hold"
    hold.write_text("yield\n", encoding="utf-8")
    try:
        with controller.transactional_hold_lease(hold, "lease"):
            assert not hold.exists()
            raise RuntimeError("simulated child failure")
    except RuntimeError as exc:
        assert str(exc) == "simulated child failure"
    else:
        raise AssertionError("simulated failure did not escape")
    assert hold.read_text(encoding="utf-8") == "yield\n"
    assert not (tmp_path / "hold.leased-lease").exists()


def test_checkpoint_transaction_restore_is_exact(tmp_path: Path) -> None:
    source = tmp_path / "weights.bin"
    backup = tmp_path / "backup.bin"
    source.write_bytes(b"before")
    backup.write_bytes(b"before")
    expected = controller.sha256_file(source)
    source.write_bytes(b"after")
    state, faults = controller.restore_checkpoint_transaction({
        "files": [{
            "source": str(source),
            "backup": str(backup),
            "sha256": expected,
        }]
    })
    assert state == "RESTORED_EXACT_PRESEGMENT_TRANSACTION"
    assert faults == []
    assert source.read_bytes() == b"before"


def test_exclusive_lease_cannot_overwrite_live_controller(tmp_path: Path) -> None:
    lease = tmp_path / "lease.json"
    controller.write_json_exclusive(lease, {"lease_id": "first"})
    with pytest.raises(FileExistsError):
        controller.write_json_exclusive(lease, {"lease_id": "second"})
    assert json.loads(lease.read_text(encoding="utf-8")) == {"lease_id": "first"}


def test_snapshot_failure_archives_lease_without_stranding_hold(
    tmp_path: Path, monkeypatch,
) -> None:
    value = config()
    hold = tmp_path / "hold"
    hold.write_text("yield\n", encoding="utf-8")
    active_lease = tmp_path / "active-lease.json"
    archive = tmp_path / "archive"
    rollback = tmp_path / "rollback"
    child = tmp_path / "child.json"
    training_config = tmp_path / "training.json"
    training_config.write_text(
        json.dumps({"host_resource_safety": {"qualified_python": sys.executable}}),
        encoding="utf-8",
    )
    value.update({
        "yield_control": str(hold),
        "active_lease": str(active_lease),
        "lease_archive_directory": str(archive),
        "rollback_staging_root": str(rollback),
        "child_report": str(child),
        "training_config": str(training_config),
    })
    monkeypatch.setattr(
        controller,
        "preflight",
        lambda *_args, **_kwargs: {
            "trigger_state": "GREEN",
            "launch_authorized": True,
            "source_binding": {"commit": "a" * 40},
        },
    )
    monkeypatch.setattr(controller, "lineage_manifest_paths", lambda _config: set())
    monkeypatch.setattr(
        controller,
        "snapshot_checkpoint_transaction",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("snapshot failed")),
    )
    report = controller.execute_one_shot(value, config_path=CONFIG_PATH)
    assert report["trigger_state"] == "RED"
    assert report["checkpoint_rollback_state"] == (
        "SNAPSHOT_NOT_CREATED_NO_TRAINING_STARTED"
    )
    assert hold.read_text(encoding="utf-8") == "yield\n"
    assert not active_lease.exists()
    leases = list(archive.glob("*.json"))
    assert len(leases) == 1
    archived = json.loads(leases[0].read_text(encoding="utf-8"))
    assert archived["state"] == "FAILED_BEFORE_SEGMENT"
    assert archived["error"] == "RuntimeError: snapshot failed"
