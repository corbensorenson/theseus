from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import neural_seed_autonomous_launch_controller as controller  # noqa: E402
import neural_seed_training_campaign as campaign  # noqa: E402


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
            "yield_requested": False,
        },
    }


def authorized_availability_policy() -> dict:
    value = json.loads(
        (ROOT / "configs/neural_seed_training_availability.json").read_text(
            encoding="utf-8"
        )
    )
    value["program_authority"].update(
        {
            "state": campaign.PROGRAM_AUTHORIZED_STATE,
            "launch_allowed": True,
        }
    )
    return value


def test_config_replaces_user_gate_with_one_machine_predicate_segment() -> None:
    value = config()
    controller.validate_config(value)
    assert value["authority"]["user_or_operator_approval_required"] is False
    assert value["authority"]["removes_hold_permanently"] is False
    assert value["authority"]["never_remove_or_modify_yield_control"] is True
    assert value["one_shot_command"]["max_segments"] == 1
    assert value["one_shot_command"]["expected_maximum_optimizer_steps"] == 64
    assert value["authority"]["D2_evaluation_authorized"] is False
    patterns = value["exclusive_accelerator_process_patterns"]
    assert "theseus_p4v2r2_campaign.py" in patterns
    assert "theseus_p4v2r2_autonomous_launch.py --execute" in patterns


def test_machine_preflight_authorizes_without_user_when_every_predicate_passes(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(controller.replacement_freeze, "verify_package_identity", lambda _: True)
    value = config()
    value["yield_control"] = str(tmp_path / "no-yield-request")
    report = controller.preflight(
        value,
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
        availability_policy_override=authorized_availability_policy(),
        review_override={"trigger_state": "READY"},
        source_binding_override=True,
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
        availability_policy_override=authorized_availability_policy(),
        review_override={"trigger_state": "READY"},
        source_binding_override=False,
    )
    assert report["trigger_state"] == "PAUSED"
    assert report["launch_authorized"] is False
    assert "source_clean_and_exactly_bound" in report["failed_gates"]
    assert "no_competing_accelerator_job" in report["failed_gates"]


def test_emergency_yield_is_a_stop_request_not_a_launch_prerequisite(
    monkeypatch, tmp_path: Path
) -> None:
    value = config()
    yield_control = tmp_path / "yield"
    value["yield_control"] = str(yield_control)
    monkeypatch.setattr(controller.replacement_freeze, "verify_package_identity", lambda _: True)
    common = {
        "config_path": CONFIG_PATH,
        "source_state_override": {
            "commit": "a" * 40,
            "branch": "main",
            "clean_at_generation": True,
            "dirty_path_count": 0,
            "dirty_paths": [],
        },
        "process_jobs_override": [],
        "package_override": green_package(),
        "independent_override": {"trigger_state": "GREEN", "failed_audits": []},
        "scale_override": {
            "trigger_state": "GREEN",
            "contract_state": "GREEN",
            "proposal_state": "AUTHORIZED_FOR_FROZEN_TRAINING_PLAN",
            "training_authorized": True,
            "boundaries": {"D2_cases_consumed": 0},
        },
        "review_override": {"trigger_state": "READY"},
        "source_binding_override": True,
        "availability_policy_override": authorized_availability_policy(),
    }
    absent = controller.preflight(
        value, availability_override=green_availability(), **common
    )
    assert absent["gates"]["emergency_yield_absent"] is True
    assert absent["trigger_state"] == "GREEN"

    yield_control.write_text("yield\n", encoding="utf-8")
    requested_availability = green_availability()
    requested_availability["snapshot"]["yield_requested"] = True
    requested = controller.preflight(
        value, availability_override=requested_availability, **common
    )
    assert requested["gates"]["emergency_yield_absent"] is False
    assert requested["trigger_state"] == "PAUSED"


def test_default_program_hold_blocks_one_shot_even_when_other_gates_pass(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        controller.replacement_freeze, "verify_package_identity", lambda _: True
    )
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
        source_binding_override=True,
    )
    assert report["trigger_state"] == "PAUSED"
    assert report["launch_authorized"] is False
    assert report["gates"]["prospective_resource_gate_green"] is False
    assert "program_authority_allows_training" in report[
        "prospective_availability_under_exclusive_lease"
    ]["failed_gates"]


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


def test_evidence_only_descendant_is_bound_but_source_drift_is_not(monkeypatch) -> None:
    value = config()
    bound_path = ROOT / "configs/neural_seed_training_availability.json"
    package = {
        "source_binding": {
            "commit": "a" * 40,
            "clean_at_generation": True,
        },
        "source_artifacts": {
            "configs/neural_seed_training_availability.json": {
                "path": "configs/neural_seed_training_availability.json",
                "sha256": controller.sha256_file(bound_path),
            }
        },
    }
    source = {
        "commit": "b" * 40,
        "clean_at_generation": True,
    }

    def evidence_only(command, **_kwargs):
        if command[1:3] == ["merge-base", "--is-ancestor"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(
            command, 0, "reports/pre_long_run_replacement_freeze.json\n", ""
        )

    monkeypatch.setattr(controller.subprocess, "run", evidence_only)
    assert controller.source_is_bound_to_package(
        value, package=package, source=source
    ) is True

    def source_drift(command, **_kwargs):
        if command[1:3] == ["merge-base", "--is-ancestor"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(
            command, 0, "scripts/moecot_language_arm_training.py\n", ""
        )

    monkeypatch.setattr(controller.subprocess, "run", source_drift)
    assert controller.source_is_bound_to_package(
        value, package=package, source=source
    ) is False


def test_snapshot_failure_archives_lease_without_modifying_yield_control(
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
