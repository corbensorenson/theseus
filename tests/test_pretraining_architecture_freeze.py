from __future__ import annotations

import copy
from dataclasses import asdict
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pretraining_architecture_freeze as freeze


def test_replacement_freeze_accepts_bound_first_campaign_dispositions() -> None:
    config = freeze.load_config()
    manifest = freeze.artifact_manifest(config)
    dispositions = freeze.architecture_dispositions(config)

    assert dispositions["required_count"] == dispositions["ready_count"]
    kerc = dispositions["rows"][
        "planned.kernel_english_hierarchical_residual_compiler_v1"
    ]
    assert kerc["status"] == "retired_by_pretraining_verdict"
    assert kerc["negative_disposition"]["kind"] == "campaign_scope_only"
    assert kerc["negative_disposition"]["scientific_falsification_claimed"] is False
    assert len(manifest) >= 100
    assert "scripts/standard_causal_transformer_model.py" in manifest
    assert "scripts/pretraining_factorized_bakeoff.py" in manifest
    assert "reports/rdc_kerc_resource_disposition.json" in manifest
    assert "scripts/host_resource_safety.py" in manifest
    assert "configs/onecell_rwm_pretraining_disposition.json" in manifest


def test_freeze_binds_generated_effect_and_governance_receipts() -> None:
    receipts = freeze.receipt_manifest(freeze.load_config())

    assert set(receipts) == {
        "reports/generation_mode_registry.json",
        "reports/generation_mode_registry.md",
        "reports/governance_rights_receipt_suite.json",
        "reports/neural_seed_50m_scale_preregistration.json",
        "reports/onecell_rwm_pretraining_disposition.json",
        "reports/policy_optimization_program.json",
        "reports/policy_optimization_program.md",
        "reports/theseus_assistant_effect_complete_canary.json",
    }
    assert all(len(row["sha256"]) == 64 and row["bytes"] > 0 for row in receipts.values())


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


def test_replacement_freeze_requires_factorized_selection() -> None:
    selection = freeze.factorized_selection(freeze.load_config())

    assert selection["campaign_id"] == "moecot_mlx_57m_active_preregistered_v1"
    assert selection["disposition"] == (
        "factorized_architecture_selected_training_not_started"
    )
    assert len(selection["selected_implementation_ids"]) == 7


def test_replacement_freeze_binds_green_selected_route_execution() -> None:
    qualification = freeze.selected_route_execution_qualification(
        freeze.load_config()
    )

    assert qualification["training_plan_sha256"] == (
        "1c7c859ecdf2112dbd9938a34631aab70545031649c4d970554395294b1c098f"
    )
    assert qualification["canonical_receipt"]["optimizer_steps"] == 3480
    assert qualification["canonical_receipt"][
        "resume_plan_identity_migration"
    ]["migration_id"] == (
        "shared_trunk_step3480_replay_swap_policy_alignment_v1"
    )
    assert (
        qualification["sustained_report"]["successful_segment_count"] == 6
    )
    assert qualification["sustained_report"]["thermal_stability"]["terminal"]
    assert qualification["fresh_process_report"]["contiguous_segment_count"] == 2
    assert qualification["fresh_process_report"]["numeric_replay_parity"]
    assert qualification["fresh_process_report"]["zero_swap_growth"] is False
    assert (
        qualification["fresh_process_report"]["swap_growth_treatment"]
        == "DIAGNOSTIC_ONLY"
    )


def test_cpu_replay_contract_is_externally_guarded_without_accelerator_authority() -> None:
    config = freeze.load_config()
    policy = freeze.replay_safety_policy(config)

    assert config["replay_safety"]["accelerator_authorization_allowed"] is False
    assert policy.max_process_memory_mib == 1024
    assert policy.minimum_available_before_launch_mib == 0
    assert policy.minimum_available_during_run_mib == 0
    assert policy.maximum_wall_seconds == 0
    assert policy.memory_guard_mode == "predicted_exhaustion"
    assert policy.swapout_growth_action == "report_only"
    assert policy.qualified_peak_inferred_unified_memory_mib == pytest.approx(
        996.281
    )
    assert policy.maximum_swapout_growth_mib == 16


def test_cpu_replay_preflight_refusal_writes_receipt_without_starting_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = copy.deepcopy(freeze.load_config())
    config["replay_commands"] = [[sys.executable, "-c", "print('never')"]]
    config["replay_safety"]["receipt_directory"] = "reports/test-replay-safety"
    original_resolve = freeze.resolve

    def isolated_resolve(value: str | Path) -> Path:
        if str(value) == "reports/test-replay-safety":
            return tmp_path
        return original_resolve(value)

    monkeypatch.setattr(freeze, "resolve", isolated_resolve)
    monkeypatch.setattr(
        freeze.host_resource_safety, "physical_memory_mib", lambda: 16384.0
    )
    monkeypatch.setattr(
        freeze.host_resource_safety,
        "run_guarded",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            freeze.host_resource_safety.HostResourceSafetyFault(
                "host_memory_preflight_failed"
            )
        ),
    )

    with pytest.raises(freeze.ArchitectureFreezeFault, match="replay_guard_failed"):
        freeze.run_replays(config)

    receipt = freeze.json.loads((tmp_path / "00.json").read_text())
    assert receipt["passed"] is False
    assert receipt["safety_receipt"]["child_started"] is False
    assert receipt["safety_receipt"]["fault"] == "host_memory_preflight_failed"


def test_cpu_replay_success_uses_replay_only_authority_and_content_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = copy.deepcopy(freeze.load_config())
    config["replay_commands"] = [[sys.executable, "-c", "print('bounded')"]]
    config["replay_safety"]["receipt_directory"] = "reports/test-replay-safety"
    original_resolve = freeze.resolve
    observed: dict[str, object] = {}

    def isolated_resolve(value: str | Path) -> Path:
        if str(value) == "reports/test-replay-safety":
            return tmp_path
        return original_resolve(value)

    def passed(command: list[str], **kwargs: object):
        observed["env"] = kwargs["env"]
        policy = kwargs["policy"]
        return freeze.host_resource_safety.GuardedProcessResult(
            returncode=0,
            stdout="bounded\n",
            stderr="",
            receipt={
                "policy": freeze.host_resource_safety.POLICY,
                "command": command,
                "passed": True,
                "child_started": True,
                "terminated_by_guard": False,
                "fault": "",
                "returncode": 0,
                "limits": asdict(policy),
            },
        )

    monkeypatch.setattr(freeze, "resolve", isolated_resolve)
    monkeypatch.setattr(
        freeze.host_resource_safety, "physical_memory_mib", lambda: 16384.0
    )
    monkeypatch.setattr(freeze.host_resource_safety, "run_guarded", passed)

    receipts = freeze.run_replays(config)

    assert receipts[0]["passed"] is True
    assert receipts[0]["receipt_path"] == "reports/test-replay-safety/00.json"
    assert len(receipts[0]["receipt_sha256"]) == 64
    assert observed["env"] == {"THESEUS_GUARDED_REPLAY_CHILD": "1"}
    assert "THESEUS_GUARDED_ACCELERATOR_CHILD" not in observed["env"]


def test_replacement_freeze_waits_for_guarded_accelerator_replay() -> None:
    config = copy.deepcopy(freeze.load_config())
    config["accelerator_replay"]["shards"][0]["receipt"] = (
        "reports/accelerator_replay/intentionally_missing_test_receipt.json"
    )
    with pytest.raises(
        freeze.ArchitectureFreezeFault,
        match="accelerator_replay_receipt_(missing|invalid)",
    ):
        freeze.accelerator_replay_receipts(config)


def test_every_accelerator_shard_has_a_tighter_source_justified_envelope() -> None:
    contract = freeze.load_config()["accelerator_replay"]
    for shard in contract["shards"]:
        freeze.validate_accelerator_shard_contract(contract, shard)
        assert shard["risk_class"]
        assert shard["resource_basis"]
        if shard["id"] == "scale_preregistration_resource_canaries":
            assert shard["maximum_process_memory_mib"] == 4096
            assert shard["generated_artifacts"] == [
                "reports/neural_seed_50m_scale_preregistration.json"
            ]
        else:
            assert shard["maximum_process_memory_mib"] <= 2048
        assert "minimum_available_before_launch_mib" in shard
        if shard["id"] == "optimizer_matched_adequacy":
            assert shard["minimum_available_memory_mib"] == 0
            assert shard["minimum_available_before_launch_mib"] == 0
        else:
            assert shard["minimum_available_memory_mib"] >= contract[
                "minimum_available_memory_mib"
            ]
            assert shard["minimum_available_before_launch_mib"] >= shard[
                "minimum_available_memory_mib"
            ]
        assert shard["maximum_swapout_growth_mib"] <= 16
        assert shard["poll_interval_seconds"] <= 0.1
        policy = freeze.accelerator_shard_policy(contract, shard)
        if shard["id"] == "optimizer_state_and_progress":
            assert policy.minimum_available_before_launch_mib == 5120
            assert (
                policy.minimum_available_before_launch_mib
                == policy.minimum_available_during_run_mib
                + policy.max_process_memory_mib
            )
        if shard["id"] == "optimizer_matched_adequacy":
            calibration = shard["measured_launch_calibration"]
            assert policy.minimum_available_before_launch_mib == 0
            assert policy.minimum_available_during_run_mib == 0
            assert policy.maximum_wall_seconds == 0
            assert policy.memory_guard_mode == "predicted_exhaustion"
            assert policy.swapout_growth_action == "report_only"
            assert (
                policy.qualified_peak_inferred_unified_memory_mib
                == pytest.approx(1529.766)
            )
            assert calibration["qualification_authority"] is False
            assert calibration["source_receipt"] == shard["receipt"]
            assert calibration["required_launch_floor_mib"] == (
                calibration["minimum_live_reserve_mib"]
                + calibration["observed_maximum_inferred_unified_memory_mib"]
                + calibration["minimum_safety_margin_mib"]
            )
            assert calibration["observed_maximum_inferred_unified_memory_mib"] == (
                policy.qualified_peak_inferred_unified_memory_mib
            )


def test_operation_specific_launch_reserve_cannot_fall_below_live_reserve() -> None:
    config = copy.deepcopy(freeze.load_config())
    contract = config["accelerator_replay"]
    shard = next(
        row for row in contract["shards"] if row["id"] == "optimizer_state_and_progress"
    )
    shard["minimum_available_before_launch_mib"] = (
        float(shard["minimum_available_memory_mib"]) - 1.0
    )
    with pytest.raises(
        freeze.ArchitectureFreezeFault,
        match="accelerator_shard_launch_reserve_below_live_reserve",
    ):
        freeze.validate_accelerator_shard_contract(contract, shard)


def test_measured_optimizer_launch_calibration_cannot_be_weakened() -> None:
    config = copy.deepcopy(freeze.load_config())
    contract = config["accelerator_replay"]
    shard = next(
        row for row in contract["shards"] if row["id"] == "optimizer_matched_adequacy"
    )
    shard["qualified_peak_inferred_unified_memory_mib"] = 1529
    with pytest.raises(
        freeze.ArchitectureFreezeFault,
        match="accelerator_shard_launch_calibration_weakened",
    ):
        freeze.validate_accelerator_shard_contract(contract, shard)

    shard["qualified_peak_inferred_unified_memory_mib"] = 1529.766
    shard["measured_launch_calibration"]["qualification_authority"] = True
    with pytest.raises(
        freeze.ArchitectureFreezeFault,
        match="accelerator_shard_launch_calibration_authority_invalid",
    ):
        freeze.validate_accelerator_shard_contract(contract, shard)


def test_aggregate_gates_replace_duplicate_native_canary_shards() -> None:
    accelerator = freeze.load_config()["accelerator_replay"]
    shard_ids = {str(shard["id"]) for shard in accelerator["shards"]}
    guarded = set(accelerator["guarded_test_nodeids"])

    assert len(accelerator["shards"]) == 14
    assert {
        "generation_mode_gate",
        "optimizer_matched_adequacy",
        "policy_optimization_gate",
    } <= shard_ids
    assert not {
        "query_chunk_compact_parity",
        "kerc_decomposed_objective_parity",
        "kerc_online_kv_representative_preflight",
        "kerc_structured_drafting",
        "generation_mtp_mechanics",
        "generation_mtp_adequacy",
        "policy_objective_mlx_parity",
    } & shard_ids
    assert {
        "tests/test_generation_architecture_contracts.py::GenerationArchitectureContractTests::test_adequate_mtp_candidates_train_heads_and_reload_exactly",
        "tests/test_generation_architecture_contracts.py::GenerationArchitectureContractTests::test_mtp_executes_on_mlx_with_shape_safe_future_offsets",
        "tests/test_kerc_structured_drafting.py::KercStructuredDraftingTests::test_mlx_heads_learn_reload_and_use_hidden_state",
        "tests/test_policy_objective_contracts.py::PolicyObjectiveContractTests::test_reference_suite_has_mlx_parity_and_zero_training_exposure",
    } <= guarded


def _runner_config() -> dict[str, object]:
    return {
        "accelerator_replay": {
            "watchdog_policy": freeze.host_resource_safety.POLICY,
            "qualified_python": sys.executable,
            "maximum_process_memory_mib": 6144,
            "minimum_available_before_launch_mib": 6144,
            "minimum_available_memory_mib": 4096,
            "maximum_swapout_growth_mib": 64,
            "poll_interval_seconds": 0.25,
            "terminate_grace_seconds": 2.0,
            "shards": [
                {
                    "id": "first",
                    "max_wall_seconds": 30,
                    "risk_class": "tiny_fixture",
                    "resource_basis": "unit_test_fixture",
                    "maximum_process_memory_mib": 1024,
                    "minimum_available_memory_mib": 4096,
                    "maximum_swapout_growth_mib": 16,
                    "poll_interval_seconds": 0.1,
                    "command": [sys.executable, "-c", "print('first')"],
                    "receipt": "reports/accelerator_replay/first.json",
                },
                {
                    "id": "second",
                    "max_wall_seconds": 45,
                    "risk_class": "tiny_fixture",
                    "resource_basis": "unit_test_fixture",
                    "maximum_process_memory_mib": 1024,
                    "minimum_available_memory_mib": 4096,
                    "maximum_swapout_growth_mib": 16,
                    "poll_interval_seconds": 0.1,
                    "command": [sys.executable, "-c", "print('second')"],
                    "receipt": "reports/accelerator_replay/second.json",
                },
            ],
        }
    }


def test_guarded_runner_writes_exact_receipts_and_completes_selected_shard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        freeze.host_resource_safety, "physical_memory_mib", lambda: 16384.0
    )
    monkeypatch.setattr(
        freeze,
        "resolve",
        lambda value: tmp_path / Path(str(value)).name,
    )

    def passed(command: list[str], **kwargs: object):
        policy = kwargs["policy"]
        return freeze.host_resource_safety.GuardedProcessResult(
            returncode=0,
            stdout="captured diagnostic output\n",
            stderr="captured diagnostic error\n",
            receipt={
                "policy": freeze.host_resource_safety.POLICY,
                "command": command,
                "passed": True,
                "child_started": True,
                "terminated_by_guard": False,
                "fault": "",
                "returncode": 0,
                "limits": asdict(policy),
            },
        )

    monkeypatch.setattr(freeze.host_resource_safety, "run_guarded", passed)
    report = freeze.run_accelerator_shards(
        _runner_config(), selected_ids={"second"}
    )

    assert report["complete"] is True
    assert report["attempted_shard_count"] == 1
    assert report["reused_shard_count"] == 0
    receipt = freeze.json.loads((tmp_path / "second.json").read_text())
    assert receipt["passed"] is True
    assert receipt["command"][-1] == "print('second')"
    assert receipt["limits"]["maximum_wall_seconds"] == 45.0
    assert receipt["stdout_tail"] == "captured diagnostic output\n"
    assert receipt["stderr_tail"] == "captured diagnostic error\n"


def test_selected_shard_refuses_before_launch_when_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _runner_config()
    config["accelerator_replay"]["shards"][1]["depends_on_shards"] = ["first"]
    monkeypatch.setattr(
        freeze.host_resource_safety, "physical_memory_mib", lambda: 16384.0
    )
    monkeypatch.setattr(
        freeze,
        "resolve",
        lambda value: tmp_path / Path(str(value)).name,
    )
    launched = False

    def unexpected_launch(*_args: object, **_kwargs: object):
        nonlocal launched
        launched = True
        raise AssertionError("dependency failure launched a child")

    monkeypatch.setattr(
        freeze.host_resource_safety, "run_guarded", unexpected_launch
    )
    report = freeze.run_accelerator_shards(config, selected_ids={"second"})

    assert report["complete"] is False
    assert report["attempted_shard_count"] == 0
    assert launched is False
    receipt = freeze.json.loads((tmp_path / "second.json").read_text())
    assert receipt["child_started"] is False
    assert receipt["fault"] == (
        "accelerator_shard_dependency_not_ready:second:first"
    )


def test_selected_shard_binds_valid_dependency_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _runner_config()
    contract = config["accelerator_replay"]
    first, second = contract["shards"]
    second["depends_on_shards"] = ["first"]
    monkeypatch.setattr(
        freeze.host_resource_safety, "physical_memory_mib", lambda: 16384.0
    )
    monkeypatch.setattr(
        freeze,
        "resolve",
        lambda value: tmp_path / Path(str(value)).name,
    )
    first_path = tmp_path / "first.json"
    freeze.atomic_write_json(
        first_path,
        {
            "policy": contract["watchdog_policy"],
            "command": first["command"],
            "passed": True,
            "child_started": True,
            "terminated_by_guard": False,
                "fault": "",
                "returncode": 0,
                "maximum_inferred_unified_memory_mib": 128.0,
                "limits": asdict(freeze.accelerator_shard_policy(contract, first)),
        },
    )

    def passed(command: list[str], **kwargs: object):
        return freeze.host_resource_safety.GuardedProcessResult(
            returncode=0,
            stdout="",
            stderr="",
            receipt={
                "policy": contract["watchdog_policy"],
                "command": command,
                "passed": True,
                "child_started": True,
                "terminated_by_guard": False,
                    "fault": "",
                    "returncode": 0,
                    "maximum_inferred_unified_memory_mib": 128.0,
                    "limits": asdict(kwargs["policy"]),
            },
        )

    monkeypatch.setattr(freeze.host_resource_safety, "run_guarded", passed)
    report = freeze.run_accelerator_shards(config, selected_ids={"second"})

    assert report["complete"] is True
    assert report["attempted_shard_count"] == 1
    receipt = freeze.json.loads((tmp_path / "second.json").read_text())
    assert receipt["dependency_receipts"] == {
        "first": {
            "receipt": "reports/accelerator_replay/first.json",
            "sha256": freeze.sha256(first_path),
        }
    }


def test_guarded_runner_records_preflight_refusal_and_stops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        freeze.host_resource_safety, "physical_memory_mib", lambda: 16384.0
    )
    monkeypatch.setattr(
        freeze,
        "resolve",
        lambda value: tmp_path / Path(str(value)).name,
    )
    attempts = 0

    def refused(*_args: object, **_kwargs: object):
        nonlocal attempts
        attempts += 1
        raise freeze.host_resource_safety.HostResourceSafetyFault(
            "host_memory_preflight_failed"
        )

    monkeypatch.setattr(freeze.host_resource_safety, "run_guarded", refused)
    report = freeze.run_accelerator_shards(_runner_config())

    assert report["complete"] is False
    assert report["attempted_shard_count"] == 1
    assert attempts == 1
    receipt = freeze.json.loads((tmp_path / "first.json").read_text())
    assert receipt["passed"] is False
    assert receipt["child_started"] is False
    assert receipt["fault"] == "host_memory_preflight_failed"
    assert not (tmp_path / "second.json").exists()


def test_shard_resource_contract_cannot_inherit_or_weaken_global_ceiling() -> None:
    config = _runner_config()
    contract = config["accelerator_replay"]
    shard = contract["shards"][0]

    missing = copy.deepcopy(shard)
    missing.pop("resource_basis")
    with pytest.raises(
        freeze.ArchitectureFreezeFault,
        match="accelerator_shard_resource_contract_incomplete",
    ):
        freeze.validate_accelerator_shard_contract(contract, missing)

    weaker = copy.deepcopy(shard)
    weaker["maximum_process_memory_mib"] = 7000
    with pytest.raises(
        freeze.ArchitectureFreezeFault,
        match="accelerator_shard_resource_contract_weaker_than_global",
    ):
        freeze.validate_accelerator_shard_contract(contract, weaker)

    slower_poll = copy.deepcopy(shard)
    slower_poll["poll_interval_seconds"] = 0.5
    with pytest.raises(
        freeze.ArchitectureFreezeFault,
        match="accelerator_shard_resource_contract_weaker_than_global",
    ):
        freeze.validate_accelerator_shard_contract(contract, slower_poll)

    forward_dependency = copy.deepcopy(shard)
    forward_dependency["depends_on_shards"] = ["second"]
    with pytest.raises(
        freeze.ArchitectureFreezeFault,
        match="accelerator_shard_dependency_contract_invalid",
    ):
        freeze.validate_accelerator_shard_contract(contract, forward_dependency)


def test_guarded_runner_reuses_valid_receipt_and_resumes_at_missing_shard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _runner_config()
    contract = config["accelerator_replay"]
    monkeypatch.setattr(
        freeze.host_resource_safety, "physical_memory_mib", lambda: 16384.0
    )
    monkeypatch.setattr(
        freeze,
        "resolve",
        lambda value: tmp_path / Path(str(value)).name,
    )
    first, second = contract["shards"]
    first_policy = freeze.accelerator_shard_policy(contract, first)
    freeze.atomic_write_json(
        tmp_path / "first.json",
        {
            "policy": contract["watchdog_policy"],
            "command": first["command"],
            "passed": True,
            "child_started": True,
            "terminated_by_guard": False,
                "fault": "",
                "returncode": 0,
                "maximum_inferred_unified_memory_mib": 128.0,
                "limits": asdict(first_policy),
        },
    )
    launched: list[list[str]] = []

    def passed(command: list[str], **kwargs: object):
        launched.append(command)
        policy = kwargs["policy"]
        return freeze.host_resource_safety.GuardedProcessResult(
            returncode=0,
            stdout="",
            stderr="",
            receipt={
                "policy": contract["watchdog_policy"],
                "command": command,
                "passed": True,
                "child_started": True,
                "terminated_by_guard": False,
                    "fault": "",
                    "returncode": 0,
                    "maximum_inferred_unified_memory_mib": 128.0,
                    "limits": asdict(policy),
                },
            )

    monkeypatch.setattr(freeze.host_resource_safety, "run_guarded", passed)
    report = freeze.run_accelerator_shards(config)

    assert report["complete"] is True
    assert report["reused_shard_count"] == 1
    assert report["attempted_shard_count"] == 1
    assert launched == [second["command"]]


def test_replay_readiness_is_derived_from_exact_receipts_and_rejects_tampering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = copy.deepcopy(freeze.load_config())
    original_resolve = freeze.resolve

    def isolated_resolve(value: str | Path) -> Path:
        if str(value).startswith("reports/"):
            return tmp_path / Path(str(value)).name
        return original_resolve(value)

    monkeypatch.setattr(freeze, "resolve", isolated_resolve)
    contract = config["accelerator_replay"]
    for shard in contract["shards"]:
        for artifact in shard.get("implementation_artifacts") or []:
            if str(artifact).startswith("reports/"):
                isolated_resolve(artifact).write_bytes(
                    original_resolve(artifact).read_bytes()
                )
        for artifact in shard.get("generated_artifacts") or []:
            artifact_path = isolated_resolve(artifact)
            artifact_path.write_text(
                '{"test_fixture":"generated_accelerator_artifact"}\n',
                encoding="utf-8",
            )
        policy = freeze.accelerator_shard_policy(contract, shard)
        freeze.atomic_write_json(
            isolated_resolve(shard["receipt"]),
            {
                "policy": contract["watchdog_policy"],
                "command": shard["command"],
                "passed": True,
                "child_started": True,
                "terminated_by_guard": False,
                    "fault": "",
                    "returncode": 0,
                    "maximum_inferred_unified_memory_mib": 128.0,
                    "limits": asdict(policy),
                    "generated_artifacts": freeze.shard_generated_artifact_manifest(
                        shard
                    ),
                    "implementation_artifacts": (
                        freeze.shard_implementation_artifact_manifest(shard)
                    ),
                    "dependency_receipts": freeze.accelerator_dependency_receipt_manifest(
                    contract, shard
                ),
            },
        )

    manifest = freeze.accelerator_replay_receipts(config)
    assert len(manifest) == len(contract["shards"])

    first = isolated_resolve(contract["shards"][0]["receipt"])
    tampered = freeze.json.loads(first.read_text())
    tampered["limits"]["minimum_available_before_launch_mib"] = 1
    freeze.atomic_write_json(first, tampered)
    with pytest.raises(
        freeze.ArchitectureFreezeFault, match="accelerator_replay_receipt_invalid"
    ):
        freeze.accelerator_replay_receipts(config)


def test_accelerator_receipt_binds_generated_gate_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _runner_config()
    contract = config["accelerator_replay"]
    shard = contract["shards"][0]
    shard["generated_artifacts"] = ["reports/gate.json"]
    output = tmp_path / "gate.json"
    output.write_text('{"trigger_state":"GREEN"}\n', encoding="utf-8")
    original_resolve = freeze.resolve

    def isolated_resolve(value: str | Path) -> Path:
        if str(value) == "reports/gate.json":
            return output
        return original_resolve(value)

    monkeypatch.setattr(freeze, "resolve", isolated_resolve)
    monkeypatch.setattr(
        freeze.host_resource_safety, "physical_memory_mib", lambda: 16384.0
    )
    receipt = {
        "policy": contract["watchdog_policy"],
        "command": shard["command"],
        "passed": True,
        "child_started": True,
        "terminated_by_guard": False,
            "fault": "",
            "returncode": 0,
            "maximum_inferred_unified_memory_mib": 128.0,
            "limits": asdict(freeze.accelerator_shard_policy(contract, shard)),
        "generated_artifacts": freeze.shard_generated_artifact_manifest(shard),
    }

    assert freeze.accelerator_receipt_valid(contract, shard, receipt) is True
    output.write_text('{"trigger_state":"RED"}\n', encoding="utf-8")
    assert freeze.accelerator_receipt_valid(contract, shard, receipt) is False


def test_accelerator_receipt_accepts_only_legacy_default_guard_fields() -> None:
    config = _runner_config()
    contract = config["accelerator_replay"]
    shard = contract["shards"][0]
    receipt = {
        "policy": contract["watchdog_policy"],
        "command": shard["command"],
        "passed": True,
        "child_started": True,
        "terminated_by_guard": False,
        "fault": "",
        "returncode": 0,
        "maximum_inferred_unified_memory_mib": 128.0,
        "limits": asdict(freeze.accelerator_shard_policy(contract, shard)),
    }
    receipt["limits"].pop("memory_guard_mode")
    receipt["limits"].pop("qualified_peak_inferred_unified_memory_mib")

    assert freeze.accelerator_receipt_valid(contract, shard, receipt) is True

    receipt["limits"]["memory_guard_mode"] = "predicted_exhaustion"
    assert freeze.accelerator_receipt_valid(contract, shard, receipt) is False
