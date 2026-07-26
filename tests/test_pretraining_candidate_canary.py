from __future__ import annotations

import copy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pretraining_candidate_canary as canary  # noqa: E402


def test_contract_covers_every_stable_slot_and_candidate_budget() -> None:
    contract = canary.load_contract()
    assert set(contract["stable_slots"]) == {row["slot"] for row in contract["implementation_cards"]}
    assert len(contract["canaries"]) >= 6


def test_candidate_lease_is_content_bound_and_candidate_scoped(tmp_path: Path) -> None:
    root = ROOT / "runtime" / "t0a_canaries" / "mtp_adequacy" / "test-run"
    lease = canary.candidate_lease(
        candidate_id="mtp_adequacy",
        max_steps=32,
        scratch_checkpoint_root=root,
        targets=["shared_trunk"],
        phase="pretraining",
        resume=False,
    )
    assert lease["authorized"] is True
    assert canary.validate_lease(lease)


def test_kerc_lease_binds_matched_control_and_behavior_panel() -> None:
    root = ROOT / "runtime" / "t0a_canaries" / "rdc_kerc_adequacy" / "test-run"
    lease = canary.candidate_lease(
        candidate_id="rdc_kerc_adequacy",
        max_steps=128,
        scratch_checkpoint_root=root,
        targets=["english_kerc", "english_surface_control"],
        phase="kernel_english",
        resume=False,
        selected_seed=20260722,
    )
    assert lease["authorized"] is True
    assert lease["behavior_eval_rows"] == 16
    assert lease["selected_seed"] == 20260722
    assert lease["seed_execution_mode"] == "single_bound_seed"
    assert lease["targets"] == ["english_kerc", "english_surface_control"]
    assert lease["host_safety_policy"]["minimum_available_before_launch_mib"] == 2048
    assert lease["host_safety_policy"]["minimum_available_during_run_mib"] == 2048
    assert lease["host_safety_policy"]["maximum_swapout_growth_mib"] == 16
    assert lease["execution_policy"]["objective_gradient_checkpointing"] is True
    assert lease["execution_policy"]["objective_gradient_decomposition"] is True
    assert lease["execution_policy"]["token_loss_position_chunk_size"] == 64
    assert lease["execution_policy"]["attention_query_chunk_size"] == 32
    assert lease["execution_policy"]["attention_key_chunk_size"] == 32
    assert canary.validate_lease(lease)

    denied = canary.candidate_lease(
        candidate_id="rdc_kerc_adequacy",
        max_steps=128,
        scratch_checkpoint_root=root,
        targets=["english_kerc", "english_surface_control"],
        phase="kernel_english",
        resume=False,
        selected_seed=99,
    )
    assert denied["authorized"] is False
    assert "seed_not_authorized" in denied["faults"]


def test_k5_measured_peak_is_advisory_and_live_reserve_is_the_launch_gate() -> None:
    mapping = canary.candidate_host_safety_mapping("rdc_kerc_k5_adequacy")
    assert mapping["maximum_swapout_growth_mib"] == 320
    assert mapping["swapout_growth_action"] == "report_only"
    preflight = mapping["measured_launch_preflight"]
    assert preflight["maximum_inferred_unified_memory_mib"] == 6268.875
    assert preflight["required_live_reserve_mib"] == 2048
    assert mapping["minimum_available_before_launch_mib"] == 2048
    assert preflight["resolved_minimum_available_before_launch_mib"] == 2048
    assert preflight["advisory_projected_available_mib"] == 8316.875
    assert preflight["measured_peak_role"] == (
        "advisory_capacity_projection_not_launch_gate"
    )
    assert preflight["launch_gate"] == "configured_live_reserve_only"
    assert preflight["runtime_enforcement"] == (
        "external_live_reserve_watchdog_with_swap_growth_telemetry"
    )
    assert preflight["qualified_maximum_training_sequence_tokens"] == 1005
    assert preflight["current_maximum_training_sequence_tokens"] == 628
    assert preflight["qualified_maximum_token_loss_position_chunk_size"] == 64
    assert preflight["current_token_loss_position_chunk_size"] == 32
    assert preflight["qualified_maximum_attention_query_chunk_size"] == 32
    assert preflight["current_attention_query_chunk_size"] == 32
    assert preflight["qualified_maximum_attention_key_chunk_size"] == 32
    assert preflight["current_attention_key_chunk_size"] == 32
    assert preflight["command_marker"] == "rdc_kerc_adequacy"
    contract = canary.load_contract()
    candidate = next(
        row
        for row in contract["canaries"]
        if row["candidate_id"] == "rdc_kerc_k5_adequacy"
    )
    assert candidate["max_steps"] == 6144
    lease = canary.candidate_lease(
        candidate_id="rdc_kerc_k5_adequacy",
        max_steps=2,
        scratch_checkpoint_root=(
            ROOT / "runtime" / "t0a_canaries" / "rdc_kerc_k5_adequacy" / "isolation"
        ),
        targets=["english_kerc"],
        phase="kernel_english",
        resume=False,
        selected_seed=20260722,
    )
    assert lease["execution_policy"]["kerc_stage_train_stage_embedding"] is True
    assert lease["execution_policy"]["kerc_stage_detach_frozen_trunk"] is False
    assert lease["execution_policy"]["retain_segment_checkpoint_generations"] is True
    assert candidate["max_wall_seconds"] == 43200
    assert candidate["execution_policy"]["kernel_optimizer_repetitions"] == 4
    assert candidate["execution_policy"].get("target_token_frequency_balance_power", 0.0) == 0.0
    lease = canary.candidate_lease(
        candidate_id="rdc_kerc_k5_adequacy",
        max_steps=256,
        scratch_checkpoint_root=(
            ROOT / "runtime" / "t0a_canaries" / "rdc_kerc_k5_adequacy" / "test"
        ),
        targets=["english_kerc"],
        phase="kernel_english",
        resume=False,
        selected_seed=20260722,
    )
    assert lease["execution_policy"]["optimizer_state_offload_between_steps"] is True
    assert (
        lease["execution_policy"][
            "optimizer_state_offload_minimum_target_positions"
        ]
        == 600
    )
    assert lease["execution_policy"]["kerc_resource_stress_prefix"] is False


def test_candidate_lease_rejects_generic_or_unsafe_training() -> None:
    outside = canary.candidate_lease(candidate_id="mtp_adequacy", max_steps=32, scratch_checkpoint_root=ROOT / "checkpoints" / "unsafe", targets=["shared_trunk"], phase="pretraining", resume=False)
    assert outside["authorized"] is False
    assert "scratch_namespace_outside_candidate" in outside["faults"]
    resumed = canary.candidate_lease(candidate_id="mtp_adequacy", max_steps=32, scratch_checkpoint_root=ROOT / "runtime" / "t0a_canaries" / "mtp_adequacy" / "run", targets=["shared_trunk"], phase="pretraining", resume=True)
    assert "candidate_resume_forbidden" in resumed["faults"]
    exceeded = canary.candidate_lease(candidate_id="mtp_adequacy", max_steps=97, scratch_checkpoint_root=ROOT / "runtime" / "t0a_canaries" / "mtp_adequacy" / "run", targets=["shared_trunk"], phase="pretraining", resume=False)
    assert "step_budget_exceeded" in exceeded["faults"]
    wrong_target = canary.candidate_lease(candidate_id="mtp_adequacy", max_steps=8, scratch_checkpoint_root=ROOT / "runtime" / "t0a_canaries" / "mtp_adequacy" / "run", targets=["rust"], phase="pretraining", resume=False)
    assert "target_not_authorized" in wrong_target["faults"]


def test_contract_mutations_fail_closed() -> None:
    contract = canary.load_contract()
    broken = copy.deepcopy(contract)
    broken["implementation_cards"][0].pop("rollback_policy")
    path = ROOT / "runtime" / "t0a_canaries" / "broken-contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(canary.canonical(broken), encoding="utf-8")
        try:
            canary.load_contract(path)
        except canary.CandidateCanaryFault as exc:
            assert "implementation_card_incomplete" in str(exc)
        else:
            raise AssertionError("mutated contract was accepted")
    finally:
        path.unlink(missing_ok=True)


def test_negative_attention_key_chunk_size_fails_closed() -> None:
    contract = copy.deepcopy(canary.load_contract())
    row = next(
        item
        for item in contract["canaries"]
        if item["candidate_id"] == "rdc_kerc_adequacy"
    )
    row["execution_policy"]["attention_key_chunk_size"] = -1
    path = ROOT / "runtime" / "t0a_canaries" / "invalid-key-chunk-contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(canary.canonical(contract), encoding="utf-8")
        try:
            canary.load_contract(path)
        except canary.CandidateCanaryFault as exc:
            assert "canary_execution_policy_invalid:rdc_kerc_adequacy" in str(exc)
        else:
            raise AssertionError("negative attention key chunk size was accepted")
    finally:
        path.unlink(missing_ok=True)


def test_out_of_range_target_frequency_balance_power_fails_closed() -> None:
    contract = copy.deepcopy(canary.load_contract())
    row = next(
        item
        for item in contract["canaries"]
        if item["candidate_id"] == "rdc_kerc_k5_adequacy"
    )
    row["execution_policy"]["target_token_frequency_balance_power"] = 1.01
    path = ROOT / "runtime" / "t0a_canaries" / "invalid-frequency-power.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(canary.canonical(contract), encoding="utf-8")
        try:
            canary.load_contract(path)
        except canary.CandidateCanaryFault as exc:
            assert "canary_execution_policy_invalid:rdc_kerc_k5_adequacy" in str(exc)
        else:
            raise AssertionError("out-of-range token-frequency power was accepted")
    finally:
        path.unlink(missing_ok=True)


def test_kerc_execution_policy_override_mutations_fail_closed() -> None:
    contract = copy.deepcopy(canary.load_contract())
    contract["execution_policy_overrides"]["rdc_kerc_adequacy"][
        "objective_gradient_decomposition"
    ] = False
    path = (
        ROOT
        / "runtime"
        / "t0a_canaries"
        / "invalid-kerc-execution-policy-override.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(canary.canonical(contract), encoding="utf-8")
        try:
            canary.load_contract(path)
        except canary.CandidateCanaryFault as exc:
            assert "kerc_execution_policy_override_invalid" in str(exc)
        else:
            raise AssertionError("disabled KERC decomposition override was accepted")
    finally:
        path.unlink(missing_ok=True)


def test_unknown_execution_policy_override_fails_closed() -> None:
    contract = copy.deepcopy(canary.load_contract())
    contract["execution_policy_overrides"]["unknown_candidate"] = {
        "objective_gradient_decomposition": True,
        "token_loss_position_chunk_size": 128,
    }
    path = (
        ROOT
        / "runtime"
        / "t0a_canaries"
        / "unknown-execution-policy-override.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(canary.canonical(contract), encoding="utf-8")
        try:
            canary.load_contract(path)
        except canary.CandidateCanaryFault as exc:
            assert "execution_policy_override_candidate_unknown" in str(exc)
        else:
            raise AssertionError("unknown execution-policy override was accepted")
    finally:
        path.unlink(missing_ok=True)


def test_runtime_monitor_enforces_disk_and_position_budgets(tmp_path: Path) -> None:
    contract = canary.load_contract()
    contract = copy.deepcopy(contract)
    row = next(item for item in contract["canaries"] if item["candidate_id"] == "mtp_adequacy")
    row["max_disk_mib"] = 1
    row["max_positions"] = 10
    root = ROOT / "runtime" / "t0a_canaries" / "mtp_adequacy" / "monitor-test"
    lease = canary.candidate_lease(candidate_id="mtp_adequacy", max_steps=1, scratch_checkpoint_root=root, targets=["shared_trunk"], phase="pretraining", resume=False, contract=contract)
    monitor = canary.CandidateCanaryMonitor(lease)
    monitor.check("before_device_step", 1)
    receipt = monitor.finalize([{"optimizer_positions": 11}])
    assert receipt["passed"] is False
    assert receipt["faults"] == ["position_budget_exceeded"]
