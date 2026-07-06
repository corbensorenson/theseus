"""Runtime helpers for SparkStream autonomy cycles.

This module keeps observation, decision, status, and command-runner helpers out of
scripts/autonomy_cycle.py so the cycle wrapper stays small enough for AI maintenance.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autonomy_cycle_support import active_frontier, failed_gates, goal_for_decision

ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "reports" / "sparkstream_status.json"


def consume_watchdog_override(decision: dict[str, Any]) -> None:
    override_id = decision.get("watchdog_override_id")
    if not override_id:
        return
    path = ROOT / "reports" / "autonomy_watchdog_override.json"
    override = read_json(path)
    if override.get("override_id") != override_id:
        return
    override["consumed_utc"] = now()
    write_json(path, override)


def consume_frontier_rotation_request(decision: dict[str, Any]) -> None:
    request_id = decision.get("rotation_request_id")
    if not request_id:
        return
    path = ROOT / "reports" / "frontier_rotation_request.json"
    request = read_json(path)
    if request.get("request_id") != request_id:
        return
    request["consumed_utc"] = now()
    write_json(path, request)


def observe() -> dict[str, Any]:
    reports = ROOT / "reports"
    return {
        "preflight": read_json(reports / "training_preflight_report.json"),
        "candidate_gate": read_json(reports / "candidate_promotion_gate.json"),
        "promotion_closure": read_json(reports / "promotion_closure.json"),
        "accepted_candidate_registry": read_json(reports / "accepted_candidate_registry.json"),
        "frontier_rotation_request": read_json(reports / "frontier_rotation_request.json"),
        "benchmark_ledger": read_json(reports / "benchmark_ledger.json"),
        "residual_escrow": read_json(reports / "residual_escrow.json"),
        "rmi": read_json(reports / "ratcheting_modular_intelligence_report.json"),
        "router": read_json(reports / "octopus_router_report.json"),
        "benchmark_seeker": read_json(reports / "benchmark_seeker_registry.json"),
        "knowledge_sources": read_json(reports / "knowledge_source_registry.json"),
        "online_source_catalog": read_json(reports / "online_source_catalog_report.json"),
        "resource_pantry": read_json(reports / "resource_pantry.json"),
        "legacy_concepts": read_json(reports / "legacy_project_concept_audit.json"),
        "legacy_port_mechanisms": read_json(reports / "legacy_port_mechanisms.json"),
        "legacy_runtime_governance_gate": read_json(reports / "legacy_runtime_governance_gate.json"),
        "legacy_port_runtime_enforcement": read_json(reports / "legacy_port_runtime_enforcement.json"),
        "coherence_delirium_gate": read_json(reports / "coherence_delirium_gate.json"),
        "old_project_registry_port": read_json(reports / "old_project_registry_port.json"),
        "legacy_training_source_audit": read_json(reports / "legacy_training_source_audit.json"),
        "legacy_training_source_sample": read_json(reports / "legacy_training_source_sample.json"),
        "legacy_rl_environment_admission": read_json(reports / "legacy_rl_environment_admission.json"),
        "legacy_rl_smoke_plan": read_json(reports / "legacy_rl_smoke_plan.json"),
        "trace_fabric_capsule_admission": read_json(reports / "trace_fabric_capsule_admission.json"),
        "trace_fabric_capsule_materialization": read_json(reports / "trace_fabric_capsule_materialization.json"),
        "legacy_adapter_bank_training_plan": read_json(reports / "legacy_adapter_bank_training_plan.json"),
        "legacy_active_inference_pilot": read_json(reports / "legacy_active_inference_pilot.json"),
        "training_data_sampler": read_json(reports / "training_data_sampler.json"),
        "checkpoint_registry": read_json(reports / "checkpoint_registry.json"),
        "checkpoint_backup": read_json(reports / "checkpoint_backup_last.json"),
        "update_status": read_json(reports / "update_status.json"),
        "update_offer": read_json(reports / "update_offer_current.json"),
        "update_apply": read_json(reports / "update_apply_last.json"),
        "training_data_inventory": read_json(reports / "training_data_inventory.json"),
        "cell_lifecycle": read_json(reports / "cell_lifecycle.json"),
        "personality_core": read_json(reports / "personality_core.json"),
        "personality_context": read_json(reports / "personality_context_last.json"),
        "personality_drift_eval": read_json(reports / "personality_drift_eval.json"),
        "belief_update_governance": read_json(reports / "belief_update_governance.json"),
        "local_rom_staging": read_json(reports / "local_rom_staging_report.json"),
        "game_asset_inventory": read_json(reports / "game_asset_inventory.json"),
        "local_rom_registry": read_json(reports / "local_rom_registry.json"),
        "minecraft_runtime_probe": read_json(reports / "minecraft_runtime_probe.json"),
        "synthetic_data": read_json(reports / "synthetic_data_curator.json"),
        "synthetic_benchmark_factory": read_json(reports / "synthetic_benchmark_factory.json"),
        "multi_stream_trace_factory": read_json(reports / "multi_stream_trace_factory.json"),
        "multi_stream_code_pressure": read_latest_json(reports, "multi_stream_code_pressure_*_seed*.json"),
        "multi_stream_monitorability_probe": read_json(reports / "multi_stream_monitorability_probe.json"),
        "multi_stream_candidate_gate": read_json(reports / "multi_stream_candidate_gate.json"),
        "rl_benchmark_registry": read_json(reports / "rl_benchmark_registry.json"),
        "resource_governor": read_json(reports / "resource_governor.json"),
        "performance_optimizer": read_json(reports / "performance_optimizer.json"),
        "hive_status": read_json(reports / "hive_status.json"),
        "hive_peers": read_json(reports / "hive_peers.json"),
        "hive_scheduler": read_json(reports / "hive_scheduler.json"),
        "public_hive_contribution": read_json(reports / "public_hive_contribution_status.json"),
        "license_status": read_json(reports / "license_status.json"),
        "autonomous_goal": read_json(reports / "autonomous_goal_last.json"),
        "arm_lifecycle_governance": read_json(reports / "arm_lifecycle_governance.json"),
        "arm_sucker_registry": read_json(reports / "arm_sucker_registry.json"),
        "autonomy_launch_readiness": read_json(reports / "autonomy_launch_readiness.json"),
        "sparkstream_history": read_json(reports / "sparkstream_history.json"),
        "context_packets": read_json(reports / "context_packet_ledger.json"),
        "virtual_context_memory": read_json(reports / "virtual_context_memory_probe.json"),
        "virtual_context_memory_bench": read_json(reports / "virtual_context_memory_bench.json"),
        "virtual_context_memory_graph": read_json(reports / "virtual_context_memory_graph.json"),
        "virtual_context_memory_snapshots": read_json(reports / "virtual_context_memory_snapshots.json"),
        "virtual_context_memory_index": read_json(reports / "virtual_context_memory_index.json"),
        "virtual_context_memory_status": read_json(reports / "virtual_context_memory_status.json"),
        "virtual_context_memory_training_admission": read_json(reports / "virtual_context_memory_training_admission.json"),
        "vcm_task_context_bridge": read_json(reports / "vcm_task_context_bridge.json"),
        "vcm_task_contexts": read_json(reports / "vcm_task_contexts.json"),
        "capability_matrix": read_json(reports / "capability_matrix.json"),
        "benchmaxx_curriculum": read_json(reports / "benchmaxx_curriculum.json"),
        "benchmark_adapter_factory": read_json(reports / "benchmark_adapter_factory.json"),
        "benchmark_pantry_unblocker": read_json(reports / "benchmark_pantry_unblocker.json"),
        "candidate_bottleneck_reducer": read_json(reports / "candidate_bottleneck_reducer.json"),
        "ai_grand_prix_spec": read_json(reports / "ai_grand_prix_spec_digest.json"),
        "python_runtime_compatibility": read_json(reports / "python_runtime_compatibility.json"),
        "architecture_experiment_governance": read_json(reports / "architecture_experiment_governance.json"),
        "architecture_experiment_runner": read_json(reports / "architecture_experiment_runner.json"),
        "autoresearch_gap_audit": read_json(reports / "autoresearch_gap_audit.json"),
        "loop_closure_harvester": read_json(reports / "loop_closure_harvester.json"),
        "loop_closure_tool_promoter": read_json(reports / "loop_closure_tool_promoter.json"),
        "native_voice_io": read_json(reports / "native_voice_io.json"),
        "native_voice_training_manifest": read_json(reports / "native_voice_training_manifest.json"),
        "transfer_eval_suite": read_json(reports / "transfer_eval_suite.json"),
        "arm_transfer_plan": read_json(reports / "arm_transfer_plan.json"),
        "arm_transfer_artifacts": read_json(reports / "arm_transfer_artifacts.json"),
        "model_growth_gate": read_json(reports / "model_growth_gate.json"),
        "self_evolution_governance": read_json(reports / "self_evolution_governance.json"),
        "teacher_self_edit": read_json(reports / "teacher_self_edit_last.json"),
        "teacher_self_edit_proof": read_json(reports / "teacher_self_edit_proof.json"),
        "attd": read_json(reports / "attd_report.json"),
        "attd_maintenance_packets": read_json(reports / "attd_maintenance_packets.json"),
        "external_inference_audit": read_json(reports / "external_inference_audit.json"),
        "viea_autonomy_spine": read_json(reports / "viea_autonomy_spine.json"),
        "viea_artifact_kernel": read_json(reports / "viea_artifact_kernel.json"),
        "viea_command_executor": read_json(reports / "viea_command_executor.json"),
        "viea_action_executor": read_json(reports / "viea_action_executor.json"),
        "feedback_action_queue": read_json(reports / "feedback_action_queue.json"),
        "broad_transfer_closure": read_json(reports / "broad_transfer_closure.json"),
        "broad_transfer_action_queue": read_json(reports / "broad_transfer_action_queue.json"),
        "repo_repair_main_curriculum": read_json(reports / "repo_repair_main_curriculum.json"),
        "viea_repo_repair_learner": read_json(reports / "viea_repo_repair_learner.json"),
        "symliquid_state_engine_queue": read_json(reports / "symliquid_state_engine_queue.json"),
        "symliquid_state_engine": read_json(reports / "symliquid_state_engine.json"),
        "teacher_architect_closure": read_json(reports / "teacher_architect_closure.json"),
        "teacher_architect_experiment_runner": read_json(reports / "teacher_architect_experiment_runner.json"),
    }


def decide_next_action(policy: dict[str, Any], state: dict[str, Any], profile: str) -> dict[str, Any]:
    preflight = state.get("preflight") or {}
    candidate = state.get("candidate_gate") or {}
    failed = failed_gates(candidate)
    pressure = frontier_pressure_state(policy, state, profile)
    rotation_request = active_frontier_rotation_request()
    frontier = active_frontier(
        state.get("benchmark_ledger"),
        preferred_family=str(pressure.get("next_frontier_family") or ""),
        preferred_card_id=str(pressure.get("next_pressure_card_id") or ""),
    )
    watchdog_override = active_watchdog_override()
    run_profile = profile in set(policy.get("allowed_profiles", []))
    reason = "continue_active_frontier_pressure"
    teacher_reason = "promotion_gate_blocked"
    selected_profile = profile
    if preflight and not preflight.get("heavy_training_allowed", False):
        reason = "repair_preflight_before_training"
        selected_profile = "smoke"
        run_profile = True
        teacher_reason = "safety_or_governance_uncertainty"
    elif rotation_request:
        reason = "promotion_rotation_request_pending"
        selected_profile = str(
            rotation_request.get("profile")
            or get_path(policy, ["frontier_policy", "default_training_profile_for_fresh_frontier"], "inner_loop")
        )
        run_profile = selected_profile in set(policy.get("allowed_profiles", []))
        teacher_reason = str(rotation_request.get("teacher_reason") or "benchmark_frontier_design")
        pressure["needs_fresh_frontier"] = True
        apply_rotation_request(pressure, rotation_request)
    elif candidate.get("promote") is True:
        reason = "candidate_promoted_rotate_to_fresh_frontier"
        selected_profile = str(
            get_path(policy, ["frontier_policy", "default_training_profile_for_fresh_frontier"], "inner_loop")
        )
        run_profile = True
        teacher_reason = "benchmark_frontier_design"
        pressure["needs_fresh_frontier"] = True
        if rotation_request:
            apply_rotation_request(pressure, rotation_request)
    elif pressure["candidate_profile_required"]:
        reason = "candidate_profile_evidence_required"
        if pressure.get("curriculum_runnable_now"):
            # Candidate evidence is a gate on the next candidate, not a license
            # to keep minting BabyLM seeds after that lane has graduated. When
            # the curriculum has a runnable frontier, gather the needed
            # evidence there.
            selected_profile = str(
                get_path(policy, ["frontier_policy", "default_training_profile_for_fresh_frontier"], "inner_loop")
            )
        else:
            selected_profile = str(get_path(policy, ["frontier_policy", "candidate_evidence_profile"], "candidate"))
        run_profile = selected_profile in set(policy.get("allowed_profiles", []))
        teacher_reason = "benchmark_frontier_design"
        pressure["needs_fresh_frontier"] = True
        pressure["next_frontier_family"] = str(pressure.get("next_frontier_family") or "babylm_mutated")
    elif pressure["frontier_exhausted"]:
        reason = "frontier_exhausted_rotate_to_fresh_frontier"
        selected_profile = str(
            get_path(policy, ["frontier_policy", "default_training_profile_for_fresh_frontier"], "inner_loop")
        )
        run_profile = True
        teacher_reason = str(get_path(policy, ["frontier_policy", "teacher_reason_when_exhausted"], "frontier_exhausted"))
        pressure["needs_fresh_frontier"] = True
        if pressure.get("curriculum_adapter_required"):
            reason = "curriculum_adapter_required_before_training"
            run_profile = False
            teacher_reason = "benchmark_frontier_design"
            pressure["needs_fresh_frontier"] = False
    elif watchdog_override:
        reason = str(watchdog_override.get("reason") or "watchdog_force_frontier_rotation")
        selected_profile = str(
            watchdog_override.get("profile")
            or get_path(policy, ["frontier_policy", "default_training_profile_for_fresh_frontier"], "inner_loop")
        )
        run_profile = selected_profile in set(policy.get("allowed_profiles", []))
        teacher_reason = str(watchdog_override.get("teacher_reason") or "architecture_wall")
        pressure["needs_fresh_frontier"] = True
        pressure["next_frontier_family"] = str(watchdog_override.get("frontier_family") or "rl_local")
        pressure["watchdog_override_id"] = watchdog_override.get("override_id")
        if watchdog_override.get("rl_frontier_env"):
            pressure["next_rl_frontier_env"] = watchdog_override.get("rl_frontier_env")
        if watchdog_override.get("rl_frontier_seed"):
            pressure["next_rl_frontier_seed"] = watchdog_override.get("rl_frontier_seed")
        if watchdog_override.get("pressure_card_id"):
            pressure["next_pressure_card_id"] = watchdog_override.get("pressure_card_id")
    elif frontier and pressure.get("should_interleave_rl_frontier"):
        reason = "frontier_wall_interleave_rl_frontier"
        selected_profile = str(
            get_path(policy, ["frontier_policy", "default_training_profile_for_fresh_frontier"], "inner_loop")
        )
        run_profile = True
        teacher_reason = "architecture_wall"
        pressure["needs_fresh_frontier"] = True
        pressure["next_frontier_family"] = "rl_local"
    elif pressure.get("architecture_upgrade_due"):
        reason = "frontier_stagnated_below_floor_architecture_upgrade"
        selected_profile = "smoke"
        run_profile = False
        teacher_reason = "architecture_wall"
    elif (
        pressure.get("curriculum_runnable_now")
        and pressure.get("next_frontier_family")
        and (
            not frontier
            or row_frontier_family(frontier) != str(pressure.get("next_frontier_family") or "")
            or (
                pressure.get("next_pressure_card_id")
                and str(pressure.get("next_pressure_card_id")) not in str(frontier.get("best_report") or frontier.get("benchmark_name") or "")
            )
        )
    ):
        reason = "curriculum_runnable_frontier_override"
        selected_profile = str(
            get_path(policy, ["frontier_policy", "default_training_profile_for_fresh_frontier"], "inner_loop")
        )
        run_profile = selected_profile in set(policy.get("allowed_profiles", []))
        teacher_reason = "benchmark_frontier_design"
        pressure["needs_fresh_frontier"] = True
    elif set(failed) & {"seed55_frontier_clears_floor", "active_frontier_clears_floor"}:
        reason = "active_frontier_below_floor_continue_or_bridge"
        teacher_reason = "architecture_wall"
    elif "active_diagnostic_delta_bounded" in failed or "max_residual_delta_bounded" in failed:
        reason = "residual_delta_worsened_diagnose"
        teacher_reason = "residual_conflict"
    elif frontier and frontier.get("wall_type") not in (None, "no_current_wall"):
        reason = "frontier_wall_diagnosis"
        teacher_reason = "architecture_wall"
    selected_profile = cap_profile(policy, selected_profile)
    frontier_seed = None
    frontier_eval = ""
    frontier_report = ""
    frontier_family = ""
    rl_frontier_env = ""
    rl_frontier_seed = None
    force_frontier_generation = False
    pressure_card_id = ""
    if pressure.get("needs_fresh_frontier"):
        frontier_family = str(pressure.get("next_frontier_family") or "babylm_mutated")
        if frontier_family == "rl_local":
            rl_frontier_env = str(pressure.get("next_rl_frontier_env") or "")
            rl_frontier_seed = int(pressure.get("next_rl_frontier_seed") or 1)
        elif frontier_family in {"minecraft_rl", "drone_rl", "coding_local_sandbox", "web_agent_local", "transfer_eval"}:
            pressure_card_id = str(pressure.get("next_pressure_card_id") or "")
            rl_frontier_seed = int(pressure.get("next_rl_frontier_seed") or 1)
        else:
            frontier_seed = pressure["next_mutated_babylm_seed"]
            frontier_eval = f"data/babylm_mutated_holdout_seed{frontier_seed}.jsonl"
            frontier_report = f"reports/babylm_mutated_holdout_seed{frontier_seed}_stateful_grammar_state_frontier.json"
            force_frontier_generation = True
    elif frontier:
        active_artifacts = active_frontier_artifacts(frontier)
        frontier_family = active_artifacts.get("frontier_family", "")
        frontier_seed = active_artifacts.get("frontier_seed")
        frontier_eval = active_artifacts.get("frontier_eval", "")
        frontier_report = active_artifacts.get("frontier_report", "")
        rl_frontier_env = active_artifacts.get("rl_frontier_env", "")
        rl_frontier_seed = active_artifacts.get("rl_frontier_seed")
        pressure_card_id = active_artifacts.get("pressure_card_id", "")
    allow_network_fetch = bool(
        get_path(policy, ["benchmark_discovery", "enabled"], True)
        and (
            pressure["frontier_exhausted"]
            or pressure.get("needs_fresh_frontier")
            or get_path(policy, ["frontier_policy", "licensed_discovery_when_frontier_exhausted"], True)
        )
    )
    discovery_queries = []
    if allow_network_fetch:
        limit = int(get_path(policy, ["benchmark_discovery", "queries_per_cycle"], 3))
        discovery_queries = list(get_path(policy, ["benchmark_discovery", "queries"], []))[: max(1, limit)]
    return {
        "reason": reason,
        "run_profile": run_profile,
        "profile": selected_profile,
        "teacher_reason": teacher_reason,
        "failed_candidate_gates": failed,
        "frontier": frontier,
        "frontier_pressure": pressure,
        "frontier_family": frontier_family,
        "frontier_seed": frontier_seed,
        "frontier_eval": frontier_eval,
        "frontier_report": frontier_report,
        "rl_frontier_env": rl_frontier_env,
        "rl_frontier_seed": rl_frontier_seed,
        "pressure_card_id": pressure_card_id,
        "force_frontier_generation": force_frontier_generation,
        "allow_network_fetch": allow_network_fetch,
        "benchmark_discovery_queries": discovery_queries,
        "goal": goal_for_decision(reason, frontier, pressure),
        "watchdog_override_id": pressure.get("watchdog_override_id"),
        "rotation_request_id": pressure.get("rotation_request_id"),
    }


def active_watchdog_override() -> dict[str, Any] | None:
    override = read_json(ROOT / "reports" / "autonomy_watchdog_override.json")
    if not override or override.get("consumed_utc"):
        return None
    expires = str(override.get("expires_utc") or "")
    if expires:
        try:
            if datetime.fromisoformat(expires).timestamp() < time.time():
                return None
        except ValueError:
            return None
    if override.get("frontier_family") not in {
        "babylm_mutated",
        "rl_local",
        "minecraft_rl",
        "drone_rl",
        "coding_local_sandbox",
        "web_agent_local",
        "transfer_eval",
    }:
        return None
    return override


def active_frontier_rotation_request() -> dict[str, Any] | None:
    path = ROOT / "reports" / "frontier_rotation_request.json"
    request = read_json(path)
    if not request or request.get("consumed_utc"):
        return None
    expires = str(request.get("expires_utc") or "")
    if expires:
        try:
            if datetime.fromisoformat(expires).timestamp() < time.time():
                return None
        except ValueError:
            return None
    if request.get("frontier_family") not in {
        "babylm_mutated",
        "rl_local",
        "minecraft_rl",
        "drone_rl",
        "coding_local_sandbox",
        "web_agent_local",
        "transfer_eval",
    }:
        return None
    superseded_by = curriculum_supersedes_rotation_request(request)
    if superseded_by:
        request["consumed_utc"] = now()
        request["superseded_by_curriculum"] = superseded_by
        write_json(path, request)
        return None
    return request


def curriculum_supersedes_rotation_request(request: dict[str, Any]) -> dict[str, Any]:
    curriculum = read_json(ROOT / "reports" / "benchmaxx_curriculum.json")
    next_frontier = curriculum.get("next_frontier") if isinstance(curriculum.get("next_frontier"), dict) else {}
    rotation = next_frontier.get("same_family_rotation") if isinstance(next_frontier.get("same_family_rotation"), dict) else {}
    recommended = str(next_frontier.get("recommended_env") or "")
    requested = str(request.get("pressure_card_id") or "")
    family = str(request.get("frontier_family") or "")
    if not recommended or not requested or recommended == requested:
        return {}
    if family != str(next_frontier.get("family") or ""):
        return {}
    if not bool(next_frontier.get("runnable_now")):
        return {}
    reason = str(rotation.get("reason") or "")
    if reason not in {
        "rotate_to_public_code_graduation_ready_card",
        "rotate_promoted_regression_card",
        "rotate_public_code_transfer_stalled_card",
    }:
        return {}
    return {
        "recommended_env": recommended,
        "requested_env": requested,
        "rotation_reason": reason,
        "curriculum_report": "reports/benchmaxx_curriculum.json",
    }


def apply_rotation_request(pressure: dict[str, Any], request: dict[str, Any]) -> None:
    pressure["next_frontier_family"] = str(request.get("frontier_family") or pressure.get("next_frontier_family") or "")
    pressure["next_pressure_card_id"] = str(request.get("pressure_card_id") or pressure.get("next_pressure_card_id") or "")
    pressure["next_rl_frontier_seed"] = int(request.get("rl_frontier_seed") or pressure.get("next_rl_frontier_seed") or 1)
    if request.get("rl_frontier_env"):
        pressure["next_rl_frontier_env"] = str(request.get("rl_frontier_env"))
    pressure["rotation_request_id"] = request.get("request_id")


def frontier_pressure_state(policy: dict[str, Any], state: dict[str, Any], requested_profile: str) -> dict[str, Any]:
    ledger = state.get("benchmark_ledger")
    rows = ledger if isinstance(ledger, list) else []
    frontiers = [row for row in rows if isinstance(row, dict) and row.get("lifecycle") == "frontier"]
    regressions = [row for row in rows if isinstance(row, dict) and row.get("lifecycle") == "regression"]
    candidate = state.get("candidate_gate") or {}
    failed = set(failed_gates(candidate))
    profile_report = read_json(ROOT / "reports" / "training_ratchet_profile_run.json")
    candidate_profile_required = (
        "candidate_profile_evidence_complete" in failed
        and bool(regressions)
        and not frontiers
        and profile_report.get("profile") != "candidate"
    )
    exhausted = bool(regressions) and not frontiers
    next_seed = next_mutated_babylm_seed(
        rows,
        min_seed=int(get_path(policy, ["frontier_policy", "fresh_babylm_min_seed"], 61)),
        step=int(get_path(policy, ["frontier_policy", "fresh_babylm_seed_step"], 6)),
    )
    latest_babylm_seed = latest_mutated_babylm_seed(rows)
    min_seed = int(get_path(policy, ["frontier_policy", "fresh_babylm_min_seed"], 61))
    seed_step = max(1, int(get_path(policy, ["frontier_policy", "fresh_babylm_seed_step"], 6)))
    babylm_rotation_count = (
        0
        if latest_babylm_seed is None or latest_babylm_seed < min_seed
        else ((latest_babylm_seed - min_seed) // seed_step) + 1
    )
    max_babylm_per_rl = max(1, int(get_path(policy, ["frontier_policy", "max_babylm_mutations_per_rl_frontier"], 2)))
    required_rl_frontiers = babylm_rotation_count // max_babylm_per_rl
    latest_rl_seed = latest_rl_frontier_seed(ROOT / "reports")
    rl_env_cycle = [
        str(item)
        for item in get_path(policy, ["frontier_policy", "rl_frontier_env_cycle"], [])
        if str(item)
    ]
    active_wall = any(
        row.get("wall_type") not in (None, "", "no_current_wall")
        for row in frontiers
        if isinstance(row, dict)
    )
    active_attempt_count = max(
        [
            int(get_path(row, ["graduation_policy", "attempt_count"], 0) or 0)
            for row in frontiers
            if isinstance(row, dict)
        ]
        or [0]
    )
    below_floor_rows = below_floor_stagnation_rows(policy, frontiers)
    architecture_upgrade_due = any(row.get("architecture_upgrade_due") for row in below_floor_rows)
    active_report_mtime = latest_active_frontier_report_mtime(frontiers)
    latest_rl_mtime = latest_rl_frontier_mtime(ROOT / "reports")
    wall_interleave_after = max(
        1,
        int(get_path(policy, ["frontier_policy", "wall_interleave_after_active_frontier_attempts"], 12)),
    )
    once_per_active_report = bool(
        get_path(policy, ["frontier_policy", "wall_interleave_once_per_active_frontier_report"], True)
    )
    should_interleave_rl = bool(
        frontiers
        and active_wall
        and get_path(policy, ["frontier_policy", "interleave_rl_when_frontier_walls"], True)
    )
    wall_interleave_pending = bool(
        should_interleave_rl
        and active_attempt_count >= wall_interleave_after
        and (
            not once_per_active_report
            or active_report_mtime <= 0
            or latest_rl_mtime < active_report_mtime
        )
    )
    should_run_rl = bool(
        (exhausted or should_interleave_rl or wall_interleave_pending)
        and rl_env_cycle
        and get_path(policy, ["rl_benchmarks", "enabled"], True)
        and (latest_rl_seed < required_rl_frontiers or wall_interleave_pending)
    )
    next_rl_seed = latest_rl_seed + 1
    next_rl_env = rl_env_cycle[(next_rl_seed - 1) % len(rl_env_cycle)] if rl_env_cycle else ""
    curriculum_next = state.get("benchmaxx_curriculum") or {}
    curriculum_family = str(get_path(curriculum_next, ["next_frontier", "family"], "") or "")
    curriculum_runner = str(get_path(curriculum_next, ["next_frontier", "runner_family"], "") or "")
    curriculum_runnable_now = bool(get_path(curriculum_next, ["next_frontier", "runnable_now"], False))
    curriculum_recommended_env = str(get_path(curriculum_next, ["next_frontier", "recommended_env"], "") or "")
    curriculum_runner_map = {
        "minecraft_rl_local": "minecraft_rl",
        "drone_rl_local": "drone_rl",
        "coding_local_sandbox": "coding_local_sandbox",
        "web_agent_local": "web_agent_local",
        "transfer_eval_local": "transfer_eval",
        "rl_local": "rl_local",
        "babylm_mutated": "babylm_mutated",
    }
    curriculum_mapped_family = curriculum_runner_map.get(curriculum_runner, curriculum_family)
    if curriculum_runnable_now and curriculum_mapped_family:
        next_family = curriculum_mapped_family
    elif should_run_rl:
        next_family = "rl_local"
    elif exhausted and curriculum_runner == "emulator_rl_adapter_smoke":
        # Until the emulator wrapper smoke runner is first-class, use the
        # local RL runner as a bridge so exhausted language frontiers do not
        # fall back into endless fresh BabyLM seeds.
        next_family = "rl_local" if rl_env_cycle else "emulator_rl"
    elif exhausted and curriculum_family and curriculum_runner == "adapter_required":
        next_family = curriculum_family
    else:
        next_family = "rl_local" if should_run_rl else "babylm_mutated"
    return {
        "requested_profile": requested_profile,
        "frontier_count": len(frontiers),
        "regression_count": len(regressions),
        "frontier_exhausted": exhausted,
        "candidate_profile_required": candidate_profile_required,
        "needs_fresh_frontier": bool(
            exhausted
            and get_path(policy, ["frontier_policy", "force_new_frontier_when_all_tracked_surfaces_regression"], True)
        ),
        "next_mutated_babylm_seed": next_seed,
        "latest_mutated_babylm_seed": latest_babylm_seed,
        "babylm_rotation_count": babylm_rotation_count,
        "max_babylm_mutations_per_rl_frontier": max_babylm_per_rl,
        "required_rl_frontier_count": required_rl_frontiers,
        "latest_rl_frontier_seed": latest_rl_seed,
        "next_rl_frontier_seed": next_rl_seed,
        "next_rl_frontier_env": next_rl_env,
        "active_frontier_wall": active_wall,
        "active_frontier_attempt_count": active_attempt_count,
        "below_floor_stagnation": below_floor_rows,
        "architecture_upgrade_due": architecture_upgrade_due,
        "wall_interleave_after_active_frontier_attempts": wall_interleave_after,
        "wall_interleave_once_per_active_frontier_report": once_per_active_report,
        "wall_interleave_pending": wall_interleave_pending,
        "active_frontier_report_mtime_utc": iso_from_ts(active_report_mtime),
        "latest_rl_frontier_mtime_utc": iso_from_ts(latest_rl_mtime),
        "should_interleave_rl_frontier": should_run_rl and should_interleave_rl,
        "next_frontier_family": next_family,
        "next_pressure_card_id": curriculum_recommended_env
        if next_family in {"minecraft_rl", "drone_rl", "coding_local_sandbox", "web_agent_local", "transfer_eval"}
        else "",
        "curriculum_stage": get_path(curriculum_next, ["summary", "current_stage_id"], None),
        "curriculum_next_frontier_family": curriculum_family or None,
        "curriculum_runner_family": curriculum_runner or None,
        "curriculum_runnable_now": curriculum_runnable_now,
        "curriculum_recommended_env": curriculum_recommended_env or None,
        "curriculum_adapter_required": bool(exhausted and curriculum_family and curriculum_runner == "adapter_required"),
        "candidate_promote": candidate.get("promote"),
        "candidate_failed_gates": sorted(failed),
    }


def next_mutated_babylm_seed(rows: list[Any], *, min_seed: int, step: int) -> int:
    latest = latest_mutated_babylm_seed(rows)
    if latest is None:
        return min_seed
    return max(min_seed, latest + max(1, step))


def latest_mutated_babylm_seed(rows: list[Any]) -> int | None:
    import re

    seeds = []
    for row in rows:
        if not isinstance(row, dict) or row.get("benchmark_name") != "babylm_mutated_holdout":
            continue
        text = str(row.get("best_report") or "")
        match = re.search(r"seed(\d+)", text)
        if match:
            seeds.append(int(match.group(1)))
    return max(seeds) if seeds else None


def below_floor_stagnation_rows(policy: dict[str, Any], frontiers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    threshold = int(get_path(policy, ["teacher_escalation", "architecture_upgrade_attempt_threshold"], 8) or 8)
    epsilon = float(get_path(policy, ["teacher_escalation", "architecture_upgrade_stagnation_epsilon"], 0.002) or 0.002)
    requires_below_floor = bool(get_path(policy, ["teacher_escalation", "architecture_upgrade_requires_below_floor"], True))
    rows: list[dict[str, Any]] = []
    for row in frontiers:
        if not isinstance(row, dict):
            continue
        policy_row = row.get("graduation_policy") if isinstance(row.get("graduation_policy"), dict) else {}
        score = number(row.get("score"), default=number(row.get("accuracy"), default=0.0))
        floor = number(policy_row.get("floor_threshold"), default=0.70)
        current_threshold = number(policy_row.get("current_threshold"), default=0.90)
        attempt_count = int(policy_row.get("attempt_count") or 0)
        recent_delta_raw = policy_row.get("recent_delta")
        recent_delta = 0.0 if recent_delta_raw is None else number(recent_delta_raw, default=0.0)
        below_target = score < (floor if requires_below_floor else current_threshold)
        stalled = abs(recent_delta) <= epsilon
        due = bool(below_target and stalled and attempt_count >= max(1, threshold))
        if below_target or due:
            rows.append(
                {
                    "benchmark_name": row.get("benchmark_name"),
                    "score": score,
                    "floor": floor,
                    "current_threshold": current_threshold,
                    "attempt_count": attempt_count,
                    "recent_delta": recent_delta,
                    "stagnation_epsilon": epsilon,
                    "attempt_threshold": threshold,
                    "architecture_upgrade_due": due,
                }
            )
    return rows


def active_frontier_artifacts(frontier: dict[str, Any]) -> dict[str, Any]:
    """Resolve explicit runner inputs for the current active frontier."""
    name = str(frontier.get("benchmark_name") or "")
    benchmark_type = str(frontier.get("benchmark_type") or "")
    best_report = str(frontier.get("best_report") or "")
    if name == "babylm_mutated_holdout" or "mutated" in benchmark_type:
        seed = seed_from_text(best_report) or seed_from_text(str(frontier.get("capability_narrative") or ""))
        if seed is None:
            return {}
        return {
            "frontier_family": "babylm_mutated",
            "frontier_seed": seed,
            "frontier_eval": f"data/babylm_mutated_holdout_seed{seed}.jsonl",
            "frontier_report": best_report
            or f"reports/babylm_mutated_holdout_seed{seed}_stateful_grammar_state_frontier.json",
        }
    if name.startswith("ocean-"):
        return {
            "frontier_family": "rl_local",
            "rl_frontier_env": name,
            "rl_frontier_seed": seed_from_text(best_report),
        }
    if "pressure_" in best_report or name.startswith(("minecraft_rl_", "drone_rl_", "coding_", "web_agent_", "asi_transfer", "transfer_")):
        return pressure_frontier_artifacts(name, best_report)
    return {}


def pressure_frontier_artifacts(name: str, best_report: str) -> dict[str, Any]:
    family = "transfer_eval" if name.startswith("asi_transfer") else ""
    if name.startswith("minecraft_rl_"):
        family = "minecraft_rl"
    elif name.startswith("drone_rl_"):
        family = "drone_rl"
    elif name.startswith("coding_"):
        family = "coding_local_sandbox"
    elif name.startswith("web_agent_"):
        family = "web_agent_local"
    elif name.startswith("transfer_"):
        family = "transfer_eval"
    card_id = pressure_card_from_report(best_report)
    if not family and (
        card_id.startswith("source_crafter")
        or card_id.startswith("source_craftax")
        or card_id.startswith("source_minerl")
        or card_id.startswith("source_minedojo")
        or card_id.startswith("source_malmo")
        or card_id.startswith("source_voyager_minecraft")
    ):
        family = "minecraft_rl"
    if not family and (
        card_id.startswith("source_gym_pybullet")
        or card_id.startswith("source_pyflyt")
        or card_id.startswith("source_mavsdk")
    ):
        family = "drone_rl"
    if not family and card_id in {
        "source_bigcodebench",
        "source_evalplus",
        "source_human_eval",
        "source_mbpp",
        "source_livecodebench",
        "source_opencode",
        "source_swe_bench",
        "source_swe_agent",
        "source_mini_swe_agent",
        "source_codeclash",
        "source_swe_polybench",
        "source_swe_gen",
    }:
        family = "coding_local_sandbox"
    if not family and card_id.startswith("source_webarena"):
        family = "web_agent_local"
    if not card_id and family == "transfer_eval":
        card_id = "transfer_eval_suite"
    return {
        "frontier_family": family or "transfer_eval",
        "pressure_card_id": card_id,
        "rl_frontier_seed": seed_from_text(best_report) or 1,
    }


def pressure_card_from_report(path: str) -> str:
    import re

    name = Path(path).stem
    match = re.match(r"pressure_(.+)_seed\d+", name)
    if not match:
        return ""
    raw = match.group(1)
    known = [
        "source_gym_pybullet_drones",
        "source_crafter",
        "source_craftax",
        "source_minerl",
        "source_minedojo",
        "source_malmo",
        "source_voyager_minecraft",
        "source_pyflyt_waypoints",
        "source_pyflyt",
        "source_mavsdk_python",
        "source_bigcodebench",
        "source_evalplus",
        "source_human_eval",
        "source_mbpp",
        "source_livecodebench",
        "source_opencode",
        "source_swe_bench",
        "source_swe_agent",
        "source_mini_swe_agent",
        "source_codeclash",
        "source_swe_polybench",
        "source_swe_gen",
        "source_webarena",
        "transfer_eval_suite",
    ]
    for item in known:
        if raw == item:
            return item
    return raw


def seed_from_text(value: str) -> int | None:
    import re

    match = re.search(r"seed(\d+)", value)
    return int(match.group(1)) if match else None


def latest_rl_frontier_seed(reports_dir: Path) -> int:
    import re

    seeds = []
    for report in reports_dir.glob("rl_frontier_*_seed*_train.json"):
        match = re.search(r"_seed(\d+)_", report.name)
        if match:
            seeds.append(int(match.group(1)))
    for report in reports_dir.glob("rl_frontier_*_seed*_smoke.json"):
        match = re.search(r"_seed(\d+)_", report.name)
        if match:
            seeds.append(int(match.group(1)))
    return max(seeds) if seeds else 0


def latest_rl_frontier_mtime(reports_dir: Path) -> float:
    reports = [
        *reports_dir.glob("rl_frontier_*_seed*_train.json"),
        *reports_dir.glob("rl_frontier_*_seed*_smoke.json"),
    ]
    return max((report.stat().st_mtime for report in reports if report.exists()), default=0.0)


def latest_active_frontier_report_mtime(frontiers: list[dict[str, Any]]) -> float:
    mtimes: list[float] = []
    for row in frontiers:
        if not isinstance(row, dict):
            continue
        report = ROOT / str(row.get("best_report") or "")
        if report.exists():
            mtimes.append(report.stat().st_mtime)
    return max(mtimes, default=0.0)


def iso_from_ts(value: float) -> str:
    if value <= 0:
        return ""
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def cap_profile(policy: dict[str, Any], profile: str) -> str:
    allowed = list(policy.get("allowed_profiles", []))
    if profile not in allowed:
        return str(policy.get("default_profile", "inner_loop"))
    max_profile = str(policy.get("max_profile_without_user_request", profile))
    order = ["smoke", "inner_loop", "candidate", "seed_sweep"]
    if profile in order and max_profile in order and order.index(profile) > order.index(max_profile):
        return max_profile
    return profile










def run_step(
    command: list[str],
    *,
    timeout: int,
    execute: bool,
    name: str,
    allow_failure: bool = False,
) -> dict[str, Any]:
    planforge_node = planforge_node_for_step(name)
    effect_contract = command_effect_contract(name, command, planforge_node)
    if not execute:
        return skipped_step(name, "not_executed", command)
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "name": name,
            "planforge_node": planforge_node,
            "taskspell_effect": effect_contract,
            "command": command,
            "allow_failure": allow_failure,
            "returncode": result.returncode,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "planforge_node": planforge_node,
            "taskspell_effect": effect_contract,
            "command": command,
            "allow_failure": allow_failure,
            "returncode": 124,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "error": "timeout",
        }


def skipped_step(name: str, reason: str, command: list[str]) -> dict[str, Any]:
    planforge_node = planforge_node_for_step(name)
    return {
        "name": name,
        "planforge_node": planforge_node,
        "taskspell_effect": command_effect_contract(name, command, planforge_node),
        "command": command,
        "allow_failure": False,
        "returncode": 0,
        "runtime_ms": 0,
        "skipped": True,
        "reason": reason,
    }


def command_effect_contract(name: str, command: list[str], planforge_node: str) -> dict[str, Any]:
    payload = {
        "name": name,
        "command": command,
        "planforge_node": planforge_node,
        "contract": "autonomy_command_effect_replay_v0",
    }
    digest = stable_hash(payload)
    return {
        "contract": payload["contract"],
        "planforge_node": planforge_node,
        "effect_contract_hash": digest,
        "replay_record_id": digest[:20],
        "verification_record": "returncode_runtime_stdout_stderr_tail",
    }


def planforge_node_for_step(name: str) -> str:
    lower = name.lower()
    buckets = [
        ("taskspell_lock", ["taskspell", "runtime_enforcement", "legacy_port_runtime"]),
        ("teacher_self_edit", ["teacher", "self_evolution", "self_edit"]),
        ("proxy_truth_audit", ["proxy", "external_inference", "legacy_runtime", "candidate"]),
        ("coherence_delirium", ["coherence", "delirium"]),
        ("active_frontier_pressure", ["pressure", "frontier", "ratchet", "training_profile"]),
        ("residual_escrow_update", ["residual"]),
        ("world_job_runtime", ["world", "drone", "emulator", "active_inference", "rl_smoke"]),
        ("adapter_bank_transfer", ["adapter", "transfer", "bridge"]),
        ("trace_fabric_exchange", ["trace", "context_packet", "rlds"]),
        ("checkpoint_and_backup", ["checkpoint", "backup", "promotion_closure"]),
    ]
    for node, tokens in buckets:
        if any(token in lower for token in tokens):
            return node
    return "observe_status"


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()










def timeout_for_profile(policy: dict[str, Any], profile: str) -> int:
    return int(get_path(policy, ["command_timeouts_seconds", profile], 1800))


def profile_train_limit(profile: str) -> int:
    profiles = read_json(ROOT / "configs" / "training_profiles_rtx2060super.json")
    return int(get_path(profiles, ["profiles", profile, "babylm", "train_limit"], 50000))






def update_status(
    cycle_id: str,
    phase: str,
    profile: str,
    message: str,
    *,
    ok: bool | None = None,
) -> None:
    payload = {
        "policy": "sparkstream_status_v0",
        "updated_utc": now(),
        "cycle_id": cycle_id,
        "phase": phase,
        "profile": profile,
        "message": message,
        "ok": ok,
    }
    write_json(STATUS_PATH, payload)


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def number(value: Any, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed:
        return default
    return parsed


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return {}
        return json.loads(text)
    except (OSError, json.JSONDecodeError):
        return {}


def read_latest_json(directory: Path, pattern: str) -> Any:
    candidates = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return read_json(candidates[0]) if candidates else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()
