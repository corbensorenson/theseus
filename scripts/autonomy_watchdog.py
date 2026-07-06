"""Strict watchdog for SparkStream autonomy.

The daemon keeps the loop alive; this watchdog decides whether "alive" is
actually useful. It turns repeated frontier loops, teacher starvation, stale
daemon state, and promotion-gate drift into explicit RED/YELLOW/GREEN reports.
With --fix it writes a bounded override for the next autonomy cycle and may
make one sparse teacher call through teacher_oracle.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from autonomy_watchdog_actions import active_code_repair_evidence, apply_fixes, code_contract_preflight_state
from autonomy_watchdog_helpers import (
    active_frontier_alignment_ok,
    add_check,
    action,
    age_seconds,
    broad_transfer_blockers,
    candidate_frontier_alignment_required,
    deterministic_taming_integrity,
    file_age_seconds,
    get_path,
    grammar_sucker_integrity,
    latest_cycle_report,
    latest_daemon_terminal_event,
    now,
    read_json,
    read_jsonl_tail,
    read_latest_json,
    read_latest_json_with_path,
    recent_daemon_failures,
    resolve_path,
    seed_from_text,
    selected_work_board_task,
    task_concept,
    teacher_budget_blocks_since_completed,
    trailing_daemon_failures,
    trailing_same_frontier_streak,
    write_json,
)
from progress_integrity_policy import non_promotable_diagnostic_reason


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "autonomy_policy.json"
REPORTS = ROOT / "reports"
OVERRIDE_PATH = REPORTS / "autonomy_watchdog_override.json"
REAL_CODE_FIX_TIMEOUT_SECONDS = 7200
DECODER_RELEVANT_SOURCES = (
    ROOT / "crates" / "symliquid-cli" / "src" / "code_lm_closure.rs",
    ROOT / "scripts" / "code_lm_closure.py",
    ROOT / "scripts" / "code_residual_curriculum.py",
    ROOT / "scripts" / "type_contract_diagnostic.py",
)
CODE_CONTRACT_PREFLIGHT_REPORT = REPORTS / "code_lm_closure_public_contract_preflight_seed23_32.json"
TRAIN_ONCE_FANOUT_CURRENT_SLUG = "frontier_private_transfer_private_only_train_once_v1"
TRAIN_ONCE_FANOUT_LEGACY_SLUG = "private_pressure_private_recovery_train_once_fanout_v1"
TRAIN_ONCE_FANOUT_CLOSURE_REPORT = REPORTS / f"code_lm_closure_{TRAIN_ONCE_FANOUT_CURRENT_SLUG}.json"
TRAIN_ONCE_FANOUT_CLOSURE_REPORTS = (
    TRAIN_ONCE_FANOUT_CLOSURE_REPORT,
    REPORTS / f"code_lm_closure_{TRAIN_ONCE_FANOUT_LEGACY_SLUG}.json",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--out", default="reports/autonomy_watchdog.json")
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    cfg = policy.get("watchdog") or {}
    state = observe()
    report = assess(policy, cfg, state)
    if args.fix:
        pre_fix_report = report
        apply_fixes(policy, cfg, report)
        applied_actions = [
            item
            for item in report.get("recommended_actions", [])
            if item.get("applied")
        ]
        report = assess(policy, cfg, observe())
        report["pre_fix_trigger_state"] = pre_fix_report.get("trigger_state")
        report["applied_actions"] = applied_actions
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 2 if report["trigger_state"] == "RED" and not args.fix else 0


def read_train_once_fanout_closure() -> dict[str, Any]:
    candidates = [path for path in TRAIN_ONCE_FANOUT_CLOSURE_REPORTS if path.exists()]
    wrapper = read_json(REPORTS / "code_lm_train_once_fanout.json")
    wrapper_report = str(wrapper.get("closure_report") or "")
    if wrapper_report:
        wrapper_path = resolve_path(Path(wrapper_report))
        if wrapper_path.exists() and wrapper_path not in candidates:
            candidates.append(wrapper_path)
    if not candidates:
        return {}
    path = max(candidates, key=lambda item: item.stat().st_mtime)
    payload = read_json(path)
    if isinstance(payload, dict):
        payload = dict(payload)
        payload["__closure_report_path"] = str(path.relative_to(ROOT)).replace("\\", "/")
    return payload


def observe() -> dict[str, Any]:
    return {
        "sparkstream": read_json(REPORTS / "sparkstream_status.json"),
        "frontier_policy": read_json(REPORTS / "frontier_policy_status.json"),
        "watchdog_override": read_json(OVERRIDE_PATH),
        "candidate": read_json(REPORTS / "candidate_promotion_gate.json"),
        "maturity_integrity": read_json(REPORTS / "maturity_integrity_audit.json"),
        "learning_scoreboard": read_json(REPORTS / "learning_scoreboard.json"),
        "a_plus_scorecard": read_json(REPORTS / "a_plus_operating_scorecard.json"),
        "broad_transfer_matrix": read_json(REPORTS / "broad_transfer_matrix.json"),
        "broad_code_calibration_scheduler": read_json(REPORTS / "broad_code_calibration_scheduler.json"),
        "overnight_readiness": read_json(REPORTS / "overnight_learning_readiness.json"),
        "cell_lifecycle": read_json(REPORTS / "cell_lifecycle.json"),
        "grammar_suckers": read_json(REPORTS / "grammar_suckers.json"),
        "deterministic_taming": read_json(REPORTS / "deterministic_taming_stack.json"),
        "architecture_guidance": read_json(REPORTS / "architecture_guidance_loop.json"),
        "teacher_budget_audit": read_json(REPORTS / "teacher_budget_audit.json"),
        "code_residual_curriculum": read_json(REPORTS / "code_residual_curriculum.json"),
        "sts_repair_ablation": read_json(REPORTS / "sts_repair_ablation.json"),
        "sts_causal_decoder_ablation": read_json(REPORTS / "sts_causal_decoder_ablation.json"),
        "agent_lane_transfer_gate": read_json(REPORTS / "agent_lane_transfer_gate.json"),
        "open_code_training_pantry": read_json(REPORTS / "open_code_training_pantry.json"),
        "sts_learning": read_json(REPORTS / "sts_learning_forge.json"),
        "sts_native": read_json(REPORTS / "sts_native_parallel_probe.json"),
        "cognitive_context_router": read_json(REPORTS / "cognitive_context_router.json"),
        "teacher_last": read_json(REPORTS / "teacher_oracle_last.json"),
        "teacher_budget": read_json(REPORTS / "teacher_budget_last.json"),
        "candidate_bottleneck": read_json(REPORTS / "candidate_bottleneck_reducer.json"),
        "benchmaxx_curriculum": read_json(REPORTS / "benchmaxx_curriculum.json"),
        "arm_sucker_registry": read_json(REPORTS / "arm_sucker_registry.json"),
        "arm_transfer_plan": read_json(REPORTS / "arm_transfer_plan.json"),
        "architecture_experiment": read_json(REPORTS / "architecture_experiment_governance.json"),
        "genesis_kernel": read_json(REPORTS / "genesis_kernel" / "report.json"),
        "reality_manipulator": read_json(REPORTS / "reality_manipulator.json"),
        "viea_autonomy_spine": read_json(REPORTS / "viea_autonomy_spine.json"),
        "viea_artifact_kernel": read_json(REPORTS / "viea_artifact_kernel.json"),
        "viea_command_executor": read_json(REPORTS / "viea_command_executor.json"),
        "viea_action_executor": read_json(REPORTS / "viea_action_executor.json"),
        "feedback_action_queue": read_json(REPORTS / "feedback_action_queue.json"),
        "broad_transfer_action_queue": read_json(REPORTS / "broad_transfer_action_queue.json"),
        "hive_work_board_executor": read_json(REPORTS / "hive_work_board_executor.json"),
        "service_process_hygiene": read_json(REPORTS / "service_process_hygiene.json"),
        "system_efficiency_audit": read_json(REPORTS / "system_efficiency_audit.json"),
        "autonomy_rotation_governor": read_json(REPORTS / "autonomy_rotation_governor_v2.json"),
        "high_transfer_curriculum_scheduler": read_json(REPORTS / "high_transfer_curriculum_scheduler.json"),
        "code_contract_preflight": read_json(CODE_CONTRACT_PREFLIGHT_REPORT),
        "code_lm_private_pressure": read_json(REPORTS / "code_lm_closure_private_pressure_private.json"),
        "code_lm_private_pressure_rust": read_json(REPORTS / "code_lm_closure_rust_private_pressure_private.json"),
        "code_lm_private_pressure_rust_heartbeat": read_json(REPORTS / "code_lm_closure_rust_private_pressure_private.heartbeat.json"),
        "code_lm_latest_rust_heartbeat": read_latest_json_with_path(REPORTS, "code_lm_closure_rust*.heartbeat.json"),
        "code_lm_train_once_fanout": read_json(REPORTS / "code_lm_train_once_fanout.json"),
        "code_lm_train_once_fanout_closure": read_train_once_fanout_closure(),
        "code_lm_shard_strategy_audit": read_json(REPORTS / "code_lm_shard_strategy_audit.json"),
        "repo_repair_main_curriculum": read_json(REPORTS / "repo_repair_main_curriculum.json"),
        "viea_repo_repair_learner": read_json(REPORTS / "viea_repo_repair_learner.json"),
        "symliquid_state_engine_queue": read_json(REPORTS / "symliquid_state_engine_queue.json"),
        "symliquid_state_engine": read_json(REPORTS / "symliquid_state_engine.json"),
        "teacher_architect_closure": read_json(REPORTS / "teacher_architect_closure.json"),
        "teacher_architect_experiment_runner": read_json(REPORTS / "teacher_architect_experiment_runner.json"),
        "personality_core": read_json(REPORTS / "personality_core.json"),
        "personality_context": read_json(REPORTS / "personality_context_last.json"),
        "personality_drift": read_json(REPORTS / "personality_drift_eval.json"),
        "personality_runtime_audit": read_json(REPORTS / "personality_runtime_audit.json"),
        "belief_update_governance": read_json(REPORTS / "belief_update_governance.json"),
        "synthetic_benchmark_factory": read_json(REPORTS / "synthetic_benchmark_factory.json"),
        "multi_stream_trace_factory": read_json(REPORTS / "multi_stream_trace_factory.json"),
        "multi_stream_code_pressure": read_latest_json(REPORTS, "multi_stream_code_pressure_*_seed*.json"),
        "multi_stream_monitorability_probe": read_json(REPORTS / "multi_stream_monitorability_probe.json"),
        "multi_stream_candidate_gate": read_json(REPORTS / "multi_stream_candidate_gate.json"),
        "code_residual_forge": read_json(REPORTS / "code_residual_forge.json"),
        "real_code_benchmark_graduation": read_json(REPORTS / "real_code_benchmark_graduation.json"),
        "code_frontier_rotation": read_json(REPORTS / "code_frontier_rotation.json"),
        "code_transfer_artifacts": read_json(REPORTS / "code_transfer_artifacts.json"),
        "code_repair_organism": read_latest_json(REPORTS, "local_code_repair_organism_*_seed*.json"),
        "self_edit_lane": read_json(REPORTS / "self_edit_experiment_lane.json"),
        "long_horizon_memory": read_json(REPORTS / "long_horizon_memory_probe.json"),
        "virtual_context_memory": read_json(REPORTS / "virtual_context_memory_probe.json"),
        "virtual_context_compiled_context": read_json(REPORTS / "virtual_context_compiled_context.json"),
        "virtual_context_memory_graph": read_json(REPORTS / "virtual_context_memory_graph.json"),
        "virtual_context_memory_training_admission": read_json(REPORTS / "virtual_context_memory_training_admission.json"),
        "virtual_context_memory_status": read_json(REPORTS / "virtual_context_memory_status.json"),
        "vcm_context_recovery_benchmark": read_json(REPORTS / "vcm_context_recovery_benchmark.json"),
        "training_budget": read_json(REPORTS / "training_budget_plan.json"),
        "profile_run": read_json(REPORTS / "training_ratchet_profile_run.json"),
        "autonomy_last": read_json(REPORTS / "autonomy_cycle_last.json"),
        "autonomy_correction": read_json(REPORTS / "autonomy_cycle_watchdog_correction.json"),
        "daemon_events": read_jsonl_tail(REPORTS / "sparkstream_daemon_ledger.jsonl", 80),
        "autonomy_events": read_jsonl_tail(REPORTS / "autonomy_ledger.jsonl", 80),
        "teacher_calls": read_jsonl_tail(REPORTS / "teacher_calls.jsonl", 200),
    }


def assess(policy: dict[str, Any], cfg: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    now_ts = time.time()
    pressure = state.get("frontier_policy", {}).get("frontier_pressure") or {}
    candidate = state.get("candidate") or {}
    maturity_integrity = state.get("maturity_integrity") if isinstance(state.get("maturity_integrity"), dict) else {}
    last_cycle = latest_cycle_report(state.get("autonomy_last") or {}, state.get("autonomy_correction") or {})
    teacher_last = state.get("teacher_last") or {}
    bottleneck = state.get("candidate_bottleneck") or {}
    profile_run = state.get("profile_run") or {}
    work_board = state.get("hive_work_board_executor") if isinstance(state.get("hive_work_board_executor"), dict) else {}
    high_transfer_scheduler = (
        state.get("high_transfer_curriculum_scheduler")
        if isinstance(state.get("high_transfer_curriculum_scheduler"), dict)
        else {}
    )
    board_selected = selected_work_board_task(work_board)
    board_selected_id = str(board_selected.get("task_id") or board_selected.get("id") or "")
    board_selected_source = str(board_selected.get("source") or "")
    board_selected_concept = task_concept(board_selected)
    board_ready_tasks = int(get_path(work_board, ["summary", "ready_tasks"], 0) or 0)
    board_rotation_current = bool(
        work_board.get("trigger_state") in {"GREEN", "YELLOW"}
        and board_selected_id
        and board_selected_concept
    )
    scheduler_ready_tasks = int(get_path(high_transfer_scheduler, ["summary", "ready_task_count"], 0) or 0)
    scheduler_critical_tasks = int(get_path(high_transfer_scheduler, ["summary", "critical_task_count"], 0) or 0)

    max_stale = int(cfg.get("max_daemon_stale_seconds", 900))
    train_once_completed_max_stale = int(
        cfg.get("train_once_completed_max_stale_seconds", max(6 * 60 * 60, max_stale))
    )
    status_age = age_seconds(state.get("sparkstream", {}).get("updated_utc"), now_ts)
    add_check(
        checks,
        "daemon_status_fresh",
        status_age <= max_stale,
        "RED",
        f"status_age_seconds={status_age} max={max_stale}",
        "restart_daemon",
    )
    if status_age > max_stale:
        actions.append(action("restart_daemon", "SparkStream status is stale."))

    failed_streak = trailing_daemon_failures(state.get("daemon_events") or [])
    recent_failures = recent_daemon_failures(state.get("daemon_events") or [], minutes=int(cfg.get("failed_cycle_window_minutes", 60)))
    max_failures = int(cfg.get("max_failed_cycles", 2))
    latest_terminal = latest_daemon_terminal_event(state.get("daemon_events") or [])
    latest_terminal_event = str(latest_terminal.get("event") or "")
    sparkstream_phase = str(state.get("sparkstream", {}).get("phase") or "")
    active_cycle_running = bool(
        status_age <= max_stale
        and state.get("sparkstream", {}).get("ok") is None
        and sparkstream_phase
        and sparkstream_phase not in {"idle", "cycle_failed", "cycle_complete"}
    )
    live_repeated_failure = bool(
        failed_streak >= max_failures
        or (
            latest_terminal_event == "cycle_failed"
            and recent_failures >= max_failures
            and not active_cycle_running
        )
    )
    add_check(
        checks,
        "daemon_failure_streak_bounded",
        not live_repeated_failure,
        "RED",
        (
            f"failed_streak={failed_streak} recent_failures={recent_failures} active_cycle_running={active_cycle_running} "
            f"latest_terminal={latest_terminal_event or 'none'} max_before_red={max_failures}"
        ),
        "restart_daemon_then_smoke",
    )
    if live_repeated_failure:
        actions.append(action("restart_daemon_then_smoke", "Repeated daemon cycle failures."))

    maturity_summary = maturity_integrity.get("summary") if isinstance(maturity_integrity.get("summary"), dict) else {}
    maturity_state = str(maturity_integrity.get("trigger_state") or "missing")
    maturity_hard_value = get_path(maturity_summary, ["hard_blocker_count"], None)
    maturity_hard = 1 if maturity_hard_value is None else int(maturity_hard_value)
    add_check(
        checks,
        "maturity_integrity_audit_no_hard_blockers",
        bool(maturity_integrity) and maturity_state in {"GREEN", "YELLOW"} and maturity_hard == 0,
        "RED",
        (
            f"state={maturity_state} hard_blockers={maturity_summary.get('hard_blocker_count')} "
            f"maturity_blockers={maturity_summary.get('maturity_blocker_count')} "
            f"public_calibration_allowed={maturity_summary.get('public_calibration_allowed')} "
            f"candidate_promotion_allowed={maturity_summary.get('candidate_promotion_allowed')}"
        ),
        "run_maturity_integrity_audit",
    )
    if not maturity_integrity or maturity_state == "RED" or maturity_hard > 0:
        actions.append(action("run_maturity_integrity_audit", "Maturity/integrity audit is missing or has hard blockers."))

    profile_failure = profile_run.get("failure") if isinstance(profile_run.get("failure"), dict) else {}
    profile_timed_out = bool(profile_failure.get("timed_out"))
    add_check(
        checks,
        "profile_timeouts_become_recoverable_evidence",
        not profile_timed_out,
        "RED",
        (
            f"failed_step={profile_failure.get('name') or 'none'} "
            f"timeout_seconds={profile_failure.get('timeout_seconds')} "
            f"runtime_ms={profile_failure.get('runtime_ms')}"
        ),
        "profile_timeout_recovery",
    )
    if profile_timed_out:
        actions.append(action("profile_timeout_recovery", "A profile step timed out and needs a bounded recovery cycle."))

    add_check(
        checks,
        "hive_work_board_rotation_source_current",
        board_rotation_current,
        "YELLOW",
        (
            f"board_state={work_board.get('trigger_state') or 'missing'} "
            f"selected_task_id={board_selected_id or 'none'} "
            f"selected_source={board_selected_source or 'none'} "
            f"selected_concept={board_selected_concept or 'none'} "
            f"ready_tasks={board_ready_tasks} scheduler_ready={scheduler_ready_tasks} "
            f"scheduler_critical={scheduler_critical_tasks}"
        ),
        "run_hive_work_board_status",
    )
    if not board_rotation_current:
        actions.append(action("run_hive_work_board_status", "Hive work board report is missing a selected rotation task."))

    selected_non_promotable_reason = non_promotable_diagnostic_reason(board_selected_concept)
    add_check(
        checks,
        "hive_selected_task_promotion_safe",
        not selected_non_promotable_reason,
        "YELLOW",
        (
            f"selected_concept={board_selected_concept or 'none'} "
            f"reason={selected_non_promotable_reason or 'promotion_safe_or_none'}"
        ),
        "run_hive_work_board_status",
    )
    if selected_non_promotable_reason:
        actions.append(
            action(
                "run_hive_work_board_status",
                "Hive selected a diagnostic-only/non-promotable concept; refresh board selection before running frontier work.",
            )
        )

    code_preflight = code_contract_preflight_state(state.get("code_contract_preflight") or {})
    code_lm_private = state.get("code_lm_private_pressure") if isinstance(state.get("code_lm_private_pressure"), dict) else {}
    code_lm_private_rust = (
        state.get("code_lm_private_pressure_rust")
        if isinstance(state.get("code_lm_private_pressure_rust"), dict)
        else {}
    )
    code_lm_heartbeat = (
        state.get("code_lm_private_pressure_rust_heartbeat")
        if isinstance(state.get("code_lm_private_pressure_rust_heartbeat"), dict)
        else {}
    )
    code_lm_heartbeat_path = REPORTS / "code_lm_closure_rust_private_pressure_private.heartbeat.json"
    code_lm_heartbeat_age = file_age_seconds(code_lm_heartbeat_path, now_ts)
    latest_code_lm_heartbeat = (
        state.get("code_lm_latest_rust_heartbeat")
        if isinstance(state.get("code_lm_latest_rust_heartbeat"), dict)
        else {}
    )
    latest_heartbeat_path_text = str(latest_code_lm_heartbeat.get("__path") or "")
    latest_heartbeat_path = resolve_path(Path(latest_heartbeat_path_text)) if latest_heartbeat_path_text else code_lm_heartbeat_path
    latest_heartbeat_age = file_age_seconds(latest_heartbeat_path, now_ts)
    if latest_code_lm_heartbeat and (not code_lm_heartbeat or latest_heartbeat_age <= code_lm_heartbeat_age):
        code_lm_heartbeat = latest_code_lm_heartbeat
        code_lm_heartbeat_path = latest_heartbeat_path
        code_lm_heartbeat_age = latest_heartbeat_age
    add_check(
        checks,
        "code_decoder_contract_preflight_current",
        bool(code_preflight.get("ok")),
        "RED",
        (
            f"reason={code_preflight.get('reason')} "
            f"report_mtime={code_preflight.get('report_mtime')} "
            f"source_mtime={code_preflight.get('source_mtime')} "
            f"varargs={code_preflight.get('varargs_task_count')} "
            f"weak_required={code_preflight.get('weak_required_construct_count')} "
            f"weak_full_body={code_preflight.get('weak_full_body_count')} "
            f"hard_blockers={code_preflight.get('hard_blockers')}"
        ),
        "run_code_contract_preflight",
    )
    if not code_preflight.get("ok"):
        actions.append(action("run_code_contract_preflight", str(code_preflight.get("reason") or "code contract preflight is stale or failed.")))
    rust_candidate_stage = "candidate_generation" in str(code_lm_private_rust.get("progress_stage") or "")
    rust_declares_heartbeat = bool(
        code_lm_private_rust.get("candidate_generation_heartbeat")
        or get_path(code_lm_private_rust, ["summary", "candidate_generation_heartbeat"], None)
    )
    if code_lm_private_rust.get("run_status") == "in_progress" and rust_candidate_stage and rust_declares_heartbeat:
        add_check(
            checks,
            "code_lm_candidate_generation_heartbeat_fresh",
            bool(code_lm_heartbeat) and code_lm_heartbeat_age <= int(cfg.get("code_lm_heartbeat_stale_seconds", 900)),
            "YELLOW",
            (
                f"heartbeat_age_seconds={code_lm_heartbeat_age} "
                f"stage={code_lm_heartbeat.get('stage') or 'missing'} "
                f"phase={code_lm_heartbeat.get('phase') or 'missing'} "
                f"progress_ratio={get_path(code_lm_heartbeat, ['progress', 'progress_ratio'], None)}"
            ),
            "inspect_code_lm_candidate_generation",
        )
    shard_strategy = (
        state.get("code_lm_shard_strategy_audit")
        if isinstance(state.get("code_lm_shard_strategy_audit"), dict)
        else {}
    )
    shard_strategy_age = age_seconds(shard_strategy.get("created_utc"), now_ts)
    shard_strategy_summary = shard_strategy.get("summary") if isinstance(shard_strategy.get("summary"), dict) else {}
    shard_strategy_ok = bool(
        shard_strategy.get("policy") == "project_theseus_code_lm_shard_strategy_audit_v1"
        and shard_strategy.get("trigger_state") in {"GREEN", "YELLOW"}
        and shard_strategy_age <= max_stale
    )
    duplicate_active_shards = shard_strategy_summary.get("duplicate_active_shards") or {}
    add_check(
        checks,
        "code_lm_shard_strategy_audit_current",
        shard_strategy_ok and not duplicate_active_shards,
        "YELLOW",
        (
            f"policy={shard_strategy.get('policy') or 'missing'} "
            f"trigger_state={shard_strategy.get('trigger_state') or 'missing'} "
            f"age_seconds={shard_strategy_age} "
            f"completed={shard_strategy_summary.get('completed_shards')} "
            f"artifact_mb={shard_strategy_summary.get('current_artifact_mb')} "
            f"projected_mb={shard_strategy_summary.get('projected_artifact_mb')} "
            f"repeats_training={shard_strategy_summary.get('repeated_training_per_shard_detected')} "
            f"duplicate_active_shards={duplicate_active_shards}"
        ),
        "refresh_code_lm_shard_strategy_audit",
    )
    if not shard_strategy_ok:
        actions.append(action("refresh_code_lm_shard_strategy_audit", "Code LM shard strategy audit is missing or stale."))
    raw_train_once_fanout = (
        state.get("code_lm_train_once_fanout")
        if isinstance(state.get("code_lm_train_once_fanout"), dict)
        else {}
    )
    train_once_closure = (
        state.get("code_lm_train_once_fanout_closure")
        if isinstance(state.get("code_lm_train_once_fanout_closure"), dict)
        else {}
    )
    train_once_closure_summary = (
        train_once_closure.get("summary") if isinstance(train_once_closure.get("summary"), dict) else {}
    )
    train_once_closure_age = age_seconds(train_once_closure.get("created_utc"), now_ts)
    train_once_closure_ok = bool(
        train_once_closure.get("policy") == "project_theseus_code_lm_closure_train_once_fanout_v1"
        and train_once_closure.get("trigger_state") in {"GREEN", "YELLOW"}
        and train_once_closure.get("run_status") == "completed"
        and train_once_closure_age <= train_once_completed_max_stale
        and train_once_closure_summary.get("train_once_checkpoint_fanout") is True
        and train_once_closure_summary.get("repeated_training_per_candidate_shard") is False
    )
    raw_train_once_run_status = str(raw_train_once_fanout.get("run_status") or "")
    raw_train_once_is_stale_plan = raw_train_once_run_status in {
        "stale_artifacts_need_fanout_refresh",
        "deferred",
    }
    train_once_closure_report_path = str(
        train_once_closure.get("__closure_report_path")
        or TRAIN_ONCE_FANOUT_CLOSURE_REPORT.relative_to(ROOT)
    ).replace("\\", "/")
    if train_once_closure_ok and raw_train_once_is_stale_plan:
        train_once_fanout = {
            "policy": "project_theseus_code_lm_train_once_fanout_v1",
            "created_utc": train_once_closure.get("created_utc"),
            "trigger_state": train_once_closure.get("trigger_state"),
            "run_status": train_once_closure.get("run_status"),
            "current_phase": train_once_closure.get("progress_stage") or "completed_from_closure_report",
            "closure_report": train_once_closure_report_path,
            "summary": train_once_closure_summary,
            "architecture": {
                "repeated_training_per_candidate_shard": train_once_closure_summary.get(
                    "repeated_training_per_candidate_shard"
                )
            },
            "evidence_source": "closure_report_fallback",
        }
    else:
        train_once_fanout = raw_train_once_fanout
    train_once_age = age_seconds(train_once_fanout.get("created_utc"), now_ts)
    train_once_summary = train_once_fanout.get("summary") if isinstance(train_once_fanout.get("summary"), dict) else {}
    train_once_run_status = str(train_once_fanout.get("run_status") or "")
    train_once_active_heartbeat_path = str(train_once_fanout.get("active_phase_heartbeat") or "")
    train_once_active_heartbeat = (
        read_json(resolve_path(Path(train_once_active_heartbeat_path)))
        if train_once_active_heartbeat_path
        else {}
    )
    train_once_active_heartbeat_age = (
        file_age_seconds(resolve_path(Path(train_once_active_heartbeat_path)), now_ts)
        if train_once_active_heartbeat_path
        else 10**9
    )
    train_once_active_heartbeat_fresh = bool(
        train_once_active_heartbeat_path
        and train_once_active_heartbeat_age <= max_stale
        and train_once_active_heartbeat.get("status") == "running"
    )
    train_once_current_status = train_once_run_status in {
        "completed",
        "active_worker_discovered",
        "running",
        "current_source_fanout_smoke_completed",
    }
    train_once_ok = bool(
        train_once_fanout.get("policy") == "project_theseus_code_lm_train_once_fanout_v1"
        and train_once_fanout.get("trigger_state") in {"PLANNED", "RUNNING", "GREEN", "YELLOW"}
        and train_once_current_status
        and (
            train_once_active_heartbeat_fresh
            or (train_once_run_status == "completed" and train_once_age <= train_once_completed_max_stale)
            or train_once_age <= max_stale
        )
    )
    add_check(
        checks,
        "code_lm_train_once_fanout_path_current",
        train_once_ok
        and get_path(train_once_fanout, ["architecture", "repeated_training_per_candidate_shard"], False) is False,
        "YELLOW",
        (
            f"policy={train_once_fanout.get('policy') or 'missing'} "
            f"trigger_state={train_once_fanout.get('trigger_state') or 'missing'} "
            f"run_status={train_once_fanout.get('run_status') or 'missing'} "
            f"age_seconds={train_once_age} "
            f"completed_max_age_seconds={train_once_completed_max_stale} "
            f"active_heartbeat_age_seconds={train_once_active_heartbeat_age} "
            f"active_heartbeat_status={train_once_active_heartbeat.get('status') or None} "
            f"evidence_source={train_once_fanout.get('evidence_source') or 'wrapper_report'} "
            f"closure_age_seconds={train_once_closure_age} "
            f"repeated_training={train_once_summary.get('repeated_training_per_candidate_shard', get_path(train_once_fanout, ['architecture', 'repeated_training_per_candidate_shard'], None))} "
            f"closure_report={train_once_fanout.get('closure_report') or ''}"
        ),
        "refresh_code_lm_train_once_fanout_plan",
    )
    if not train_once_ok:
        actions.append(action("refresh_code_lm_train_once_fanout_plan", "Code LM train-once fanout plan/status is missing or stale."))

    system_efficiency = state.get("system_efficiency_audit") if isinstance(state.get("system_efficiency_audit"), dict) else {}
    system_efficiency_age = age_seconds(system_efficiency.get("created_utc"), now_ts)
    system_efficiency_summary = system_efficiency.get("summary") if isinstance(system_efficiency.get("summary"), dict) else {}
    system_efficiency_ok = bool(
        system_efficiency.get("policy") == "project_theseus_system_efficiency_audit_v1"
        and system_efficiency.get("trigger_state") in {"GREEN", "YELLOW", "RED"}
        and system_efficiency_age <= max_stale
    )
    add_check(
        checks,
        "system_efficiency_audit_current",
        system_efficiency_ok,
        "YELLOW",
        (
            f"policy={system_efficiency.get('policy') or 'missing'} "
            f"trigger_state={system_efficiency.get('trigger_state') or 'missing'} "
            f"age_seconds={system_efficiency_age} "
            f"finding_count={system_efficiency_summary.get('finding_count')} "
            f"top_bottleneck={system_efficiency_summary.get('top_loop_bottleneck')} "
            f"maintainability_score={system_efficiency_summary.get('maintainability_score')} "
            f"attd_score={system_efficiency_summary.get('attd_score')} "
            f"attd_top_component={system_efficiency_summary.get('attd_top_component')} "
            f"cleanup_queue={system_efficiency_summary.get('architecture_cleanup_queue_count')} "
            f"top_cleanup={system_efficiency_summary.get('top_architecture_cleanup_item')} "
            f"active_code_lm={system_efficiency_summary.get('active_code_lm_process_count')} "
            f"duplicate_services={system_efficiency_summary.get('duplicate_service_count')}"
        ),
        "run_system_efficiency_audit",
    )
    if not system_efficiency_ok:
        actions.append(action("run_system_efficiency_audit", "System efficiency audit is missing or stale."))

    service_hygiene = state.get("service_process_hygiene") if isinstance(state.get("service_process_hygiene"), dict) else {}
    service_hygiene_age = age_seconds(service_hygiene.get("created_utc"), now_ts)
    service_hygiene_ok = bool(
        service_hygiene.get("policy") == "project_theseus_service_process_hygiene_v1"
        and service_hygiene.get("trigger_state") in {"GREEN", "YELLOW"}
        and service_hygiene_age <= max_stale
    )
    duplicate_services = int(get_path(service_hygiene, ["summary", "duplicate_service_count"], 0) or 0)
    missing_services = int(get_path(service_hygiene, ["summary", "missing_required_service_count"], 0) or 0)
    add_check(
        checks,
        "service_process_hygiene_current",
        service_hygiene_ok and duplicate_services == 0,
        "YELLOW",
        (
            f"policy={service_hygiene.get('policy') or 'missing'} "
            f"trigger_state={service_hygiene.get('trigger_state') or 'missing'} "
            f"age_seconds={service_hygiene_age} duplicates={duplicate_services} missing_required={missing_services}"
        ),
        "run_service_process_hygiene",
    )
    if not service_hygiene_ok:
        actions.append(action("run_service_process_hygiene", "Service process hygiene report is missing or stale."))

    rotation_governor = state.get("autonomy_rotation_governor") if isinstance(state.get("autonomy_rotation_governor"), dict) else {}
    rotation_governor_age = age_seconds(rotation_governor.get("created_utc"), now_ts)
    rotation_governor_ok = bool(
        rotation_governor.get("policy") == "project_theseus_autonomy_rotation_governor_v2"
        and rotation_governor.get("trigger_state") in {"GREEN", "YELLOW"}
        and rotation_governor_age <= max_stale
        and int(get_path(rotation_governor, ["summary", "fresh_selected_targets_remaining"], 0) or 0) == 0
    )
    add_check(
        checks,
        "autonomy_rotation_governor_current",
        rotation_governor_ok,
        "YELLOW",
        (
            f"policy={rotation_governor.get('policy') or 'missing'} "
            f"trigger_state={rotation_governor.get('trigger_state') or 'missing'} "
            f"age_seconds={rotation_governor_age} "
            f"fresh_selected_targets_remaining={get_path(rotation_governor, ['summary', 'fresh_selected_targets_remaining'], None)}"
        ),
        "run_autonomy_rotation_governor",
    )
    if not rotation_governor_ok:
        actions.append(action("run_autonomy_rotation_governor", "Rotation governor is missing, stale, or still points at an already-satisfied target."))

    same_frontier_streak = trailing_same_frontier_streak(state.get("autonomy_events") or [])
    max_same = int(cfg.get("max_same_frontier_cycles", 3))
    latest_family = get_path(last_cycle, ["decision", "frontier_family"], "")
    active_wall = bool(pressure.get("active_frontier_wall"))
    attempt_count = int(pressure.get("active_frontier_attempt_count") or 0)
    interleave_after = int(
        pressure.get("wall_interleave_after_active_frontier_attempts")
        or get_path(policy, ["frontier_policy", "wall_interleave_after_active_frontier_attempts"], 12)
    )
    stuck_on_active = bool(
        active_wall
        and attempt_count >= interleave_after
        and latest_family == "babylm_mutated"
        and same_frontier_streak >= max_same
    )
    add_check(
        checks,
        "frontier_not_repeating_past_watchdog_limit",
        not stuck_on_active,
        "RED",
        (
            f"latest_family={latest_family} same_frontier_streak={same_frontier_streak} "
            f"active_wall={active_wall} attempts={attempt_count} interleave_after={interleave_after}"
        ),
        "force_rl_interleave",
    )
    if stuck_on_active:
        actions.append(action("force_rl_interleave", "Active frontier is repeating past the watchdog limit."))

    expected_family = str(state.get("frontier_policy", {}).get("frontier_family") or latest_family or "")
    expected_card = str(state.get("frontier_policy", {}).get("pressure_card_id") or "")
    needs_fresh_frontier = bool(pressure.get("needs_fresh_frontier"))
    direct_curriculum = state.get("benchmaxx_curriculum") if isinstance(state.get("benchmaxx_curriculum"), dict) else {}
    direct_next = direct_curriculum.get("next_frontier") if isinstance(direct_curriculum.get("next_frontier"), dict) else {}
    direct_runner = str(direct_next.get("runner_family") or "")
    direct_family = str(direct_next.get("family") or "")
    direct_card = str(direct_next.get("recommended_env") or "")
    direct_runner_map = {
        "minecraft_rl_local": "minecraft_rl",
        "drone_rl_local": "drone_rl",
        "coding_local_sandbox": "coding_local_sandbox",
        "web_agent_local": "web_agent_local",
        "transfer_eval_local": "transfer_eval",
    }
    direct_mapped_family = direct_runner_map.get(direct_runner, direct_family)
    direct_runnable = bool(direct_next.get("runnable_now")) and bool(direct_mapped_family)
    if direct_runnable:
        curriculum_family = direct_mapped_family
        curriculum_card = direct_card
        curriculum_runnable = True
        expected_family = curriculum_family or expected_family
        expected_card = curriculum_card or expected_card
    else:
        curriculum_family = str(get_path(state, ["frontier_policy", "frontier_pressure", "curriculum_next_frontier_family"], ""))
        curriculum_card = str(get_path(state, ["frontier_policy", "frontier_pressure", "next_pressure_card_id"], ""))
        curriculum_runnable = bool(get_path(state, ["frontier_policy", "frontier_pressure", "curriculum_runnable_now"], False))

    blocked_since_completed = teacher_budget_blocks_since_completed(state.get("teacher_calls") or [])
    max_blocks = int(cfg.get("max_teacher_budget_blocks_since_completion", 3))
    teacher_needed_recent = any(bool(row.get("teacher_needed")) for row in (state.get("autonomy_events") or [])[-3:])
    last_teacher_age = age_seconds(teacher_last.get("completed_utc") or teacher_last.get("created_utc"), now_ts)
    max_teacher_age = int(cfg.get("max_teacher_completed_age_seconds_when_needed", 1800))
    teacher_starved = bool(
        teacher_needed_recent
        and (teacher_last.get("status") != "completed" or last_teacher_age > max_teacher_age or blocked_since_completed > max_blocks)
    )
    add_check(
        checks,
        "teacher_escalation_not_starved",
        not teacher_starved,
        "RED",
        (
            f"teacher_needed_recent={teacher_needed_recent} last_status={teacher_last.get('status')} "
            f"last_teacher_age_seconds={last_teacher_age} blocked_since_completed={blocked_since_completed}"
        ),
        "force_teacher_call",
    )
    if teacher_starved and bool(cfg.get("force_teacher_call_on_starvation", False)):
        actions.append(action("force_teacher_call", "Teacher escalation is needed but not producing fresh guidance."))

    artifacts = candidate.get("artifacts") or {}
    active_frontier_family = str(artifacts.get("active_frontier_family") or "")
    active_frontier_path = str(artifacts.get("active_frontier") or "")
    active_frontier_seed = seed_from_text(str(artifacts.get("active_frontier") or ""))
    latest_mutated_seed = int(pressure.get("latest_mutated_babylm_seed") or -1)
    active_is_babylm = "babylm_mutated_holdout_seed" in active_frontier_path.replace("\\", "/")
    candidate_expected_family = expected_family or latest_family
    candidate_alignment_required = candidate_frontier_alignment_required(
        expected_family,
        expected_card,
        active_frontier_family,
    )
    family_mismatch = bool(
        candidate_alignment_required
        and
        candidate_expected_family
        and active_frontier_family
        and active_frontier_family != candidate_expected_family
        and candidate_expected_family != "babylm_mutated"
    )
    stale_frontier_gate = bool(
        candidate_alignment_required
        and
        active_is_babylm
        and latest_mutated_seed > 0
        and active_frontier_seed > 0
        and active_frontier_seed < latest_mutated_seed
    ) or (family_mismatch and not needs_fresh_frontier)
    add_check(
        checks,
        "candidate_gate_uses_current_frontier",
        not stale_frontier_gate,
        "RED",
        (
            f"active_frontier_source={artifacts.get('active_frontier_source')} "
            f"active_frontier_family={active_frontier_family} latest_family={latest_family} "
            f"active_frontier_seed={active_frontier_seed} latest_seed={latest_mutated_seed} "
            f"candidate_alignment_required={candidate_alignment_required}"
        ),
        "rerun_candidate_gate",
    )
    if stale_frontier_gate:
        actions.append(action("rerun_candidate_gate", "Candidate gate is using a stale frontier artifact."))

    curriculum_override_ok = bool(
        not curriculum_runnable
        or not curriculum_family
        or (
            expected_family == curriculum_family
            and (not curriculum_card or expected_card == curriculum_card)
        )
    )
    add_check(
        checks,
        "runnable_curriculum_overrides_stale_frontier",
        curriculum_override_ok,
        "RED",
        (
            f"curriculum={curriculum_family}/{curriculum_card or 'none'} runnable={curriculum_runnable} "
            f"selected={expected_family}/{expected_card or 'none'}"
        ),
        "force_curriculum_frontier_override",
    )
    if not curriculum_override_ok:
        actions.append(action("force_curriculum_frontier_override", "Runnable curriculum frontier is being ignored by the selected active frontier."))
    transfer_family = str(get_path(state, ["arm_transfer_plan", "summary", "frontier_family"], ""))
    transfer_card = str(get_path(state, ["arm_transfer_plan", "summary", "pressure_card_id"], ""))
    architecture_family = str(get_path(state, ["architecture_experiment", "state", "frontier_family"], ""))
    architecture_card = str(get_path(state, ["architecture_experiment", "state", "pressure_card_id"], ""))
    candidate_card_matches = bool(
        not candidate_alignment_required
        or
        needs_fresh_frontier
        or
        not expected_card
        or not active_frontier_path
        or expected_card in active_frontier_path
        or expected_card in str(get_path(candidate, ["artifacts", "active_frontier"], ""))
    )
    alignment_ok = active_frontier_alignment_ok(
        expected_family=expected_family,
        expected_card=expected_card,
        candidate_family=active_frontier_family if candidate_alignment_required else "",
        transfer_family=transfer_family,
        transfer_card=transfer_card,
        architecture_family=architecture_family,
        architecture_card=architecture_card,
    ) and candidate_card_matches
    add_check(
        checks,
        "governance_reports_share_active_frontier",
        alignment_ok,
        "RED",
        (
            f"expected={expected_family}/{expected_card or 'none'} "
            f"candidate={active_frontier_family}/{active_frontier_path or 'none'} "
            f"transfer={transfer_family}/{transfer_card or 'none'} "
            f"architecture={architecture_family}/{architecture_card or 'none'} "
            f"candidate_card_matches={candidate_card_matches} "
            f"candidate_alignment_required={candidate_alignment_required}"
        ),
        "refresh_alignment_reports",
    )
    if not alignment_ok:
        actions.append(action("refresh_alignment_reports", "Transfer, architecture, or candidate reports drifted from frontier_policy_status."))

    target_suckers = get_path(state, ["arm_transfer_plan", "summary", "target_suckers"], [])
    if not isinstance(target_suckers, list):
        target_suckers = []
    sucker_ready = bool(get_path(state, ["arm_sucker_registry", "summary", "ready_for_transfer_routing"], False))
    sucker_alignment_ok = bool(
        expected_family not in {"minecraft_rl", "drone_rl"}
        or (sucker_ready and target_suckers)
    )
    add_check(
        checks,
        "active_rl_frontier_has_arm_sucker_route",
        sucker_alignment_ok,
        "RED",
        f"family={expected_family or 'none'} sucker_ready={sucker_ready} target_suckers={target_suckers}",
        "refresh_alignment_reports",
    )
    if not sucker_alignment_ok:
        actions.append(action("refresh_alignment_reports", "Active RL frontier is missing its arm-sucker transfer route."))

    budget = state.get("training_budget") if isinstance(state.get("training_budget"), dict) else {}
    budget_family = str(budget.get("frontier_family") or "")
    budget_card = str(budget.get("pressure_card_id") or "")
    budget_sufficient = bool(get_path(budget, ["summary", "sufficient"], False))
    budget_matches = bool(
        budget
        and (not expected_family or budget_family == expected_family)
        and (not expected_card or budget_card == expected_card)
    )
    budget_ok = bool(budget_matches and budget_sufficient)
    add_check(
        checks,
        "training_budget_sufficient_before_eval",
        budget_ok,
        "RED",
        (
            f"budget_family={budget_family or 'missing'} budget_card={budget_card or 'missing'} "
            f"expected={expected_family}/{expected_card or 'none'} "
            f"sufficient={budget_sufficient} "
            f"train_env_steps={get_path(budget, ['summary', 'train_env_steps_budget'], None)}"
        ),
        "refresh_training_budget_plan",
    )
    if not budget_ok:
        actions.append(action("refresh_training_budget_plan", "Training budget is missing, stale, or too small for the active frontier."))

    genesis = state.get("genesis_kernel") if isinstance(state.get("genesis_kernel"), dict) else {}
    genesis_age = age_seconds(genesis.get("created_utc"), now_ts)
    genesis_max_age = int(cfg.get("max_genesis_kernel_age_seconds", 7200))
    genesis_hard_gate_failures = [
        row.get("gate")
        for row in genesis.get("release_gates", [])
        if isinstance(row, dict) and row.get("severity") == "hard" and not row.get("passed")
    ]
    genesis_ok = bool(
        genesis.get("policy") == "project_theseus_genesis_kernel_report_v0"
        and genesis_age <= genesis_max_age
        and not genesis_hard_gate_failures
    )
    add_check(
        checks,
        "genesis_kernel_snapshot_fresh",
        genesis_ok,
        "YELLOW",
        (
            f"age_seconds={genesis_age} max={genesis_max_age} "
            f"trigger_state={get_path(genesis, ['summary', 'trigger_state'], 'missing')} "
            f"failed_hard_gates={genesis_hard_gate_failures}"
        ),
        "refresh_genesis_kernel",
    )
    if not genesis_ok:
        actions.append(action("refresh_genesis_kernel", "Genesis artifact substrate is missing, stale, or has hard release-gate failures."))

    reality = state.get("reality_manipulator") if isinstance(state.get("reality_manipulator"), dict) else {}
    reality_age = age_seconds(reality.get("created_utc"), now_ts)
    reality_max_age = int(cfg.get("max_reality_manipulator_age_seconds", 7200))
    reality_hard_gate_failures = [
        row.get("gate")
        for row in reality.get("gates", [])
        if isinstance(row, dict) and row.get("severity") == "hard" and not row.get("passed")
    ]
    high_risk_approved = int(get_path(reality, ["safety_model", "high_risk_approved_without_gate_count"], 999) or 0)
    reality_acceptance = reality.get("acceptance_scenario") if isinstance(reality.get("acceptance_scenario"), dict) else {}
    reality_ok = bool(
        reality.get("policy") == "project_theseus_reality_manipulator_mvp_v1"
        and reality_age <= reality_max_age
        and not reality_hard_gate_failures
        and high_risk_approved == 0
        and reality_acceptance.get("world_created") is True
        and reality_acceptance.get("release_manifest_ready") is True
    )
    add_check(
        checks,
        "reality_manipulator_world_kernel_fresh",
        reality_ok,
        "YELLOW",
        (
            f"age_seconds={reality_age} max={reality_max_age} "
            f"trigger_state={reality.get('trigger_state') or 'missing'} "
            f"failed_hard_gates={reality_hard_gate_failures} "
            f"high_risk_approved_without_gate={high_risk_approved}"
        ),
        "refresh_reality_manipulator",
    )
    if not reality_ok:
        actions.append(action("refresh_reality_manipulator", "Reality Manipulator world/artifact/gate kernel is missing, stale, or unsafe."))

    viea_spine = state.get("viea_autonomy_spine") if isinstance(state.get("viea_autonomy_spine"), dict) else {}
    viea_kernel = state.get("viea_artifact_kernel") if isinstance(state.get("viea_artifact_kernel"), dict) else {}
    command_executor = state.get("viea_command_executor") if isinstance(state.get("viea_command_executor"), dict) else {}
    action_executor = state.get("viea_action_executor") if isinstance(state.get("viea_action_executor"), dict) else {}
    feedback_queue = state.get("feedback_action_queue") if isinstance(state.get("feedback_action_queue"), dict) else {}
    viea_spine_age = age_seconds(viea_spine.get("created_utc"), now_ts)
    viea_max_age = int(cfg.get("max_viea_autonomy_spine_age_seconds", 7200))
    viea_ok = bool(
        viea_spine.get("policy") == "project_theseus_viea_autonomy_spine_v1"
        and viea_spine_age <= viea_max_age
        and viea_spine.get("trigger_state") in {"GREEN", "YELLOW"}
        and get_path(viea_kernel, ["summary", "object_count"], 0)
        and command_executor.get("policy") == "project_theseus_viea_command_executor_v1"
        and int(get_path(feedback_queue, ["summary", "action_count"], 0) or 0) > 0
    )
    add_check(
        checks,
        "viea_autonomy_spine_controls_cycle",
        viea_ok,
        "YELLOW",
        (
            f"age_seconds={viea_spine_age} max={viea_max_age} "
            f"state={viea_spine.get('trigger_state') or 'missing'} "
            f"kernel_objects={get_path(viea_kernel, ['summary', 'object_count'], None)} "
            f"command_executor={command_executor.get('trigger_state') or 'missing'} "
            f"feedback_actions={get_path(feedback_queue, ['summary', 'action_count'], None)}"
        ),
        "refresh_viea_autonomy_spine",
    )
    if not viea_ok:
        actions.append(action("refresh_viea_autonomy_spine", "VIEA is missing from the autonomy control path or has no executable feedback actions."))
    action_executor_ok = bool(
        action_executor.get("policy") == "project_theseus_viea_action_executor_v1"
        and action_executor.get("trigger_state") in {"GREEN", "YELLOW"}
        and get_path(action_executor, ["summary", "queue_action_count"], 0) is not None
    )
    add_check(
        checks,
        "viea_action_executor_resume_ready",
        action_executor_ok,
        "YELLOW",
        (
            f"state={action_executor.get('trigger_state') or 'missing'} "
            f"queue_actions={get_path(action_executor, ['summary', 'queue_action_count'], None)} "
            f"ready={get_path(action_executor, ['summary', 'ready_action_count'], None)} "
            f"paused={get_path(action_executor, ['summary', 'paused'], None)}"
        ),
        "refresh_viea_autonomy_spine",
    )

    personality_runtime = state.get("personality_runtime_audit") if isinstance(state.get("personality_runtime_audit"), dict) else {}
    personality_core = state.get("personality_core") if isinstance(state.get("personality_core"), dict) else {}
    personality_context = state.get("personality_context") if isinstance(state.get("personality_context"), dict) else {}
    personality_drift = state.get("personality_drift") if isinstance(state.get("personality_drift"), dict) else {}
    belief_governance = state.get("belief_update_governance") if isinstance(state.get("belief_update_governance"), dict) else {}
    personality_age = age_seconds(personality_runtime.get("created_utc"), now_ts)
    personality_max_age = int(cfg.get("max_personality_runtime_audit_age_seconds", 7200))
    personality_ok = bool(
        personality_runtime.get("policy") == "sparkstream_personality_runtime_audit_v0"
        and personality_runtime.get("trigger_state") == "GREEN"
        and personality_age <= personality_max_age
        and personality_core.get("status") == "ready"
        and personality_context.get("status") == "ready"
        and personality_drift.get("passed") is True
        and belief_governance.get("status") in {"ready", "evaluated"}
    )
    add_check(
        checks,
        "personality_core_is_live_runtime_substrate",
        personality_ok,
        "YELLOW",
        (
            f"audit_state={personality_runtime.get('trigger_state') or 'missing'} age_seconds={personality_age} "
            f"core={personality_core.get('status') or 'missing'} context={personality_context.get('status') or 'missing'} "
            f"drift_passed={personality_drift.get('passed')} belief_status={belief_governance.get('status') or 'missing'}"
        ),
        "refresh_personality_runtime_audit",
    )
    if not personality_ok:
        actions.append(action("refresh_personality_runtime_audit", "Personality core is not freshly proven as a live runtime substrate."))

    code_forge = state.get("code_residual_forge") if isinstance(state.get("code_residual_forge"), dict) else {}
    code_transfer = state.get("code_transfer_artifacts") if isinstance(state.get("code_transfer_artifacts"), dict) else {}
    coding_frontier_active = expected_family == "coding_local_sandbox" or active_frontier_family == "coding_local_sandbox"
    code_forge_card = str(get_path(code_forge, ["summary", "active_card_id"], "") or "")
    code_forge_selected = str(get_path(code_forge, ["summary", "selected_card_id"], "") or get_path(code_forge, ["rotation", "selected_card_id"], "") or "")
    code_forge_ok = bool(
        not coding_frontier_active
        or (
            code_forge.get("policy") == "project_theseus_code_residual_forge_report_v1"
            and code_forge.get("trigger_state") != "RED"
            and int(get_path(code_forge, ["summary", "cluster_count"], 0) or 0) > 0
            and int(get_path(code_forge, ["summary", "transfer_artifacts"], 0) or 0) > 0
            and (not expected_card or code_forge_card == expected_card or code_forge_selected == expected_card)
            and int(get_path(code_transfer, ["summary", "artifact_count"], 0) or 0) > 0
        )
    )
    add_check(
        checks,
        "code_frontier_exports_residual_transfer_artifacts",
        code_forge_ok,
        "RED",
        (
            f"coding_frontier_active={coding_frontier_active} expected_card={expected_card or 'none'} "
            f"forge_card={code_forge_card or 'missing'} forge_selected={code_forge_selected or 'missing'} "
            f"trigger_state={code_forge.get('trigger_state') or 'missing'} "
            f"clusters={get_path(code_forge, ['summary', 'cluster_count'], None)} "
            f"transfer_artifacts={get_path(code_forge, ['summary', 'transfer_artifacts'], None)} "
            f"transfer_index={get_path(code_transfer, ['summary', 'artifact_count'], None)}"
        ),
        "run_code_residual_forge",
    )
    if not code_forge_ok:
        actions.append(action("run_code_residual_forge", "Coding frontier is missing residual classifications or reusable transfer artifacts."))

    code_repair = active_code_repair_evidence(state, expected_family, expected_card)
    code_repair_card = str(code_repair.get("card_id") or "")
    code_repair_ok = bool(
        not coding_frontier_active
        or (
            code_repair.get("policy") == "project_theseus_local_code_repair_organism_v1"
            and (not expected_card or not code_repair_card or code_repair_card == expected_card)
            and bool(get_path(code_repair, ["summary", "transfer_loaded"], False))
            and bool(get_path(code_repair, ["summary", "transfer_altered_behavior"], False))
            and float(get_path(code_repair, ["summary", "pass_rate_delta"], 0.0) or 0.0) > 0.0
            and int(code_repair.get("external_inference_calls") or 0) == 0
        )
    )
    add_check(
        checks,
        "code_frontier_consumes_transfer_artifacts",
        code_repair_ok,
        "RED",
        (
            f"coding_frontier_active={coding_frontier_active} policy={code_repair.get('policy') or 'missing'} "
            f"card_id={code_repair.get('card_id') or 'missing'} source={code_repair.get('source') or 'missing'} "
            f"loaded={get_path(code_repair, ['summary', 'transfer_loaded'], None)} "
            f"altered={get_path(code_repair, ['summary', 'transfer_altered_behavior'], None)} "
            f"delta={get_path(code_repair, ['summary', 'pass_rate_delta'], None)}"
        ),
        "run_code_repair_organism",
    )
    if not code_repair_ok:
        actions.append(action("run_code_repair_organism", "Coding frontier transfer artifacts exist but have not been consumed by a local repair organism."))

    real_code = state.get("real_code_benchmark_graduation") if isinstance(state.get("real_code_benchmark_graduation"), dict) else {}
    real_code_summary = real_code.get("summary") if isinstance(real_code.get("summary"), dict) else {}
    real_code_ok = bool(
        not coding_frontier_active
        or (
            real_code.get("policy") == "project_theseus_real_code_benchmark_graduation_v1"
            and real_code.get("trigger_state") in {"GREEN", "YELLOW"}
            and real_code.get("candidate_source")
            in {
                "local_theseus_student_checkpoint",
                "student_learning_checkpoint_v1",
                "student_neural_checkpoint_v1",
                "student_token_generator_checkpoint_v1",
                "student_code_lm_checkpoint_v1",
            }
            and int(real_code_summary.get("public_task_count") or 0) >= 32
            and int(real_code_summary.get("total_case_count") or 0) >= 32
            and int(real_code_summary.get("student_candidate_count") or 0) > 0
            and bool(real_code_summary.get("student_candidate_provenance_valid"))
            and bool(real_code_summary.get("student_candidate_benchmark_integrity_valid"))
            and bool(real_code_summary.get("token_level_code_generation_learned"))
            and int(real_code_summary.get("benchmark_promotion_eligible_candidate_count") or 0) > 0
            and int(real_code_summary.get("template_like_candidate_count") or 0) == 0
            and int(real_code_summary.get("loop_closure_candidate_count") or 0) == 0
            and int(real_code_summary.get("task_level_regressions_vs_single_stream") or 0) == 0
            and real_code.get("public_benchmark_score_claim")
            in {
                "student_checkpoint_public_task_calibration_only",
                "student_learning_checkpoint_public_task_calibration_only",
                "student_neural_checkpoint_public_task_calibration_only",
                "student_token_generator_checkpoint_public_task_calibration_only",
                "student_code_lm_checkpoint_public_task_calibration_only",
            }
            and int(real_code.get("external_inference_calls") or 0) == 0
        )
    )
    add_check(
        checks,
        "real_code_benchmark_graduation_ready",
        real_code_ok,
        "RED",
        (
            f"coding_frontier_active={coding_frontier_active} policy={real_code.get('policy') or 'missing'} "
            f"trigger_state={real_code.get('trigger_state') or 'missing'} "
            f"public_tasks={real_code_summary.get('public_task_count')} "
            f"total_cases={real_code_summary.get('total_case_count')} "
            "min_public_tasks=32 "
            f"delta={real_code_summary.get('pass_rate_delta')} "
            f"regressions={real_code_summary.get('task_level_regressions_vs_single_stream')} "
            f"candidate_source={real_code.get('candidate_source') or 'missing'} "
            f"student_candidates={real_code_summary.get('student_candidate_count')} "
            f"student_provenance_valid={real_code_summary.get('student_candidate_provenance_valid')} "
            f"benchmark_integrity_valid={real_code_summary.get('student_candidate_benchmark_integrity_valid')} "
            f"token_level_code_generation_learned={real_code_summary.get('token_level_code_generation_learned')} "
            f"template_like={real_code_summary.get('template_like_candidate_count')} "
            f"loop_closure={real_code_summary.get('loop_closure_candidate_count')} "
            f"eligible={real_code_summary.get('benchmark_promotion_eligible_candidate_count')} "
            f"score_claim={real_code.get('public_benchmark_score_claim') or 'missing'}"
        ),
        "run_real_code_benchmark_graduation",
    )
    if not real_code_ok:
        actions.append(action("run_real_code_benchmark_graduation", "Coding frontier lacks token-level learned student code generation evidence for public-task graduation."))

    open_code_pantry = state.get("open_code_training_pantry") if isinstance(state.get("open_code_training_pantry"), dict) else {}
    open_code_pantry_ok = bool(
        not coding_frontier_active
        or (
            open_code_pantry.get("policy") == "project_theseus_open_code_training_pantry_v1"
            and open_code_pantry.get("trigger_state") in {"GREEN", "YELLOW"}
            and int(get_path(open_code_pantry, ["summary", "private_train_expression_count"], 0) or 0) > 0
            and bool(get_path(open_code_pantry, ["summary", "public_benchmark_solutions_included"], True)) is False
        )
    )
    add_check(
        checks,
        "open_code_training_pantry_ready",
        open_code_pantry_ok,
        "YELLOW",
        (
            f"coding_frontier_active={coding_frontier_active} policy={open_code_pantry.get('policy') or 'missing'} "
            f"trigger_state={open_code_pantry.get('trigger_state') or 'missing'} "
            f"train_expressions={get_path(open_code_pantry, ['summary', 'private_train_expression_count'], None)} "
            f"public_benchmark_solutions_included={get_path(open_code_pantry, ['summary', 'public_benchmark_solutions_included'], None)}"
        ),
        "refresh_open_code_training_pantry",
    )
    if not open_code_pantry_ok:
        actions.append(action("refresh_open_code_training_pantry", "Governed permissive open-code training pantry is missing or empty."))

    sts_learning = state.get("sts_learning") if isinstance(state.get("sts_learning"), dict) else {}
    sts_learning_ok = bool(
        not coding_frontier_active
        or (
            sts_learning.get("policy") == "project_theseus_sts_learning_forge_v1"
            and sts_learning.get("trigger_state") == "GREEN"
            and bool(get_path(sts_learning, ["summary", "sts_training_substrate_ready"], False))
            and bool(get_path(sts_learning, ["summary", "public_benchmark_solutions_included"], True)) is False
        )
    )
    add_check(
        checks,
        "sts_parallel_stream_learning_substrate_ready",
        sts_learning_ok,
        "YELLOW",
        (
            f"coding_frontier_active={coding_frontier_active} policy={sts_learning.get('policy') or 'missing'} "
            f"trigger_state={sts_learning.get('trigger_state') or 'missing'} "
            f"rows={get_path(sts_learning, ['summary', 'row_count'], None)} "
            f"native_parallel_token_generation_proven={get_path(sts_learning, ['summary', 'native_parallel_token_generation_proven'], None)}"
        ),
        "run_sts_learning_forge",
    )
    if not sts_learning_ok:
        actions.append(action("run_sts_learning_forge", "Parallel stream-token learning substrate is missing or stale."))

    sts_native = state.get("sts_native") if isinstance(state.get("sts_native"), dict) else {}
    sts_native_ok = bool(
        not coding_frontier_active
        or (
            sts_native.get("policy") == "project_theseus_sts_native_parallel_probe_v1"
            and sts_native.get("trigger_state") == "GREEN"
            and bool(get_path(sts_native, ["summary", "native_parallel_token_generation_proven"], False))
            and bool(get_path(sts_native, ["summary", "one_token_per_output_stream_per_step"], False))
            and bool(get_path(sts_native, ["summary", "public_benchmark_solutions_included"], True)) is False
        )
    )
    add_check(
        checks,
        "sts_native_parallel_generation_probe_ready",
        sts_native_ok,
        "YELLOW",
        (
            f"coding_frontier_active={coding_frontier_active} policy={sts_native.get('policy') or 'missing'} "
            f"trigger_state={sts_native.get('trigger_state') or 'missing'} "
            f"output_streams={get_path(sts_native, ['summary', 'output_stream_count'], None)} "
            f"eval_delta={get_path(sts_native, ['summary', 'eval_token_accuracy_delta'], None)} "
            f"native_parallel_token_generation_proven={get_path(sts_native, ['summary', 'native_parallel_token_generation_proven'], None)}"
        ),
        "run_sts_native_parallel_probe",
    )
    if not sts_native_ok:
        actions.append(action("run_sts_native_parallel_probe", "Native STS parallel-token decoder probe is missing or stale."))

    cognitive_context = state.get("cognitive_context_router") if isinstance(state.get("cognitive_context_router"), dict) else {}
    cognitive_age = age_seconds(cognitive_context.get("created_utc"), now_ts)
    cognitive_summary = cognitive_context.get("summary") if isinstance(cognitive_context.get("summary"), dict) else {}
    cognitive_ok = bool(
        cognitive_context.get("policy") == "project_theseus_cognitive_context_router_v1"
        and cognitive_context.get("trigger_state") == "GREEN"
        and cognitive_age <= max_stale
        and int(cognitive_summary.get("context_row_count") or 0) > 0
        and bool(cognitive_summary.get("visible_report_requires_review"))
        and str(cognitive_summary.get("raw_chain_of_thought_exposure") or "") == "forbidden"
        and bool(cognitive_summary.get("public_benchmark_solutions_included")) is False
        and bool(cognitive_summary.get("public_tests_included")) is False
    )
    add_check(
        checks,
        "cognitive_context_spaces_ready",
        cognitive_ok,
        "YELLOW",
        (
            f"policy={cognitive_context.get('policy') or 'missing'} trigger_state={cognitive_context.get('trigger_state') or 'missing'} "
            f"age_seconds={cognitive_age} rows={cognitive_summary.get('context_row_count')} "
            f"visible_review={cognitive_summary.get('visible_report_requires_review')} "
            f"raw_chain={cognitive_summary.get('raw_chain_of_thought_exposure')}"
        ),
        "run_cognitive_context_router",
    )
    if not cognitive_ok:
        actions.append(action("run_cognitive_context_router", "Cognitive context spaces are missing, stale, or not review-gated."))

    self_edit = state.get("self_edit_lane") if isinstance(state.get("self_edit_lane"), dict) else {}
    self_edit_ok = bool(
        self_edit.get("policy") == "project_theseus_self_edit_experiment_lane_v1"
        and self_edit.get("trigger_state") in {"GREEN", "YELLOW"}
        and int(self_edit.get("external_inference_calls") or 0) == 0
    )
    add_check(
        checks,
        "self_edit_experiment_lane_ready",
        self_edit_ok,
        "YELLOW",
        f"policy={self_edit.get('policy') or 'missing'} trigger_state={self_edit.get('trigger_state') or 'missing'}",
        "run_self_edit_experiment_lane",
    )
    if not self_edit_ok:
        actions.append(action("run_self_edit_experiment_lane", "Bounded residual-to-source-patch experiment lane is missing or stale."))

    memory = state.get("long_horizon_memory") if isinstance(state.get("long_horizon_memory"), dict) else {}
    memory_ok = bool(
        memory.get("policy") == "project_theseus_long_horizon_memory_probe_v1"
        and memory.get("trigger_state") == "GREEN"
        and float(get_path(memory, ["score", "overall"], 0.0) or 0.0) >= 0.90
        and int(memory.get("external_inference_calls") or 0) == 0
    )
    add_check(
        checks,
        "long_horizon_memory_probe_green",
        memory_ok,
        "YELLOW",
        (
            f"policy={memory.get('policy') or 'missing'} trigger_state={memory.get('trigger_state') or 'missing'} "
            f"overall={get_path(memory, ['score', 'overall'], None)}"
        ),
        "run_long_horizon_memory_probe",
    )
    if not memory_ok:
        actions.append(action("run_long_horizon_memory_probe", "Long-horizon memory recovery has not been proven from compact traces."))

    vcm = state.get("virtual_context_memory") if isinstance(state.get("virtual_context_memory"), dict) else {}
    vcm_summary = vcm.get("summary") if isinstance(vcm.get("summary"), dict) else {}
    vcm_ok = bool(
        vcm.get("policy") == "project_theseus_virtual_context_memory_probe_v1"
        and vcm.get("trigger_state") == "GREEN"
        and int(vcm_summary.get("event_count") or 0) > 0
        and int(vcm_summary.get("graph_edge_count") or 0) > 0
        and vcm_summary.get("vcm_bench_state") == "GREEN"
        and int(vcm.get("external_inference_calls") or vcm_summary.get("external_inference_calls") or 0) == 0
    )
    add_check(
        checks,
        "virtual_context_memory_v1_green",
        vcm_ok,
        "YELLOW",
        (
            f"policy={vcm.get('policy') or 'missing'} trigger_state={vcm.get('trigger_state') or 'missing'} "
            f"events={vcm_summary.get('event_count')} edges={vcm_summary.get('graph_edge_count')} bench={vcm_summary.get('vcm_bench_state')}"
        ),
        "refresh_virtual_context_memory",
    )
    if not vcm_ok:
        actions.append(action("refresh_virtual_context_memory", "Virtual Context Memory v1 is missing, stale, or not bench-green."))

    vcm_compiled = state.get("virtual_context_compiled_context") if isinstance(state.get("virtual_context_compiled_context"), dict) else {}
    vcm_graph = state.get("virtual_context_memory_graph") if isinstance(state.get("virtual_context_memory_graph"), dict) else {}
    vcm_training = state.get("virtual_context_memory_training_admission") if isinstance(state.get("virtual_context_memory_training_admission"), dict) else {}
    vcm_context_recovery = state.get("vcm_context_recovery_benchmark") if isinstance(state.get("vcm_context_recovery_benchmark"), dict) else {}
    vcm_faults = vcm_compiled.get("semantic_page_faults") if isinstance(vcm_compiled.get("semantic_page_faults"), list) else []
    vcm_fault_counts: dict[str, int] = {}
    for row in vcm_faults:
        if isinstance(row, dict):
            kind = str(row.get("fault_type") or "unknown")
            vcm_fault_counts[kind] = vcm_fault_counts.get(kind, 0) + 1
    vcm_faults_explicit = all(
        isinstance(row, dict) and bool(row.get("fault_type")) and bool(row.get("safe_behavior"))
        for row in vcm_faults
    )
    conflict_edges = [
        row
        for row in (vcm_graph.get("edges") if isinstance(vcm_graph.get("edges"), list) else [])
        if isinstance(row, dict) and row.get("type") in {"contradicts", "supersedes", "invalidates"}
    ]
    invalidated = get_path(vcm_graph, ["invalidation", "invalidated_addresses"], [])
    training_ok = bool(
        vcm_training.get("policy") == "project_theseus_vcm_training_admission_bridge_v1"
        and vcm_training.get("trigger_state") == "GREEN"
        and int(get_path(vcm_training, ["summary", "public_training_leaks"], 0) or 0) == 0
        and int(get_path(vcm_training, ["summary", "teacher_boundary_leaks"], 0) or 0) == 0
        and int(get_path(vcm_training, ["summary", "deletion_leaks"], 0) or 0) == 0
    )
    add_check(
        checks,
        "virtual_context_memory_faults_explicit",
        vcm_ok and vcm_faults_explicit,
        "YELLOW",
        f"fault_counts={vcm_fault_counts}",
        "refresh_virtual_context_memory",
    )
    add_check(
        checks,
        "virtual_context_memory_training_admission_green",
        training_ok,
        "YELLOW",
        (
            f"policy={vcm_training.get('policy') or 'missing'} trigger_state={vcm_training.get('trigger_state') or 'missing'} "
            f"summary={vcm_training.get('summary') or {}}"
        ),
        "refresh_virtual_context_memory",
    )
    add_check(
        checks,
        "virtual_context_memory_graph_closure_tracked",
        vcm_ok and isinstance(invalidated, list) and bool(vcm_graph.get("invalidation")),
        "YELLOW",
        f"conflict_edges={len(conflict_edges)} invalidated={len(invalidated) if isinstance(invalidated, list) else 'missing'}",
        "refresh_virtual_context_memory",
    )
    context_recovery_ok = bool(
        vcm_context_recovery.get("policy") == "project_theseus_vcm_context_recovery_benchmark_v1"
        and vcm_context_recovery.get("trigger_state") == "GREEN"
        and float(get_path(vcm_context_recovery, ["summary", "vcm_answer_accuracy"], 0.0) or 0.0)
        > float(get_path(vcm_context_recovery, ["summary", "best_baseline_answer_accuracy"], 0.0) or 0.0)
        and int(vcm_context_recovery.get("external_inference_calls") or 0) == 0
        and int(vcm_context_recovery.get("public_training_rows_written") or 0) == 0
        and int(vcm_context_recovery.get("fallback_return_count") or 0) == 0
    )
    add_check(
        checks,
        "virtual_context_memory_context_recovery_green",
        context_recovery_ok,
        "YELLOW",
        (
            f"policy={vcm_context_recovery.get('policy') or 'missing'} trigger_state={vcm_context_recovery.get('trigger_state') or 'missing'} "
            f"vcm_accuracy={get_path(vcm_context_recovery, ['summary', 'vcm_answer_accuracy'], 'missing')} "
            f"best_baseline={get_path(vcm_context_recovery, ['summary', 'best_baseline_answer_accuracy'], 'missing')}"
        ),
        "run_vcm_context_recovery_benchmark",
    )
    if not training_ok:
        actions.append(action("refresh_virtual_context_memory", "VCM training-admission bridge is not green; memory-backed training must stay blocked."))
    elif not vcm_faults_explicit:
        actions.append(action("refresh_virtual_context_memory", "VCM faults are missing explicit safe-behavior metadata."))
    if not context_recovery_ok:
        actions.append(action("run_vcm_context_recovery_benchmark", "VCM context recovery has not proven it beats flat baselines under public-calibration firewall."))

    learning_scoreboard = state.get("learning_scoreboard") if isinstance(state.get("learning_scoreboard"), dict) else {}
    scoreboard_age = age_seconds(learning_scoreboard.get("created_utc"), now_ts)
    learning_scoreboard_ok = bool(
        learning_scoreboard.get("policy") == "project_theseus_learning_scoreboard_v1"
        and learning_scoreboard.get("trigger_state") in {"GREEN", "YELLOW", "RED"}
        and scoreboard_age <= max_stale
    )
    add_check(
        checks,
        "learning_truth_scoreboard_fresh",
        learning_scoreboard_ok,
        "YELLOW",
        (
            f"policy={learning_scoreboard.get('policy') or 'missing'} "
            f"trigger_state={learning_scoreboard.get('trigger_state') or 'missing'} "
            f"age_seconds={scoreboard_age} max={max_stale}"
        ),
        "refresh_learning_scoreboard",
    )
    if not learning_scoreboard_ok:
        actions.append(action("refresh_learning_scoreboard", "Learning truth scoreboard is missing or stale."))

    a_plus_scorecard = state.get("a_plus_scorecard") if isinstance(state.get("a_plus_scorecard"), dict) else {}
    a_plus_age = age_seconds(a_plus_scorecard.get("created_utc"), now_ts)
    a_plus_ok = bool(
        a_plus_scorecard.get("policy") == "project_theseus_a_plus_operating_scorecard_v1"
        and a_plus_scorecard.get("trigger_state") in {"GREEN", "YELLOW"}
        and a_plus_age <= max_stale
    )
    add_check(
        checks,
        "a_plus_operating_scorecard_fresh",
        a_plus_ok,
        "YELLOW",
        (
            f"policy={a_plus_scorecard.get('policy') or 'missing'} "
            f"trigger_state={a_plus_scorecard.get('trigger_state') or 'missing'} "
            f"grade={get_path(a_plus_scorecard, ['summary', 'overall_grade'], None)} "
            f"age_seconds={a_plus_age} max={max_stale}"
        ),
        "refresh_a_plus_scorecard",
    )
    if not a_plus_ok:
        actions.append(action("refresh_a_plus_scorecard", "A+ operating scorecard is missing or stale."))

    broad_matrix = state.get("broad_transfer_matrix") if isinstance(state.get("broad_transfer_matrix"), dict) else {}
    broad_age = age_seconds(broad_matrix.get("created_utc"), now_ts)
    broad_summary = broad_matrix.get("summary") if isinstance(broad_matrix.get("summary"), dict) else {}
    broad_ok = bool(
        broad_matrix.get("policy") == "project_theseus_broad_transfer_matrix_v1"
        and broad_matrix.get("trigger_state") in {"GREEN", "YELLOW"}
        and broad_age <= max_stale
        and int(broad_summary.get("real_public_task_count") or 0) >= 32
    )
    add_check(
        checks,
        "broad_public_transfer_matrix_fresh",
        broad_ok,
        "YELLOW",
        (
            f"policy={broad_matrix.get('policy') or 'missing'} "
            f"trigger_state={broad_matrix.get('trigger_state') or 'missing'} "
            f"age_seconds={broad_age} max={max_stale} "
            f"tasks={broad_summary.get('real_public_task_count')} "
            f"pass_rate={broad_summary.get('real_public_pass_rate')} "
            f"below_floor={broad_summary.get('cards_below_floor')} "
            f"no_clean={broad_summary.get('no_clean_student_evidence_cards')}"
        ),
        "refresh_broad_transfer_matrix",
    )
    if not broad_ok:
        actions.append(action("refresh_broad_transfer_matrix", "Broad public-transfer matrix is missing, stale, or too small to anchor learning truth."))

    broad_scheduler = state.get("broad_code_calibration_scheduler") if isinstance(state.get("broad_code_calibration_scheduler"), dict) else {}
    broad_scheduler_age = age_seconds(broad_scheduler.get("created_utc"), now_ts)
    broad_selected = broad_scheduler.get("selected") if isinstance(broad_scheduler.get("selected"), dict) else {}
    broad_blockers = broad_transfer_blockers(broad_summary)
    broad_progressing_ok = bool(
        not coding_frontier_active
        or (
            broad_ok
            and not broad_blockers
        )
        or (
            broad_ok
            and broad_blockers
            and broad_scheduler.get("policy") == "project_theseus_broad_code_calibration_scheduler_v1"
            and broad_scheduler_age <= max_stale
            and str(broad_selected.get("action") or "") not in {"", "no_action"}
        )
    )
    add_check(
        checks,
        "broad_public_transfer_matrix_drives_next_pressure",
        broad_progressing_ok,
        "YELLOW",
        (
            f"coding_frontier_active={coding_frontier_active} broad_trigger={broad_matrix.get('trigger_state') or 'missing'} "
            f"blockers={broad_blockers} scheduler_age_seconds={broad_scheduler_age} "
            f"selected={broad_selected.get('card_id') or 'none'} action={broad_selected.get('action') or 'none'} "
            f"can_run_real_code={broad_selected.get('can_run_real_code')}"
        ),
        "run_broad_code_calibration_step",
    )
    if not broad_progressing_ok:
        actions.append(action("run_broad_code_calibration_step", "Broad public-transfer blockers need to select and run the next honest calibration target."))

    cell_lifecycle = state.get("cell_lifecycle") if isinstance(state.get("cell_lifecycle"), dict) else {}
    cell_lifecycle_age = age_seconds(cell_lifecycle.get("created_utc"), now_ts)
    cell_summary = cell_lifecycle.get("summary") if isinstance(cell_lifecycle.get("summary"), dict) else {}
    prune_summary = get_path(cell_lifecycle, ["training_data_prune_plan", "summary"], {})
    if not isinstance(prune_summary, dict):
        prune_summary = {}
    cell_lifecycle_ok = bool(
        cell_lifecycle.get("policy") == "project_theseus_cell_lifecycle_v1"
        and cell_lifecycle.get("trigger_state") in {"GREEN", "YELLOW"}
        and cell_lifecycle_age <= max_stale
        and bool(get_path(cell_lifecycle, ["training_data_prune_plan", "delete_performed"], False)) is False
        and int(prune_summary.get("unsafe_prune_requests") or 0) == 0
    )
    add_check(
        checks,
        "cell_lifecycle_pressure_current_and_non_destructive",
        cell_lifecycle_ok,
        "YELLOW",
        (
            f"policy={cell_lifecycle.get('policy') or 'missing'} "
            f"trigger_state={cell_lifecycle.get('trigger_state') or 'missing'} "
            f"age_seconds={cell_lifecycle_age} max={max_stale} "
            f"improve={cell_summary.get('improve_candidates')} "
            f"split={cell_summary.get('split_or_compress_candidates')} "
            f"retire={cell_summary.get('retire_candidates')} "
            f"archive_candidates={cell_summary.get('training_data_archive_candidates')} "
            f"delete_performed={get_path(cell_lifecycle, ['training_data_prune_plan', 'delete_performed'], None)} "
            f"unsafe_prune={prune_summary.get('unsafe_prune_requests')}"
        ),
        "refresh_cell_lifecycle",
    )
    if not cell_lifecycle_ok:
        actions.append(action("refresh_cell_lifecycle", "Cell lifecycle pressure is missing, stale, or attempting unsafe pruning."))

    overnight = state.get("overnight_readiness") if isinstance(state.get("overnight_readiness"), dict) else {}
    overnight_age = age_seconds(overnight.get("created_utc"), now_ts)
    overnight_ok = bool(
        overnight.get("policy") == "project_theseus_overnight_learning_readiness_v1"
        and overnight.get("trigger_state") in {"GREEN", "YELLOW"}
        and overnight_age <= max_stale
        and bool(overnight.get("overnight_launch_ready"))
    )
    add_check(
        checks,
        "overnight_learning_readiness_current",
        overnight_ok,
        "YELLOW",
        (
            f"policy={overnight.get('policy') or 'missing'} "
            f"trigger_state={overnight.get('trigger_state') or 'missing'} "
            f"launch_ready={overnight.get('overnight_launch_ready')} "
            f"age_seconds={overnight_age} max={max_stale}"
        ),
        "refresh_overnight_learning_readiness",
    )
    if not overnight_ok:
        actions.append(action("refresh_overnight_learning_readiness", "Overnight learning readiness report is missing, stale, or not launch-ready."))

    grammar_suckers = state.get("grammar_suckers") if isinstance(state.get("grammar_suckers"), dict) else {}
    grammar_age = age_seconds(grammar_suckers.get("created_utc"), now_ts)
    grammar_integrity = grammar_sucker_integrity(grammar_suckers)
    grammar_summary = grammar_integrity["summary"]
    grammar_report_ok = bool(
        grammar_suckers.get("policy") == "project_theseus_grammar_suckers_v0"
        and grammar_suckers.get("trigger_state") in {"GREEN", "YELLOW", "RED"}
        and grammar_age <= max_stale
    )
    add_check(
        checks,
        "grammar_suckers_fresh",
        grammar_report_ok,
        "YELLOW",
        (
            f"policy={grammar_suckers.get('policy') or 'missing'} "
            f"trigger_state={grammar_suckers.get('trigger_state') or 'missing'} "
            f"age_seconds={grammar_age} max={max_stale}"
        ),
        "run_grammar_suckers",
    )
    if not grammar_report_ok:
        actions.append(action("run_grammar_suckers", "Language rule substrate report is missing or stale."))
    grammar_invalid_promotion = int(grammar_integrity["invalid_promotion"])
    add_check(
        checks,
        "grammar_suckers_no_invalid_python_promotion_candidates",
        bool(grammar_integrity["integrity_ok"]),
        "RED",
        (
            f"invalid_promotion={grammar_invalid_promotion} "
            f"python_parse={grammar_summary.get('python_parse_pass_rate')} "
            f"english_surface={grammar_summary.get('english_surface_pass_rate')} "
            f"sbl_traces={grammar_summary.get('sbl_trace_count')}"
        ),
        "run_grammar_suckers",
    )
    if grammar_integrity["hard_action_needed"]:
        actions.append(action("run_grammar_suckers", "Grammar sucker found malformed Python candidates that must not be promotion evidence."))

    taming = state.get("deterministic_taming") if isinstance(state.get("deterministic_taming"), dict) else {}
    taming_age = age_seconds(taming.get("created_utc"), now_ts)
    taming_integrity = deterministic_taming_integrity(taming)
    taming_summary = taming_integrity["summary"]
    taming_ok = bool(taming_integrity["integrity_ok"] and taming_age <= max_stale)
    add_check(
        checks,
        "deterministic_taming_stack_ready",
        taming_ok,
        str(taming_integrity["stale_severity"]),
        (
            f"policy={taming.get('policy') or 'missing'} trigger_state={taming.get('trigger_state') or 'missing'} "
            f"age_seconds={taming_age} hard_failures={taming_summary.get('hard_failure_count')} "
            f"python_invalid={taming_summary.get('python_invalid_promotion_candidates')}"
        ),
        "run_deterministic_taming_stack",
    )
    if not taming_ok:
        if not taming_integrity["integrity_ok"]:
            actions.append(action("run_deterministic_taming_stack", "Deterministic taming stack has hard verifier failures."))
        else:
            actions.append(action("run_deterministic_taming_stack", "Deterministic taming stack is missing or stale."))

    residual_curriculum = state.get("code_residual_curriculum") if isinstance(state.get("code_residual_curriculum"), dict) else {}
    residual_curriculum_age = age_seconds(residual_curriculum.get("created_utc"), now_ts)
    residual_curriculum_rows = int(get_path(residual_curriculum, ["summary", "private_row_count"], 0) or 0)
    residual_curriculum_ok = bool(
        residual_curriculum.get("policy") == "project_theseus_code_residual_curriculum_v1"
        and residual_curriculum.get("trigger_state") in {"GREEN", "YELLOW"}
        and residual_curriculum_age <= max_stale
        and residual_curriculum_rows > 0
        and not bool(get_path(residual_curriculum, ["summary", "public_benchmark_solutions_included"], True))
        and not bool(get_path(residual_curriculum, ["summary", "public_tests_included"], True))
    )
    add_check(
        checks,
        "code_residual_curriculum_private_and_clean",
        residual_curriculum_ok,
        "YELLOW",
        (
            f"policy={residual_curriculum.get('policy') or 'missing'} rows={residual_curriculum_rows} "
            f"age_seconds={residual_curriculum_age} public_solutions={get_path(residual_curriculum, ['summary', 'public_benchmark_solutions_included'], None)} "
            f"public_tests={get_path(residual_curriculum, ['summary', 'public_tests_included'], None)}"
        ),
        "run_code_residual_curriculum",
    )
    active_code_lm_count = int(system_efficiency_summary.get("active_code_lm_process_count") or 0)
    train_once_active = str(train_once_fanout.get("trigger_state") or "") == "RUNNING" or str(
        train_once_fanout.get("run_status") or ""
    ) in {"running", "active_worker_discovered"}
    code_lm_training_input_mutation_deferred = bool(active_code_lm_count > 0 or train_once_active)
    if not residual_curriculum_ok:
        if code_lm_training_input_mutation_deferred:
            actions.append(
                action(
                    "defer_code_residual_curriculum_refresh",
                    (
                        "Residual-targeted private code curriculum is missing or stale, but Code LM work is active; "
                        "defer canonical private-row mutation until the checkpoint/fanout worker is idle."
                    ),
                )
            )
        else:
            actions.append(action("run_code_residual_curriculum", "Residual-targeted private code curriculum is missing or not clean."))

    sts_ablation = state.get("sts_repair_ablation") if isinstance(state.get("sts_repair_ablation"), dict) else {}
    sts_ablation_age = age_seconds(sts_ablation.get("created_utc"), now_ts)
    sts_delta = float(get_path(sts_ablation, ["summary", "pass_rate_delta"], 0.0) or 0.0)
    sts_regressions = int(get_path(sts_ablation, ["summary", "task_level_regressions"], 0) or 0)
    sts_ablation_ok = bool(
        sts_ablation.get("policy") == "project_theseus_sts_repair_ablation_v1"
        and sts_ablation.get("trigger_state") in {"GREEN", "YELLOW"}
        and sts_ablation_age <= max_stale
        and sts_delta > 0.0
        and sts_regressions == 0
    )
    add_check(
        checks,
        "sts_repair_ablation_positive",
        sts_ablation_ok,
        "YELLOW",
        (
            f"policy={sts_ablation.get('policy') or 'missing'} age_seconds={sts_ablation_age} "
            f"delta={sts_delta} regressions={sts_regressions}"
        ),
        "run_sts_repair_ablation",
    )
    if not sts_ablation_ok:
        actions.append(action("run_sts_repair_ablation", "STS repair ablation is missing/stale or not showing positive no-regression delta."))

    architecture_guidance = state.get("architecture_guidance") if isinstance(state.get("architecture_guidance"), dict) else {}
    architecture_guidance_age = age_seconds(architecture_guidance.get("created_utc"), now_ts)
    architecture_guidance_ok = bool(
        architecture_guidance.get("policy") == "project_theseus_architecture_guidance_loop_v1"
        and architecture_guidance.get("trigger_state") in {"GREEN", "YELLOW"}
        and architecture_guidance_age <= max_stale
        and len(architecture_guidance.get("experiments") if isinstance(architecture_guidance.get("experiments"), list) else []) > 0
    )
    add_check(
        checks,
        "architecture_guidance_loop_ready",
        architecture_guidance_ok,
        "YELLOW",
        (
            f"policy={architecture_guidance.get('policy') or 'missing'} trigger_state={architecture_guidance.get('trigger_state') or 'missing'} "
            f"age_seconds={architecture_guidance_age} wall={get_path(architecture_guidance, ['diagnosis', 'wall'], None)} "
            f"teacher={get_path(architecture_guidance, ['teacher', 'status'], None)}"
        ),
        "run_architecture_guidance_loop",
    )
    if not architecture_guidance_ok:
        actions.append(action("run_architecture_guidance_loop", "Architecture guidance loop is missing/stale or has no safe experiments."))

    teacher_budget_audit = state.get("teacher_budget_audit") if isinstance(state.get("teacher_budget_audit"), dict) else {}
    teacher_budget_age = age_seconds(teacher_budget_audit.get("created_utc"), now_ts)
    teacher_budget_ok = bool(
        teacher_budget_audit.get("policy") == "project_theseus_teacher_budget_audit_v1"
        and teacher_budget_audit.get("trigger_state") in {"GREEN", "YELLOW"}
        and teacher_budget_age <= max_stale
    )
    add_check(
        checks,
        "teacher_budget_audit_current",
        teacher_budget_ok,
        "YELLOW",
        (
            f"policy={teacher_budget_audit.get('policy') or 'missing'} "
            f"trigger_state={teacher_budget_audit.get('trigger_state') or 'missing'} "
            f"age_seconds={teacher_budget_age} architecture_allowed="
            f"{get_path(teacher_budget_audit, ['reason_decisions', 'architecture_wall', 'budget', 'allowed'], None)}"
        ),
        "refresh_teacher_budget_audit",
    )
    if not teacher_budget_ok:
        actions.append(action("refresh_teacher_budget_audit", "Teacher budget audit is missing or stale."))

    synthetic_benchmarks = state.get("synthetic_benchmark_factory") if isinstance(state.get("synthetic_benchmark_factory"), dict) else {}
    synthetic_benchmarks_ok = bool(
        synthetic_benchmarks.get("policy") == "project_theseus_synthetic_benchmark_factory_v1"
        and synthetic_benchmarks.get("trigger_state") == "GREEN"
        and int(get_path(synthetic_benchmarks, ["summary", "ready_cards"], 0) or 0) > 0
        and int(get_path(synthetic_benchmarks, ["summary", "case_count"], 0) or 0) > 0
        and int(synthetic_benchmarks.get("external_inference_calls") or 0) == 0
    )
    add_check(
        checks,
        "synthetic_benchmark_pressure_backstop_ready",
        synthetic_benchmarks_ok,
        "YELLOW",
        (
            f"policy={synthetic_benchmarks.get('policy') or 'missing'} "
            f"trigger_state={synthetic_benchmarks.get('trigger_state') or 'missing'} "
            f"cards={get_path(synthetic_benchmarks, ['summary', 'cards'], None)} "
            f"cases={get_path(synthetic_benchmarks, ['summary', 'case_count'], None)}"
        ),
        "run_synthetic_benchmark_factory",
    )
    if not synthetic_benchmarks_ok:
        actions.append(action("run_synthetic_benchmark_factory", "Synthetic benchmark pressure backstop is missing or stale."))

    multi_stream_factory = state.get("multi_stream_trace_factory") if isinstance(state.get("multi_stream_trace_factory"), dict) else {}
    multi_stream_pressure = state.get("multi_stream_code_pressure") if isinstance(state.get("multi_stream_code_pressure"), dict) else {}
    multi_stream_probe = state.get("multi_stream_monitorability_probe") if isinstance(state.get("multi_stream_monitorability_probe"), dict) else {}
    multi_stream_gate = state.get("multi_stream_candidate_gate") if isinstance(state.get("multi_stream_candidate_gate"), dict) else {}
    multi_stream_factory_ok = bool(
        multi_stream_factory.get("policy") == "project_theseus_multi_stream_trace_factory_v1"
        and multi_stream_factory.get("trigger_state") == "GREEN"
        and int(get_path(multi_stream_factory, ["summary", "ready_cards"], 0) or 0) > 0
        and int(get_path(multi_stream_factory, ["summary", "case_count"], 0) or 0) > 0
        and int(multi_stream_factory.get("external_inference_calls") or 0) == 0
    )
    add_check(
        checks,
        "multi_stream_trace_factory_ready",
        multi_stream_factory_ok,
        "YELLOW",
        (
            f"policy={multi_stream_factory.get('policy') or 'missing'} "
            f"trigger_state={multi_stream_factory.get('trigger_state') or 'missing'} "
            f"cases={get_path(multi_stream_factory, ['summary', 'case_count'], None)}"
        ),
        "run_multi_stream_trace_factory",
    )
    if not multi_stream_factory_ok:
        actions.append(action("run_multi_stream_trace_factory", "Multi-stream code pressure trace factory is missing or stale."))
    multi_stream_expected = str(expected_card or "") == "multistream_code_repair_pressure"
    multi_stream_pressure_ok = bool(
        (not multi_stream_expected)
        or (
            multi_stream_pressure.get("policy") == "project_theseus_multi_stream_code_pressure_v1"
            and get_path(multi_stream_pressure, ["verifier", "trigger_state"], "") == "GREEN"
            and float(get_path(multi_stream_pressure, ["summary", "apples_to_apples_overlap"], 0.0) or 0.0) >= 1.0
            and float(get_path(multi_stream_pressure, ["summary", "pass_rate_delta"], 0.0) or 0.0) > 0.0
            and int(get_path(multi_stream_pressure, ["summary", "task_level_improvements_over_single_stream"], 0) or 0) > 0
            and int(get_path(multi_stream_pressure, ["summary", "task_level_regressions_vs_single_stream"], 0) or 0) == 0
            and int(multi_stream_pressure.get("external_inference_calls") or 0) == 0
        )
    )
    add_check(
        checks,
        "multi_stream_code_pressure_apples_to_apples",
        multi_stream_pressure_ok,
        "YELLOW",
        (
            f"expected={multi_stream_expected} policy={multi_stream_pressure.get('policy') or 'missing'} "
            f"verifier={get_path(multi_stream_pressure, ['verifier', 'trigger_state'], None)} "
            f"overlap={get_path(multi_stream_pressure, ['summary', 'apples_to_apples_overlap'], None)} "
            f"delta={get_path(multi_stream_pressure, ['summary', 'pass_rate_delta'], None)} "
            f"improved={get_path(multi_stream_pressure, ['summary', 'task_level_improvements_over_single_stream'], None)} "
            f"regressed={get_path(multi_stream_pressure, ['summary', 'task_level_regressions_vs_single_stream'], None)} "
            f"score={multi_stream_pressure.get('score')}"
        ),
        "write_frontier_override",
    )
    multi_stream_probe_ok = bool(
        (not multi_stream_expected)
        or (
            multi_stream_probe.get("policy") == "project_theseus_multi_stream_monitorability_probe_v1"
            and multi_stream_probe.get("trigger_state") == "GREEN"
            and int(multi_stream_probe.get("external_inference_calls") or 0) == 0
        )
    )
    add_check(
        checks,
        "multi_stream_monitorability_probe_green",
        multi_stream_probe_ok,
        "YELLOW",
        (
            f"expected={multi_stream_expected} trigger_state={multi_stream_probe.get('trigger_state') or 'missing'} "
            f"score={get_path(multi_stream_probe, ['summary', 'monitorability_score'], None)}"
        ),
        "run_multi_stream_monitorability_probe",
    )
    if multi_stream_expected and not multi_stream_probe_ok:
        actions.append(action("run_multi_stream_monitorability_probe", "Multi-stream pressure is active but monitorability has not been probed."))
    multi_stream_gate_ok = bool(
        (not multi_stream_expected)
        or (
            multi_stream_gate.get("policy") == "project_theseus_multi_stream_candidate_gate_v1"
            and multi_stream_gate.get("trigger_state") == "GREEN"
            and int(multi_stream_gate.get("external_inference_calls") or 0) == 0
        )
    )
    add_check(
        checks,
        "multi_stream_candidate_gate_green",
        multi_stream_gate_ok,
        "YELLOW",
        (
            f"expected={multi_stream_expected} trigger_state={multi_stream_gate.get('trigger_state') or 'missing'} "
            f"promotion_allowed={get_path(multi_stream_gate, ['decision', 'promotion_allowed'], None)}"
        ),
        "run_multi_stream_candidate_gate",
    )
    if multi_stream_expected and not multi_stream_gate_ok:
        actions.append(action("run_multi_stream_candidate_gate", "Multi-stream pressure needs its private-pressure candidate gate report."))

    reducer_status = str(bottleneck.get("status") or "missing")
    reducer_flow_ready = bool(bottleneck.get("candidate_flow_ready"))
    remaining_safe_actions = bottleneck.get("remaining_safe_auto_actions")
    remaining_safe_count = len(remaining_safe_actions) if isinstance(remaining_safe_actions, list) else 0
    reducer_needs_fix = bool(
        reducer_status in {"missing", "RED", "YELLOW_FRONTIER_SETUP_REQUIRED", "YELLOW_RUNTIME_BLOCKERS"}
        or (not reducer_flow_ready and remaining_safe_count > 0)
    )
    add_check(
        checks,
        "candidate_bottlenecks_reduced_before_teacher",
        not reducer_needs_fix,
        "RED",
        (
            f"status={reducer_status} candidate_flow_ready={reducer_flow_ready} "
            f"remaining_safe_auto_actions={remaining_safe_count}"
        ),
        "run_candidate_bottleneck_reducer",
    )
    if reducer_needs_fix:
        actions.append(action("run_candidate_bottleneck_reducer", "Candidate flow has unresolved local setup bottlenecks."))

    state_name = "GREEN"
    if any(item["severity"] == "RED" and not item["passed"] for item in checks):
        state_name = "RED"
    elif any(not item["passed"] for item in checks):
        state_name = "YELLOW"

    return {
        "policy": "sparkstream_autonomy_watchdog_v0",
        "created_utc": now(),
        "trigger_state": state_name,
        "summary": {
            "status_age_seconds": status_age,
            "failed_cycle_streak": failed_streak,
            "failed_cycles_recent_window": recent_failures,
            "latest_daemon_terminal_event": latest_terminal_event or None,
            "same_frontier_streak": same_frontier_streak,
            "rotation_source_of_truth": "hive_work_board",
            "board_rotation_current": board_rotation_current,
            "board_trigger_state": work_board.get("trigger_state"),
            "board_selected_task_id": board_selected_id or None,
            "board_selected_source": board_selected_source or None,
            "board_selected_concept": board_selected_concept or None,
            "board_selected_command": board_selected.get("command") if board_selected else None,
            "board_ready_tasks": board_ready_tasks,
            "service_process_hygiene_trigger_state": service_hygiene.get("trigger_state"),
            "service_process_hygiene_age_seconds": service_hygiene_age,
            "service_process_duplicate_count": duplicate_services,
            "service_process_missing_required_count": missing_services,
            "system_efficiency_trigger_state": system_efficiency.get("trigger_state"),
            "system_efficiency_finding_count": system_efficiency_summary.get("finding_count"),
            "system_efficiency_loop_bottleneck_count": system_efficiency_summary.get("loop_bottleneck_count"),
            "system_efficiency_top_loop_bottleneck": system_efficiency_summary.get("top_loop_bottleneck"),
            "system_efficiency_maintainability_score": system_efficiency_summary.get("maintainability_score"),
            "system_efficiency_hard_maintainability_hotspots": system_efficiency_summary.get("hard_maintainability_hotspot_count"),
            "system_efficiency_attd_score": system_efficiency_summary.get("attd_score"),
            "system_efficiency_attd_top_component": system_efficiency_summary.get("attd_top_component"),
            "system_efficiency_attd_runtime_overlap_count": system_efficiency_summary.get("attd_runtime_overlap_count"),
            "system_efficiency_architecture_cleanup_queue_count": system_efficiency_summary.get("architecture_cleanup_queue_count"),
            "system_efficiency_top_architecture_cleanup_item": system_efficiency_summary.get("top_architecture_cleanup_item"),
            "system_efficiency_active_code_lm_process_count": system_efficiency_summary.get("active_code_lm_process_count"),
            "system_efficiency_public_calibration_locked": system_efficiency_summary.get("public_calibration_locked"),
            "rotation_governor_trigger_state": rotation_governor.get("trigger_state"),
            "rotation_governor_fresh_selected_targets_remaining": get_path(rotation_governor, ["summary", "fresh_selected_targets_remaining"], None),
            "rotation_governor_lane_backlog": get_path(rotation_governor, ["summary", "lane_backlog"], None),
            "scheduler_trigger_state": high_transfer_scheduler.get("trigger_state"),
            "scheduler_ready_tasks": scheduler_ready_tasks,
            "scheduler_critical_tasks": scheduler_critical_tasks,
            "teacher_blocks_since_completed": blocked_since_completed,
            "teacher_last_status": teacher_last.get("status"),
            "teacher_last_age_seconds": last_teacher_age,
            "active_frontier_wall": active_wall,
            "active_frontier_attempt_count": attempt_count,
            "latest_frontier_family": latest_family,
            "candidate_promote": candidate.get("promote"),
            "candidate_failed_gates": [c.get("gate") for c in candidate.get("checks", []) if not c.get("passed")],
            "candidate_bottleneck_status": reducer_status,
            "candidate_flow_ready": reducer_flow_ready,
            "candidate_remaining_safe_auto_actions": remaining_safe_count,
            "alignment_expected_frontier_family": expected_family,
            "alignment_expected_pressure_card_id": expected_card,
            "alignment_transfer_family": transfer_family,
            "alignment_architecture_family": architecture_family,
            "training_budget_sufficient": budget_ok,
            "training_budget_train_env_steps": get_path(budget, ["summary", "train_env_steps_budget"], None),
            "genesis_kernel_trigger_state": get_path(genesis, ["summary", "trigger_state"], None),
            "genesis_kernel_age_seconds": genesis_age,
            "reality_manipulator_trigger_state": reality.get("trigger_state"),
            "reality_manipulator_age_seconds": reality_age,
            "reality_manipulator_world": get_path(reality, ["world", "name"], None),
            "reality_manipulator_compile_targets": get_path(reality, ["world", "compile_targets"], None),
            "reality_manipulator_high_risk_approved_without_gate": high_risk_approved,
            "personality_runtime_trigger_state": personality_runtime.get("trigger_state"),
            "personality_runtime_age_seconds": personality_age,
            "personality_selected_cards": get_path(personality_context, ["summary", "selected_cards"], None),
            "personality_drift_score": get_path(personality_drift, ["summary", "average_score"], None),
            "personality_belief_quarantined": get_path(belief_governance, ["summary", "quarantined"], None),
            "synthetic_benchmark_trigger_state": get_path(synthetic_benchmarks, ["trigger_state"], None),
            "synthetic_benchmark_cases": get_path(synthetic_benchmarks, ["summary", "case_count"], None),
            "multi_stream_factory_trigger_state": get_path(multi_stream_factory, ["trigger_state"], None),
            "multi_stream_cases": get_path(multi_stream_factory, ["summary", "case_count"], None),
            "multi_stream_pressure_score": multi_stream_pressure.get("score"),
            "multi_stream_pressure_pass_rate": get_path(multi_stream_pressure, ["summary", "multi_stream_pass_rate"], None),
            "multi_stream_pressure_delta": get_path(multi_stream_pressure, ["summary", "pass_rate_delta"], None),
            "multi_stream_task_level_improvements": get_path(multi_stream_pressure, ["summary", "task_level_improvements_over_single_stream"], None),
            "multi_stream_task_level_regressions": get_path(multi_stream_pressure, ["summary", "task_level_regressions_vs_single_stream"], None),
            "multi_stream_patch_synthesis_used": get_path(multi_stream_pressure, ["summary", "patch_stream_synthesis_used_count"], None),
            "multi_stream_monitorability_score": get_path(multi_stream_probe, ["summary", "monitorability_score"], None),
            "real_code_graduation_trigger_state": real_code.get("trigger_state"),
            "real_code_graduation_public_tasks": get_path(real_code, ["summary", "public_task_count"], None),
            "real_code_graduation_total_cases": get_path(real_code, ["summary", "total_case_count"], None),
            "real_code_graduation_delta": get_path(real_code, ["summary", "pass_rate_delta"], None),
            "real_code_graduation_regressions": get_path(real_code, ["summary", "task_level_regressions_vs_single_stream"], None),
            "broad_transfer_matrix_trigger_state": broad_matrix.get("trigger_state"),
            "broad_transfer_matrix_public_tasks": broad_summary.get("real_public_task_count"),
            "broad_transfer_matrix_pass_rate": broad_summary.get("real_public_pass_rate"),
            "broad_transfer_matrix_blockers": broad_blockers,
            "broad_code_calibration_selected": broad_selected.get("card_id"),
            "broad_code_calibration_action": broad_selected.get("action"),
            "broad_code_calibration_can_run_real_code": broad_selected.get("can_run_real_code"),
            "code_lm_private_run_status": code_lm_private.get("run_status"),
            "code_lm_private_progress_stage": code_lm_private.get("progress_stage"),
            "code_lm_rust_run_status": code_lm_private_rust.get("run_status"),
            "code_lm_rust_progress_stage": code_lm_private_rust.get("progress_stage"),
            "code_lm_candidate_heartbeat_age_seconds": code_lm_heartbeat_age if code_lm_heartbeat else None,
            "code_lm_candidate_heartbeat_path": str(code_lm_heartbeat_path.relative_to(ROOT)).replace("\\", "/")
            if code_lm_heartbeat
            else None,
            "code_lm_candidate_heartbeat_status": code_lm_heartbeat.get("heartbeat_status"),
            "code_lm_candidate_heartbeat_stage": code_lm_heartbeat.get("stage"),
            "code_lm_candidate_heartbeat_phase": code_lm_heartbeat.get("phase"),
            "code_lm_candidate_heartbeat_progress_ratio": get_path(code_lm_heartbeat, ["progress", "progress_ratio"], None),
            "code_lm_candidate_heartbeat_completed_tasks": get_path(code_lm_heartbeat, ["progress", "completed_tasks"], None),
            "code_lm_candidate_heartbeat_total_tasks": get_path(code_lm_heartbeat, ["progress", "total_tasks"], None),
            "code_lm_candidate_heartbeat_current_task": get_path(code_lm_heartbeat, ["progress", "current_task", "task_id"], None),
            "code_lm_candidate_heartbeat_rejections": get_path(code_lm_heartbeat, ["progress", "rejection_counts_for_current_task"], None),
            "code_lm_shard_strategy_trigger_state": shard_strategy.get("trigger_state"),
            "code_lm_shard_strategy_completed_shards": shard_strategy_summary.get("completed_shards"),
            "code_lm_shard_strategy_current_artifact_mb": shard_strategy_summary.get("current_artifact_mb"),
            "code_lm_shard_strategy_projected_artifact_mb": shard_strategy_summary.get("projected_artifact_mb"),
            "code_lm_shard_strategy_repeats_training": shard_strategy_summary.get("repeated_training_per_shard_detected"),
            "code_lm_shard_strategy_duplicate_active_shards": duplicate_active_shards,
            "code_lm_train_once_fanout_trigger_state": train_once_fanout.get("trigger_state"),
            "code_lm_train_once_fanout_run_status": train_once_fanout.get("run_status"),
            "code_lm_train_once_fanout_current_phase": train_once_fanout.get("current_phase"),
            "code_lm_train_once_fanout_active_heartbeat_path": train_once_fanout.get("active_phase_heartbeat"),
            "code_lm_train_once_fanout_active_heartbeat_stage": get_path(
                train_once_fanout, ["active_phase_heartbeat_summary", "latest_progress_stage"], None
            ),
            "code_lm_train_once_fanout_active_heartbeat_age_seconds": get_path(
                train_once_fanout, ["active_phase_heartbeat_summary", "age_seconds"], None
            ),
            "code_lm_train_once_fanout_repeated_training": train_once_summary.get(
                "repeated_training_per_candidate_shard",
                get_path(train_once_fanout, ["architecture", "repeated_training_per_candidate_shard"], None),
            ),
            "code_lm_train_once_fanout_closure_report": train_once_fanout.get("closure_report"),
            "open_code_training_pantry_expressions": get_path(open_code_pantry, ["summary", "private_train_expression_count"], None),
            "sts_learning_rows": get_path(sts_learning, ["summary", "row_count"], None),
            "sts_native_parallel_generation_proven": get_path(sts_native, ["summary", "native_parallel_token_generation_proven"], None),
            "sts_native_eval_delta": get_path(sts_native, ["summary", "eval_token_accuracy_delta"], None),
            "cognitive_context_trigger_state": cognitive_context.get("trigger_state"),
            "cognitive_context_rows": cognitive_summary.get("context_row_count"),
            "cognitive_context_visible_review": cognitive_summary.get("visible_report_requires_review"),
            "learning_scoreboard_trigger_state": learning_scoreboard.get("trigger_state"),
            "learning_scoreboard_public_pass_rate": get_path(learning_scoreboard, ["public_transfer", "real_public_task_pass_rate"], None),
            "learning_scoreboard_promotion_allowed": get_path(learning_scoreboard, ["promotion", "promotion_allowed"], None),
            "overnight_readiness_trigger_state": get_path(state, ["overnight_readiness", "trigger_state"], None),
            "overnight_launch_ready": get_path(state, ["overnight_readiness", "overnight_launch_ready"], None),
            "overnight_next_card": get_path(state, ["overnight_readiness", "current_truth", "next_recommended_card"], None),
            "overnight_rotation_reason": get_path(state, ["overnight_readiness", "current_truth", "rotation_reason"], None),
            "cell_lifecycle_trigger_state": get_path(state, ["cell_lifecycle", "trigger_state"], None),
            "cell_lifecycle_improve_candidates": get_path(state, ["cell_lifecycle", "summary", "improve_candidates"], None),
            "cell_lifecycle_split_or_compress_candidates": get_path(state, ["cell_lifecycle", "summary", "split_or_compress_candidates"], None),
            "cell_lifecycle_retire_candidates": get_path(state, ["cell_lifecycle", "summary", "retire_candidates"], None),
            "cell_lifecycle_tool_creation_pressure": get_path(state, ["cell_lifecycle", "summary", "tool_creation_pressure_count"], None),
            "cell_lifecycle_data_archive_candidates": get_path(state, ["cell_lifecycle", "summary", "training_data_archive_candidates"], None),
            "grammar_suckers_trigger_state": grammar_suckers.get("trigger_state"),
            "grammar_suckers_python_parse_pass_rate": grammar_summary.get("python_parse_pass_rate"),
            "grammar_suckers_invalid_python_promotion_candidates": grammar_invalid_promotion,
            "grammar_suckers_english_surface_pass_rate": grammar_summary.get("english_surface_pass_rate"),
            "grammar_suckers_sbl_traces": grammar_summary.get("sbl_trace_count"),
            "deterministic_taming_trigger_state": taming.get("trigger_state"),
            "deterministic_taming_hard_failures": taming_summary.get("hard_failure_count"),
            "code_residual_curriculum_rows": residual_curriculum_rows,
            "code_lm_training_input_mutation_deferred": code_lm_training_input_mutation_deferred,
            "code_residual_curriculum_refresh_action": (
                "defer_while_code_lm_active"
                if code_lm_training_input_mutation_deferred and not residual_curriculum_ok
                else "not_needed"
            ),
            "sts_repair_ablation_delta": sts_delta,
            "sts_repair_ablation_regressions": sts_regressions,
            "architecture_guidance_wall": get_path(architecture_guidance, ["diagnosis", "wall"], None),
            "architecture_guidance_teacher_status": get_path(architecture_guidance, ["teacher", "status"], None),
            "teacher_budget_audit_trigger_state": get_path(state, ["teacher_budget_audit", "trigger_state"], None),
            "teacher_budget_architecture_allowed": get_path(state, ["teacher_budget_audit", "reason_decisions", "architecture_wall", "budget", "allowed"], None),
        },
        "checks": checks,
        "recommended_actions": actions,
    }

if __name__ == "__main__":
    raise SystemExit(main())
