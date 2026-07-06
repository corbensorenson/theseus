"""Pre-overnight readiness gate for honest Theseus learning runs.

This report is intentionally separate from candidate promotion. A night run can
be ready to keep learning while still being blocked from promotion. The gate
checks that curriculum rotation, STS/context streams, sparse teacher guidance,
runtime storage, and anti-cheat evidence are wired well enough to run unattended
without turning a stale/private signal into a false learning claim.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import theseus_runtime


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MAX_REPORT_AGE_SECONDS = 3600
MIN_RUNTIME_FREE_GIB = 100.0
MIN_RUNTIME_FREE_GIB_NON_WINDOWS = 20.0
MIN_SOURCE_DRIVE_FREE_GIB = 10.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/overnight_learning_readiness.json")
    parser.add_argument("--markdown-out", default="reports/overnight_learning_readiness.md")
    parser.add_argument("--max-age-seconds", type=int, default=MAX_REPORT_AGE_SECONDS)
    args = parser.parse_args()

    state = load_state()
    report = build_report(state, max_age_seconds=args.max_age_seconds)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def load_state() -> dict[str, Any]:
    runtime = theseus_runtime.runtime_report(create=True, write_report=True)
    return {
        "sparkstream": read_json(REPORTS / "sparkstream_status.json"),
        "hive": read_json(REPORTS / "hive_status.json"),
        "watchdog": read_json(REPORTS / "autonomy_watchdog.json"),
        "vacation_mode": read_json(REPORTS / "vacation_mode_supervisor_overnight.json"),
        "learning": read_json(REPORTS / "learning_scoreboard.json"),
        "reality_manipulator": read_json(REPORTS / "reality_manipulator.json"),
        "candidate": read_json(REPORTS / "candidate_promotion_gate.json"),
        "benchmaxx": read_json(REPORTS / "benchmaxx_curriculum.json"),
        "frontier": read_json(REPORTS / "frontier_policy_status.json"),
        "architecture": read_json(REPORTS / "architecture_guidance_loop.json"),
        "teacher_budget": read_json(REPORTS / "teacher_budget_last.json"),
        "cognitive": read_json(REPORTS / "cognitive_context_router.json"),
        "sts_native": read_json(REPORTS / "sts_native_parallel_probe.json"),
        "grammar": read_json(REPORTS / "grammar_suckers.json"),
        "taming": read_json(REPORTS / "deterministic_taming_stack.json"),
        "real_code": read_json(REPORTS / "real_code_benchmark_graduation.json"),
        "wide_public_code": read_json(REPORTS / "real_code_benchmark_graduation_wide_public_seed23_5x32_interface_floor_v1.json"),
        "public_residual": read_json(REPORTS / "public_code_transfer_residual_report_wide_public_seed23_5x32_interface_floor_v1.json"),
        "generalization_governor": read_json(REPORTS / "theseus_generalization_governor_v1.json"),
        "readiness_packet": read_json(REPORTS / "public_calibration_readiness_packet.json"),
        "v4_learned": read_json(REPORTS / "public_safe_broad_transfer_maturity_v4_learned_distillation_gate.json"),
        "v4_sts_streams": read_json(REPORTS / "public_safe_broad_transfer_maturity_v4_private_safe_sts_streams.json"),
        "v5_refresh": read_json(REPORTS / "private_ecology_generalization_v5_refresh.json"),
        "residual_curriculum": read_json(REPORTS / "code_residual_curriculum.json"),
        "open_code": read_json(REPORTS / "open_code_training_pantry.json"),
        "open_conversation": read_json(REPORTS / "open_conversation_training_pantry.json"),
        "self_edit": read_json(REPORTS / "self_edit_experiment_lane.json"),
        "runtime": runtime,
        "git": git_state(),
    }


def build_report(state: dict[str, Any], *, max_age_seconds: int) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    learning = state["learning"]
    public_transfer = object_field(learning, "public_transfer")
    promotion = object_field(learning, "promotion")
    governor = state["generalization_governor"]
    governor_summary = object_field(governor, "summary")
    readiness_packet = state["readiness_packet"]
    readiness_summary = object_field(readiness_packet, "summary")
    benchmaxx = state["benchmaxx"]
    next_frontier = object_field(benchmaxx, "next_frontier")
    rotation = object_field(next_frontier, "same_family_rotation")
    transfer_interleave = object_field(next_frontier, "transfer_interleave")
    public_stall = object_field(rotation, "public_code_transfer_stall")
    real_summary = object_field(state["real_code"], "summary")
    wide_public_summary = object_field(state["wide_public_code"], "summary")
    v4_learned_summary = object_field(state["v4_learned"], "summary")
    v4_sts_summary = object_field(state["v4_sts_streams"], "summary")
    v5_summary = object_field(state["v5_refresh"], "summary")
    runtime_summary = runtime_readiness(state["runtime"])
    git = state["git"]

    spark_age = report_age_seconds(state["sparkstream"], max_age_seconds=max_age_seconds)
    spark_phase = str(state["sparkstream"].get("phase") or "")
    add_check(
        checks,
        "sparkstream_daemon_alive_or_sleeping",
        spark_age <= max_age_seconds and spark_phase not in {"cycle_failed", "stopped", ""},
        "RED",
        f"phase={spark_phase or 'missing'} age_seconds={spark_age}",
    )
    watchdog_state = str(state["watchdog"].get("trigger_state") or "")
    watchdog_status_age = number(get_path(state["watchdog"], ["summary", "status_age_seconds"]))
    watchdog_failed_streak = int(number(get_path(state["watchdog"], ["summary", "failed_cycle_streak"])))
    watchdog_red_failures = {
        str(row.get("name") or "")
        for row in state["watchdog"].get("checks", [])
        if isinstance(row, dict) and row.get("severity") == "RED" and not row.get("passed")
    }
    vacation = state["vacation_mode"]
    vacation_age = report_age_seconds(vacation, max_age_seconds=max_age_seconds)
    vacation_supervisor_ok = (
        vacation.get("trigger_state") in {"GREEN", "YELLOW"}
        and vacation_age <= max_age_seconds
        and bool(get_path(vacation, ["last_cycle", "progress_contract", "passed"], False))
        and not bool(get_path(vacation, ["summary", "paused"], False))
        and not bool(get_path(vacation, ["summary", "stop_requested"], False))
    )
    teacher_budget_exhausted = (
        str(state["teacher_budget"].get("status") or "") == "blocked_by_teacher_budget"
        and str(get_path(state["teacher_budget"], ["budget_decision", "reason"], "")) == "daily_call_budget_exhausted"
    )
    overrideable_watchdog_failures = {
        "candidate_gate_uses_current_frontier",
        "governance_reports_share_active_frontier",
        "overnight_learning_readiness_current",
    }
    if teacher_budget_exhausted and vacation_supervisor_ok:
        overrideable_watchdog_failures.add("teacher_escalation_not_starved")
    watchdog_red_overrideable = bool(watchdog_red_failures) and watchdog_red_failures <= overrideable_watchdog_failures
    watchdog_operational = (
        watchdog_status_age <= max_age_seconds
        and watchdog_failed_streak == 0
        and (
            watchdog_state in {"GREEN", "YELLOW"}
            or (watchdog_red_overrideable and vacation_supervisor_ok)
        )
    )
    add_check(
        checks,
        "watchdog_operational_not_red",
        watchdog_operational,
        "RED",
        (
            f"trigger_state={watchdog_state or 'missing'} "
            f"status_age_seconds={watchdog_status_age} failed_streak={watchdog_failed_streak} "
            f"red_failures={sorted(watchdog_red_failures)} "
            f"vacation_supervisor_ok={vacation_supervisor_ok} vacation_age_seconds={vacation_age} "
            f"teacher_budget_exhausted={teacher_budget_exhausted}"
        ),
    )
    scoreboard_age = report_age_seconds(learning, max_age_seconds=max_age_seconds)
    governor_age = report_age_seconds(governor, max_age_seconds=max_age_seconds)
    readiness_age = report_age_seconds(readiness_packet, max_age_seconds=max_age_seconds)
    scoreboard_current = bool(
        learning.get("policy") == "project_theseus_learning_scoreboard_v1"
        and scoreboard_age <= max_age_seconds
        and bool(get_path(learning, ["dashboard_truth", "show_public_transfer_as_active_truth"], False))
        and bool(get_path(learning, ["dashboard_truth", "stale_lanes_are_historical_context_not_active_blocks"], False))
    )
    governor_current = bool(
        governor.get("policy") == "project_theseus_generalization_governor_v1"
        and governor_age <= max_age_seconds
        and governor.get("trigger_state") in {"GREEN", "YELLOW"}
        and governor_summary.get("public_calibration_allowed") is False
        and governor_summary.get("operator_lock_active") is True
        and int(number(governor_summary.get("forbidden_post_v4_public_artifact_count"))) == 0
    )
    add_check(
        checks,
        "generalization_truth_is_canonical_and_fresh",
        scoreboard_current or governor_current,
        "RED",
        (
            f"scoreboard_trigger_state={learning.get('trigger_state')} scoreboard_age_seconds={scoreboard_age} "
            f"governor_trigger_state={governor.get('trigger_state')} governor_age_seconds={governor_age} "
            f"readiness_age_seconds={readiness_age} public_calibration_allowed={governor_summary.get('public_calibration_allowed')}"
        ),
    )
    reality = state["reality_manipulator"]
    reality_age = report_age_seconds(reality, max_age_seconds=max_age_seconds)
    add_check(
        checks,
        "reality_manipulator_world_kernel_ready",
        reality.get("policy") == "project_theseus_reality_manipulator_mvp_v1"
        and reality_age <= max_age_seconds
        and int(get_path(reality, ["safety_model", "high_risk_approved_without_gate_count"], 999) or 0) == 0
        and bool(get_path(reality, ["acceptance_scenario", "world_created"], False))
        and bool(get_path(reality, ["acceptance_scenario", "release_manifest_ready"], False)),
        "RED",
        (
            f"trigger_state={reality.get('trigger_state') or 'missing'} age_seconds={reality_age} "
            f"world={get_path(reality, ['world', 'name'], 'missing')} "
            f"high_risk_approved_without_gate={get_path(reality, ['safety_model', 'high_risk_approved_without_gate_count'], None)}"
        ),
    )
    public_rate = float(first_number(
        governor_summary.get("public_pass_rate"),
        readiness_summary.get("broad_public_pass_rate"),
        wide_public_summary.get("real_public_task_pass_rate"),
        wide_public_summary.get("multi_stream_pass_rate"),
        public_transfer.get("real_public_task_pass_rate"),
        0.0,
    ))
    public_floor = float(first_number(
        public_transfer.get("required_floor"),
        governor_summary.get("public_floor"),
        readiness_summary.get("public_floor"),
        0.70,
    ))
    promotion_allowed = bool(promotion.get("promotion_allowed"))
    candidate_promote = bool(state["candidate"].get("promote"))
    add_check(
        checks,
        "promotion_blocked_until_public_transfer_floor",
        (public_rate >= public_floor and promotion_allowed == candidate_promote)
        or (public_rate < public_floor and not promotion_allowed and not candidate_promote),
        "RED",
        f"public_rate={public_rate:.6f} floor={public_floor:.2f} scoreboard_promotion={promotion_allowed} candidate_promote={candidate_promote}",
    )
    rotation_reason = str(rotation.get("reason") or "")
    transfer_reason = str(transfer_interleave.get("reason") or "")
    transfer_rung_ready = bool(
        transfer_interleave.get("apply")
        or transfer_interleave.get("same_family_is_moving")
        or transfer_interleave.get("force_cross_family")
        or transfer_reason in {
            "same_family_rotation_precedes_transfer_interleave",
            "cross_family_transfer_interleave_due",
        }
    )
    attempts = int(number(public_stall.get("observed_attempts")))
    threshold = int(number(public_stall.get("attempts_before_rotate")) or 1)
    public_below_floor = bool(public_rate < public_floor)
    rotation_due = bool(public_below_floor and attempts >= threshold)
    add_check(
        checks,
        "code_family_rotation_handles_public_transfer_stall",
        (
            not rotation_due
            or rotation_reason
            in {
                "rotate_public_code_transfer_stalled_card",
                "public_code_transfer_stalled_but_no_ready_alternate",
                "rotate_to_public_code_graduation_ready_card",
                "continue_current_public_code_card_until_calibration_runs",
            }
            or transfer_rung_ready
        ),
        "RED",
        (
            f"reason={rotation_reason or 'missing'} transfer={transfer_reason or 'missing'} "
            f"public_below_floor={public_below_floor} attempts={attempts} threshold={threshold} "
            f"forced_at={transfer_interleave.get('max_same_family_attempts_before_forced_interleave')}"
        ),
    )
    teacher = object_field(state["architecture"], "teacher")
    diagnosis = object_field(state["architecture"], "diagnosis")
    add_check(
        checks,
        "teacher_is_sparse_architecture_guidance_only",
        str(diagnosis.get("teacher_role") or "") == "proposal_only_architecture_guidance"
        and int(teacher.get("external_inference_calls") or 0) <= 1
        and str(teacher.get("mode") or "proposal") == "proposal",
        "RED",
        f"teacher_status={teacher.get('status')} mode={teacher.get('mode', 'proposal')} external_calls={teacher.get('external_inference_calls')}",
    )
    cognitive = object_field(state["cognitive"], "summary")
    sts = object_field(state["sts_native"], "summary")
    private_sts_current = bool(
        state["v4_sts_streams"].get("trigger_state") == "GREEN"
        and int(number(
            v4_sts_summary.get("sts_stream_task_count")
            or v4_sts_summary.get("stream_row_count")
            or v4_sts_summary.get("row_count")
        )) >= 1008
        and int(number(v4_sts_summary.get("public_data_leakage_hit_count") or v4_sts_summary.get("public_leak_hit_count"))) == 0
        and int(number(state["v4_sts_streams"].get("external_inference_calls") or v4_sts_summary.get("external_inference_calls"))) == 0
    )
    add_check(
        checks,
        "sts_context_spaces_are_active_and_review_gated",
        (
            state["cognitive"].get("trigger_state") == "GREEN"
            and state["sts_native"].get("trigger_state") == "GREEN"
            and int(cognitive.get("merged_row_count") or 0) > int(cognitive.get("base_row_count") or 0)
            and bool(cognitive.get("visible_report_requires_review"))
            and str(cognitive.get("raw_chain_of_thought_exposure") or "") == "forbidden"
            and bool(sts.get("native_parallel_token_generation_proven"))
        )
        or private_sts_current,
        "RED",
        (
            f"context_rows={cognitive.get('context_row_count')} merged={cognitive.get('merged_row_count')} "
            f"native_streams={sts.get('output_stream_count')} private_v4_sts_current={private_sts_current} "
            f"private_v4_sts_rows={v4_sts_summary.get('sts_stream_task_count') or v4_sts_summary.get('stream_row_count') or v4_sts_summary.get('row_count')}"
        ),
    )
    grammar = object_field(state["grammar"], "summary")
    taming = object_field(state["taming"], "summary")
    add_check(
        checks,
        "deterministic_rule_substrate_clean_for_code",
        state["grammar"].get("trigger_state") in {"GREEN", "YELLOW"}
        and state["taming"].get("trigger_state") in {"GREEN", "YELLOW"}
        and int(grammar.get("python_invalid_promotion_eligible_count") or 0) == 0
        and int(taming.get("hard_failure_count") or 0) == 0,
        "RED",
        (
            f"grammar={state['grammar'].get('trigger_state')} taming={state['taming'].get('trigger_state')} "
            f"invalid_python={grammar.get('python_invalid_promotion_eligible_count')} "
            f"taming_hard_failures={taming.get('hard_failure_count')} taming_soft_failures={taming.get('soft_failure_count')}"
        ),
    )
    add_check(
        checks,
        "anti_cheat_public_code_evidence_clean",
        (
            state["real_code"].get("candidate_source") == "student_code_lm_checkpoint_v1"
            and bool(real_summary.get("student_candidate_benchmark_integrity_valid"))
            and bool(real_summary.get("token_level_code_generation_learned"))
            and int(real_summary.get("template_like_candidate_count") or 0) == 0
            and int(real_summary.get("loop_closure_candidate_count") or 0) == 0
            and int(state["real_code"].get("external_inference_calls") or 0) == 0
        )
        or (
            state["wide_public_code"].get("policy") == "project_theseus_real_code_benchmark_graduation_v1"
            and state["wide_public_code"].get("trigger_state") == "GREEN"
            and bool(wide_public_summary.get("student_candidate_benchmark_integrity_valid"))
            and bool(wide_public_summary.get("token_level_code_generation_learned"))
            and int(number(wide_public_summary.get("template_like_candidate_count"))) == 0
            and int(number(wide_public_summary.get("loop_closure_candidate_count"))) == 0
            and int(number(wide_public_summary.get("external_inference_calls"))) == 0
            and int(number(governor_summary.get("forbidden_post_v4_public_artifact_count"))) == 0
            and readiness_summary.get("public_tests_or_solutions_visible") is False
        ),
        "RED",
        (
            f"legacy_candidate_source={state['real_code'].get('candidate_source')} "
            f"legacy_templates={real_summary.get('template_like_candidate_count')} "
            f"wide_integrity={wide_public_summary.get('student_candidate_benchmark_integrity_valid')} "
            f"wide_templates={wide_public_summary.get('template_like_candidate_count')} "
            f"wide_loop_closure={wide_public_summary.get('loop_closure_candidate_count')} "
            f"public_tests_or_solutions_visible={readiness_summary.get('public_tests_or_solutions_visible')}"
        ),
    )
    residual = object_field(state["residual_curriculum"], "summary")
    current_private_training_clean = bool(
        state["v4_learned"].get("trigger_state") == "GREEN"
        and float(number(v4_learned_summary.get("learned_only_pass_rate"))) >= 1.0
        and int(number(v4_learned_summary.get("prototype_pass_count"))) == 0
        and int(number(v4_learned_summary.get("exact_train_body_memory_pass_count"))) == 0
        and int(number(v4_learned_summary.get("learned_token_pass_count"))) >= 1008
        and state["v5_refresh"].get("trigger_state") == "GREEN"
        and int(number(v5_summary.get("learned_token_pass_count"))) >= 480
        and int(number(governor_summary.get("external_inference_calls"))) == 0
        and readiness_summary.get("public_tests_or_solutions_visible") is False
    )
    add_check(
        checks,
        "private_training_evidence_clean_and_current",
        (
            state["residual_curriculum"].get("trigger_state") in {"GREEN", "YELLOW"}
            and bool(residual.get("public_benchmark_solutions_included")) is False
            and bool(residual.get("public_tests_included")) is False
            and int(residual.get("private_row_count") or 0) > 0
        )
        or current_private_training_clean,
        "RED",
        (
            f"legacy_rows={residual.get('private_row_count')} "
            f"legacy_public_solutions={residual.get('public_benchmark_solutions_included')} "
            f"legacy_public_tests={residual.get('public_tests_included')} "
            f"v4_learned_only_pass_rate={v4_learned_summary.get('learned_only_pass_rate')} "
            f"v4_learned_tokens={v4_learned_summary.get('learned_token_pass_count')} "
            f"v5_learned_tokens={v5_summary.get('learned_token_pass_count')} "
            f"current_private_training_clean={current_private_training_clean}"
        ),
    )
    add_check(
        checks,
        "runtime_storage_has_room",
        runtime_summary["runtime_paths_exist"]
        and runtime_summary["runtime_location_ready"]
        and runtime_summary["runtime_free_gib"] >= runtime_summary["min_runtime_free_gib"]
        and (
            runtime_summary["source_free_gib"] >= MIN_SOURCE_DRIVE_FREE_GIB
            or runtime_summary["source_drive_floor_suppressed"]
        ),
        "RED",
        (
            f"platform={runtime_summary['platform']} runtime_anchor={runtime_summary['runtime_anchor']} "
            f"runtime_free_gib={runtime_summary['runtime_free_gib']} "
            f"min_runtime_free_gib={runtime_summary['min_runtime_free_gib']} "
            f"source_free_gib={runtime_summary['source_free_gib']} "
            f"runtime_paths_exist={runtime_summary['runtime_paths_exist']} "
            f"runtime_location_ready={runtime_summary['runtime_location_ready']} "
            f"source_floor_suppressed={runtime_summary['source_drive_floor_suppressed']}"
        ),
    )
    dirty_source_count = len(git["source_or_config_changed"])
    self_edit_commit_allowed = bool(state["self_edit"].get("commit_allowed"))
    add_check(
        checks,
        "self_edit_guarded_when_source_boundary_dirty",
        dirty_source_count == 0 or not self_edit_commit_allowed,
        "YELLOW",
        f"dirty_source_or_config={dirty_source_count} self_edit_commit_allowed={self_edit_commit_allowed}",
    )

    red_failures = [row for row in checks if row["severity"] == "RED" and not row["passed"]]
    yellow_failures = [row for row in checks if row["severity"] == "YELLOW" and not row["passed"]]
    launch_ready = not red_failures
    trigger = "GREEN" if launch_ready and not yellow_failures and public_rate >= public_floor else "YELLOW"
    if red_failures:
        trigger = "RED"
    report = {
        "policy": "project_theseus_overnight_learning_readiness_v1",
        "created_utc": now(),
        "trigger_state": trigger,
        "overnight_launch_ready": launch_ready,
        "promotion_ready": bool(public_rate >= public_floor and promotion_allowed),
        "current_truth": {
            "public_code_pass_rate": public_rate,
            "required_public_code_floor": public_floor,
            "promotion_allowed": promotion_allowed,
            "active_frontier_family": get_path(learning, ["frontier_truth", "frontier_family"], None),
            "pressure_card_id": get_path(learning, ["frontier_truth", "pressure_card_id"], None),
            "next_recommended_card": next_frontier.get("recommended_env"),
            "rotation_reason": rotation_reason,
            "transfer_interleave_reason": transfer_reason,
            "transfer_interleave_apply": bool(transfer_interleave.get("apply")),
            "transfer_force_cross_family": bool(transfer_interleave.get("force_cross_family")),
            "teacher_status": teacher.get("status"),
        },
        "runtime": runtime_summary,
        "git": git,
        "checks": checks,
        "red_failures": [row["name"] for row in red_failures],
        "yellow_failures": [row["name"] for row in yellow_failures],
        "next_best_action": next_best_action(trigger, public_rate, public_floor, rotation_reason, dirty_source_count),
        "external_inference_calls": 0,
    }
    return report


def next_best_action(trigger: str, public_rate: float, public_floor: float, rotation_reason: str, dirty_source_count: int) -> str:
    if trigger == "RED":
        return "Fix RED readiness gates before unattended overnight learning."
    if public_rate < public_floor and rotation_reason == "rotate_public_code_transfer_stalled_card":
        return "Run the next inner_loop cycle so code-family rotation attacks the next public-shaped code card with transfer artifacts loaded."
    if public_rate < public_floor:
        return "Continue private residual/STS code training, then rerun public calibration; promotion remains blocked honestly."
    if dirty_source_count:
        return "Promotion can remain blocked or close honestly, but commit/shelve source changes before enabling unattended self-edit commits."
    return "Ready for overnight learning; keep watchdog hourly and promotion gated by public transfer."


def runtime_readiness(runtime: dict[str, Any]) -> dict[str, Any]:
    paths = runtime.get("paths") if isinstance(runtime.get("paths"), dict) else {}
    runtime_paths = [
        str(get_path(paths, [key, "path"], ""))
        for key in ["runtime_root", "data_dir", "cache_dir", "reports_dir", "checkpoints_dir", "cargo_target_dir"]
    ]
    system = platform.system() or os.name
    runtime_root = Path(str(get_path(paths, ["runtime_root", "path"], "") or default_storage_anchor()))
    runtime_anchor = storage_anchor(runtime_root)
    source_anchor = storage_anchor(ROOT)
    runtime_paths_exist = all(Path(path).exists() for path in runtime_paths if path)
    paths_on_d = all(path.upper().startswith("D:\\") or path.upper().startswith("D:/") for path in runtime_paths if path)
    runtime_location_ready = paths_on_d if system == "Windows" and Path("D:/").exists() else bool(runtime_anchor)
    min_runtime_free = configured_min_runtime_free_gib(system)
    runtime_free = disk_free_gib(runtime_anchor)
    source_free = disk_free_gib(source_anchor)
    migration = runtime.get("migration", {}) if isinstance(runtime.get("migration"), dict) else {}
    managed = migration.get("managed_directories") if isinstance(migration.get("managed_directories"), list) else []
    redirected_names = {
        str(row.get("name") or "")
        for row in managed
        if isinstance(row, dict)
        and row.get("status") == "redirected"
        and bool(row.get("source_is_reparse_point"))
    }
    heavy_generated_redirected = {"reports", "checkpoints", "target"}.issubset(redirected_names)
    source_drive_floor_suppressed = bool(
        runtime_location_ready
        and runtime_free >= min_runtime_free
        and source_free < MIN_SOURCE_DRIVE_FREE_GIB
        and heavy_generated_redirected
    )
    return {
        "platform": system,
        "paths_on_d": paths_on_d,
        "runtime_location_ready": runtime_location_ready,
        "runtime_paths_exist": runtime_paths_exist,
        "runtime_paths": runtime_paths,
        "runtime_anchor": runtime_anchor,
        "source_anchor": source_anchor,
        "runtime_free_gib": runtime_free,
        "source_free_gib": source_free,
        "min_runtime_free_gib": min_runtime_free,
        "min_source_drive_free_gib": MIN_SOURCE_DRIVE_FREE_GIB,
        "heavy_generated_dirs_redirected": heavy_generated_redirected,
        "source_drive_floor_suppressed": source_drive_floor_suppressed,
        "source_drive_floor_reason": (
            "source_drive_low_but_heavy_generated_dirs_are_redirected_to_runtime_storage"
            if source_drive_floor_suppressed
            else "source_drive_free_space_floor_enforced"
        ),
        "migration": migration,
    }


def git_state() -> dict[str, Any]:
    result = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, timeout=30)
    rows = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        status = line[:2].strip()
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip()
        normalized = path.replace("\\", "/")
        rows.append({"status": status, "path": normalized, "kind": git_path_kind(normalized)})
    source = [row["path"] for row in rows if row["kind"] == "source_or_config"]
    generated = [row["path"] for row in rows if row["kind"] == "generated_or_data"]
    return {
        "changed_path_count": len(rows),
        "source_or_config_changed": source[:80],
        "generated_or_data_changed_count": len(generated),
        "source_or_config_changed_count": len(source),
        "note": "Dirty source/config blocks unattended self-edit commits; generated reports/data may continue updating.",
    }


def git_path_kind(path: str) -> str:
    generated_prefixes = (
        "reports/",
        "data/",
        "logs/",
        "checkpoints/",
        "target/",
        ".venv",
        "runtime/",
        "games/",
    )
    if path.startswith(generated_prefixes):
        return "generated_or_data"
    if path.endswith((".jsonl", ".log", ".tmp")):
        return "generated_or_data"
    return "source_or_config"


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, severity: str, evidence: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence})


def report_age_seconds(report: dict[str, Any], *, max_age_seconds: int) -> int:
    raw = report.get("created_utc") or report.get("updated_utc")
    if not raw:
        return max_age_seconds + 1
    try:
        return max(0, int(time.time() - datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()))
    except ValueError:
        return max_age_seconds + 1


def disk_free_gib(anchor: str) -> float:
    try:
        return round(shutil.disk_usage(anchor).free / 1024**3, 2)
    except OSError:
        return 0.0


def configured_min_runtime_free_gib(system: str) -> float:
    raw = os.environ.get("THESEUS_MIN_RUNTIME_FREE_GIB")
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    if system == "Windows":
        return MIN_RUNTIME_FREE_GIB
    return MIN_RUNTIME_FREE_GIB_NON_WINDOWS


def storage_anchor(path: Path) -> str:
    try:
        resolved = path.expanduser().resolve(strict=False)
    except OSError:
        resolved = path.expanduser()
    if platform.system() == "Windows":
        drive = resolved.drive or Path.cwd().drive or "C:"
        return f"{drive}/"
    return str(resolved if resolved.exists() else first_existing_parent(resolved))


def first_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def default_storage_anchor() -> str:
    if platform.system() == "Windows":
        return "D:/" if Path("D:/").exists() else "C:/"
    return str(Path.home())


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Overnight Learning Readiness",
        "",
        f"State: **{report.get('trigger_state')}**",
        f"Launch ready: **{report.get('overnight_launch_ready')}**",
        f"Promotion ready: **{report.get('promotion_ready')}**",
        "",
        "## Current Truth",
    ]
    truth = report.get("current_truth", {})
    lines.extend(
        [
            f"- Public code pass rate: {truth.get('public_code_pass_rate')} / {truth.get('required_public_code_floor')}",
            f"- Pressure card: {truth.get('pressure_card_id')}",
            f"- Next card: {truth.get('next_recommended_card')}",
            f"- Rotation reason: {truth.get('rotation_reason')}",
            f"- Transfer interleave: {truth.get('transfer_interleave_reason')} apply={truth.get('transfer_interleave_apply')} force={truth.get('transfer_force_cross_family')}",
            f"- Teacher status: {truth.get('teacher_status')}",
            "",
            "## Checks",
        ]
    )
    for row in report.get("checks", []):
        mark = "PASS" if row.get("passed") else row.get("severity", "FAIL")
        lines.append(f"- {mark}: {row.get('name')} -- {row.get('evidence')}")
    lines.extend(["", f"Next: {report.get('next_best_action')}"])
    return "\n".join(lines) + "\n"


def object_field(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    return item if isinstance(item, dict) else {}


def get_path(value: Any, path: list[Any], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def first_number(*values: Any) -> float:
    for value in values:
        try:
            if value is not None and value != "":
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
