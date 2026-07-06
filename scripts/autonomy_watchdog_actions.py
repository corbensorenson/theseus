"""Action and refresh runners for the SparkStream autonomy watchdog."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from autonomy_watchdog_helpers import (
    get_path,
    parse_json_object,
    read_json,
    read_jsonl_tail,
    resolve_path,
    safe_name,
    seed_from_text,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
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
TRAIN_ONCE_FANOUT_SLUG = "private_pressure_private_recovery_train_once_fanout_v1"
TRAIN_ONCE_FANOUT_PUBLIC_CARDS = "source_mbpp,source_evalplus,source_bigcodebench,source_livecodebench"


__all__ = [
    "apply_fixes",
    "code_contract_preflight_state",
    "run_code_contract_preflight",
    "run_maturity_integrity_audit",
    "write_override",
    "run_teacher_call",
    "recent_teacher_blocked_correction",
    "run_candidate_gate",
    "active_code_repair_evidence",
    "run_candidate_bottleneck_reducer",
    "refresh_alignment_reports",
    "run_candidate_evidence_profile",
    "candidate_evidence_profile_command",
    "seed_from_pressure_report",
    "refresh_training_budget_plan",
    "run_code_residual_forge",
    "run_code_repair_organism",
    "refresh_open_code_training_pantry",
    "run_sts_learning_forge",
    "run_sts_native_parallel_probe",
    "run_cognitive_context_router",
    "run_real_code_benchmark_graduation",
    "real_code_graduation_card_id",
    "real_code_public_case_budget",
    "code_lm_work_step_budget",
    "effective_frontier_target",
    "pressure_report_for",
    "learned_manifest_matches_card",
    "run_legacy_real_code_benchmark_graduation",
    "run_real_code_fix_command",
    "run_self_edit_experiment_lane",
    "run_long_horizon_memory_probe",
    "refresh_virtual_context_memory",
    "run_synthetic_benchmark_factory",
    "run_multi_stream_trace_factory",
    "run_multi_stream_monitorability_probe",
    "run_multi_stream_candidate_gate",
    "refresh_code_lm_shard_strategy_audit",
    "refresh_code_lm_train_once_fanout_plan",
    "refresh_genesis_kernel",
    "refresh_reality_manipulator",
    "refresh_viea_autonomy_spine",
    "refresh_personality_runtime_audit",
    "refresh_learning_scoreboard",
    "refresh_a_plus_scorecard",
    "refresh_broad_transfer_matrix",
    "refresh_broad_code_calibration_scheduler",
    "run_broad_code_calibration_step",
    "stage_public_code_adapter",
    "refresh_cell_lifecycle",
    "refresh_overnight_learning_readiness",
    "run_hive_work_board_status",
    "run_service_process_hygiene",
    "run_system_efficiency_audit",
    "run_autonomy_rotation_governor",
    "run_grammar_suckers",
    "run_deterministic_taming_stack",
    "run_code_residual_curriculum",
    "run_sts_repair_ablation",
    "run_architecture_guidance_loop",
    "refresh_teacher_budget_audit",
    "restart_sparkstream_stack",
]

def apply_fixes(policy: dict[str, Any], cfg: dict[str, Any], report: dict[str, Any]) -> None:
    for item in report.get("recommended_actions", []):
        kind = item.get("kind")
        if kind in {"restart_daemon", "restart_daemon_then_smoke"}:
            item["applied"] = restart_sparkstream_stack()
        elif kind == "force_rl_interleave":
            write_override(policy, "force_rl_interleave", item.get("reason", "watchdog frontier correction"))
            item["applied"] = True
            item["artifact"] = str(OVERRIDE_PATH.relative_to(ROOT))
        elif kind == "force_curriculum_frontier_override":
            write_override(policy, "force_curriculum_frontier_override", item.get("reason", "watchdog curriculum correction"))
            item["applied"] = True
            item["artifact"] = str(OVERRIDE_PATH.relative_to(ROOT))
        elif kind == "profile_timeout_recovery":
            write_override(policy, "profile_timeout_recovery", item.get("reason", "watchdog profile timeout correction"))
            item["applied"] = True
            item["artifact"] = str(OVERRIDE_PATH.relative_to(ROOT))
        elif kind == "force_teacher_call" and bool(cfg.get("force_teacher_call_on_starvation", True)):
            item.update(run_teacher_call(report))
        elif kind == "rerun_candidate_gate":
            item["applied"] = run_candidate_gate()
        elif kind == "run_candidate_bottleneck_reducer":
            item["applied"] = run_candidate_bottleneck_reducer()
        elif kind == "refresh_alignment_reports":
            item["applied"] = refresh_alignment_reports()
        elif kind == "refresh_training_budget_plan":
            item["applied"] = refresh_training_budget_plan()
        elif kind == "run_code_residual_forge":
            item["applied"] = run_code_residual_forge()
        elif kind == "run_code_repair_organism":
            item["applied"] = run_code_repair_organism()
        elif kind == "run_real_code_benchmark_graduation":
            item["applied"] = run_real_code_benchmark_graduation()
        elif kind == "refresh_open_code_training_pantry":
            item["applied"] = refresh_open_code_training_pantry()
        elif kind == "run_sts_learning_forge":
            item["applied"] = run_sts_learning_forge()
        elif kind == "run_sts_native_parallel_probe":
            item["applied"] = run_sts_native_parallel_probe()
        elif kind == "run_cognitive_context_router":
            item["applied"] = run_cognitive_context_router()
        elif kind == "run_self_edit_experiment_lane":
            item["applied"] = run_self_edit_experiment_lane()
        elif kind == "run_long_horizon_memory_probe":
            item["applied"] = run_long_horizon_memory_probe()
        elif kind == "refresh_virtual_context_memory":
            item["applied"] = refresh_virtual_context_memory()
        elif kind == "refresh_genesis_kernel":
            item["applied"] = refresh_genesis_kernel()
        elif kind == "refresh_reality_manipulator":
            item["applied"] = refresh_reality_manipulator()
        elif kind == "refresh_viea_autonomy_spine":
            item["applied"] = refresh_viea_autonomy_spine()
        elif kind == "refresh_learning_scoreboard":
            item["applied"] = refresh_learning_scoreboard()
        elif kind == "refresh_a_plus_scorecard":
            item["applied"] = refresh_a_plus_scorecard()
        elif kind == "refresh_broad_transfer_matrix":
            item["applied"] = refresh_broad_transfer_matrix()
        elif kind == "run_broad_code_calibration_step":
            item["applied"] = run_broad_code_calibration_step()
        elif kind == "refresh_cell_lifecycle":
            item["applied"] = refresh_cell_lifecycle()
        elif kind == "refresh_overnight_learning_readiness":
            item["applied"] = refresh_overnight_learning_readiness()
        elif kind == "run_hive_work_board_status":
            item["applied"] = run_hive_work_board_status()
        elif kind == "run_service_process_hygiene":
            item["applied"] = run_service_process_hygiene()
        elif kind == "run_system_efficiency_audit":
            item["applied"] = run_system_efficiency_audit()
        elif kind == "run_autonomy_rotation_governor":
            item["applied"] = run_autonomy_rotation_governor()
        elif kind == "run_grammar_suckers":
            item["applied"] = run_grammar_suckers()
        elif kind == "run_deterministic_taming_stack":
            item["applied"] = run_deterministic_taming_stack()
        elif kind == "run_code_residual_curriculum":
            item["applied"] = run_code_residual_curriculum()
        elif kind == "run_sts_repair_ablation":
            item["applied"] = run_sts_repair_ablation()
        elif kind == "run_architecture_guidance_loop":
            item["applied"] = run_architecture_guidance_loop()
        elif kind == "refresh_teacher_budget_audit":
            item["applied"] = refresh_teacher_budget_audit()
        elif kind == "refresh_personality_runtime_audit":
            item["applied"] = refresh_personality_runtime_audit()
        elif kind == "run_synthetic_benchmark_factory":
            item["applied"] = run_synthetic_benchmark_factory()
        elif kind == "run_multi_stream_trace_factory":
            item["applied"] = run_multi_stream_trace_factory()
        elif kind == "run_code_contract_preflight":
            item["applied"] = run_code_contract_preflight()
        elif kind == "run_maturity_integrity_audit":
            item["applied"] = run_maturity_integrity_audit()
        elif kind == "run_multi_stream_monitorability_probe":
            item["applied"] = run_multi_stream_monitorability_probe()
        elif kind == "run_multi_stream_candidate_gate":
            item["applied"] = run_multi_stream_candidate_gate()
        elif kind == "refresh_code_lm_shard_strategy_audit":
            item["applied"] = refresh_code_lm_shard_strategy_audit()
        elif kind == "refresh_code_lm_train_once_fanout_plan":
            item["applied"] = refresh_code_lm_train_once_fanout_plan()


def code_contract_preflight_state(report: dict[str, Any]) -> dict[str, Any]:
    source_mtime = max((path.stat().st_mtime for path in DECODER_RELEVANT_SOURCES if path.exists()), default=0.0)
    report_mtime = CODE_CONTRACT_PREFLIGHT_REPORT.stat().st_mtime if CODE_CONTRACT_PREFLIGHT_REPORT.exists() else 0.0
    raw_preflight = get_path(report, ["summary", "public_decoder_contract_preflight"], {})
    preflight = raw_preflight if isinstance(raw_preflight, dict) else {}
    hard_blockers = preflight.get("hard_blockers") if isinstance(preflight, dict) else []
    varargs = int(get_path(preflight, ["varargs_task_count"], 1) or 0)
    weak_required = int(get_path(preflight, ["weak_required_construct_count"], 1) or 0)
    weak_full_body = int(get_path(preflight, ["weak_full_body_count"], 1) or 0)
    arithmetic_obligations = int(get_path(preflight, ["construct_counts", "arithmetic_formula"], 0) or 0)
    passed = bool(
        report.get("policy") == "project_theseus_code_lm_closure_preflight_v1"
        and report.get("trigger_state") == "GREEN"
        and report.get("run_status") == "completed"
        and preflight.get("passed") is True
        and varargs == 0
        and weak_required == 0
        and weak_full_body == 0
        and arithmetic_obligations > 0
        and not hard_blockers
    )
    current = bool(report_mtime and source_mtime and report_mtime >= source_mtime)
    if not CODE_CONTRACT_PREFLIGHT_REPORT.exists():
        reason = "code_contract_preflight_missing"
    elif not current:
        reason = "code_contract_preflight_stale_after_decoder_source_change"
    elif not passed:
        reason = "code_contract_preflight_failed"
    else:
        reason = "code_contract_preflight_green_current"
    return {
        "ok": bool(passed and current),
        "reason": reason,
        "report": str(CODE_CONTRACT_PREFLIGHT_REPORT.relative_to(ROOT)).replace("\\", "/"),
        "report_mtime": report_mtime or None,
        "source_mtime": source_mtime or None,
        "varargs_task_count": varargs,
        "weak_required_construct_count": weak_required,
        "weak_full_body_count": weak_full_body,
        "arithmetic_formula_obligation_count": arithmetic_obligations,
        "hard_blockers": hard_blockers if isinstance(hard_blockers, list) else [],
    }


def run_code_contract_preflight() -> bool:
    command = [
        sys.executable,
        "scripts/code_lm_closure.py",
        "--public-cards",
        "source_mbpp,source_evalplus,source_bigcodebench,source_livecodebench",
        "--seed",
        "23",
        "--max-public-cases-per-card",
        "32",
        "--private-count",
        "20",
        "--preflight-only",
        "--allow-concurrent",
        "--private-curriculum-out",
        "reports/code_lm_preflight_private_curriculum_seed23_32.jsonl",
        "--public-task-manifest-out",
        "reports/code_lm_public_tasks_preflight_seed23_32.jsonl",
        "--out",
        "reports/code_lm_closure_public_contract_preflight_seed23_32.json",
        "--lock-path",
        "reports/code_lm_closure_public_contract_preflight_seed23_32.lock",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=1800)
    return result.returncode == 0


def run_maturity_integrity_audit() -> bool:
    command = [
        sys.executable,
        "scripts/maturity_integrity_audit.py",
        "--out",
        "reports/maturity_integrity_audit.json",
        "--markdown-out",
        "reports/maturity_integrity_audit.md",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=600)
    return result.returncode in {0, 2}


def write_override(policy: dict[str, Any], reason: str, detail: str) -> None:
    pressure = read_json(REPORTS / "frontier_policy_status.json").get("frontier_pressure") or {}
    curriculum = read_json(REPORTS / "benchmaxx_curriculum.json")
    next_frontier = curriculum.get("next_frontier") if isinstance(curriculum.get("next_frontier"), dict) else {}
    curriculum_family = str(next_frontier.get("family") or "")
    runner_family = str(next_frontier.get("runner_family") or "")
    recommended_env = str(next_frontier.get("recommended_env") or "")
    runner_map = {
        "minecraft_rl_local": "minecraft_rl",
        "drone_rl_local": "drone_rl",
        "coding_local_sandbox": "coding_local_sandbox",
        "web_agent_local": "web_agent_local",
        "transfer_eval_local": "transfer_eval",
    }
    if bool(next_frontier.get("runnable_now")) and runner_family in runner_map:
        frontier_family = runner_map[runner_family]
    elif bool(next_frontier.get("runnable_now")) and curriculum_family in {
        "minecraft_rl",
        "drone_rl",
        "coding_local_sandbox",
        "web_agent_local",
        "transfer_eval",
    }:
        frontier_family = curriculum_family
    else:
        frontier_family = "rl_local"
    env = str(pressure.get("next_rl_frontier_env") or "ocean-noisy-memory")
    seed = int(pressure.get("next_rl_frontier_seed") or ((pressure.get("latest_rl_frontier_seed") or 0) + 1))
    payload = {
        "policy": "sparkstream_watchdog_override_v0",
        "override_id": f"watchdog_{int(time.time() * 1000)}",
        "created_utc": now(),
        "expires_utc": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
        "consumed_utc": "",
        "reason": reason,
        "detail": detail,
        "frontier_family": frontier_family,
        "rl_frontier_env": env,
        "rl_frontier_seed": seed,
        "pressure_card_id": recommended_env
        if frontier_family in {"minecraft_rl", "drone_rl", "coding_local_sandbox", "web_agent_local", "transfer_eval"}
        else "",
        "curriculum_next_frontier_family": curriculum_family,
        "profile": get_path(policy, ["frontier_policy", "default_training_profile_for_fresh_frontier"], "inner_loop"),
        "teacher_reason": "architecture_wall",
    }
    write_json(OVERRIDE_PATH, payload)


def run_teacher_call(report: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        "The SparkStream watchdog detected a RED autonomy condition. "
        "Diagnose the single smallest local correction that would restore unattended learning. "
        "Use the attached reports as evidence; do not propose broad rewrites. Watchdog summary: "
        + json.dumps(report.get("summary", {}), default=str)
    )
    command = [
        sys.executable,
        "scripts/teacher_oracle.py",
        "--reason",
        "architecture_wall",
        "--mode",
        "proposal",
        "--prompt",
        prompt,
        "--local-evidence",
        "reports/autonomy_watchdog.json",
        "reports/vacation_mode_supervisor_overnight.json",
        "reports/hive_work_board_executor.json",
        "reports/high_transfer_curriculum_scheduler.json",
        "reports/broad_transfer_matrix.json",
        "reports/learning_scoreboard.json",
        "reports/teacher_budget_last.json",
        "--allow-teacher",
        "--out",
        "reports/teacher_oracle_last.json",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=1800)
    teacher_result = parse_json_object(result.stdout)
    teacher_last = read_json(REPORTS / "teacher_oracle_last.json")
    teacher_budget = read_json(REPORTS / "teacher_budget_last.json")
    status = str(teacher_result.get("status") or teacher_last.get("status") or "")
    request_id = str(teacher_result.get("request_id") or "")
    completed = bool(
        result.returncode == 0
        and status == "completed"
        and request_id
        and teacher_last.get("request_id") == request_id
    )
    if completed:
        write_json(REPORTS / "teacher_architecture_guidance_last.json", teacher_last)
    payload: dict[str, Any] = {
        "applied": completed,
        "teacher_status": status,
        "teacher_report": "reports/teacher_oracle_last.json",
        "returncode": result.returncode,
    }
    if not completed:
        blocked_reason = teacher_result.get("blocked_reason") or teacher_budget.get("blocked_reason")
        payload["blocked_reason"] = blocked_reason
        payload["budget_decision"] = teacher_result.get("budget_decision") or teacher_budget.get("budget_decision")
        if blocked_reason == "daily_call_budget_exhausted" and recent_teacher_blocked_correction():
            payload["local_correction_recent"] = True
            payload["local_correction_artifact"] = "reports/autonomy_cycle_watchdog_correction.json"
        else:
            write_override(policy={}, reason="teacher_blocked_local_correction", detail="Teacher force-call did not complete; run a local bounded correction cycle.")
            payload["artifact"] = str(OVERRIDE_PATH.relative_to(ROOT))
    return payload


def recent_teacher_blocked_correction(max_age_seconds: int = 7200) -> bool:
    for path in (
        REPORTS / "autonomy_cycle_watchdog_correction.json",
        REPORTS / "autonomy_cycle_last.json",
    ):
        report = read_json(path)
        if not report or report.get("ok") is not True:
            continue
        if str(get_path(report, ["decision", "reason"], "")) != "teacher_blocked_local_correction":
            continue
        created = parse_time(report.get("created_utc"))
        if created and time.time() - created <= max_age_seconds:
            return True
    return False


def run_candidate_gate() -> bool:
    expected_family, expected_card = effective_frontier_target()
    frontier_report = pressure_report_for(expected_family, expected_card)
    run_candidate_evidence_profile(expected_family, expected_card, frontier_report)
    command = [
        sys.executable,
        "scripts/candidate_promotion_gate.py",
        "--runtime-report",
        "reports/preflight_cuda_rollout_smoke.json",
        "--out",
        "reports/candidate_promotion_gate.json",
    ]
    if frontier_report:
        command.extend(["--frontier-report", frontier_report])
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=180)
    return result.returncode in (0, 1)


def active_code_repair_evidence(state: dict[str, Any], expected_family: str, expected_card: str) -> dict[str, Any]:
    frontier_report = pressure_report_for(expected_family, expected_card)
    frontier = read_json(ROOT / frontier_report) if frontier_report else {}
    embedded = get_path(frontier, ["metrics", "code_repair_organism"], {})
    if isinstance(embedded, dict) and embedded.get("ran"):
        return {
            "policy": "project_theseus_local_code_repair_organism_v1",
            "card_id": expected_card,
            "source": f"active_pressure_report:{frontier_report}",
            "summary": embedded,
            "external_inference_calls": int(frontier.get("external_inference_calls") or 0),
        }
    standalone = state.get("code_repair_organism") if isinstance(state.get("code_repair_organism"), dict) else {}
    if standalone:
        result = dict(standalone)
        result["source"] = "latest_standalone_report"
        return result
    return {}


def run_candidate_bottleneck_reducer() -> bool:
    command = [
        sys.executable,
        "scripts/candidate_bottleneck_reducer.py",
        "--fix",
        "--out",
        "reports/candidate_bottleneck_reducer.json",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=1800)
    return result.returncode == 0


def refresh_alignment_reports() -> bool:
    expected_family, expected_card = effective_frontier_target()
    frontier_report = pressure_report_for(expected_family, expected_card)
    evidence_command = candidate_evidence_profile_command(expected_family, expected_card, frontier_report)
    candidate_command = [
        sys.executable,
        "scripts/candidate_promotion_gate.py",
        "--runtime-report",
        "reports/preflight_cuda_rollout_smoke.json",
        "--out",
        "reports/candidate_promotion_gate.json",
    ]
    if frontier_report:
        candidate_command.extend(["--frontier-report", frontier_report])
    commands = []
    if evidence_command:
        commands.append(evidence_command)
    commands.extend(
        [
            candidate_command,
            [
            sys.executable,
            "scripts/octopus_router.py",
            "--router-head-report",
            "reports/octopus_router_head_report.json",
            "--router-head-eval",
            "reports/octopus_router_head_eval.json",
            "--out",
            "reports/octopus_router_report.json",
        ],
        [
            sys.executable,
            "scripts/architecture_experiment_governor.py",
            "--out",
            "reports/architecture_experiment_governance.json",
            "--markdown-out",
            "reports/architecture_experiment_governance.md",
        ],
        [
            sys.executable,
            "scripts/arm_sucker_registry.py",
            "--out",
            "reports/arm_sucker_registry.json",
            "--markdown-out",
            "reports/arm_sucker_registry.md",
        ],
        [
            sys.executable,
            "scripts/arm_transfer_planner.py",
            "--out",
            "reports/arm_transfer_plan.json",
            "--markdown-out",
            "reports/arm_transfer_plan.md",
        ],
        [
            sys.executable,
            "scripts/transfer_artifact_builder.py",
            "--plan",
            "reports/arm_transfer_plan.json",
            "--out",
            "reports/arm_transfer_artifacts.json",
        ],
        ]
    )
    ok = True
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=300)
        accepted = result.returncode == 0
        if (
            len(command) > 1
            and str(command[1]).replace("\\", "/") == "scripts/candidate_promotion_gate.py"
            and result.returncode == 1
            and (REPORTS / "candidate_promotion_gate.json").exists()
        ):
            accepted = True
        ok = ok and accepted
    return ok


def run_candidate_evidence_profile(expected_family: str, expected_card: str, frontier_report: str) -> bool:
    command = candidate_evidence_profile_command(expected_family, expected_card, frontier_report)
    if not command:
        return True
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def candidate_evidence_profile_command(expected_family: str, expected_card: str, frontier_report: str) -> list[str]:
    if expected_family != "coding_local_sandbox" or not expected_card or not frontier_report:
        return []
    return [
        sys.executable,
        "scripts/candidate_evidence_profile.py",
        "--frontier-family",
        expected_family,
        "--card-id",
        expected_card,
        "--seed",
        str(seed_from_pressure_report(frontier_report) or 1),
        "--frontier-report",
        frontier_report,
        "--out",
        "reports/training_ratchet_candidate_evidence_profile.json",
    ]


def seed_from_pressure_report(path: str) -> int:
    match = re.search(r"_seed(\d+)", str(path))
    return int(match.group(1)) if match else 0


def refresh_training_budget_plan() -> bool:
    frontier = read_json(REPORTS / "frontier_policy_status.json")
    expected_family, expected_card = effective_frontier_target()
    command = [
        sys.executable,
        "scripts/training_budget_planner.py",
        "--profile",
        str(frontier.get("selected_profile") or frontier.get("requested_profile") or "inner_loop"),
        "--frontier-family",
        str(expected_family or frontier.get("frontier_family") or "drone_rl"),
        "--pressure-card-id",
        str(expected_card or frontier.get("pressure_card_id") or ""),
        "--mode",
        "auto",
        "--out",
        "reports/training_budget_plan.json",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60)
    return result.returncode == 0


def run_code_residual_forge() -> bool:
    expected_family, expected_card = effective_frontier_target()
    frontier_report = pressure_report_for(expected_family, expected_card)
    command = [
        sys.executable,
        "scripts/code_residual_forge.py",
        "--out",
        "reports/code_residual_forge.json",
        "--transfer-out",
        "reports/code_transfer_artifacts.json",
        "--rotation-out",
        "reports/code_frontier_rotation.json",
        "--trace-out",
        "reports/code_repair_traces.jsonl",
    ]
    if expected_card:
        command.extend(["--active-card-id", expected_card])
    if frontier_report:
        command.extend(["--active-report", frontier_report])
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def run_code_repair_organism() -> bool:
    _family, card_id = effective_frontier_target()
    card_id = str(card_id or "source_livecodebench")
    card = read_json(ROOT / "benchmarks" / "cards" / f"{card_id}.json")
    source_path = str(card.get("resource_pantry_path") or card.get("staged_path") or "")
    command = [
        sys.executable,
        "scripts/local_code_repair_organism.py",
        "--card-id",
        card_id,
        "--seed",
        "14",
        "--source-path",
        source_path,
        "--transfer-artifacts",
        "reports/code_transfer_artifacts.json",
        "--out",
        f"reports/local_code_repair_organism_{safe_name(card_id)}_seed14.json",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def refresh_open_code_training_pantry() -> bool:
    command = [
        sys.executable,
        "scripts/open_code_training_pantry.py",
        "--root",
        "D:/ProjectTheseus/training_data/open_code_pantry",
        "--repo-config",
        "configs/open_code_training_pantry_expanded.json",
        "--out",
        "reports/open_code_training_pantry.json",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=900)
    return result.returncode == 0


def run_sts_learning_forge() -> bool:
    command = [
        sys.executable,
        "scripts/sts_learning_forge.py",
        "--out",
        "reports/sts_learning_forge.json",
        "--out-data",
        "data/sts_learning/sts_code_streams_seed14.jsonl",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def run_sts_native_parallel_probe() -> bool:
    merged = ROOT / "data" / "sts_learning" / "sts_code_context_spaces_seed14.jsonl"
    if not merged.exists():
        run_cognitive_context_router()
    command = [
        sys.executable,
        "scripts/sts_native_parallel_probe.py",
        "--input",
        "data/sts_learning/sts_code_context_spaces_seed14.jsonl",
        "--out",
        "reports/sts_native_parallel_probe.json",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=360)
    return result.returncode == 0


def run_cognitive_context_router() -> bool:
    command = [
        sys.executable,
        "scripts/cognitive_context_router.py",
        "--policy",
        "configs/cognitive_context_policy.json",
        "--base-sts",
        "data/sts_learning/sts_code_streams_seed14.jsonl",
        "--out-data",
        "data/sts_learning/cognitive_context_spaces_seed14.jsonl",
        "--merged-out-data",
        "data/sts_learning/sts_code_context_spaces_seed14.jsonl",
        "--out",
        "reports/cognitive_context_router.json",
        "--markdown-out",
        "reports/cognitive_context_router.md",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def run_real_code_benchmark_graduation() -> bool:
    card_id = real_code_graduation_card_id()
    max_cases = str(real_code_public_case_budget(card_id))
    card_safe = safe_name(card_id)
    seed = "14"
    candidate_manifest = f"reports/student_code_candidates_{card_safe}_seed{seed}.jsonl"
    checkpoint = f"reports/student_code_lm_checkpoint_{card_safe}_seed{seed}.json"
    private_candidates = f"reports/code_lm_private_candidates_{card_safe}_seed{seed}.jsonl"
    public_tasks = f"reports/code_lm_public_tasks_{card_safe}_seed{seed}.jsonl"
    rust_report = f"reports/code_lm_closure_rust_{card_safe}_seed{seed}.json"
    code_lm_report = f"reports/code_lm_closure_{card_safe}_seed{seed}.json"
    sts_input = f"reports/code_lm_sts_conditioning_input_{card_safe}_seed{seed}.jsonl"
    sts_generations = f"reports/code_lm_sts_public_generations_{card_safe}_seed{seed}.jsonl"
    sts_checkpoint = f"reports/code_lm_sts_conditioning_checkpoint_{card_safe}_seed{seed}.json"
    sts_report = f"reports/code_lm_sts_conditioning_report_{card_safe}_seed{seed}.json"

    commands = [
        [
            sys.executable,
            "scripts/open_code_training_pantry.py",
            "--root",
            "D:/ProjectTheseus/training_data/open_code_pantry",
            "--repo-config",
            "configs/open_code_training_pantry_expanded.json",
            "--out",
            "reports/open_code_training_pantry.json",
        ],
        [
            sys.executable,
            "scripts/code_lm_closure.py",
            "--public-cards",
            card_id,
            "--seed",
            seed,
            "--private-count",
            "300",
            "--max-extra-private-train",
            "120",
            "--max-residual-private-train",
            "160",
            "--max-public-cases-per-card",
            max_cases,
            "--hv-dim",
            "128",
            "--max-vocab",
            "128",
            "--epochs",
            "1",
            "--candidates-per-task",
            "2",
            "--max-rust-work-steps",
            str(code_lm_work_step_budget(int(max_cases))),
            "--rust-timeout-seconds",
            "0",
            "--public-timeout-seconds",
            "0",
            "--sts-timeout-seconds",
            "0",
            "--public-task-manifest-out",
            public_tasks,
            "--public-candidate-out",
            candidate_manifest,
            "--checkpoint-out",
            checkpoint,
            "--private-candidate-out",
            private_candidates,
            "--rust-report-out",
            rust_report,
            "--sts-conditioning-input-out",
            sts_input,
            "--sts-generation-out",
            sts_generations,
            "--sts-conditioning-checkpoint-out",
            sts_checkpoint,
            "--sts-conditioning-report-out",
            sts_report,
            "--skip-public-calibration",
            "--out",
            code_lm_report,
        ],
        [
            sys.executable,
            "scripts/real_code_benchmark_graduation.py",
            "--cards",
            card_id,
            "--seed",
            seed,
            "--max-cases-per-card",
            max_cases,
            "--skip-student-candidate-generation",
            "--student-candidate-manifest",
            candidate_manifest,
            "--out",
            f"reports/student_learning_baseline_{card_safe}_seed{seed}_real_code.json",
            "--trace-out",
            f"reports/student_learning_baseline_{card_safe}_seed{seed}_traces.jsonl",
            "--transfer-artifact-out",
            f"reports/transfer_artifacts/code/student_learning_baseline_{card_safe}_seed{seed}_transfer_artifact.json",
        ],
        [
            str(ROOT / "target" / "release" / "symliquid-cli.exe"),
            "train-code-ranker",
            "--use-cuda-readout",
            "--candidate-manifest",
            candidate_manifest,
            "--trace-in",
            f"reports/student_learning_baseline_{card_safe}_seed{seed}_traces.jsonl",
            "--seed",
            seed,
            "--model-out",
            "reports/student_neural_code_checkpoint.json",
            "--candidate-out",
            "reports/student_learning_code_candidates.jsonl",
            "--training-examples-out",
            "reports/student_learning_training_examples.jsonl",
            "--transfer-artifact-out",
            "reports/transfer_artifacts/code/student_neural_learning_closure_transfer_artifact.json",
            "--code-transfer-artifacts",
            "reports/code_transfer_artifacts.json",
            "--out",
            "reports/student_learning_closure.json",
        ],
        [
            sys.executable,
            "scripts/real_code_benchmark_graduation.py",
            "--cards",
            card_id,
            "--seed",
            seed,
            "--max-cases-per-card",
            max_cases,
            "--skip-student-candidate-generation",
            "--student-candidate-manifest",
            candidate_manifest,
            "--out",
            "reports/real_code_benchmark_graduation.json",
            "--trace-out",
            "reports/real_code_benchmark_traces.jsonl",
            "--transfer-artifact-out",
            "reports/transfer_artifacts/code/real_code_benchmark_graduation_transfer_artifact.json",
        ],
    ]
    ok = True
    for command in commands:
        result = run_real_code_fix_command(command)
        ok = ok and result.returncode == 0
        if result.returncode != 0:
            break
    return ok


def real_code_graduation_card_id() -> str:
    scheduler = read_json(REPORTS / "broad_code_calibration_scheduler.json")
    selected = scheduler.get("selected") if isinstance(scheduler.get("selected"), dict) else {}
    scheduled_card = str(selected.get("card_id") or "")
    if bool(selected.get("can_run_real_code")) and scheduled_card in {"source_mbpp", "source_human_eval", "source_evalplus"}:
        return scheduled_card
    _family, card_id = effective_frontier_target()
    public_task_cards = ["source_mbpp", "source_human_eval", "source_evalplus"]
    if card_id in public_task_cards:
        return card_id
    learned_manifest = REPORTS / "student_learning_code_candidates.jsonl"
    for candidate in public_task_cards:
        if learned_manifest_matches_card(learned_manifest, candidate):
            return candidate
    return "source_mbpp"


def real_code_public_case_budget(card_id: str) -> int:
    # Public calibration must be large enough to expose transfer failures. Small smoke slices
    # are useful for adapter checks, but they are not promotion-facing learning evidence.
    if card_id in {"source_human_eval", "source_mbpp", "source_evalplus"}:
        return 32
    return 32


def code_lm_work_step_budget(max_cases: int) -> int:
    # Work steps are the primary duration control; subprocess timeouts are only recovery fuses.
    return max(120_000, int(max_cases) * 4_000)


def effective_frontier_target() -> tuple[str, str]:
    curriculum = read_json(REPORTS / "benchmaxx_curriculum.json")
    next_frontier = curriculum.get("next_frontier") if isinstance(curriculum.get("next_frontier"), dict) else {}
    family = str(next_frontier.get("family") or "")
    runner = str(next_frontier.get("runner_family") or "")
    runner_map = {
        "minecraft_rl_local": "minecraft_rl",
        "drone_rl_local": "drone_rl",
        "coding_local_sandbox": "coding_local_sandbox",
        "web_agent_local": "web_agent_local",
        "transfer_eval_local": "transfer_eval",
    }
    mapped = runner_map.get(runner, family)
    card = str(next_frontier.get("recommended_env") or "")
    if bool(next_frontier.get("runnable_now")) and mapped:
        return mapped, card
    frontier = read_json(REPORTS / "frontier_policy_status.json")
    return str(frontier.get("frontier_family") or ""), str(frontier.get("pressure_card_id") or "")


def pressure_report_for(expected_family: str, expected_card: str) -> str:
    frontier = read_json(REPORTS / "frontier_policy_status.json")
    active = frontier.get("frontier") if isinstance(frontier.get("frontier"), dict) else {}
    report = str(active.get("best_report") or "")
    if report and (not expected_card or expected_card in report):
        return report
    ledger = read_json(REPORTS / "benchmark_ledger.json")
    if isinstance(ledger, list):
        for row in ledger:
            if not isinstance(row, dict):
                continue
            candidate_report = str(row.get("best_report") or "")
            name = str(row.get("benchmark_name") or "")
            if not candidate_report:
                continue
            if expected_family and expected_family not in name and expected_family not in candidate_report:
                continue
            if expected_card and expected_card not in name and expected_card not in candidate_report:
                continue
            return candidate_report
    if expected_card:
        matches = sorted(
            REPORTS.glob(f"pressure_{safe_name(expected_card)}_seed*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return str(matches[0].relative_to(ROOT)).replace("\\", "/")
    return ""


def learned_manifest_matches_card(path: Path, card_id: str) -> bool:
    if not path.exists() or not card_id:
        return False
    prefix = f"{card_id}_"
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("candidate_source") or "") not in {
            "student_learning_checkpoint_v1",
            "student_neural_checkpoint_v1",
            "student_token_generator_checkpoint_v1",
            "student_code_lm_checkpoint_v1",
        }:
            continue
        if str(row.get("task_id") or "").startswith(prefix):
            return True
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        if str(provenance.get("card_id") or "") == card_id:
            return True
    return False


def run_legacy_real_code_benchmark_graduation() -> bool:
    command = [
        sys.executable,
        "scripts/real_code_benchmark_graduation.py",
        "--cards",
        "source_evalplus,source_human_eval,source_bigcodebench,source_livecodebench",
        "--seed",
        "14",
        "--out",
        "reports/real_code_benchmark_graduation.json",
        "--trace-out",
        "reports/real_code_benchmark_traces.jsonl",
        "--transfer-artifact-out",
        "reports/transfer_artifacts/code/real_code_benchmark_graduation_transfer_artifact.json",
    ]
    result = run_real_code_fix_command(command)
    return result.returncode == 0


def run_real_code_fix_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=REAL_CODE_FIX_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stderr = (
            stderr
            + f"\nwatchdog real-code correction timed out after {REAL_CODE_FIX_TIMEOUT_SECONDS}s"
        )
        return subprocess.CompletedProcess(command, 124, stdout=stdout, stderr=stderr)


def run_self_edit_experiment_lane() -> bool:
    command = [sys.executable, "scripts/self_edit_experiment_lane.py", "--out", "reports/self_edit_experiment_lane.json"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def run_long_horizon_memory_probe() -> bool:
    command = [sys.executable, "scripts/long_horizon_memory_probe.py", "--out", "reports/long_horizon_memory_probe.json"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def refresh_virtual_context_memory() -> bool:
    command = [
        sys.executable,
        "scripts/virtual_context_memory.py",
        "--task",
        "Project Theseus watchdog VCM refresh",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=180)
    return result.returncode == 0


def run_synthetic_benchmark_factory() -> bool:
    command = [
        sys.executable,
        "scripts/synthetic_benchmark_factory.py",
        "--policy",
        "configs/synthetic_benchmark_policy.json",
        "--write-cards",
        "--out",
        "reports/synthetic_benchmark_factory.json",
        "--markdown-out",
        "reports/synthetic_benchmark_factory.md",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def run_multi_stream_trace_factory() -> bool:
    command = [
        sys.executable,
        "scripts/multi_stream_trace_factory.py",
        "--policy",
        "configs/multi_stream_policy.json",
        "--write-cards",
        "--out",
        "reports/multi_stream_trace_factory.json",
        "--markdown-out",
        "reports/multi_stream_trace_factory.md",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def run_multi_stream_monitorability_probe() -> bool:
    command = [sys.executable, "scripts/multi_stream_monitorability_probe.py", "--out", "reports/multi_stream_monitorability_probe.json"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def run_multi_stream_candidate_gate() -> bool:
    command = [sys.executable, "scripts/multi_stream_candidate_gate.py", "--out", "reports/multi_stream_candidate_gate.json"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def refresh_code_lm_shard_strategy_audit() -> bool:
    command = [
        sys.executable,
        "scripts/code_lm_shard_strategy_audit.py",
        "--slug",
        "private_pressure_private_recovery_cuda_program_loop_v6",
        "--shard-count",
        "16",
        "--out",
        "reports/code_lm_shard_strategy_audit.json",
        "--markdown-out",
        "reports/code_lm_shard_strategy_audit.md",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode in {0, 2}


def refresh_code_lm_train_once_fanout_plan() -> bool:
    command = [
        sys.executable,
        "scripts/code_lm_train_once_fanout.py",
        "--slug",
        TRAIN_ONCE_FANOUT_SLUG,
        "--public-cards",
        TRAIN_ONCE_FANOUT_PUBLIC_CARDS,
        "--max-public-cases-per-card",
        "32",
        "--out",
        "reports/code_lm_train_once_fanout_plan.json",
        "--markdown-out",
        "reports/code_lm_train_once_fanout_plan.md",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def refresh_genesis_kernel() -> bool:
    command = [
        sys.executable,
        "scripts/genesis_kernel.py",
        "ingest-theseus",
        "--out",
        "reports/genesis_kernel/report.json",
        "--bundle-dir",
        "reports/genesis_kernel/latest_release",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def refresh_reality_manipulator() -> bool:
    command = [
        sys.executable,
        "scripts/reality_manipulator.py",
        "--out",
        "reports/reality_manipulator.json",
        "--markdown-out",
        "reports/reality_manipulator.md",
        "--bundle-dir",
        "reports/reality_manipulator/latest_world",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def refresh_viea_autonomy_spine() -> bool:
    command = [
        sys.executable,
        "scripts/viea_autonomy_spine.py",
        "--max-steps",
        "64",
        "--timeout-seconds",
        "7200",
        "--out",
        "reports/viea_autonomy_spine.json",
        "--markdown-out",
        "reports/viea_autonomy_spine.md",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=7200)
    return result.returncode == 0


def refresh_personality_runtime_audit() -> bool:
    command = [
        sys.executable,
        "scripts/personality_runtime_audit.py",
        "--refresh",
        "--out",
        "reports/personality_runtime_audit.json",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=900)
    return result.returncode == 0


def refresh_learning_scoreboard() -> bool:
    command = [
        sys.executable,
        "scripts/learning_scoreboard.py",
        "--out",
        "reports/learning_scoreboard.json",
        "--markdown-out",
        "reports/learning_scoreboard.md",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def refresh_a_plus_scorecard() -> bool:
    command = [
        sys.executable,
        "scripts/a_plus_operating_scorecard.py",
        "--out",
        "reports/a_plus_operating_scorecard.json",
        "--markdown-out",
        "reports/a_plus_operating_scorecard.md",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def refresh_broad_transfer_matrix() -> bool:
    command = [
        sys.executable,
        "scripts/broad_transfer_matrix.py",
        "--min-public-tasks",
        "32",
        "--out",
        "reports/broad_transfer_matrix.json",
        "--markdown-out",
        "reports/broad_transfer_matrix.md",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=180)
    return result.returncode in (0, 1, 2)


def refresh_broad_code_calibration_scheduler() -> bool:
    command = [
        sys.executable,
        "scripts/broad_code_calibration_scheduler.py",
        "--min-public-tasks",
        "32",
        "--out",
        "reports/broad_code_calibration_scheduler.json",
        "--markdown-out",
        "reports/broad_code_calibration_scheduler.md",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def run_broad_code_calibration_step() -> bool:
    matrix_ok = refresh_broad_transfer_matrix()
    scheduler_ok = refresh_broad_code_calibration_scheduler()
    scheduler = read_json(REPORTS / "broad_code_calibration_scheduler.json")
    selected = scheduler.get("selected") if isinstance(scheduler.get("selected"), dict) else {}
    adapter_ok = True
    if str(selected.get("action") or "") == "stage_or_upgrade_public_task_adapter":
        adapter_ok = stage_public_code_adapter(str(selected.get("card_id") or ""))
        refresh_broad_code_calibration_scheduler()
        scheduler = read_json(REPORTS / "broad_code_calibration_scheduler.json")
        selected = scheduler.get("selected") if isinstance(scheduler.get("selected"), dict) else {}
    ran_real_code = True
    if bool(selected.get("can_run_real_code")):
        ran_real_code = run_real_code_benchmark_graduation()
        refresh_broad_transfer_matrix()
    refresh_learning_scoreboard()
    refresh_overnight_learning_readiness()
    return bool(matrix_ok and scheduler_ok and adapter_ok and ran_real_code)


def stage_public_code_adapter(card_id: str) -> bool:
    if card_id == "source_evalplus":
        result = subprocess.run(
            [
                sys.executable,
                "scripts/stage_evalplus_public_data.py",
                "--out",
                "reports/stage_evalplus_public_data.json",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=180,
        )
        return result.returncode == 0
    # Other public-code cards need explicit adapter implementations rather than
    # spending cycles on small or loader-only slices.
    return False


def refresh_cell_lifecycle() -> bool:
    command = [
        sys.executable,
        "scripts/cell_lifecycle.py",
        "--out",
        "reports/cell_lifecycle.json",
        "--markdown-out",
        "reports/cell_lifecycle.md",
        "--prune-plan-out",
        "reports/cell_lifecycle_prune_plan.json",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def refresh_overnight_learning_readiness() -> bool:
    command = [
        sys.executable,
        "scripts/overnight_learning_readiness.py",
        "--out",
        "reports/overnight_learning_readiness.json",
        "--markdown-out",
        "reports/overnight_learning_readiness.md",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode in (0, 2)


def run_hive_work_board_status() -> bool:
    command = [
        sys.executable,
        "scripts/hive_work_board_executor.py",
        "--status",
        "--resume",
        "--max-tasks",
        "1",
        "--out",
        "reports/hive_work_board_executor.json",
        "--markdown-out",
        "reports/hive_work_board_executor.md",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=180)
    return result.returncode in (0, 2)


def run_service_process_hygiene() -> bool:
    command = [
        sys.executable,
        "scripts/service_process_hygiene.py",
        "--fix",
        "--out",
        "reports/service_process_hygiene.json",
        "--markdown-out",
        "reports/service_process_hygiene.md",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode in (0, 2)


def refresh_efficiency_upstream_evidence() -> bool:
    """Refresh cheap upstream evidence before judging whole-system efficiency.

    system_efficiency_audit imports performance_optimizer bottlenecks directly.
    Refreshing the resource/scheduler/optimizer chain here prevents stale
    worker-plan or resource-throttle rows from turning into overnight goals.
    """
    commands = [
        [
            sys.executable,
            "scripts/resource_governor.py",
            "--profile",
            "smoke",
            "--out",
            "reports/resource_governor.json",
        ],
        [
            sys.executable,
            "scripts/hive_scheduler.py",
            "--worker-chunks",
            "--out",
            "reports/hive_scheduler.json",
        ],
        [
            sys.executable,
            "scripts/runtime_bottleneck_optimizer_worker_chunk_plan.py",
            "--out",
            "reports/runtime_bottleneck_optimizer_worker_chunk_plan.json",
            "--markdown-out",
            "reports/runtime_bottleneck_optimizer_worker_chunk_plan.md",
            "--lease-out",
            "reports/runtime_bottleneck_optimizer_worker_chunk_leases.jsonl",
        ],
        [
            sys.executable,
            "scripts/performance_optimizer.py",
            "--out",
            "reports/performance_optimizer.json",
            "--markdown-out",
            "reports/performance_optimizer.md",
        ],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=180)
        if result.returncode not in (0, 2):
            return False
    return True


def run_system_efficiency_audit() -> bool:
    if not refresh_efficiency_upstream_evidence():
        return False
    command = [
        sys.executable,
        "scripts/system_efficiency_audit.py",
        "--out",
        "reports/system_efficiency_audit.json",
        "--markdown-out",
        "reports/system_efficiency_audit.md",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode in (0, 2)


def run_autonomy_rotation_governor() -> bool:
    command = [
        sys.executable,
        "scripts/autonomy_rotation_governor.py",
        "--execute",
        "--out",
        "reports/autonomy_rotation_governor_v2.json",
        "--markdown-out",
        "reports/autonomy_rotation_governor_v2.md",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=180)
    return result.returncode in (0, 2)


def run_grammar_suckers() -> bool:
    command = [
        sys.executable,
        "scripts/grammar_suckers.py",
        "--config",
        "configs/grammar_suckers.json",
        "--out",
        "reports/grammar_suckers.json",
        "--markdown-out",
        "reports/grammar_suckers.md",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def run_deterministic_taming_stack() -> bool:
    command = [
        sys.executable,
        "scripts/deterministic_taming_stack.py",
        "--run-cargo-check",
        "--out",
        "reports/deterministic_taming_stack.json",
        "--markdown-out",
        "reports/deterministic_taming_stack.md",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def run_code_residual_curriculum() -> bool:
    command = [
        sys.executable,
        "scripts/code_residual_curriculum.py",
        "--trace-in",
        "reports/real_code_benchmark_traces.jsonl",
        "--private-out",
        "D:/ProjectTheseus/training_data/residual_code_curriculum/private_train/residual_code_lm_tasks.jsonl",
        "--out",
        "reports/code_residual_curriculum.json",
        "--markdown-out",
        "reports/code_residual_curriculum.md",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def run_sts_repair_ablation() -> bool:
    command = [
        sys.executable,
        "scripts/sts_repair_ablation.py",
        "--real-code-report",
        "reports/real_code_benchmark_graduation.json",
        "--trace-in",
        "reports/real_code_benchmark_traces.jsonl",
        "--out",
        "reports/sts_repair_ablation.json",
        "--markdown-out",
        "reports/sts_repair_ablation.md",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def run_architecture_guidance_loop() -> bool:
    command = [
        sys.executable,
        "scripts/architecture_guidance_loop.py",
        "--out",
        "reports/architecture_guidance_loop.json",
        "--markdown-out",
        "reports/architecture_guidance_loop.md",
        "--experiments-out",
        "reports/architecture_guided_experiments.json",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode == 0


def refresh_teacher_budget_audit() -> bool:
    command = [
        sys.executable,
        "scripts/teacher_budget_audit.py",
        "--out",
        "reports/teacher_budget_audit.json",
        "--markdown-out",
        "reports/teacher_budget_audit.md",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return result.returncode in (0, 1)


def restart_sparkstream_stack() -> bool:
    commands = [
        [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            "scripts\\start_theseus_hive.ps1",
            "-Restart",
            "-NoDashboard",
        ],
        [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            "scripts\\start_sparkstream.ps1",
            "-StartDaemon",
            "-Profile",
            "inner_loop",
            "-Execute",
            "-AllowTeacher",
            "-AllowNetworkFetch",
            "-DurationHours",
            "10",
            "-Port",
            "8787",
            "-Restart",
        ],
    ]
    ok = True
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=300)
        ok = ok and result.returncode == 0
    return ok
