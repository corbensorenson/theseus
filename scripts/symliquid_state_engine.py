"""Materialize SymLiquid state into live VIEA routing hints.

SymLiquid should be the compact recurrent state substrate where state and
dynamics matter. This report turns the existing slot map into concrete route
weights and control hints consumed by the action executor and teacher runner.
It is local, deterministic, and does not claim student-learning progress.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from code_lm_private_rows import default_no_admissible_repair_policy_jsonl


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def training_data_root() -> Path:
    configured = os.environ.get("THESEUS_TRAINING_DATA_ROOT", "").strip()
    if configured:
        return Path(configured)
    if sys.platform.startswith("win"):
        return Path("D:/ProjectTheseus/training_data")
    return ROOT / "data" / "training_data"


def training_data_path(*parts: str) -> Path:
    return training_data_root().joinpath(*parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="reports/symliquid_state_engine_queue.json")
    parser.add_argument("--out", default="reports/symliquid_state_engine.json")
    parser.add_argument("--markdown-out", default="reports/symliquid_state_engine.md")
    args = parser.parse_args()

    state = observe(args)
    slots = build_slots(state)
    weights = action_kind_weights(state, slots)
    code_lm_args = code_lm_closure_args(state)
    edge_v2 = edge_contract_v2_state(state)
    candidate_coverage = candidate_coverage_state(state)
    sts_control = sts_decoder_control_state(state)
    gates = [
        gate("state_slots_loaded", len(slots) >= 7, len(slots)),
        gate("action_weights_materialized", bool(weights), weights),
        gate("executor_influence_declared", "viea_action_executor" in influence_targets(slots), influence_targets(slots)),
        gate("repo_repair_bridge_available", bool(code_lm_args.get("repo_repair_private_train_jsonl")), code_lm_args),
        gate("candidate_coverage_wall_materialized", candidate_coverage["observed"], candidate_coverage, severity="soft"),
        gate(
            "candidate_coverage_routes_private_pressure",
            (not candidate_coverage["wall_present"]) or weights.get("recover_code_candidate_coverage", 0.0) >= 8.0,
            {"candidate_coverage": candidate_coverage, "recover_code_candidate_coverage": weights.get("recover_code_candidate_coverage")},
            severity="hard",
        ),
        gate("sts_decoder_control_contract_materialized", sts_control["observed"], sts_control, severity="hard"),
        gate(
            "sts_decoder_control_has_named_consumers",
            sts_control["named_consumer_count"] >= 3,
            sts_control,
            severity="hard",
        ),
        gate(
            "sts_decoder_control_consumed_by_code_lm_args",
            bool(code_lm_args.get("sts_decoder_control_policy_jsonl")),
            code_lm_args,
            severity="hard",
        ),
    ]
    payload = {
        "policy": "project_theseus_symliquid_state_engine_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(row["passed"] for row in gates) else "YELLOW",
        "summary": {
            "slot_count": len(slots),
            "active_slot_count": sum(1 for row in slots if row.get("status") == "active"),
            "strongest_action_kind": max(weights, key=weights.get) if weights else "",
            "broad_transfer_gap": broad_gap(state),
            "repo_repair_trace_count": get_path(state["repo_learner"], ["summary", "validated_private_trace_count"], 0),
            "edge_contract_v2_private_rows": edge_v2["private_rows"],
            "edge_contract_v2_ready_for_public_calibration": edge_v2["ready_for_public_calibration"],
            "edge_contract_v2_control_reason": edge_v2["reason"],
            "candidate_coverage_ready_for_public_calibration": candidate_coverage["ready_for_public_calibration"],
            "candidate_coverage_control_reason": candidate_coverage["reason"],
            "sts_decoder_control_observed": sts_control["observed"],
            "sts_decoder_control_reason": sts_control["reason"],
            "sts_decoder_control_rows": sts_control["control_rows_written"],
            "sts_decoder_control_named_consumer_count": sts_control["named_consumer_count"],
            "sts_decoder_control_requires_same_seed_comparator": sts_control["force_same_seed_non_sts_comparator"],
            "asi_wall_next_primary_action": get_path(state["asi_governor"], ["summary", "next_primary_action"], ""),
            "asi_wall_hard_blocker_count": get_path(state["asi_governor"], ["summary", "hard_blocker_count"], 0),
            "asi_wall_public_calibration_allowed": get_path(state["asi_governor"], ["summary", "public_calibration_allowed"], False),
            "a_plus_overall_grade": get_path(state["a_plus_scorecard"], ["summary", "overall_grade"], ""),
            "promotion_evidence": False,
            "external_inference_calls": 0,
        },
        "state_slots": slots,
        "action_kind_weights": weights,
        "route_hints": route_hints(state, weights),
        "code_lm_closure_args": code_lm_args,
        "sts_decoder_control": sts_control,
        "influence_targets": influence_targets(slots),
        "gates": gates,
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 0 if payload["trigger_state"] in {"GREEN", "YELLOW"} else 2


def observe(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "queue": read_json(resolve(args.queue), {}),
        "feedback_queue": read_json(REPORTS / "feedback_action_queue.json", {}),
        "broad": read_json(REPORTS / "broad_transfer_matrix.json", read_json(REPORTS / "broad_transfer_closure.json", {})),
        "repo": read_json(REPORTS / "repo_repair_main_curriculum.json", {}),
        "repo_learner": read_json(REPORTS / "viea_repo_repair_learner.json", {}),
        "teacher": read_json(REPORTS / "teacher_architect_closure.json", {}),
        "workflow_tools": read_json(REPORTS / "workflow_tool_compiler_v2.json", {}),
        "sts": read_json(REPORTS / "sts_repair_ablation.json", {}),
        "executor": read_json(REPORTS / "viea_action_executor.json", {}),
        "scheduler": read_json(REPORTS / "high_transfer_curriculum_scheduler.json", {}),
        "conversation_hard_v2": read_json(REPORTS / "high_transfer_multi_turn_conversation_hard_v2.json", {}),
        "conversation_hard_v3": read_json(REPORTS / "high_transfer_multi_turn_conversation_hard_v3.json", {}),
        "board_game": read_json(REPORTS / "board_game_learned_policy.json", {}),
        "long_horizon_tool_use": read_json(REPORTS / "high_transfer_long_horizon_tool_use.json", {}),
        "edge_contract_v2_verifier": read_json(REPORTS / "edge_contract_v2_private_verifier.json", {}),
        "edge_contract_v2_pressure": read_json(REPORTS / "high_transfer_edge_contract_v2_private_residual_curriculum_code_residual_curriculum.json", {}),
        "decoder_ablation_gate": read_json(REPORTS / "decoder_v2_private_ablation_gate.json", {}),
        "execution_shape_smoke": latest_execution_shape_candidate_coverage_report(),
        "candidate_bottleneck": read_json(REPORTS / "candidate_bottleneck_reducer.json", {}),
        "candidate_floor_v2": read_json(REPORTS / "high_transfer_candidate_floor_v2_private_residual_curriculum_code_residual_curriculum.json", {}),
        "no_admissible_residuals": read_json(REPORTS / "no_admissible_candidate_residuals.json", {}),
        "sts_decoder_control": read_json(REPORTS / "sts_decoder_control_contract.json", {}),
        "cross_domain_capsules": read_json(REPORTS / "cross_domain_sts_capsules.json", {}),
        "asi_governor": read_json(REPORTS / "asi_wall_breaker_governor.json", {}),
        "a_plus_scorecard": read_json(REPORTS / "a_plus_operating_scorecard.json", {}),
    }


def latest_execution_shape_candidate_coverage_report() -> dict[str, Any]:
    candidates: list[tuple[float, dict[str, Any]]] = []
    paths = [REPORTS / "execution_shape_private_ablation_smoke.json"]
    paths.extend(REPORTS.glob("execution_shape_candidate_coverage*.json"))
    for path in paths:
        name = path.name
        if name.endswith("_rust.json") or name.endswith("_checkpoint.json"):
            continue
        report = read_json(path, {})
        if not isinstance(report, dict) or not report:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        report.setdefault("source_report_path", str(path))
        candidates.append((mtime, report))
    if not candidates:
        return {}

    def score(item: tuple[float, dict[str, Any]]) -> tuple[int, float, float, float]:
        mtime, report = item
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        trigger_bonus = 1 if report.get("trigger_state") == "GREEN" else 0
        pass_rate = safe_float(summary.get("learned_token_decoder_pass_rate")) or 0.0
        no_admissible = safe_float(summary.get("learned_token_decoder_no_admissible_candidate_rate"))
        no_admissible = 1.0 if no_admissible is None else no_admissible
        return (trigger_bonus, pass_rate, -no_admissible, mtime)

    return max(candidates, key=score)[1]


def build_slots(state: dict[str, Any]) -> list[dict[str, Any]]:
    queue = state["queue"].get("queue") if isinstance(state["queue"].get("queue"), list) else []
    slots = []
    for item in queue:
        if not isinstance(item, dict):
            continue
        capability = str(item.get("capability") or "")
        activation = activation_for(capability, state)
        slots.append(
            {
                **item,
                "activation": activation,
                "influences": influences_for(capability),
                "routing_signal": routing_signal(capability, state, activation),
            }
        )
    return slots


def activation_for(capability: str, state: dict[str, Any]) -> float:
    gap = broad_gap(state)
    if capability == "residual_clusters":
        return clamp(0.4 + gap)
    if capability == "sts_conditioning":
        sts_delta = float(get_path(state["broad"], ["summary", "real_public_sts_delta"], 0.0) or 0.0)
        return clamp(0.6 if sts_delta <= 0.0 else 0.4 + min(0.4, sts_delta))
    if capability == "repo_repair_state":
        tasks = int(get_path(state["repo"], ["summary", "task_count"], 0) or 0)
        traces = int(get_path(state["repo_learner"], ["summary", "validated_private_trace_count"], 0) or 0)
        return clamp(0.3 + min(tasks, 96) / 160 + min(traces, 96) / 320)
    if capability == "command_route_memory":
        ready = int(get_path(state["executor"], ["summary", "ready_action_count"], 0) or 0)
        return clamp(0.3 + min(ready, 20) / 30)
    if capability == "tool_selection":
        return 0.45
    if capability == "long_autonomy_state":
        return 0.5
    if capability == "small_control_policies":
        return clamp(0.35 + gap)
    if capability == "candidate_coverage":
        coverage = candidate_coverage_state(state)
        return clamp(0.2 + coverage["severity"])
    return 0.25


def influences_for(capability: str) -> list[str]:
    table = {
        "command_route_memory": ["viea_action_executor", "viea_command_executor"],
        "residual_clusters": ["feedback_action_queue", "teacher_architect_experiment_runner", "code_lm_closure"],
        "tool_selection": ["viea_action_executor", "workflow_tool_compiler_v2"],
        "sts_conditioning": ["sts_repair_ablation", "code_lm_closure"],
        "sts_decoder_control": ["code_lm_closure", "decoder_v2_private_ablation_gate", "hive_work_board_executor"],
        "repo_repair_state": ["viea_repo_repair_learner", "code_lm_closure"],
        "long_autonomy_state": ["autonomy_cycle", "autonomy_watchdog"],
        "small_control_policies": ["benchmaxx_curriculum", "viea_action_executor"],
        "candidate_coverage": ["code_lm_closure", "decoder_v2_private_ablation_gate", "hive_work_board_executor"],
    }
    return table.get(capability, [])


def routing_signal(capability: str, state: dict[str, Any], activation: float) -> str:
    if capability == "residual_clusters" and activation >= 0.5:
        return "boost_private_semantic_residual_training"
    if capability == "repo_repair_state" and activation >= 0.5:
        return "boost_repo_repair_trace_checkpoint"
    if capability == "sts_conditioning" and activation >= 0.5:
        return "boost_same_seed_sts_ablation"
    if capability == "small_control_policies" and broad_gap(state) > 0:
        return "keep_promotion_blocked_until_broad_floor_closes"
    if capability == "candidate_coverage" and candidate_coverage_state(state)["wall_present"]:
        return "boost_candidate_coverage_recovery_private_gate"
    return "observe"


def action_kind_weights(state: dict[str, Any], slots: list[dict[str, Any]]) -> dict[str, float]:
    gap = broad_gap(state)
    loader_only = get_path(state["broad"], ["summary", "loader_only_cards"], []) or []
    teacher_experiments = int(get_path(state["teacher"], ["summary", "experiment_count"], 0) or 0)
    repo_tasks = int(get_path(state["repo"], ["summary", "task_count"], 0) or 0)
    sts_causal_missing = any(
        "sts_not_causal_on_card" in (row.get("blockers") or [])
        for row in (state["broad"].get("rows") if isinstance(state["broad"].get("rows"), list) else [])
        if isinstance(row, dict)
    )
    edge_v2 = edge_contract_v2_state(state)
    capsule_rows = int(get_path(state["cross_domain_capsules"], ["summary", "sts_row_count"], 0) or 0)
    conversation_v3_cases = int(get_path(state["conversation_hard_v3"], ["summary", "case_count"], 0) or 0)
    conversation_v3_accuracy = float(get_path(state["conversation_hard_v3"], ["summary", "accuracy"], 0.0) or 0.0)
    conversation_v3_graduated = bool(get_path(state["conversation_hard_v3"], ["summary", "graduated"], False))
    conversation_accuracy = conversation_v3_accuracy or float(get_path(state["conversation_hard_v2"], ["summary", "accuracy"], 0.0) or 0.0)
    board_policy_rows = int(get_path(state["board_game"], ["summary", "policy_train_row_count"], 0) or 0)
    tool_use_pass = float(get_path(state["long_horizon_tool_use"], ["summary", "pass_rate"], 0.0) or 0.0)
    tool_use_cases = int(get_path(state["long_horizon_tool_use"], ["summary", "case_count"], 0) or 0)
    candidate_coverage = candidate_coverage_state(state)
    sts_control = sts_decoder_control_state(state)
    asi_next = str(get_path(state["asi_governor"], ["summary", "next_primary_action"], "") or "")
    asi_hard_blockers = int(get_path(state["asi_governor"], ["summary", "hard_blocker_count"], 0) or 0)
    asi_public_allowed = bool(get_path(state["asi_governor"], ["summary", "public_calibration_allowed"], False))
    a_plus_domains = state["a_plus_scorecard"].get("domains") if isinstance(state["a_plus_scorecard"].get("domains"), dict) else {}
    breadth_score = float(get_path(a_plus_domains, ["breadth_cross_domain_learning", "score"], 0.0) or 0.0)
    autonomy_score = float(get_path(a_plus_domains, ["autonomy_unattended_operation", "score"], 0.0) or 0.0)
    return {
        "recover_code_candidate_coverage": round(candidate_coverage["weight"], 4),
        "train_edge_contract_v2_private_gate": round(edge_v2["weight"], 4),
        "train_private_semantic_residual_family": round(5.0 + gap * 20.0, 4),
        "refresh_cross_domain_sts_capsules": 3.5 if capsule_rows <= 0 else 0.75,
        "train_repo_repair_trace_checkpoint": round(3.0 + min(repo_tasks, 96) / 24.0, 4),
        "run_long_horizon_tool_use": round(6.0 if tool_use_cases < 64 else (3.5 + gap * 4.0 if tool_use_pass < 1.0 else 1.0), 4),
        "run_board_game_self_play": round(3.0 if board_policy_rows <= 0 else 1.0 + min(capsule_rows, 256) / 256.0, 4),
        "run_hard_conversation_frontier": round(5.5 if (not conversation_v3_graduated or conversation_v3_cases < 256) else max(0.75, 1.5 - conversation_accuracy), 4),
        "repair_autonomy_rotation": round(4.0 if 0.0 < autonomy_score < 0.9 else 0.5, 4),
        "broaden_cross_domain_transfer": round(4.0 if 0.0 < breadth_score < 0.9 else 1.0, 4),
        "expand_public_adapter_clean_slice": 4.0 if loader_only else 1.0,
        "run_same_seed_sts_repair_ablation": 7.0
        if (sts_causal_missing or sts_control["force_same_seed_non_sts_comparator"])
        else 1.5,
        "refresh_sts_decoder_control_contract": 6.0 if not sts_control["observed"] else 0.75,
        "request_teacher_architecture_diagnosis": 4.0 if teacher_experiments and gap > 0.0 else 1.0,
        "run_fresh_private_code_closure": 10.0 if "fresh private_pressure_private_closure" in asi_next else 0.5,
        "refresh_decoder_v2_private_gate": 9.0 if "decoder_v2_private_ablation_gate" in asi_next else 0.5,
        "permit_single_public_calibration": 7.0 if asi_public_allowed else 0.0,
        "enforce_asi_wall_governor": 4.0 if asi_hard_blockers else 1.0,
        "refresh_symliquid_state_engine": 0.25,
        "write_repo_repair_tasks": 2.0 if repo_tasks < 48 else 0.75,
        "promote_regression_surface": 0.5,
        "renew_useful_tool": 0.1,
        "expire_stale_tool": 1.0,
    }


def route_hints(state: dict[str, Any], weights: dict[str, float]) -> list[dict[str, Any]]:
    hints = []
    for kind, weight in sorted(weights.items(), key=lambda item: item[1], reverse=True):
        hints.append(
            {
                "kind": kind,
                "weight": weight,
                "route": route_for_kind(kind),
                "public_data_rule": "public_benchmarks_calibration_only",
            }
        )
    return hints


def route_for_kind(kind: str) -> str:
    if kind in {
        "train_private_semantic_residual_family",
        "train_repo_repair_trace_checkpoint",
        "write_repo_repair_tasks",
        "train_edge_contract_v2_private_gate",
        "recover_code_candidate_coverage",
        "run_fresh_private_code_closure",
        "refresh_cross_domain_sts_capsules",
        "run_long_horizon_tool_use",
        "run_board_game_self_play",
        "run_hard_conversation_frontier",
        "broaden_cross_domain_transfer",
        "refresh_sts_decoder_control_contract",
    }:
        return "private_training_pressure"
    if kind in {"refresh_decoder_v2_private_gate", "enforce_asi_wall_governor"}:
        return "local_gate_or_state_refresh"
    if kind == "permit_single_public_calibration":
        return "public_calibration_or_ablation_only"
    if kind == "repair_autonomy_rotation":
        return "local_lifecycle_or_state"
    if kind in {"expand_public_adapter_clean_slice", "promote_regression_surface", "run_same_seed_sts_repair_ablation"}:
        return "public_calibration_or_ablation_only"
    if kind == "request_teacher_architecture_diagnosis":
        return "teacher_architect_proposal_only"
    return "local_lifecycle_or_state"


def code_lm_closure_args(state: dict[str, Any]) -> dict[str, Any]:
    repo_path = training_data_path("long_horizon_programming", "private_train", "repo_repair_code_lm_rows.jsonl")
    edge_v2_path = training_data_path(
        "high_transfer",
        "private_train",
        "edge_contract_v2_private_residual_curriculum_residual_code_lm_tasks.jsonl",
    )
    no_admissible_path = Path(default_no_admissible_repair_policy_jsonl())
    candidate_coverage = candidate_coverage_state(state)
    sts_control = sts_decoder_control_state(state)
    return {
        "repo_repair_private_train_jsonl": str(repo_path).replace("\\", "/") if repo_path.exists() else "",
        "edge_contract_v2_private_train_jsonl": str(edge_v2_path).replace("\\", "/") if edge_v2_path.exists() else "",
        "no_admissible_repair_policy_jsonl": str(no_admissible_path).replace("\\", "/") if no_admissible_path.exists() else "",
        "sts_decoder_control_contract": sts_control["control_contract_path"] if sts_control["observed"] else "",
        "sts_decoder_control_policy_jsonl": sts_control["control_rows_path"] if sts_control["control_rows_path_exists"] else "",
        "recommended_max_repo_repair_private_train": 1200,
        "recommended_max_edge_contract_v2_private_train": 960,
        "recommended_max_no_admissible_control_rows": 640,
        "recommended_max_sts_decoder_control_rows": 128,
        "candidate_coverage_recovery_v2": candidate_coverage,
        "sts_decoder_control_v1": sts_control,
        "decoder_control_hints": decoder_control_hints(candidate_coverage),
        "sts_control_hints": sts_control_hints(sts_control),
        "public_benchmarks": "visible_prompt_calibration_only",
    }


def edge_contract_v2_state(state: dict[str, Any]) -> dict[str, Any]:
    verifier = state.get("edge_contract_v2_verifier") if isinstance(state.get("edge_contract_v2_verifier"), dict) else {}
    pressure = state.get("edge_contract_v2_pressure") if isinstance(state.get("edge_contract_v2_pressure"), dict) else {}
    rows = int(get_path(verifier, ["summary", "private_rows"], 0) or get_path(pressure, ["summary", "edge_contract_v2_rows"], 0) or 0)
    ready = bool(verifier.get("ready_for_public_calibration"))
    closure_green = bool(
        get_path(verifier, ["summary", "closure_trigger_state"], "") == "GREEN"
        and get_path(verifier, ["summary", "closure_run_status"], "") == "completed"
    )
    gap = broad_gap(state)
    if ready:
        weight = 0.5
        reason = "edge_contract_v2_private_gate_ready"
    elif rows > 0 and gap > 0:
        weight = 8.0 + gap * 18.0
        reason = "edge_contract_v2_rows_need_private_closure_or_verifier"
    elif gap > 0:
        weight = 6.0 + gap * 12.0
        reason = "edge_contract_v2_pressure_missing_or_stale"
    else:
        weight = 0.5
        reason = "broad_floor_closed_or_no_pressure"
    return {
        "private_rows": rows,
        "ready_for_public_calibration": ready,
        "closure_green": closure_green,
        "weight": weight,
        "reason": reason,
    }


def sts_decoder_control_state(state: dict[str, Any]) -> dict[str, Any]:
    """Expose STS decoder control as a live routing/control contract."""

    report = state.get("sts_decoder_control") if isinstance(state.get("sts_decoder_control"), dict) else {}
    consumers = get_path(report, ["consumer_contract", "consumers"], [])
    effects = get_path(report, ["consumer_contract", "effects"], [])
    decoder_hints = get_path(report, ["decoder_hints"], {})
    control_rows_path = str(report.get("control_rows_path") or "")
    control_contract_path = str(report.get("control_contract_path") or "reports/sts_decoder_control_contract.json")
    rows_abs = resolve(control_rows_path) if control_rows_path else Path("")
    rows_exist = bool(control_rows_path and rows_abs.exists() and rows_abs.stat().st_size > 0)
    observed = bool(report) and report.get("trigger_state") in {"GREEN", "YELLOW"} and rows_exist
    if observed:
        reason = "sts_decoder_control_contract_materialized_for_next_private_closure"
    elif report:
        reason = "sts_decoder_control_contract_present_but_rows_missing"
    else:
        reason = "sts_decoder_control_contract_missing"
    return {
        "observed": observed,
        "reason": reason,
        "trigger_state": report.get("trigger_state"),
        "control_contract_path": control_contract_path,
        "control_rows_path": control_rows_path,
        "control_rows_path_exists": rows_exist,
        "control_rows_written": int(get_path(report, ["control_rows_written"], 0) or 0),
        "named_consumer_count": len(consumers) if isinstance(consumers, list) else 0,
        "consumer_effect_count": len(effects) if isinstance(effects, list) else 0,
        "consumers": consumers if isinstance(consumers, list) else [],
        "effects": effects if isinstance(effects, list) else [],
        "force_same_seed_non_sts_comparator": bool(
            get_path(decoder_hints, ["force_same_seed_non_sts_comparator"], False)
        ),
        "prefer_sts_when_verifier_passes": bool(
            get_path(decoder_hints, ["prefer_sts_when_verifier_passes"], False)
        ),
        "sts_positive_same_seed_lift": bool(get_path(decoder_hints, ["sts_positive_same_seed_lift"], False)),
        "sts_coverage_non_regressive": bool(get_path(decoder_hints, ["sts_coverage_non_regressive"], False)),
        "sts_conditioning_regressed_candidate_coverage": bool(
            get_path(decoder_hints, ["sts_conditioning_regressed_candidate_coverage"], False)
        ),
        "min_non_sts_comparator_task_coverage": safe_float(
            get_path(decoder_hints, ["min_non_sts_comparator_task_coverage"], None)
        ),
        "min_sts_conditioned_task_coverage": safe_float(
            get_path(decoder_hints, ["min_sts_conditioned_task_coverage"], None)
        ),
        "no_admissible_rate_floor": safe_float(get_path(decoder_hints, ["no_admissible_rate_floor"], None)),
        "targeted_capability_families": get_path(report, ["targeted_capability_families"], []),
        "promotion_evidence_required": get_path(
            report,
            ["consumer_contract", "promotion_evidence_required"],
            "same_seed_sts_vs_non_sts_delta_after_completed_private_closure",
        ),
        "public_calibration_locked": True,
    }


def candidate_coverage_state(state: dict[str, Any]) -> dict[str, Any]:
    """Materialize learned-token candidate sparsity as a live control signal."""

    gate_report = state.get("decoder_ablation_gate") if isinstance(state.get("decoder_ablation_gate"), dict) else {}
    smoke = state.get("execution_shape_smoke") if isinstance(state.get("execution_shape_smoke"), dict) else {}
    bottleneck = state.get("candidate_bottleneck") if isinstance(state.get("candidate_bottleneck"), dict) else {}
    floor = state.get("candidate_floor_v2") if isinstance(state.get("candidate_floor_v2"), dict) else {}
    no_admissible = state.get("no_admissible_residuals") if isinstance(state.get("no_admissible_residuals"), dict) else {}
    no_admissible_summary = no_admissible.get("summary") if isinstance(no_admissible.get("summary"), dict) else {}
    no_admissible_outputs = no_admissible.get("outputs") if isinstance(no_admissible.get("outputs"), dict) else {}

    public_eligible = safe_float(get_path(gate_report, ["summary", "public_eligible_task_coverage"], None))
    public_actual = safe_float(get_path(gate_report, ["summary", "public_actual_token_task_coverage"], None))
    public_no_admissible = safe_float(get_path(gate_report, ["summary", "public_no_admissible_task_rate"], None))
    ready = bool(gate_report.get("ready_for_public_calibration"))
    learned_pass = safe_float(get_path(smoke, ["summary", "learned_token_decoder_pass_rate"], None))
    learned_no_admissible = safe_float(get_path(smoke, ["summary", "learned_token_decoder_no_admissible_candidate_rate"], None))
    learned_public_gate = bool(get_path(smoke, ["summary", "learned_token_public_gate_ready"], False))

    observed = any(
        value is not None
        for value in [public_eligible, public_actual, public_no_admissible, learned_pass, learned_no_admissible]
    )
    public_eligible_value = public_eligible if public_eligible is not None else 0.0
    public_actual_value = public_actual if public_actual is not None else 0.0
    public_no_admissible_value = public_no_admissible if public_no_admissible is not None else 1.0
    learned_pass_value = learned_pass if learned_pass is not None else 0.0
    learned_no_admissible_value = learned_no_admissible if learned_no_admissible is not None else 1.0

    blockers = []
    if public_eligible_value < 0.60:
        blockers.append("public_eligible_task_coverage_below_0.60")
    if public_actual_value < 0.60:
        blockers.append("public_actual_token_task_coverage_below_0.60")
    if public_no_admissible_value > 0.25:
        blockers.append("public_no_admissible_task_rate_above_0.25")
    if learned_pass_value < 0.70:
        blockers.append("learned_execution_shape_pass_rate_below_0.70")
    if learned_no_admissible_value > 0.25:
        blockers.append("learned_execution_shape_no_admissible_rate_above_0.25")

    if ready and learned_public_gate and not blockers:
        reason = "candidate_coverage_gate_ready"
        wall_present = False
    elif observed:
        reason = "candidate_coverage_wall_requires_private_recovery"
        wall_present = True
    else:
        reason = "candidate_coverage_not_observed_yet"
        wall_present = True

    severity = max(
        0.0,
        0.60 - public_eligible_value,
        0.60 - public_actual_value,
        public_no_admissible_value - 0.25,
        0.70 - learned_pass_value,
        learned_no_admissible_value - 0.25,
    )
    rows = int(get_path(floor, ["summary", "rows"], 0) or get_path(floor, ["summary", "task_count"], 0) or 0)
    if rows <= 0:
        rows = int(get_path(bottleneck, ["summary", "candidate_floor_v2_rows"], 0) or 0)
    residual_rows = int(get_path(no_admissible_summary, ["residual_record_count"], 0) or 0)
    policy_rows = int(get_path(no_admissible_summary, ["policy_row_count"], 0) or 0)
    weight = 0.5 if not wall_present else 9.0 + severity * 10.0
    if rows <= 0 and wall_present:
        weight += 2.0
    if residual_rows > 0 and wall_present:
        weight += min(2.0, residual_rows / 80.0)
    return {
        "observed": observed,
        "ready_for_public_calibration": bool(ready and learned_public_gate and not blockers),
        "wall_present": wall_present,
        "reason": reason,
        "severity": round(severity, 6),
        "weight": round(weight, 6),
        "public_actual_token_task_coverage": round(public_actual_value, 6),
        "public_eligible_task_coverage": round(public_eligible_value, 6),
        "public_no_admissible_task_rate": round(public_no_admissible_value, 6),
        "learned_token_decoder_pass_rate": round(learned_pass_value, 6),
        "learned_token_decoder_no_admissible_candidate_rate": round(learned_no_admissible_value, 6),
        "candidate_floor_private_rows": rows,
        "no_admissible_residual_record_count": residual_rows,
        "no_admissible_policy_row_count": policy_rows,
        "no_admissible_residuals_out": no_admissible_outputs.get("residual_jsonl_out") or "",
        "no_admissible_policy_rows_out": no_admissible_outputs.get("policy_rows_out") or "",
        "top_no_admissible_rejection_reasons": no_admissible_summary.get("top_rejection_reasons") or {},
        "top_no_admissible_missing_capability_families": no_admissible_summary.get("top_missing_capability_families") or {},
        "blockers": blockers,
        "promotion_rule": "public_4card_locked_until_private_candidate_coverage_gate_green",
    }


def decoder_control_hints(candidate_coverage: dict[str, Any]) -> list[str]:
    hints = [
        "public_calibration_locked_until_private_candidate_coverage_gate_green",
        "public_benchmarks_visible_prompt_calibration_only",
    ]
    if candidate_coverage.get("wall_present"):
        hints.extend(
            [
                "prioritize_parser_ast_constrained_learned_generation",
                "prioritize_no_admissible_candidate_residual_rows",
                "require_exact_signature_and_argument_use",
                "rank_non_vacuous_minimal_code_before_generic_scaffolds",
                "emit_rejection_reason_histograms_and_sample_rejected_bodies",
            ]
        )
    return hints


def sts_control_hints(sts_control: dict[str, Any]) -> list[str]:
    hints = [
        "sts_capsules_must_have_named_decoder_consumer",
        "public_calibration_locked_until_same_seed_sts_delta_exists",
    ]
    if not sts_control.get("observed"):
        hints.append("run_sts_causal_decoder_ablation_to_materialize_control_contract")
        return hints
    hints.extend(
        [
            "pass_sts_decoder_control_policy_jsonl_into_code_lm_closure",
            "emit_same_seed_non_sts_comparator_candidates",
            "measure_sts_conditioned_vs_non_sts_candidate_distribution_delta",
        ]
    )
    if (
        sts_control.get("sts_conditioning_regressed_candidate_coverage")
        or not sts_control.get("prefer_sts_when_verifier_passes")
    ):
        hints.extend(
            [
                "demote_sts_conditioned_rank_bias_until_positive_same_seed_lift",
                "prefer_same_seed_non_sts_candidate_when_both_verify",
                "repair_sts_candidate_coverage_before_promotion",
            ]
        )
        for family in sts_control.get("targeted_capability_families") or []:
            hints.append(f"repair_sts_decoder_family:{family}")
    else:
        hints.append("rank_sts_conditioned_candidates_only_when_verifier_passes")
        for family in sts_control.get("targeted_capability_families") or []:
            hints.append(f"boost_sts_decoder_family:{family}")
    return hints[:18]


def influence_targets(slots: list[dict[str, Any]]) -> list[str]:
    out = set()
    for slot in slots:
        for target in slot.get("influences") or []:
            out.add(str(target))
    return sorted(out)


def broad_gap(state: dict[str, Any]) -> float:
    summary = state["broad"].get("summary") if isinstance(state["broad"].get("summary"), dict) else {}
    explicit_gap = get_path(state["broad"], ["summary", "aggregate_floor_gap"], None)
    if explicit_gap is not None:
        return float(explicit_gap or 0.0)
    rate = summary.get("real_public_pass_rate")
    if rate is None:
        rate = summary.get("aggregate_pass_rate")
    try:
        return max(0.0, 0.70 - float(rate or 0.0))
    except Exception:
        return 0.0


def gate(name: str, passed: bool, evidence: Any, *, severity: str = "hard") -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# SymLiquid State Engine",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- slots: `{summary.get('slot_count')}`",
        f"- strongest_action_kind: `{summary.get('strongest_action_kind')}`",
        f"- broad_transfer_gap: `{summary.get('broad_transfer_gap')}`",
        "",
        "## Route Hints",
        "",
    ]
    for row in payload.get("route_hints", [])[:12]:
        lines.append(f"- `{row.get('kind')}` weight `{row.get('weight')}` -> `{row.get('route')}`")
    lines.append("")
    return "\n".join(lines)


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def get_path(data: Any, path: list[str], default: Any = None) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
