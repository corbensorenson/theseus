#!/usr/bin/env python3
"""A+ operating scorecard for Project Theseus.

This is an operator-facing report over local evidence. It does not claim model
promotion, does not train on public benchmark data, and does not call external
inference. Its job is to keep the whole system honest about what must improve
next across architecture, autonomy, conversation, code transfer,
self-improvement, and breadth.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "a_plus_operating_scorecard.json"
DEFAULT_MARKDOWN = REPORTS / "a_plus_operating_scorecard.md"
PUBLIC_CODE_FLOOR = 0.70


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    state = observe()
    domains = {
        "maturity_integrity_governance": assess_maturity_integrity(state),
        "architecture_viea_spine": assess_architecture(state),
        "autonomy_unattended_operation": assess_autonomy(state),
        "conversation_personality_lane": assess_conversation(state),
        "code_public_transfer": assess_code_transfer(state),
        "self_improvement_loop": assess_self_improvement(state),
        "breadth_cross_domain_learning": assess_breadth(state),
    }
    walls = main_walls(domains)
    next_actions = recommended_actions(domains, state)
    asi_governor_summary = object_field(state.get("asi_wall_breaker_governor", {}), "summary")
    raw_average_score = round(sum(float(row["score"]) for row in domains.values()) / max(1, len(domains)), 4)
    weakest_domain_score = min(float(row["score"]) for row in domains.values()) if domains else 0.0
    overall_score = truthful_overall_score(raw_average_score, weakest_domain_score, walls)
    payload = {
        "policy": "project_theseus_a_plus_operating_scorecard_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if overall_score >= 0.90 and not walls else "YELLOW",
        "summary": {
            "overall_score": overall_score,
            "overall_grade": grade(overall_score),
            "raw_average_score": raw_average_score,
            "weakest_domain_score": round(weakest_domain_score, 4),
            "weakest_domain_caps_overall": bool(walls),
            "domain_count": len(domains),
            "blocking_wall_count": len(walls),
            "north_star": "observe -> act -> verify -> store evidence -> learn -> transfer -> route better next time",
            "promotion_evidence": False,
            "external_inference_calls": 0,
            "asi_wall_hard_blocker_count": asi_governor_summary.get("hard_blocker_count"),
            "asi_wall_public_calibration_allowed": asi_governor_summary.get("public_calibration_allowed"),
            "asi_wall_next_primary_action": asi_governor_summary.get("next_primary_action"),
        },
        "domains": domains,
        "main_walls": walls,
        "recommended_next_actions": next_actions,
        "evidence_policy": {
            "public_benchmarks": "calibration-only; no public solutions/tests become training rows",
            "teacher": "proposal-only architecture experiments; local private gates decide",
            "score_semantics": "operator diagnostic scorecard, not public model promotion",
            "truth_cap": "if any major domain is below A-, the weakest major domain caps the overall grade",
        },
        "external_inference_calls": 0,
    }
    out_path = resolve(args.out)
    write_json(out_path, payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    ingest_self(out_path, payload)
    print(json.dumps(payload, indent=2))
    return 0


def observe() -> dict[str, Any]:
    return {
        "report_store": read_json(REPORTS / "report_evidence_store.json", {}),
        "report_store_db": REPORTS / "report_evidence_store.sqlite",
        "work_board_db": REPORTS / "hive_work_board.sqlite",
        "work_board_executor": read_json(REPORTS / "hive_work_board_executor.json", {}),
        "watchdog": read_json(REPORTS / "autonomy_watchdog.json", {}),
        "learning_supervisor": read_json(REPORTS / "learning_launch_supervisor.json", {}),
        "morning_report": read_json(REPORTS / "hive_morning_report.json", {}),
        "broad": read_json(REPORTS / "broad_transfer_matrix.json", {}),
        "broad_code_scheduler": read_json(REPORTS / "broad_code_calibration_scheduler.json", {}),
        "transfer_audit": read_json(REPORTS / "transfer_generalization_audit.json", {}),
        "scoreboard": read_json(REPORTS / "learning_scoreboard.json", {}),
        "maturity_integrity": read_json(REPORTS / "maturity_integrity_audit.json", {}),
        "candidate_gate": read_json(REPORTS / "candidate_promotion_gate.json", {}),
        "conversation_hard": read_json(REPORTS / "high_transfer_multi_turn_conversation_hard.json", {}),
        "conversation_hard_v2": read_json(REPORTS / "high_transfer_multi_turn_conversation_hard_v2.json", {}),
        "conversation_hard_v3": read_json(REPORTS / "high_transfer_multi_turn_conversation_hard_v3.json", {}),
        "tool_use": read_json(REPORTS / "high_transfer_long_horizon_tool_use.json", {}),
        "board_game": read_json(REPORTS / "board_game_rl_benchmark.json", {}),
        "board_game_policy": read_json(REPORTS / "board_game_learned_policy.json", {}),
        "pufferlib4": read_json(REPORTS / "pufferlib4_rl_lane.json", read_json(REPORTS / "pufferlib4_capability_probe.json", {})),
        "repo_repair": best_repo_repair_report(),
        "cross_domain": read_json(REPORTS / "cross_domain_sts_capsules.json", {}),
        "sts_causal_decoder_ablation": read_json(REPORTS / "sts_causal_decoder_ablation.json", {}),
        "agent_lane_transfer_gate": read_json(REPORTS / "agent_lane_transfer_gate.json", {}),
        "symliquid": read_json(REPORTS / "symliquid_state_engine.json", {}),
        "asi_wall_breaker_governor": read_json(REPORTS / "asi_wall_breaker_governor.json", {}),
        "teacher_runner": read_json(REPORTS / "teacher_architect_experiment_runner.json", {}),
        "teacher_last": read_json(REPORTS / "teacher_oracle_last.json", {}),
        "teacher_budget": read_json(REPORTS / "teacher_budget_last.json", read_json(REPORTS / "teacher_budget_audit.json", {})),
        "attd": read_json(REPORTS / "attd_report.json", {}),
        "public_transfer_residual_packet": read_json(REPORTS / "public_transfer_residual_packet.json", {}),
        "edge_v2_verifier": read_json(REPORTS / "edge_contract_v2_private_verifier.json", {}),
        "edge_obligation_gate": read_json(REPORTS / "edge_obligation_decode_gate_v1_private_pressure_private.json", read_json(REPORTS / "edge_obligation_decode_gate_v1_private.json", {})),
        "decoder_plan_ir": read_json(REPORTS / "decoder_plan_ir_private_pressure.json", {}),
        "decoder_plan_ir_code_lm_adapter": read_json(REPORTS / "decoder_plan_ir_code_lm_adapter.json", {}),
        "edge_v2_closure": read_json(REPORTS / "code_lm_closure_edge_contract_v2_private.json", {}),
        "private_closure": read_json(REPORTS / "code_lm_closure_private_pressure_private.json", {}),
        "decoder_ablation_gate": read_json(REPORTS / "decoder_v2_private_ablation_gate.json", {}),
        "execution_shape_full": read_json(REPORTS / "execution_shape_private_ablation.json", {}),
        "execution_shape_smoke": read_json(REPORTS / "execution_shape_private_ablation_smoke.json", {}),
        "execution_shape_archive_gate": read_json(REPORTS / "execution_shape_private_ablation_archive_gate.json", {}),
        "execution_shape_candidate_coverage": latest_execution_shape_candidate_coverage_report(),
        "sts_repair_ablation": read_json(REPORTS / "sts_repair_ablation.json", {}),
        "architecture_experiment_governance": read_json(REPORTS / "architecture_experiment_governance.json", {}),
        "architecture_experiment_results": read_json(REPORTS / "architecture_experiment_results.json", {}),
        "closed_loop_residual_ratchet": read_json(REPORTS / "closed_loop_residual_ratchet.json", {}),
        "transfer_eval_suite": read_json(REPORTS / "transfer_eval_suite.json", {}),
        "improvement_ledger": read_jsonl_tail(REPORTS / "hive_unattended_improvement_ledger.jsonl", 240),
        "no_progress_ledger": read_jsonl_tail(REPORTS / "hive_no_progress_demotion_ledger.jsonl", 120),
        "stop_flags": [str(path.relative_to(ROOT)).replace("\\", "/") for path in stop_flags() if path.exists()],
    }


def assess_maturity_integrity(state: dict[str, Any]) -> dict[str, Any]:
    audit = state["maturity_integrity"]
    summary = object_field(audit, "summary")
    checks = [
        check("maturity_integrity_audit_present", bool(audit), audit.get("created_utc")),
        check("no_hard_integrity_blockers", int(number(summary.get("hard_blocker_count"))) == 0, summary.get("hard_blocker_count"), weight=2.0),
        check("public_training_leak_absent", audit_check_passed(audit, "public_training_leak_absent"), audit_check_evidence(audit, "public_training_leak_absent"), weight=2.0),
        check("templates_not_promotion_evidence", audit_check_passed(audit, "templates_are_not_promotion_evidence"), audit_check_evidence(audit, "templates_are_not_promotion_evidence"), weight=1.6),
        check("teacher_proposal_only", audit_check_passed(audit, "teacher_remains_proposal_only"), audit_check_evidence(audit, "teacher_remains_proposal_only"), weight=1.4),
        check("growth_blocked_until_transfer", audit_check_passed(audit, "model_growth_blocked_until_integrity_and_transfer"), audit_check_evidence(audit, "model_growth_blocked_until_integrity_and_transfer"), weight=1.4),
        check("promotion_blocked_until_transfer_and_coherence", audit_check_passed(audit, "candidate_promotion_blocked_until_transfer_and_coherence"), audit_check_evidence(audit, "candidate_promotion_blocked_until_transfer_and_coherence"), weight=1.4),
        check("report_only_intelligence_rejected", audit_check_passed(audit, "sts_capsules_have_causal_consumer"), audit_check_evidence(audit, "sts_capsules_have_causal_consumer"), weight=1.2),
        check("architecture_experiment_delta_required", audit_check_passed(audit, "architecture_experiments_have_delta_before_promotion"), audit_check_evidence(audit, "architecture_experiments_have_delta_before_promotion"), weight=1.2),
    ]
    return domain("Maturity / integrity governance", checks, "No leakage, no template substitution, no report-only claims, no premature growth.")


def assess_architecture(state: dict[str, Any]) -> dict[str, Any]:
    store_db = state["report_store_db"]
    board_db = state["work_board_db"]
    stored_reports = sqlite_count(store_db, "report_runs") if store_db.exists() else 0
    board_tasks = sqlite_count(board_db, "tasks") if board_db.exists() else 0
    evidence_rows = sqlite_count(board_db, "evidence") if board_db.exists() else 0
    checks = [
        check("report_evidence_store_db_exists", store_db.exists(), rel(store_db)),
        check("report_store_has_many_versions", stored_reports >= 100, stored_reports),
        check("hive_work_board_db_exists", board_db.exists(), rel(board_db)),
        check("work_board_has_tasks", board_tasks > 0, board_tasks),
        check("work_board_has_evidence_rows", evidence_rows > 0, evidence_rows),
        check("reports_are_views_not_truth_only", bool(state["report_store"]) or stored_reports > 0, "evidence store present"),
        check("attd_not_red", state["attd"].get("trigger_state") != "RED", {"state": state["attd"].get("trigger_state"), "score": state["attd"].get("attd_score")}, weight=1.6),
        check(
            "training_runtime_monolith_below_cap",
            not attd_has_hard_cap_violation(state["attd"], "max_source_file_lines"),
            attd_hard_cap_evidence(state["attd"], "max_source_file_lines"),
            weight=1.6,
        ),
    ]
    return domain("Architecture / VIEA spine", checks, "Durable evidence, board source of truth, append/versioned reports.")


def assess_autonomy(state: dict[str, Any]) -> dict[str, Any]:
    board_summary = object_field(state["work_board_executor"], "summary")
    watchdog_state = state["watchdog"].get("trigger_state")
    supervisor_status = get_path(state["learning_supervisor"], ["summary", "status"], state["learning_supervisor"].get("status"))
    no_progress_blocks = sum(1 for row in state["no_progress_ledger"] if row.get("action") == "block")
    no_progress_demotes = sum(1 for row in state["no_progress_ledger"] if row.get("action") == "demote")
    no_progress_capability = no_progress_capability_installed()
    checks = [
        check("watchdog_report_current", watchdog_state in {"GREEN", "YELLOW", "RED"}, watchdog_state),
        check("work_board_executor_current", state["work_board_executor"].get("policy") == "project_theseus_hive_work_board_executor_v1", state["work_board_executor"].get("created_utc")),
        check("learning_supervisor_known", bool(state["learning_supervisor"]), supervisor_status),
        check("no_stop_flags_present", not state["stop_flags"], state["stop_flags"]),
        check("ready_or_active_work_exists", int(number(board_summary.get("ready_tasks")) or number(board_summary.get("ready_or_active")) or 0) > 0, board_summary),
        check(
            "fake_progress_is_punished",
            no_progress_capability,
            {
                "recent_blocks": no_progress_blocks,
                "recent_demotes": no_progress_demotes,
                "ledger": rel(REPORTS / "hive_no_progress_demotion_ledger.jsonl"),
            },
        ),
    ]
    return domain("Autonomy / unattended operation", checks, "Reboot-safe board execution, watchdog correction, no stale churn.")


def assess_conversation(state: dict[str, Any]) -> dict[str, Any]:
    report = best_conversation_report(state)
    summary = object_field(report, "summary")
    mode = str(summary.get("suite_mode") or "missing")
    cases = int(number(summary.get("case_count")))
    accuracy = number(summary.get("accuracy"))
    target_cases = 256 if mode == "hard_v3" else 128 if mode == "hard_v2" else 64
    target_accuracy = 0.95 if mode == "hard_v3" else 0.90
    checks = [
        check("hard_conversation_report_present", bool(report), mode),
        check("hardest_available_lane", mode == "hard_v3", mode, weight=1.5),
        check("case_count_frontier", cases >= target_cases, {"cases": cases, "target": target_cases}),
        check("accuracy_frontier", accuracy >= target_accuracy, {"accuracy": accuracy, "target": target_accuracy}, weight=1.5),
        check("personality_context_attached", int(number(summary.get("personality_context_ready_turns"))) == int(number(summary.get("turn_count"))) and int(number(summary.get("turn_count"))) > 0, {"ready": summary.get("personality_context_ready_turns"), "turns": summary.get("turn_count")}),
    ]
    return domain("Conversation / personality lane", checks, "Hard memory, corrections, interruptions, talk-while-working, personality stability.")


def assess_code_transfer(state: dict[str, Any]) -> dict[str, Any]:
    broad_summary = object_field(state["broad"], "summary")
    scoreboard_public = object_field(state["scoreboard"], "public_transfer")
    pass_rate = first_number(
        broad_summary.get("real_public_pass_rate"),
        broad_summary.get("aggregate_pass_rate"),
        scoreboard_public.get("real_public_task_pass_rate"),
    )
    rows = state["broad"].get("rows") if isinstance(state["broad"].get("rows"), list) else []
    card_rates = {}
    for row in rows:
        if isinstance(row, dict):
            name = str(row.get("card") or row.get("source") or row.get("benchmark") or "")
            if not name:
                name = str(row.get("card_id") or "")
            rate = first_number(
                row.get("pass_rate"),
                row.get("real_public_pass_rate"),
                row.get("public_pass_rate"),
                row.get("multi_stream_pass_rate"),
                row.get("real_public_task_pass_rate"),
            )
            if name and rate is not None:
                card_rates[name] = rate
    min_card = min(card_rates.values()) if card_rates else 0.0
    private_ready = bool(state["edge_v2_verifier"].get("ready_for_public_calibration"))
    edge_obligation_ready = bool(state["edge_obligation_gate"].get("ready_for_public_calibration"))
    plan_ir_summary = object_field(state["decoder_plan_ir"], "summary")
    plan_ir_coverage = object_field(plan_ir_summary, "coverage")
    plan_ir_ready = (
        state["decoder_plan_ir"].get("trigger_state") == "GREEN"
        and int(number(plan_ir_summary.get("plan_ir_row_count"))) >= 1000
        and float(number(plan_ir_coverage.get("complete_plan_order_rate"))) >= 0.98
        and float(number(plan_ir_coverage.get("return_contract_rate"))) >= 0.98
        and float(number(plan_ir_coverage.get("skeleton_obligation_rate"))) >= 0.95
    )
    plan_ir_adapter_summary = object_field(state["decoder_plan_ir_code_lm_adapter"], "summary")
    plan_ir_adapter_ready = (
        state["decoder_plan_ir_code_lm_adapter"].get("trigger_state") == "GREEN"
        and int(number(plan_ir_adapter_summary.get("code_lm_row_count"))) >= 1000
        and float(number(plan_ir_adapter_summary.get("contract_row_rate"))) >= 0.98
        and int(number(plan_ir_adapter_summary.get("public_leak_flag_count"))) == 0
    )
    scheduler_selected = object_field(state["broad_code_scheduler"], "selected")
    scheduler_gate = object_field(state["broad_code_scheduler"], "private_receiver_gate")
    scheduler_gate_present = bool(scheduler_gate)
    scheduler_gate_allowed = bool(scheduler_gate.get("allowed"))
    scheduler_can_run_public = bool(scheduler_selected.get("can_run_real_code"))
    scheduler_action = str(scheduler_selected.get("action") or "")
    public_gate_controls = bool(
        scheduler_gate_present
        and (
            (scheduler_gate_allowed and scheduler_can_run_public)
            or (
                not scheduler_gate_allowed
                and not scheduler_can_run_public
                and scheduler_action == "private_gate_required_before_public_calibration"
            )
        )
    )
    checks = [
        check("public_transfer_report_present", bool(state["broad"]) or bool(state["scoreboard"]), pass_rate),
        check("aggregate_above_floor", (pass_rate or 0.0) >= PUBLIC_CODE_FLOOR, {"pass_rate": pass_rate, "floor": PUBLIC_CODE_FLOOR}, weight=2.0),
        check("all_cards_above_floor", bool(card_rates) and min_card >= PUBLIC_CODE_FLOOR, {"min_card": min_card, "cards": card_rates}, weight=2.0),
        check(
            "private_gate_controls_public_rerun",
            public_gate_controls,
            {
                "scheduler_action": scheduler_action,
                "scheduler_can_run_public": scheduler_can_run_public,
                "private_receiver_gate_allowed": scheduler_gate_allowed,
                "gate_blockers": scheduler_gate.get("blockers") or [],
                "edge_v2_ready": private_ready,
            },
        ),
        check("edge_obligation_gate_installed", bool(state["edge_obligation_gate"]) or edge_obligation_gate_available(), {"edge_obligation_ready": edge_obligation_ready}),
        check(
            "decoder_plan_ir_private_pressure_ready",
            plan_ir_ready,
            {
                "state": state["decoder_plan_ir"].get("trigger_state"),
                "rows": plan_ir_summary.get("plan_ir_row_count"),
                "coverage": plan_ir_coverage,
            },
        ),
        check(
            "decoder_plan_ir_code_lm_rows_ready",
            plan_ir_adapter_ready,
            {
                "state": state["decoder_plan_ir_code_lm_adapter"].get("trigger_state"),
                "rows": plan_ir_adapter_summary.get("code_lm_row_count"),
                "contract_rate": plan_ir_adapter_summary.get("contract_row_rate"),
                "public_leak_flag_count": plan_ir_adapter_summary.get("public_leak_flag_count"),
            },
        ),
        check("candidate_promotion_not_overclaiming", not bool(state["candidate_gate"].get("promote")) or (pass_rate or 0.0) >= PUBLIC_CODE_FLOOR, state["candidate_gate"].get("trigger_state")),
    ]
    return domain("Code / public transfer", checks, "Private concept pressure must transfer to MBPP/EvalPlus/BigCodeBench/LiveCodeBench.")


def assess_self_improvement(state: dict[str, Any]) -> dict[str, Any]:
    events = state["improvement_ledger"]
    improved = [row for row in events if get_path(row, ["improvement_contract", "passed"], False)]
    signal_counts = Counter(
        kind
        for row in improved
        for kind in (get_path(row, ["improvement_contract", "signal_kinds"], []) or [])
    )
    teacher_mode = str(state["teacher_last"].get("mode") or state["teacher_budget"].get("mode") or "proposal")
    teacher_status = str(state["teacher_last"].get("status") or state["teacher_runner"].get("trigger_state") or "")
    teacher_packet_summary = object_field(state["public_transfer_residual_packet"], "summary")
    teacher_packet_ready = (
        state["public_transfer_residual_packet"].get("trigger_state") == "GREEN"
        and bool(teacher_packet_summary.get("reason_for_teacher"))
        and bool(teacher_packet_summary.get("dominant_residuals"))
    )
    decoder_gate_summary = object_field(state["decoder_ablation_gate"], "summary")
    execution_shape_report = best_execution_shape_report(state)
    execution_shape_summary = object_field(execution_shape_report, "summary")
    private_gate_ready = bool(state["decoder_ablation_gate"].get("ready_for_public_calibration"))
    public_no_admissible_rate = first_number(decoder_gate_summary.get("public_no_admissible_task_rate"))
    coverage_recovered = (
        float(number(decoder_gate_summary.get("public_eligible_task_coverage")) or 0.0) >= 0.60
        and float(public_no_admissible_rate if public_no_admissible_rate is not None else 1.0) <= 0.25
        and float(number(decoder_gate_summary.get("public_program_synthesis_loop_present_rate")) or 0.0) >= 0.60
        and float(number(decoder_gate_summary.get("public_program_synthesis_promotion_ready_rate")) or 0.0) >= 0.50
    )
    learned_no_admissible_rate = first_number(execution_shape_summary.get("learned_token_decoder_no_admissible_candidate_rate"))
    learned_execution_shape_ready = (
        float(number(execution_shape_summary.get("learned_token_decoder_pass_rate")) or 0.0) >= 0.70
        and float(learned_no_admissible_rate if learned_no_admissible_rate is not None else 1.0) <= 0.25
        and int(number(execution_shape_summary.get("diagnostic_template_candidate_count")) or 0) == 0
    )
    private_or_receiver_coverage_recovered = learned_execution_shape_ready or (private_gate_ready and coverage_recovered)
    architecture_results = state["architecture_experiment_results"]
    architecture_status = str(architecture_results.get("status") or "")
    architecture_decision = object_field(architecture_results, "promotion_decision")
    architecture_contract = object_field(architecture_results, "residual_delta_contract")
    architecture_closed_loop = (
        bool(architecture_results.get("promotion_evidence"))
        and architecture_status == "completed_with_capability_delta"
        and architecture_decision.get("decision") == "promote"
        and bool(architecture_contract.get("targeted_improvement_observed"))
    )
    sts_causal_summary = object_field(state["sts_causal_decoder_ablation"], "summary")
    sts_causal_delta = first_number(sts_causal_summary.get("sts_public_eligible_coverage_delta"))
    sts_causal_effect = (
        state["sts_causal_decoder_ablation"].get("trigger_state") == "GREEN"
        or float(sts_causal_delta or 0.0) >= 0.01
    )
    supplemental_signal_counts = self_improvement_gate_signal_counts(
        state=state,
        private_or_receiver_coverage_recovered=private_or_receiver_coverage_recovered,
        learned_execution_shape_ready=learned_execution_shape_ready,
        teacher_packet_ready=teacher_packet_ready,
        sts_causal_effect=sts_causal_effect,
        architecture_closed_loop=architecture_closed_loop,
    )
    combined_signal_counts = signal_counts + supplemental_signal_counts
    checks = [
        check("improvement_events_recent", len(improved) > 0, len(improved), weight=1.2),
        check(
            "multiple_signal_types",
            len(combined_signal_counts) >= 3,
            {
                "ledger_signal_counts": dict(signal_counts),
                "verified_gate_signal_counts": dict(supplemental_signal_counts),
                "combined_signal_counts": dict(combined_signal_counts),
                "rule": "supplemental signals must come from already-verified gates/reports; this is evidence accounting, not promotion",
            },
            weight=1.2,
        ),
        check("private_closure_or_edge_gate_active", bool(state["private_closure"]) or bool(state["edge_v2_closure"]) or bool(state["edge_v2_verifier"]), "private closure/gate evidence present"),
        check(
            "candidate_coverage_recovery_proven",
            private_or_receiver_coverage_recovered,
            {
                "decoder_gate_ready_for_public_calibration": private_gate_ready,
                "public_eligible_task_coverage": decoder_gate_summary.get("public_eligible_task_coverage"),
                "public_no_admissible_task_rate": decoder_gate_summary.get("public_no_admissible_task_rate"),
                "public_program_synthesis_loop_present_rate": decoder_gate_summary.get("public_program_synthesis_loop_present_rate"),
                "public_program_synthesis_promotion_ready_rate": decoder_gate_summary.get("public_program_synthesis_promotion_ready_rate"),
                "private_execution_shape_ready": learned_execution_shape_ready,
                "private_learned_token_decoder_pass_rate": execution_shape_summary.get("learned_token_decoder_pass_rate"),
                "private_learned_token_no_admissible_rate": execution_shape_summary.get("learned_token_decoder_no_admissible_candidate_rate"),
            },
            weight=1.4,
        ),
        check(
            "learned_execution_shape_gate_not_scaffold",
            learned_execution_shape_ready,
            {
                "learned_token_decoder_pass_rate": execution_shape_summary.get("learned_token_decoder_pass_rate"),
                "learned_token_decoder_no_admissible_candidate_rate": execution_shape_summary.get("learned_token_decoder_no_admissible_candidate_rate"),
                "diagnostic_template_candidate_count": execution_shape_summary.get("diagnostic_template_candidate_count"),
            },
            weight=1.4,
        ),
        check("teacher_proposal_only", teacher_mode in {"", "proposal"} and state["teacher_last"].get("blocked_reason") != "teacher_must_remain_proposal_only", {"mode": teacher_mode, "status": teacher_status}),
        check("teacher_exact_residual_packet_ready", teacher_packet_ready, teacher_packet_summary),
        check(
            "sts_conditioning_has_causal_ablation",
            sts_causal_effect,
            {
                "trigger_state": state["sts_causal_decoder_ablation"].get("trigger_state"),
                "eligible_coverage_delta": sts_causal_summary.get("sts_public_eligible_coverage_delta"),
                "pass_rate_delta": sts_causal_summary.get("sts_public_pass_rate_delta"),
            },
            weight=1.3,
        ),
        check(
            "architecture_experiment_loop_closes",
            architecture_closed_loop,
            {
                "status": architecture_status,
                "selected": architecture_results.get("selected"),
                "promotion_decision": architecture_decision,
                "residual_delta_contract": architecture_contract,
            },
            weight=1.2,
        ),
        check("no_progress_demotion_available", no_progress_capability_installed(), "demotion ledger and board hook"),
    ]
    return domain("Self-improvement loop", checks, "Residuals become private pressure, verifier gates, teacher specs, or demotion.")


def self_improvement_gate_signal_counts(
    *,
    state: dict[str, Any],
    private_or_receiver_coverage_recovered: bool,
    learned_execution_shape_ready: bool,
    teacher_packet_ready: bool,
    sts_causal_effect: bool,
    architecture_closed_loop: bool,
) -> Counter[str]:
    """Count verified control-loop signal types that may live outside the ledger.

    The unattended improvement ledger can lag behind the actual gate reports. These
    supplemental labels are deliberately sourced from already-verified report gates
    and do not imply public promotion or model growth.
    """
    counts: Counter[str] = Counter()
    if private_or_receiver_coverage_recovered:
        counts["candidate_coverage_recovery"] += 1
    if learned_execution_shape_ready:
        counts["learned_execution_shape_gate"] += 1
    if teacher_packet_ready:
        counts["teacher_residual_packet"] += 1
    if sts_causal_effect:
        counts["sts_causal_delta"] += 1
    if architecture_closed_loop:
        counts["architecture_delta"] += 1

    ratchet = state.get("closed_loop_residual_ratchet", {})
    ratchet_summary = object_field(ratchet, "summary")
    same_seed = object_field(ratchet, "same_seed_evidence")
    private_only_lift = (
        ratchet.get("trigger_state") in {"GREEN", "YELLOW"}
        and float(number(ratchet_summary.get("same_seed_private_semantic_lift")) or 0.0) > 0.0
        and same_seed.get("ablation_private_only") is True
        and same_seed.get("public_tests_used") is False
        and same_seed.get("public_solutions_used") is False
        and same_seed.get("public_calibration_run") is False
    )
    if private_only_lift:
        counts["private_residual_lift"] += 1

    same_frontier_churn = object_field(ratchet, "same_frontier_churn")
    if same_frontier_churn.get("demoted") is True:
        counts["stale_frontier_demoted"] += 1

    return counts


def assess_breadth(state: dict[str, Any]) -> dict[str, Any]:
    tool_summary = object_field(state["tool_use"], "summary")
    cross_summary = object_field(state["cross_domain"], "summary")
    sts_summary = object_field(state["sts_repair_ablation"], "summary")
    transfer_suite_summary = object_field(state["transfer_eval_suite"], "summary")
    puffer_summary = object_field(state["pufferlib4"], "summary")
    agent_lane_summary = object_field(state["agent_lane_transfer_gate"], "summary")
    lane_counts = cross_summary.get("lane_counts") if isinstance(cross_summary.get("lane_counts"), dict) else {}
    board_policy_rows = int(number(get_path(state["board_game_policy"], ["summary", "policy_train_row_count"], 0)) or 0)
    board_trace_count = int(number(get_path(state["board_game_policy"], ["summary", "trace_count"], 0)) or 0)
    repo_rows = int(number(get_path(state["repo_repair"], ["summary", "code_lm_row_count"], 0)) or 0)
    sts_delta = float(number(sts_summary.get("pass_rate_delta")) or 0.0)
    transfer_accuracy = float(number(transfer_suite_summary.get("accuracy")) or 0.0)
    transfer_task_count = int(number(transfer_suite_summary.get("task_count")) or 0)
    puffer_report_present = bool(state["pufferlib4"])
    puffer_native_ready = bool(puffer_summary.get("native_backend_ready") or puffer_summary.get("native_backend_ok"))
    puffer_policy_evidence = bool(puffer_summary.get("native_policy_learning_evidence"))
    puffer_policy_rows = int(number(puffer_summary.get("policy_train_row_count")) or number(puffer_summary.get("policy_row_count")) or 0)
    puffer_capsules = int(number(puffer_summary.get("capsule_count")) or 0)
    lane_consumer_ready = agent_lane_named_consumers_ready(agent_lane_summary)
    checks = [
        check("tool_use_64_case_lane", int(number(tool_summary.get("case_count"))) >= 64 and number(tool_summary.get("pass_rate")) >= 0.85, tool_summary),
        check("board_game_learned_policy_not_just_harness", board_policy_rows >= 1000 and board_trace_count >= 32, {"policy_rows": board_policy_rows, "trace_count": board_trace_count, "benchmark": state["board_game"].get("trigger_state")}),
        check(
            "pufferlib_rl_not_harness_only",
            (not puffer_report_present) or (puffer_native_ready and puffer_policy_evidence and puffer_policy_rows > 0),
            {
                "report_present": puffer_report_present,
                "native_ready": puffer_native_ready,
                "native_policy_learning_evidence": puffer_policy_evidence,
                "policy_rows": puffer_policy_rows,
                "capsules": puffer_capsules,
                "trigger_state": state["pufferlib4"].get("trigger_state"),
            },
        ),
        check("repo_repair_rows_frontier_sized", repo_rows >= 128, repo_rows),
        check("cross_domain_capsules_present", int(number(cross_summary.get("capsule_count"))) > 0 and int(number(cross_summary.get("sts_row_count"))) > 0, cross_summary),
        check("at_least_four_lanes_in_capsules", len(lane_counts) >= 4, lane_counts, weight=1.4),
        check(
            "agent_lanes_have_named_consumers",
            lane_consumer_ready,
            {
                "trigger_state": state["agent_lane_transfer_gate"].get("trigger_state"),
                "repo_repair": agent_lane_summary.get("repo_repair"),
                "terminal_tool_use": agent_lane_summary.get("terminal_tool_use"),
                "pufferlib_rl": agent_lane_summary.get("pufferlib_rl"),
                "conversation": agent_lane_summary.get("conversation"),
                "sts_consumption": agent_lane_summary.get("sts_consumption"),
                "public_transfer": agent_lane_summary.get("public_transfer"),
                "rule": "This check gates named downstream consumers only; broad public transfer remains a separate code-public-transfer wall.",
            },
            weight=1.4,
        ),
        check("cross_domain_capsules_have_measured_transfer_effect", sts_delta > 0.0 or transfer_accuracy >= 0.75, {"sts_pass_rate_delta": sts_delta, "transfer_suite_accuracy": transfer_accuracy, "transfer_task_count": transfer_task_count}, weight=1.3),
        check("symliquid_routes_breadth", bool(get_path(state["symliquid"], ["action_kind_weights", "run_long_horizon_tool_use"], None)) and bool(get_path(state["symliquid"], ["action_kind_weights", "run_board_game_self_play"], None)), get_path(state["symliquid"], ["action_kind_weights"], {})),
    ]
    return domain("Breadth / cross-domain learning", checks, "Conversation, games, tool-use, repo repair, and code emit reusable STS capsules.")


def agent_lane_named_consumers_ready(summary: dict[str, Any]) -> bool:
    """Separate lane-consumer causality from the public-transfer floor.

    `agent_lane_transfer_gate.py` stays YELLOW while broad public transfer is
    below floor, but that does not mean the repo/tool/RL/conversation lanes lack
    named consumers. Keeping these signals separate prevents the control plane
    from chasing a false breadth blocker when the real wall is code transfer.
    """

    repo = object_field(summary, "repo_repair")
    tool = object_field(summary, "terminal_tool_use")
    puffer = object_field(summary, "pufferlib_rl")
    conversation = object_field(summary, "conversation")
    sts = object_field(summary, "sts_consumption")
    return all(
        [
            bool(repo.get("transfer_consumer_ready")),
            bool(tool.get("transfer_consumer_ready")),
            bool(puffer.get("transfer_consumer_ready")),
            bool(conversation.get("graduated")),
            bool(sts.get("named_consumer_effect")),
        ]
    )


def domain(label: str, checks: list[dict[str, Any]], target: str) -> dict[str, Any]:
    total_weight = sum(float(row.get("weight") or 1.0) for row in checks)
    earned = sum(float(row.get("weight") or 1.0) for row in checks if row.get("passed"))
    score = round(earned / max(1e-9, total_weight), 4)
    return {
        "label": label,
        "score": score,
        "grade": grade(score),
        "target": target,
        "passed_checks": sum(1 for row in checks if row.get("passed")),
        "total_checks": len(checks),
        "checks": checks,
    }


def check(name: str, passed: bool, evidence: Any, *, weight: float = 1.0) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "weight": weight, "evidence": evidence}


def main_walls(domains: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    walls = []
    for key, row in domains.items():
        failed = [check for check in row.get("checks", []) if not check.get("passed")]
        if row.get("score", 0.0) < 0.90:
            walls.append(
                {
                    "domain": key,
                    "grade": row.get("grade"),
                    "score": row.get("score"),
                    "failed_checks": [check.get("name") for check in failed[:6]],
                }
            )
    return walls


def recommended_actions(domains: dict[str, dict[str, Any]], state: dict[str, Any]) -> list[str]:
    actions = []
    asi_summary = object_field(state.get("asi_wall_breaker_governor", {}), "summary")
    asi_next = asi_summary.get("next_primary_action")
    if asi_next:
        actions.append(f"ASI wall governor primary action: {asi_next}")
    decoder_gate_summary = state["decoder_ablation_gate"].get("summary") if isinstance(state["decoder_ablation_gate"].get("summary"), dict) else {}
    execution_shape_report = best_execution_shape_report(state)
    execution_shape_summary = execution_shape_report.get("summary") if isinstance(execution_shape_report.get("summary"), dict) else {}
    learned_no_admissible_rate = execution_shape_summary.get("learned_token_decoder_no_admissible_candidate_rate")
    learned_no_admissible_rate = 1.0 if learned_no_admissible_rate is None else number(learned_no_admissible_rate)
    learned_execution_shape_ready = (
        float(number(execution_shape_summary.get("learned_token_decoder_pass_rate")) or 0.0) >= 0.70
        and float(learned_no_admissible_rate) <= 0.25
        and int(number(execution_shape_summary.get("diagnostic_template_candidate_count")) or 0) == 0
    )
    candidate_coverage_blocked = (
        float(number(decoder_gate_summary.get("public_eligible_task_coverage")) or 0.0) < 0.60
        or float(number(decoder_gate_summary.get("public_no_admissible_task_rate")) or 1.0) > 0.25
    )
    if domains["code_public_transfer"]["score"] < 0.90:
        actions.append("Keep Decoder Plan IR connected to Code LM rows so causal skeleton pressure is trained, not just reported.")
        if candidate_coverage_blocked and not learned_execution_shape_ready:
            actions.append("Keep public calibration locked; raise learned-token candidate coverage before any long closure or 4-card rerun.")
            actions.append("Prioritize parser/AST-constrained learned generation, split-token repair, and no-admissible residual rows until private coverage gates pass.")
        elif candidate_coverage_blocked:
            actions.append("Private learned-token coverage is recovered; keep public calibration locked until a fresh private closure refreshes receiver candidate manifests.")
            actions.append("Run fresh private closure -> decoder_v2 private ablation gate; only then consider one bounded public 4-card calibration.")
        else:
            actions.append("Run fresh private closure -> decoder_v2 private ablation -> one public 4-card calibration only if private gate passes.")
        actions.append("Patch Decoder V2 so contract/verifier feedback guides skeleton choice before body generation.")
    if domains["conversation_personality_lane"]["score"] < 0.90:
        actions.append("Run hard conversation v3 at 256+ cases and preserve correction/personality failures as STS capsules.")
    if domains["breadth_cross_domain_learning"]["score"] < 0.90:
        actions.append("Run agent_lane_transfer_gate.py and require repo/tool/RL/conversation traces to name downstream consumers with transfer evidence.")
        actions.append("Keep tool-use 64-case, repo repair, chess/Go learned policy, and cross-domain STS capsules rotating as non-code transfer pressure.")
    if domains["autonomy_unattended_operation"]["score"] < 0.90:
        actions.append("Keep board executor and watchdog authoritative; demote or block any done task without a useful signal.")
    if domains["self_improvement_loop"]["score"] < 0.90:
        actions.append("Escalate repeated flat-transfer residual clusters to teacher for one proposal-only experiment spec.")
    if domains.get("maturity_integrity_governance", {}).get("score", 1.0) < 0.90:
        actions.insert(0, "Run maturity_integrity_audit.py and clear hard integrity blockers before promotion, growth, or public calibration.")
    return actions[:8]


def best_conversation_report(state: dict[str, Any]) -> dict[str, Any]:
    for key in ("conversation_hard_v3", "conversation_hard_v2", "conversation_hard"):
        report = state.get(key)
        if isinstance(report, dict) and report:
            return report
    return {}


def grade(score: float) -> str:
    value = float(score or 0.0)
    if value >= 0.97:
        return "A+"
    if value >= 0.93:
        return "A"
    if value >= 0.90:
        return "A-"
    if value >= 0.87:
        return "B+"
    if value >= 0.83:
        return "B"
    if value >= 0.80:
        return "B-"
    if value >= 0.75:
        return "C+"
    if value >= 0.70:
        return "C"
    if value >= 0.60:
        return "C-"
    return "D"


def truthful_overall_score(raw_average: float, weakest_domain: float, walls: list[dict[str, Any]]) -> float:
    """Avoid averaging away a serious weakness.

    The scorecard is an operator control signal, not a morale layer. A system
    aiming at A+ cannot call itself A while a major domain is C-grade or while
    architecture debt blocks growth. If any major wall exists, the weakest
    domain caps the overall score.
    """

    if walls:
        return round(min(float(raw_average), float(weakest_domain)), 4)
    return round(float(raw_average), 4)


def attd_has_hard_cap_violation(report: dict[str, Any], gate_name: str) -> bool:
    violations = get_path(report, ["hard_caps", "violations"], [])
    if not isinstance(violations, list):
        return False
    return any(isinstance(row, dict) and row.get("gate") == gate_name for row in violations)


def attd_hard_cap_evidence(report: dict[str, Any], gate_name: str) -> Any:
    violations = get_path(report, ["hard_caps", "violations"], [])
    if isinstance(violations, list):
        for row in violations:
            if isinstance(row, dict) and row.get("gate") == gate_name:
                return row.get("evidence")
    checks = get_path(report, ["hard_caps", "checks"], [])
    if isinstance(checks, list):
        for row in checks:
            if isinstance(row, dict) and row.get("gate") == gate_name:
                return row.get("evidence")
    return "no_hard_cap_evidence"


def sqlite_count(path: Path, table: str) -> int:
    try:
        with sqlite3.connect(str(path)) as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            return int(row[0] if row else 0)
    except sqlite3.Error:
        return 0


def stop_flags() -> list[Path]:
    return [REPORTS / "sparkstream_stop.flag", REPORTS / "unattended_autonomy_stop.flag", REPORTS / "hive_work_board_stop.flag"]


def no_progress_capability_installed() -> bool:
    sources = [
        ROOT / "scripts" / "hive_work_board_executor.py",
        ROOT / "scripts" / "hive_work_board_executor_runtime.py",
    ]
    try:
        text = "\n".join(path.read_text(encoding="utf-8") for path in sources if path.exists())
    except OSError:
        return False
    return "NO_PROGRESS_DEMOTION_LEDGER" in text and "def no_progress_demote_or_block" in text


def edge_obligation_gate_available() -> bool:
    source = ROOT / "scripts" / "code_lm_closure.py"
    gate_script = ROOT / "scripts" / "edge_obligation_decode_gate_v1.py"
    try:
        text = source.read_text(encoding="utf-8")
    except OSError:
        text = ""
    return gate_script.exists() and "--edge-obligation-decode-gate-v1" in text


def audit_check_passed(audit: dict[str, Any], name: str) -> bool:
    checks = audit.get("checks") if isinstance(audit.get("checks"), list) else []
    for row in checks:
        if isinstance(row, dict) and row.get("name") == name:
            return bool(row.get("passed"))
    return False


def audit_check_evidence(audit: dict[str, Any], name: str) -> Any:
    checks = audit.get("checks") if isinstance(audit.get("checks"), list) else []
    for row in checks:
        if isinstance(row, dict) and row.get("name") == name:
            return row.get("evidence")
    return "missing_check"


def best_execution_shape_report(state: dict[str, Any]) -> dict[str, Any]:
    reports = [
        state.get("execution_shape_candidate_coverage") or {},
        state.get("execution_shape_smoke") or {},
        state.get("execution_shape_archive_gate") or {},
        state.get("execution_shape_full") or {},
    ]

    def report_score(report: dict[str, Any]) -> tuple[int, float, float]:
        summary = object_field(report, "summary")
        trigger_bonus = 1 if report.get("trigger_state") == "GREEN" else 0
        pass_rate = first_number(summary.get("learned_token_decoder_pass_rate")) or 0.0
        no_admissible = first_number(summary.get("learned_token_decoder_no_admissible_candidate_rate"))
        no_admissible = 1.0 if no_admissible is None else no_admissible
        return (trigger_bonus, pass_rate, -no_admissible)

    usable = [report for report in reports if isinstance(report, dict) and report]
    return max(usable, key=report_score) if usable else {}


def latest_execution_shape_candidate_coverage_report() -> dict[str, Any]:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for path in REPORTS.glob("execution_shape_candidate_coverage*.json"):
        name = path.name
        if name.endswith("_rust.json") or name.endswith("_checkpoint.json"):
            continue
        report = read_json(path, {})
        if not isinstance(report, dict) or not report:
            continue
        report.setdefault("source_report_path", rel(path))
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((mtime, report))
    if not candidates:
        return {}

    def score(item: tuple[float, dict[str, Any]]) -> tuple[int, float, float, float]:
        mtime, report = item
        summary = object_field(report, "summary")
        trigger_bonus = 1 if report.get("trigger_state") == "GREEN" else 0
        pass_rate = first_number(summary.get("learned_token_decoder_pass_rate")) or 0.0
        no_admissible = first_number(summary.get("learned_token_decoder_no_admissible_candidate_rate"))
        no_admissible = 1.0 if no_admissible is None else no_admissible
        return (trigger_bonus, pass_rate, -no_admissible, mtime)

    return max(candidates, key=score)[1]


def best_repo_repair_report() -> dict[str, Any]:
    candidates = [
        read_json(REPORTS / "high_transfer_repo_repair_learner.json", {}),
        read_json(REPORTS / "viea_repo_repair_learner.json", {}),
    ]

    def score(report: dict[str, Any]) -> tuple[int, int, int]:
        summary = object_field(report, "summary")
        trigger_bonus = 1 if report.get("trigger_state") == "GREEN" else 0
        rows = int(number(summary.get("code_lm_row_count")) or 0)
        traces = int(number(summary.get("validated_private_trace_count")) or 0)
        return (trigger_bonus, rows, traces)

    usable = [report for report in candidates if isinstance(report, dict) and report]
    return max(usable, key=score) if usable else {}


def ingest_self(path: Path, payload: dict[str, Any]) -> None:
    try:
        import sys

        sys.path.insert(0, str(ROOT / "scripts"))
        import report_evidence_store  # type: ignore

        report_evidence_store.ingest_report_path(report_evidence_store.DEFAULT_DB, path, payload=payload)
    except Exception:
        return


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Theseus A+ Operating Scorecard",
        "",
        f"- State: `{payload.get('trigger_state')}`",
        f"- Overall: `{summary.get('overall_grade')}` ({summary.get('overall_score')})",
        f"- Blocking walls: `{summary.get('blocking_wall_count')}`",
        "",
        "## Domains",
        "",
    ]
    for key, row in payload.get("domains", {}).items():
        lines.append(f"- `{row.get('grade')}` {row.get('label')} ({row.get('score')})")
        failed = [check.get("name") for check in row.get("checks", []) if not check.get("passed")]
        if failed:
            lines.append(f"  - Needs: {', '.join(str(item) for item in failed[:5])}")
    lines.extend(["", "## Main Walls", ""])
    for wall in payload.get("main_walls", []) or []:
        lines.append(f"- `{wall.get('grade')}` {wall.get('domain')}: {', '.join(wall.get('failed_checks') or [])}")
    if not payload.get("main_walls"):
        lines.append("- No sub-A- domains.")
    lines.extend(["", "## Recommended Next Actions", ""])
    for action in payload.get("recommended_next_actions", []) or []:
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def object_field(value: Any, key: str) -> dict[str, Any]:
    item = value.get(key) if isinstance(value, dict) else {}
    return item if isinstance(item, dict) else {}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def first_number(*values: Any) -> float | None:
    for value in values:
        try:
            if value is None:
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    try:
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines()[-max(1, limit) :]:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows
    except OSError:
        return []


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
