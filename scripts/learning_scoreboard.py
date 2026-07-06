"""Truth-layer scoreboard for Theseus/SparkStream learning evidence.

This report deliberately separates operational health from actual learning:

- operational_health: daemon/watchdog/resources are alive;
- private_training: private/synthetic training gains, never promotion proof;
- public_transfer: public calibration evidence and score semantics;
- promotion: whether gates allow a candidate;
- stale_or_superseded_lanes: reports that are red or obsolete without being
  confused with the active learned generator.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PUBLIC_CODE_FLOOR = 0.70


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/learning_scoreboard.json")
    parser.add_argument("--markdown-out", default="reports/learning_scoreboard.md")
    args = parser.parse_args()

    state = load_state()
    payload = build_scoreboard(state)
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 0 if payload["trigger_state"] in {"GREEN", "YELLOW"} else 2


def load_state() -> dict[str, Any]:
    return {
        "watchdog": read_json(REPORTS / "autonomy_watchdog.json"),
        "sparkstream": read_json(REPORTS / "sparkstream_status.json"),
        "hive": read_json(REPORTS / "hive_status.json"),
        "candidate": read_json(REPORTS / "candidate_promotion_gate.json"),
        "code_lm": read_json(REPORTS / "code_lm_closure.json"),
        "code_lm_rust": read_json(REPORTS / "code_lm_closure_rust.json"),
        "open_code_pantry": read_json(REPORTS / "open_code_training_pantry.json"),
        "open_conversation_pantry": read_json(REPORTS / "open_conversation_training_pantry.json"),
        "grammar_suckers": read_json(REPORTS / "grammar_suckers.json"),
        "deterministic_taming": read_json(REPORTS / "deterministic_taming_stack.json"),
        "architecture_guidance": read_json(REPORTS / "architecture_guidance_loop.json"),
        "teacher_budget": read_json(REPORTS / "teacher_budget_audit.json"),
        "code_residual_curriculum": read_json(REPORTS / "code_residual_curriculum.json"),
        "code_lm_residual_smoke": read_json(REPORTS / "code_lm_residual_smoke.json"),
        "sts_repair_ablation": read_json(REPORTS / "sts_repair_ablation.json"),
        "sts_learning": read_json(REPORTS / "sts_learning_forge.json"),
        "sts_native": read_json(REPORTS / "sts_native_parallel_probe.json"),
        "cognitive_context_router": read_json(REPORTS / "cognitive_context_router.json"),
        "real_code": read_best_public_code_report(),
        "broad_transfer_matrix": read_json(REPORTS / "broad_transfer_matrix.json"),
        "transfer_generalization": read_json(REPORTS / "transfer_generalization_audit.json"),
        "student_learning": read_json(REPORTS / "student_learning_closure.json"),
        "student_first_evidence": read_json(REPORTS / "student_first_evidence_audit.json"),
        "long_horizon_programming": read_json(REPORTS / "long_horizon_programming_curriculum.json"),
        "frontier": read_json(REPORTS / "frontier_policy_status.json"),
        "benchmaxx": read_json(REPORTS / "benchmaxx_curriculum.json"),
        "model_growth": read_json(REPORTS / "model_growth_gate.json"),
        "resource": read_json(REPORTS / "resource_governor.json"),
        "performance": read_json(REPORTS / "performance_optimizer.json"),
        "genesis": read_json(REPORTS / "genesis_kernel" / "report.json"),
        "reality_manipulator": read_json(REPORTS / "reality_manipulator.json"),
        "transfer_eval": read_json(REPORTS / "transfer_eval_suite.json"),
        "cell_lifecycle": read_json(REPORTS / "cell_lifecycle.json"),
    }


def build_scoreboard(state: dict[str, Any]) -> dict[str, Any]:
    real_code = state["real_code"]
    real_summary = object_field(real_code, "summary")
    broad_matrix = state["broad_transfer_matrix"]
    broad_matrix_summary = object_field(broad_matrix, "summary")
    transfer_generalization = state["transfer_generalization"]
    transfer_generalization_summary = object_field(transfer_generalization, "summary")
    code_lm = state["code_lm"]
    code_summary = object_field(code_lm, "summary")
    code_rust_summary = object_field(state["code_lm_rust"], "summary")
    open_code_pantry = state["open_code_pantry"]
    open_code_summary = object_field(open_code_pantry, "summary")
    open_conversation_pantry = state["open_conversation_pantry"]
    open_conversation_summary = object_field(open_conversation_pantry, "summary")
    grammar_suckers = state["grammar_suckers"]
    grammar_summary = object_field(grammar_suckers, "summary")
    taming = state["deterministic_taming"]
    taming_summary = object_field(taming, "summary")
    guidance = state["architecture_guidance"]
    guidance_diagnosis = object_field(guidance, "diagnosis")
    teacher_budget = state["teacher_budget"]
    residual_curriculum = state["code_residual_curriculum"]
    residual_curriculum_summary = object_field(residual_curriculum, "summary")
    residual_smoke = state["code_lm_residual_smoke"]
    residual_smoke_summary = object_field(residual_smoke, "summary")
    student_first = state["student_first_evidence"]
    student_first_summary = object_field(student_first, "summary")
    long_horizon = state["long_horizon_programming"]
    long_horizon_summary = object_field(long_horizon, "summary")
    sts_repair_ablation = state["sts_repair_ablation"]
    sts_repair_summary = object_field(sts_repair_ablation, "summary")
    candidate = state["candidate"]
    sts_learning = state["sts_learning"]
    sts_summary = object_field(sts_learning, "summary")
    sts_native = state["sts_native"]
    sts_native_summary = object_field(sts_native, "summary")
    cognitive_context = state["cognitive_context_router"]
    cognitive_summary = object_field(cognitive_context, "summary")
    cell_lifecycle = state["cell_lifecycle"]
    cell_lifecycle_summary = object_field(cell_lifecycle, "summary")
    cell_prune_plan = object_field(cell_lifecycle, "training_data_prune_plan")
    cell_prune_summary = object_field(cell_prune_plan, "summary")
    reality = state["reality_manipulator"]
    reality_acceptance = object_field(reality, "acceptance_scenario")
    reality_safety = object_field(reality, "safety_model")
    candidate_failed = failed_gates(candidate)
    public_pass = number(real_summary.get("real_public_task_pass_rate"))
    full_body_public_pass_count = int(real_summary.get("full_body_public_pass_count") or 0)
    public_floor_gap = round(max(0.0, PUBLIC_CODE_FLOOR - public_pass), 6)
    token_level_valid = bool(
        real_code.get("policy") == "project_theseus_real_code_benchmark_graduation_v1"
        and real_code.get("candidate_source") == "student_code_lm_checkpoint_v1"
        and real_code.get("public_benchmark_score_claim") == "student_code_lm_checkpoint_public_task_calibration_only"
        and bool(real_summary.get("token_level_code_generation_learned"))
        and bool(real_summary.get("student_candidate_benchmark_integrity_valid"))
        and int(real_summary.get("template_like_candidate_count") or 0) == 0
        and int(real_summary.get("loop_closure_candidate_count") or 0) == 0
        and int(real_code.get("external_inference_calls") or 0) == 0
    )
    public_transfer_ready = bool(
        token_level_valid
        and int(real_summary.get("public_task_count") or 0) > 0
        and int(real_summary.get("total_case_count") or 0) > 0
        and int(real_summary.get("benchmark_promotion_eligible_candidate_count") or 0) > 0
        and int(real_summary.get("task_level_regressions_vs_single_stream") or 0) == 0
    )
    promotion_allowed = bool(candidate.get("promote")) and public_pass >= PUBLIC_CODE_FLOOR
    stale = stale_or_superseded_lanes(state, token_level_valid=token_level_valid)
    hard_issues = []
    if not public_transfer_ready:
        hard_issues.append("missing_valid_public_transfer_evidence")
    if public_pass < PUBLIC_CODE_FLOOR:
        hard_issues.append("public_code_pass_rate_below_floor")
    if full_body_public_pass_count <= 0:
        hard_issues.append("public_code_passes_still_fallback_origin")
    if grammar_suckers.get("trigger_state") == "RED":
        hard_issues.append("grammar_sucker_red")
    if taming.get("trigger_state") == "RED":
        hard_issues.append("deterministic_taming_stack_red")
    if cognitive_context.get("trigger_state") == "RED":
        hard_issues.append("cognitive_context_spaces_red")
    if cell_lifecycle.get("trigger_state") == "RED":
        hard_issues.append("cell_lifecycle_red")
    if student_first.get("trigger_state") == "RED":
        hard_issues.append("student_first_evidence_audit_red")
    if broad_matrix.get("trigger_state") == "RED":
        hard_issues.append("broad_transfer_matrix_red")
    if transfer_generalization.get("trigger_state") == "RED":
        hard_issues.append("transfer_generalization_audit_red")
    if reality.get("trigger_state") == "RED":
        hard_issues.append("reality_manipulator_red")
    if long_horizon.get("trigger_state") == "RED":
        hard_issues.append("long_horizon_programming_curriculum_red")
    if int(grammar_summary.get("python_invalid_promotion_eligible_count") or 0) > 0:
        hard_issues.append("invalid_python_promotion_candidate")
    if "synthetic_benchmark_private_pressure_only" in candidate_failed:
        hard_issues.append("synthetic_pressure_is_private_not_promotion_evidence")
    operational = operational_health(state)
    trigger_state = "GREEN"
    if hard_issues or stale:
        trigger_state = "YELLOW"
    if not operational["alive"]:
        trigger_state = "RED"
    return {
        "policy": "project_theseus_learning_scoreboard_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "thesis": "Operational health is not learning. Private pressure is not public promotion. Public code transfer is the hard frontier.",
        "operational_health": operational,
        "private_training": {
            "policy": code_lm.get("policy"),
            "trigger_state": code_lm.get("trigger_state"),
            "private_task_count": code_summary.get("private_task_count"),
            "generated_private_task_count": code_summary.get("generated_private_task_count"),
            "open_code_train_expression_count": open_code_summary.get("private_train_expression_count"),
            "open_code_license_policy": open_code_summary.get("allowed_licenses"),
            "open_conversation_train_rows": open_conversation_summary.get("private_train_rows", 0),
            "open_conversation_sts_rows": open_conversation_summary.get("sts_rows", 0),
            "open_conversation_source_count": open_conversation_summary.get("source_count", 0),
            "open_conversation_license_policy": open_conversation_summary.get("allowed_licenses", []),
            "open_conversation_promotion_evidence": False,
            "private_train_task_count": code_summary.get("private_train_task_count"),
            "private_eval_task_count": code_summary.get("private_eval_task_count"),
            "sts_conditioning_used": bool(code_summary.get("sts_conditioning_used")),
            "sts_conditioned_public_task_count": code_summary.get("sts_conditioned_public_task_count"),
            "sts_conditioning_report": code_summary.get("sts_conditioning_report"),
            "before_next_token_accuracy": code_summary.get("before_next_token_accuracy"),
            "after_next_token_accuracy": code_summary.get("after_next_token_accuracy"),
            "next_token_accuracy_delta": code_summary.get("next_token_accuracy_delta"),
            "private_baseline_pass_rate": code_summary.get("private_baseline_pass_rate"),
            "private_sts_off_pass_rate": code_summary.get("private_sts_off_pass_rate"),
            "private_trained_pass_rate": code_summary.get("private_trained_pass_rate"),
            "private_pass_rate_delta": code_summary.get("private_pass_rate_delta"),
            "private_sts_repair_pass_rate_delta": code_summary.get("private_sts_repair_pass_rate_delta"),
            "private_sts_repair_task_level_improvements": code_summary.get("private_sts_repair_task_level_improvements"),
            "private_sts_repair_task_level_regressions": code_summary.get("private_sts_repair_task_level_regressions"),
            "private_concept_residual_counts": code_summary.get("private_concept_residual_counts", {}),
            "private_concept_family_pass_rates": code_summary.get("private_concept_family_pass_rates", {}),
            "promotion_evidence": False,
            "score_semantics": "private_training_gain_and_open_code_pantry_are_not_public_mastery",
        },
        "public_transfer": {
            "policy": real_code.get("policy"),
            "trigger_state": real_code.get("trigger_state"),
            "candidate_source": real_code.get("candidate_source"),
            "score_claim": real_code.get("public_benchmark_score_claim"),
            "public_task_count": real_summary.get("public_task_count"),
            "total_case_count": real_summary.get("total_case_count"),
            "real_public_task_pass_rate": public_pass,
            "required_floor": PUBLIC_CODE_FLOOR,
            "floor_gap": public_floor_gap,
            "pass_rate_delta": real_summary.get("pass_rate_delta"),
            "candidate_generation_modes": real_summary.get("candidate_generation_modes"),
            "multi_stream_pass_origin_counts": real_summary.get("multi_stream_pass_origin_counts", {}),
            "full_body_public_pass_count": full_body_public_pass_count,
            "expression_fallback_public_pass_count": real_summary.get("expression_fallback_public_pass_count", 0),
            "code_lm_candidate_generation_mode": code_rust_summary.get("candidate_generation_mode"),
            "sts_stream_conditioned_task_count": code_rust_summary.get("sts_stream_conditioned_task_count"),
            "regressions": real_summary.get("task_level_regressions_vs_single_stream"),
            "token_level_student_generation_valid": token_level_valid,
            "template_like_candidate_count": real_summary.get("template_like_candidate_count"),
            "loop_closure_candidate_count": real_summary.get("loop_closure_candidate_count"),
            "benchmark_promotion_eligible_candidate_count": real_summary.get("benchmark_promotion_eligible_candidate_count"),
            "promotion_evidence": public_transfer_ready and public_pass >= PUBLIC_CODE_FLOOR and full_body_public_pass_count > 0,
            "score_semantics": real_code.get("score_semantics"),
        },
        "broad_transfer_matrix": {
            "policy": broad_matrix.get("policy"),
            "trigger_state": broad_matrix.get("trigger_state"),
            "requested_card_count": broad_matrix_summary.get("requested_card_count"),
            "covered_card_count": broad_matrix_summary.get("covered_card_count"),
            "clean_covered_card_count": broad_matrix_summary.get("clean_covered_card_count"),
            "real_public_task_count": broad_matrix_summary.get("real_public_task_count"),
            "real_public_pass_rate": broad_matrix_summary.get("real_public_pass_rate"),
            "real_public_single_stream_pass_rate": broad_matrix_summary.get("real_public_single_stream_pass_rate"),
            "real_public_sts_delta": broad_matrix_summary.get("real_public_sts_delta"),
            "cards_below_floor": broad_matrix_summary.get("cards_below_floor", []),
            "no_clean_student_evidence_cards": broad_matrix_summary.get("no_clean_student_evidence_cards", []),
            "missing_cards": broad_matrix_summary.get("missing_cards", []),
            "loader_only_cards": broad_matrix_summary.get("loader_only_cards", []),
            "coverage_warning_cards": broad_matrix_summary.get("coverage_warning_cards", []),
            "no_cheat_violation_count": broad_matrix_summary.get("no_cheat_violation_count"),
            "best_single_public_report": broad_matrix.get("best_single_public_report", {}),
            "promotion_evidence": False,
            "score_semantics": broad_matrix.get("score_semantics"),
        },
        "transfer_generalization": {
            "policy": transfer_generalization.get("policy"),
            "trigger_state": transfer_generalization.get("trigger_state"),
            "clean_public_card_count": transfer_generalization_summary.get("clean_public_card_count"),
            "above_floor_transfer_card_count": transfer_generalization_summary.get("above_floor_transfer_card_count"),
            "aggregate_pass_rate": transfer_generalization_summary.get("aggregate_pass_rate"),
            "card_pass_rate_spread": transfer_generalization_summary.get("card_pass_rate_spread"),
            "card_pass_rate_stddev": transfer_generalization_summary.get("card_pass_rate_stddev"),
            "overfit_risk_count": transfer_generalization_summary.get("overfit_risk_count"),
            "top_shared_concepts": [
                row.get("concept")
                for row in transfer_generalization.get("shared_concept_targets", [])[:5]
                if isinstance(row, dict)
            ],
            "promotion_evidence": False,
            "score_semantics": transfer_generalization.get("score_semantics"),
        },
        "rule_substrate": {
            "policy": grammar_suckers.get("policy"),
            "trigger_state": grammar_suckers.get("trigger_state"),
            "python_parse_pass_rate": grammar_summary.get("python_parse_pass_rate"),
            "python_invalid_promotion_eligible_count": grammar_summary.get("python_invalid_promotion_eligible_count"),
            "english_surface_pass_rate": grammar_summary.get("english_surface_pass_rate"),
            "sbl_trace_count": grammar_summary.get("sbl_trace_count"),
            "legacy_sbl_found": bool(grammar_summary.get("legacy_sbl_found")),
            "public_benchmark_solutions_used": bool(grammar_summary.get("public_benchmark_solutions_used")),
            "public_tests_visible_to_rule_layer": bool(grammar_summary.get("public_tests_visible_to_rule_layer")),
            "external_inference_calls": grammar_suckers.get("external_inference_calls", 0),
            "promotion_evidence": False,
            "score_semantics": "grammar suckers validate legal form, route language arms, and emit SBL frames; they never provide benchmark answers",
        },
        "deterministic_taming_stack": {
            "policy": taming.get("policy"),
            "trigger_state": taming.get("trigger_state"),
            "arm_count": taming_summary.get("arm_count"),
            "passed_arms": taming_summary.get("passed_arms"),
            "hard_failure_count": taming_summary.get("hard_failure_count"),
            "soft_failure_count": taming_summary.get("soft_failure_count"),
            "python_invalid_promotion_candidates": taming_summary.get("python_invalid_promotion_candidates"),
            "rust_cargo_checked": taming_summary.get("rust_cargo_checked"),
            "promotion_evidence": False,
            "score_semantics": "deterministic linters/verifiers constrain legal form and tool safety; they are not answer sources",
        },
        "architecture_guidance": {
            "policy": guidance.get("policy"),
            "trigger_state": guidance.get("trigger_state"),
            "wall": guidance_diagnosis.get("wall"),
            "dominant_residual": guidance_diagnosis.get("dominant_residual"),
            "interpretation": guidance_diagnosis.get("interpretation"),
            "experiment_count": len(guidance.get("experiments", [])) if isinstance(guidance.get("experiments"), list) else 0,
            "teacher_status": get_path(guidance, ["teacher", "status"], "not_requested"),
            "teacher_external_inference_calls": get_path(guidance, ["teacher", "external_inference_calls"], 0),
            "promotion_evidence": False,
            "score_semantics": "teacher/guidance proposes experiments only; local measured evals decide adoption",
        },
        "teacher_budget": {
            "policy": teacher_budget.get("policy"),
            "trigger_state": teacher_budget.get("trigger_state"),
            "completed_today": teacher_budget.get("completed_today"),
            "completed_architecture_today": teacher_budget.get("completed_architecture_today"),
            "architecture_wall_budget_allowed": get_path(
                teacher_budget,
                ["reason_decisions", "architecture_wall", "budget", "allowed"],
                None,
            ),
            "architecture_wall_budget_reason": get_path(
                teacher_budget,
                ["reason_decisions", "architecture_wall", "budget", "reason"],
                None,
            ),
            "architecture_wall_evidence_allowed": get_path(
                teacher_budget,
                ["reason_decisions", "architecture_wall", "local_wall_evidence", "allowed"],
                None,
            ),
            "proposal_only_no_distillation": teacher_budget.get("proposal_only_no_distillation"),
            "promotion_evidence": False,
            "score_semantics": teacher_budget.get("score_semantics"),
        },
        "cell_lifecycle": {
            "policy": cell_lifecycle.get("policy"),
            "trigger_state": cell_lifecycle.get("trigger_state"),
            "cell_count": cell_lifecycle_summary.get("cell_count"),
            "expired_cells": cell_lifecycle_summary.get("expired_cells"),
            "renewed_or_protected": cell_lifecycle_summary.get("renewed_or_protected"),
            "improve_candidates": cell_lifecycle_summary.get("improve_candidates"),
            "split_or_compress_candidates": cell_lifecycle_summary.get("split_or_compress_candidates"),
            "retire_candidates": cell_lifecycle_summary.get("retire_candidates"),
            "training_data_archive_candidates": cell_lifecycle_summary.get("training_data_archive_candidates"),
            "training_data_archive_candidate_bytes": cell_lifecycle_summary.get("training_data_archive_candidate_bytes"),
            "tool_creation_pressure_count": cell_lifecycle_summary.get("tool_creation_pressure_count"),
            "prune_plan_mode": cell_prune_plan.get("mode"),
            "delete_performed": bool(cell_prune_plan.get("delete_performed")),
            "unsafe_prune_requests": cell_prune_summary.get("unsafe_prune_requests"),
            "teacher_recommended": get_path(cell_lifecycle, ["teacher_escalation", "recommended"], False),
            "promotion_evidence": False,
            "score_semantics": "cell death is anti-bloat pressure and architecture renewal; deletion is quarantine-only unless approved",
        },
        "reality_manipulator": {
            "policy": reality.get("policy"),
            "trigger_state": reality.get("trigger_state"),
            "world": get_path(reality, ["world", "name"], None),
            "world_type": get_path(reality, ["world", "type"], None),
            "artifact_count": get_path(reality, ["world", "artifact_graph", "artifact_count"], None),
            "claim_count": len(reality.get("claim_ledger", [])) if isinstance(reality.get("claim_ledger"), list) else 0,
            "critique_count": len(reality.get("critique_log", [])) if isinstance(reality.get("critique_log"), list) else 0,
            "compile_targets": get_path(reality, ["world", "compile_targets"], []),
            "acceptance_world_created": reality_acceptance.get("world_created"),
            "acceptance_release_manifest_ready": reality_acceptance.get("release_manifest_ready"),
            "high_risk_approved_without_gate_count": reality_safety.get("high_risk_approved_without_gate_count"),
            "promotion_evidence": False,
            "score_semantics": "intent-to-artifact world substrate only; learning progress still requires student checkpoint evidence and public/private eval reports",
        },
        "residual_curriculum": {
            "policy": residual_curriculum.get("policy"),
            "trigger_state": residual_curriculum.get("trigger_state"),
            "private_row_count": residual_curriculum_summary.get("private_row_count"),
            "residual_class_counts": residual_curriculum_summary.get("residual_class_counts", {}),
            "concept_residual_counts": residual_curriculum_summary.get("concept_residual_counts", {}),
            "target_wall_family_counts": residual_curriculum_summary.get("target_wall_family_counts", {}),
            "smoke_residual_private_rows_loaded": residual_smoke_summary.get("residual_private_train_task_count"),
            "smoke_next_token_accuracy_delta": residual_smoke_summary.get("next_token_accuracy_delta"),
            "smoke_private_pass_rate_delta": residual_smoke_summary.get("private_pass_rate_delta"),
            "public_benchmark_solutions_included": residual_curriculum_summary.get("public_benchmark_solutions_included"),
            "public_tests_included": residual_curriculum_summary.get("public_tests_included"),
            "promotion_evidence": False,
            "score_semantics": "private generated residual pressure only; public failures inform categories, not answers",
        },
        "student_first_evidence": {
            "policy": student_first.get("policy"),
            "trigger_state": student_first.get("trigger_state"),
            "candidate_source": student_first_summary.get("candidate_source"),
            "public_task_pass_rate": student_first_summary.get("public_task_pass_rate"),
            "required_public_task_floor": student_first_summary.get("required_public_task_floor"),
            "student_first_public_transfer_valid": student_first_summary.get("student_first_public_transfer_valid"),
            "promotion_allowed_by_evidence": student_first_summary.get("promotion_allowed_by_evidence"),
            "full_body_token_candidate_count": student_first_summary.get("full_body_token_candidate_count"),
            "template_like_candidate_count": student_first_summary.get("template_like_candidate_count"),
            "loop_closure_candidate_count": student_first_summary.get("loop_closure_candidate_count"),
            "ranker_counted_as_token_learning": student_first_summary.get("ranker_counted_as_token_learning"),
            "promotion_evidence": False,
            "score_semantics": "audit-only truth layer; it prevents helper/ranker evidence from being cited as token learning",
        },
        "long_horizon_programming": {
            "policy": long_horizon.get("policy"),
            "trigger_state": long_horizon.get("trigger_state"),
            "task_count": long_horizon_summary.get("task_count"),
            "sts_row_count": long_horizon_summary.get("sts_row_count"),
            "category_count": long_horizon_summary.get("category_count"),
            "categories": long_horizon_summary.get("categories", []),
            "task_out": long_horizon_summary.get("task_out"),
            "sts_out": long_horizon_summary.get("sts_out"),
            "public_benchmark_solutions_included": long_horizon_summary.get("public_benchmark_solutions_included"),
            "public_tests_included": long_horizon_summary.get("public_tests_included"),
            "promotion_evidence": False,
            "score_semantics": "private repo-repair hidden-test pressure; public SWE-style surfaces remain calibration-only",
        },
        "promotion": {
            "candidate_promote": bool(candidate.get("promote")),
            "promotion_allowed": promotion_allowed,
            "failed_gates": candidate_failed,
            "honest_blockers": hard_issues,
        },
        "frontier_truth": {
            "frontier_family": state["frontier"].get("frontier_family"),
            "pressure_card_id": state["frontier"].get("pressure_card_id"),
            "benchmaxx_next_family": get_path(state["benchmaxx"], ["next_frontier", "family"], ""),
            "benchmaxx_next_card": get_path(state["benchmaxx"], ["next_frontier", "recommended_env"], ""),
            "transfer_interleave": get_path(state["benchmaxx"], ["next_frontier", "transfer_interleave"], {}),
            "synthetic_pressure_role": "private_training_support_only",
            "promotion_facing_role": "real_public_code_calibration",
        },
        "parallel_stream_learning": {
            "policy": sts_learning.get("policy"),
            "trigger_state": sts_learning.get("trigger_state"),
            "sts_training_substrate_ready": bool(sts_summary.get("sts_training_substrate_ready")),
            "row_count": sts_summary.get("row_count"),
            "train_row_count": sts_summary.get("train_row_count"),
            "eval_row_count": sts_summary.get("eval_row_count"),
            "stream_count": sts_summary.get("stream_count"),
            "independent_output_stream_count": sts_summary.get("independent_output_stream_count"),
            "native_policy": sts_native.get("policy"),
            "native_trigger_state": sts_native.get("trigger_state"),
            "native_output_stream_count": sts_native_summary.get("output_stream_count"),
            "native_before_eval_token_accuracy": sts_native_summary.get("before_eval_token_accuracy"),
            "native_after_eval_token_accuracy": sts_native_summary.get("after_eval_token_accuracy"),
            "native_eval_token_accuracy_delta": sts_native_summary.get("eval_token_accuracy_delta"),
            "native_parallel_token_generation_proven": bool(sts_native_summary.get("native_parallel_token_generation_proven")),
            "one_token_per_output_stream_per_step": bool(sts_native_summary.get("one_token_per_output_stream_per_step")),
            "score_semantics": "STS is an independent private train/eval stream decoder probe, not public code promotion evidence",
        },
        "cognitive_context_spaces": {
            "policy": cognitive_context.get("policy"),
            "trigger_state": cognitive_context.get("trigger_state"),
            "context_row_count": cognitive_summary.get("context_row_count"),
            "merged_row_count": cognitive_summary.get("merged_row_count"),
            "stream_count": cognitive_summary.get("stream_count"),
            "target_context_space_count": cognitive_summary.get("target_context_space_count"),
            "target_context_spaces": cognitive_summary.get("target_context_spaces", []),
            "visible_report_requires_review": cognitive_summary.get("visible_report_requires_review"),
            "raw_chain_of_thought_exposure": cognitive_summary.get("raw_chain_of_thought_exposure"),
            "public_benchmark_solutions_included": cognitive_summary.get("public_benchmark_solutions_included"),
            "public_tests_included": cognitive_summary.get("public_tests_included"),
            "promotion_evidence": False,
            "score_semantics": cognitive_context.get("score_semantics"),
        },
        "sts_repair_ablation": {
            "policy": sts_repair_ablation.get("policy"),
            "trigger_state": sts_repair_ablation.get("trigger_state"),
            "single_stream_pass_rate": sts_repair_summary.get("single_stream_pass_rate"),
            "multi_stream_pass_rate": sts_repair_summary.get("multi_stream_pass_rate"),
            "pass_rate_delta": sts_repair_summary.get("pass_rate_delta"),
            "task_level_improvements": sts_repair_summary.get("task_level_improvements"),
            "task_level_regressions": sts_repair_summary.get("task_level_regressions"),
            "private_sts_repair_pass_rate_delta": sts_repair_summary.get("private_sts_repair_pass_rate_delta"),
            "private_concept_residual_counts": sts_repair_summary.get("private_concept_residual_counts", {}),
            "promotion_evidence": False,
            "score_semantics": "STS repair ablation tests causal utility but does not by itself promote a checkpoint",
        },
        "stale_or_superseded_lanes": stale,
        "dashboard_truth": {
            "current_learning_report": "reports/learning_scoreboard.json",
            "retire_raw_stale_lane_status": True,
            "stale_lanes_are_historical_context_not_active_blocks": bool(stale),
            "show_public_transfer_as_active_truth": True,
        },
        "model_growth": {
            "allowed": bool(state["model_growth"].get("model_growth_allowed")),
            "next_action": state["model_growth"].get("next_action"),
            "missing_evidence": state["model_growth"].get("missing_evidence", []),
        },
        "next_best_action": next_best_action(public_pass, public_floor_gap, stale, broad_matrix_summary),
        "external_inference_calls": 0,
    }


def operational_health(state: dict[str, Any]) -> dict[str, Any]:
    spark = state["sparkstream"]
    watchdog = state["watchdog"]
    performance = state["performance"]
    status_age = get_path(watchdog, ["summary", "status_age_seconds"], None)
    spark_status_fresh = isinstance(status_age, (int, float)) and status_age <= 900
    spark_policy_ok = spark.get("policy") in {"sparkstream_daemon_status_v0", "sparkstream_status_v0"}
    return {
        "alive": bool(
            spark_policy_ok
            and spark_status_fresh
            and watchdog.get("trigger_state") in {"GREEN", "YELLOW"}
            and performance.get("trigger_state") in {"GREEN", "YELLOW"}
        ),
        "watchdog_trigger_state": watchdog.get("trigger_state"),
        "sparkstream_policy": spark.get("policy"),
        "sparkstream_status_age_seconds": status_age,
        "sparkstream_phase": spark.get("phase"),
        "sparkstream_cycle": spark.get("cycle"),
        "performance_trigger_state": performance.get("trigger_state"),
        "resource_policy": state["resource"].get("policy"),
    }


def stale_or_superseded_lanes(state: dict[str, Any], *, token_level_valid: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    closure = state["student_learning"]
    if closure.get("policy") == "project_theseus_student_learning_closure_v1" and closure.get("trigger_state") == "RED":
        rows.append(
            {
                "lane": "student_learning_closure_ranker",
                "trigger_state": "RED",
                "status": "superseded_by_token_level_code_lm" if token_level_valid else "red",
                "failed_gates": failed_gates(closure),
                "why_it_matters": "old ranker/readout lane did not learn; do not cite it as student learning evidence",
            }
        )
    genesis = state["genesis"]
    if get_path(genesis, ["summary", "trigger_state"], "") == "YELLOW":
        rows.append(
            {
                "lane": "genesis_kernel",
                "trigger_state": "YELLOW",
                "status": "artifact_debt_or_open_critique",
                "failed_gates": [],
                "why_it_matters": "artifact substrate exists but should not be described as finished invention infrastructure",
            }
        )
    return rows


def next_best_action(
    public_pass: float,
    floor_gap: float,
    stale: list[dict[str, Any]],
    broad_matrix_summary: dict[str, Any] | None = None,
) -> str:
    if public_pass < PUBLIC_CODE_FLOOR:
        return f"Improve real Code LM transfer with governed private code/STS training, then rerun public calibration; current floor gap is {floor_gap:.3f}. Keep stale lanes superseded."
    broad_matrix_summary = broad_matrix_summary or {}
    if (
        broad_matrix_summary.get("missing_cards")
        or broad_matrix_summary.get("cards_below_floor")
        or broad_matrix_summary.get("no_clean_student_evidence_cards")
        or broad_matrix_summary.get("coverage_warning_cards")
    ):
        return "Broaden public-transfer calibration across benchmark families, train private residual lookalikes for below-floor cards, and keep the single-report promotion path honest."
    if stale:
        return "Use the scoreboard as the truth source and retire/supersede stale red learning lanes in dashboards."
    return "Run promotion closure with residual escrow and regression-only marking; public code floor is met."


def failed_gates(report: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    for key in ("checks", "gates"):
        values = report.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict) or item.get("passed") is not False:
                continue
            rows.append(str(item.get("gate") or item.get("name") or "unnamed"))
    return rows


def object_field(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    return item if isinstance(item, dict) else {}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
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


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def read_best_public_code_report() -> dict[str, Any]:
    """Prefer the broadest honest public calibration report available.

    The canonical report may be a small smoke slice. For the truth layer, a
    wider clean calibration should supersede it without deleting the smoke run.
    """
    candidates = []
    for path in REPORTS.glob("real_code_benchmark_graduation*.json"):
        payload = read_json(path)
        if not payload:
            continue
        summary = object_field(payload, "summary")
        candidates.append(
            (
                int(summary.get("public_task_count") or 0),
                str(payload.get("created_utc") or ""),
                path,
                payload,
            )
        )
    if not candidates:
        return {}
    candidates.sort(key=lambda item: (item[0], item[1], str(item[2])), reverse=True)
    selected = candidates[0][3]
    selected.setdefault("selected_truth_source", display_path(candidates[0][2]))
    return selected


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    private = payload["private_training"]
    public = payload["public_transfer"]
    broad = payload["broad_transfer_matrix"]
    transfer = payload["transfer_generalization"]
    rule = payload["rule_substrate"]
    taming = payload["deterministic_taming_stack"]
    guidance = payload["architecture_guidance"]
    teacher_budget = payload["teacher_budget"]
    lifecycle = payload["cell_lifecycle"]
    reality = payload["reality_manipulator"]
    residual = payload["residual_curriculum"]
    student_first = payload["student_first_evidence"]
    long_horizon = payload["long_horizon_programming"]
    cognitive = payload["cognitive_context_spaces"]
    sts_ablation = payload["sts_repair_ablation"]
    promotion = payload["promotion"]
    return "\n".join(
        [
            "# Learning Scoreboard",
            "",
            f"Generated: {payload.get('created_utc')}",
            "",
            f"Trigger: **{payload.get('trigger_state')}**",
            "",
            "## Public Transfer",
            "",
            f"- Pass rate: {public.get('real_public_task_pass_rate')} / floor {public.get('required_floor')}",
            f"- Candidate source: {public.get('candidate_source')}",
            f"- Generation modes: {public.get('candidate_generation_modes')}",
            f"- Pass origins: {public.get('multi_stream_pass_origin_counts')}",
            f"- Full-body public passes: {public.get('full_body_public_pass_count')}; expression fallback public passes: {public.get('expression_fallback_public_pass_count')}",
            f"- STS-conditioned tasks: {public.get('sts_stream_conditioned_task_count')}",
            f"- Score claim: {public.get('score_claim')}",
            f"- Token-level valid: {public.get('token_level_student_generation_valid')}",
            f"- Templates / loop-closure candidates: {public.get('template_like_candidate_count')} / {public.get('loop_closure_candidate_count')}",
            "",
            "## Broad Transfer Matrix",
            "",
            f"- State: {broad.get('trigger_state')}; clean covered cards: {broad.get('clean_covered_card_count')} / {broad.get('requested_card_count')}",
            f"- Public tasks: {broad.get('real_public_task_count')}; aggregate pass: {broad.get('real_public_pass_rate')}; STS delta: {broad.get('real_public_sts_delta')}",
            f"- Below-floor cards: {', '.join(broad.get('cards_below_floor') or []) or 'none'}",
            f"- No clean student evidence: {', '.join(broad.get('no_clean_student_evidence_cards') or []) or 'none'}",
            f"- Missing / loader-only cards: {', '.join(broad.get('missing_cards') or []) or 'none'} / {', '.join(broad.get('loader_only_cards') or []) or 'none'}",
            f"- No-cheat violations: {broad.get('no_cheat_violation_count')}",
            f"- Best single report: {broad.get('best_single_public_report')}",
            "",
            "## Transfer Generalization",
            "",
            f"- State: {transfer.get('trigger_state')}; above-floor transfer cards: {transfer.get('above_floor_transfer_card_count')} / {transfer.get('clean_public_card_count')}",
            f"- Aggregate pass: {transfer.get('aggregate_pass_rate')}; spread: {transfer.get('card_pass_rate_spread')}; stddev: {transfer.get('card_pass_rate_stddev')}",
            f"- Overfit risks: {transfer.get('overfit_risk_count')}",
            f"- Shared concepts: {', '.join(transfer.get('top_shared_concepts') or []) or 'none'}",
            "",
            "## Private Training",
            "",
            f"- Private pass: {private.get('private_baseline_pass_rate')} -> {private.get('private_trained_pass_rate')}",
            f"- Next-token accuracy: {private.get('before_next_token_accuracy')} -> {private.get('after_next_token_accuracy')}",
            f"- STS conditioning: {private.get('sts_conditioning_used')} on {private.get('sts_conditioned_public_task_count')} public calibration tasks",
            "",
            "## Rule Substrate",
            "",
            f"- Grammar suckers: {rule.get('trigger_state')}",
            f"- Python parse pass: {rule.get('python_parse_pass_rate')}; invalid promotion candidates: {rule.get('python_invalid_promotion_eligible_count')}",
            f"- English surface pass: {rule.get('english_surface_pass_rate')}",
            f"- SBL traces: {rule.get('sbl_trace_count')}; legacy SBL found: {rule.get('legacy_sbl_found')}",
            f"- Semantics: {rule.get('score_semantics')}",
            "",
            "## Taming Stack",
            "",
            f"- State: {taming.get('trigger_state')}; arms: {taming.get('passed_arms')} / {taming.get('arm_count')}",
            f"- Python invalid promotion candidates: {taming.get('python_invalid_promotion_candidates')}",
            f"- Rust cargo checked: {taming.get('rust_cargo_checked')}",
            "",
            "## Architecture Guidance",
            "",
            f"- Wall: {guidance.get('wall')}",
            f"- Dominant residual: {guidance.get('dominant_residual')}",
            f"- Experiments: {guidance.get('experiment_count')}; teacher: {guidance.get('teacher_status')}",
            f"- Interpretation: {guidance.get('interpretation')}",
            "",
            "## Teacher Budget",
            "",
            f"- State: {teacher_budget.get('trigger_state')}; completed today: {teacher_budget.get('completed_today')}",
            f"- Architecture calls today: {teacher_budget.get('completed_architecture_today')}",
            f"- Architecture wall allowed: budget={teacher_budget.get('architecture_wall_budget_allowed')} reason={teacher_budget.get('architecture_wall_budget_reason')} evidence={teacher_budget.get('architecture_wall_evidence_allowed')}",
            f"- Proposal-only/no-distillation: {teacher_budget.get('proposal_only_no_distillation')}",
            "",
            "## Cell Lifecycle",
            "",
            f"- State: {lifecycle.get('trigger_state')}; cells: {lifecycle.get('cell_count')}; expired: {lifecycle.get('expired_cells')}",
            f"- Improve / split / retire: {lifecycle.get('improve_candidates')} / {lifecycle.get('split_or_compress_candidates')} / {lifecycle.get('retire_candidates')}",
            f"- Tool creation pressure: {lifecycle.get('tool_creation_pressure_count')}",
            f"- Data archive candidates: {lifecycle.get('training_data_archive_candidates')} ({lifecycle.get('training_data_archive_candidate_bytes')} bytes)",
            f"- Delete performed / unsafe prune requests: {lifecycle.get('delete_performed')} / {lifecycle.get('unsafe_prune_requests')}",
            f"- Semantics: {lifecycle.get('score_semantics')}",
            "",
            "## Reality Manipulator",
            "",
            f"- State: {reality.get('trigger_state')}; world: {reality.get('world')} ({reality.get('world_type')})",
            f"- Artifacts / claims / critiques: {reality.get('artifact_count')} / {reality.get('claim_count')} / {reality.get('critique_count')}",
            f"- Compile targets: {reality.get('compile_targets')}",
            f"- High-risk approvals without gate: {reality.get('high_risk_approved_without_gate_count')}",
            f"- Semantics: {reality.get('score_semantics')}",
            "",
            "## Residual Curriculum",
            "",
            f"- Private rows: {residual.get('private_row_count')}",
            f"- Smoke loaded rows / token delta: {residual.get('smoke_residual_private_rows_loaded')} / {residual.get('smoke_next_token_accuracy_delta')}",
            f"- Classes: {residual.get('residual_class_counts')}",
            f"- Public solutions/tests included: {residual.get('public_benchmark_solutions_included')} / {residual.get('public_tests_included')}",
            "",
            "## Student-First Evidence",
            "",
            f"- State: {student_first.get('trigger_state')}; valid: {student_first.get('student_first_public_transfer_valid')}",
            f"- Candidate source: {student_first.get('candidate_source')}",
            f"- Public pass/floor: {student_first.get('public_task_pass_rate')} / {student_first.get('required_public_task_floor')}",
            f"- Full-body token candidates: {student_first.get('full_body_token_candidate_count')}",
            f"- Templates / loop tools: {student_first.get('template_like_candidate_count')} / {student_first.get('loop_closure_candidate_count')}",
            f"- Ranker counted as token learning: {student_first.get('ranker_counted_as_token_learning')}",
            "",
            "## Long-Horizon Programming",
            "",
            f"- State: {long_horizon.get('trigger_state')}; tasks: {long_horizon.get('task_count')}; STS rows: {long_horizon.get('sts_row_count')}",
            f"- Categories: {long_horizon.get('categories')}",
            f"- Outputs: {long_horizon.get('task_out')} / {long_horizon.get('sts_out')}",
            f"- Public solutions/tests included: {long_horizon.get('public_benchmark_solutions_included')} / {long_horizon.get('public_tests_included')}",
            "",
            "## STS Ablation",
            "",
            f"- Single vs multi: {sts_ablation.get('single_stream_pass_rate')} -> {sts_ablation.get('multi_stream_pass_rate')}",
            f"- Delta/regressions: {sts_ablation.get('pass_rate_delta')} / {sts_ablation.get('task_level_regressions')}",
            "",
            "## Cognitive Context Spaces",
            "",
            f"- State: {cognitive.get('trigger_state')}; rows: {cognitive.get('context_row_count')}; merged rows: {cognitive.get('merged_row_count')}",
            f"- Spaces: {cognitive.get('target_context_spaces')}",
            f"- Visible requires review: {cognitive.get('visible_report_requires_review')}",
            f"- Raw chain exposure: {cognitive.get('raw_chain_of_thought_exposure')}",
            f"- Public solutions/tests included: {cognitive.get('public_benchmark_solutions_included')} / {cognitive.get('public_tests_included')}",
            "",
            "## Promotion",
            "",
            f"- Candidate promote: {promotion.get('candidate_promote')}",
            f"- Promotion allowed: {promotion.get('promotion_allowed')}",
            f"- Blockers: {', '.join(promotion.get('honest_blockers') or []) or 'none'}",
            "",
            f"Next: {payload.get('next_best_action')}",
            "",
        ]
    )


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
