#!/usr/bin/env python3
"""Build the Capability Transfer Closure v1 evidence packet.

This packet is an audit artifact, not a benchmark runner. It consolidates the
current private transfer gates, public calibration locks, neural substrate
comparators, VCM deltas, and Mac accelerator evidence so the next decision is
grounded in the same artifacts the gates already use.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_INPUTS = {
    "private_admissibility": "reports/private_full_body_candidate_admissibility_gate_capability_transfer_closure_v1.json",
    "private_residual_gate": "reports/private_residual_repair_v3_gate_capability_transfer_closure_v1.json",
    "private_heldout": "reports/private_residual_repair_v3_heldout_score_capability_transfer_closure_v1.json",
    "broad_transfer": "reports/broad_transfer_matrix.json",
    "public_audit": "reports/bounded_public_transfer_calibration_goal_audit_v1.json",
    "public_residual_mining_current": "reports/bounded_public_transfer_residual_mining_current_private_full_body_v1.json",
    "private_residual_target_consumer": "reports/private_residual_target_consumer_v1.json",
    "public_shape_generator": "reports/student_token_code_generator_capability_transfer_closure_v1.json",
    "public_shape_return_audit": "reports/public_shape_return_contract_audit_v1.json",
    "public_shape_semantic_structure_audit": "reports/public_shape_semantic_structure_audit_v1.json",
    "neural_seed": "reports/neural_seed_growth_gate.json",
    "vcm": "reports/vcm_release_conformance_audit.json",
    "mac_mlx": "reports/macos_mlx_structural_action_smoke.json",
    "mac_metal": "reports/macos_metal_production_route_readiness.json",
    "candidate_promotion": "reports/candidate_promotion_gate.json",
    "operator_lock": "reports/public_calibration_operator_lock.flag",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    for name, default in DEFAULT_INPUTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", default=default)
    parser.add_argument("--out", default="reports/capability_transfer_closure_v1.json")
    parser.add_argument("--markdown-out", default="reports/capability_transfer_closure_v1.md")
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    inputs = {name: resolve(getattr(args, name)) for name in DEFAULT_INPUTS}
    private_admissibility = read_json(inputs["private_admissibility"], {})
    private_residual_gate = read_json(inputs["private_residual_gate"], {})
    private_heldout = read_json(inputs["private_heldout"], {})
    broad_transfer = read_json(inputs["broad_transfer"], {})
    public_audit = read_json(inputs["public_audit"], {})
    public_residual_mining_current = read_json(inputs["public_residual_mining_current"], {})
    private_residual_target_consumer = read_json(inputs["private_residual_target_consumer"], {})
    public_shape_generator = read_json(inputs["public_shape_generator"], {})
    neural_seed = read_json(inputs["neural_seed"], {})
    vcm = read_json(inputs["vcm"], {})
    mac_mlx = read_json(inputs["mac_mlx"], {})
    mac_metal = read_json(inputs["mac_metal"], {})
    candidate_promotion = read_json(inputs["candidate_promotion"], {})
    operator_lock_active = inputs["operator_lock"].exists()

    adm = obj(private_admissibility, "summary")
    residual = obj(private_residual_gate, "summary")
    heldout = obj(private_heldout, "summary")
    broad = obj(broad_transfer, "summary")
    public_summary = obj(public_audit, "summary")
    public_residual_mining_current_summary = obj(public_residual_mining_current, "summary")
    private_residual_target_consumer_summary = obj(private_residual_target_consumer, "summary")
    public_shape = obj(public_shape_generator, "summary")
    public_shape_return_audit = read_json(inputs["public_shape_return_audit"], {})
    public_shape_return = obj(public_shape_return_audit, "summary")
    public_shape_semantic_audit = read_json(inputs["public_shape_semantic_structure_audit"], {})
    public_shape_semantic = obj(public_shape_semantic_audit, "summary")
    neural = obj(neural_seed, "summary")
    vcm_summary = obj(vcm, "summary")
    mac_mlx_summary = obj(mac_mlx, "summary")
    mac_metal_summary = obj(mac_metal, "summary")
    promotion = obj(candidate_promotion, "summary")

    public_score = obj(public_summary, "current_public_score")
    previous_public_score = obj(public_summary, "previous_locked_baseline")
    code_proposer = obj(neural, "code_proposer_comparator")
    token_decoder = obj(neural, "token_decoder_comparator")
    architecture_sweep = obj(neural, "architecture_sweep")
    arch_aggregate = obj(architecture_sweep, "aggregate")
    budget_ladder = obj(arch_aggregate, "one_notch_budget_ladder")

    no_cheat = {
        "fallback_return_count": int_number(adm.get("fallback_return_candidate_count"))
        + int_number(heldout.get("fallback_return_candidate_count")),
        "template_like_count": int_number(adm.get("template_like_candidate_count")),
        "public_leakage_count": int_number(adm.get("public_leakage_count")),
        "external_inference_calls": int_number(adm.get("external_inference_calls"))
        + int_number(heldout.get("external_inference_calls"))
        + int_number(public_shape.get("external_inference_calls"))
        + int_number(public_shape_return.get("external_inference_calls"))
        + int_number(public_shape_semantic.get("external_inference_calls"))
        + int_number(vcm_summary.get("external_inference_calls"))
        + int_number(mac_mlx_summary.get("external_inference_calls"))
        + int_number(mac_metal_summary.get("external_inference_calls")),
        "public_training_rows": int_number(mac_mlx_summary.get("public_training_rows")),
        "teacher_used_count": int_number(mac_mlx_summary.get("teacher_used"))
        + int_number(mac_metal_summary.get("teacher_used_count")),
    }
    private_semantic_ready = bool(
        private_admissibility.get("trigger_state") == "GREEN"
        and private_residual_gate.get("trigger_state") == "GREEN"
        and number(adm.get("selected_pass_rate")) >= 0.70
        and number(adm.get("no_admissible_task_rate")) <= 0.03
        and number(heldout.get("learned_candidate_task_pass_rate")) >= 0.70
    )
    public_candidate_coverage = {
        "trigger_state": public_shape_generator.get("trigger_state"),
        "task_count": int_number(public_shape.get("task_count")),
        "candidate_count": int_number(public_shape.get("candidate_count")),
        "benchmark_promotion_eligible_candidate_count": int_number(
            public_shape.get("benchmark_promotion_eligible_candidate_count")
        ),
        "full_body_token_candidate_count": int_number(public_shape.get("full_body_token_candidate_count")),
        "grammar_masked_learned_token_candidate_count": int_number(
            public_shape.get("grammar_masked_learned_token_candidate_count")
        ),
        "expression_memory_fallback_count": int_number(public_shape.get("expression_memory_fallback_count")),
        "template_like_candidate_count": int_number(public_shape.get("template_like_candidate_count")),
        "loop_closure_candidate_count": int_number(public_shape.get("loop_closure_candidate_count")),
        "public_tests_visible_to_generator": public_shape.get("public_tests_visible_to_generator"),
        "canonical_solution_seen_by_solver": public_shape.get("canonical_solution_seen_by_solver"),
        "external_inference_calls": int_number(public_shape.get("external_inference_calls")),
        "ready_training_sources": int_number(public_shape.get("ready_training_sources")),
        "training_rows_used": int_number(public_shape.get("training_rows_used")),
        "return_contract_audit_state": public_shape_return_audit.get("trigger_state"),
        "return_contract_expected_shape_coverage_rate": number(
            public_shape_return.get("expected_shape_coverage_rate")
        ),
        "return_contract_selected_shape_compatible_task_rate": number(
            public_shape_return.get("selected_shape_compatible_task_rate")
        ),
        "return_contract_any_shape_compatible_task_rate": number(
            public_shape_return.get("any_shape_compatible_task_rate")
        ),
        "return_contract_candidate_ast_parse_rate": number(public_shape_return.get("candidate_ast_parse_rate")),
        "return_contract_selected_ast_parse_rate": number(public_shape_return.get("selected_ast_parse_rate")),
        "return_contract_cheat_counts": public_shape_return.get("cheat_counts", {}),
        "semantic_structure_audit_state": public_shape_semantic_audit.get("trigger_state"),
        "semantic_structure_selected_obligation_satisfaction_rate": number(
            public_shape_semantic.get("selected_obligation_satisfaction_rate")
        ),
        "semantic_structure_any_obligation_satisfaction_rate": number(
            public_shape_semantic.get("any_obligation_satisfaction_rate")
        ),
        "semantic_structure_best_single_obligation_satisfaction_rate": number(
            public_shape_semantic.get("any_obligation_satisfaction_rate")
        ),
        "semantic_structure_selected_full_obligation_task_rate": number(
            public_shape_semantic.get("selected_task_full_obligation_rate")
        ),
        "semantic_structure_best_single_full_obligation_task_rate": number(
            public_shape_semantic.get("any_task_full_obligation_rate")
        ),
        "semantic_structure_fragmented_candidate_union_only_task_count": int_number(
            public_shape_semantic.get("fragmented_candidate_union_only_task_count")
        ),
        "semantic_structure_fragmented_candidate_union_only_task_rate": number(
            public_shape_semantic.get("fragmented_candidate_union_only_task_rate")
        ),
        "semantic_structure_candidate_union_missing_structure_counts": public_shape_semantic.get(
            "candidate_union_missing_structure_counts",
            public_shape_semantic.get("any_missing_structure_counts", {}),
        ),
        "semantic_structure_best_single_missing_structure_counts": public_shape_semantic.get(
            "any_single_candidate_missing_structure_counts", {}
        ),
        "semantic_structure_multi_statement_candidate_count": int_number(
            public_shape_semantic.get("multi_statement_generated_body_candidate_count")
        ),
        "semantic_structure_multi_statement_candidate_rate": number(
            public_shape_semantic.get("multi_statement_generated_body_candidate_rate")
        ),
        "semantic_structure_expression_wrapped_candidate_rate": number(
            public_shape_semantic.get("expression_wrapped_body_candidate_rate")
        ),
        "semantic_structure_cheat_counts": public_shape_semantic.get("cheat_counts", {}),
    }
    public_candidate_coverage_ready = bool(
        public_candidate_coverage["trigger_state"] == "GREEN"
        and public_candidate_coverage["ready_training_sources"] > 0
        and public_candidate_coverage["training_rows_used"] > 0
        and public_candidate_coverage["task_count"] >= 160
        and public_candidate_coverage["benchmark_promotion_eligible_candidate_count"] > 0
        and public_candidate_coverage["full_body_token_candidate_count"] > 0
        and public_candidate_coverage["grammar_masked_learned_token_candidate_count"] > 0
        and public_candidate_coverage["expression_memory_fallback_count"] == 0
        and public_candidate_coverage["template_like_candidate_count"] == 0
        and public_candidate_coverage["loop_closure_candidate_count"] == 0
        and public_candidate_coverage["external_inference_calls"] == 0
        and public_candidate_coverage["public_tests_visible_to_generator"] is False
        and public_candidate_coverage["canonical_solution_seen_by_solver"] is False
        and public_candidate_coverage["return_contract_audit_state"] == "GREEN"
        and public_candidate_coverage["semantic_structure_audit_state"] == "GREEN"
    )
    public_candidate_coverage["ready"] = public_candidate_coverage_ready
    public_promotion_ready = bool(
        public_candidate_coverage_ready
        and not operator_lock_active
        and private_semantic_ready
    )
    no_cheat_clean = all(value == 0 for value in no_cheat.values())
    public_transfer_wall = {
        "locked_broad_public_pass_rate": number(broad.get("real_public_pass_rate")),
        "locked_broad_public_pass_count": int_number(broad.get("real_public_multi_passed")),
        "locked_broad_public_task_count": int_number(broad.get("real_public_task_count")),
        "cards_below_floor": broad.get("cards_below_floor", []),
        "last_public_audit_pass_rate": number(public_score.get("pass_rate")),
        "last_public_audit_pass_count": int_number(public_score.get("pass_count")),
        "last_public_audit_task_count": int_number(public_score.get("task_count")),
        "previous_locked_baseline_pass_rate": number(previous_public_score.get("pass_rate")),
        "previous_locked_baseline_pass_count": int_number(previous_public_score.get("pass_count")),
        "delta_vs_previous_locked_baseline": number(public_summary.get("delta_vs_previous_locked_baseline")),
        "dominant_failure": public_summary.get("dominant_failure"),
        "public_calibration_allowed": bool(public_summary.get("public_calibration_allowed")),
        "operator_lock_active": operator_lock_active,
        "current_full_body_residual_mining": {
            "trigger_state": public_residual_mining_current.get("trigger_state"),
            "candidate_manifest_score_claim_allowed": public_residual_mining_current_summary.get(
                "candidate_manifest_score_claim_allowed"
            ),
            "candidate_manifest_slice_alignment": public_residual_mining_current_summary.get(
                "candidate_manifest_slice_alignment"
            ),
            "residual_category_counts": public_residual_mining_current_summary.get("residual_category_counts"),
            "full_body_token_candidate_count": int_number(
                public_residual_mining_current_summary.get("full_body_token_candidate_count")
            ),
            "countable_integrity_candidate_count": int_number(
                public_residual_mining_current_summary.get("countable_integrity_candidate_count")
            ),
            "promotable_candidate_count": int_number(
                public_residual_mining_current_summary.get("promotable_candidate_count")
            ),
            "fallback_return_candidate_count": int_number(
                public_residual_mining_current_summary.get("fallback_return_candidate_count")
            ),
            "template_like_candidate_count": int_number(
                public_residual_mining_current_summary.get("template_like_candidate_count")
            ),
            "training_rows_written": int_number(
                public_residual_mining_current_summary.get("training_rows_written")
            ),
            "private_residual_target_rows_written": int_number(
                public_residual_mining_current_summary.get("private_residual_target_rows_written")
            ),
            "private_residual_target_manifest": public_residual_mining_current_summary.get(
                "private_residual_target_manifest"
            ),
            "external_inference_calls": int_number(
                public_residual_mining_current_summary.get("external_inference_calls")
            ),
            "score_semantics": (
                "Hash-only residual mining over the current full-body manifest. If "
                "candidate_manifest_score_claim_allowed is false, this is slice-alignment evidence, "
                "not a new public score or a semantic pass/fail claim."
            ),
        },
        "private_residual_target_consumer": {
            "trigger_state": private_residual_target_consumer.get("trigger_state"),
            "target_rows": int_number(private_residual_target_consumer_summary.get("target_rows")),
            "valid_target_rows": int_number(private_residual_target_consumer_summary.get("valid_target_rows")),
            "repair_queue_rows": int_number(private_residual_target_consumer_summary.get("repair_queue_rows")),
            "covered_target_count": int_number(private_residual_target_consumer_summary.get("covered_target_count")),
            "ready_target_count": int_number(private_residual_target_consumer_summary.get("ready_target_count")),
            "needs_private_ablation_count": int_number(
                private_residual_target_consumer_summary.get("needs_private_ablation_count")
            ),
            "blocked_target_count": int_number(private_residual_target_consumer_summary.get("blocked_target_count")),
            "unresolved_target_count": int_number(
                private_residual_target_consumer_summary.get("unresolved_target_count")
            ),
            "unresolved_target_state_counts": private_residual_target_consumer_summary.get(
                "unresolved_target_state_counts", {}
            ),
            "unresolved_target_category_counts": private_residual_target_consumer_summary.get(
                "unresolved_target_category_counts", {}
            ),
            "target_coverage_rate": number(private_residual_target_consumer_summary.get("target_coverage_rate")),
            "all_valid_targets_closed_by_current_private_evidence": bool(
                private_residual_target_consumer_summary.get("all_valid_targets_closed_by_current_private_evidence")
            ),
            "queue_path": private_residual_target_consumer_summary.get("queue_path"),
            "training_rows_written": int_number(private_residual_target_consumer_summary.get("training_rows_written")),
            "public_training_rows_written": int_number(
                private_residual_target_consumer_summary.get("public_training_rows_written")
            ),
            "external_inference_calls": int_number(private_residual_target_consumer_summary.get("external_inference_calls")),
        },
    }
    substrate = {
        "code_proposer_best_arm": code_proposer.get("best_sts_on_arm_by_verifier_pass_rate"),
        "code_proposer_symliquid_minus_transformer": number(
            code_proposer.get("symliquid_minus_transformer_sts_on_verifier_pass_rate")
        ),
        "token_decoder_best_arm": token_decoder.get("best_sts_on_arm_by_verifier_pass_rate"),
        "token_decoder_symliquid_minus_transformer": number(
            token_decoder.get("symliquid_minus_transformer_sts_on_verifier_pass_rate")
        ),
        "budget_ladder_symliquid_minus_transformer_mean": number(
            obj(budget_ladder, "symliquid_minus_transformer").get("mean")
        ),
        "neural_student_ready": bool(neural.get("neural_student_ready")),
        "architecture_sweep_seed_count": int_number(obj(budget_ladder, "symliquid_minus_transformer").get("count")),
    }
    vcm_delta = {
        "public_memory_prompt_state": vcm_summary.get("public_memory_prompt_calibration_state"),
        "public_memory_prompt_vcm_on_pass_rate": number(vcm_summary.get("public_memory_prompt_vcm_on_pass_rate")),
        "public_memory_prompt_vcm_off_pass_rate": number(vcm_summary.get("public_memory_prompt_vcm_off_pass_rate")),
        "public_memory_prompt_vcm_over_flat_tail_delta": number(
            vcm_summary.get("public_memory_prompt_vcm_over_flat_tail_delta")
        ),
        "public_memory_prompt_vcm_over_best_non_vcm_delta": number(
            vcm_summary.get("public_memory_prompt_vcm_over_best_non_vcm_delta")
        ),
        "runtime_profile_state": obj(vcm_summary, "profile_states").get("VCM-Runtime"),
    }
    accelerator = {
        "mac_mlx_state": mac_mlx.get("trigger_state"),
        "mlx_available": mac_mlx_summary.get("mlx_available"),
        "mlx_used": mac_mlx_summary.get("mlx_used"),
        "mlx_default_device": mac_mlx_summary.get("mlx_default_device"),
        "mlx_structural_action_verifier_pass_rate": number(mac_mlx_summary.get("verifier_pass_rate")),
        "mlx_structural_action_train_wall_time_ms": int_number(mac_mlx_summary.get("train_wall_time_ms")),
        "mlx_fallback_return_rows": int_number(mac_mlx_summary.get("fallback_return_rows")),
        "metal_route_state": mac_metal.get("trigger_state"),
        "metal_production_route_allowed": mac_metal_summary.get("production_route_allowed"),
        "metal_kernel_parity_pending_count": int_number(mac_metal_summary.get("kernel_parity_pending_count")),
        "native_hot_loop_parity_claim_allowed": mac_metal_summary.get("native_hot_loop_parity_claim_allowed"),
    }
    runtime_cost = {
        "private_admissibility_runtime_ms": int_number(adm.get("runtime_ms")),
        "public_shape_generator_runtime_ms": int_number(public_shape_generator.get("runtime_ms")),
        "mac_mlx_structural_action_train_wall_time_ms": accelerator["mlx_structural_action_train_wall_time_ms"],
    }
    recommendation = recommend(
        private_semantic_ready,
        public_promotion_ready,
        no_cheat_clean,
        substrate,
        public_transfer_wall,
        public_candidate_coverage,
        operator_lock_active,
    )
    gates = [
        gate("private_semantic_transfer_green", private_semantic_ready, {
            "private_admissibility": private_admissibility.get("trigger_state"),
            "private_residual_gate": private_residual_gate.get("trigger_state"),
            "selected_pass_rate": adm.get("selected_pass_rate"),
            "learned_candidate_pass_rate": heldout.get("learned_candidate_task_pass_rate"),
            "no_admissible_task_rate": adm.get("no_admissible_task_rate"),
        }),
        gate("no_cheat_clean", no_cheat_clean, no_cheat),
        gate("strict_public_promotion_not_claimed", not public_promotion_ready, {
            "strict_public_promotion_candidate_count": public_candidate_coverage[
                "benchmark_promotion_eligible_candidate_count"
            ],
            "operator_lock_active": operator_lock_active,
        }),
        gate("public_shaped_candidate_coverage_fresh_green", public_candidate_coverage_ready, public_candidate_coverage),
        gate("public_transfer_wall_still_present", public_transfer_wall["locked_broad_public_pass_rate"] < 0.90, public_transfer_wall),
        gate("mac_mlx_evidence_present", mac_mlx.get("trigger_state") == "GREEN", accelerator),
        gate("vcm_delta_evidence_present", vcm_delta["public_memory_prompt_state"] == "GREEN", vcm_delta),
        gate(
            "private_residual_targets_prepared_without_public_training_rows",
            int_number(public_residual_mining_current_summary.get("private_residual_target_rows_written")) > 0
            and int_number(public_residual_mining_current_summary.get("training_rows_written")) == 0,
            {
                "target_rows": int_number(public_residual_mining_current_summary.get("private_residual_target_rows_written")),
                "target_manifest": public_residual_mining_current_summary.get("private_residual_target_manifest"),
                "training_rows_written": int_number(public_residual_mining_current_summary.get("training_rows_written")),
            },
        ),
        gate(
            "private_residual_targets_consumed_without_public_training_rows",
            private_residual_target_consumer.get("trigger_state") in {"GREEN", "YELLOW"}
            and int_number(private_residual_target_consumer_summary.get("valid_target_rows")) > 0
            and int_number(private_residual_target_consumer_summary.get("repair_queue_rows")) > 0
            and int_number(private_residual_target_consumer_summary.get("training_rows_written")) == 0
            and int_number(private_residual_target_consumer_summary.get("public_training_rows_written")) == 0,
            {
                "trigger_state": private_residual_target_consumer.get("trigger_state"),
                "valid_target_rows": int_number(private_residual_target_consumer_summary.get("valid_target_rows")),
                "repair_queue_rows": int_number(private_residual_target_consumer_summary.get("repair_queue_rows")),
                "queue_path": private_residual_target_consumer_summary.get("queue_path"),
                "training_rows_written": int_number(private_residual_target_consumer_summary.get("training_rows_written")),
                "public_training_rows_written": int_number(
                    private_residual_target_consumer_summary.get("public_training_rows_written")
                ),
            },
        ),
        gate(
            "private_residual_targets_all_closed",
            private_residual_target_consumer.get("trigger_state") == "GREEN"
            and int_number(private_residual_target_consumer_summary.get("unresolved_target_count")) == 0
            and int_number(private_residual_target_consumer_summary.get("blocked_target_count")) == 0,
            {
                "trigger_state": private_residual_target_consumer.get("trigger_state"),
                "unresolved_target_count": int_number(
                    private_residual_target_consumer_summary.get("unresolved_target_count")
                ),
                "unresolved_target_state_counts": private_residual_target_consumer_summary.get(
                    "unresolved_target_state_counts", {}
                ),
                "unresolved_target_category_counts": private_residual_target_consumer_summary.get(
                    "unresolved_target_category_counts", {}
                ),
                "blocked_target_count": int_number(private_residual_target_consumer_summary.get("blocked_target_count")),
                "target_coverage_rate": number(private_residual_target_consumer_summary.get("target_coverage_rate")),
            },
        ),
    ]
    trigger_state = "YELLOW" if private_semantic_ready and no_cheat_clean else "RED"
    return {
        "policy": "project_theseus_capability_transfer_closure_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "inputs": {name: rel(path) for name, path in inputs.items()},
        "summary": {
            "private_semantic_ready": private_semantic_ready,
            "public_promotion_ready": public_promotion_ready,
            "no_cheat_clean": no_cheat_clean,
            "private_selected_pass_rate": number(adm.get("selected_pass_rate")),
            "private_pass_if_any_rate": number(adm.get("pass_if_any_rate")),
            "private_no_admissible_task_rate": number(adm.get("no_admissible_task_rate")),
            "private_semantic_eligible_candidate_count": int_number(adm.get("private_semantic_eligible_candidate_count")),
            "private_learned_candidate_pass_rate": number(heldout.get("learned_candidate_task_pass_rate")),
            "public_candidate_coverage_ready": public_candidate_coverage_ready,
            "strict_public_promotion_candidate_count": public_candidate_coverage[
                "benchmark_promotion_eligible_candidate_count"
            ],
            "public_candidate_coverage": public_candidate_coverage,
            "sts_private_selected_delta": first_number(obj(adm, "sts_on_vs_matched_sts_off"), "sts_delta_selected_pass_rate"),
            "sts_matched_selected_delta": number(residual.get("matched_sts_selected_pass_rate_delta")),
            "sts_matched_oracle_delta": number(residual.get("matched_sts_oracle_pass_rate_delta")),
            "vcm": vcm_delta,
            "public_transfer_wall": public_transfer_wall,
            "substrate": substrate,
            "accelerator": accelerator,
            "runtime_cost": runtime_cost,
            "no_cheat": no_cheat,
            "candidate_promotion_allowed": bool(promotion.get("candidate_promotion_allowed")),
        },
        "gates": gates,
        "recommendation": recommendation,
        "next_actions": next_actions(
            public_promotion_ready,
            public_transfer_wall,
            substrate,
            public_candidate_coverage=public_candidate_coverage,
            public_candidate_coverage_ready=public_candidate_coverage_ready,
            operator_lock_active=operator_lock_active,
        ),
    }


def recommend(
    private_semantic_ready: bool,
    public_promotion_ready: bool,
    no_cheat_clean: bool,
    substrate: dict[str, Any],
    public_transfer_wall: dict[str, Any],
    public_candidate_coverage: dict[str, Any],
    operator_lock_active: bool,
) -> dict[str, Any]:
    if not private_semantic_ready or not no_cheat_clean:
        return {
            "decision": "continue_private_candidate_repair",
            "reason": "Private semantic transfer or no-cheat hygiene is not clean enough for the next gate.",
        }
    target_consumer = public_transfer_wall.get("private_residual_target_consumer")
    if isinstance(target_consumer, dict) and int_number(target_consumer.get("unresolved_target_count")) > 0:
        return {
            "decision": "close_private_residual_target_alignment_queue_before_public_review",
            "survival_lane": "transformer_or_hybrid_structural_student_candidate_path",
            "discovery_lane": "symliquid_protected_but_not_survival_default",
            "reason": (
                "Private semantic transfer is clean, but "
                f"{target_consumer.get('unresolved_target_count')} residual target(s) are not closed by current private evidence. "
                "The open category is a calibration slice/candidate-manifest alignment audit, so public review should wait until "
                "the future one-shot packet can prove exact candidate/slice alignment before execution."
            ),
            "substrate_read": (
                "Matched neural seed evidence still favors the transformer control on proposer and budget-ladder means; "
                "keep SymLiquid as discovery unless it wins a matched transfer gate."
            )
            if substrate.get("code_proposer_best_arm") == "transformer_control"
            else "Substrate evidence is mixed; keep matched controls mandatory.",
        }
    if public_candidate_coverage.get("ready") and operator_lock_active:
        return {
            "decision": "private_candidate_coverage_ready_public_calibration_still_locked",
            "survival_lane": "transformer_or_hybrid_structural_student_candidate_path",
            "discovery_lane": "symliquid_protected_but_not_survival_default",
            "reason": (
                "Private semantic transfer is clean and the prompt-only public-shaped manifest now has "
                f"{public_candidate_coverage['benchmark_promotion_eligible_candidate_count']} strict full-body "
                "promotion-eligible candidates. Public scoring is still locked by the project charter; the locked "
                f"broad public score remains {public_transfer_wall['locked_broad_public_pass_rate']}."
            ),
            "substrate_read": (
                "Matched neural seed evidence still favors the transformer control on proposer and budget-ladder means; "
                "keep SymLiquid as discovery unless it wins a matched transfer gate."
            )
            if substrate.get("code_proposer_best_arm") == "transformer_control"
            else "Substrate evidence is mixed; keep matched controls mandatory.",
        }
    if public_candidate_coverage.get("semantic_structure_audit_state") != "GREEN":
        return {
            "decision": "build_real_multistatement_structural_decoder_before_public_spend",
            "survival_lane": "transformer_or_hybrid_structural_student_candidate_path",
            "discovery_lane": "symliquid_protected_but_not_survival_default",
            "reason": (
                "The prompt-only public-shaped manifest is clean on leakage, fallback, AST parseability, and return shape, "
                "but semantic structure is not ready: selected obligation satisfaction is "
                f"{public_candidate_coverage.get('semantic_structure_selected_obligation_satisfaction_rate')} and "
                f"{public_candidate_coverage.get('semantic_structure_multi_statement_candidate_count')} real multi-statement "
                "generated bodies were present."
            ),
            "substrate_read": (
                "Matched neural seed evidence still favors the transformer control on proposer and budget-ladder means; "
                "keep SymLiquid as discovery unless it wins a matched transfer gate."
            )
            if substrate.get("code_proposer_best_arm") == "transformer_control"
            else "Substrate evidence is mixed; keep matched controls mandatory.",
        }
    if not public_promotion_ready:
        return {
            "decision": "build_strict_public_promotion_candidate_manifest_before_public_spend",
            "survival_lane": "transformer_or_hybrid_structural_student_candidate_path",
            "discovery_lane": "symliquid_protected_but_not_survival_default",
            "reason": (
                "Private semantic full-body candidates now transfer on the v3 heldout set without fallback/template/leakage, "
                "but the strict public-promotion manifest is still not claimed and the locked broad public score remains "
                f"{public_transfer_wall['locked_broad_public_pass_rate']}."
            ),
            "substrate_read": (
                "Matched neural seed evidence still favors the transformer control on proposer and budget-ladder means; "
                "keep SymLiquid as discovery unless it wins a matched transfer gate."
            )
            if substrate.get("code_proposer_best_arm") == "transformer_control"
            else "Substrate evidence is mixed; keep matched controls mandatory.",
        }
    return {
        "decision": "operator_may_review_one_bounded_public_calibration",
        "reason": "Private candidate coverage and strict promotion evidence are clean; public calibration remains one-shot and locked.",
    }


def next_actions(
    public_promotion_ready: bool,
    public_transfer_wall: dict[str, Any],
    substrate: dict[str, Any],
    *,
    public_candidate_coverage: dict[str, Any] | None = None,
    public_candidate_coverage_ready: bool = False,
    operator_lock_active: bool = False,
) -> list[str]:
    actions = []
    if not public_candidate_coverage_ready:
        coverage = public_candidate_coverage or {}
        if coverage.get("semantic_structure_audit_state") != "GREEN":
            actions.append(
                "Build a real learned multi-statement structural body decoder; the current public-shaped manifest is expression-wrapped and static semantic-structure readiness is YELLOW."
            )
            actions.append(
                "Rerun the public-shaped prompt-only generator, return-contract audit, and semantic-structure audit; require zero fallback/templates/public tests and nonzero real multi-statement generated bodies."
            )
        else:
            actions.append(
                "Generate a strict public-shaped candidate manifest from the repaired learned path, with public task prompts only and no public tests/solutions/templates."
            )
            actions.append(
                "Run the candidate coverage/promotion gate on that manifest without executing public tests; require nonzero promotion-eligible full-body candidates."
            )
    elif operator_lock_active and not public_promotion_ready:
        actions.append(
            "Public-shaped candidate coverage is ready; keep public calibration locked until an explicit one-shot calibration decision exists."
        )
        target_consumer = public_transfer_wall.get("private_residual_target_consumer")
        if isinstance(target_consumer, dict) and int_number(target_consumer.get("unresolved_target_count")) > 0:
            actions.append(
                "Close the private residual target queue before public review: "
                f"{target_consumer.get('unresolved_target_count')} target(s) remain unresolved, "
                f"with categories {target_consumer.get('unresolved_target_category_counts')}."
            )
        current_mining = public_transfer_wall.get("current_full_body_residual_mining")
        if isinstance(current_mining, dict) and current_mining.get("candidate_manifest_score_claim_allowed") is False:
            alignment = current_mining.get("candidate_manifest_slice_alignment")
            missing = alignment.get("missing_residual_task_count") if isinstance(alignment, dict) else None
            actions.append(
                "Before any future one-shot public calibration, freeze the exact task slice and candidate manifest together; "
                f"the current full-body residual miner is hash-clean but has {missing} residual tasks missing from that spent slice, "
                "so it is alignment evidence rather than a score claim."
            )
    if public_transfer_wall["locked_broad_public_pass_rate"] < 0.90 and not public_candidate_coverage_ready:
        actions.append(
            "Keep public calibration locked until the candidate-coverage gate is fresh and green; do not rerun the spent public surface to fish."
        )
    if number(substrate.get("code_proposer_symliquid_minus_transformer")) < 0:
        actions.append(
            "Use the transformer or hybrid structural path as the survival lane for utility while keeping SymLiquid in matched discovery experiments."
        )
    actions.append(
        "Use VCM as an integrated context layer for long-context tasks; semantic runtime-cache lifecycle is proven, "
        "native KV/prefix lifecycle is proven through the local no-download Transformers/Torch tiny-model DynamicCache forward-pass route when VCM-Runtime is GREEN; "
        "KV-aware cross-backend scheduling and MLX/CUDA/Metal-specific KV parity remain unclaimed."
    )
    return actions


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def obj(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key) if isinstance(row, dict) else None
    return value if isinstance(value, dict) else {}


def first_number(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in row and row.get(key) is not None:
            return number(row.get(key))
    return None


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def int_number(value: Any) -> int:
    return int(number(value))


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    public_wall = summary["public_transfer_wall"]
    substrate = summary["substrate"]
    accelerator = summary["accelerator"]
    vcm = summary["vcm"]
    public_coverage = summary["public_candidate_coverage"]
    runtime_cost = summary["runtime_cost"]
    lines = [
        "# Capability Transfer Closure v1",
        "",
        f"State: **{report['trigger_state']}**",
        "",
        "## Core Evidence",
        "",
        f"- Private semantic ready: {summary['private_semantic_ready']}",
        f"- Private selected pass rate: {summary['private_selected_pass_rate']}",
        f"- Private pass-if-any rate: {summary['private_pass_if_any_rate']}",
        f"- Private no-admissible rate: {summary['private_no_admissible_task_rate']}",
        f"- Private semantic eligible candidates: {summary['private_semantic_eligible_candidate_count']}",
        f"- Public-shaped candidate coverage ready: {summary['public_candidate_coverage_ready']}",
        f"- Public-shaped tasks/candidates: {public_coverage['task_count']} / {public_coverage['candidate_count']}",
        f"- Strict public-promotion candidates: {summary['strict_public_promotion_candidate_count']}",
        f"- Public-shaped curated training sources/rows: {public_coverage.get('ready_training_sources')} / {public_coverage.get('training_rows_used')}",
        f"- Public return-contract audit: {public_coverage.get('return_contract_audit_state')}",
        f"- Public return-contract selected/any compatible: {public_coverage.get('return_contract_selected_shape_compatible_task_rate')} / {public_coverage.get('return_contract_any_shape_compatible_task_rate')}",
        f"- Public semantic-structure audit: {public_coverage.get('semantic_structure_audit_state')}",
        f"- Public semantic-structure selected/best-single obligation satisfaction: {public_coverage.get('semantic_structure_selected_obligation_satisfaction_rate')} / {public_coverage.get('semantic_structure_best_single_obligation_satisfaction_rate')}",
        f"- Public semantic-structure fragmented union-only tasks: {public_coverage.get('semantic_structure_fragmented_candidate_union_only_task_count')}",
        f"- Public semantic-structure multi-statement candidate rate: {public_coverage.get('semantic_structure_multi_statement_candidate_rate')}",
        f"- No-cheat clean: {summary['no_cheat_clean']}",
        f"- STS selected delta: {summary['sts_private_selected_delta']}",
        f"- Matched STS selected delta: {summary['sts_matched_selected_delta']}",
        f"- Matched STS oracle delta: {summary['sts_matched_oracle_delta']}",
        f"- Runtime ms, private/public-shaped/MLX train: {runtime_cost['private_admissibility_runtime_ms']} / {runtime_cost['public_shape_generator_runtime_ms']} / {runtime_cost['mac_mlx_structural_action_train_wall_time_ms']}",
        "",
        "## Public Transfer Wall",
        "",
        f"- Locked broad public score: {public_wall['locked_broad_public_pass_count']}/{public_wall['locked_broad_public_task_count']} = {public_wall['locked_broad_public_pass_rate']}",
        f"- Last public audit score: {public_wall['last_public_audit_pass_count']}/{public_wall['last_public_audit_task_count']} = {public_wall['last_public_audit_pass_rate']}",
        f"- Operator lock active: {public_wall['operator_lock_active']}",
    ]
    current_mining = public_wall.get("current_full_body_residual_mining")
    if isinstance(current_mining, dict):
        alignment = current_mining.get("candidate_manifest_slice_alignment")
        if not isinstance(alignment, dict):
            alignment = {}
        lines.extend(
            [
                f"- Current full-body residual mining state: {current_mining.get('trigger_state')}",
                f"- Current full-body score claim allowed: {current_mining.get('candidate_manifest_score_claim_allowed')}",
                f"- Current full-body residual slice coverage: {alignment.get('covered_residual_task_count')}/{alignment.get('residual_task_count')} = {alignment.get('covered_residual_task_rate')}",
                f"- Current full-body residual categories: {current_mining.get('residual_category_counts')}",
                "",
            ]
        )
    else:
        lines.append("")
    target_consumer = public_wall.get("private_residual_target_consumer")
    if isinstance(target_consumer, dict):
        lines.extend(
            [
                "## Private Residual Targets",
                "",
                f"- Consumer state: {target_consumer.get('trigger_state')}",
                f"- Covered targets: {target_consumer.get('covered_target_count')}/{target_consumer.get('valid_target_rows')}",
                f"- Unresolved targets: {target_consumer.get('unresolved_target_count')}",
                f"- Unresolved categories: {target_consumer.get('unresolved_target_category_counts')}",
                f"- Training rows written: {target_consumer.get('training_rows_written')}",
                f"- Public training rows written: {target_consumer.get('public_training_rows_written')}",
                "",
            ]
        )
    lines.extend([
        "## VCM And Mac",
        "",
        f"- VCM public memory prompt on/off: {vcm['public_memory_prompt_vcm_on_pass_rate']} / {vcm['public_memory_prompt_vcm_off_pass_rate']}",
        f"- VCM over flat-tail delta: {vcm['public_memory_prompt_vcm_over_flat_tail_delta']}",
        f"- VCM-Runtime state: {vcm['runtime_profile_state']}",
        f"- MLX state: {accelerator['mac_mlx_state']} on {accelerator['mlx_default_device']}",
        f"- MLX structural-action verifier pass rate: {accelerator['mlx_structural_action_verifier_pass_rate']}",
        f"- Metal production route allowed: {accelerator['metal_production_route_allowed']}",
        f"- Native parity claim allowed: {accelerator['native_hot_loop_parity_claim_allowed']}",
        "",
        "## Substrate Read",
        "",
        f"- Code proposer best arm: {substrate['code_proposer_best_arm']}",
        f"- Code proposer SymLiquid minus transformer: {substrate['code_proposer_symliquid_minus_transformer']}",
        f"- Token decoder best arm: {substrate['token_decoder_best_arm']}",
        f"- Token decoder SymLiquid minus transformer: {substrate['token_decoder_symliquid_minus_transformer']}",
        f"- Budget ladder SymLiquid minus transformer mean: {substrate['budget_ladder_symliquid_minus_transformer_mean']}",
        "",
        "## Recommendation",
        "",
        f"- Decision: {report['recommendation']['decision']}",
        f"- Reason: {report['recommendation']['reason']}",
    ])
    for action in report["next_actions"]:
        lines.append(f"- Next: {action}")
    return "\n".join(lines) + "\n"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
