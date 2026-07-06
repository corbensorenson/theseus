#!/usr/bin/env python3
"""Pre-training readiness gate for the practical neural seed survival lane.

This gate closes the architecture/readiness question, not the capability
question. It consumes existing replay, integrity, private residual, MLX route,
and no-cheat receipts to decide whether the practical transformer/hybrid
survival lane is ready for the next governed training/adaptation pass.

It does not run training, public calibration, teacher inference, generation, or
verification. It also does not permit promotion while semantic behavior remains
weak.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import report_evidence_store  # noqa: E402


REPORTS = ROOT / "reports"
DEFAULT_C1_GATE = REPORTS / "correctness_generator_survival_lane_gate.json"
DEFAULT_FANOUT = REPORTS / "neural_seed_strict_generator_fanout_receipt.json"
DEFAULT_FANOUT_BUDGET8 = REPORTS / "neural_seed_strict_generator_fanout_receipt_budget8_private_probe.json"
DEFAULT_BODY_TRANSITION = REPORTS / "strict_generator_mlx_decode_eval_body_transition_guard_broad4_v1.json"
DEFAULT_OPERATION_CONDITION = REPORTS / "strict_generator_mlx_decode_eval_prompt_operation_condition_strict_broad4_v3.json"
DEFAULT_NEGATIVE_REPLAY_ADAPTATION = (
    REPORTS / "strict_generator_mlx_private_adaptation_source_condition_operation_integrity_negative_replay_smoke_v1.json"
)
DEFAULT_NEGATIVE_REPLAY_DECODE = (
    REPORTS / "strict_generator_mlx_decode_eval_source_condition_operation_integrity_negative_replay_broad4_v1.json"
)
DEFAULT_EXPRESSION_GUARD = REPORTS / "strict_generator_mlx_decode_eval_expression_value_isinstance_guard_broad4_v3.json"
DEFAULT_RESOURCE_ROUTE = REPORTS / "resource_mlx_route_readiness_gate.json"
DEFAULT_TRAIN_ONCE_FANOUT = REPORTS / "code_lm_train_once_fanout.json"
DEFAULT_OUT = REPORTS / "neural_seed_survival_readiness_gate.json"
DEFAULT_MARKDOWN = REPORTS / "neural_seed_survival_readiness_gate.md"

FORBIDDEN_FIELDS = {
    "tests",
    "hidden_tests",
    "solution",
    "solution_body",
    "solution_expr",
    "expected",
    "answer",
    "category",
    "source_task_id",
    "decoder_contract.return_shape",
    "decoder_contract.type_family",
    "decoder_contract.required_constructs",
    "public_benchmark_payloads",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c1-gate", default=rel(DEFAULT_C1_GATE))
    parser.add_argument("--fanout", default=rel(DEFAULT_FANOUT))
    parser.add_argument("--fanout-budget8", default=rel(DEFAULT_FANOUT_BUDGET8))
    parser.add_argument("--body-transition", default=rel(DEFAULT_BODY_TRANSITION))
    parser.add_argument("--operation-condition", default=rel(DEFAULT_OPERATION_CONDITION))
    parser.add_argument("--negative-replay-adaptation", default=rel(DEFAULT_NEGATIVE_REPLAY_ADAPTATION))
    parser.add_argument("--negative-replay-decode", default=rel(DEFAULT_NEGATIVE_REPLAY_DECODE))
    parser.add_argument("--expression-guard", default=rel(DEFAULT_EXPRESSION_GUARD))
    parser.add_argument("--resource-route", default=rel(DEFAULT_RESOURCE_ROUTE))
    parser.add_argument("--train-once-fanout", default=rel(DEFAULT_TRAIN_ONCE_FANOUT))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    inputs = {
        "c1_gate": resolve(args.c1_gate),
        "fanout": resolve(args.fanout),
        "fanout_budget8": resolve(args.fanout_budget8),
        "body_transition": resolve(args.body_transition),
        "operation_condition": resolve(args.operation_condition),
        "negative_replay_adaptation": resolve(args.negative_replay_adaptation),
        "negative_replay_decode": resolve(args.negative_replay_decode),
        "expression_guard": resolve(args.expression_guard),
        "resource_route": resolve(args.resource_route),
        "train_once_fanout": resolve(args.train_once_fanout),
    }
    report = build_report(inputs, started=started)
    report_evidence_store.write_json_report(
        resolve(args.out),
        report,
        markdown_path=resolve(args.markdown_out),
        markdown_text=render_markdown(report),
    )
    view = gate_view(report) if args.gate else report
    print(json.dumps(view, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def build_report(inputs: dict[str, Path], *, started: float) -> dict[str, Any]:
    reports = {name: read_json(path) for name, path in inputs.items()}
    summaries = {name: summary(report) for name, report in reports.items()}

    checks = [
        check_c1_gate(summaries["c1_gate"]),
        check_fanout_receipt(summaries["fanout"]),
        check_fanout_width_not_the_wall(summaries["fanout"], summaries["fanout_budget8"]),
        check_body_transition_control(summaries["body_transition"]),
        check_operation_conditioning_control(summaries["operation_condition"]),
        check_negative_replay_adaptation(summaries["negative_replay_adaptation"]),
        check_negative_replay_decode(summaries["negative_replay_decode"]),
        check_expression_guard_control(summaries["expression_guard"]),
        check_resource_route_ready_but_blocked(summaries["resource_route"]),
        check_vcm_fanout_context(summaries["train_once_fanout"]),
        check_all_no_cheat_counters(summaries),
    ]
    expected_invalid_controls = [
        expected_invalid("old_router_template_ngram_credit_must_not_qualify", check_forbidden_credit_paths_blocked(summaries)),
        expected_invalid("promotion_must_remain_blocked_while_behavior_wall_exists", check_promotion_blocked(summaries)),
        expected_invalid("public_training_or_external_inference_fault_must_block", check_all_no_cheat_counters(summaries)),
        expected_invalid("selector_oracle_delta_zero_identifies_candidate_pool_wall", check_selector_oracle_wall(summaries["fanout"])),
        expected_invalid("fanout_width_increase_must_not_be_mislabeled_as_progress", check_fanout_width_not_the_wall(summaries["fanout"], summaries["fanout_budget8"])),
        expected_invalid("zero_behavior_resource_route_must_fail_closed", check_resource_route_ready_but_blocked(summaries["resource_route"])),
        expected_invalid("vcm_context_required_for_fanout_readiness", check_vcm_fanout_context(summaries["train_once_fanout"])),
        expected_invalid("integrity_mismatches_are_negative_evidence_not_promotion", check_negative_replay_decode(summaries["negative_replay_decode"])),
    ]
    failed_checks = [row for row in checks if not row["passed"]]
    failed_expected_invalid = [row for row in expected_invalid_controls if not row["passed"]]
    trigger_state = "GREEN" if not failed_checks and not failed_expected_invalid else "RED"

    fanout = summaries["fanout"]
    c1 = summaries["c1_gate"]
    operation = summaries["operation_condition"]
    negative_decode = summaries["negative_replay_decode"]
    expression = summaries["expression_guard"]
    resource = summaries["resource_route"]
    return {
        "policy": "project_theseus_neural_seed_survival_readiness_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "phase": 10,
            "phase_title": "Practical Neural Seed Survival Lane",
            "phase10_survival_lane_state": trigger_state,
            "phase10_survival_lane_support_state": "synthetic-test-backed",
            "readiness_scope": "architecture_ready_for_next_governed_training_or_adaptation_not_model_quality",
            "eligible_full_body_candidate_count": fanout.get("eligible_full_body_candidate_count"),
            "replayed_task_count": nested(fanout, "combined", "task_count"),
            "runtime_load_task_rate": nested(fanout, "combined", "runtime_load_task_rate"),
            "current_behavior_pass_rate": nested(fanout, "combined", "intended_behavior_pass_rate"),
            "budget8_behavior_pass_rate": nested(summaries["fanout_budget8"], "combined", "intended_behavior_pass_rate"),
            "c1_pass_if_any_rate": c1.get("pass_if_any_rate"),
            "c1_selected_behavior_pass_rate": c1.get("selected_intended_behavior_pass_rate"),
            "operation_condition_integrity_verified_count": nested(operation, "candidate_integrity", "integrity_verified_candidate_count"),
            "operation_condition_nontrivial_return_rate": nested(operation, "split_nontrivial_return_rates", "broad_private_heldout"),
            "operation_condition_private_passes": nested(operation, "split_passes", "broad_private_heldout"),
            "negative_replay_integrity_verified_count": nested(negative_decode, "candidate_integrity", "integrity_verified_candidate_count"),
            "negative_replay_private_passes": nested(negative_decode, "split_passes", "broad_private_heldout"),
            "expression_guard_private_passes": nested(expression, "split_passes", "broad_private_heldout"),
            "resource_route_state": resource.get("phase8_resource_mlx_route_state"),
            "resource_route_production_eligible": resource.get("production_route_eligible"),
            "resource_route_block_reason": resource.get("production_route_block_reason"),
            "check_count": len(checks),
            "failed_check_count": len(failed_checks),
            "expected_invalid_count": len(expected_invalid_controls),
            "failed_expected_invalid_count": len(failed_expected_invalid),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "inputs": {name: rel(path) for name, path in inputs.items()},
        "checks": checks,
        "expected_invalid_controls": expected_invalid_controls,
        "hard_gaps": failed_checks + failed_expected_invalid,
        "support_state_basis": {
            "candidate_generation_contract_bound": True,
            "direct_body_token_generation_measured": True,
            "candidate_integrity_independent": True,
            "private_verifier_replay_present": True,
            "blind_residual_and_selector_ablation_present": True,
            "negative_replay_training_pressure_present": True,
            "vcm_context_attached_to_fanout": True,
            "resource_route_ready_but_fail_closed_on_behavior": True,
            "promotion_claim_allowed": False,
            "public_calibration_allowed": False,
        },
        "non_claims": [
            "This gate does not claim promotion-grade learned code generation.",
            "This gate does not claim public transfer improvement.",
            "This gate does not run or authorize public calibration.",
            "This gate does not run training, teacher inference, or external inference.",
            "Routers, templates, deterministic tools, ngrams, structural adapters, and fallback returns remain non-credit for learned-generation claims.",
            "The next step is a governed private training/adaptation pass over the existing survival lane, not a new side lane.",
        ],
        "next_governed_step": {
            "owner_surface": "neural_seed_and_decoder",
            "implementation_id": "impl.neural_seed_strict_token_decoder.v1",
            "task": "private semantic update/final-return adaptation",
            "allowed_inputs": [
                "prompt/signature source text",
                "governed private/licensed rows",
                "integrity-negative private replay rows",
                "private verifier labels attached after generation",
                "generated-prefix AST/state features",
                "VCM context packets with public calibration taint excluded",
            ],
            "forbidden_inputs": sorted(FORBIDDEN_FIELDS | {"public_benchmark_training_rows", "teacher_runtime_tokens", "fallback_return_templates"}),
            "success_evidence_required_after_training": [
                "same private heldout split improves behavior without public/external/fallback/tool credit",
                "candidate integrity remains independently clean",
                "selector/pass-if-any and selected-pass rates improve because candidate semantics improve",
                "public calibration remains proposal-gated until private evidence justifies it",
            ],
        },
    }


def check_c1_gate(c1: dict[str, Any]) -> dict[str, Any]:
    return check(
        "c1_generator_survival_gate_records_clean_falsifying_wall",
        c1.get("c1_correctness_generator_survival_lane_state") == "GREEN"
        and c1.get("c1_synthetic_support_ready") is True
        and int_value(c1.get("eligible_candidate_count")) >= 1
        and float_value(c1.get("pass_if_any_rate")) == 0.0
        and c1.get("falsifying_wall_recorded") is True
        and int_value(c1.get("public_training_rows_written")) == 0
        and int_value(c1.get("external_inference_calls")) == 0
        and int_value(c1.get("fallback_return_count")) == 0,
        {
            "state": c1.get("c1_correctness_generator_survival_lane_state"),
            "support_state": c1.get("c1_correctness_generator_survival_lane_support_state"),
            "eligible_candidate_count": c1.get("eligible_candidate_count"),
            "pass_if_any_rate": c1.get("pass_if_any_rate"),
            "falsifying_wall_recorded": c1.get("falsifying_wall_recorded"),
            "public_training_rows_written": c1.get("public_training_rows_written"),
            "external_inference_calls": c1.get("external_inference_calls"),
            "fallback_return_count": c1.get("fallback_return_count"),
        },
    )


def check_fanout_receipt(fanout: dict[str, Any]) -> dict[str, Any]:
    residual = dict_value(fanout.get("semantic_residual_diagnosis"))
    selector = dict_value(fanout.get("selector_ablation"))
    combined = dict_value(fanout.get("combined"))
    forbidden = {str(item) for item in list_value(residual.get("forbidden_fields_excluded"))}
    selector_forbidden = {str(item) for item in list_value(selector.get("forbidden_fields_excluded"))}
    return check(
        "strict_fanout_receipt_measures_direct_learned_candidates",
        int_value(fanout.get("eligible_full_body_candidate_count")) >= 90
        and int_value(fanout.get("integrity_mismatch_count")) == 0
        and float_value(combined.get("runtime_load_task_rate")) >= 1.0
        and float_value(combined.get("intended_behavior_pass_rate")) > 0.0
        and residual.get("failed_runtime_loaded_task_count") == 17
        and selector.get("selector_diagnosis") == "candidate_pool_semantic_quality_gap"
        and int_value(selector.get("baseline_to_oracle_pass_delta")) == 0
        and FORBIDDEN_FIELDS.issubset(forbidden)
        and FORBIDDEN_FIELDS.issubset(selector_forbidden)
        and fanout.get("learned_generation_claim_allowed") is False
        and fanout.get("model_promotion_allowed") is False
        and int_value(fanout.get("public_training_rows_written")) == 0
        and int_value(fanout.get("external_inference_calls")) == 0
        and int_value(fanout.get("fallback_return_count")) == 0,
        {
            "eligible_full_body_candidate_count": fanout.get("eligible_full_body_candidate_count"),
            "integrity_mismatch_count": fanout.get("integrity_mismatch_count"),
            "runtime_load_task_rate": combined.get("runtime_load_task_rate"),
            "intended_behavior_pass_rate": combined.get("intended_behavior_pass_rate"),
            "selector_diagnosis": selector.get("selector_diagnosis"),
            "baseline_to_oracle_pass_delta": selector.get("baseline_to_oracle_pass_delta"),
            "learned_generation_claim_allowed": fanout.get("learned_generation_claim_allowed"),
            "model_promotion_allowed": fanout.get("model_promotion_allowed"),
        },
    )


def check_fanout_width_not_the_wall(fanout: dict[str, Any], budget8: dict[str, Any]) -> dict[str, Any]:
    base = nested(fanout, "combined", "intended_behavior_pass_rate")
    wider = nested(budget8, "combined", "intended_behavior_pass_rate")
    return check(
        "wider_fanout_does_not_mask_semantic_candidate_pool_wall",
        float_value(base, -1.0) == float_value(wider, -2.0)
        and int_value(budget8.get("max_candidates_per_task")) >= 8
        and int_value(budget8.get("integrity_mismatch_count")) == 0
        and int_value(budget8.get("public_training_rows_written")) == 0
        and int_value(budget8.get("external_inference_calls")) == 0
        and int_value(budget8.get("fallback_return_count")) == 0,
        {
            "base_behavior_pass_rate": base,
            "budget8_behavior_pass_rate": wider,
            "max_candidates_per_task": budget8.get("max_candidates_per_task"),
            "integrity_mismatch_count": budget8.get("integrity_mismatch_count"),
        },
    )


def check_body_transition_control(report: dict[str, Any]) -> dict[str, Any]:
    return check_decode_control(
        "body_transition_control_removes_decode_starvation_but_keeps_quality_wall",
        report,
        min_integrity_verified=8,
        expected_nontrivial_return_rate=0.0,
    )


def check_operation_conditioning_control(report: dict[str, Any]) -> dict[str, Any]:
    return check_decode_control(
        "operation_conditioning_control_has_nontrivial_returns_without_behavior_win",
        report,
        min_integrity_verified=8,
        expected_nontrivial_return_rate=1.0,
    )


def check_decode_control(name: str, report: dict[str, Any], *, min_integrity_verified: int, expected_nontrivial_return_rate: float) -> dict[str, Any]:
    integrity = dict_value(report.get("candidate_integrity"))
    starvation = nested(report, "split_decode_starvation", "broad_private_heldout", "zero_candidate_task_count")
    nontrivial = nested(report, "split_nontrivial_return_rates", "broad_private_heldout")
    passes = nested(report, "split_passes", "broad_private_heldout")
    return check(
        name,
        int_value(report.get("generated_candidate_rows")) >= min_integrity_verified
        and int_value(integrity.get("integrity_verified_candidate_count")) >= min_integrity_verified
        and int_value(starvation) == 0
        and float_value(nontrivial, -1.0) == expected_nontrivial_return_rate
        and int_value(passes) == 0
        and int_value(report.get("public_training_rows")) == 0
        and int_value(report.get("external_inference_calls")) == 0
        and int_value(report.get("fallback_template_router_tool_credit_count")) == 0,
        {
            "generated_candidate_rows": report.get("generated_candidate_rows"),
            "integrity_verified_candidate_count": integrity.get("integrity_verified_candidate_count"),
            "zero_candidate_task_count": starvation,
            "nontrivial_return_rate": nontrivial,
            "private_passes": passes,
            "public_training_rows": report.get("public_training_rows"),
            "external_inference_calls": report.get("external_inference_calls"),
            "fallback_template_router_tool_credit_count": report.get("fallback_template_router_tool_credit_count"),
        },
    )


def check_negative_replay_adaptation(report: dict[str, Any]) -> dict[str, Any]:
    negative = dict_value(report.get("negative_replay_unlikelihood"))
    loop_semantic = dict_value(report.get("loop_semantic_operation_weighting"))
    semantic_plan = dict_value(report.get("semantic_plan_visible_operation_weighting"))
    return check(
        "integrity_negative_replay_training_pressure_is_wired_without_generation_credit",
        bool(negative.get("active"))
        and int_value(negative.get("candidate_integrity_verified_rows")) >= 12
        and int_value(negative.get("candidate_integrity_unverified_rows")) >= 4
        and bool(loop_semantic.get("enabled"))
        and bool(semantic_plan.get("enabled"))
        and int_value(negative.get("candidate_generation_credit")) == 0
        and int_value(loop_semantic.get("candidate_generation_credit")) == 0
        and int_value(semantic_plan.get("candidate_generation_credit")) == 0
        and report.get("public_calibration_eligible") is False
        and int_value(report.get("public_training_rows")) == 0
        and int_value(report.get("external_inference_calls")) == 0
        and int_value(report.get("fallback_template_router_tool_credit_count")) == 0,
        {
            "negative_replay_active": negative.get("active"),
            "candidate_integrity_verified_rows": negative.get("candidate_integrity_verified_rows"),
            "candidate_integrity_unverified_rows": negative.get("candidate_integrity_unverified_rows"),
            "loop_semantic_operation_enabled": loop_semantic.get("enabled"),
            "semantic_plan_visible_operation_enabled": semantic_plan.get("enabled"),
            "public_calibration_eligible": report.get("public_calibration_eligible"),
        },
    )


def check_negative_replay_decode(report: dict[str, Any]) -> dict[str, Any]:
    integrity = dict_value(report.get("candidate_integrity"))
    return check(
        "negative_replay_decode_keeps_integrity_faults_as_negative_evidence",
        int_value(report.get("generated_candidate_rows")) >= 7
        and int_value(integrity.get("integrity_verified_candidate_count")) >= 6
        and int_value(integrity.get("integrity_mismatch_count")) >= 1
        and float_value(nested(report, "split_nontrivial_return_rates", "broad_private_heldout")) >= 0.8
        and int_value(nested(report, "split_passes", "broad_private_heldout")) == 0
        and int_value(report.get("public_training_rows")) == 0
        and int_value(report.get("external_inference_calls")) == 0
        and int_value(report.get("fallback_template_router_tool_credit_count")) == 0,
        {
            "generated_candidate_rows": report.get("generated_candidate_rows"),
            "integrity_verified_candidate_count": integrity.get("integrity_verified_candidate_count"),
            "integrity_mismatch_count": integrity.get("integrity_mismatch_count"),
            "nontrivial_return_rate": nested(report, "split_nontrivial_return_rates", "broad_private_heldout"),
            "private_passes": nested(report, "split_passes", "broad_private_heldout"),
        },
    )


def check_expression_guard_control(report: dict[str, Any]) -> dict[str, Any]:
    guard = dict_value(report.get("decode_guard"))
    integrity = dict_value(report.get("candidate_integrity"))
    return check(
        "expression_value_guard_is_enabled_and_keeps_pathology_as_negative_evidence",
        guard.get("enable_expression_value_guard") is True
        and guard.get("enable_expression_closure_guard") is True
        and guard.get("uses_public_data") is False
        and guard.get("uses_eval_tests_or_solutions") is False
        and int_value(report.get("generated_candidate_rows")) >= 6
        and int_value(integrity.get("integrity_mismatch_count")) >= 1
        and int_value(nested(report, "split_passes", "broad_private_heldout")) == 0
        and int_value(report.get("public_training_rows")) == 0
        and int_value(report.get("external_inference_calls")) == 0
        and int_value(report.get("fallback_template_router_tool_credit_count")) == 0,
        {
            "decode_guard": guard,
            "generated_candidate_rows": report.get("generated_candidate_rows"),
            "integrity_mismatch_count": integrity.get("integrity_mismatch_count"),
            "private_passes": nested(report, "split_passes", "broad_private_heldout"),
        },
    )


def check_resource_route_ready_but_blocked(report: dict[str, Any]) -> dict[str, Any]:
    return check(
        "mlx_resource_route_ready_and_production_route_fail_closed_on_zero_behavior",
        report.get("phase8_resource_mlx_route_state") == "GREEN"
        and report.get("production_route_eligible") is False
        and report.get("production_route_block_reason") == "fail_closed_behavior_quality_zero"
        and report.get("parity_claim_allowed") is False,
        {
            "phase8_resource_mlx_route_state": report.get("phase8_resource_mlx_route_state"),
            "production_route_eligible": report.get("production_route_eligible"),
            "production_route_block_reason": report.get("production_route_block_reason"),
            "parity_claim_allowed": report.get("parity_claim_allowed"),
        },
    )


def check_vcm_fanout_context(report: dict[str, Any]) -> dict[str, Any]:
    receipt = dict_value(report.get("vcm_context_governor_receipt"))
    return check(
        "vcm_context_governor_attached_to_train_once_fanout_contract",
        report.get("vcm_context_governor_ready") is True
        and report.get("vcm_context_governor_state") == "GREEN"
        and report.get("vcm_context_adequacy_state") == "governed_sufficient_for_generation_fanout"
        and receipt.get("trigger_state") == "GREEN"
        and int_value(receipt.get("hard_gap_count")) == 0
        and int_value(report.get("external_inference_calls")) == 0,
        {
            "trigger_state": report.get("trigger_state"),
            "vcm_context_governor_ready": report.get("vcm_context_governor_ready"),
            "vcm_context_governor_state": report.get("vcm_context_governor_state"),
            "vcm_context_adequacy_state": report.get("vcm_context_adequacy_state"),
            "receipt": receipt,
        },
    )


def check_all_no_cheat_counters(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    faults = {}
    for name, report in summaries.items():
        public_rows = int_value(report.get("public_training_rows_written")) + int_value(report.get("public_training_rows"))
        external = int_value(report.get("external_inference_calls"))
        fallback = (
            int_value(report.get("fallback_return_count"))
            + int_value(report.get("fallback_template_router_tool_credit_count"))
            + int_value(report.get("fallback_return_candidate_count"))
        )
        if public_rows or external or fallback:
            faults[name] = {"public_training_rows": public_rows, "external_inference_calls": external, "fallback_or_tool_credit": fallback}
    return check("all_survival_readiness_no_cheat_counters_zero", not faults, faults)


def check_forbidden_credit_paths_blocked(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fanout = summaries["fanout"]
    c1 = summaries["c1_gate"]
    return check(
        "forbidden_credit_paths_are_not_counted_as_learned_generation",
        fanout.get("learned_generation_claim_allowed") is False
        and fanout.get("model_promotion_allowed") is False
        and int_value(c1.get("fallback_return_candidate_count")) == 0
        and int_value(c1.get("unconditional_constant_return_candidate_count")) == 0,
        {
            "learned_generation_claim_allowed": fanout.get("learned_generation_claim_allowed"),
            "model_promotion_allowed": fanout.get("model_promotion_allowed"),
            "fallback_return_candidate_count": c1.get("fallback_return_candidate_count"),
            "unconditional_constant_return_candidate_count": c1.get("unconditional_constant_return_candidate_count"),
        },
    )


def check_promotion_blocked(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fanout = summaries["fanout"]
    c1 = summaries["c1_gate"]
    return check(
        "promotion_and_public_calibration_stay_blocked_until_semantic_quality_improves",
        fanout.get("model_promotion_allowed") is False
        and fanout.get("learned_generation_claim_allowed") is False
        and float_value(c1.get("pass_if_any_rate"), -1.0) == 0.0
        and float_value(c1.get("selected_intended_behavior_pass_rate"), -1.0) == 0.0,
        {
            "model_promotion_allowed": fanout.get("model_promotion_allowed"),
            "learned_generation_claim_allowed": fanout.get("learned_generation_claim_allowed"),
            "c1_pass_if_any_rate": c1.get("pass_if_any_rate"),
            "c1_selected_intended_behavior_pass_rate": c1.get("selected_intended_behavior_pass_rate"),
        },
    )


def check_selector_oracle_wall(fanout: dict[str, Any]) -> dict[str, Any]:
    selector = dict_value(fanout.get("selector_ablation"))
    return check(
        "selector_oracle_delta_zero_proves_pool_semantic_quality_wall",
        selector.get("selector_diagnosis") == "candidate_pool_semantic_quality_gap"
        and int_value(selector.get("baseline_to_oracle_pass_delta")) == 0,
        {
            "selector_diagnosis": selector.get("selector_diagnosis"),
            "baseline_to_oracle_pass_delta": selector.get("baseline_to_oracle_pass_delta"),
            "baseline_to_blind_pass_delta": selector.get("baseline_to_blind_pass_delta"),
        },
    )


def check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def expected_invalid(name: str, check_row: dict[str, Any]) -> dict[str, Any]:
    row = dict(check_row)
    row["name"] = name
    row["expected_invalid_control"] = True
    return row


def gate_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report["trigger_state"],
        "summary": report["summary"],
        "hard_gaps": report["hard_gaps"],
        "non_claims": report["non_claims"],
        "next_governed_step": report["next_governed_step"],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Neural Seed Survival Readiness Gate",
        "",
        f"- state: `{report['trigger_state']}`",
        f"- support: `{report['summary']['phase10_survival_lane_support_state']}`",
        f"- scope: `{report['summary']['readiness_scope']}`",
        f"- current behavior pass rate: `{report['summary']['current_behavior_pass_rate']}`",
        f"- C1 pass-if-any rate: `{report['summary']['c1_pass_if_any_rate']}`",
        "",
        "## Checks",
    ]
    for row in report["checks"]:
        mark = "PASS" if row["passed"] else "FAIL"
        lines.append(f"- `{mark}` {row['name']}")
    lines.extend(["", "## Non-Claims"])
    for item in report["non_claims"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Next Governed Step"])
    lines.append(str(report["next_governed_step"]["task"]))
    return "\n".join(lines) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summary(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("summary")
    return value if isinstance(value, dict) else report


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def nested(value: Any, *path: str) -> Any:
    current = value
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
