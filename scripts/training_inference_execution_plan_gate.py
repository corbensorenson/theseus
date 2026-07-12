#!/usr/bin/env python3
"""Gate the post-readiness training and inference execution roadmap.

This does not train a model and does not run public calibration. It checks that
the next execution plan is complete, tied to current evidence, and still obeys
the Project Theseus no-cheat boundaries before training/inference work begins.
"""

from __future__ import annotations

import argparse
import hashlib
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
from theseus_archive_resolver import is_archive_pointer, resolve_archived_path  # noqa: E402


REPORTS = ROOT / "reports"
DEFAULT_PLAN = ROOT / "configs" / "training_inference_execution_roadmap.json"
DEFAULT_OUT = REPORTS / "training_inference_execution_plan_gate.json"
DEFAULT_MARKDOWN = REPORTS / "training_inference_execution_plan_gate.md"

DEFAULT_CURRENT_REPORTS = {
    "roadmap_pre_training": REPORTS / "roadmap_pre_training_architecture_readiness_gate.json",
    "project_registry": REPORTS / "theseus_project_registry.json",
    "training_data_admission": REPORTS / "training_data_admission_v1.json",
    "neural_seed_survival": REPORTS / "neural_seed_survival_readiness_gate.json",
    "resource_mlx_route": REPORTS / "resource_mlx_route_readiness_gate.json",
    "t2_private_training_smoke": REPORTS / "strict_generator_mlx_pretraining_probe_t2_private_smoke_20260706.json",
    "assistant_product_lane": REPORTS / "theseus_assistant_product_lane_gate.json",
    "public_calibration_proposal": REPORTS / "public_calibration_proposal_gate.json",
}

REQUIRED_RULES = {
    "public_benchmarks_calibration_only": True,
    "fresh_public_surfaces_authorized_when_governed": True,
    "exact_consumed_public_surface_rerun_allowed": False,
    "public_benchmark_payloads_allowed_in_training": False,
    "licensed_open_nonbenchmark_training_allowed_with_provenance": True,
    "external_runtime_inference_allowed": False,
    "teacher_training_rows_allowed_only_through_distillation_gate": True,
    "teacher_tokens_allowed_at_runtime_serving": False,
    "fallback_template_router_tool_credit_as_learned_generation": False,
    "deterministic_tool_results_count_as_tool_assisted_not_model_only": True,
    "arbitrary_remote_execution_allowed": False,
    "raw_private_user_text_training_default": False,
    "candidate_integrity_independent_audit_required": True,
    "vcm_public_calibration_taint_excluded_from_training": True,
}

REQUIRED_LANES = [
    "T0_preflight_freeze",
    "T1_data_and_context_manifest",
    "T2_private_training_smoke",
    "T3_bounded_private_training_rung",
    "T4_private_eval_and_candidate_integrity",
    "T5_local_assisted_inference_canary",
    "T6_dogfood_feedback_to_training_rows",
    "T7_fresh_public_calibration_proposal",
    "T8_hive_fleet_training_scaleout",
]

ALLOWED_LANE_STATES = {
    "ready",
    "planned",
    "blocked_until_private_quality_positive",
    "blocked_external_peers_unreachable",
}

REQUIRED_LANE_FIELDS = {
    "id",
    "title",
    "status",
    "goal",
    "entry_criteria",
    "required_gates",
    "allowed_inputs",
    "forbidden_inputs",
    "evidence_outputs",
    "exit_criteria",
    "rollback_or_stop",
    "no_claims",
}

FORBIDDEN_TRAINING_TERMS = {
    "public benchmark prompts as training rows",
    "public benchmark tests or hidden tests",
    "public benchmark solutions or answer templates",
    "public benchmark payloads",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=rel(DEFAULT_PLAN))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    plan_path = resolve(args.plan)
    plan = read_json(plan_path)
    current_reports = {name: read_json(path) for name, path in DEFAULT_CURRENT_REPORTS.items()}
    report = build_report(plan_path, plan, current_reports, started=started)
    report_evidence_store.write_json_report(
        resolve(args.out),
        report,
        markdown_path=resolve(args.markdown_out),
        markdown_text=render_markdown(report),
    )
    view = gate_view(report) if args.gate else report
    print(json.dumps(view, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def build_report(plan_path: Path, plan: dict[str, Any], current_reports: dict[str, dict[str, Any]], *, started: float) -> dict[str, Any]:
    current = {name: compact_report(report) for name, report in current_reports.items()}
    checks = [
        check_plan_shape(plan),
        check_required_rules(plan),
        check_lane_inventory(plan),
        check_lane_shapes(plan),
        check_training_lanes_forbid_public_benchmarks(plan),
        check_public_calibration_lane_freshness_policy(plan),
        check_hive_lane_blocks_unreachable_peers(plan),
        check_current_architecture_ready(current),
        check_current_data_admission(current),
        check_current_neural_seed_readiness(current),
        check_current_resource_route(current),
        check_current_t2_training_smoke_if_present(current),
        check_current_assistant_canary(current),
        check_current_public_calibration_status(current),
    ]
    expected_invalid_controls = [
        expected_invalid("public_benchmark_training_payloads_must_be_refused", check_public_benchmark_training_refused(plan, current)),
        expected_invalid("external_runtime_inference_must_be_refused", check_external_runtime_refused(plan)),
        expected_invalid("fallback_template_router_tool_credit_must_be_refused", check_fallback_credit_refused(plan)),
        expected_invalid("exact_consumed_public_surface_rerun_must_be_refused", check_consumed_public_surface_refused(current)),
        expected_invalid("production_mlx_route_must_fail_closed_while_behavior_quality_zero", check_mlx_route_fail_closed(current)),
        expected_invalid("model_only_general_chat_serving_not_claimed", check_model_only_general_chat_not_claimed(plan)),
        expected_invalid("teacher_rows_outside_distillation_gate_must_be_refused", check_teacher_boundary(plan, current)),
        expected_invalid("hive_fleet_scaleout_must_block_when_peers_unreachable", check_hive_scaleout_blocked(plan)),
    ]
    failed_checks = [row for row in checks if not row["passed"]]
    failed_expected_invalid = [row for row in expected_invalid_controls if not row["passed"]]
    trigger_state = "GREEN" if not failed_checks and not failed_expected_invalid else "RED"
    t2_smoke_passed = current_t2_training_smoke_passed(current)
    architecture_ready = current["roadmap_pre_training"].get("pre_training_architecture_ready") is True
    bounded_rung_ready = ready_for_bounded_private_training_rung(
        plan,
        current,
        failed_checks,
        failed_expected_invalid,
        t2_smoke_passed=t2_smoke_passed,
    )
    summary = {
        "plan": rel(plan_path),
        "training_inference_execution_plan_state": trigger_state,
        "lane_count": len(list_dicts(plan.get("lanes"))),
        "required_lane_count": len(REQUIRED_LANES),
        "ready_lane_count": sum(1 for lane in list_dicts(plan.get("lanes")) if lane.get("status") == "ready"),
        "planned_lane_count": sum(1 for lane in list_dicts(plan.get("lanes")) if lane.get("status") == "planned"),
        "blocked_lane_count": sum(1 for lane in list_dicts(plan.get("lanes")) if str(lane.get("status", "")).startswith("blocked")),
        "ready_for_governed_private_training_focus": ready_for_private_training(current, failed_checks, failed_expected_invalid),
        "ready_for_private_training_smoke": ready_for_private_training_smoke(plan, current, failed_checks, failed_expected_invalid),
        "t2_private_training_smoke_passed": t2_smoke_passed,
        "ready_for_bounded_private_training_rung": bounded_rung_ready,
        "bounded_private_training_rung_block_reason": (
            "none"
            if bounded_rung_ready
            else (
                "pre-training architecture readiness is RED; complete or falsify the partial book-derived implementation phases before making training the primary focus"
                if not architecture_ready
                else "requires a clean T2 private training smoke checkpoint before longer training"
            )
        ),
        "ready_for_local_assisted_inference_canary": ready_for_local_assisted_inference(current, failed_checks, failed_expected_invalid),
        "ready_for_model_only_general_chat_runtime": False,
        "ready_for_public_calibration": False,
        "public_calibration_block_reason": "requires fresh non-consumed surface and positive private semantic behavior evidence first",
        "ready_for_production_mlx_route": False,
        "production_mlx_route_block_reason": current["resource_mlx_route"].get("production_route_block_reason"),
        "ready_for_hive_fleet_training": False,
        "hive_fleet_training_block_reason": "trusted peers currently unreachable from this Mac",
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "expected_invalid_count": len(expected_invalid_controls),
        "failed_expected_invalid_count": len(failed_expected_invalid),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }
    return {
        "policy": "project_theseus_training_inference_execution_plan_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": summary,
        "current_evidence": {name: rel(path) for name, path in DEFAULT_CURRENT_REPORTS.items()},
        "checks": checks,
        "expected_invalid_controls": expected_invalid_controls,
        "hard_gaps": failed_checks + failed_expected_invalid,
        "execution_order": list_value(plan.get("execution_order")),
        "readiness_decision": {
            "next_allowed_step": (
                "T3_bounded_private_training_rung"
                if bounded_rung_ready
                else (
                    "complete_partial_book_derived_phases_before_training_focus"
                    if not architecture_ready
                    else "T2_private_training_smoke"
                )
            ),
            "next_allowed_inference_step": (
                "T5_local_assisted_inference_canary"
                if ready_for_local_assisted_inference(current, failed_checks, failed_expected_invalid)
                else "none_until_gate_green"
            ),
            "next_disallowed_steps": [
                "model-only general chat serving",
                "production MLX route",
                "exact consumed public calibration rerun",
                "Hive fleet training while peers are unreachable",
            ],
            "why": (
                "The private MLX smoke checkpoint is clean, so the next training step may advance to a bounded private semantic-quality rung."
                if bounded_rung_ready
                else (
                    "Pre-training architecture readiness is RED because the roadmap now has partial book-derived implementation phases. Resolve or falsify those partial phases before making training or public calibration the primary focus."
                    if not architecture_ready
                    else "Architecture and governance are ready, but a clean private smoke checkpoint is required before longer training; semantic behavior quality and external peer reachability are not yet sufficient for promotion, public calibration, production routing, or distributed training claims."
                )
            ),
        },
        "non_claims": [
            "This gate does not train a model.",
            "This gate does not run public calibration.",
            "This gate does not claim model-only ChatGPT-grade inference.",
            "This gate does not claim public transfer or ASI capability.",
            "This gate does not allow public benchmark artifacts in training.",
            "This gate does not allow fallback/template/router/tool results to count as learned generation.",
        ],
    }


def check_plan_shape(plan: dict[str, Any]) -> dict[str, Any]:
    return check(
        "plan_shape_and_policy_present",
        plan.get("policy") == "project_theseus_training_inference_execution_roadmap_v1"
        and isinstance(plan.get("rules"), dict)
        and isinstance(plan.get("execution_order"), list)
        and isinstance(plan.get("lanes"), list)
        and isinstance(plan.get("success_metrics"), dict)
        and isinstance(plan.get("next_concrete_sequence"), list),
        {
            "policy": plan.get("policy"),
            "has_rules": isinstance(plan.get("rules"), dict),
            "has_execution_order": isinstance(plan.get("execution_order"), list),
            "lane_count": len(list_dicts(plan.get("lanes"))),
        },
    )


def check_required_rules(plan: dict[str, Any]) -> dict[str, Any]:
    rules = dict_value(plan.get("rules"))
    mismatches = {key: {"expected": expected, "actual": rules.get(key)} for key, expected in REQUIRED_RULES.items() if rules.get(key) is not expected}
    return check("required_no_cheat_rules_are_exact", not mismatches, mismatches)


def check_lane_inventory(plan: dict[str, Any]) -> dict[str, Any]:
    lanes = list_dicts(plan.get("lanes"))
    ids = [str(lane.get("id", "")) for lane in lanes]
    missing = [lane_id for lane_id in REQUIRED_LANES if lane_id not in ids]
    duplicates = sorted({lane_id for lane_id in ids if ids.count(lane_id) > 1})
    order_ok = list_value(plan.get("execution_order")) == REQUIRED_LANES
    return check(
        "required_lane_inventory_and_order_present",
        not missing and not duplicates and order_ok,
        {"missing": missing, "duplicates": duplicates, "execution_order": plan.get("execution_order")},
    )


def check_lane_shapes(plan: dict[str, Any]) -> dict[str, Any]:
    faults = {}
    for lane in list_dicts(plan.get("lanes")):
        lane_id = str(lane.get("id", ""))
        missing = sorted(REQUIRED_LANE_FIELDS - set(lane))
        bad_lists = [
            field
            for field in ["entry_criteria", "required_gates", "allowed_inputs", "forbidden_inputs", "evidence_outputs", "exit_criteria", "no_claims"]
            if not isinstance(lane.get(field), list) or not lane.get(field)
        ]
        bad_status = lane.get("status") not in ALLOWED_LANE_STATES
        if missing or bad_lists or bad_status:
            faults[lane_id or "<missing-id>"] = {"missing": missing, "bad_lists": bad_lists, "status": lane.get("status")}
    return check("all_lanes_have_operational_shape", not faults, faults)


def check_training_lanes_forbid_public_benchmarks(plan: dict[str, Any]) -> dict[str, Any]:
    training_ids = {"T1_data_and_context_manifest", "T2_private_training_smoke", "T3_bounded_private_training_rung", "T6_dogfood_feedback_to_training_rows"}
    faults = {}
    for lane in list_dicts(plan.get("lanes")):
        lane_id = str(lane.get("id", ""))
        if lane_id not in training_ids:
            continue
        forbidden = {str(item) for item in list_value(lane.get("forbidden_inputs"))}
        if not any(term in forbidden for term in FORBIDDEN_TRAINING_TERMS):
            faults[lane_id] = sorted(forbidden)
        # The raw-text check is only required for data/dogfood lanes.
        if lane_id in {"T1_data_and_context_manifest", "T6_dogfood_feedback_to_training_rows"} and not any(
            "raw private user text" in item for item in forbidden
        ):
            faults[lane_id] = sorted(forbidden)
    return check("training_lanes_explicitly_forbid_public_benchmark_payloads", not faults, faults)


def check_public_calibration_lane_freshness_policy(plan: dict[str, Any]) -> dict[str, Any]:
    lane = lane_by_id(plan, "T7_fresh_public_calibration_proposal")
    forbidden = " ".join(str(item) for item in list_value(lane.get("forbidden_inputs"))).lower()
    entry = " ".join(str(item) for item in list_value(lane.get("entry_criteria"))).lower()
    return check(
        "public_calibration_lane_requires_fresh_non_consumed_surface",
        lane.get("status") == "blocked_until_private_quality_positive"
        and "fresh" in entry
        and "consumed" in forbidden
        and any("public" in gate for gate in list_value(lane.get("required_gates"))),
        {"status": lane.get("status"), "entry_criteria": lane.get("entry_criteria"), "forbidden_inputs": lane.get("forbidden_inputs")},
    )


def check_hive_lane_blocks_unreachable_peers(plan: dict[str, Any]) -> dict[str, Any]:
    lane = lane_by_id(plan, "T8_hive_fleet_training_scaleout")
    return check(
        "hive_scaleout_lane_is_external_blocked",
        lane.get("status") == "blocked_external_peers_unreachable"
        and any("arbitrary remote shell" in str(item).lower() for item in list_value(lane.get("forbidden_inputs"))),
        {"status": lane.get("status"), "forbidden_inputs": lane.get("forbidden_inputs")},
    )


def check_current_architecture_ready(current: dict[str, dict[str, Any]]) -> dict[str, Any]:
    roadmap = current["roadmap_pre_training"]
    registry = current["project_registry"]
    return check(
        "current_architecture_and_registry_ready",
        roadmap.get("pre_training_architecture_ready") is True
        and int_value(roadmap.get("pre_training_architecture_blocker_count")) == 0
        and int_value(registry.get("registry_hard_governance_violation_count")) == 0
        and int_value(registry.get("unregistered_active_source_count")) == 0,
        {
            "pre_training_architecture_ready": roadmap.get("pre_training_architecture_ready"),
            "pre_training_architecture_blocker_count": roadmap.get("pre_training_architecture_blocker_count"),
            "registry_hard_governance_violation_count": registry.get("registry_hard_governance_violation_count"),
            "unregistered_active_source_count": registry.get("unregistered_active_source_count"),
        },
    )


def check_current_data_admission(current: dict[str, dict[str, Any]]) -> dict[str, Any]:
    admission = current["training_data_admission"]
    return check(
        "current_training_data_admission_hard_gates_clean",
        bool(admission.get("public_benchmark_payload_admitted")) is False
        and bool(admission.get("public_benchmark_training_allowed")) is False
        and int_value(admission.get("external_inference_calls")) == 0
        and bool(admission.get("teacher_rows_admitted_outside_distillation_gate")) is False
        and bool(admission.get("vcm_context_governor_ready")) is True,
        {
            "trigger_state": admission.get("trigger_state"),
            "public_benchmark_payload_admitted": admission.get("public_benchmark_payload_admitted"),
            "public_benchmark_training_allowed": admission.get("public_benchmark_training_allowed"),
            "external_inference_calls": admission.get("external_inference_calls"),
            "teacher_rows_admitted_outside_distillation_gate": admission.get("teacher_rows_admitted_outside_distillation_gate"),
            "vcm_context_governor_ready": admission.get("vcm_context_governor_ready"),
        },
    )


def check_current_neural_seed_readiness(current: dict[str, dict[str, Any]]) -> dict[str, Any]:
    neural = current["neural_seed_survival"]
    return check(
        "current_neural_seed_ready_for_training_focus_not_quality_claim",
        neural.get("phase10_survival_lane_state") == "GREEN"
        and neural.get("readiness_scope") == "architecture_ready_for_next_governed_training_or_adaptation_not_model_quality"
        and int_value(neural.get("failed_check_count")) == 0
        and int_value(neural.get("failed_expected_invalid_count")) == 0
        and float_value(neural.get("current_behavior_pass_rate")) >= 0.0,
        {
            "phase10_survival_lane_state": neural.get("phase10_survival_lane_state"),
            "readiness_scope": neural.get("readiness_scope"),
            "current_behavior_pass_rate": neural.get("current_behavior_pass_rate"),
            "c1_pass_if_any_rate": neural.get("c1_pass_if_any_rate"),
        },
    )


def check_current_resource_route(current: dict[str, dict[str, Any]]) -> dict[str, Any]:
    resource = current["resource_mlx_route"]
    return check(
        "current_mlx_route_mechanically_ready_but_production_disabled",
        resource.get("phase8_resource_mlx_route_state") == "GREEN"
        and resource.get("accelerator_backend") == "mlx_apple"
        and resource.get("production_route_eligible") is False
        and resource.get("production_route_block_reason") == "fail_closed_behavior_quality_zero"
        and resource.get("parity_claim_allowed") is False,
        {
            "phase8_resource_mlx_route_state": resource.get("phase8_resource_mlx_route_state"),
            "accelerator_backend": resource.get("accelerator_backend"),
            "production_route_eligible": resource.get("production_route_eligible"),
            "production_route_block_reason": resource.get("production_route_block_reason"),
            "parity_claim_allowed": resource.get("parity_claim_allowed"),
        },
    )


def check_current_t2_training_smoke_if_present(current: dict[str, dict[str, Any]]) -> dict[str, Any]:
    smoke = current["t2_private_training_smoke"]
    if not smoke:
        return check("current_t2_private_training_smoke_if_present", True, {"present": False, "state": "not_yet_run"})
    checkpoint = str(smoke.get("checkpoint") or "")
    vocab = str(smoke.get("vocab") or "")
    training_plan = dict_value(smoke.get("training_plan"))
    target_token_positions = int_value(training_plan.get("target_token_positions"))
    no_cheat_counts = {
        "public_training_rows": int_value(smoke.get("public_training_rows")),
        "public_training_rows_written": int_value(smoke.get("public_training_rows_written")),
        "external_inference_calls": int_value(smoke.get("external_inference_calls")),
        "fallback_return_count": int_value(smoke.get("fallback_return_count")),
        "fallback_template_router_tool_credit_count": int_value(smoke.get("fallback_template_router_tool_credit_count")),
    }
    faults = {}
    checkpoint_logical = resolve(checkpoint) if checkpoint else None
    checkpoint_resolved = (
        resolve_archived_path(checkpoint_logical) if checkpoint_logical is not None else None
    )
    checkpoint_from_archive = bool(
        checkpoint_logical is not None and is_archive_pointer(checkpoint_logical)
    )
    vocab_path = resolve(vocab) if vocab else None
    if smoke.get("trigger_state") != "GREEN":
        faults["trigger_state"] = smoke.get("trigger_state")
    if smoke.get("backend") != "mlx_high_level_transformer":
        faults["backend"] = smoke.get("backend")
    if "gpu" not in str(smoke.get("device", "")).lower():
        faults["device"] = smoke.get("device")
    if not checkpoint or checkpoint_resolved is None or not checkpoint_resolved.exists():
        faults["checkpoint"] = checkpoint or None
    elif file_sha256(checkpoint_resolved) != str(smoke.get("checkpoint_sha256") or ""):
        faults["checkpoint_sha256_mismatch"] = {
            "expected": smoke.get("checkpoint_sha256"),
            "actual": file_sha256(checkpoint_resolved),
        }
    if not vocab or vocab_path is None or not vocab_path.exists():
        faults["vocab"] = vocab or None
    elif file_sha256(vocab_path) != str(smoke.get("vocab_sha256") or ""):
        faults["vocab_sha256_mismatch"] = {
            "expected": smoke.get("vocab_sha256"),
            "actual": file_sha256(vocab_path),
        }
    if not valid_sha256(smoke.get("checkpoint_sha256")):
        faults["checkpoint_sha256"] = smoke.get("checkpoint_sha256")
    if not valid_sha256(smoke.get("vocab_sha256")):
        faults["vocab_sha256"] = smoke.get("vocab_sha256")
    dirty_counts = {key: value for key, value in no_cheat_counts.items() if value != 0}
    if dirty_counts:
        faults["no_cheat_counts"] = dirty_counts
    if smoke.get("open_or_pretrained_model_weights_used") is not False:
        faults["open_or_pretrained_model_weights_used"] = smoke.get("open_or_pretrained_model_weights_used")
    if target_token_positions <= 0 or int_value(smoke.get("optimizer_token_positions_consumed")) < target_token_positions:
        faults["optimizer_token_positions_consumed"] = {
            "consumed": smoke.get("optimizer_token_positions_consumed"),
            "target": target_token_positions,
        }
    if float_value(smoke.get("parameter_update_fraction")) < 0.95:
        faults["parameter_update_fraction"] = smoke.get("parameter_update_fraction")
    if float_value(smoke.get("parameter_tensor_update_fraction")) < 0.95:
        faults["parameter_tensor_update_fraction"] = smoke.get("parameter_tensor_update_fraction")
    if smoke.get("heldout_lm_improved") is not True:
        faults["heldout_lm_improved"] = smoke.get("heldout_lm_improved")
    if float_value(smoke.get("training_tokens_per_second")) <= 0:
        faults["training_tokens_per_second"] = smoke.get("training_tokens_per_second")
    return check(
        "current_t2_private_training_smoke_clean_when_present",
        not faults,
        {
            "present": True,
            "faults": faults,
            "checkpoint": checkpoint,
            "checkpoint_resolved": rel(checkpoint_resolved) if checkpoint_resolved else None,
            "checkpoint_from_archive": checkpoint_from_archive,
            "checkpoint_sha256": smoke.get("checkpoint_sha256"),
            "vocab": vocab,
            "vocab_sha256": smoke.get("vocab_sha256"),
            "optimizer_token_positions_consumed": smoke.get("optimizer_token_positions_consumed"),
            "target_token_positions": target_token_positions,
            "training_tokens_per_second": smoke.get("training_tokens_per_second"),
            "heldout_lm_loss_before": smoke.get("heldout_lm_loss_before"),
            "heldout_lm_loss_after": smoke.get("heldout_lm_loss_after"),
            "semantic_plan_accuracy_before": smoke.get("semantic_plan_accuracy_before"),
            "semantic_plan_accuracy_after": smoke.get("semantic_plan_accuracy_after"),
            "no_cheat_counts": no_cheat_counts,
        },
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_current_assistant_canary(current: dict[str, dict[str, Any]]) -> dict[str, Any]:
    assistant = current["assistant_product_lane"]
    return check(
        "current_assisted_local_inference_canary_ready",
        assistant.get("b1_assisted_verified_assistant_product_lane_state") == "GREEN"
        and assistant.get("b1_synthetic_support_ready") is True
        and int_value(assistant.get("passed_case_count")) >= 4
        and int_value(assistant.get("external_inference_calls")) == 0
        and int_value(assistant.get("public_training_rows_written")) == 0
        and int_value(assistant.get("fallback_return_count")) == 0,
        {
            "state": assistant.get("b1_assisted_verified_assistant_product_lane_state"),
            "support_state": assistant.get("b1_assisted_verified_assistant_product_lane_support_state"),
            "b1_synthetic_support_ready": assistant.get("b1_synthetic_support_ready"),
            "b1_empirical_support_ready": assistant.get("b1_empirical_support_ready"),
            "b1_empirical_block_reason": assistant.get("b1_empirical_block_reason"),
            "passed_case_count": assistant.get("passed_case_count"),
            "recent_event_count": assistant.get("recent_event_count"),
            "completed_or_accepted_count": assistant.get("completed_or_accepted_count"),
            "external_inference_calls": assistant.get("external_inference_calls"),
        },
    )


def check_current_public_calibration_status(current: dict[str, dict[str, Any]]) -> dict[str, Any]:
    public = current["public_calibration_proposal"]
    return check(
        "current_public_calibration_status_is_known_and_not_training",
        public.get("public_benchmark_training_payload_admitted") is False
        and public.get("execution_allowed") is False
        and public.get("proposal_decision") == "REFUSED_EXACT_CONSUMED_SURFACE_RERUN",
        {
            "surface_slug": public.get("surface_slug"),
            "proposal_decision": public.get("proposal_decision"),
            "execution_allowed": public.get("execution_allowed"),
            "exact_consumed_surface": public.get("exact_consumed_surface"),
            "public_benchmark_training_payload_admitted": public.get("public_benchmark_training_payload_admitted"),
        },
    )


def check_public_benchmark_training_refused(plan: dict[str, Any], current: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rules = dict_value(plan.get("rules"))
    admission = current["training_data_admission"]
    return check(
        "public_benchmark_training_refused",
        rules.get("public_benchmark_payloads_allowed_in_training") is False
        and admission.get("public_benchmark_payload_admitted") is False
        and admission.get("public_benchmark_training_allowed") is False,
        {
            "rule": rules.get("public_benchmark_payloads_allowed_in_training"),
            "public_benchmark_payload_admitted": admission.get("public_benchmark_payload_admitted"),
            "public_benchmark_training_allowed": admission.get("public_benchmark_training_allowed"),
        },
    )


def check_external_runtime_refused(plan: dict[str, Any]) -> dict[str, Any]:
    rules = dict_value(plan.get("rules"))
    lane = lane_by_id(plan, "T5_local_assisted_inference_canary")
    forbidden = " ".join(str(item).lower() for item in list_value(lane.get("forbidden_inputs")))
    return check(
        "external_runtime_inference_refused",
        rules.get("external_runtime_inference_allowed") is False and "external inference tokens" in forbidden,
        {"rule": rules.get("external_runtime_inference_allowed"), "t5_forbidden_inputs": lane.get("forbidden_inputs")},
    )


def check_fallback_credit_refused(plan: dict[str, Any]) -> dict[str, Any]:
    rules = dict_value(plan.get("rules"))
    return check(
        "fallback_template_router_tool_credit_refused",
        rules.get("fallback_template_router_tool_credit_as_learned_generation") is False
        and rules.get("deterministic_tool_results_count_as_tool_assisted_not_model_only") is True,
        {
            "fallback_credit_rule": rules.get("fallback_template_router_tool_credit_as_learned_generation"),
            "tool_assisted_rule": rules.get("deterministic_tool_results_count_as_tool_assisted_not_model_only"),
        },
    )


def check_consumed_public_surface_refused(current: dict[str, dict[str, Any]]) -> dict[str, Any]:
    public = current["public_calibration_proposal"]
    return check(
        "consumed_public_surface_rerun_refused",
        public.get("exact_consumed_surface") is True
        and public.get("proposal_decision") == "REFUSED_EXACT_CONSUMED_SURFACE_RERUN"
        and public.get("execution_allowed") is False,
        {
            "proposal_decision": public.get("proposal_decision"),
            "exact_consumed_surface": public.get("exact_consumed_surface"),
            "execution_allowed": public.get("execution_allowed"),
        },
    )


def check_mlx_route_fail_closed(current: dict[str, dict[str, Any]]) -> dict[str, Any]:
    resource = current["resource_mlx_route"]
    return check(
        "mlx_route_fail_closed",
        resource.get("production_route_eligible") is False
        and resource.get("production_route_block_reason") == "fail_closed_behavior_quality_zero",
        {
            "production_route_eligible": resource.get("production_route_eligible"),
            "production_route_block_reason": resource.get("production_route_block_reason"),
        },
    )


def check_model_only_general_chat_not_claimed(plan: dict[str, Any]) -> dict[str, Any]:
    lane = lane_by_id(plan, "T5_local_assisted_inference_canary")
    no_claims = " ".join(str(item).lower() for item in list_value(lane.get("no_claims")))
    return check(
        "model_only_general_chat_not_claimed",
        "does not claim chatgpt-grade model-only inference" in no_claims,
        {"t5_no_claims": lane.get("no_claims")},
    )


def check_teacher_boundary(plan: dict[str, Any], current: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rules = dict_value(plan.get("rules"))
    admission = current["training_data_admission"]
    return check(
        "teacher_boundary_refuses_unadmitted_rows_and_runtime_serving",
        rules.get("teacher_training_rows_allowed_only_through_distillation_gate") is True
        and rules.get("teacher_tokens_allowed_at_runtime_serving") is False
        and admission.get("teacher_rows_admitted_outside_distillation_gate") is False,
        {
            "teacher_training_rows_allowed_only_through_distillation_gate": rules.get("teacher_training_rows_allowed_only_through_distillation_gate"),
            "teacher_tokens_allowed_at_runtime_serving": rules.get("teacher_tokens_allowed_at_runtime_serving"),
            "teacher_rows_admitted_outside_distillation_gate": admission.get("teacher_rows_admitted_outside_distillation_gate"),
        },
    )


def check_hive_scaleout_blocked(plan: dict[str, Any]) -> dict[str, Any]:
    lane = lane_by_id(plan, "T8_hive_fleet_training_scaleout")
    return check(
        "hive_scaleout_blocked",
        lane.get("status") == "blocked_external_peers_unreachable",
        {"status": lane.get("status"), "rollback_or_stop": lane.get("rollback_or_stop")},
    )


def ready_for_private_training(current: dict[str, dict[str, Any]], failed_checks: list[dict[str, Any]], failed_expected_invalid: list[dict[str, Any]]) -> bool:
    return (
        not failed_checks
        and not failed_expected_invalid
        and current["roadmap_pre_training"].get("pre_training_architecture_ready") is True
        and current["neural_seed_survival"].get("phase10_survival_lane_state") == "GREEN"
        and current["training_data_admission"].get("public_benchmark_payload_admitted") is False
    )


def ready_for_private_training_smoke(plan: dict[str, Any], current: dict[str, dict[str, Any]], failed_checks: list[dict[str, Any]], failed_expected_invalid: list[dict[str, Any]]) -> bool:
    return ready_for_private_training(current, failed_checks, failed_expected_invalid) and lane_by_id(plan, "T2_private_training_smoke").get("status") == "ready"


def current_t2_training_smoke_passed(current: dict[str, dict[str, Any]]) -> bool:
    return check_current_t2_training_smoke_if_present(current)["passed"] and bool(current["t2_private_training_smoke"])


def ready_for_bounded_private_training_rung(
    plan: dict[str, Any],
    current: dict[str, dict[str, Any]],
    failed_checks: list[dict[str, Any]],
    failed_expected_invalid: list[dict[str, Any]],
    *,
    t2_smoke_passed: bool,
) -> bool:
    return (
        ready_for_private_training(current, failed_checks, failed_expected_invalid)
        and t2_smoke_passed
        and lane_by_id(plan, "T3_bounded_private_training_rung").get("status") in {"ready", "planned"}
    )


def ready_for_local_assisted_inference(current: dict[str, dict[str, Any]], failed_checks: list[dict[str, Any]], failed_expected_invalid: list[dict[str, Any]]) -> bool:
    return (
        not failed_checks
        and not failed_expected_invalid
        and current["assistant_product_lane"].get("b1_assisted_verified_assistant_product_lane_state") == "GREEN"
        and int_value(current["assistant_product_lane"].get("external_inference_calls")) == 0
    )


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    compact: dict[str, Any] = {}
    summary = report.get("summary")
    if isinstance(summary, dict):
        compact.update(summary)
    for key, value in report.items():
        if key == "summary":
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[key] = value
    return compact


def lane_by_id(plan: dict[str, Any], lane_id: str) -> dict[str, Any]:
    for lane in list_dicts(plan.get("lanes")):
        if lane.get("id") == lane_id:
            return lane
    return {}


def expected_invalid(name: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(row.get("passed")),
        "expected_invalid_rejected": bool(row.get("passed")),
        "evidence": row.get("evidence", {}),
    }


def check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def gate_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report["trigger_state"],
        "summary": report["summary"],
        "failed_checks": [row for row in report["checks"] if not row["passed"]],
        "failed_expected_invalid_controls": [row for row in report["expected_invalid_controls"] if not row["passed"]],
        "readiness_decision": report["readiness_decision"],
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Training And Inference Execution Plan Gate",
        "",
        f"- State: `{report['trigger_state']}`",
        f"- Ready for governed private training focus: `{summary['ready_for_governed_private_training_focus']}`",
        f"- Ready for private training smoke: `{summary['ready_for_private_training_smoke']}`",
        f"- T2 private training smoke passed: `{summary['t2_private_training_smoke_passed']}`",
        f"- Ready for bounded private training rung: `{summary['ready_for_bounded_private_training_rung']}`",
        f"- Ready for local assisted inference canary: `{summary['ready_for_local_assisted_inference_canary']}`",
        f"- Ready for model-only general chat runtime: `{summary['ready_for_model_only_general_chat_runtime']}`",
        f"- Ready for public calibration: `{summary['ready_for_public_calibration']}`",
        f"- Ready for production MLX route: `{summary['ready_for_production_mlx_route']}`",
        f"- Ready for Hive fleet training: `{summary['ready_for_hive_fleet_training']}`",
        f"- Failed checks: `{summary['failed_check_count']}`",
        f"- Failed expected-invalid controls: `{summary['failed_expected_invalid_count']}`",
        "",
        "## Next Allowed Steps",
        "",
        f"- Training: `{report['readiness_decision']['next_allowed_step']}`",
        f"- Inference: `{report['readiness_decision']['next_allowed_inference_step']}`",
        "",
        "## Non-Claims",
        "",
    ]
    lines.extend(f"- {item}" for item in report["non_claims"])
    lines.append("")
    return "\n".join(lines)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


def valid_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text.lower())


def resolve(path: str | Path) -> Path:
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
