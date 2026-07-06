"""Concept build orchestration for the high-transfer scheduler."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from high_transfer_scheduler_common import *  # noqa: F401,F403
from high_transfer_scheduler_code_state import *  # noqa: F401,F403
from high_transfer_scheduler_rotation import *  # noqa: F401,F403
from progress_integrity_policy import apply_non_promotable_diagnostic_policy  # noqa: E402


def build_concepts(
    transfer: dict[str, Any],
    broad: dict[str, Any],
    guidance: dict[str, Any],
    conversation: dict[str, Any],
    conversation_hard: dict[str, Any],
    conversation_hard_v2: dict[str, Any],
    conversation_hard_v3: dict[str, Any],
    conversation_hard_v4: dict[str, Any],
    conversation_pantry: dict[str, Any],
    repo: dict[str, Any],
    board_game: dict[str, Any],
    pufferlib4_rl: dict[str, Any],
    long_horizon: dict[str, Any],
    cross_domain_capsules: dict[str, Any],
    type_contract: dict[str, Any],
    autonomy_policy: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    aggregate_pass_rate = float(
        get_path(
            broad,
            ["summary", "aggregate_pass_rate"],
            get_path(broad, ["summary", "real_public_pass_rate"], 0.0),
        )
        or 0.0
    )
    floor_gap = float(get_path(broad, ["summary", "aggregate_floor_gap"], None) or max(0.0, 0.70 - aggregate_pass_rate))
    overfit_risk = int(get_path(transfer, ["summary", "overfit_risk_count"], 0) or 0)
    transfer_pressure_by_concept = concept_transfer_pressure_index(transfer)
    no_lift_cooldown_concepts = no_lift_cooldown_concept_set()
    max_shared_pressure = max(
        (
            int(row.get("residual_count") or 0)
            for row in transfer_pressure_by_concept.values()
            if int(row.get("card_count") or 0) >= 2
        ),
        default=0,
    )
    conversation_focus = conversation_focus_enabled(autonomy_policy)
    rotation_epoch = conversation_rotation_epoch()
    conversation_state = conversation_lifecycle(conversation)
    conversation_hard_state = conversation_lifecycle(conversation_hard)
    conversation_hard_v2_state = conversation_lifecycle(conversation_hard_v2)
    conversation_hard_v3_state = conversation_lifecycle(conversation_hard_v3)
    conversation_hard_v4_state = conversation_lifecycle(conversation_hard_v4)
    generalist_rotation = generalist_rotation_state(
        broad=broad,
        conversation_hard=conversation_hard,
        conversation_hard_v2=conversation_hard_v2,
        conversation_hard_v3=conversation_hard_v3,
        conversation_hard_v4=conversation_hard_v4,
        board_game=board_game,
        pufferlib4_rl=pufferlib4_rl,
        long_horizon=long_horizon,
        repo=repo,
        cross_domain_capsules=cross_domain_capsules,
    )
    for spec in CONCEPTS:
        concept = dict(spec)
        if conversation_focus and concept["concept"] in {"type_and_return_shape", "edge_conditions"} and not conversation_state["graduated"]:
            concept["priority"] = "high"
        if conversation_focus and concept["concept"] in {"open_conversation_pantry", "multi_turn_conversation"} and not conversation_state["graduated"]:
            concept["priority"] = "critical"
            concept["rotation_epoch"] = rotation_epoch
        concept["transfer_checks"] = [
            {"donor": donor, "receiver": receiver, "status": "pending"}
            for donor in concept["donors"]
            for receiver in concept["receivers"]
            if donor != receiver
        ]
        concept["public_data_rule"] = "public_benchmarks_calibration_only"
        concept["training_surface"] = "private_hidden_or_local_trace_only_with_sts_default_conditioning"
        concept["sts_training_policy"] = {
            "default": "native_sts_conditioning_on",
            "disable_flag": "--disable-native-sts-conditioning",
            "sts_off_role": "control_ablation_only",
            "public_tests_or_solutions_used": False,
        }
        if concept["concept"] in {
            "type_contract_diagnostic",
            "type_and_return_shape",
            "typed_interface_skeleton",
            "edge_contract_4card",
            "edge_conditions",
            "admissibility_and_interface",
            "algorithmic_planning",
            "execution_shaped_programs",
            BALANCED_EDGE_CONTRACT_CONCEPT,
            EDGE_CASE_FULL_BODY_CONCEPT,
            EDGE_CONTRACT_V2_CONCEPT,
            RESIDUAL_EDGE_CASE_CONTRACT_CONCEPT,
        }:
            feedback_rows = int(get_path(type_contract, ["summary", "feedback_rows_written"], 0) or 0)
            concept_pressure = concept_private_pressure_state(concept["concept"])
            if concept["concept"] == "type_contract_diagnostic" and feedback_rows >= 900:
                concept["status"] = "regression_only"
            elif (
                concept["concept"] == "execution_shaped_programs"
                and not concept_pressure["waiting_for_recalibration"]
                and edge_exec_repair_lifecycle(
                    read_json(
                        REPORTS
                        / "real_code_benchmark_graduation_source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32.json",
                        {},
                    ),
                    broad,
                )["graduated"]
            ):
                concept["status"] = "regression_only"
            elif concept["concept"] == BALANCED_EDGE_CONTRACT_CONCEPT:
                balanced_state = balanced_edge_contract_experiment_state(guidance)
                concept["status"] = (
                    "ready"
                    if balanced_state["needs_pressure"] and floor_gap > 0
                    else ("waiting_private_closure" if balanced_state["needs_private_closure"] else "regression_only")
                )
            elif concept["concept"] == EDGE_CASE_FULL_BODY_CONCEPT:
                edge_full_state = edge_case_full_body_experiment_state(guidance)
                concept["status"] = (
                    "ready"
                    if edge_full_state["needs_pressure"] and floor_gap > 0
                    else ("waiting_private_closure" if edge_full_state["needs_private_closure"] else "regression_only")
                )
            elif concept["concept"] == EDGE_CONTRACT_V2_CONCEPT:
                edge_v2_state = edge_contract_v2_experiment_state(guidance)
                concept["status"] = (
                    "ready"
                    if edge_v2_state["needs_pressure"] and floor_gap > 0
                    else ("waiting_private_closure" if edge_v2_state["needs_private_closure"] else "regression_only")
                )
            elif concept_pressure["waiting_for_recalibration"]:
                concept["status"] = "waiting_recalibration"
            else:
                concept["status"] = "ready" if floor_gap > 0 or overfit_risk > 0 else "diagnostic"
            if concept["status"] == "ready":
                base_epoch = code_transfer_rotation_epoch(
                    concept["concept"],
                    transfer=transfer,
                    broad=broad,
                    guidance=guidance,
                )
                epoch_payload = {
                    "base_epoch": base_epoch,
                    "concept": concept["concept"],
                    "last_private_pressure_mtime": concept_pressure.get("report_mtime"),
                    "last_four_card_calibration_mtime": concept_pressure.get("calibration_mtime"),
                    "pressure_state_reason": concept_pressure.get("reason"),
                }
                concept["rotation_epoch"] = hashlib.sha256(
                    json.dumps(epoch_payload, sort_keys=True).encode("utf-8")
                ).hexdigest()[:12]
            apply_dynamic_code_priority(
                concept,
                transfer_pressure_by_concept=transfer_pressure_by_concept,
                max_shared_pressure=max_shared_pressure,
                no_lift_cooldown_concepts=no_lift_cooldown_concepts,
            )
            concept["evidence"] = {
                "aggregate_floor_gap": floor_gap,
                "overfit_risk_count": overfit_risk,
                "rotation_epoch": concept.get("rotation_epoch"),
                "dynamic_priority_reason": concept.get("dynamic_priority_reason"),
                "dynamic_priority_evidence": concept.get("dynamic_priority_evidence"),
                "teacher_recommendation": get_path(guidance, ["teacher", "response_json", "recommended_intervention"], ""),
                "decoder_feedback_path": "D:/ProjectTheseus/training_data/high_transfer/private_train/type_contract_decoder_feedback.jsonl"
                if concept["concept"] == "type_contract_diagnostic"
                else None,
                "feedback_rows_written": feedback_rows if concept["concept"] == "type_contract_diagnostic" else None,
                "concept_private_pressure_state": concept_pressure,
                "balanced_edge_contract_experiment_state": balanced_edge_contract_experiment_state(guidance)
                if concept["concept"] == BALANCED_EDGE_CONTRACT_CONCEPT
                else None,
                "edge_case_full_body_experiment_state": edge_case_full_body_experiment_state(guidance)
                if concept["concept"] == EDGE_CASE_FULL_BODY_CONCEPT
                else None,
                "edge_contract_v2_experiment_state": edge_contract_v2_experiment_state(guidance)
                if concept["concept"] == EDGE_CONTRACT_V2_CONCEPT
                else None,
                "residual_edge_case_contract_private_rows": file_mtime(RESIDUAL_EDGE_CASE_CONTRACT_TRAIN_JSONL)
                if concept["concept"] == RESIDUAL_EDGE_CASE_CONTRACT_CONCEPT
                else None,
            }
        elif concept["concept"] == "typed_interface_private_closure":
            closure_state = typed_interface_private_closure_state()
            concept["status"] = (
                "ready"
                if closure_state["needs_private_closure"] and floor_gap > 0
                else ("regression_only" if closure_state["closure_current"] else "diagnostic")
            )
            if concept["status"] == "ready":
                epoch_payload = {
                    "concept": concept["concept"],
                    "typed_private_pressure_mtime": closure_state["typed_private_pressure_mtime"],
                    "closure_report_mtime": closure_state["closure_report_mtime"],
                    "closure_trigger_state": closure_state["closure_trigger_state"],
                }
                concept["rotation_epoch"] = hashlib.sha256(
                    json.dumps(epoch_payload, sort_keys=True).encode("utf-8")
                ).hexdigest()[:12]
            concept["evidence"] = {
                "aggregate_floor_gap": floor_gap,
                "typed_interface_private_closure_state": closure_state,
                "public_data_rule": "public_calibration_skipped_private_eval_only",
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == "decoder_plan_ir_code_lm_adapter":
            adapter_state = decoder_plan_ir_code_lm_adapter_state()
            concept["status"] = (
                "ready"
                if adapter_state["needs_adapter"] and floor_gap > 0
                else ("regression_only" if adapter_state["adapter_current"] else "diagnostic")
            )
            if concept["status"] == "ready":
                epoch_payload = {
                    "concept": concept["concept"],
                    "plan_report_mtime": adapter_state["plan_report_mtime"],
                    "adapter_report_mtime": adapter_state["adapter_report_mtime"],
                    "rows_mtime": adapter_state["rows_mtime"],
                    "reason": adapter_state["reason"],
                }
                concept["rotation_epoch"] = hashlib.sha256(
                    json.dumps(epoch_payload, sort_keys=True).encode("utf-8")
                ).hexdigest()[:12]
            concept["evidence"] = {
                "aggregate_floor_gap": floor_gap,
                "decoder_plan_ir_code_lm_adapter_state": adapter_state,
                "public_data_rule": "public_calibration_skipped_private_generated_rows_only",
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == "private_pressure_private_closure":
            closure_state = private_pressure_private_closure_state()
            balanced_state = balanced_edge_contract_experiment_state(guidance)
            edge_full_state = edge_case_full_body_experiment_state(guidance)
            edge_v2_state = edge_contract_v2_experiment_state(guidance)
            broad_decoder_refresh = (
                closure_state.get("reason") == "decoder_or_generator_source_newer_than_private_closure"
                or bool(closure_state.get("decoder_source_changed_after_closure"))
            )
            # The broad private-pressure closure is the unifying gate before
            # Decoder V2 ablation/public calibration. It consumes the
            # specialized private rows directly, so stale specialized closure
            # reports must not demote this task to diagnostic-only.
            concept["status"] = (
                "ready"
                if closure_state["needs_private_closure"] and floor_gap > 0
                else ("regression_only" if closure_state["closure_current"] else "diagnostic")
            )
            if concept["status"] == "ready":
                epoch_payload = {
                    "concept": concept["concept"],
                    "latest_private_pressure_mtime": closure_state["latest_private_pressure_mtime"],
                    "closure_report_mtime": closure_state["closure_report_mtime"],
                    "closure_trigger_state": closure_state["closure_trigger_state"],
                    "fresh_private_pressure_reports": closure_state["fresh_private_pressure_reports"],
                    "missing_required_private_pressure_reports": closure_state["missing_required_private_pressure_reports"],
                }
                concept["rotation_epoch"] = hashlib.sha256(
                    json.dumps(epoch_payload, sort_keys=True).encode("utf-8")
                ).hexdigest()[:12]
            concept["evidence"] = {
                "aggregate_floor_gap": floor_gap,
                "private_pressure_private_closure_state": closure_state,
                "balanced_edge_contract_state": balanced_state,
                "edge_case_full_body_state": edge_full_state,
                "edge_contract_v2_state": edge_v2_state,
                "broad_decoder_refresh_allowed": broad_decoder_refresh,
                "public_data_rule": "public_calibration_skipped_private_eval_only",
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == "edge_contract_private_closure":
            closure_state = edge_contract_private_closure_state()
            concept["status"] = (
                "ready"
                if closure_state["needs_private_closure"] and floor_gap > 0
                else ("regression_only" if closure_state["closure_current"] else "diagnostic")
            )
            if concept["status"] == "ready":
                epoch_payload = {
                    "concept": concept["concept"],
                    "edge_contract_pressure_mtime": closure_state["edge_contract_pressure_mtime"],
                    "closure_report_mtime": closure_state["closure_report_mtime"],
                    "closure_trigger_state": closure_state["closure_trigger_state"],
                    "decoder_relevant_source_fingerprint": closure_state["decoder_relevant_source_fingerprint"],
                }
                concept["rotation_epoch"] = hashlib.sha256(
                    json.dumps(epoch_payload, sort_keys=True).encode("utf-8")
                ).hexdigest()[:12]
            concept["evidence"] = {
                "aggregate_floor_gap": floor_gap,
                "edge_contract_private_closure_state": closure_state,
                "public_data_rule": "public_calibration_skipped_private_eval_only",
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == BALANCED_EDGE_CONTRACT_CLOSURE_CONCEPT:
            balanced_state = balanced_edge_contract_experiment_state(guidance)
            concept["status"] = (
                "ready"
                if balanced_state["needs_private_closure"] and floor_gap > 0
                else ("regression_only" if balanced_state["closure_current"] else "diagnostic")
            )
            if concept["status"] == "ready":
                epoch_payload = {
                    "concept": concept["concept"],
                    "pressure_mtime": balanced_state["pressure_report_mtime"],
                    "closure_report_mtime": balanced_state["closure_report_mtime"],
                    "teacher_guidance_created_utc": balanced_state["teacher_guidance_created_utc"],
                    "decoder_relevant_source_fingerprint": balanced_state["decoder_relevant_source_fingerprint"],
                }
                concept["rotation_epoch"] = hashlib.sha256(
                    json.dumps(epoch_payload, sort_keys=True).encode("utf-8")
                ).hexdigest()[:12]
            concept["evidence"] = {
                "aggregate_floor_gap": floor_gap,
                "balanced_edge_contract_experiment_state": balanced_state,
                "public_data_rule": "public_calibration_skipped_private_eval_only",
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == EDGE_CASE_FULL_BODY_CLOSURE_CONCEPT:
            edge_full_state = edge_case_full_body_experiment_state(guidance)
            concept["status"] = (
                "ready"
                if edge_full_state["needs_private_closure"] and floor_gap > 0
                else ("regression_only" if edge_full_state["closure_current"] else "diagnostic")
            )
            if concept["status"] == "ready":
                epoch_payload = {
                    "concept": concept["concept"],
                    "pressure_mtime": edge_full_state["pressure_report_mtime"],
                    "closure_report_mtime": edge_full_state["closure_report_mtime"],
                    "teacher_guidance_created_utc": edge_full_state["teacher_guidance_created_utc"],
                    "decoder_relevant_source_fingerprint": edge_full_state["decoder_relevant_source_fingerprint"],
                    "private_gate": edge_full_state["private_gate"],
                }
                concept["rotation_epoch"] = hashlib.sha256(
                    json.dumps(epoch_payload, sort_keys=True).encode("utf-8")
                ).hexdigest()[:12]
            concept["evidence"] = {
                "aggregate_floor_gap": floor_gap,
                "edge_case_full_body_experiment_state": edge_full_state,
                "public_data_rule": "public_calibration_skipped_private_eval_only",
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == EDGE_CONTRACT_V2_CLOSURE_CONCEPT:
            edge_v2_state = edge_contract_v2_experiment_state(guidance)
            concept["status"] = (
                "ready"
                if edge_v2_state["needs_private_closure"] and floor_gap > 0
                else ("regression_only" if edge_v2_state["closure_current"] else "diagnostic")
            )
            if concept["status"] == "ready":
                epoch_payload = {
                    "concept": concept["concept"],
                    "pressure_mtime": edge_v2_state["pressure_report_mtime"],
                    "closure_report_mtime": edge_v2_state["closure_report_mtime"],
                    "verifier_report_mtime": edge_v2_state["verifier_report_mtime"],
                    "teacher_guidance_created_utc": edge_v2_state["teacher_guidance_created_utc"],
                    "decoder_relevant_source_fingerprint": edge_v2_state["decoder_relevant_source_fingerprint"],
                    "private_gate": edge_v2_state["private_gate"],
                }
                concept["rotation_epoch"] = hashlib.sha256(
                    json.dumps(epoch_payload, sort_keys=True).encode("utf-8")
                ).hexdigest()[:12]
            concept["evidence"] = {
                "aggregate_floor_gap": floor_gap,
                "edge_contract_v2_experiment_state": edge_v2_state,
                "public_data_rule": "public_calibration_skipped_private_eval_only",
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == "decoder_v2_private_ablation_gate":
            ablation_state = decoder_v2_private_ablation_gate_state()
            concept["status"] = (
                "ready"
                if ablation_state["needs_ablation"]
                else ("regression_only" if ablation_state["ready_for_public_calibration"] else "diagnostic")
            )
            if concept["status"] == "ready":
                epoch_payload = {
                    "concept": concept["concept"],
                    "latest_private_closure_mtime": ablation_state["latest_private_closure_mtime"],
                    "ablation_report_mtime": ablation_state["ablation_report_mtime"],
                    "decoder_relevant_source_fingerprint": ablation_state["decoder_relevant_source_fingerprint"],
                    "reported_decoder_relevant_source_fingerprint": ablation_state["reported_decoder_relevant_source_fingerprint"],
                }
                concept["rotation_epoch"] = hashlib.sha256(
                    json.dumps(epoch_payload, sort_keys=True).encode("utf-8")
                ).hexdigest()[:12]
            concept["evidence"] = {
                "aggregate_floor_gap": floor_gap,
                "decoder_v2_private_ablation_gate_state": ablation_state,
                "public_data_rule": "private_ablation_only_public_benchmarks_calibration_only",
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == "private_type_shape_receiver_veto_ablation":
            receiver_state = private_type_shape_receiver_ablation_state()
            concept["status"] = (
                "ready"
                if receiver_state["needs_ablation"] and floor_gap > 0
                else ("regression_only" if receiver_state["ready_for_public_calibration"] else "diagnostic")
            )
            if concept["status"] == "ready":
                epoch_payload = {
                    "concept": concept["concept"],
                    "teacher_spec_mtime": receiver_state["teacher_spec_mtime"],
                    "candidate_manifest_mtime": receiver_state["candidate_manifest_mtime"],
                    "latest_private_closure_mtime": receiver_state["latest_private_closure_mtime"],
                    "ablation_report_mtime": receiver_state["ablation_report_mtime"],
                    "reason": receiver_state["reason"],
                }
                concept["rotation_epoch"] = hashlib.sha256(
                    json.dumps(epoch_payload, sort_keys=True).encode("utf-8")
                ).hexdigest()[:12]
            concept["evidence"] = {
                "aggregate_floor_gap": floor_gap,
                "private_type_shape_receiver_ablation_state": receiver_state,
                "public_data_rule": "private_ablation_only_public_benchmarks_calibration_only",
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == "execution_shape_private_ablation":
            execution_pressure = concept_private_pressure_state("execution_shaped_programs")
            ablation = read_json(REPORTS / "execution_shape_private_ablation.json", {})
            ablation_path = REPORTS / "execution_shape_private_ablation.json"
            ablation_mtime = file_mtime(ablation_path)
            skeleton_ok = bool(get_path(ablation, ["summary", "skeleton_competitive_with_semantic"], False))
            private_gate = execution_shape_private_gate(ablation)
            decoder_fingerprint = decoder_relevant_source_fingerprint()
            report_fingerprint = str(get_path(ablation, ["summary", "decoder_relevant_source_fingerprint"], "") or "")
            calibration_mtime = four_card_calibration_mtime()
            decoder_current = (
                report_fingerprint == decoder_fingerprint
                if report_fingerprint
                else calibration_mtime >= ablation_mtime
            )
            ablation_current = (
                private_gate["ready_for_public_calibration"]
                and ablation_mtime >= float(execution_pressure.get("report_mtime") or 0.0)
                and decoder_current
            )
            report_current_but_not_ready = (
                ablation_mtime
                and ablation_mtime >= float(execution_pressure.get("report_mtime") or 0.0)
                and decoder_current
                and not private_gate["ready_for_public_calibration"]
            )
            concept["status"] = (
                "regression_only"
                if ablation_current
                else ("blocked_needs_decoder_patch" if report_current_but_not_ready else "ready")
            )
            if concept["status"] == "ready":
                epoch_payload = {
                    "concept": concept["concept"],
                    "execution_report_mtime": execution_pressure.get("report_mtime"),
                    "decoder_relevant_source_fingerprint": decoder_fingerprint,
                    "reported_decoder_relevant_source_fingerprint": report_fingerprint,
                    "ablation_mtime": ablation_mtime,
                    "ablation_trigger": ablation.get("trigger_state"),
                    "skeleton_ok": skeleton_ok,
                }
                concept["rotation_epoch"] = hashlib.sha256(
                    json.dumps(epoch_payload, sort_keys=True).encode("utf-8")
                ).hexdigest()[:12]
            concept["evidence"] = {
                "execution_shaped_private_pressure_state": execution_pressure,
                "ablation_report": "reports/execution_shape_private_ablation.json",
                "ablation_trigger_state": ablation.get("trigger_state"),
                "ablation_mtime": ablation_mtime or None,
                "decoder_relevant_source_fingerprint": decoder_fingerprint,
                "reported_decoder_relevant_source_fingerprint": report_fingerprint or None,
                "decoder_current": decoder_current,
                "skeleton_competitive_with_semantic": skeleton_ok,
                "private_ablation_gate": private_gate,
                "public_data_rule": "private_ablation_only_public_benchmarks_calibration_only",
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == "execution_shaped_four_card_calibration":
            execution_pressure = concept_private_pressure_state("execution_shaped_programs")
            feedback_rows = int(get_path(type_contract, ["summary", "feedback_rows_written"], 0) or 0)
            execution_contract_rows = int(
                get_path(type_contract, ["summary", "type_family_counts", "execution_shaped_program"], 0) or 0
            )
            ablation = read_json(REPORTS / "execution_shape_private_ablation.json", {})
            ablation_path = REPORTS / "execution_shape_private_ablation.json"
            ablation_mtime = file_mtime(ablation_path)
            skeleton_ok = bool(get_path(ablation, ["summary", "skeleton_competitive_with_semantic"], False))
            private_gate = execution_shape_private_gate(ablation)
            decoder_fingerprint = decoder_relevant_source_fingerprint()
            report_fingerprint = str(get_path(ablation, ["summary", "decoder_relevant_source_fingerprint"], "") or "")
            calibration_mtime = four_card_calibration_mtime()
            decoder_current = (
                report_fingerprint == decoder_fingerprint
                if report_fingerprint
                else calibration_mtime >= ablation_mtime
            )
            private_gate_current = (
                private_gate["ready_for_public_calibration"]
                and ablation_mtime >= float(execution_pressure.get("report_mtime") or 0.0)
                and decoder_current
            )
            calibration_stale_after_private_gate = bool(ablation_mtime) and (
                not calibration_mtime or calibration_mtime < ablation_mtime
            )
            ready = (
                feedback_rows > 0
                and execution_contract_rows > 0
                and private_gate_current
                and (
                    bool(execution_pressure.get("waiting_for_recalibration"))
                    or calibration_stale_after_private_gate
                )
            )
            concept["status"] = "ready" if ready else ("regression_only" if execution_pressure.get("report_mtime") else "diagnostic")
            if concept["status"] == "ready":
                epoch_payload = {
                    "concept": concept["concept"],
                    "execution_report_mtime": execution_pressure.get("report_mtime"),
                    "last_four_card_calibration_mtime": execution_pressure.get("calibration_mtime"),
                    "execution_contract_rows": execution_contract_rows,
                    "private_ablation_mtime": ablation_mtime,
                    "decoder_relevant_source_fingerprint": decoder_fingerprint,
                }
                concept["rotation_epoch"] = hashlib.sha256(
                    json.dumps(epoch_payload, sort_keys=True).encode("utf-8")
                ).hexdigest()[:12]
            concept["evidence"] = {
                "aggregate_floor_gap": floor_gap,
                "execution_shaped_private_pressure_state": execution_pressure,
                "type_contract_feedback_rows": feedback_rows,
                "execution_shaped_decoder_contract_rows": execution_contract_rows,
                "private_ablation_gate_current": private_gate_current,
                "private_ablation_gate": private_gate,
                "calibration_stale_after_private_gate": calibration_stale_after_private_gate,
                "private_ablation_trigger_state": ablation.get("trigger_state"),
                "private_ablation_skeleton_competitive": skeleton_ok,
                "private_ablation_mtime": ablation_mtime or None,
                "decoder_relevant_source_fingerprint": decoder_fingerprint,
                "reported_decoder_relevant_source_fingerprint": report_fingerprint or None,
                "decoder_current": decoder_current,
                "public_data_rule": "public_calibration_only_visible_prompts_and_scorer_tests",
                "success_bar": "BigCodeBench_ge_1_of_32_no_LiveCodeBench_HumanEval_MBPP_EvalPlus_STS_leakage_template_wrapper_regression",
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == "type_contract_four_card_calibration":
            feedback_rows = int(get_path(type_contract, ["summary", "feedback_rows_written"], 0) or 0)
            last_summary = read_json(REPORTS / "broad_transfer_closure_runner_source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32.json", {})
            last_rows = int(get_path(last_summary, ["summary", "high_transfer_private_train_task_count"], 0) or 0)
            teacher = read_json(REPORTS / "teacher_architecture_guidance_edge_conditions_last.json", {})
            teacher_experiment = str(get_path(teacher, ["response_json", "experiment_spec", "id"], "") or "")
            concept["status"] = (
                "regression_only"
                if last_rows > 0 and teacher_experiment == "edge_exec_repair_v1_private_first"
                else ("ready" if feedback_rows > 0 and floor_gap > 0 else "diagnostic")
            )
            if concept["status"] == "ready":
                concept["rotation_epoch"] = code_transfer_rotation_epoch(
                    concept["concept"],
                    transfer=transfer,
                    broad=broad,
                    guidance=guidance,
                )
            concept["evidence"] = {
                "type_contract_feedback_rows": feedback_rows,
                "last_four_card_high_transfer_rows": last_rows,
                "aggregate_floor_gap": floor_gap,
                "public_data_rule": "public_calibration_only_visible_prompts_and_scorer_tests",
                "superseded_by_teacher_experiment": teacher_experiment,
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == "edge_exec_repair_four_card_calibration":
            teacher = read_json(REPORTS / "teacher_architecture_guidance_edge_conditions_last.json", {})
            experiment_id = str(get_path(teacher, ["response_json", "experiment_spec", "id"], "") or "")
            edge_report = read_json(
                REPORTS
                / "real_code_benchmark_graduation_source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32.json",
                {},
            )
            edge_lifecycle = edge_exec_repair_lifecycle(edge_report, broad)
            edge_pressure = next(
                (
                    row
                    for row in transfer.get("concept_transfer_pressure", [])
                    if isinstance(row, dict) and row.get("concept") == "edge_conditions"
                ),
                {},
            )
            edge_residuals = int(edge_pressure.get("residual_count") or 0)
            concept["status"] = (
                "regression_only"
                if edge_lifecycle["graduated"]
                else ("ready" if floor_gap > 0 and edge_residuals > 0 else "diagnostic")
            )
            if concept["status"] == "ready":
                base_epoch = code_transfer_rotation_epoch(
                    concept["concept"],
                    transfer=transfer,
                    broad=broad,
                    guidance=teacher or guidance,
                )
                epoch_payload = {
                    "base_epoch": base_epoch,
                    "edge_lifecycle_reason": edge_lifecycle.get("reason"),
                    "bigcodebench_rate": edge_lifecycle.get("bigcodebench_rate"),
                    "livecodebench_rate": edge_lifecycle.get("livecodebench_rate"),
                    "decoder_source_mtime": file_mtime(
                        ROOT / "crates" / "symliquid-cli" / "src" / "code_lm_closure.rs"
                    ),
                }
                concept["rotation_epoch"] = hashlib.sha256(
                    json.dumps(epoch_payload, sort_keys=True).encode("utf-8")
                ).hexdigest()[:12]
            concept["evidence"] = {
                "teacher_experiment_id": experiment_id,
                "edge_residual_count": edge_residuals,
                "aggregate_floor_gap": floor_gap,
                "public_data_rule": "public_calibration_only_visible_prompts_and_scorer_tests",
                "success_bar": "BigCodeBench_nonzero_no_HumanEval_STS_leakage_template_wrapper_regression",
                "edge_exec_lifecycle": edge_lifecycle,
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == "private_pressure_four_card_recalibration":
            pressure_state = private_pressure_recalibration_state()
            balanced_state = balanced_edge_contract_experiment_state(guidance)
            edge_full_state = edge_case_full_body_experiment_state(guidance)
            edge_v2_state = edge_contract_v2_experiment_state(guidance)
            specialized_blocks = (
                balanced_state["blocks_public_recalibration"]
                or edge_full_state["blocks_public_recalibration"]
                or edge_v2_state["blocks_public_recalibration"]
            )
            broad_private_gate_ready = bool(
                pressure_state.get("private_pressure_closure_after_calibration")
                and get_path(
                    pressure_state,
                    ["decoder_v2_private_ablation_gate", "ready_for_public_calibration"],
                    False,
                )
            )
            specialized_blocks_superseded = bool(specialized_blocks and broad_private_gate_ready)
            concept["status"] = (
                "ready"
                if pressure_state["ready_for_recalibration"]
                and floor_gap > 0
                and (not specialized_blocks or specialized_blocks_superseded)
                else "diagnostic"
            )
            if concept["status"] == "ready":
                concept["rotation_epoch"] = pressure_state["rotation_epoch"]
            concept["evidence"] = {
                "aggregate_floor_gap": floor_gap,
                "ready_for_recalibration": pressure_state["ready_for_recalibration"],
                "reason": pressure_state["reason"],
                "broad_private_gate_ready": broad_private_gate_ready,
                "specialized_blocks_superseded_by_broad_private_gate": specialized_blocks_superseded,
                "balanced_edge_contract_block": balanced_state["blocks_public_recalibration"],
                "balanced_edge_contract_state": balanced_state,
                "edge_case_full_body_block": edge_full_state["blocks_public_recalibration"],
                "edge_case_full_body_state": edge_full_state,
                "edge_contract_v2_block": edge_v2_state["blocks_public_recalibration"],
                "edge_contract_v2_state": edge_v2_state,
                "latest_private_pressure_report": pressure_state["latest_private_pressure_report"],
                "latest_private_pressure_mtime": pressure_state["latest_private_pressure_mtime"],
                "last_four_card_calibration_mtime": pressure_state["last_four_card_calibration_mtime"],
                "public_data_rule": "public_calibration_only_visible_prompts_and_scorer_tests",
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == "multi_turn_conversation":
            conv_score = conversation_state["accuracy"]
            concept["status"] = "regression_only" if conversation_state["graduated"] else "ready"
            concept["evidence"] = {
                "conversation_accuracy": conv_score,
                "case_count": conversation_state["case_count"],
                "turn_count": conversation_state["turn_count"],
                "suite_mode": conversation_state["suite_mode"],
                "graduated": conversation_state["graduated"],
                "graduation_reason": conversation_state["reason"],
                "conversation_focus": conversation_focus,
                "rotation_epoch": concept.get("rotation_epoch"),
            }
            if not conversation_state["graduated"]:
                concept["suite_mode"] = "large"
                concept["large_case_target"] = CONVERSATION_LARGE_CASE_TARGET
        elif concept["concept"] == "multi_turn_conversation_hard":
            hard_due = "multi_turn_conversation_hard" in generalist_rotation["due_concepts"]
            hard_unmastered = not conversation_hard_state["graduated"]
            concept["status"] = (
                "ready"
                if conversation_state["graduated"]
                and hard_unmastered
                and not conversation_hard_v2_state["graduated"]
                else "regression_only"
            )
            if concept["status"] == "ready":
                concept["rotation_epoch"] = generalist_rotation["rotation_epoch"] if hard_due else rotation_epoch
                if generalist_rotation["first_due_concept"] == "multi_turn_conversation_hard":
                    concept["priority"] = "critical"
            concept["evidence"] = {
                "base_conversation_graduated": conversation_state["graduated"],
                "base_graduation_reason": conversation_state["reason"],
                "hard_conversation_accuracy": conversation_hard_state["accuracy"],
                "hard_case_count": conversation_hard_state["case_count"],
                "hard_turn_count": conversation_hard_state["turn_count"],
                "hard_suite_mode": conversation_hard_state["suite_mode"],
                "hard_graduated": conversation_hard_state["graduated"],
                "hard_graduation_reason": conversation_hard_state["reason"],
                "hard_due_but_mastered": hard_due and conversation_hard_state["graduated"],
                "post_graduation_rule": "mastered conversation hard lanes are regression-only; generalist rotation cannot rerun them as fresh progress",
                "hard_case_target": CONVERSATION_HARD_CASE_TARGET,
                "conversation_focus": conversation_focus,
                "generalist_rotation": generalist_rotation,
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == "multi_turn_conversation_hard_v2":
            hard_v2_due = "multi_turn_conversation_hard_v2" in generalist_rotation["due_concepts"]
            hard_v2_unmastered = not conversation_hard_v2_state["graduated"]
            concept["status"] = (
                "ready"
                if conversation_state["graduated"]
                and conversation_hard_state["graduated"]
                and hard_v2_unmastered
                and not conversation_hard_v3_state["graduated"]
                else "regression_only"
            )
            if concept["status"] == "ready":
                concept["rotation_epoch"] = generalist_rotation["rotation_epoch"] if hard_v2_due else rotation_epoch
                if generalist_rotation["first_due_concept"] == "multi_turn_conversation_hard_v2":
                    concept["priority"] = "critical"
            concept["evidence"] = {
                "base_conversation_graduated": conversation_state["graduated"],
                "hard_conversation_graduated": conversation_hard_state["graduated"],
                "hard_v2_accuracy": conversation_hard_v2_state["accuracy"],
                "hard_v2_case_count": conversation_hard_v2_state["case_count"],
                "hard_v2_turn_count": conversation_hard_v2_state["turn_count"],
                "hard_v2_suite_mode": conversation_hard_v2_state["suite_mode"],
                "hard_v2_graduated": conversation_hard_v2_state["graduated"],
                "hard_v2_graduation_reason": conversation_hard_v2_state["reason"],
                "hard_v2_due_but_mastered": hard_v2_due and conversation_hard_v2_state["graduated"],
                "post_graduation_rule": "mastered conversation hard lanes are regression-only; generalist rotation cannot rerun them as fresh progress",
                "hard_v2_case_target": CONVERSATION_HARD_V2_CASE_TARGET,
                "conversation_focus": conversation_focus,
                "generalist_rotation": generalist_rotation,
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == "multi_turn_conversation_hard_v3":
            hard_v3_due = "multi_turn_conversation_hard_v3" in generalist_rotation["due_concepts"]
            hard_v3_unmastered = not conversation_hard_v3_state["graduated"]
            concept["status"] = (
                "ready"
                if conversation_state["graduated"]
                and conversation_hard_state["graduated"]
                and conversation_hard_v2_state["graduated"]
                and hard_v3_unmastered
                else "regression_only"
            )
            if concept["status"] == "ready":
                concept["rotation_epoch"] = generalist_rotation["rotation_epoch"] if hard_v3_due else rotation_epoch
                if generalist_rotation["first_due_concept"] == "multi_turn_conversation_hard_v3":
                    concept["priority"] = "critical"
            concept["evidence"] = {
                "base_conversation_graduated": conversation_state["graduated"],
                "hard_conversation_graduated": conversation_hard_state["graduated"],
                "hard_v2_graduated": conversation_hard_v2_state["graduated"],
                "hard_v3_accuracy": conversation_hard_v3_state["accuracy"],
                "hard_v3_case_count": conversation_hard_v3_state["case_count"],
                "hard_v3_turn_count": conversation_hard_v3_state["turn_count"],
                "hard_v3_suite_mode": conversation_hard_v3_state["suite_mode"],
                "hard_v3_graduated": conversation_hard_v3_state["graduated"],
                "hard_v3_graduation_reason": conversation_hard_v3_state["reason"],
                "hard_v3_due_but_mastered": hard_v3_due and conversation_hard_v3_state["graduated"],
                "post_graduation_rule": "mastered conversation hard lanes are regression-only; generalist rotation cannot rerun them as fresh progress",
                "hard_v3_case_target": CONVERSATION_HARD_V3_CASE_TARGET,
                "hard_v3_graduation_accuracy": 0.95,
                "conversation_focus": conversation_focus,
                "generalist_rotation": generalist_rotation,
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == "multi_turn_conversation_hard_v4":
            hard_v4_due = "multi_turn_conversation_hard_v4" in generalist_rotation["due_concepts"]
            hard_v4_unmastered = not conversation_hard_v4_state["graduated"]
            concept["status"] = (
                "ready"
                if conversation_state["graduated"]
                and conversation_hard_state["graduated"]
                and conversation_hard_v2_state["graduated"]
                and conversation_hard_v3_state["graduated"]
                and hard_v4_unmastered
                else "regression_only"
            )
            if concept["status"] == "ready":
                concept["rotation_epoch"] = generalist_rotation["rotation_epoch"] if hard_v4_due else rotation_epoch
                if generalist_rotation["first_due_concept"] == "multi_turn_conversation_hard_v4":
                    concept["priority"] = "critical"
            concept["evidence"] = {
                "base_conversation_graduated": conversation_state["graduated"],
                "hard_conversation_graduated": conversation_hard_state["graduated"],
                "hard_v2_graduated": conversation_hard_v2_state["graduated"],
                "hard_v3_graduated": conversation_hard_v3_state["graduated"],
                "hard_v4_accuracy": conversation_hard_v4_state["accuracy"],
                "hard_v4_case_count": conversation_hard_v4_state["case_count"],
                "hard_v4_turn_count": conversation_hard_v4_state["turn_count"],
                "hard_v4_suite_mode": conversation_hard_v4_state["suite_mode"],
                "hard_v4_graduated": conversation_hard_v4_state["graduated"],
                "hard_v4_graduation_reason": conversation_hard_v4_state["reason"],
                "hard_v4_due_but_mastered": hard_v4_due and conversation_hard_v4_state["graduated"],
                "post_graduation_rule": "hard_v4 is the current hardest conversation surface; once graduated, it becomes regression-only instead of rerunning as a generalist rotation task",
                "hard_v4_case_target": CONVERSATION_HARD_V4_CASE_TARGET,
                "hard_v4_graduation_accuracy": 0.97,
                "conversation_focus": conversation_focus,
                "generalist_rotation": generalist_rotation,
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == "open_conversation_pantry":
            private_rows = get_path(conversation_pantry, ["summary", "private_train_rows"], None)
            sts_rows = get_path(conversation_pantry, ["summary", "sts_rows"], None)
            enough_rows = int(private_rows or 0) >= 900 and int(sts_rows or 0) >= 900
            concept["status"] = "regression_only" if conversation_state["graduated"] and enough_rows else "ready"
            concept["evidence"] = {
                "private_train_rows": private_rows,
                "sts_rows": sts_rows,
                "conversation_graduated": conversation_state["graduated"],
                "graduation_reason": conversation_state["reason"],
                "conversation_focus": conversation_focus,
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == "repo_repair":
            trace_count = get_path(repo, ["summary", "training_rows"], get_path(repo, ["summary", "task_count"], None))
            due = "repo_repair" in generalist_rotation["due_concepts"]
            concept["status"] = "ready" if due or not trace_count else "regression_only"
            if due:
                concept["rotation_epoch"] = generalist_rotation["rotation_epoch"]
                concept["priority"] = "critical" if generalist_rotation["first_due_concept"] == "repo_repair" else "high"
            concept["evidence"] = {
                "repo_repair_training_rows": trace_count,
                "generalist_rotation": generalist_rotation,
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == "board_game_rl":
            due = "board_game_rl" in generalist_rotation["due_concepts"]
            concept["status"] = "ready" if due or not board_game else "regression_only"
            if due:
                concept["rotation_epoch"] = generalist_rotation["rotation_epoch"]
                concept["priority"] = "critical" if generalist_rotation["first_due_concept"] == "board_game_rl" else "high"
            concept["evidence"] = {
                "board_game_trigger_state": board_game.get("trigger_state") or board_game.get("status"),
                "board_game_report_age_seconds": report_age_seconds(REPORTS / "board_game_rl_benchmark.json"),
                "generalist_rotation": generalist_rotation,
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == "pufferlib4_rl":
            due = "pufferlib4_rl" in generalist_rotation["due_concepts"]
            concept["status"] = "ready" if due or not pufferlib4_rl else "regression_only"
            if due:
                concept["rotation_epoch"] = generalist_rotation["rotation_epoch"]
                concept["priority"] = "critical" if generalist_rotation["first_due_concept"] == "pufferlib4_rl" else "high"
            concept["evidence"] = {
                "pufferlib4_trigger_state": pufferlib4_rl.get("trigger_state") or pufferlib4_rl.get("status"),
                "pufferlib4_native_backend_ready": get_path(pufferlib4_rl, ["summary", "native_backend_ready"]),
                "pufferlib4_atari_enabled": get_path(pufferlib4_rl, ["summary", "atari_enabled"]),
                "pufferlib4_report_age_seconds": report_age_seconds(REPORTS / "pufferlib4_rl_lane.json"),
                "generalist_rotation": generalist_rotation,
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == "long_horizon_tool_use":
            due = "long_horizon_tool_use" in generalist_rotation["due_concepts"]
            concept["status"] = "ready" if due or not long_horizon else "regression_only"
            if due:
                concept["rotation_epoch"] = generalist_rotation["rotation_epoch"]
                concept["priority"] = "critical" if generalist_rotation["first_due_concept"] == "long_horizon_tool_use" else "high"
            concept["evidence"] = {
                "long_horizon_trigger_state": long_horizon.get("trigger_state"),
                "long_horizon_report_age_seconds": report_age_seconds(REPORTS / "high_transfer_long_horizon_tool_use.json"),
                "generalist_rotation": generalist_rotation,
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        elif concept["concept"] == "cross_domain_sts_capsules":
            due = "cross_domain_sts_capsules" in generalist_rotation["due_concepts"]
            capsule_count = int(get_path(cross_domain_capsules, ["summary", "capsule_count"], 0) or 0)
            concept["status"] = "ready" if due or capsule_count <= 0 else "regression_only"
            if concept["status"] == "ready":
                concept["rotation_epoch"] = generalist_rotation["rotation_epoch"]
                concept["priority"] = "critical" if generalist_rotation["first_due_concept"] == "cross_domain_sts_capsules" else "high"
            concept["evidence"] = {
                "capsule_count": capsule_count,
                "sts_row_count": get_path(cross_domain_capsules, ["summary", "sts_row_count"], None),
                "generalist_rotation": generalist_rotation,
                "rotation_epoch": concept.get("rotation_epoch"),
            }
        else:
            concept["status"] = "ready"
            concept["evidence"] = {"source": "operator_long_horizon"}
        apply_non_promotable_diagnostic_policy(concept)
        rows.append(concept)
    return rows

