#!/usr/bin/env python3
"""Private-row selection and guard weighting for strict MLX adaptation.

These helpers audit admitted private target bodies and select existing private
training rows for adaptation. They do not generate candidates, inspect eval
verifier tests, use public benchmark data, call teachers, or grant learned-code
capability credit.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from neural_seed_code_proposer_comparator import (  # noqa: E402
    deterministic_sample,
    dict_or_empty,
    load_private_rows,
    rel,
    resolve,
    row_id,
    stable_hash,
)
from neural_seed_decode_static_guard import decode_static_guard  # noqa: E402
from strict_generator_mlx_decode_eval import (  # noqa: E402
    source_condition_expectation_from_source_text,
    stable_hash_file,
    strict_generator_decode_source_text,
)
from strict_generator_mlx_replay_selection import (  # noqa: E402
    private_train_replay_tier_inventory,
    private_train_replay_tier_match,
)
from strict_generator_mlx_pretraining_probe import allowed_parameter_names_from_source_text  # noqa: E402


def config_with_teacher_curriculum_mode(
    config: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Apply an explicit teacher-curriculum ablation without mutating caller state."""

    selected = str(mode or "auto").strip().lower()
    if selected not in {"auto", "on", "off"}:
        raise ValueError(f"unsupported teacher curriculum mode: {mode}")
    result = copy.deepcopy(config)
    if selected == "auto":
        return result
    teacher_cfg = dict_or_empty(result.get("teacher_distillation"))
    teacher_cfg["enabled"] = selected == "on"
    result["teacher_distillation"] = teacher_cfg
    return result


def strict_target_guard_rows(
    bodies: list[str],
    source_texts: list[str],
    *,
    split_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    failure_counts: dict[str, int] = {}
    pass_count = 0
    for index, body in enumerate(bodies):
        source_text = source_texts[index] if index < len(source_texts) else ""
        allowed = allowed_parameter_names_from_source_text(source_text)
        if not allowed:
            allowed = {"data"}
        guard = decode_static_guard(
            str(body or ""),
            allowed_names=allowed,
            require_parameter_use=True,
            require_nontrivial_return=True,
            require_top_level_return=True,
        )
        passed = bool(guard.get("passed"))
        if passed:
            pass_count += 1
        for failure in list(guard.get("failures") or []):
            key = str(failure)
            failure_counts[key] = failure_counts.get(key, 0) + 1
        rows.append(
            {
                "index": index,
                "passed": passed,
                "failures": [str(item) for item in list(guard.get("failures") or [])],
                "allowed_parameter_count": len(allowed),
                "dependency": guard.get("dependency"),
                "definite_assignment": guard.get("definite_assignment"),
                "control_flow_pathology": guard.get("control_flow_pathology"),
            }
        )
    total = len(rows)
    examples = [
        {
            "index": row["index"],
            "failures": row["failures"],
            "dependency": row.get("dependency"),
            "definite_assignment": row.get("definite_assignment"),
            "control_flow_pathology": row.get("control_flow_pathology"),
        }
        for row in rows
        if not bool(row.get("passed"))
    ][:8]
    return rows, {
        "enabled": True,
        "policy": "private_target_body_strict_decode_guard_audit_v1",
        "split": split_name,
        "rows": total,
        "passed": pass_count,
        "failed": total - pass_count,
        "pass_rate": round(pass_count / max(1, total), 6),
        "failure_counts": dict(sorted(failure_counts.items())),
        "failure_examples": examples,
        "score_semantics": (
            "Audits admitted private target bodies against the same task-blind strict decode guard "
            "used for MLX candidate admission. The guard sees only target body text and visible "
            "callable parameter names from source text; it does not inspect tests, eval solutions, "
            "public benchmarks, verifier results, or answer metadata."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def apply_guard_clean_target_weights(
    token_weight_rows: list[list[float]],
    target_guard_rows: list[dict[str, Any]],
    *,
    guard_clean_boost: float,
    rejected_weight: float,
) -> tuple[list[list[float]], dict[str, Any]]:
    clean_boost = max(0.0, float(guard_clean_boost if guard_clean_boost is not None else 1.0))
    rejected = max(0.0, float(rejected_weight if rejected_weight is not None else 1.0))
    if clean_boost == 1.0 and rejected == 1.0:
        return token_weight_rows, {
            "enabled": False,
            "policy": "private_target_body_strict_decode_guard_loss_weighting_v1",
            "reason": "identity_weights",
            "guard_clean_target_loss_boost": clean_boost,
            "guard_rejected_target_loss_weight": rejected,
            "rows": len(token_weight_rows),
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }
    adjusted: list[list[float]] = []
    clean_rows = 0
    rejected_rows = 0
    weighted_token_count = 0
    for index, row in enumerate(token_weight_rows):
        guard = target_guard_rows[index] if index < len(target_guard_rows) else {}
        multiplier = clean_boost if bool(guard.get("passed")) else rejected
        if bool(guard.get("passed")):
            clean_rows += 1
        else:
            rejected_rows += 1
        weighted_row = [float(value) * multiplier for value in row]
        if multiplier != 1.0:
            weighted_token_count += len(weighted_row)
        adjusted.append(weighted_row)
    return adjusted, {
        "enabled": True,
        "policy": "private_target_body_strict_decode_guard_loss_weighting_v1",
        "guard_clean_target_loss_boost": clean_boost,
        "guard_rejected_target_loss_weight": rejected,
        "rows": len(token_weight_rows),
        "guard_clean_rows": clean_rows,
        "guard_rejected_rows": rejected_rows,
        "weighted_token_count": weighted_token_count,
        "score_semantics": (
            "Optional supervised loss weighting over admitted private target rows, derived only from "
            "task-blind strict-guard audit of the private target body and visible callable parameter "
            "names. It does not synthesize code, inspect tests/solutions, use public data, alter "
            "verifier scoring, or grant candidate-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def select_private_train_tier_rows(rows: list[dict[str, Any]], *, tier: str) -> dict[str, Any]:
    tier_name = str(tier or "any").strip() or "any"
    inventory = private_train_replay_tier_inventory(rows)
    if tier_name == "any":
        selected = list(rows)
    else:
        selected = [row for row in rows if private_train_replay_tier_match(row, tier_name)]
    return {
        "enabled": tier_name != "any",
        "tier": tier_name,
        "policy": "private_adaptation_existing_row_tier_selection_v1",
        "input_rows": len(rows),
        "selected_rows": len(selected),
        "inventory": inventory,
        "rows": selected,
        "score_semantics": (
            "Selects existing private train rows by private solution-body structural complexity before "
            "sampling adaptation rows. This affects only which admitted private targets are used for "
            "training/eval loss. Generation source text remains strict prompt/signature-only, and this "
            "is not heldout transfer evidence, public calibration, or learned-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def tier_balanced_private_train_sample(rows: list[dict[str, Any]], *, limit: int, seed: int) -> dict[str, Any]:
    requested = max(0, int(limit or 0))
    tier_names = ["simple_return", "loop_accumulate", "algorithmic_small"]
    buckets: dict[str, list[dict[str, Any]]] = {
        tier_name: [row for row in rows if private_train_replay_tier_match(row, tier_name)]
        for tier_name in tier_names
    }
    active_tiers = [tier_name for tier_name in tier_names if buckets.get(tier_name)]
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    tier_counts: dict[str, int] = {tier_name: 0 for tier_name in tier_names}
    target_per_tier = max(1, requested // max(1, len(active_tiers))) if requested else 0
    remainder_budget = max(0, requested - (target_per_tier * max(1, len(active_tiers))))

    for tier_index, tier_name in enumerate(active_tiers):
        bucket = buckets[tier_name]
        quota = min(len(bucket), target_per_tier + (1 if tier_index < remainder_budget else 0))
        sampled = deterministic_sample(bucket, quota, seed + 1009 * (tier_index + 1))
        for row in sampled:
            key = private_train_row_key(row)
            if key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(key)
            tier_counts[tier_name] += 1

    if len(selected) < requested:
        remaining = [row for row in rows if private_train_row_key(row) not in selected_keys]
        fill = deterministic_sample(remaining, requested - len(selected), seed + 7919)
        for row in fill:
            key = private_train_row_key(row)
            if key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(key)
            matched = [tier_name for tier_name in tier_names if private_train_replay_tier_match(row, tier_name)]
            tier_counts[matched[0] if matched else "unclassified"] = tier_counts.get(matched[0] if matched else "unclassified", 0) + 1

    selected.sort(key=lambda row: stable_hash(f"tier-balanced:{seed}:{private_train_row_key(row)}"))
    return {
        "enabled": True,
        "policy": "private_adaptation_existing_row_tier_balanced_sampling_v1",
        "requested_rows": requested,
        "input_rows": len(rows),
        "active_tiers": active_tiers,
        "target_per_active_tier": target_per_tier,
        "selected_rows": len(selected),
        "tier_counts": tier_counts,
        "rows": selected,
        "score_semantics": (
            "Deterministically samples admitted private training rows across existing structural replay tiers "
            "before train/eval splitting so one abundant family cannot dominate the learned prefix/body update. "
            "This does not create synthetic rows, inspect public data, inspect eval tests/solutions, or grant "
            "candidate-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def private_train_row_key(row: dict[str, Any]) -> str:
    if not isinstance(row, dict):
        return stable_hash(repr(row))
    preferred = row.get("task_id") or row.get("source_task_id") or row.get("entry_point")
    if preferred:
        return str(preferred)
    return stable_hash(json.dumps(row, sort_keys=True, default=str))


def private_train_balanced_sample_summary(selection: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in selection.items() if key != "rows"}


def private_train_tier_summary(selection: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in selection.items() if key != "rows"}


def private_train_tier_vocab_summary(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(selection.get("enabled")),
        "tier": selection.get("tier"),
        "policy": selection.get("policy"),
        "input_rows": int(selection.get("input_rows") or 0),
        "selected_rows": int(selection.get("selected_rows") or 0),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }

SEMANTIC_CONSTRUCTION_REPAIR_PROFILES: dict[str, dict[str, Any]] = {
    "strict_full_body_semantic_construction_v1": {
        "policy": "private_strict_full_body_semantic_construction_repair_profile_v1",
        "target_failure_modes": [
            "prompt_implies_branching_but_no_branch",
            "prompt_implies_structured_output_but_no_collection_construction",
            "prompt_implies_string_processing_but_no_string_ops",
            "loop_update_semantics_collapse_to_identity_or_additive_state",
            "body_token_transition_after_semantic_prefix_collapses_to_generic_accumulator",
        ],
        "required_components": [
            "semantic_plan_auxiliary",
            "semantic_slot_auxiliary",
            "semantic_plan_visible_operation_weighting",
            "source_condition_internalization_weighting",
            "semantic_slot_prefix_weighting",
            "loop_semantic_operation_weighting",
            "loop_expression_synthesis_weighting",
            "plan_conditioned_body_weighting",
            "update_contract_consistency_weighting",
            "semantic_ir_obligation_weighting",
            "source_contrastive_loss",
        ],
        "overrides": {
            "train_tier": "any",
            "tier_balanced_sampling": True,
            "source_condition_operation_coverage_min_rows": 2,
            "semantic_plan_loss_weight": 0.08,
            "semantic_slot_loss_weight": 0.06,
            "semantic_plan_visible_operation_loss_boost": 1.5,
            "source_contrastive_loss_weight": 0.05,
            "source_contrastive_prefix_tokens": 48,
            "enable_primary_dataflow_weights": True,
            "primary_dataflow_weight_scale": 1.2,
            "return_expression_loss_boost": 2.0,
            "source_condition_internalization_loss_boost": 2.4,
            "loop_semantic_operation_loss_boost": 3.0,
            "loop_semantic_operation_roles": "loop_semantic_update,top_level_semantic_finalizer",
            "semantic_ir_obligation_loss_boost": 3.0,
            "semantic_slot_prefix_loss_boost": 2.2,
            "semantic_slot_prefix_roles": "plan,loop_source,update,guard,finalizer,state,binding,statement",
            "loop_expression_synthesis_loss_boost": 3.0,
            "loop_expression_synthesis_roles": (
                "loop_condition_expression,loop_update_expression,top_level_finalizer_expression"
            ),
            "plan_conditioned_body_loss_boost": 2.8,
            "plan_conditioned_body_roles": (
                "guard_expression,loop_source_expression,loop_condition_expression,"
                "loop_update_statement,plan_key_call_expression,final_return_expression"
            ),
            "update_contract_consistency_loss_boost": 3.0,
        },
        "score_semantics": (
            "Composes existing strict-generator private-only objectives that directly target learned "
            "full-body semantic construction. It does not add templates, renderers, tools, answer-family "
            "labels, public benchmark data, eval tests, eval solutions, teacher output, or candidate credit. "
            "The resulting checkpoint is private repair evidence only until verifier-passing behavior moves."
        ),
    },
    "strict_direct_body_emission_path_v1": {
        "policy": "private_strict_direct_body_emission_path_repair_profile_v1",
        "target_failure_modes": [
            "strict_decode_emits_zero_candidate_rows",
            "body_token_transition_after_plan_prefix_never_reaches_top_level_return",
            "non_loop_task_missing_state_binding_and_return_path",
            "loop_update_semantics_not_connected_to_final_return",
            "prompt_visible_source_condition_operation_not_internalized",
            "policy_gap_improves_without_replay_candidate_emission",
        ],
        "required_components": [
            "source_condition_internalization_weighting",
            "direct_body_emission_path_weighting",
            "local_return_closure_weighting",
            "closed_state_transition_weighting",
            "body_transition_auxiliary",
            "body_action_auxiliary",
            "body_operand_auxiliary",
            "body_aux_semantic_event_weighting",
            "body_state_machine_event_weighting",
            "body_state_event_auxiliary",
            "loop_semantic_operation_weighting",
            "loop_expression_synthesis_weighting",
            "plan_conditioned_body_weighting",
            "update_contract_consistency_weighting",
            "semantic_ir_obligation_weighting",
            "source_contrastive_loss",
        ],
        "overrides": {
            "train_tier": "any",
            "tier_balanced_sampling": True,
            "source_condition_operation_coverage_min_rows": 2,
            "source_contrastive_loss_weight": 0.05,
            "source_contrastive_prefix_tokens": 48,
            "enable_primary_dataflow_weights": True,
            "primary_dataflow_weight_scale": 1.2,
            "return_expression_loss_boost": 3.0,
            "source_condition_internalization_loss_boost": 2.6,
            "direct_body_emission_loss_boost": 4.0,
            "direct_body_emission_roles": (
                "top_level_state_binding,top_level_state_update,top_level_branch_guard,"
                "top_level_loop_statement,loop_source_expression,loop_condition_expression,"
                "loop_body_state_transition,branch_body_state_transition,top_level_local_state_return,"
                "top_level_return_expression"
            ),
            "local_return_closure_loss_boost": 5.0,
            "local_return_closure_roles": (
                "previous_local_state_transition,block_exit_local_return_closure,"
                "final_return_local_name,final_return_local_expression"
            ),
            "closed_state_transition_loss_boost": 3.8,
            "closed_state_transition_roles": (
                "closed_top_level_control_block,closed_nested_control_block,"
                "control_body_state_transition,block_exit_to_next_statement,"
                "top_level_finalizer_return"
            ),
            "body_transition_loss_weight": 0.08,
            "body_action_class_balance": True,
            "body_action_class_balance_min_weight": 0.35,
            "body_action_class_balance_max_weight": 3.0,
            "body_operand_class_balance": True,
            "body_operand_class_balance_min_weight": 0.35,
            "body_operand_class_balance_max_weight": 3.0,
            "body_aux_semantic_event_weighting": True,
            "body_aux_semantic_event_min_factor": 1.0,
            "body_aux_semantic_event_max_factor": 3.0,
            "body_state_machine_event_weighting": True,
            "body_state_machine_event_factor": 2.25,
            "body_state_machine_non_event_scale": 0.35,
            "body_state_machine_operand_event_factor": 1.25,
            "body_state_machine_operand_non_event_scale": 1.0,
            "body_state_event_loss_weight": 0.04,
            "body_state_event_target_event_weight": 1.0,
            "body_state_event_target_none_weight": 0.20,
            "body_action_loss_weight": 0.08,
            "body_operand_loss_weight": 0.10,
            "loop_semantic_operation_loss_boost": 3.2,
            "loop_semantic_operation_roles": "loop_semantic_update,top_level_semantic_finalizer",
            "semantic_ir_obligation_loss_boost": 4.0,
            "loop_expression_synthesis_loss_boost": 3.2,
            "loop_expression_synthesis_roles": (
                "loop_condition_expression,loop_update_expression,top_level_finalizer_expression"
            ),
            "plan_conditioned_body_loss_boost": 3.2,
            "plan_conditioned_body_roles": (
                "guard_expression,loop_source_expression,loop_condition_expression,"
                "loop_update_statement,plan_key_call_expression,final_return_expression"
            ),
            "update_contract_consistency_loss_boost": 3.4,
        },
        "score_semantics": (
            "Composes private-only strict-generator objectives around the current zero-candidate wall: "
            "the direct prompt/signature body-token path must learn top-level state bindings, branch/loop "
            "guards, prompt-visible source-condition operations, loop body updates, and final return "
            "statements from admitted private solution bodies. It does not add templates, renderers, tools, "
            "answer-family labels, public benchmark data, eval tests, eval solutions, teacher output, or "
            "candidate credit. The checkpoint remains private repair evidence only until strict replay emits "
            "non-fallback learned candidates and verifier behavior moves."
        ),
    },
}


def apply_semantic_construction_repair_profile(args: argparse.Namespace) -> dict[str, Any]:
    profile_id = str(getattr(args, "semantic_construction_repair_profile", "none") or "none")
    if profile_id == "none":
        return {
            "enabled": False,
            "profile_id": "none",
            "policy": "not_enabled",
            "candidate_generation_credit": 0,
        }
    profile = dict_or_empty(SEMANTIC_CONSTRUCTION_REPAIR_PROFILES.get(profile_id))
    overrides = dict_or_empty(profile.get("overrides"))
    applied: dict[str, Any] = {}
    preserved: dict[str, Any] = {}

    for key, value in overrides.items():
        current = getattr(args, key)
        should_replace = False
        if isinstance(value, bool):
            should_replace = bool(value) and not bool(current)
        elif isinstance(value, (int, float)):
            should_replace = float(current or 0.0) <= 0.0 or (
                key == "primary_dataflow_weight_scale" and float(current or 0.0) <= 1.0
            )
        elif isinstance(value, str):
            should_replace = not str(current or "").strip() or (key == "train_tier" and str(current or "") == "any")
        else:
            should_replace = current in (None, "", 0, 0.0, False)
        if should_replace:
            setattr(args, key, value)
            applied[key] = value
        else:
            preserved[key] = current

    return {
        "enabled": True,
        "profile_id": profile_id,
        "policy": str(profile.get("policy") or "private_strict_generator_semantic_construction_repair_profile_v1"),
        "target_failure_modes": list(profile.get("target_failure_modes") or []),
        "required_components": list(profile.get("required_components") or []),
        "applied_overrides": applied,
        "preserved_operator_overrides": preserved,
        "public_calibration_eligible": False,
        "candidate_generation_credit": 0,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "external_inference_calls": 0,
        "score_semantics": str(profile.get("score_semantics") or ""),
    }


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def load_adaptation_private_train_rows(
    data_cfg: dict[str, Any],
    *,
    supplemental_private_train_jsonl: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary_path = resolve(str(data_cfg.get("train_jsonl") or ""))
    primary_rows = load_private_rows(primary_path, data_cfg)
    rows: list[dict[str, Any]] = list(primary_rows)
    seen = {row_id(row) for row in rows}
    configured_paths = [resolve(path) for path in string_list(data_cfg.get("adaptation_supplemental_train_jsonl"))]
    requested_paths = [resolve(path) for path in supplemental_private_train_jsonl]
    supplemental_paths: list[Path] = []
    supplemental_seen: set[str] = set()
    for path in [*configured_paths, *requested_paths]:
        key = str(path)
        if key in supplemental_seen:
            continue
        supplemental_seen.add(key)
        supplemental_paths.append(path)

    source_reports: list[dict[str, Any]] = []
    duplicate_row_count = 0
    added_row_count = 0
    for path in supplemental_paths:
        loaded = load_private_rows(path, data_cfg)
        path_added = 0
        path_duplicate = 0
        for row in loaded:
            key = row_id(row)
            if key in seen:
                duplicate_row_count += 1
                path_duplicate += 1
                continue
            seen.add(key)
            rows.append(row)
            added_row_count += 1
            path_added += 1
        source_reports.append(
            {
                "path": rel(path),
                "loaded_rows": len(loaded),
                "added_rows": path_added,
                "duplicate_rows": path_duplicate,
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
                "candidate_generation_credit": 0,
            }
        )

    audit = {
        "enabled": bool(supplemental_paths),
        "policy": "private_adaptation_supplemental_train_jsonl_v1",
        "primary_path": rel(primary_path),
        "primary_rows": len(primary_rows),
        "supplemental_sources": source_reports,
        "supplemental_source_count": len(supplemental_paths),
        "supplemental_added_rows": added_row_count,
        "supplemental_duplicate_rows": duplicate_row_count,
        "total_rows": len(rows),
        "score_semantics": (
            "Loads additional admitted private-training JSONL files into the existing strict-generator "
            "adaptation pool. The same forbidden public-row flag checks apply to every supplemental row. "
            "This changes private training coverage only; generation remains prompt/signature-only, and "
            "the report is public-calibration-ineligible until heldout replay behavior improves."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }
    return rows, audit


def reserve_governed_teacher_train_rows(
    *,
    selected_rows: list[dict[str, Any]],
    eligible_rows: list[dict[str, Any]],
    injected_teacher_rows: list[dict[str, Any]],
    max_train_rows: int,
    max_eval_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    teacher_row_ids = {row_id(row) for row in injected_teacher_rows}
    eligible_teacher_rows = [row for row in eligible_rows if row_id(row) in teacher_row_ids]
    selected_base_rows = [row for row in selected_rows if row_id(row) not in teacher_row_ids]
    eval_count = min(max_eval_rows, max(1, len(selected_base_rows) // 5))
    eval_rows = selected_base_rows[:eval_count]
    train_rows = [
        *eligible_teacher_rows,
        *selected_base_rows[eval_count : eval_count + max(0, max_train_rows - len(eligible_teacher_rows))],
    ]
    return train_rows, eval_rows, {
        "eligible_row_count_after_holdout_and_tier_filters": len(eligible_teacher_rows),
        "reserved_train_position_count": len(eligible_teacher_rows),
        "teacher_rows_in_eval_count": sum(1 for row in eval_rows if row_id(row) in teacher_row_ids),
    }


def source_condition_operation_tags_for_row(
    row: dict[str, Any],
    *,
    source_fields: list[Any],
    source_text_style: str,
    source_vocab: dict[str, int],
) -> set[str]:
    source_text = strict_generator_decode_source_text(
        row,
        source_fields,
        source_text_style=source_text_style,
        source_vocab=source_vocab,
    )
    expectation = source_condition_expectation_from_source_text(source_text)
    return {str(tag) for tag in list(expectation.get("operation_tags") or []) if str(tag)}


def apply_source_condition_operation_coverage_sampling(
    selected: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    *,
    requested_rows: int,
    min_rows_per_tag: int,
    seed: int,
    source_fields: list[Any],
    source_text_style: str,
    source_vocab: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = max(0, int(min_rows_per_tag or 0))
    if target <= 0 or not selected:
        return selected, {
            "enabled": False,
            "policy": "not_enabled",
            "min_rows_per_tag": target,
            "candidate_generation_credit": 0,
        }

    selected_rows = list(selected)
    selected_keys = {row_id(row) for row in selected_rows}
    row_tags: dict[str, set[str]] = {}
    pool_by_tag: dict[str, list[dict[str, Any]]] = {}
    for row in pool:
        key = row_id(row)
        tags = source_condition_operation_tags_for_row(
            row,
            source_fields=source_fields,
            source_text_style=source_text_style,
            source_vocab=source_vocab,
        )
        row_tags[key] = tags
        for tag in tags:
            pool_by_tag.setdefault(tag, []).append(row)

    def tag_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            for tag in row_tags.get(row_id(row), set()):
                counts[tag] = counts.get(tag, 0) + 1
        return counts

    inserted_rows: list[dict[str, Any]] = []
    before_counts = tag_counts(selected_rows)
    for tag in sorted(pool_by_tag):
        current_count = before_counts.get(tag, 0) + sum(1 for row in inserted_rows if tag in row_tags.get(row_id(row), set()))
        need = max(0, target - current_count)
        if need <= 0:
            continue
        candidates = [row for row in pool_by_tag[tag] if row_id(row) not in selected_keys]
        for row in deterministic_sample(candidates, need, seed + stable_int(tag)):
            key = row_id(row)
            if key in selected_keys:
                continue
            selected_keys.add(key)
            inserted_rows.append(row)

    if inserted_rows:
        selected_rows.extend(inserted_rows)
    requested = max(0, int(requested_rows or len(selected_rows)))
    dropped_rows: list[dict[str, Any]] = []
    if requested and len(selected_rows) > requested:
        protected = {row_id(row) for row in inserted_rows}
        selected_rows.sort(key=lambda row: stable_hash(f"source-condition-coverage:{seed}:{row_id(row)}"))
        while len(selected_rows) > requested:
            drop_index = next(
                (
                    index
                    for index in range(len(selected_rows) - 1, -1, -1)
                    if row_id(selected_rows[index]) not in protected
                ),
                len(selected_rows) - 1,
            )
            dropped_rows.append(selected_rows.pop(drop_index))

    after_counts = tag_counts(selected_rows)
    selected_rows.sort(key=lambda row: stable_hash(f"source-condition-coverage-final:{seed}:{row_id(row)}"))
    return selected_rows, {
        "enabled": True,
        "policy": "private_prompt_visible_source_condition_operation_coverage_sampling_v1",
        "min_rows_per_tag": target,
        "requested_rows": requested_rows,
        "pool_rows": len(pool),
        "selected_rows_before": len(selected),
        "selected_rows_after": len(selected_rows),
        "inserted_rows": len(inserted_rows),
        "dropped_rows": len(dropped_rows),
        "operation_tag_counts_before": dict(sorted(before_counts.items())),
        "operation_tag_counts_after": dict(sorted(after_counts.items())),
        "pool_operation_tag_counts": {
            tag: len(rows)
            for tag, rows in sorted(pool_by_tag.items())
        },
        "inserted_row_ids": [row_id(row) for row in inserted_rows[:24]],
        "dropped_row_ids": [row_id(row) for row in dropped_rows[:24]],
        "score_semantics": (
            "Deterministically ensures rare prompt-visible operation contracts are represented in the "
            "private adaptation sample. Tags are derived only from strict prompt/signature source text, "
            "not from hidden tests, eval solutions, verifier labels, answer-family labels, public data, "
            "or target-derived decoder metadata. This is data coverage for supervised private training, "
            "not candidate-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def stable_int(text: str) -> int:
    return int(stable_hash(text)[:8], 16)


def role_csv_to_set(value: str) -> set[str]:
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def merge_role_csv(*values: str) -> str:
    roles: set[str] = set()
    for value in values:
        roles.update(role_csv_to_set(value))
    return ",".join(sorted(roles))


def optional_json_report(path: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if path is None:
        return {}, {
            "path": "",
            "present": False,
            "loaded": False,
            "reason": "not_requested",
        }
    try:
        resolved = resolve(path)
    except Exception:
        resolved = Path(path)
    if not resolved.exists():
        return {}, {
            "path": rel(resolved),
            "present": False,
            "loaded": False,
            "reason": "missing_report",
        }
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, {
            "path": rel(resolved),
            "present": True,
            "loaded": False,
            "reason": "read_failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:240],
        }
    return payload, {
        "path": rel(resolved),
        "present": True,
        "loaded": True,
        "sha256": stable_hash_file(resolved),
    }


def numeric_counts(value: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(value, dict):
        return counts
    for key, raw in value.items():
        try:
            count = int(raw or 0)
        except (TypeError, ValueError):
            continue
        if count > 0:
            counts[str(key)] = count
    return dict(sorted(counts.items()))


def build_semantic_ir_obligation_weighting_plan(
    *,
    semantic_ir_obligation_report: Path | None,
    current_wall_report: Path | None,
    boost: float,
    source_condition_internalization_loss_boost: float,
    source_condition_operation_coverage_min_rows: int,
    loop_semantic_operation_loss_boost: float,
    loop_semantic_operation_roles: str,
    loop_expression_synthesis_loss_boost: float,
    loop_expression_synthesis_roles: str,
    plan_conditioned_body_loss_boost: float,
    plan_conditioned_body_roles: str,
    direct_body_emission_loss_boost: float,
    direct_body_emission_roles: str,
    local_return_closure_loss_boost: float,
    local_return_closure_roles: str,
    closed_state_transition_loss_boost: float,
    closed_state_transition_roles: str,
) -> dict[str, Any]:
    requested_boost = max(0.0, float(boost if boost is not None else 0.0))
    bridge, bridge_source = optional_json_report(semantic_ir_obligation_report)
    wall, wall_source = optional_json_report(current_wall_report)
    issue_counts = numeric_counts(dict_or_empty(bridge.get("summary")).get("issue_label_counts"))
    wall_summary = dict_or_empty(wall.get("summary"))
    dominant_residuals = numeric_counts(
        wall_summary.get("current_wall_dominant_residuals")
        or dict_or_empty(wall.get("current_wall")).get("dominant_residuals")
    )

    local_labels = {
        "missing_local_return_closure",
        "missing_local_return",
        "current_line_starts_return",
        "unfinished_return_expression",
        "block_exit_without_finalizer",
    }
    closed_state_labels = {
        "repeated_nested_guard_without_progress",
        "block_exit_without_finalizer",
        "loop_without_decision_or_state_update",
        "inside_loop",
        "inside_loop_without_update",
        "inside_loop_at_blank_line",
        "zero_candidate_task",
    }
    semantic_operation_labels = {
        "missing_semantic_update_value",
        "missing_gcd_call",
        "missing_list_construction",
        "missing_numeric_accumulation",
        "missing_windowed_finalizer",
        "missing_rle_branch_or_update",
        "shallow_identity_accumulation",
        "append_accumulator_to_itself",
        "loop_without_decision_or_state_update",
        "inside_loop_without_update",
    }
    direct_body_labels = {
        "zero_candidate_task",
        "missing_local_return",
        "current_line_starts_return",
        "missing_list_construction",
        "missing_semantic_update_value",
        "loop_without_decision_or_state_update",
    }
    source_condition_labels = {
        "missing_semantic_update_value",
        "missing_gcd_call",
        "missing_list_construction",
        "missing_numeric_accumulation",
        "missing_windowed_finalizer",
        "missing_rle_branch_or_update",
    }

    all_counts: dict[str, int] = {}
    for key, count in issue_counts.items():
        all_counts[key] = all_counts.get(key, 0) + int(count)
    for key, count in dominant_residuals.items():
        all_counts[key] = all_counts.get(key, 0) + int(count)

    def matched_count(labels: set[str]) -> int:
        return sum(int(all_counts.get(label, 0)) for label in labels)

    objective_counts = {
        "local_return_closure": matched_count(local_labels),
        "closed_state_transition": matched_count(closed_state_labels),
        "loop_semantic_operation": matched_count(semantic_operation_labels),
        "loop_expression_synthesis": matched_count(semantic_operation_labels | closed_state_labels),
        "plan_conditioned_body": matched_count(semantic_operation_labels | local_labels | closed_state_labels),
        "direct_body_emission": matched_count(direct_body_labels),
        "source_condition_internalization": matched_count(source_condition_labels),
    }
    active_objectives = [
        name for name, count in objective_counts.items() if requested_boost > 0.0 and count > 0
    ]
    enabled = bool(active_objectives)

    effective = {
        "source_condition_internalization_loss_boost": float(source_condition_internalization_loss_boost or 0.0),
        "source_condition_operation_coverage_min_rows": int(source_condition_operation_coverage_min_rows or 0),
        "loop_semantic_operation_loss_boost": float(loop_semantic_operation_loss_boost or 0.0),
        "loop_semantic_operation_roles": str(loop_semantic_operation_roles or ""),
        "loop_expression_synthesis_loss_boost": float(loop_expression_synthesis_loss_boost or 0.0),
        "loop_expression_synthesis_roles": str(loop_expression_synthesis_roles or ""),
        "plan_conditioned_body_loss_boost": float(plan_conditioned_body_loss_boost or 0.0),
        "plan_conditioned_body_roles": str(plan_conditioned_body_roles or ""),
        "direct_body_emission_loss_boost": float(direct_body_emission_loss_boost or 0.0),
        "direct_body_emission_roles": str(direct_body_emission_roles or ""),
        "local_return_closure_loss_boost": float(local_return_closure_loss_boost or 0.0),
        "local_return_closure_roles": str(local_return_closure_roles or ""),
        "closed_state_transition_loss_boost": float(closed_state_transition_loss_boost or 0.0),
        "closed_state_transition_roles": str(closed_state_transition_roles or ""),
    }

    if enabled:
        if objective_counts["source_condition_internalization"] > 0:
            effective["source_condition_internalization_loss_boost"] = max(
                effective["source_condition_internalization_loss_boost"],
                requested_boost,
            )
            effective["source_condition_operation_coverage_min_rows"] = max(
                effective["source_condition_operation_coverage_min_rows"],
                2,
            )
        if objective_counts["loop_semantic_operation"] > 0:
            effective["loop_semantic_operation_loss_boost"] = max(
                effective["loop_semantic_operation_loss_boost"],
                requested_boost,
            )
            effective["loop_semantic_operation_roles"] = merge_role_csv(
                effective["loop_semantic_operation_roles"],
                "loop_semantic_update,top_level_semantic_finalizer",
            )
        if objective_counts["loop_expression_synthesis"] > 0:
            effective["loop_expression_synthesis_loss_boost"] = max(
                effective["loop_expression_synthesis_loss_boost"],
                requested_boost,
            )
            effective["loop_expression_synthesis_roles"] = merge_role_csv(
                effective["loop_expression_synthesis_roles"],
                "loop_condition_expression,loop_update_expression,top_level_finalizer_expression",
            )
        if objective_counts["plan_conditioned_body"] > 0:
            effective["plan_conditioned_body_loss_boost"] = max(
                effective["plan_conditioned_body_loss_boost"],
                requested_boost,
            )
            effective["plan_conditioned_body_roles"] = merge_role_csv(
                effective["plan_conditioned_body_roles"],
                "guard_expression,loop_source_expression,loop_condition_expression,"
                "loop_update_statement,plan_key_call_expression,final_return_expression",
            )
        if objective_counts["direct_body_emission"] > 0:
            effective["direct_body_emission_loss_boost"] = max(
                effective["direct_body_emission_loss_boost"],
                requested_boost,
            )
            effective["direct_body_emission_roles"] = merge_role_csv(
                effective["direct_body_emission_roles"],
                "top_level_state_binding,top_level_state_update,top_level_branch_guard,"
                "top_level_loop_statement,loop_source_expression,loop_condition_expression,"
                "loop_body_state_transition,branch_body_state_transition,top_level_local_state_return,"
                "top_level_return_expression",
            )
        if objective_counts["local_return_closure"] > 0:
            effective["local_return_closure_loss_boost"] = max(
                effective["local_return_closure_loss_boost"],
                requested_boost,
            )
            effective["local_return_closure_roles"] = merge_role_csv(
                effective["local_return_closure_roles"],
                "previous_local_state_transition,block_exit_local_return_closure,"
                "final_return_local_name,final_return_local_expression",
            )
        if objective_counts["closed_state_transition"] > 0:
            effective["closed_state_transition_loss_boost"] = max(
                effective["closed_state_transition_loss_boost"],
                requested_boost,
            )
            effective["closed_state_transition_roles"] = merge_role_csv(
                effective["closed_state_transition_roles"],
                "closed_top_level_control_block,closed_nested_control_block,"
                "control_body_state_transition,block_exit_to_next_statement,"
                "top_level_finalizer_return",
            )

    return {
        "enabled": enabled,
        "policy": "private_semantic_ir_current_wall_obligation_weighting_plan_v1" if enabled else "not_enabled",
        "requested_loss_boost": requested_boost,
        "source_reports": {
            "semantic_ir_obligation_report": bridge_source,
            "current_wall_report": wall_source,
        },
        "issue_label_counts": issue_counts,
        "current_wall_dominant_residuals": dominant_residuals,
        "objective_match_counts": objective_counts,
        "active_objectives": active_objectives,
        "effective_overrides": effective,
        "score_semantics": (
            "Reads aggregate private failure reports and maps their labels to existing private target-side "
            "span-weighting objectives. It does not train on public benchmark artifacts, eval tests, eval "
            "solutions, generated candidate bodies, deterministic repair bodies, teacher output, hidden "
            "answer metadata, or report text as model targets. It only changes supervised weights over "
            "already-admitted private prompt/signature-to-body rows, and grants no candidate-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "uses_generated_candidate_bodies": False,
        "uses_deterministic_repair_bodies": False,
        "candidate_generation_credit": 0,
    }
