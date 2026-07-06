#!/usr/bin/env python3
"""Candidate evidence summaries for neural seed token decoders.

This module owns task-blind report summaries and no-cheat candidate evidence
helpers shared by the token comparator and strict MLX decoder. It does not
train, generate, call teachers, run public calibration, or grant promotion
credit to fallback/template/router/tool rows.
"""

from __future__ import annotations

import ast
from collections import Counter
from typing import Any

from code_lm_private_verifier import evaluate_private_candidates  # noqa: E402

TOKEN_DECODER_ARM_IDS = ("symliquid_style", "transformer_control")


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def get_path(payload: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def ratio(num: int | float, den: int | float) -> float:
    return round(float(num) / float(den), 6) if den else 0.0


def structural_action_family_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict_or_empty(dict_or_empty(config.get("body_structure_decoder")).get("structural_action_family"))


def candidate_schema_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    phases = Counter(str(row.get("phase") or "") for row in rows)
    modes = Counter(str(row.get("candidate_generation_mode") or "") for row in rows if row.get("phase") != "private_baseline")
    return {
        "candidate_rows": len(rows),
        "phase_counts": dict(sorted(phases.items())),
        "generated_mode_counts": dict(sorted(modes.items())),
        "all_rows_have_code": all(bool(row.get("code")) for row in rows),
        "token_decoded_rows": sum(1 for row in rows if row.get("candidate_generation_mode") == "token_level_code_decoder"),
        "structural_action_rows": sum(1 for row in rows if row.get("candidate_generation_mode") == "private_train_structural_action_sequence_decoder"),
        "strict_action_renderer_rows": sum(1 for row in rows if row.get("strict_action_renderer")),
        "visible_contract_semantic_beam_rows": sum(1 for row in rows if row.get("visible_contract_semantic_beam")),
        "learned_internal_semantic_route_rows": sum(1 for row in rows if row.get("learned_internal_semantic_route")),
        "grammar_repair_rows": sum(1 for row in rows if isinstance(row.get("grammar_repair"), dict)),
        "static_coherence_rows": sum(1 for row in rows if isinstance(row.get("static_coherence"), dict)),
        "body_structure_rows": sum(1 for row in rows if isinstance(row.get("body_structure_decode"), dict)),
        "internal_semantic_routing_rows": sum(1 for row in rows if get_path(row, ["internal_semantic_routing", "enabled"], False)),
        "candidate_sha256_unique": len({str(row.get("candidate_sha256") or "") for row in rows}),
    }


def no_cheat_candidate_evidence(eval_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    generated = [row for row in rows if row.get("phase") != "private_baseline"]
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    by_arm_rows: dict[str, list[dict[str, Any]]] = {"symliquid_style": [], "transformer_control": []}
    disallowed_not_promotion_eligible = True
    for row in generated:
        reasons = no_cheat_exclusion_reasons(row)
        if reasons:
            for reason in reasons:
                reason_counts[reason] += 1
            if bool(row.get("benchmark_promotion_eligible")) or bool(get_path(row, ["provenance", "model_promotion_allowed"], False)):
                disallowed_not_promotion_eligible = False
            if len(excluded) < 16:
                excluded.append(
                    {
                        "task_id": row.get("task_id"),
                        "phase": row.get("phase"),
                        "substrate_arm": row.get("substrate_arm"),
                        "candidate_generation_mode": row.get("candidate_generation_mode"),
                        "reasons": reasons,
                        "candidate_sha256": row.get("candidate_sha256"),
                    }
                )
            continue
        eligible.append(row)
        arm = str(row.get("substrate_arm") or "")
        if arm in by_arm_rows:
            by_arm_rows[arm].append(row)

    private_eval_eligible = [row for row in eligible if row.get("phase") == "private_eval"]
    all_eval = evaluate_private_candidates(eval_rows, private_eval_eligible)
    by_arm: dict[str, Any] = {}
    for arm, arm_rows in by_arm_rows.items():
        arm_private_eval = [row for row in arm_rows if row.get("phase") == "private_eval"]
        by_arm[arm] = {
            "eligible_generated_rows": len(arm_rows),
            "eligible_private_eval_rows": len(arm_private_eval),
            "private_verifier": evaluate_private_candidates(eval_rows, arm_private_eval),
        }

    summary = {
        "policy": "project_theseus_no_cheat_candidate_evidence_v0",
        "generated_rows": len(generated),
        "private_eval_generated_rows": sum(1 for row in generated if row.get("phase") == "private_eval"),
        "eligible_generated_rows": len(eligible),
        "eligible_private_eval_rows": len(private_eval_eligible),
        "excluded_generated_rows": len(generated) - len(eligible),
        "exclusion_reason_counts": dict(reason_counts.most_common()),
        "semantic_family_renderer_rows": int(reason_counts.get("semantic_family_body_renderer", 0)),
        "strict_action_fixed_renderer_rows": int(reason_counts.get("strict_action_fixed_renderer", 0)),
        "visible_contract_semantic_prior_rows": int(reason_counts.get("visible_contract_semantic_prior", 0)),
        "null_baseline_rows": int(reason_counts.get("null_baseline", 0)),
        "terminal_null_return_rows": int(reason_counts.get("terminal_null_return", 0)),
        "fallback_return_rows": int(reason_counts.get("fallback_return", 0)),
        "task_identity_keyed_rows": int(reason_counts.get("task_identity_keyed_candidate", 0)),
        "leakage_or_public_rows": sum(
            int(reason_counts.get(reason, 0))
            for reason in [
                "public_or_eval_leakage_flag",
                "body_template_selector",
                "teacher_or_external_inference",
            ]
        ),
        "promotion_eligible_rows": int(reason_counts.get("unexpected_promotion_eligible", 0)),
        "disallowed_rows_not_promotion_eligible": disallowed_not_promotion_eligible,
        "filtered_private_verifier_pass_rate": all_eval.get("trained_pass_rate"),
        "filtered_private_verifier_passed": all_eval.get("trained_passed"),
        "filtered_private_verifier_residual_count": all_eval.get("residual_count"),
        "external_inference_calls": 0,
        "teacher_used": False,
        "public_training_rows": 0,
        "model_promotion_allowed": False,
    }
    return {
        "policy": "project_theseus_no_cheat_candidate_evidence_v0",
        "summary": summary,
        "private_verifier": all_eval,
        "by_arm": by_arm,
        "excluded_examples": excluded,
        "score_semantics": (
            "No-cheat evidence excludes null baselines, fallback returns, visible-contract semantic priors, "
            "semantic family body renderers, body-template selectors, task-id keyed candidates, public/eval leakage, "
            "teacher/external-inference rows, and unexpected promotion-eligible diagnostics. The normal comparator may "
            "still carry quarantined diagnostic rows, but this filtered view is the capability evidence slice."
        ),
    }


def no_cheat_exclusion_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    structure = dict_or_empty(row.get("body_structure_decode"))
    repair = dict_or_empty(row.get("grammar_repair"))
    provenance = dict_or_empty(row.get("provenance"))
    if row.get("phase") == "private_baseline" or row.get("substrate_adapter") == "shared_null_baseline":
        reasons.append("null_baseline")
    if bool(row.get("visible_contract_semantic_beam")) or bool(structure.get("visible_contract_semantic_beam")):
        reasons.append("visible_contract_semantic_prior")
    if bool(structure.get("rendered_from_semantic_slots")) and bool(structure.get("semantic_plan_supported")):
        reasons.append("semantic_family_body_renderer")
    if bool(structure.get("rendered_from_strict_actions")) or bool(row.get("strict_action_renderer")):
        reasons.append("strict_action_fixed_renderer")
    if bool(repair.get("fallback_return_used")) or bool(structure.get("fallback_return_used")):
        reasons.append("fallback_return")
    if terminal_null_return_candidate(str(row.get("code") or "")):
        reasons.append("terminal_null_return")
    if candidate_contains_task_identity(row):
        reasons.append("task_identity_keyed_candidate")
    if bool(row.get("benchmark_promotion_eligible")):
        reasons.append("unexpected_promotion_eligible")
    if bool(provenance.get("body_template_selected")) or row.get("template_id") or row.get("template_sha256"):
        reasons.append("body_template_selector")
    if (
        bool(row.get("public_tests_visible_to_generator"))
        or bool(row.get("public_solutions_visible_to_generator"))
        or bool(row.get("eval_tests_visible_to_generator"))
        or bool(row.get("eval_solution_visible_to_generator"))
        or bool(provenance.get("tests_used_for_generation"))
        or bool(provenance.get("solutions_used_for_generation"))
    ):
        reasons.append("public_or_eval_leakage_flag")
    if int(row.get("external_inference_calls") or 0) != 0 or bool(provenance.get("teacher_used")):
        reasons.append("teacher_or_external_inference")
    return sorted(set(reasons))


def candidate_contains_task_identity(row: dict[str, Any]) -> bool:
    code = str(row.get("code") or "")
    for key in ["task_id", "source_task_id"]:
        value = str(row.get(key) or "").strip()
        if value and value in code:
            return True
    return False


def terminal_null_return_candidate(code: str) -> bool:
    try:
        parsed = ast.parse(code)
    except SyntaxError:
        return False
    function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    if function is None or len(function.body) != 1:
        return False
    stmt = function.body[0]
    return isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Constant) and stmt.value.value is None


def grammar_repair_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    repairs = [dict_or_empty(row.get("grammar_repair")) for row in rows if isinstance(row.get("grammar_repair"), dict)]
    total = len(repairs)
    raw_pass = sum(1 for row in repairs if row.get("raw_syntax_ok"))
    repaired_pass = sum(1 for row in repairs if row.get("repaired_syntax_ok"))
    changed = sum(1 for row in repairs if row.get("changed"))
    fallback = sum(1 for row in repairs if row.get("strategy") == "shape_compatible_terminal_fallback" or row.get("fallback_return_used"))
    strategies = Counter(str(row.get("strategy") or "") for row in repairs)
    failures = Counter(str(row.get("raw_failure") or "") for row in repairs if row.get("raw_failure"))
    return {
        "candidate_rows": len(rows),
        "repair_rows": total,
        "raw_syntax_pass_count": raw_pass,
        "raw_syntax_pass_rate": ratio(raw_pass, total),
        "repaired_syntax_pass_count": repaired_pass,
        "repaired_syntax_pass_rate": ratio(repaired_pass, total),
        "changed_count": changed,
        "changed_rate": ratio(changed, total),
        "fallback_count": fallback,
        "fallback_rate": ratio(fallback, total),
        "strategy_counts": dict(strategies.most_common(8)),
        "raw_failure_counts": dict(failures.most_common(8)),
    }


def static_coherence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    coherence_rows = [row for row in rows if isinstance(row.get("static_coherence"), dict)]
    records = [dict_or_empty(row.get("static_coherence")) for row in coherence_rows]
    rankers = [dict_or_empty(row.get("static_coherence_ranker")) for row in rows if isinstance(row.get("static_coherence_ranker"), dict)]
    total = len(records)

    def dependency_or_static_count(dependency_key: str, static_key: str) -> int:
        count = 0
        for source_row, static in zip(coherence_rows, records):
            dependency = dict_or_empty(get_path(source_row, ["decode_static_guard", "dependency"], {}))
            if dependency.get("parse_ok"):
                count += 1 if int(dependency.get(dependency_key) or 0) > 0 else 0
            else:
                count += 1 if int(static.get(static_key) or 0) > 0 else 0
        return count

    parse_ok = sum(1 for row in records if row.get("parse_ok") and row.get("has_function"))
    has_return = sum(1 for row in records if row.get("has_return"))
    valued_return = dependency_or_static_count("valued_return_count", "valued_return_count")
    trivial_return = sum(1 for row in records if int(row.get("trivial_return_count") or 0) > 0)
    nontrivial_return = dependency_or_static_count("nontrivial_return_count", "nontrivial_return_count")
    top_level_valued_return = dependency_or_static_count("top_level_valued_return_count", "top_level_valued_return_count")
    top_level_nontrivial_return = dependency_or_static_count("top_level_nontrivial_return_count", "nontrivial_return_count")
    nested_return = sum(1 for row in records if int(row.get("nested_return_count") or 0) > 0)
    parameter_use = sum(1 for row in records if int(row.get("used_parameter_count") or 0) > 0)
    primary_parameter_use = sum(1 for row in records if int(row.get("primary_parameter_load_count") or 0) > 0)
    primary_parameter_return = sum(1 for row in records if int(row.get("primary_parameter_return_count") or 0) > 0)
    aux_only_return = sum(1 for row in records if int(row.get("return_only_uses_auxiliary_parameter_count") or 0) > 0)
    self_dependent_assignment = sum(1 for row in records if int(row.get("self_dependent_assignment_count") or 0) > 0)
    repeated_condition_chain = sum(
        1 for row in records if int(row.get("max_repeated_identical_condition_chain") or 0) >= 4
    )
    inert_stub = sum(1 for row in records if bool(row.get("inert_stub", False)))
    clean_names = sum(1 for row in records if int(row.get("undefined_name_count") or 0) == 0)
    clean_signature = sum(1 for row in records if int(row.get("unexpected_signature_name_count") or 0) == 0)
    invalid_receiver = sum(1 for row in records if int(row.get("invalid_receiver_count") or 0) > 0)
    builtin_type_descriptor_receiver = sum(
        1 for row in records if int(row.get("builtin_type_descriptor_receiver_count") or 0) > 0
    )
    bare_builtin_condition = sum(1 for row in records if int(row.get("bare_builtin_condition_count") or 0) > 0)
    invalid_known_builtin_arity = sum(1 for row in records if int(row.get("invalid_known_builtin_arity_count") or 0) > 0)
    invalid_known_local_receiver = sum(
        1 for row in records if int(row.get("invalid_known_local_receiver_count") or 0) > 0
    )
    invalid_known_local_call = sum(1 for row in records if int(row.get("invalid_known_local_call_count") or 0) > 0)
    invalid_known_local_iter = sum(1 for row in records if int(row.get("invalid_known_local_iter_count") or 0) > 0)
    invalid_multi_assign = sum(1 for row in records if int(row.get("invalid_multi_assign_from_scalar_count") or 0) > 0)
    mutating_method_return_value = sum(
        1 for row in records if int(row.get("mutating_method_return_value_count") or 0) > 0
    )
    ignored_pure_call_expression = sum(
        1 for row in records if int(row.get("ignored_pure_call_expression_count") or 0) > 0
    )
    parameter_free_literal_return = sum(
        1 for row in records if int(row.get("parameter_free_literal_expression_return_count") or 0) > 0
    )
    changed = sum(1 for row in rankers if row.get("rank_changed"))
    undefined_names: Counter[str] = Counter()
    unexpected_names: Counter[str] = Counter()
    for row in records:
        undefined_names.update(str(name) for name in row.get("undefined_names", []) if str(name))
        unexpected_names.update(str(name) for name in row.get("unexpected_signature_names", []) if str(name))
    return {
        "candidate_rows": len(rows),
        "static_coherence_rows": total,
        "policy": "prompt_signature_static_coherence_v1",
        "parse_function_ok_count": parse_ok,
        "parse_function_ok_rate": ratio(parse_ok, total),
        "has_return_count": has_return,
        "has_return_rate": ratio(has_return, total),
        "valued_return_count": valued_return,
        "valued_return_rate": ratio(valued_return, total),
        "trivial_return_count": trivial_return,
        "trivial_return_rate": ratio(trivial_return, total),
        "nontrivial_return_count": nontrivial_return,
        "nontrivial_return_rate": ratio(nontrivial_return, total),
        "top_level_valued_return_count": top_level_valued_return,
        "top_level_valued_return_rate": ratio(top_level_valued_return, total),
        "top_level_nontrivial_return_count": top_level_nontrivial_return,
        "top_level_nontrivial_return_rate": ratio(top_level_nontrivial_return, total),
        "return_dependency_summary_policy": "strict_decode_dependency_when_available_else_static_coherence_v1",
        "nested_return_count": nested_return,
        "nested_return_rate": ratio(nested_return, total),
        "parameter_use_count": parameter_use,
        "parameter_use_rate": ratio(parameter_use, total),
        "primary_parameter_use_count": primary_parameter_use,
        "primary_parameter_use_rate": ratio(primary_parameter_use, total),
        "primary_parameter_return_count": primary_parameter_return,
        "primary_parameter_return_rate": ratio(primary_parameter_return, total),
        "auxiliary_only_return_count": aux_only_return,
        "auxiliary_only_return_rate": ratio(aux_only_return, total),
        "self_dependent_assignment_count": self_dependent_assignment,
        "self_dependent_assignment_rate": ratio(self_dependent_assignment, total),
        "repeated_identical_condition_chain_count": repeated_condition_chain,
        "repeated_identical_condition_chain_rate": ratio(repeated_condition_chain, total),
        "inert_stub_count": inert_stub,
        "inert_stub_rate": ratio(inert_stub, total),
        "undefined_clean_count": clean_names,
        "undefined_clean_rate": ratio(clean_names, total),
        "signature_clean_count": clean_signature,
        "signature_clean_rate": ratio(clean_signature, total),
        "invalid_receiver_count": invalid_receiver,
        "invalid_receiver_rate": ratio(invalid_receiver, total),
        "builtin_type_descriptor_receiver_count": builtin_type_descriptor_receiver,
        "builtin_type_descriptor_receiver_rate": ratio(builtin_type_descriptor_receiver, total),
        "bare_builtin_condition_count": bare_builtin_condition,
        "bare_builtin_condition_rate": ratio(bare_builtin_condition, total),
        "invalid_known_builtin_arity_count": invalid_known_builtin_arity,
        "invalid_known_builtin_arity_rate": ratio(invalid_known_builtin_arity, total),
        "invalid_known_local_receiver_count": invalid_known_local_receiver,
        "invalid_known_local_receiver_rate": ratio(invalid_known_local_receiver, total),
        "invalid_known_local_call_count": invalid_known_local_call,
        "invalid_known_local_call_rate": ratio(invalid_known_local_call, total),
        "invalid_known_local_iter_count": invalid_known_local_iter,
        "invalid_known_local_iter_rate": ratio(invalid_known_local_iter, total),
        "invalid_multi_assign_from_scalar_count": invalid_multi_assign,
        "invalid_multi_assign_from_scalar_rate": ratio(invalid_multi_assign, total),
        "mutating_method_return_value_count": mutating_method_return_value,
        "mutating_method_return_value_rate": ratio(mutating_method_return_value, total),
        "ignored_pure_call_expression_count": ignored_pure_call_expression,
        "ignored_pure_call_expression_rate": ratio(ignored_pure_call_expression, total),
        "parameter_free_literal_expression_return_count": parameter_free_literal_return,
        "parameter_free_literal_expression_return_rate": ratio(parameter_free_literal_return, total),
        "rank_changed_count": changed,
        "rank_changed_rate": ratio(changed, len(rankers)),
        "undefined_name_counts": dict(undefined_names.most_common(12)),
        "unexpected_signature_name_counts": dict(unexpected_names.most_common(12)),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
    }


def body_structure_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    structures = [dict_or_empty(row.get("body_structure_decode")) for row in rows if isinstance(row.get("body_structure_decode"), dict)]
    total = len(structures)
    rendered = sum(1 for row in structures if row.get("rendered_from_statement_skeleton"))
    semantic_rendered = sum(1 for row in structures if row.get("rendered_from_semantic_slots"))
    structural_rendered = sum(1 for row in structures if row.get("rendered_from_structural_actions"))
    strict_action_rendered = sum(1 for row in structures if row.get("rendered_from_strict_actions"))
    semantic_supported = sum(1 for row in structures if row.get("semantic_plan_supported"))
    contract_beams = sum(1 for row in structures if row.get("visible_contract_semantic_beam"))
    learned_routes = sum(1 for row in structures if row.get("learned_internal_semantic_route"))
    return_shapes = sum(1 for row in structures if row.get("predicted_return_shape"))
    loops = sum(1 for row in structures if row.get("predicted_loop"))
    branches = sum(1 for row in structures if row.get("predicted_branch"))
    fallback_returns = sum(1 for row in structures if row.get("fallback_return_used"))
    plans = Counter(str(row.get("semantic_plan") or "") for row in structures if row.get("semantic_plan"))
    families = Counter(str(row.get("task_family") or "unknown") for row in structures)
    return {
        "candidate_rows": len(rows),
        "body_structure_rows": total,
        "statement_skeleton_render_count": rendered,
        "statement_skeleton_render_rate": ratio(rendered, total),
        "semantic_slot_render_count": semantic_rendered,
        "semantic_slot_render_rate": ratio(semantic_rendered, total),
        "structural_action_render_count": structural_rendered,
        "structural_action_render_rate": ratio(structural_rendered, total),
        "strict_action_render_count": strict_action_rendered,
        "strict_action_render_rate": ratio(strict_action_rendered, total),
        "semantic_plan_supported_count": semantic_supported,
        "semantic_plan_supported_rate": ratio(semantic_supported, total),
        "visible_contract_semantic_beam_count": contract_beams,
        "visible_contract_semantic_beam_rate": ratio(contract_beams, total),
        "learned_internal_semantic_route_count": learned_routes,
        "learned_internal_semantic_route_rate": ratio(learned_routes, total),
        "predicted_return_shape_count": return_shapes,
        "predicted_return_shape_rate": ratio(return_shapes, total),
        "predicted_loop_count": loops,
        "predicted_branch_count": branches,
        "fallback_return_used_count": fallback_returns,
        "fallback_return_used_rate": ratio(fallback_returns, total),
        "semantic_plan_counts": dict(plans.most_common(12)),
        "task_family_counts": dict(families.most_common(12)),
    }


def grammar_repair_metadata_recorded(rows: list[dict[str, Any]]) -> bool:
    generated = [row for row in rows if row.get("phase") != "private_baseline"]
    return bool(generated) and all(isinstance(row.get("grammar_repair"), dict) for row in generated)


def raw_syntax_measured(rows: list[dict[str, Any]]) -> bool:
    generated = [row for row in rows if row.get("phase") != "private_baseline"]
    return bool(generated) and all("raw_syntax_ok" in dict_or_empty(row.get("grammar_repair")) for row in generated)


def grammar_repair_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    generated = [row for row in rows if row.get("phase") != "private_baseline"]
    repairs = [dict_or_empty(row.get("grammar_repair")) for row in generated if isinstance(row.get("grammar_repair"), dict)]
    strategies = Counter(str(row.get("strategy") or "") for row in repairs)
    return {
        "generated_rows": len(generated),
        "repair_rows": len(repairs),
        "fallback_return_used_rows": sum(1 for row in repairs if row.get("fallback_return_used")),
        "strategy_counts": dict(strategies.most_common(8)),
    }


def body_structure_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    generated = [row for row in rows if row.get("phase") != "private_baseline"]
    structures = [dict_or_empty(row.get("body_structure_decode")) for row in generated if isinstance(row.get("body_structure_decode"), dict)]
    return {
        "generated_rows": len(generated),
        "body_structure_rows": len(structures),
        "statement_skeleton_rows": sum(1 for row in structures if row.get("rendered_from_statement_skeleton")),
        "semantic_slot_rows": sum(1 for row in structures if row.get("rendered_from_semantic_slots")),
        "structural_action_rows": sum(1 for row in structures if row.get("rendered_from_structural_actions")),
        "strict_action_rows": sum(1 for row in structures if row.get("rendered_from_strict_actions")),
        "semantic_plan_supported_rows": sum(1 for row in structures if row.get("semantic_plan_supported")),
        "visible_contract_semantic_beam_rows": sum(1 for row in structures if row.get("visible_contract_semantic_beam")),
        "fallback_return_used_rows": sum(1 for row in structures if row.get("fallback_return_used")),
    }


def post_repair_syntax_pass_nonzero(arm_reports: dict[str, Any], required_arms: list[str] | None = None) -> bool:
    for arm_id in required_arms or list(TOKEN_DECODER_ARM_IDS):
        if float(get_path(arm_reports, [arm_id, "summary", "syntax_pass_rate_sts_on"], 0.0) or 0.0) <= 0.0:
            return False
    return True


def raw_syntax_pass_nonzero(arm_reports: dict[str, Any], required_arms: list[str] | None = None) -> bool:
    for arm_id in required_arms or list(TOKEN_DECODER_ARM_IDS):
        if float(get_path(arm_reports, [arm_id, "summary", "raw_syntax_pass_rate_sts_on"], 0.0) or 0.0) <= 0.0:
            return False
    return True


def fallback_return_rate_zero(arm_reports: dict[str, Any], required_arms: list[str] | None = None) -> bool:
    for arm_id in required_arms or list(TOKEN_DECODER_ARM_IDS):
        if float(get_path(arm_reports, [arm_id, "summary", "grammar_repair_fallback_rate_sts_on"], 0.0) or 0.0) != 0.0:
            return False
        if int(get_path(arm_reports, [arm_id, "summary", "structural_action_candidate_rows"], 0) or 0) > 0:
            if float(get_path(arm_reports, [arm_id, "summary", "structural_action_fallback_rate"], 0.0) or 0.0) != 0.0:
                return False
    return True


def body_decode_metadata_recorded(rows: list[dict[str, Any]]) -> bool:
    generated = [row for row in rows if row.get("phase") != "private_baseline"]
    return bool(generated) and all(isinstance(row.get("body_structure_decode"), dict) for row in generated)


def allowed_generated_candidate_modes(config: dict[str, Any]) -> set[str]:
    modes = {"token_level_code_decoder"}
    if bool(structural_action_family_config(config).get("enabled", False)):
        modes.add("private_train_structural_action_sequence_decoder")
    return modes


def candidate_generation_modes_allowed(config: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    allowed = allowed_generated_candidate_modes(config)
    return all(
        str(row.get("candidate_generation_mode") or "") in allowed or str(row.get("phase")) == "private_baseline"
        for row in rows
    )


def syntax_evidence(arm_reports: dict[str, Any]) -> dict[str, Any]:
    return {
        arm_id: {
            "sts_on": get_path(report, ["summary", "syntax_pass_rate_sts_on"], 0.0),
            "sts_off": get_path(report, ["summary", "syntax_pass_rate_sts_off"], 0.0),
            "raw_sts_on": get_path(report, ["summary", "raw_syntax_pass_rate_sts_on"], 0.0),
            "fallback_sts_on": get_path(report, ["summary", "grammar_repair_fallback_rate_sts_on"], 0.0),
            "statement_skeleton_render_sts_on": get_path(report, ["summary", "statement_skeleton_render_rate_sts_on"], 0.0),
            "semantic_slot_render_sts_on": get_path(report, ["summary", "semantic_slot_render_rate_sts_on"], 0.0),
            "strict_action_render_sts_on": get_path(report, ["summary", "strict_action_render_rate_sts_on"], 0.0),
            "structural_action_render_rate": get_path(report, ["summary", "structural_action_render_rate"], 0.0),
            "structural_action_syntax_pass_rate": get_path(report, ["summary", "structural_action_syntax_pass_rate"], 0.0),
            "structural_action_fallback_rate": get_path(report, ["summary", "structural_action_fallback_rate"], 0.0),
            "semantic_plan_supported_sts_on": get_path(report, ["summary", "semantic_plan_supported_rate_sts_on"], 0.0),
            "learned_internal_route_sts_on": get_path(report, ["summary", "learned_internal_semantic_route_rate_sts_on"], 0.0),
        }
        for arm_id, report in arm_reports.items()
    }


def both_arms_emit_token_code(arm_reports: dict[str, Any], required_arms: list[str] | None = None) -> bool:
    for arm_id in required_arms or list(TOKEN_DECODER_ARM_IDS):
        if int(get_path(arm_reports, [arm_id, "candidate_schema", "token_decoded_rows"], 0) or 0) <= 0:
            return False
        if not bool(get_path(arm_reports, [arm_id, "candidate_schema", "all_rows_have_code"], False)):
            return False
    return True
