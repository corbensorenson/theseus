#!/usr/bin/env python3
"""Private strict-generator target weighting and span extraction helpers.

These helpers build supervised private-training token weights and AST-derived
span metadata for the strict MLX adaptation path. They use admitted private
solution bodies as training targets only. They do not inspect eval tests,
public benchmark payloads, teacher output, or runtime user prompts, and they do
not grant candidate-generation credit.
"""

from __future__ import annotations

import ast
from typing import Any

from neural_seed_code_proposer_comparator import dict_or_empty
from neural_seed_token_decoder_support import learned_body_decision_prefix_tokens_for_body, semantic_plan_from_body, target_tokens
from strict_generator_mlx_decode_plans import (
    source_condition_adequacy_for_body,
    source_condition_expectation_from_source_text,
)


def apply_return_expression_weight_override(budget: dict[str, Any], *, boost: float) -> dict[str, Any]:
    value = max(0.0, float(boost if boost is not None else 0.0))
    if value <= 0.0:
        return {
            "enabled": False,
            "policy": "not_enabled",
            "return_expression_loss_boost": value,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }
    semantic_cfg = dict_or_empty(budget.get("semantic_token_loss_weights"))
    max_weight = max(value, float(semantic_cfg.get("max_weight") or 4.0))
    semantic_cfg.update(
        {
            "enabled": True,
            "policy": "private_return_expression_loss_weight_override_v1",
            "structural_trajectory_weights_enabled": True,
            "top_level_return_weight": value,
            "final_dedent_weight": max(float(semantic_cfg.get("final_dedent_weight") or 1.8), min(value, max_weight)),
            "max_weight": max_weight,
        }
    )
    budget["semantic_token_loss_weights"] = semantic_cfg
    budget["return_token_loss_weight"] = max(value, float(budget.get("return_token_loss_weight") or 1.0))
    return {
        "enabled": True,
        "policy": "private_return_expression_loss_weight_override_v1",
        "return_expression_loss_boost": value,
        "return_token_loss_weight": budget["return_token_loss_weight"],
        "top_level_return_weight": semantic_cfg["top_level_return_weight"],
        "final_dedent_weight": semantic_cfg["final_dedent_weight"],
        "max_weight": semantic_cfg["max_weight"],
        "score_semantics": (
            "Raises supervised CE weight on admitted private target return tokens and top-level return "
            "lines. It uses private training solution bodies only as training targets; generation still "
            "sees only prompt/signature source text. It does not inspect eval tests/solutions, use public "
            "data, synthesize code, or grant candidate-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def apply_default_parameter_return_weights(
    token_weight_rows: list[list[float]],
    target_rows: list[list[int]],
    bodies: list[str],
    source_texts: list[str],
    *,
    target_vocab: dict[str, int],
    boost: float,
) -> tuple[list[list[float]], dict[str, Any]]:
    value = max(0.0, float(boost if boost is not None else 0.0))
    if value <= 0.0:
        return token_weight_rows, {
            "enabled": False,
            "policy": "not_enabled",
            "default_parameter_return_loss_boost": value,
            "matched_rows": 0,
            "weighted_token_positions": 0,
            "candidate_generation_credit": 0,
        }
    return_id = target_vocab.get("NAME:return")
    if return_id is None:
        return token_weight_rows, {
            "enabled": True,
            "policy": "private_visible_default_parameter_return_weighting_v1",
            "default_parameter_return_loss_boost": value,
            "matched_rows": 0,
            "weighted_token_positions": 0,
            "reason": "target_vocab_missing_NAME_return",
            "candidate_generation_credit": 0,
        }

    adjusted = [list(row) for row in token_weight_rows]
    matched_rows = 0
    weighted_positions = 0
    candidate_arg_counts: dict[str, int] = {}
    matched_arg_counts: dict[str, int] = {}
    for index, target in enumerate(target_rows):
        source_text = source_texts[index] if index < len(source_texts) else ""
        body = bodies[index] if index < len(bodies) else ""
        arg_names = visible_default_argument_candidates(source_text)
        for arg in arg_names:
            candidate_arg_counts[arg] = candidate_arg_counts.get(arg, 0) + 1
        if not arg_names:
            continue
        body_lines = [line.strip() for line in str(body or "").splitlines()]
        matched_args = [arg for arg in arg_names if f"return {arg}" in body_lines]
        if not matched_args:
            continue
        matched_rows += 1
        for arg in matched_args:
            matched_arg_counts[arg] = matched_arg_counts.get(arg, 0) + 1
            arg_id = target_vocab.get(f"NAME:{arg}")
            if arg_id is None:
                continue
            for pos in range(0, max(0, len(target) - 1)):
                if int(target[pos]) == int(return_id) and int(target[pos + 1]) == int(arg_id):
                    adjusted[index][pos] = max(float(adjusted[index][pos]), value)
                    adjusted[index][pos + 1] = max(float(adjusted[index][pos + 1]), value)
                    weighted_positions += 2
    return adjusted, {
        "enabled": True,
        "policy": "private_visible_default_parameter_return_weighting_v1",
        "default_parameter_return_loss_boost": value,
        "matched_rows": matched_rows,
        "weighted_token_positions": weighted_positions,
        "candidate_argument_counts": dict(sorted(candidate_arg_counts.items())),
        "matched_argument_counts": dict(sorted(matched_arg_counts.items())),
        "score_semantics": (
            "Boosts supervised CE weight only for admitted private target lines that return a visible "
            "default argument when the prompt/signature source text itself contains default-like intent. "
            "It uses no eval tests, eval solutions, public data, teacher output, candidate templates, "
            "or runtime fallback returns, and grants no candidate-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def visible_default_argument_candidates(source_text: str) -> list[str]:
    text = str(source_text or "")
    lower = text.lower()
    if "default" not in lower and "empty" not in lower:
        return []
    args: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("arguments "):
            args = [part for part in stripped.split()[1:] if part.isidentifier()]
            break
    if len(args) <= 1:
        return []
    defaultish = {"default", "fallback", "other", "otherwise", "missing", "empty"}
    named = [arg for arg in args[1:] if any(token in arg.lower() for token in defaultish)]
    if named:
        return named
    return args[1:]


def apply_truthiness_guard_weights(
    token_weight_rows: list[list[float]],
    target_rows: list[list[int]],
    bodies: list[str],
    source_texts: list[str],
    *,
    target_vocab: dict[str, int],
    boost: float,
) -> tuple[list[list[float]], dict[str, Any]]:
    value = max(0.0, float(boost if boost is not None else 0.0))
    if value <= 0.0:
        return token_weight_rows, {
            "enabled": False,
            "policy": "not_enabled",
            "truthiness_guard_loss_boost": value,
            "matched_rows": 0,
            "weighted_token_positions": 0,
            "candidate_generation_credit": 0,
        }
    and_id = target_vocab.get("NAME:and")
    if and_id is None:
        return token_weight_rows, {
            "enabled": True,
            "policy": "private_visible_empty_input_truthiness_guard_weighting_v1",
            "truthiness_guard_loss_boost": value,
            "matched_rows": 0,
            "weighted_token_positions": 0,
            "reason": "target_vocab_missing_NAME_and",
            "candidate_generation_credit": 0,
        }
    adjusted = [list(row) for row in token_weight_rows]
    matched_rows = 0
    weighted_positions = 0
    candidate_arg_counts: dict[str, int] = {}
    matched_arg_counts: dict[str, int] = {}
    for index, target in enumerate(target_rows):
        source_text = source_texts[index] if index < len(source_texts) else ""
        body = bodies[index] if index < len(bodies) else ""
        arg_names = visible_truthiness_argument_candidates(source_text)
        for arg in arg_names:
            candidate_arg_counts[arg] = candidate_arg_counts.get(arg, 0) + 1
        if not arg_names:
            continue
        body_text = str(body or "")
        matched_args = [arg for arg in arg_names if f"and {arg}" in body_text or f"if {arg}" in body_text]
        if not matched_args:
            continue
        matched_rows += 1
        for arg in matched_args:
            matched_arg_counts[arg] = matched_arg_counts.get(arg, 0) + 1
            arg_id = target_vocab.get(f"NAME:{arg}")
            if arg_id is None:
                continue
            for pos in range(0, max(0, len(target) - 1)):
                if int(target[pos]) == int(and_id) and int(target[pos + 1]) == int(arg_id):
                    adjusted[index][pos] = max(float(adjusted[index][pos]), value)
                    adjusted[index][pos + 1] = max(float(adjusted[index][pos + 1]), value)
                    weighted_positions += 2
    return adjusted, {
        "enabled": True,
        "policy": "private_visible_empty_input_truthiness_guard_weighting_v1",
        "truthiness_guard_loss_boost": value,
        "matched_rows": matched_rows,
        "weighted_token_positions": weighted_positions,
        "candidate_argument_counts": dict(sorted(candidate_arg_counts.items())),
        "matched_argument_counts": dict(sorted(matched_arg_counts.items())),
        "score_semantics": (
            "Boosts supervised CE weight only for admitted private target truthiness guards such as "
            "`and data` when the prompt/signature source text visibly mentions empty/default handling. "
            "It does not inspect eval tests, eval solutions, public data, teacher output, or candidate "
            "templates, and grants no candidate-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def visible_truthiness_argument_candidates(source_text: str) -> list[str]:
    text = str(source_text or "")
    lower = text.lower()
    if "empty" not in lower and "non-empty" not in lower and "nonempty" not in lower:
        return []
    args: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("arguments "):
            args = [part for part in stripped.split()[1:] if part.isidentifier()]
            break
    return args[:1]


def apply_source_condition_internalization_weights(
    token_weight_rows: list[list[float]],
    target_rows: list[list[int]],
    bodies: list[str],
    source_texts: list[str],
    *,
    target_vocab: dict[str, int],
    boost: float,
) -> tuple[list[list[float]], dict[str, Any]]:
    value = max(0.0, float(boost if boost is not None else 0.0))
    if value <= 0.0:
        return token_weight_rows, {
            "enabled": False,
            "policy": "not_enabled",
            "source_condition_internalization_loss_boost": value,
            "matched_rows": 0,
            "weighted_token_positions": 0,
            "candidate_generation_credit": 0,
        }
    adjusted = [list(row) for row in token_weight_rows]
    inverse = {int(token_id): str(token) for token, token_id in target_vocab.items()}
    matched_rows = 0
    adequate_rows = 0
    weighted_positions = 0
    operation_weighted_positions = 0
    missing_feature_counts: dict[str, int] = {}
    operation_tag_counts: dict[str, int] = {}
    operation_hit_tag_counts: dict[str, int] = {}
    token_counts: dict[str, int] = {}
    skipped: dict[str, int] = {}
    for index, target in enumerate(target_rows):
        source_text = source_texts[index] if index < len(source_texts) else ""
        body = bodies[index] if index < len(bodies) else ""
        expectation = source_condition_expectation_from_source_text(source_text)
        if not bool(expectation.get("enabled")):
            continue
        matched_rows += 1
        adequacy = source_condition_adequacy_for_body(body, expectation, allowed_names=None)
        if bool(adequacy.get("adequate")):
            adequate_rows += 1
        else:
            for feature in list(adequacy.get("missing_features") or []):
                name = str(feature)
                missing_feature_counts[name] = missing_feature_counts.get(name, 0) + 1
            continue
        operation_evidence = dict_or_empty(adequacy.get("operation_evidence"))
        operation_tags = {str(tag) for tag in list(expectation.get("operation_tags") or []) if str(tag)}
        hit_operation_tags = {str(tag) for tag in list(operation_evidence.get("hit_operation_tags") or []) if str(tag)}
        for tag in operation_tags:
            operation_tag_counts[tag] = operation_tag_counts.get(tag, 0) + 1
        for tag in hit_operation_tags:
            operation_hit_tag_counts[tag] = operation_hit_tag_counts.get(tag, 0) + 1
        expected_arg_names = {
            str(expectation.get("truthiness_arg") or ""),
            str(expectation.get("default_arg") or ""),
        }
        relevant_token_texts: set[str] = set()
        if (
            bool(expectation.get("requires_truthiness_guard"))
            or bool(expectation.get("requires_default_return"))
            or bool(expectation.get("requires_sequence_type_guard"))
            or bool(expectation.get("requires_first_item_return"))
        ):
            relevant_token_texts.update(
                {
                    "NAME:if",
                    "NAME:isinstance",
                    "NAME:list",
                    "NAME:tuple",
                    "NAME:and",
                    "NAME:return",
                    "OP:(",
                    "OP:)",
                    "OP:,",
                    "OP:[",
                    "OP:]",
                    "OP::",
                    "INDENT:",
                    "DEDENT:",
                    "NEWLINE:",
                    "NUMBER:0",
                }
            )
            for arg in expected_arg_names:
                if arg:
                    relevant_token_texts.add(f"NAME:{arg}")
        operation_token_texts = source_condition_operation_weight_token_texts(hit_operation_tags or operation_tags)
        relevant_token_texts.update(operation_token_texts)
        if not relevant_token_texts:
            skipped["no_relevant_source_condition_tokens"] = skipped.get("no_relevant_source_condition_tokens", 0) + 1
            continue
        for pos, token_id in enumerate(target):
            token_text = inverse.get(int(token_id), "")
            if token_text not in relevant_token_texts:
                continue
            adjusted[index][pos] = max(float(adjusted[index][pos]), value)
            weighted_positions += 1
            if token_text in operation_token_texts:
                operation_weighted_positions += 1
            token_counts[token_text] = token_counts.get(token_text, 0) + 1
    return adjusted, {
        "enabled": True,
        "policy": "private_prompt_visible_source_condition_internalization_weighting_v1",
        "source_condition_internalization_loss_boost": value,
        "matched_rows": matched_rows,
        "adequate_target_rows": adequate_rows,
        "weighted_token_positions": weighted_positions,
        "operation_weighted_token_positions": operation_weighted_positions,
        "weighted_token_counts": dict(sorted(token_counts.items())),
        "operation_tag_counts": dict(sorted(operation_tag_counts.items())),
        "operation_hit_tag_counts": dict(sorted(operation_hit_tag_counts.items())),
        "missing_feature_counts": dict(sorted(missing_feature_counts.items())),
        "skipped_counts": dict(sorted(skipped.items())),
        "score_semantics": (
            "Boosts supervised CE weight on admitted private target tokens that implement prompt-visible "
            "source-condition contracts already present in the source text, including empty/default guards "
            "and broad operation tags such as clamp, threshold filtering, tolerance-window filtering, round, "
            "and numeric summary. It uses the private solution body only as the admitted training target, "
            "does not inspect eval tests/solutions, public benchmark payloads, verifier labels, teacher "
            "output, answer metadata, or candidate templates, and grants no candidate-generation credit. "
            "Success must be evaluated separately with this constraint off."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def source_condition_operation_weight_token_texts(operation_tags: set[str]) -> set[str]:
    tokens: set[str] = set()
    if "op_gcd_reduce" in operation_tags:
        tokens.update({"NAME:gcd", "NAME:math", "NAME:abs", "OP:.", "OP:(", "OP:)"})
    if "op_abs_positive_filter" in operation_tags:
        tokens.update({"NAME:abs", "OP:>", "OP:<", "OP:>=", "OP:<=", "NUMBER:0"})
    if "op_abs_tolerance_filter" in operation_tags:
        tokens.update({"NAME:abs", "NAME:float", "OP:-", "OP:+", "OP:<=", "OP:(", "OP:)", "NUMBER:0"})
    if "op_threshold_filter" in operation_tags:
        tokens.update({"NAME:if", "OP::", "OP:>", "OP:>=", "OP:<", "OP:<=", "NUMBER:0"})
    if "op_windowed_delta" in operation_tags:
        tokens.update({"NAME:abs", "NAME:min", "NAME:max", "NAME:range", "NAME:len", "OP:-", "OP:+", "OP:<", "OP:>"})
    if "op_clip_to_range" in operation_tags:
        tokens.update({"NAME:min", "NAME:max", "OP:<", "OP:>", "OP:<=", "OP:>="})
    if "op_round_values" in operation_tags:
        tokens.add("NAME:round")
    if "op_numeric_summary" in operation_tags:
        tokens.update({"NAME:sum", "NAME:abs", "NAME:min", "NAME:max", "NAME:round", "OP:+", "OP:-", "OP:*", "OP:/"})
    return tokens


def apply_loop_operation_weights(
    token_weight_rows: list[list[float]],
    target_rows: list[list[int]],
    bodies: list[str],
    *,
    target_vocab: dict[str, int],
    boost: float,
) -> tuple[list[list[float]], dict[str, Any]]:
    value = max(0.0, float(boost if boost is not None else 0.0))
    if value <= 0.0:
        return token_weight_rows, {
            "enabled": False,
            "policy": "not_enabled",
            "loop_operation_loss_boost": value,
            "matched_rows": 0,
            "weighted_token_positions": 0,
            "candidate_generation_credit": 0,
        }
    inverse = {int(token_id): str(token) for token, token_id in target_vocab.items()}
    adjusted = [list(row) for row in token_weight_rows]
    matched_rows = 0
    weighted_positions = 0
    token_counts: dict[str, int] = {}
    operation_counts: dict[str, int] = {}
    skipped: dict[str, int] = {}
    for index, target in enumerate(target_rows):
        body = bodies[index] if index < len(bodies) else ""
        extraction = loop_operation_tokens_for_body(body)
        if not bool(extraction.get("matched")):
            reason = str(extraction.get("reason") or "no_loop_operation_tokens")
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        matched_rows += 1
        for operation, count in dict_or_empty(extraction.get("operation_counts")).items():
            key = str(operation)
            operation_counts[key] = operation_counts.get(key, 0) + int(count or 0)
        relevant = {str(token) for token in list(extraction.get("tokens") or [])}
        for pos, token_id in enumerate(target):
            token_text = inverse.get(int(token_id), "")
            if token_text not in relevant:
                continue
            adjusted[index][pos] = max(float(adjusted[index][pos]), value)
            weighted_positions += 1
            token_counts[token_text] = token_counts.get(token_text, 0) + 1
    return adjusted, {
        "enabled": True,
        "policy": "private_loop_operation_body_loss_weighting_v1",
        "loop_operation_loss_boost": value,
        "rows": len(token_weight_rows),
        "matched_rows": matched_rows,
        "weighted_token_positions": weighted_positions,
        "weighted_token_counts": dict(sorted(token_counts.items())),
        "operation_counts": dict(sorted(operation_counts.items())),
        "skipped_counts": dict(sorted(skipped.items())),
        "score_semantics": (
            "Boosts supervised CE weight on admitted private target tokens that implement AST-visible "
            "loop, branch, update, comparison, and finalizer operations. It uses private solution bodies "
            "only as already-admitted training targets; generation still sees strict prompt/signature "
            "source text. It does not render code, inspect eval tests/solutions, use public data, use "
            "teacher output, or grant learned-generation candidate credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def apply_loop_statement_action_weights(
    token_weight_rows: list[list[float]],
    target_rows: list[list[int]],
    bodies: list[str],
    *,
    target_vocab: dict[str, int],
    boost: float,
    roles: str,
) -> tuple[list[list[float]], dict[str, Any]]:
    value = max(0.0, float(boost if boost is not None else 0.0))
    allowed_roles = parse_role_filter(roles)
    if value <= 0.0:
        return token_weight_rows, {
            "enabled": False,
            "policy": "not_enabled",
            "loop_statement_action_loss_boost": value,
            "role_filter": sorted(allowed_roles),
            "matched_rows": 0,
            "weighted_token_positions": 0,
            "candidate_generation_credit": 0,
        }
    inverse = {int(token_id): str(token) for token, token_id in target_vocab.items()}
    adjusted = [list(row) for row in token_weight_rows]
    matched_rows = 0
    weighted_positions = 0
    excluded_positions = 0
    role_counts: dict[str, int] = {}
    token_counts: dict[str, int] = {}
    excluded_token_counts: dict[str, int] = {}
    skipped: dict[str, int] = {}
    for index, target in enumerate(target_rows):
        body = bodies[index] if index < len(bodies) else ""
        extraction = loop_statement_action_spans_for_body(body)
        spans = list(extraction.get("spans") or [])
        if not bool(extraction.get("matched")) or not spans:
            reason = str(extraction.get("reason") or "no_loop_statement_action_spans")
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        target_texts = [inverse.get(int(token_id), "") for token_id in target]
        row_matched = False
        for span in spans:
            item = dict_or_empty(span)
            role = str(item.get("role") or "unknown")
            if allowed_roles and role not in allowed_roles:
                skipped[f"role_filtered_{role}"] = skipped.get(f"role_filtered_{role}", 0) + 1
                continue
            span_tokens = [str(token) for token in list(item.get("tokens") or []) if str(token)]
            excluded_positive_tokens = {str(token) for token in list(item.get("excluded_positive_tokens") or [])}
            if not span_tokens:
                skipped["empty_span_tokens"] = skipped.get("empty_span_tokens", 0) + 1
                continue
            matches = find_subsequence_positions(target_texts, span_tokens)
            if not matches:
                skipped[f"span_not_found_{role}"] = skipped.get(f"span_not_found_{role}", 0) + 1
                continue
            role_counts[role] = role_counts.get(role, 0) + len(matches)
            row_matched = True
            for start in matches:
                for offset, token_text in enumerate(span_tokens):
                    if token_text in excluded_positive_tokens:
                        excluded_positions += 1
                        excluded_token_counts[token_text] = excluded_token_counts.get(token_text, 0) + 1
                        continue
                    pos = start + offset
                    if pos >= len(adjusted[index]):
                        continue
                    adjusted[index][pos] = max(float(adjusted[index][pos]), value)
                    weighted_positions += 1
                    token_counts[token_text] = token_counts.get(token_text, 0) + 1
        if row_matched:
            matched_rows += 1
    return adjusted, {
        "enabled": True,
        "policy": "private_loop_statement_action_span_weighting_v1",
        "loop_statement_action_loss_boost": value,
        "role_filter": sorted(allowed_roles),
        "role_filter_active": bool(allowed_roles),
        "rows": len(token_weight_rows),
        "matched_rows": matched_rows,
        "weighted_token_positions": weighted_positions,
        "excluded_positive_token_positions": excluded_positions,
        "weighted_token_counts": dict(sorted(token_counts.items())),
        "excluded_positive_token_counts": dict(sorted(excluded_token_counts.items())),
        "role_match_counts": dict(sorted(role_counts.items())),
        "skipped_counts": dict(sorted(skipped.items())),
        "score_semantics": (
            "Boosts supervised CE weight on exact token spans for selected AST-visible loop-body "
            "update/decision/finalizer roles in admitted private target bodies. It matches spans "
            "against the encoded private target row before weighting; unmatched or role-filtered spans are skipped. Bare "
            "`continue`/`break` control-flow terminals are retained for span matching context but excluded "
            "from positive weighting so they are not taught as update actions. Generation still sees only "
            "strict prompt/signature source text. This does not render code, inspect eval tests/solutions, "
            "use public data, use teacher output, or grant candidate-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def apply_loop_semantic_operation_weights(
    token_weight_rows: list[list[float]],
    target_rows: list[list[int]],
    bodies: list[str],
    *,
    target_vocab: dict[str, int],
    boost: float,
    roles: str,
) -> tuple[list[list[float]], dict[str, Any]]:
    value = max(0.0, float(boost if boost is not None else 0.0))
    allowed_roles = parse_role_filter(roles)
    if value <= 0.0:
        return token_weight_rows, {
            "enabled": False,
            "policy": "not_enabled",
            "loop_semantic_operation_loss_boost": value,
            "role_filter": sorted(allowed_roles),
            "matched_rows": 0,
            "weighted_token_positions": 0,
            "candidate_generation_credit": 0,
        }
    inverse = {int(token_id): str(token) for token, token_id in target_vocab.items()}
    adjusted = [list(row) for row in token_weight_rows]
    matched_rows = 0
    weighted_positions = 0
    excluded_positions = 0
    role_counts: dict[str, int] = {}
    semantic_kind_counts: dict[str, int] = {}
    token_counts: dict[str, int] = {}
    excluded_token_counts: dict[str, int] = {}
    skipped: dict[str, int] = {}
    for index, target in enumerate(target_rows):
        body = bodies[index] if index < len(bodies) else ""
        extraction = loop_semantic_operation_spans_for_body(body)
        spans = list(extraction.get("spans") or [])
        for key, count in dict_or_empty(extraction.get("skipped_counts")).items():
            skipped[str(key)] = skipped.get(str(key), 0) + int(count or 0)
        if not bool(extraction.get("matched")) or not spans:
            reason = str(extraction.get("reason") or "no_loop_semantic_operation_spans")
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        target_texts = [inverse.get(int(token_id), "") for token_id in target]
        row_matched = False
        for span in spans:
            item = dict_or_empty(span)
            role = str(item.get("role") or "unknown")
            if allowed_roles and role not in allowed_roles:
                skipped[f"role_filtered_{role}"] = skipped.get(f"role_filtered_{role}", 0) + 1
                continue
            span_tokens = [str(token) for token in list(item.get("tokens") or []) if str(token)]
            excluded_positive_tokens = {str(token) for token in list(item.get("excluded_positive_tokens") or [])}
            if not span_tokens:
                skipped["empty_span_tokens"] = skipped.get("empty_span_tokens", 0) + 1
                continue
            matches = find_subsequence_positions(target_texts, span_tokens)
            if not matches:
                kind = str(item.get("semantic_kind") or role)
                skipped[f"span_not_found_{kind}"] = skipped.get(f"span_not_found_{kind}", 0) + 1
                continue
            role_counts[role] = role_counts.get(role, 0) + len(matches)
            semantic_kind = str(item.get("semantic_kind") or "unknown")
            semantic_kind_counts[semantic_kind] = semantic_kind_counts.get(semantic_kind, 0) + len(matches)
            row_matched = True
            for start in matches:
                for offset, token_text in enumerate(span_tokens):
                    if token_text in excluded_positive_tokens:
                        excluded_positions += 1
                        excluded_token_counts[token_text] = excluded_token_counts.get(token_text, 0) + 1
                        continue
                    pos = start + offset
                    if pos >= len(adjusted[index]):
                        continue
                    adjusted[index][pos] = max(float(adjusted[index][pos]), value)
                    weighted_positions += 1
                    token_counts[token_text] = token_counts.get(token_text, 0) + 1
        if row_matched:
            matched_rows += 1
    return adjusted, {
        "enabled": True,
        "policy": "private_loop_semantic_operation_span_weighting_v1",
        "loop_semantic_operation_loss_boost": value,
        "role_filter": sorted(allowed_roles),
        "role_filter_active": bool(allowed_roles),
        "rows": len(token_weight_rows),
        "matched_rows": matched_rows,
        "weighted_token_positions": weighted_positions,
        "excluded_positive_token_positions": excluded_positions,
        "weighted_token_counts": dict(sorted(token_counts.items())),
        "excluded_positive_token_counts": dict(sorted(excluded_token_counts.items())),
        "role_match_counts": dict(sorted(role_counts.items())),
        "semantic_kind_match_counts": dict(sorted(semantic_kind_counts.items())),
        "skipped_counts": dict(sorted(skipped.items())),
        "score_semantics": (
            "Boosts supervised CE weight on exact token spans for semantic loop operations in admitted "
            "private target bodies: non-identity loop updates, accumulator transforms, nested/projected "
            "mutation calls, and top-level transform finalizers. It explicitly excludes shallow direct "
            "loop-variable append/add spans from this semantic boost. Generation still sees only strict "
            "prompt/signature source text. This does not render code, inspect eval tests/solutions, use "
            "public data, use teacher output, or grant candidate-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def apply_semantic_slot_prefix_weights(
    token_weight_rows: list[list[float]],
    target_rows: list[list[int]],
    *,
    target_vocab: dict[str, int],
    boost: float,
    roles: str,
) -> tuple[list[list[float]], dict[str, Any]]:
    value = max(0.0, float(boost if boost is not None else 0.0))
    allowed_roles = parse_role_filter(roles)
    if value <= 0.0:
        return token_weight_rows, {
            "enabled": False,
            "policy": "not_enabled",
            "semantic_slot_prefix_loss_boost": value,
            "role_filter": sorted(allowed_roles),
            "matched_rows": 0,
            "weighted_token_positions": 0,
            "candidate_generation_credit": 0,
        }
    inverse = {int(token_id): str(token) for token, token_id in target_vocab.items()}
    adjusted = [list(row) for row in token_weight_rows]
    matched_rows = 0
    weighted_positions = 0
    role_counts: dict[str, int] = {}
    token_counts: dict[str, int] = {}
    skipped: dict[str, int] = {}
    for index, target in enumerate(target_rows):
        body_started = False
        row_matched = False
        for pos, token_id in enumerate(target):
            token_text = inverse.get(int(token_id), "")
            if token_text == "SLOT:BODY_START":
                body_started = True
                break
            if not token_text.startswith("SLOT:"):
                continue
            role = semantic_slot_prefix_role(token_text)
            if allowed_roles and role not in allowed_roles:
                skipped[f"role_filtered_{role}"] = skipped.get(f"role_filtered_{role}", 0) + 1
                continue
            if pos >= len(adjusted[index]):
                continue
            adjusted[index][pos] = max(float(adjusted[index][pos]), value)
            weighted_positions += 1
            row_matched = True
            role_counts[role] = role_counts.get(role, 0) + 1
            token_counts[token_text] = token_counts.get(token_text, 0) + 1
        if not body_started:
            skipped["missing_body_start"] = skipped.get("missing_body_start", 0) + 1
        if row_matched:
            matched_rows += 1
        else:
            skipped["row_without_weighted_slot"] = skipped.get("row_without_weighted_slot", 0) + 1
    return adjusted, {
        "enabled": True,
        "policy": "private_semantic_slot_prefix_weighting_v1",
        "semantic_slot_prefix_loss_boost": value,
        "role_filter": sorted(allowed_roles),
        "role_filter_active": bool(allowed_roles),
        "rows": len(token_weight_rows),
        "matched_rows": matched_rows,
        "weighted_token_positions": weighted_positions,
        "role_match_counts": dict(sorted(role_counts.items())),
        "weighted_token_counts": dict(sorted(token_counts.items())),
        "skipped_counts": dict(sorted(skipped.items())),
        "score_semantics": (
            "Boosts supervised CE weight on generated semantic slot-prefix tokens before SLOT:BODY_START "
            "for admitted private target bodies. This trains learned plan/update/guard/finalizer slot "
            "selection from prompt/signature source text; it does not render code, inspect tests or "
            "solutions, use public data, use teacher output, or grant candidate-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def apply_loop_expression_synthesis_weights(
    token_weight_rows: list[list[float]],
    target_rows: list[list[int]],
    bodies: list[str],
    *,
    target_vocab: dict[str, int],
    boost: float,
    roles: str,
) -> tuple[list[list[float]], dict[str, Any]]:
    value = max(0.0, float(boost if boost is not None else 0.0))
    allowed_roles = parse_role_filter(roles)
    if value <= 0.0:
        return token_weight_rows, {
            "enabled": False,
            "policy": "not_enabled",
            "loop_expression_synthesis_loss_boost": value,
            "role_filter": sorted(allowed_roles),
            "matched_rows": 0,
            "weighted_token_positions": 0,
            "candidate_generation_credit": 0,
        }
    inverse = {int(token_id): str(token) for token, token_id in target_vocab.items()}
    adjusted = [list(row) for row in token_weight_rows]
    matched_rows = 0
    weighted_positions = 0
    role_counts: dict[str, int] = {}
    expression_kind_counts: dict[str, int] = {}
    token_counts: dict[str, int] = {}
    skipped: dict[str, int] = {}
    for index, target in enumerate(target_rows):
        body = bodies[index] if index < len(bodies) else ""
        extraction = loop_expression_synthesis_spans_for_body(body)
        spans = list(extraction.get("spans") or [])
        for key, count in dict_or_empty(extraction.get("skipped_counts")).items():
            skipped[str(key)] = skipped.get(str(key), 0) + int(count or 0)
        if not bool(extraction.get("matched")) or not spans:
            reason = str(extraction.get("reason") or "no_loop_expression_synthesis_spans")
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        target_texts = [inverse.get(int(token_id), "") for token_id in target]
        row_matched = False
        for span in spans:
            item = dict_or_empty(span)
            role = str(item.get("role") or "unknown")
            if allowed_roles and role not in allowed_roles:
                skipped[f"role_filtered_{role}"] = skipped.get(f"role_filtered_{role}", 0) + 1
                continue
            span_tokens = [str(token) for token in list(item.get("tokens") or []) if str(token)]
            if not span_tokens:
                skipped["empty_expression_tokens"] = skipped.get("empty_expression_tokens", 0) + 1
                continue
            matches = find_subsequence_positions(target_texts, span_tokens)
            if not matches:
                kind = str(item.get("expression_kind") or role)
                skipped[f"span_not_found_{kind}"] = skipped.get(f"span_not_found_{kind}", 0) + 1
                continue
            role_counts[role] = role_counts.get(role, 0) + len(matches)
            expression_kind = str(item.get("expression_kind") or "unknown")
            expression_kind_counts[expression_kind] = expression_kind_counts.get(expression_kind, 0) + len(matches)
            row_matched = True
            for start in matches:
                for offset, token_text in enumerate(span_tokens):
                    pos = start + offset
                    if pos >= len(adjusted[index]):
                        continue
                    adjusted[index][pos] = max(float(adjusted[index][pos]), value)
                    weighted_positions += 1
                    token_counts[token_text] = token_counts.get(token_text, 0) + 1
        if row_matched:
            matched_rows += 1
    return adjusted, {
        "enabled": True,
        "policy": "private_loop_expression_synthesis_span_weighting_v1",
        "loop_expression_synthesis_loss_boost": value,
        "role_filter": sorted(allowed_roles),
        "role_filter_active": bool(allowed_roles),
        "rows": len(token_weight_rows),
        "matched_rows": matched_rows,
        "weighted_token_positions": weighted_positions,
        "weighted_token_counts": dict(sorted(token_counts.items())),
        "role_match_counts": dict(sorted(role_counts.items())),
        "expression_kind_match_counts": dict(sorted(expression_kind_counts.items())),
        "skipped_counts": dict(sorted(skipped.items())),
        "score_semantics": (
            "Boosts supervised CE weight on expression-level spans inside admitted private loop targets: "
            "loop condition tests, non-identity update RHS/arguments, and semantic top-level finalizer "
            "expressions. It deliberately targets the expression that makes a loop update meaningful "
            "instead of the surrounding renderer-like skeleton. Generation still sees only strict "
            "prompt/signature source text. This does not render code, inspect eval tests/solutions, use "
            "public data, use teacher output, or grant candidate-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def apply_plan_conditioned_body_semantic_weights(
    token_weight_rows: list[list[float]],
    target_rows: list[list[int]],
    bodies: list[str],
    *,
    target_vocab: dict[str, int],
    boost: float,
    roles: str,
) -> tuple[list[list[float]], dict[str, Any]]:
    value = max(0.0, float(boost if boost is not None else 0.0))
    allowed_roles = parse_role_filter(roles)
    if value <= 0.0:
        return token_weight_rows, {
            "enabled": False,
            "policy": "not_enabled",
            "plan_conditioned_body_loss_boost": value,
            "role_filter": sorted(allowed_roles),
            "matched_rows": 0,
            "weighted_token_positions": 0,
            "candidate_generation_credit": 0,
        }
    inverse = {int(token_id): str(token) for token, token_id in target_vocab.items()}
    adjusted = [list(row) for row in token_weight_rows]
    matched_rows = 0
    weighted_positions = 0
    role_counts: dict[str, int] = {}
    plan_counts: dict[str, int] = {}
    semantic_kind_counts: dict[str, int] = {}
    token_counts: dict[str, int] = {}
    skipped: dict[str, int] = {}
    for index, target in enumerate(target_rows):
        body = bodies[index] if index < len(bodies) else ""
        extraction = plan_conditioned_body_semantic_spans_for_body(body)
        for key, count in dict_or_empty(extraction.get("skipped_counts")).items():
            skipped[str(key)] = skipped.get(str(key), 0) + int(count or 0)
        spans = list(extraction.get("spans") or [])
        if not bool(extraction.get("matched")) or not spans:
            reason = str(extraction.get("reason") or "no_plan_conditioned_spans")
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        plan = str(extraction.get("plan") or "UNKNOWN")
        target_texts = [inverse.get(int(token_id), "") for token_id in target]
        row_matched = False
        for span in spans:
            item = dict_or_empty(span)
            role = str(item.get("role") or "unknown")
            if allowed_roles and role not in allowed_roles:
                skipped[f"role_filtered_{role}"] = skipped.get(f"role_filtered_{role}", 0) + 1
                continue
            span_tokens = [str(token) for token in list(item.get("tokens") or []) if str(token)]
            if not span_tokens:
                skipped["empty_span_tokens"] = skipped.get("empty_span_tokens", 0) + 1
                continue
            matches = find_subsequence_positions(target_texts, span_tokens)
            if not matches:
                kind = str(item.get("semantic_kind") or role)
                skipped[f"span_not_found_{kind}"] = skipped.get(f"span_not_found_{kind}", 0) + 1
                continue
            row_matched = True
            role_counts[role] = role_counts.get(role, 0) + len(matches)
            semantic_kind = str(item.get("semantic_kind") or "unknown")
            semantic_kind_counts[semantic_kind] = semantic_kind_counts.get(semantic_kind, 0) + len(matches)
            for start in matches:
                for offset, token_text in enumerate(span_tokens):
                    pos = start + offset
                    if pos >= len(adjusted[index]):
                        continue
                    adjusted[index][pos] = max(float(adjusted[index][pos]), value)
                    weighted_positions += 1
                    token_counts[token_text] = token_counts.get(token_text, 0) + 1
        if row_matched:
            matched_rows += 1
            plan_counts[plan] = plan_counts.get(plan, 0) + 1
    return adjusted, {
        "enabled": True,
        "policy": "private_plan_conditioned_body_semantic_weighting_v1",
        "plan_conditioned_body_loss_boost": value,
        "role_filter": sorted(allowed_roles),
        "role_filter_active": bool(allowed_roles),
        "rows": len(token_weight_rows),
        "matched_rows": matched_rows,
        "weighted_token_positions": weighted_positions,
        "weighted_token_counts": dict(sorted(token_counts.items())),
        "role_match_counts": dict(sorted(role_counts.items())),
        "plan_match_counts": dict(sorted(plan_counts.items())),
        "semantic_kind_match_counts": dict(sorted(semantic_kind_counts.items())),
        "skipped_counts": dict(sorted(skipped.items())),
        "score_semantics": (
            "Boosts supervised CE weight on admitted private target body-token spans selected by the "
            "existing semantic plan taxonomy. It targets body semantics conditioned on the plan label: "
            "branch guards, loop update/source expressions, aggregate/text/dict/list key calls, and final "
            "return expressions. The plan and spans are derived only from admitted private target ASTs as "
            "training targets; generation still sees strict prompt/signature source text. This does not "
            "render code, inspect eval tests/solutions, use public data, use teacher output, route tools, "
            "or grant learned-generation candidate credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def apply_update_contract_consistency_weights(
    token_weight_rows: list[list[float]],
    target_rows: list[list[int]],
    bodies: list[str],
    *,
    target_vocab: dict[str, int],
    boost: float,
) -> tuple[list[list[float]], dict[str, Any]]:
    value = max(0.0, float(boost if boost is not None else 0.0))
    if value <= 0.0:
        return token_weight_rows, {
            "enabled": False,
            "policy": "not_enabled",
            "update_contract_consistency_loss_boost": value,
            "matched_rows": 0,
            "weighted_token_positions": 0,
            "candidate_generation_credit": 0,
        }
    inverse = {int(token_id): str(token) for token, token_id in target_vocab.items()}
    adjusted = [list(row) for row in token_weight_rows]
    matched_rows = 0
    weighted_positions = 0
    prefix_weighted_positions = 0
    body_weighted_positions = 0
    contract_counts: dict[str, int] = {}
    token_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    skipped: dict[str, int] = {}

    for index, target in enumerate(target_rows):
        body = bodies[index] if index < len(bodies) else ""
        contract = update_contract_consistency_spans_for_body(body)
        for key, count in dict_or_empty(contract.get("skipped_counts")).items():
            skipped[str(key)] = skipped.get(str(key), 0) + int(count or 0)
        spans = list(contract.get("spans") or [])
        prefix_slots = [str(token) for token in list(contract.get("prefix_slots") or []) if str(token)]
        if not bool(contract.get("matched")) or not (spans or prefix_slots):
            reason = str(contract.get("reason") or "no_update_contract")
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        target_texts = [inverse.get(int(token_id), "") for token_id in target]
        row_matched = False
        for pos, token_text in enumerate(target_texts):
            if token_text not in prefix_slots:
                continue
            if pos >= len(adjusted[index]):
                continue
            adjusted[index][pos] = max(float(adjusted[index][pos]), value)
            weighted_positions += 1
            prefix_weighted_positions += 1
            token_counts[token_text] = token_counts.get(token_text, 0) + 1
            role_counts["prefix_contract_slot"] = role_counts.get("prefix_contract_slot", 0) + 1
            row_matched = True
        for span in spans:
            item = dict_or_empty(span)
            role = str(item.get("role") or "unknown")
            span_tokens = [str(token) for token in list(item.get("tokens") or []) if str(token)]
            if not span_tokens:
                skipped[f"empty_span_{role}"] = skipped.get(f"empty_span_{role}", 0) + 1
                continue
            matches = find_subsequence_positions(target_texts, span_tokens)
            if not matches:
                skipped[f"span_not_found_{role}"] = skipped.get(f"span_not_found_{role}", 0) + 1
                continue
            row_matched = True
            role_counts[role] = role_counts.get(role, 0) + len(matches)
            for start in matches:
                for offset, token_text in enumerate(span_tokens):
                    pos = start + offset
                    if pos >= len(adjusted[index]):
                        continue
                    adjusted[index][pos] = max(float(adjusted[index][pos]), value)
                    weighted_positions += 1
                    body_weighted_positions += 1
                    token_counts[token_text] = token_counts.get(token_text, 0) + 1
        if row_matched:
            matched_rows += 1
            for name, count in dict_or_empty(contract.get("contract_counts")).items():
                contract_counts[str(name)] = contract_counts.get(str(name), 0) + int(count or 0)

    return adjusted, {
        "enabled": True,
        "policy": "private_update_contract_consistency_weighting_v1",
        "update_contract_consistency_loss_boost": value,
        "rows": len(token_weight_rows),
        "matched_rows": matched_rows,
        "weighted_token_positions": weighted_positions,
        "prefix_weighted_token_positions": prefix_weighted_positions,
        "body_weighted_token_positions": body_weighted_positions,
        "weighted_token_counts": dict(sorted(token_counts.items())),
        "role_match_counts": dict(sorted(role_counts.items())),
        "contract_match_counts": dict(sorted(contract_counts.items())),
        "skipped_counts": dict(sorted(skipped.items())),
        "score_semantics": (
            "Boosts supervised CE weight on admitted private target update-contract evidence: learned "
            "SLOT:UPDATE_/SLOT:STATE_UPDATE_/SLOT:BIND_UPDATE_ prefix slots and the exact loop update "
            "statement span that implements that state transition. This is target-side private "
            "supervision only. Generation still sees strict prompt/signature source text, and this does "
            "not inspect eval tests/solutions, use public data, use teacher output, render code, call "
            "tools, or grant learned-generation candidate credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def apply_direct_body_emission_path_weights(
    token_weight_rows: list[list[float]],
    target_rows: list[list[int]],
    bodies: list[str],
    *,
    target_vocab: dict[str, int],
    boost: float,
    roles: str,
) -> tuple[list[list[float]], dict[str, Any]]:
    value = max(0.0, float(boost if boost is not None else 0.0))
    allowed_roles = parse_role_filter(roles)
    if value <= 0.0:
        return token_weight_rows, {
            "enabled": False,
            "policy": "not_enabled",
            "direct_body_emission_loss_boost": value,
            "role_filter": sorted(allowed_roles),
            "matched_rows": 0,
            "weighted_token_positions": 0,
            "candidate_generation_credit": 0,
        }

    inverse = {int(token_id): str(token) for token, token_id in target_vocab.items()}
    adjusted = [list(row) for row in token_weight_rows]
    matched_rows = 0
    weighted_positions = 0
    role_counts: dict[str, int] = {}
    token_counts: dict[str, int] = {}
    skipped: dict[str, int] = {}

    for index, target in enumerate(target_rows):
        body = bodies[index] if index < len(bodies) else ""
        extraction = direct_body_emission_path_spans_for_body(body)
        for key, count in dict_or_empty(extraction.get("skipped_counts")).items():
            skipped[str(key)] = skipped.get(str(key), 0) + int(count or 0)
        spans = list(extraction.get("spans") or [])
        if not bool(extraction.get("matched")) or not spans:
            reason = str(extraction.get("reason") or "no_direct_body_emission_spans")
            skipped[reason] = skipped.get(reason, 0) + 1
            continue

        target_texts = [inverse.get(int(token_id), "") for token_id in target]
        row_matched = False
        for span in spans:
            item = dict_or_empty(span)
            role = str(item.get("role") or "unknown")
            if allowed_roles and role not in allowed_roles:
                skipped[f"role_filtered_{role}"] = skipped.get(f"role_filtered_{role}", 0) + 1
                continue
            span_tokens = [str(token) for token in list(item.get("tokens") or []) if str(token)]
            if not span_tokens:
                skipped[f"empty_span_tokens_{role}"] = skipped.get(f"empty_span_tokens_{role}", 0) + 1
                continue
            matches = find_subsequence_positions(target_texts, span_tokens)
            if not matches:
                skipped[f"span_not_found_{role}"] = skipped.get(f"span_not_found_{role}", 0) + 1
                continue
            row_matched = True
            role_counts[role] = role_counts.get(role, 0) + len(matches)
            for start in matches:
                for offset, token_text in enumerate(span_tokens):
                    pos = start + offset
                    if pos >= len(adjusted[index]):
                        continue
                    adjusted[index][pos] = max(float(adjusted[index][pos]), value)
                    weighted_positions += 1
                    token_counts[token_text] = token_counts.get(token_text, 0) + 1
        if row_matched:
            matched_rows += 1

    return adjusted, {
        "enabled": True,
        "policy": "private_direct_body_emission_path_weighting_v1",
        "direct_body_emission_loss_boost": value,
        "role_filter": sorted(allowed_roles),
        "role_filter_active": bool(allowed_roles),
        "rows": len(token_weight_rows),
        "matched_rows": matched_rows,
        "weighted_token_positions": weighted_positions,
        "weighted_token_counts": dict(sorted(token_counts.items())),
        "role_match_counts": dict(sorted(role_counts.items())),
        "skipped_counts": dict(sorted(skipped.items())),
        "score_semantics": (
            "Boosts supervised CE weight on exact body-token spans that form the reachable emission path "
            "in admitted private target bodies: top-level state bindings, branch guards, loop headers, "
            "loop body updates, local-state returns, and nontrivial return expressions. It fixes a gap in "
            "loop-only weighting by covering non-loop bodies and direct `return local` finalizers. It uses "
            "private target bodies only as already-admitted training targets. Generation still sees strict "
            "prompt/signature source text. It does not render code, inspect eval tests/solutions, use "
            "public data, use teacher output, route tools, or grant learned-generation candidate credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def direct_body_emission_path_spans_for_body(body: str) -> dict[str, Any]:
    text = str(body or "")
    try:
        parsed = ast.parse("def _candidate(data, other=None):\n" + "\n".join(f"    {line}" for line in text.splitlines()))
    except SyntaxError as exc:
        return {
            "matched": False,
            "reason": "parse_error",
            "error": str(exc)[:160],
            "spans": [],
            "skipped_counts": {},
        }
    function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    if function is None:
        return {"matched": False, "reason": "missing_function", "spans": [], "skipped_counts": {}}

    spans: list[dict[str, Any]] = []
    skipped_counts: dict[str, int] = {}
    local_bindings = function_local_binding_names(function)
    parameter_names = {arg.arg for arg in list(function.args.args or []) if str(arg.arg)}

    def skip(reason: str) -> None:
        skipped_counts[reason] = skipped_counts.get(reason, 0) + 1

    def add_stmt(role: str, stmt: ast.stmt, *, semantic_kind: str = "") -> None:
        tokens = body_statement_tokens(stmt)
        if not tokens:
            skip(f"empty_tokens_{role}")
            return
        spans.append(
            {
                "role": role,
                "semantic_kind": semantic_kind or role,
                "tokens": tokens,
                "statement": safe_unparse(stmt)[:240],
            }
        )

    def add_expr(role: str, expr: ast.AST | None, *, semantic_kind: str, context: str) -> None:
        if expr is None:
            skip(f"empty_expr_{role}")
            return
        tokens = expression_body_tokens(expr)
        if not tokens:
            skip(f"empty_expr_tokens_{role}")
            return
        if expression_tokens_are_trivial(tokens):
            skip(f"trivial_expr_tokens_{role}")
            return
        spans.append(
            {
                "role": role,
                "semantic_kind": semantic_kind,
                "tokens": tokens,
                "expression": safe_unparse(expr)[:240],
                "context": context[:240],
            }
        )

    for stmt in function.body:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            value = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
            if value_has_semantic_operation(value) or ast_load_names(value):
                add_stmt("top_level_state_binding", stmt, semantic_kind=expression_kind_for_ast(value))
            else:
                skip("plain_top_level_binding")
        elif isinstance(stmt, ast.AugAssign):
            add_stmt("top_level_state_update", stmt, semantic_kind="augmented_state_update")
        elif isinstance(stmt, ast.If):
            add_expr(
                "top_level_branch_guard",
                stmt.test,
                semantic_kind=expression_kind_for_ast(stmt.test),
                context=safe_unparse(stmt),
            )
            for child in list(stmt.body or []) + list(stmt.orelse or []):
                if isinstance(child, ast.Return):
                    add_return_span(child, spans, skipped_counts, local_bindings=local_bindings, parameter_names=parameter_names)
                elif isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr)):
                    add_stmt("branch_body_state_transition", child, semantic_kind=type(child).__name__)
        elif isinstance(stmt, (ast.For, ast.While)):
            add_stmt("top_level_loop_statement", stmt, semantic_kind=type(stmt).__name__)
            if isinstance(stmt, ast.For):
                add_expr(
                    "loop_source_expression",
                    stmt.iter,
                    semantic_kind=expression_kind_for_ast(stmt.iter),
                    context=safe_unparse(stmt),
                )
            else:
                add_expr(
                    "loop_condition_expression",
                    stmt.test,
                    semantic_kind=expression_kind_for_ast(stmt.test),
                    context=safe_unparse(stmt),
                )
            loop_targets = ast_store_names(getattr(stmt, "target", None))
            for child in semantic_candidate_statements_from_loop(stmt):
                classification = classify_loop_semantic_statement(child, loop_targets=loop_targets)
                if bool(classification.get("matched")):
                    add_stmt(
                        "loop_body_state_transition",
                        child,
                        semantic_kind=str(classification.get("semantic_kind") or "loop_body_state_transition"),
                    )
                else:
                    skip(str(classification.get("reason") or "loop_child_not_semantic"))
        elif isinstance(stmt, ast.Return):
            add_return_span(stmt, spans, skipped_counts, local_bindings=local_bindings, parameter_names=parameter_names)

    if not spans:
        return {
            "matched": False,
            "reason": "no_direct_body_emission_spans",
            "spans": [],
            "skipped_counts": dict(sorted(skipped_counts.items())),
        }

    role_counts: dict[str, int] = {}
    semantic_kind_counts: dict[str, int] = {}
    for span in spans:
        role = str(span.get("role") or "unknown")
        semantic_kind = str(span.get("semantic_kind") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
        semantic_kind_counts[semantic_kind] = semantic_kind_counts.get(semantic_kind, 0) + 1
    return {
        "matched": True,
        "policy": "private_direct_body_emission_path_span_extraction_v1",
        "spans": spans,
        "role_counts": dict(sorted(role_counts.items())),
        "semantic_kind_counts": dict(sorted(semantic_kind_counts.items())),
        "skipped_counts": dict(sorted(skipped_counts.items())),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def add_return_span(
    stmt: ast.Return,
    spans: list[dict[str, Any]],
    skipped_counts: dict[str, int],
    *,
    local_bindings: set[str],
    parameter_names: set[str],
) -> None:
    value = stmt.value
    if value is None:
        skipped_counts["empty_return"] = skipped_counts.get("empty_return", 0) + 1
        return
    if isinstance(value, ast.Constant):
        skipped_counts["constant_return"] = skipped_counts.get("constant_return", 0) + 1
        return
    if isinstance(value, ast.Name) and value.id in parameter_names and value.id not in local_bindings:
        skipped_counts["direct_parameter_return"] = skipped_counts.get("direct_parameter_return", 0) + 1
        return
    role = "top_level_local_state_return" if isinstance(value, ast.Name) and value.id in local_bindings else "top_level_return_expression"
    tokens = body_statement_tokens(stmt)
    if not tokens:
        skipped_counts[f"empty_tokens_{role}"] = skipped_counts.get(f"empty_tokens_{role}", 0) + 1
        return
    spans.append(
        {
            "role": role,
            "semantic_kind": expression_kind_for_ast(value),
            "tokens": tokens,
            "statement": safe_unparse(stmt)[:240],
        }
    )


def function_local_binding_names(function: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(ast_store_names(target))
        elif isinstance(node, ast.AnnAssign):
            names.update(ast_store_names(node.target))
        elif isinstance(node, ast.AugAssign):
            names.update(ast_store_names(node.target))
        elif isinstance(node, ast.For):
            names.update(ast_store_names(node.target))
        elif isinstance(node, ast.With):
            for item in node.items:
                names.update(ast_store_names(item.optional_vars))
    return names


def update_contract_consistency_spans_for_body(body: str) -> dict[str, Any]:
    text = str(body or "")
    try:
        parsed = ast.parse("def _candidate(data, other=None):\n" + "\n".join(f"    {line}" for line in text.splitlines()))
    except SyntaxError as exc:
        return {
            "matched": False,
            "reason": "parse_error",
            "error": str(exc)[:160],
            "prefix_slots": [],
            "spans": [],
            "skipped_counts": {},
        }
    function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    if function is None:
        return {"matched": False, "reason": "missing_function", "prefix_slots": [], "spans": [], "skipped_counts": {}}

    prefix_slots = [
        token
        for token in learned_body_decision_prefix_tokens_for_body(text)
        if token.startswith("SLOT:UPDATE_")
        or token.startswith("SLOT:STATE_UPDATE_")
        or token.startswith("SLOT:BIND_UPDATE_")
    ]
    spans: list[dict[str, Any]] = []
    skipped_counts: dict[str, int] = {}
    contract_counts: dict[str, int] = {}
    init_names: set[str] = set()
    for stmt in function.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                init_names.update(ast_store_names(target))
        elif isinstance(stmt, ast.AnnAssign):
            init_names.update(ast_store_names(stmt.target))

    def add_span(role: str, stmt: ast.stmt, *, contract: str) -> None:
        tokens = body_statement_tokens(stmt)
        if not tokens:
            skipped_counts[f"empty_tokens_{role}"] = skipped_counts.get(f"empty_tokens_{role}", 0) + 1
            return
        spans.append(
            {
                "role": role,
                "contract": contract,
                "tokens": tokens,
                "statement": safe_unparse(stmt)[:240],
            }
        )
        contract_counts[contract] = contract_counts.get(contract, 0) + 1

    for loop in [node for node in ast.walk(function) if isinstance(node, (ast.For, ast.While))]:
        loop_targets = ast_store_names(getattr(loop, "target", None))
        for stmt in list(getattr(loop, "body", []) or []):
            if isinstance(stmt, ast.AugAssign):
                add_span("loop_update_augassign", stmt, contract="update_augassign")
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                value = stmt.value if isinstance(stmt, ast.Assign) else stmt.value
                target_names: set[str] = set()
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        target_names.update(ast_store_names(target))
                else:
                    target_names.update(ast_store_names(stmt.target))
                value_names = ast_load_names(value)
                if target_names & init_names and (value_names & (init_names | loop_targets) or value_has_semantic_operation(value)):
                    add_span("loop_update_assign_transform", stmt, contract="update_assign_transform")
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if semantic_mutation_call_kind(call):
                    add_span("loop_update_mutation_call", stmt, contract=str(semantic_mutation_call_kind(call)))

    if not spans and not prefix_slots:
        return {
            "matched": False,
            "reason": "no_update_contract_spans_or_slots",
            "prefix_slots": [],
            "spans": [],
            "skipped_counts": dict(sorted(skipped_counts.items())),
        }
    return {
        "matched": True,
        "policy": "private_update_contract_consistency_span_extraction_v1",
        "prefix_slots": prefix_slots,
        "spans": spans,
        "contract_counts": dict(sorted(contract_counts.items())),
        "skipped_counts": dict(sorted(skipped_counts.items())),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def plan_conditioned_body_semantic_spans_for_body(body: str) -> dict[str, Any]:
    text = str(body or "")
    try:
        parsed = ast.parse("def _candidate(data, other=None):\n" + "\n".join(f"    {line}" for line in text.splitlines()))
    except SyntaxError as exc:
        return {
            "matched": False,
            "reason": "parse_error",
            "error": str(exc)[:160],
            "plan": "AST_INVALID",
            "spans": [],
            "skipped_counts": {},
        }
    function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    if function is None:
        return {"matched": False, "reason": "missing_function", "plan": "AST_INVALID", "spans": [], "skipped_counts": {}}
    plan = semantic_plan_from_body(text)
    plan_upper = str(plan or "").upper()
    spans: list[dict[str, Any]] = []
    skipped_counts: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped_counts[reason] = skipped_counts.get(reason, 0) + 1

    def add_expr(role: str, expr: ast.AST | None, *, semantic_kind: str, context: str) -> None:
        if expr is None:
            skip(f"empty_{role}")
            return
        tokens = expression_body_tokens(expr)
        if not tokens:
            skip(f"empty_tokens_{role}")
            return
        if expression_tokens_are_trivial(tokens):
            skip(f"trivial_tokens_{role}")
            return
        spans.append(
            {
                "role": role,
                "semantic_kind": semantic_kind,
                "plan": plan,
                "tokens": tokens,
                "expression": safe_unparse(expr)[:240],
                "context": context[:240],
            }
        )

    def add_stmt(role: str, stmt: ast.stmt, *, semantic_kind: str) -> None:
        tokens = body_statement_tokens(stmt)
        if not tokens:
            skip(f"empty_tokens_{role}")
            return
        spans.append(
            {
                "role": role,
                "semantic_kind": semantic_kind,
                "plan": plan,
                "tokens": tokens,
                "statement": safe_unparse(stmt)[:240],
            }
        )

    branch_plan = "BRANCH" in plan_upper or plan_upper in {"SAFE_HEAD_DEFAULT", "DEVICE_ROUTE_WORKER", "VOICE_OUTPUT_ROUTE"}
    loop_plan = "LOOP" in plan_upper or plan_upper in {
        "AST_LIST_ACCUMULATE",
        "AST_LIST_FILTER_MAP",
        "AST_DICT_GROUP_APPEND",
        "AST_DICT_GROUP_SET",
        "AST_DICT_COUNT",
        "AST_SET_ACCUMULATE",
        "AST_MAPPING_UPDATE",
        "AST_TEXT_BUILD_JOIN",
        "AST_TEXT_SPLIT_TRANSFORM",
        "GRAPH_COMPONENTS",
        "SHORTEST_HOPS",
        "LONGEST_EVEN_RUN",
        "RLE_ENCODE",
        "TOP_K_FREQUENT",
    }
    text_plan = "TEXT" in plan_upper or "PARSE" in plan_upper or plan_upper.startswith("STDIN_")
    aggregate_plan = "AGGREGATE" in plan_upper or plan_upper in {"INTERVAL_COVERAGE", "LCS_LENGTH", "MAX_NON_ADJACENT_SUM"}
    collection_plan = any(fragment in plan_upper for fragment in ["LIST", "DICT", "SET", "GROUP", "COUNT", "MERGE", "PROJECT", "SORT"])

    if branch_plan:
        for node in ast.walk(function):
            if isinstance(node, ast.If):
                add_expr(
                    "guard_expression",
                    node.test,
                    semantic_kind=expression_kind_for_ast(node.test),
                    context=safe_unparse(node),
                )

    if loop_plan:
        for node in ast.walk(function):
            if isinstance(node, ast.For):
                add_expr(
                    "loop_source_expression",
                    node.iter,
                    semantic_kind=expression_kind_for_ast(node.iter),
                    context=safe_unparse(node),
                )
                loop_targets = ast_store_names(node.target)
                for stmt in semantic_candidate_statements_from_loop(node):
                    classification = classify_loop_semantic_statement(stmt, loop_targets=loop_targets)
                    if bool(classification.get("matched")):
                        add_stmt(
                            "loop_update_statement",
                            stmt,
                            semantic_kind=str(classification.get("semantic_kind") or "loop_update_statement"),
                        )
            elif isinstance(node, ast.While):
                add_expr(
                    "loop_condition_expression",
                    node.test,
                    semantic_kind=expression_kind_for_ast(node.test),
                    context=safe_unparse(node),
                )

    key_call_names = plan_key_call_names(plan_upper, text_plan=text_plan, aggregate_plan=aggregate_plan, collection_plan=collection_plan)
    if key_call_names:
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            name = normalized_call_name(node)
            if name not in key_call_names:
                continue
            add_expr(
                "plan_key_call_expression",
                node,
                semantic_kind=f"call_{name}",
                context=safe_unparse(node),
            )

    for stmt in function.body:
        if not isinstance(stmt, ast.Return):
            continue
        classification = classify_top_level_semantic_finalizer(stmt)
        if bool(classification.get("matched")) or aggregate_plan or collection_plan or text_plan or branch_plan or "RETURN" in plan_upper:
            add_expr(
                "final_return_expression",
                stmt.value,
                semantic_kind=str(classification.get("semantic_kind") or expression_kind_for_ast(stmt.value)),
                context=safe_unparse(stmt),
            )

    if not spans:
        return {
            "matched": False,
            "reason": "no_plan_conditioned_spans",
            "plan": plan,
            "spans": [],
            "skipped_counts": dict(sorted(skipped_counts.items())),
        }
    role_counts: dict[str, int] = {}
    semantic_kind_counts: dict[str, int] = {}
    for span in spans:
        role = str(span.get("role") or "unknown")
        semantic_kind = str(span.get("semantic_kind") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
        semantic_kind_counts[semantic_kind] = semantic_kind_counts.get(semantic_kind, 0) + 1
    return {
        "matched": True,
        "policy": "private_plan_conditioned_body_semantic_span_extraction_v1",
        "plan": plan,
        "spans": spans,
        "role_counts": dict(sorted(role_counts.items())),
        "semantic_kind_counts": dict(sorted(semantic_kind_counts.items())),
        "skipped_counts": dict(sorted(skipped_counts.items())),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def plan_key_call_names(plan_upper: str, *, text_plan: bool, aggregate_plan: bool, collection_plan: bool) -> set[str]:
    names: set[str] = set()
    if text_plan:
        names.update({"split", "splitlines", "join", "int", "float", "str"})
    if aggregate_plan:
        names.update({"sum", "all", "any", "max", "min", "len", "range"})
    if collection_plan:
        names.update({"append", "add", "extend", "update", "setdefault", "get", "sorted", "sort", "items", "keys", "values"})
    if "GCD" in plan_upper:
        names.add("gcd")
    if "SHORTEST" in plan_upper:
        names.update({"deque", "popleft", "append"})
    return names


def normalized_call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Attribute):
        return str(call.func.attr or "")
    if isinstance(call.func, ast.Name):
        return str(call.func.id or "")
    return ""


def loop_expression_synthesis_spans_for_body(body: str) -> dict[str, Any]:
    text = str(body or "")
    try:
        parsed = ast.parse("def _candidate(data, other=None):\n" + "\n".join(f"    {line}" for line in text.splitlines()))
    except SyntaxError as exc:
        return {
            "matched": False,
            "reason": "parse_error",
            "error": str(exc)[:160],
            "spans": [],
            "skipped_counts": {},
        }
    function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    if function is None:
        return {"matched": False, "reason": "missing_function", "spans": [], "skipped_counts": {}}
    loops = [node for node in ast.walk(function) if isinstance(node, (ast.For, ast.While))]
    if not loops:
        return {"matched": False, "reason": "no_loop", "spans": [], "skipped_counts": {}}
    spans: list[dict[str, Any]] = []
    skipped_counts: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped_counts[reason] = skipped_counts.get(reason, 0) + 1

    def add_span(role: str, expr: ast.AST | None, *, expression_kind: str, context: str, loop_targets: set[str] | None = None) -> None:
        if expr is None:
            skip(f"empty_{role}")
            return
        if expression_is_plain_loop_identity(expr, loop_targets=set(loop_targets or set())):
            skip(f"plain_loop_identity_{role}")
            return
        tokens = expression_body_tokens(expr)
        if not tokens:
            skip(f"empty_tokens_{role}")
            return
        if expression_tokens_are_trivial(tokens):
            skip(f"trivial_tokens_{role}")
            return
        spans.append(
            {
                "role": role,
                "expression_kind": expression_kind,
                "tokens": tokens,
                "expression": safe_unparse(expr)[:240],
                "context": context[:240],
            }
        )

    def visit_loop_stmt(stmt: ast.stmt, *, loop_targets: set[str]) -> None:
        if isinstance(stmt, ast.If):
            add_span(
                "loop_condition_expression",
                stmt.test,
                expression_kind=expression_kind_for_ast(stmt.test),
                context=safe_unparse(stmt),
                loop_targets=loop_targets,
            )
            for child in list(stmt.body or []) + list(stmt.orelse or []):
                if isinstance(child, ast.stmt):
                    visit_loop_stmt(child, loop_targets=loop_targets)
            return
        if isinstance(stmt, ast.Try):
            for child in list(stmt.body or []) + list(stmt.orelse or []) + list(stmt.finalbody or []):
                if isinstance(child, ast.stmt):
                    visit_loop_stmt(child, loop_targets=loop_targets)
            for handler in list(stmt.handlers or []):
                for child in list(handler.body or []):
                    if isinstance(child, ast.stmt):
                        visit_loop_stmt(child, loop_targets=loop_targets)
            return
        if isinstance(stmt, ast.Assign):
            classification = classify_loop_semantic_statement(stmt, loop_targets=loop_targets)
            if bool(classification.get("matched")):
                add_span(
                    "loop_update_expression",
                    stmt.value,
                    expression_kind=str(classification.get("semantic_kind") or expression_kind_for_ast(stmt.value)),
                    context=safe_unparse(stmt),
                    loop_targets=loop_targets,
                )
            else:
                skip(str(classification.get("reason") or "non_semantic_assignment_expression"))
            return
        if isinstance(stmt, ast.AnnAssign):
            classification = classify_loop_semantic_statement(stmt, loop_targets=loop_targets)
            if bool(classification.get("matched")):
                add_span(
                    "loop_update_expression",
                    stmt.value,
                    expression_kind=str(classification.get("semantic_kind") or expression_kind_for_ast(stmt.value)),
                    context=safe_unparse(stmt),
                    loop_targets=loop_targets,
                )
            else:
                skip(str(classification.get("reason") or "non_semantic_annotated_assignment_expression"))
            return
        if isinstance(stmt, ast.AugAssign):
            add_span(
                "loop_update_expression",
                stmt.value,
                expression_kind="augmented_accumulator_rhs",
                context=safe_unparse(stmt),
                loop_targets=loop_targets,
            )
            return
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            classification = classify_loop_semantic_statement(stmt, loop_targets=loop_targets)
            if not bool(classification.get("matched")):
                skip(str(classification.get("reason") or "non_semantic_call_expression"))
                return
            expression_nodes = semantic_call_expression_nodes(call)
            if not expression_nodes:
                skip("semantic_call_without_argument_expression")
                return
            for expr in expression_nodes:
                add_span(
                    "loop_update_expression",
                    expr,
                    expression_kind=str(classification.get("semantic_kind") or expression_kind_for_ast(expr)),
                    context=safe_unparse(stmt),
                    loop_targets=loop_targets,
                )

    for loop in loops:
        loop_targets = ast_store_names(getattr(loop, "target", None))
        if isinstance(loop, ast.While):
            add_span(
                "loop_condition_expression",
                loop.test,
                expression_kind=expression_kind_for_ast(loop.test),
                context=safe_unparse(loop),
                loop_targets=loop_targets,
            )
        for child in list(getattr(loop, "body", []) or []):
            if isinstance(child, ast.stmt):
                visit_loop_stmt(child, loop_targets=loop_targets)

    for stmt in function.body:
        if not isinstance(stmt, ast.Return):
            continue
        classification = classify_top_level_semantic_finalizer(stmt)
        if not bool(classification.get("matched")):
            skip(str(classification.get("reason") or "non_semantic_finalizer_expression"))
            continue
        add_span(
            "top_level_finalizer_expression",
            stmt.value,
            expression_kind=str(classification.get("semantic_kind") or expression_kind_for_ast(stmt.value)),
            context=safe_unparse(stmt),
        )

    if not spans:
        return {
            "matched": False,
            "reason": "no_loop_expression_synthesis_spans",
            "spans": [],
            "skipped_counts": dict(sorted(skipped_counts.items())),
        }
    role_counts: dict[str, int] = {}
    expression_kind_counts: dict[str, int] = {}
    for span in spans:
        role = str(span.get("role") or "unknown")
        expression_kind = str(span.get("expression_kind") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
        expression_kind_counts[expression_kind] = expression_kind_counts.get(expression_kind, 0) + 1
    return {
        "matched": True,
        "policy": "private_loop_expression_synthesis_span_extraction_v1",
        "spans": spans,
        "role_counts": dict(sorted(role_counts.items())),
        "expression_kind_counts": dict(sorted(expression_kind_counts.items())),
        "skipped_counts": dict(sorted(skipped_counts.items())),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def expression_body_tokens(expr: ast.AST | None) -> list[str]:
    text = safe_unparse(expr) if expr is not None else ""
    if not text:
        return []
    tokens = [
        str(token)
        for token in target_tokens(text, target_mode="body_tokens")
        if str(token) not in {"<pad>", "<bos>", "<eos>", "<unk>"}
    ]
    while tokens and tokens[-1] == "NEWLINE:":
        tokens.pop()
    return tokens


def expression_is_plain_loop_identity(expr: ast.AST | None, *, loop_targets: set[str]) -> bool:
    if not loop_targets:
        return False
    if isinstance(expr, ast.Name) and expr.id in loop_targets:
        return True
    return False


def expression_tokens_are_trivial(tokens: list[str]) -> bool:
    values = [str(token) for token in tokens if str(token)]
    if not values:
        return True
    if len(values) == 1 and (values[0].startswith("NAME:") or values[0].startswith("NUMBER:") or values[0].startswith("STRING:")):
        return True
    if all(token.startswith(("OP:[", "OP:]", "OP:(", "OP:)", "OP:{", "OP:}", "OP:,")) for token in values):
        return True
    return False


def semantic_call_expression_nodes(call: ast.Call) -> list[ast.AST]:
    nodes: list[ast.AST] = []
    for nested in ast.walk(call):
        if not isinstance(nested, ast.Call):
            continue
        nodes.extend(list(nested.args or []))
        nodes.extend([keyword.value for keyword in list(nested.keywords or []) if keyword.value is not None])
    return nodes


def expression_kind_for_ast(expr: ast.AST | None) -> str:
    if expr is None:
        return "empty_expression"
    if isinstance(expr, ast.BoolOp):
        return "boolean_expression"
    if isinstance(expr, ast.BinOp):
        return "binary_expression"
    if isinstance(expr, ast.Compare):
        return "comparison_expression"
    if isinstance(expr, ast.Call):
        return "call_expression"
    if isinstance(expr, ast.Subscript):
        return "subscript_expression"
    if isinstance(expr, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return "comprehension_expression"
    if isinstance(expr, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
        return "literal_container_expression"
    if isinstance(expr, ast.IfExp):
        return "conditional_expression"
    if isinstance(expr, ast.UnaryOp):
        return "unary_expression"
    return type(expr).__name__


def semantic_slot_prefix_role(token: str) -> str:
    text = str(token or "")
    if text.startswith("SLOT:PLAN_"):
        return "plan"
    if text.startswith("SLOT:RETURN_SHAPE_"):
        return "return_shape"
    if text.startswith("SLOT:LOOP_SOURCE_"):
        return "loop_source"
    if text.startswith("SLOT:INIT_"):
        return "init"
    if text.startswith("SLOT:UPDATE_"):
        return "update"
    if text.startswith("SLOT:GUARD_"):
        return "guard"
    if text.startswith("SLOT:FINALIZER_"):
        return "finalizer"
    if text.startswith("SLOT:STATE_"):
        return "state"
    if text.startswith("SLOT:BIND_"):
        return "binding"
    if text.startswith("SLOT:STMT_"):
        return "statement"
    return "other"


def parse_role_filter(text: str) -> set[str]:
    return {
        item.strip()
        for item in str(text or "").split(",")
        if item.strip()
    }


def loop_semantic_operation_spans_for_body(body: str) -> dict[str, Any]:
    text = str(body or "")
    try:
        parsed = ast.parse("def _candidate(data, other=None):\n" + "\n".join(f"    {line}" for line in text.splitlines()))
    except SyntaxError as exc:
        return {
            "matched": False,
            "reason": "parse_error",
            "error": str(exc)[:160],
            "spans": [],
            "skipped_counts": {},
        }
    function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    if function is None:
        return {"matched": False, "reason": "missing_function", "spans": [], "skipped_counts": {}}
    loops = [node for node in ast.walk(function) if isinstance(node, (ast.For, ast.While))]
    if not loops:
        return {"matched": False, "reason": "no_loop", "spans": [], "skipped_counts": {}}
    spans: list[dict[str, Any]] = []
    skipped_counts: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped_counts[reason] = skipped_counts.get(reason, 0) + 1

    for loop in loops:
        loop_targets = ast_store_names(getattr(loop, "target", None))
        for stmt in semantic_candidate_statements_from_loop(loop):
            classification = classify_loop_semantic_statement(stmt, loop_targets=loop_targets)
            if not bool(classification.get("matched")):
                skip(str(classification.get("reason") or "non_semantic_loop_statement"))
                continue
            tokens = body_statement_tokens(stmt)
            if not tokens:
                skip("empty_loop_semantic_statement_tokens")
                continue
            spans.append(
                {
                    "role": "loop_semantic_update",
                    "semantic_kind": str(classification.get("semantic_kind") or "loop_semantic_update"),
                    "tokens": tokens,
                    "excluded_positive_tokens": sorted(loop_statement_positive_excluded_tokens(stmt)),
                    "statement": safe_unparse(stmt)[:240],
                }
            )

    for stmt in function.body:
        if not isinstance(stmt, ast.Return):
            continue
        classification = classify_top_level_semantic_finalizer(stmt)
        if not bool(classification.get("matched")):
            skip(str(classification.get("reason") or "non_semantic_finalizer"))
            continue
        tokens = body_statement_tokens(stmt)
        if not tokens:
            skip("empty_top_level_semantic_finalizer_tokens")
            continue
        spans.append(
            {
                "role": "top_level_semantic_finalizer",
                "semantic_kind": str(classification.get("semantic_kind") or "top_level_semantic_finalizer"),
                "tokens": tokens,
                "excluded_positive_tokens": [],
                "statement": safe_unparse(stmt)[:240],
            }
        )

    if not spans:
        return {
            "matched": False,
            "reason": "no_semantic_operation_spans",
            "spans": [],
            "skipped_counts": dict(sorted(skipped_counts.items())),
        }
    role_counts: dict[str, int] = {}
    semantic_kind_counts: dict[str, int] = {}
    for span in spans:
        role = str(span.get("role") or "unknown")
        semantic_kind = str(span.get("semantic_kind") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
        semantic_kind_counts[semantic_kind] = semantic_kind_counts.get(semantic_kind, 0) + 1
    return {
        "matched": True,
        "policy": "private_loop_semantic_operation_span_extraction_v1",
        "spans": spans,
        "role_counts": dict(sorted(role_counts.items())),
        "semantic_kind_counts": dict(sorted(semantic_kind_counts.items())),
        "skipped_counts": dict(sorted(skipped_counts.items())),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def semantic_candidate_statements_from_loop(loop: ast.AST) -> list[ast.stmt]:
    statements: list[ast.stmt] = []

    def visit_stmt(stmt: ast.stmt) -> None:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr)):
            statements.append(stmt)
        for attr in ("body", "orelse", "finalbody"):
            for child in list(getattr(stmt, attr, []) or []):
                if isinstance(child, ast.stmt):
                    visit_stmt(child)
        handlers = list(getattr(stmt, "handlers", []) or [])
        for handler in handlers:
            for child in list(getattr(handler, "body", []) or []):
                if isinstance(child, ast.stmt):
                    visit_stmt(child)

    for child in list(getattr(loop, "body", []) or []):
        if isinstance(child, ast.stmt):
            visit_stmt(child)
    return statements


def classify_loop_semantic_statement(stmt: ast.stmt, *, loop_targets: set[str]) -> dict[str, Any]:
    if isinstance(stmt, ast.AugAssign):
        return {"matched": True, "semantic_kind": "augmented_accumulator_update"}
    if isinstance(stmt, ast.Assign):
        if value_is_plain_loop_identity(stmt.value, loop_targets=loop_targets):
            return {"matched": False, "reason": "plain_loop_identity_assignment"}
        if value_has_semantic_operation(stmt.value):
            return {"matched": True, "semantic_kind": "assignment_transform_update"}
        return {"matched": False, "reason": "plain_assignment"}
    if isinstance(stmt, ast.AnnAssign):
        if value_is_plain_loop_identity(stmt.value, loop_targets=loop_targets):
            return {"matched": False, "reason": "plain_loop_identity_assignment"}
        if value_has_semantic_operation(stmt.value):
            return {"matched": True, "semantic_kind": "annotated_assignment_transform_update"}
        return {"matched": False, "reason": "plain_annotated_assignment"}
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
        if call_is_shallow_loop_identity_accumulation(call, loop_targets=loop_targets):
            return {"matched": False, "reason": "shallow_loop_identity_accumulation_excluded"}
        semantic_kind = semantic_mutation_call_kind(call)
        if semantic_kind:
            return {"matched": True, "semantic_kind": semantic_kind}
        return {"matched": False, "reason": "non_semantic_expression_call"}
    return {"matched": False, "reason": "unsupported_statement_type"}


def classify_top_level_semantic_finalizer(stmt: ast.Return) -> dict[str, Any]:
    value = stmt.value
    if value is None:
        return {"matched": False, "reason": "empty_return"}
    if isinstance(value, ast.Name):
        return {"matched": False, "reason": "direct_local_return"}
    if isinstance(value, ast.Constant):
        return {"matched": False, "reason": "constant_return"}
    if value_has_semantic_operation(value) or isinstance(value, (ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp)):
        return {"matched": True, "semantic_kind": "transform_finalizer"}
    return {"matched": False, "reason": "plain_finalizer"}


def value_is_plain_loop_identity(value: ast.AST | None, *, loop_targets: set[str]) -> bool:
    return isinstance(value, ast.Name) and value.id in loop_targets


def value_has_semantic_operation(value: ast.AST | None) -> bool:
    if value is None:
        return False
    semantic_nodes = (
        ast.BinOp,
        ast.BoolOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Call,
        ast.Subscript,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.IfExp,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Set,
    )
    return any(isinstance(node, semantic_nodes) for node in ast.walk(value))


def semantic_mutation_call_kind(call: ast.Call) -> str:
    attr = call_attribute_name(call)
    if attr in {"append", "add"}:
        return "projected_append_or_add_update"
    if attr in {"extend", "update", "setdefault"}:
        return "collection_merge_or_mapping_update"
    if attr in {"pop", "remove", "discard", "clear"}:
        return "collection_removal_update"
    if nested_call_attribute_names(call) & {"append", "add", "extend", "update", "setdefault", "pop"}:
        return "nested_mutation_call_update"
    return ""


def call_is_shallow_loop_identity_accumulation(call: ast.Call, *, loop_targets: set[str]) -> bool:
    attr = call_attribute_name(call)
    if attr not in {"append", "add"} or len(call.args) != 1:
        return False
    arg = call.args[0]
    return isinstance(arg, ast.Name) and arg.id in loop_targets


def call_attribute_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Attribute):
        return str(call.func.attr or "")
    return ""


def nested_call_attribute_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {
        str(child.func.attr)
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
    }


def loop_statement_action_spans_for_body(body: str) -> dict[str, Any]:
    text = str(body or "")
    try:
        parsed = ast.parse("def _candidate(data, other=None):\n" + "\n".join(f"    {line}" for line in text.splitlines()))
    except SyntaxError as exc:
        return {
            "matched": False,
            "reason": "parse_error",
            "error": str(exc)[:160],
            "spans": [],
        }
    function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    if function is None:
        return {"matched": False, "reason": "missing_function", "spans": []}
    loops = [node for node in ast.walk(function) if isinstance(node, (ast.For, ast.While))]
    if not loops:
        return {"matched": False, "reason": "no_loop", "spans": []}
    spans: list[dict[str, Any]] = []
    skipped_counts: dict[str, int] = {}
    for loop in loops:
        for stmt in list(getattr(loop, "body", []) or []):
            if isinstance(stmt, ast.Return):
                skipped_counts["loop_body_return"] = skipped_counts.get("loop_body_return", 0) + 1
                continue
            if isinstance(stmt, (ast.Continue, ast.Break)):
                skipped_counts["bare_loop_control_flow_terminal"] = skipped_counts.get("bare_loop_control_flow_terminal", 0) + 1
                continue
            role = "loop_body_decision" if isinstance(stmt, ast.If) else "loop_body_update"
            tokens = body_statement_tokens(stmt)
            if tokens:
                excluded = sorted(loop_statement_positive_excluded_tokens(stmt))
                spans.append(
                    {
                        "role": role,
                        "tokens": tokens,
                        "excluded_positive_tokens": excluded,
                        "statement": safe_unparse(stmt)[:240],
                    }
                )
    for stmt in function.body:
        if isinstance(stmt, ast.Return):
            tokens = body_statement_tokens(stmt)
            if tokens:
                spans.append({"role": "top_level_finalizer", "tokens": tokens, "statement": safe_unparse(stmt)[:240]})
    if not spans:
        return {"matched": False, "reason": "no_action_spans", "spans": []}
    role_counts: dict[str, int] = {}
    for span in spans:
        role = str(span.get("role") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
    return {
        "matched": True,
        "policy": "private_loop_statement_action_span_extraction_v1",
        "spans": spans,
        "role_counts": dict(sorted(role_counts.items())),
        "skipped_counts": dict(sorted(skipped_counts.items())),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def loop_statement_positive_excluded_tokens(stmt: ast.stmt) -> set[str]:
    excluded: set[str] = set()
    for node in ast.walk(stmt):
        if isinstance(node, ast.Continue):
            excluded.add("NAME:continue")
        elif isinstance(node, ast.Break):
            excluded.add("NAME:break")
        elif isinstance(node, ast.Return):
            excluded.add("NAME:return")
    return excluded


def body_statement_tokens(stmt: ast.stmt) -> list[str]:
    text = safe_unparse(stmt)
    if not text:
        return []
    return [
        str(token)
        for token in target_tokens(text, target_mode="body_tokens")
        if str(token) not in {"<pad>", "<bos>", "<eos>", "<unk>"}
    ]


def safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node).strip()
    except Exception:
        return ""


def find_subsequence_positions(haystack: list[str], needle: list[str]) -> list[int]:
    if not haystack or not needle or len(needle) > len(haystack):
        return []
    out: list[int] = []
    width = len(needle)
    for start in range(0, len(haystack) - width + 1):
        if haystack[start : start + width] == needle:
            out.append(start)
    return out


def loop_operation_tokens_for_body(body: str) -> dict[str, Any]:
    try:
        parsed = ast.parse("def _candidate(data, other=None):\n" + "\n".join(f"    {line}" for line in str(body or "").splitlines()))
    except SyntaxError as exc:
        return {
            "matched": False,
            "reason": "parse_error",
            "error": str(exc)[:160],
            "tokens": [],
            "operation_counts": {},
        }
    function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    if function is None:
        return {
            "matched": False,
            "reason": "missing_function",
            "tokens": [],
            "operation_counts": {},
        }
    tokens: set[str] = {
        "NAME:return",
        "NEWLINE:",
    }
    operation_counts: dict[str, int] = {}

    def add_token(token: str) -> None:
        if token:
            tokens.add(token)

    def add_name(name: str) -> None:
        if str(name).isidentifier():
            add_token(f"NAME:{name}")

    def add_operation(name: str) -> None:
        operation_counts[name] = operation_counts.get(name, 0) + 1

    for node in ast.walk(function):
        if isinstance(node, ast.For):
            add_operation("for_loop")
            add_token("NAME:for")
            add_token("NAME:in")
            add_token("OP::")
            add_token("INDENT:")
            add_token("DEDENT:")
            for name in ast_store_names(node.target) | ast_load_names(node.iter):
                add_name(name)
        elif isinstance(node, ast.While):
            add_operation("while_loop")
            add_token("NAME:while")
            add_token("OP::")
            add_token("INDENT:")
            add_token("DEDENT:")
            for name in ast_load_names(node.test):
                add_name(name)
        elif isinstance(node, ast.If):
            add_operation("branch")
            add_token("NAME:if")
            add_token("OP::")
            add_token("INDENT:")
            add_token("DEDENT:")
            if node.orelse:
                add_token("NAME:else")
            for name in ast_load_names(node.test):
                add_name(name)
        elif isinstance(node, ast.Assign):
            add_operation("assignment")
            add_token("OP:=")
            for target in node.targets:
                for name in ast_store_names(target):
                    add_name(name)
            for name in ast_load_names(node.value):
                add_name(name)
        elif isinstance(node, ast.AugAssign):
            add_operation("augmented_update")
            add_token(augassign_token(node.op))
            for name in ast_store_names(node.target) | ast_load_names(node.value):
                add_name(name)
        elif isinstance(node, ast.Return):
            add_operation("finalizer_return")
            add_token("NAME:return")
            for name in ast_load_names(node.value):
                add_name(name)
        elif isinstance(node, ast.Call):
            add_operation("call")
            add_token("OP:(")
            add_token("OP:)")
            if isinstance(node.func, ast.Attribute):
                add_token("OP:.")
                add_name(node.func.attr)
                for name in ast_load_names(node.func.value):
                    add_name(name)
            elif isinstance(node.func, ast.Name):
                add_name(node.func.id)
            for arg in node.args:
                for name in ast_load_names(arg):
                    add_name(name)
        elif isinstance(node, ast.Compare):
            add_operation("comparison")
            for op in node.ops:
                add_token(compare_token(op))
            for name in ast_load_names(node.left):
                add_name(name)
            for comparator in node.comparators:
                for name in ast_load_names(comparator):
                    add_name(name)
        elif isinstance(node, ast.BoolOp):
            add_operation("bool_op")
            add_token("NAME:and" if isinstance(node.op, ast.And) else "NAME:or")
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            add_operation("not_op")
            add_token("NAME:not")
        elif isinstance(node, ast.BinOp):
            add_operation("binary_op")
            add_token(binop_token(node.op))

    meaningful = any(name in operation_counts for name in ["for_loop", "while_loop", "branch", "call", "comparison", "augmented_update"])
    if not meaningful:
        return {
            "matched": False,
            "reason": "no_loop_or_operation_body",
            "tokens": sorted(tokens),
            "operation_counts": dict(sorted(operation_counts.items())),
        }
    return {
        "matched": True,
        "policy": "private_loop_operation_ast_token_extraction_v1",
        "tokens": sorted(tokens),
        "operation_counts": dict(sorted(operation_counts.items())),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def ast_load_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)}


def ast_store_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)}


def compare_token(op: ast.cmpop) -> str:
    if isinstance(op, ast.Eq):
        return "OP:=="
    if isinstance(op, ast.NotEq):
        return "OP:!="
    if isinstance(op, ast.Lt):
        return "OP:<"
    if isinstance(op, ast.LtE):
        return "OP:<="
    if isinstance(op, ast.Gt):
        return "OP:>"
    if isinstance(op, ast.GtE):
        return "OP:>="
    if isinstance(op, ast.In):
        return "NAME:in"
    if isinstance(op, ast.NotIn):
        return "NAME:not"
    if isinstance(op, ast.Is):
        return "NAME:is"
    if isinstance(op, ast.IsNot):
        return "NAME:is"
    return ""


def binop_token(op: ast.operator) -> str:
    if isinstance(op, ast.Add):
        return "OP:+"
    if isinstance(op, ast.Sub):
        return "OP:-"
    if isinstance(op, ast.Mult):
        return "OP:*"
    if isinstance(op, ast.Div):
        return "OP:/"
    if isinstance(op, ast.FloorDiv):
        return "OP://"
    if isinstance(op, ast.Mod):
        return "OP:%"
    if isinstance(op, ast.Pow):
        return "OP:**"
    return ""


def augassign_token(op: ast.operator) -> str:
    token = binop_token(op)
    if token.startswith("OP:"):
        return f"{token}="
    return "OP:+="
