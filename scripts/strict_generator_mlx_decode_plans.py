#!/usr/bin/env python3
"""Task-blind strict MLX decode plan, condition, and trace helpers.

This module owns source-condition adequacy, learned plan-prefix adequacy,
loop-plan exploration, expression closure, and AST action-trace summaries for
`strict_generator_mlx_decode_eval.py`. It does not train, render templates,
inspect tests/solutions, use public benchmark payloads, or grant learned-code
generation credit.
"""

from __future__ import annotations

import ast
from collections import Counter
from typing import Any

from neural_seed_code_proposer_comparator import dict_or_empty
from neural_seed_token_decoder_comparator import decode_beam_sort_key, final_decode_beam_sort_key
from neural_seed_token_decoder_support import (
    PLAN_BODY_START_TOKEN,
    body_like_target_mode,
    body_tokens_for_target_mode,
    current_line_tokens,
    decode_body_tokens,
    learned_plan_prefix_target_mode,
    split_learned_plan_prefix_tokens,
    token_allowed_by_policy,
    token_values,
)
from neural_seed_decode_static_guard import (
    action_trace_call_name,
    decode_guard_return_dependency_summary,
    expression_load_names,
    expression_store_names,
    parsed_decode_guard_function,
    return_value_is_none_like,
    static_guard_candidate_code,
)
from neural_seed_expression_value_guard import (
    EXPRESSION_VALUE_BARE_BUILTINS,
    expression_value_has_bare_builtin_value,
    expression_value_quality_summary,
)
from strict_generator_mlx_decode_guards import (
    first_call_argument_values,
    isinstance_first_arg_values_invalid,
    prefix_lines_with_depth,
    prefix_mentions_allowed_parameter,
    token_blocked_by_strict_decode_guard,
)

LOOP_PLAN_MUTATION_METHODS = {
    "append",
    "add",
    "extend",
    "update",
    "setdefault",
    "discard",
    "remove",
    "pop",
}


def beam_mentions_allowed_parameter(
    row: dict[str, Any],
    inverse: dict[int, str],
    *,
    allowed_names: set[str],
    target_mode: str,
) -> bool:
    decoded_tokens = [inverse.get(int(idx), "<unk>") for idx in list(row.get("generated") or [])[1:]]
    body_prefix = body_tokens_for_target_mode(decoded_tokens, target_mode=target_mode)
    return prefix_mentions_allowed_parameter(body_prefix, allowed_names=allowed_names)


def visible_parameter_exploration_choices(
    arr: Any,
    inverse: dict[int, str],
    prefix: list[str],
    *,
    allowed_names: set[str] | None,
    seen: set[int],
    token_policy: str,
    require_parameter_use: bool,
    input_type_hints: dict[str, str] | None,
) -> list[tuple[int, float]]:
    if not require_parameter_use:
        return []
    names = sorted(name for name in set(allowed_names or set()) if name)
    if not names or prefix_mentions_allowed_parameter(prefix, allowed_names=set(names)):
        return []
    token_to_id = {text: int(idx) for idx, text in inverse.items()}
    choices: list[tuple[int, float]] = []
    for name in names:
        token_text = f"NAME:{name}"
        idx = token_to_id.get(token_text)
        if idx is None or idx in seen:
            continue
        if token_blocked_by_strict_decode_guard(
            prefix,
            token_text,
            require_nontrivial_return=require_parameter_use,
            allowed_names=set(names),
            input_type_hints=input_type_hints,
        ):
            continue
        if not token_allowed_by_policy(prefix, token_text, policy=token_policy, allowed_names=set(names)):
            continue
        seen.add(idx)
        choices.append((idx, float(max(arr[idx], 1e-9))))
    return choices


def learned_prefix_decision_expectation_from_tokens(prefix: list[str]) -> dict[str, Any]:
    prefix_tokens = [str(tok) for tok in prefix if str(tok).startswith("SLOT:")]
    if PLAN_BODY_START_TOKEN not in prefix_tokens:
        return {"enabled": False, "policy": "learned_prefix_decision_expectation_v1"}
    truthiness_arg = ""
    sequence_arg = ""
    default_arg = ""
    guarded_head_arg = ""
    for token in prefix_tokens:
        if token.startswith("SLOT:COND_TRUTHY_ARG_"):
            truthiness_arg = token.removeprefix("SLOT:COND_TRUTHY_ARG_").lower()
        elif token.startswith("SLOT:COND_SEQUENCE_ARG_"):
            sequence_arg = token.removeprefix("SLOT:COND_SEQUENCE_ARG_").lower()
        elif token.startswith("SLOT:RETURN_DEFAULT_ARG_"):
            default_arg = token.removeprefix("SLOT:RETURN_DEFAULT_ARG_").lower()
        elif token.startswith("SLOT:RETURN_GUARDED_HEAD_ARG_"):
            guarded_head_arg = token.removeprefix("SLOT:RETURN_GUARDED_HEAD_ARG_").lower()
    main_arg = truthiness_arg or sequence_arg or guarded_head_arg
    if not main_arg:
        return {"enabled": False, "policy": "learned_prefix_decision_expectation_v1"}
    return {
        "enabled": True,
        "policy": "learned_prefix_decision_expectation_v1",
        "source": "model_generated_prefix_tokens",
        "truthiness_arg": main_arg,
        "default_arg": default_arg,
        "requires_truthiness_guard": bool(truthiness_arg),
        "requires_sequence_type_guard": bool(sequence_arg),
        "requires_guarded_first_item_return": bool(guarded_head_arg),
        "requires_default_return": bool(default_arg),
        "expected_sequence_types": ["list", "tuple"] if sequence_arg else [],
        "candidate_generation_credit": 0,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
    }


def source_condition_exploration_choices(
    arr: Any,
    inverse: dict[int, str],
    prefix: list[str],
    *,
    expectation: dict[str, Any],
    seen: set[int],
    token_policy: str,
    allowed_names: set[str] | None,
    input_type_hints: dict[str, str] | None,
    enabled: bool,
    enable_operation_value_construction: bool = False,
) -> list[tuple[int, float]]:
    if not enabled or not bool(expectation.get("enabled")):
        return []
    token_to_id = {text: int(idx) for idx, text in inverse.items()}
    values = tuple(token_values(current_line_tokens(prefix)))
    choices: list[tuple[int, float]] = []

    def add_token(token_text: str) -> None:
        idx = token_to_id.get(token_text)
        if idx is None or idx in seen:
            return
        if token_blocked_by_strict_decode_guard(
            prefix,
            token_text,
            require_nontrivial_return=False,
            allowed_names=allowed_names,
            input_type_hints=input_type_hints,
        ):
            return
        if not token_allowed_by_policy(prefix, token_text, policy=token_policy, allowed_names=allowed_names):
            return
        seen.add(idx)
        choices.append((idx, float(max(arr[idx], 1e-9))))

    truthiness_arg = str(expectation.get("truthiness_arg") or "")
    default_arg = str(expectation.get("default_arg") or "")
    operation_tags = {str(item) for item in list(expectation.get("operation_tags") or []) if str(item)}
    _lines, current_depth, current_values = prefix_lines_with_depth(prefix)
    progress = source_condition_prefix_progress(prefix, expectation)
    if operation_tags:
        body_text = decode_body_tokens(prefix)
        accumulator = loop_plan_first_assigned_local(body_text)
        loop_target = source_condition_prefix_loop_target(prefix)
        if current_depth > 0 and loop_plan_has_update_call(body_text) and not values:
            return []
        for token_text in source_condition_operation_value_prefix_tokens(
            values,
            operation_tags=operation_tags,
            loop_target=loop_target,
            default_arg=default_arg,
            allowed_names=set(allowed_names or set()),
            enabled=enable_operation_value_construction,
        ):
            add_token(token_text)
        if choices:
            return choices
        for token_text in source_condition_operation_prefix_tokens(
            values,
            operation_tags=operation_tags,
            accumulator=accumulator,
            loop_target=loop_target,
            default_arg=default_arg,
            inside_loop=current_depth > 0,
            has_operation_evidence=bool(progress.get("has_operation_evidence")),
        ):
            add_token(token_text)
        for token_text in source_condition_operation_exploration_tokens(
            values,
            operation_tags=operation_tags,
            loop_target=loop_target,
            default_arg=default_arg,
        ):
            add_token(token_text)
        if choices:
            return choices
    if (
        truthiness_arg
        and bool(expectation.get("requires_truthiness_guard"))
        and current_depth <= 0
        and not values
        and not progress.get("has_truthiness_guard")
    ):
        add_token("NAME:if")
    elif truthiness_arg and bool(expectation.get("requires_truthiness_guard")) and values == ("if",):
        add_token("NAME:isinstance")
    elif truthiness_arg and bool(expectation.get("requires_truthiness_guard")) and values == ("if", "isinstance"):
        add_token("OP:(")
    elif truthiness_arg and bool(expectation.get("requires_truthiness_guard")) and values == ("if", "isinstance", "("):
        add_token(f"NAME:{truthiness_arg}")
    elif truthiness_arg and bool(expectation.get("requires_truthiness_guard")) and values == ("if", "isinstance", "(", truthiness_arg):
        add_token("OP:,")
    if truthiness_arg and bool(expectation.get("requires_sequence_type_guard")):
        if values == ("if", "isinstance", "(", truthiness_arg, ","):
            add_token("OP:(")
        elif values == ("if", "isinstance", "(", truthiness_arg, ",", "("):
            add_token("NAME:list")
        elif values == ("if", "isinstance", "(", truthiness_arg, ",", "(", "list"):
            add_token("OP:,")
        elif values == ("if", "isinstance", "(", truthiness_arg, ",", "(", "list", ","):
            add_token("NAME:tuple")
        elif values == ("if", "isinstance", "(", truthiness_arg, ",", "(", "list", ",", "tuple"):
            add_token("OP:)")
        elif values == ("if", "isinstance", "(", truthiness_arg, ",", "(", "list", ",", "tuple", ")"):
            add_token("OP:)")
    if truthiness_arg and bool(expectation.get("requires_truthiness_guard")):
        if values[-1:] == ("and",):
            add_token(f"NAME:{truthiness_arg}")
        elif source_condition_prefix_can_add_truthiness(values, truthiness_arg):
            add_token("NAME:and")
    if (
        truthiness_arg
        and current_depth > 0
        and bool(expectation.get("requires_first_item_return"))
        and not values
        and not progress.get("has_guarded_first_item_return")
    ):
        add_token("NAME:return")
    if (
        truthiness_arg
        and current_depth > 0
        and bool(expectation.get("requires_first_item_return"))
        and values == ("return", truthiness_arg)
    ):
        add_token("OP:[")
    elif (
        truthiness_arg
        and current_depth > 0
        and bool(expectation.get("requires_first_item_return"))
        and values == ("return", truthiness_arg, "[")
    ):
        add_token("NUMBER:0")
    elif (
        truthiness_arg
        and current_depth > 0
        and bool(expectation.get("requires_first_item_return"))
        and values == ("return", truthiness_arg, "[", "0")
    ):
        add_token("OP:]")
    elif (
        truthiness_arg
        and current_depth > 0
        and bool(expectation.get("requires_first_item_return"))
        and values == ("return", truthiness_arg, "[", "0", "]")
    ):
        add_token("NEWLINE:")
    elif values == ("return",) and truthiness_arg and current_depth > 0:
        add_token(f"NAME:{truthiness_arg}")
    elif (
        default_arg
        and bool(expectation.get("requires_default_return"))
        and current_depth > 0
        and not current_values
        and progress.get("has_guarded_first_item_return")
        and not progress.get("has_default_return")
    ):
        add_token("DEDENT:")
    elif (
        default_arg
        and bool(expectation.get("requires_default_return"))
        and current_depth <= 0
        and not values
        and not progress.get("has_default_return")
        and (
            not bool(expectation.get("requires_truthiness_guard"))
            or progress.get("has_guarded_first_item_return")
            or progress.get("has_truthiness_guard")
        )
    ):
        add_token("NAME:return")
    elif default_arg and bool(expectation.get("requires_default_return")) and values == ("return",):
        add_token(f"NAME:{default_arg}")
    elif (
        default_arg
        and bool(expectation.get("requires_default_return"))
        and values == ("return", default_arg)
    ):
        add_token("NEWLINE:")
    return choices


def expression_closure_guard_choices(
    arr: Any,
    inverse: dict[int, str],
    prefix: list[str],
    *,
    expectation: dict[str, Any],
    seen: set[int],
    token_policy: str,
    allowed_names: set[str] | None,
    input_type_hints: dict[str, str] | None,
    require_nontrivial_return: bool,
    block_shallow_identity_update: bool,
    enabled: bool,
    enable_expression_value_guard: bool = False,
) -> list[tuple[int, float]]:
    if not enabled or not prefix:
        return []
    token_to_id = {text: int(idx) for idx, text in inverse.items()}
    line_tokens = current_line_tokens(prefix)
    values = token_values(line_tokens)
    body_text = decode_body_tokens(prefix)
    choices: list[tuple[int, float]] = []

    def add_token(token_text: str) -> None:
        idx = token_to_id.get(token_text)
        if idx is None or idx in seen:
            return
        if token_blocked_by_strict_decode_guard(
            prefix,
            token_text,
            require_nontrivial_return=require_nontrivial_return,
            allowed_names=allowed_names,
            input_type_hints=input_type_hints,
        ):
            return
        if token_blocked_by_loop_plan(
            prefix,
            token_text,
            expectation=expectation,
            block_shallow_identity_update=block_shallow_identity_update,
        ):
            return
        if token_blocked_by_expression_value_guard(
            prefix,
            token_text,
            expectation=expectation,
            enabled=enable_expression_value_guard,
        ):
            return
        if not token_allowed_by_policy(prefix, token_text, policy=token_policy, allowed_names=allowed_names):
            return
        seen.add(idx)
        choices.append((idx, float(max(arr[idx], 1e-9))))

    if values:
        if expression_value_inside_update_call(values) and current_line_tail_needs_operand(values):
            loop_var = loop_plan_loop_var_name(expectation, body_text=body_text, inverse=inverse, arr=arr)
            if loop_var:
                add_token(f"NAME:{loop_var}")
            if choices:
                return choices
        expected_close = current_line_expected_closer(values)
        if expected_close and not current_line_tail_needs_operand(values):
            add_token(f"OP:{expected_close}")
            return choices
        if current_line_needs_colon_from_values(values):
            add_token("OP::")
            return choices
        if expression_closure_line_can_end(values):
            add_token("NEWLINE:")
            return choices
        return []

    _lines, current_depth, _current_values = prefix_lines_with_depth(prefix)
    if current_depth > 0 and expression_closure_can_dedent(prefix, body_text=body_text, expectation=expectation):
        add_token("DEDENT:")
        return choices

    accumulator = loop_plan_first_assigned_local(body_text)
    if current_depth <= 0 and accumulator and not loop_plan_has_local_return(body_text, accumulator=accumulator):
        add_token("NAME:return")
    return choices


def direct_local_return_continuation_choices(
    arr: Any,
    inverse: dict[int, str],
    prefix: list[str],
    *,
    seen: set[int],
    token_policy: str,
    allowed_names: set[str] | None,
    input_type_hints: dict[str, str] | None,
    require_nontrivial_return: bool,
    enabled: bool,
) -> list[tuple[int, float]]:
    """Expose task-blind finalizer tokens for already-generated local state.

    This is not a body renderer and not a fallback return. It can only continue
    a current ``return`` line or start one for a local that the model already
    bound at top level from visible signature data, or for a local accumulator
    the model already updated while looping over visible signature data. It never
    inspects tests, solutions, target bodies, benchmark labels, or verifier
    outcomes.
    """

    if not enabled:
        return []
    token_to_id = {text: int(idx) for idx, text in inverse.items()}
    values = token_values(current_line_tokens(prefix))
    _lines, current_depth, _current_values = prefix_lines_with_depth(prefix)
    if current_depth > 0:
        return []
    bound_locals = list(
        dict.fromkeys(
            [
                *direct_return_parameter_dependent_locals(prefix, allowed_names=allowed_names),
                *direct_return_generated_state_locals(prefix, allowed_names=allowed_names),
                *direct_return_visible_updated_state_locals(prefix, allowed_names=allowed_names),
            ]
        )
    )
    if not bound_locals:
        return []

    choices: list[tuple[int, float]] = []

    def add_token(token_text: str) -> None:
        idx = token_to_id.get(token_text)
        if idx is None or idx in seen:
            return
        if token_blocked_by_strict_decode_guard(
            prefix,
            token_text,
            require_nontrivial_return=require_nontrivial_return,
            allowed_names=allowed_names,
            input_type_hints=input_type_hints,
        ):
            return
        if not token_allowed_by_policy(prefix, token_text, policy=token_policy, allowed_names=allowed_names):
            return
        seen.add(idx)
        choices.append((idx, float(max(arr[idx], 1e-9))))

    if not values:
        add_token("NAME:return")
    elif values == ["return"]:
        for name in bound_locals[:4]:
            add_token(f"NAME:{name}")
    elif len(values) == 2 and values[0] == "return" and values[1] in set(bound_locals):
        add_token("NEWLINE:")
    return choices


def direct_return_generated_state_locals(prefix: list[str], *, allowed_names: set[str] | None) -> list[str]:
    """Return model-created accumulator locals that are visibly stateful.

    The strict decoder often reaches a valid stateful prefix such as
    ``out = []`` plus ``for item in data: out.append(item)`` and then starves at
    the top-level return expression. This helper allows only that final local
    name after the generated body itself proves state and visible-input flow:
    a top-level local exists, a mutation/update exists, and the loop iterates
    over a visible callable argument. It does not choose the loop, update, or
    algorithm.
    """

    visible = {str(name) for name in set(allowed_names or set()) if str(name).isidentifier()}
    if not visible:
        return []
    body_text = decode_body_tokens(prefix)
    function = parsed_decode_guard_function(body_text, allowed_names=None)
    if function is not None and function_has_top_level_valued_return(function):
        return []
    accumulator = loop_plan_first_assigned_local(body_text)
    if not accumulator or accumulator in visible or not accumulator.isidentifier():
        return []
    if loop_plan_has_local_return(body_text, accumulator=accumulator):
        return []
    if not loop_plan_has_update_call(body_text):
        return []
    if not any(loop_plan_has_loop_over_source(body_text, source_arg=name) for name in visible):
        return []
    return [accumulator]


def direct_return_visible_updated_state_locals(prefix: list[str], *, allowed_names: set[str] | None) -> list[str]:
    """Return model-created locals updated from visible input/control flow.

    This is a finalizer-only helper for prefixes such as ``out = []`` followed
    by generated statements like ``if data: out.append(0)`` or
    ``out.append(data)``. It does not pick the local, mutation, branch, loop, or
    value; those must already exist in the generated prefix. It refuses to add a
    second top-level return and only returns locals that were definitely bound
    at top level before visible-input-dependent updates.
    """

    visible = {str(name) for name in set(allowed_names or set()) if str(name).isidentifier()}
    if not visible:
        return []
    body_text = decode_body_tokens(prefix)
    function = parsed_decode_guard_function(body_text, allowed_names=None)
    if function is None or function_has_top_level_valued_return(function):
        return []

    top_level_locals = top_level_bound_locals(function, visible=visible)
    if not top_level_locals:
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def remember(name: str) -> None:
        if name in top_level_locals and name not in seen:
            seen.add(name)
            candidates.append(name)

    def statement_visible_names(stmt: ast.AST | None) -> set[str]:
        return expression_load_names(stmt) & visible if stmt is not None else set()

    def visit_statements(statements: list[ast.stmt], *, visible_control: bool) -> None:
        for stmt in statements:
            if isinstance(stmt, ast.If):
                nested_visible = visible_control or bool(statement_visible_names(stmt.test))
                visit_statements(list(stmt.body or []), visible_control=nested_visible)
                visit_statements(list(stmt.orelse or []), visible_control=nested_visible)
                continue
            if isinstance(stmt, (ast.For, ast.While)):
                control_expr = stmt.iter if isinstance(stmt, ast.For) else stmt.test
                nested_visible = visible_control or bool(statement_visible_names(control_expr))
                visit_statements(list(stmt.body or []), visible_control=nested_visible)
                visit_statements(list(getattr(stmt, "orelse", []) or []), visible_control=nested_visible)
                continue
            if isinstance(stmt, ast.AugAssign):
                targets = expression_store_names(stmt.target)
                if visible_control or bool(statement_visible_names(stmt.value)):
                    for name in targets:
                        remember(name)
                continue
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                value = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
                if visible_control or bool(statement_visible_names(value)):
                    targets: set[str] = set()
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            targets.update(expression_store_names(target))
                    else:
                        targets.update(expression_store_names(stmt.target))
                    for name in targets:
                        remember(name)
            for child in ast.walk(stmt):
                if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
                    continue
                method = str(child.func.attr or "")
                if method not in LOOP_PLAN_MUTATION_METHODS:
                    continue
                receiver_names = expression_load_names(child.func.value)
                if not receiver_names & top_level_locals:
                    continue
                argument_visible = any(bool(statement_visible_names(arg)) for arg in list(child.args or []))
                keyword_visible = any(bool(statement_visible_names(keyword.value)) for keyword in list(child.keywords or []))
                if not (visible_control or argument_visible or keyword_visible):
                    continue
                for name in sorted(receiver_names & top_level_locals):
                    remember(name)

    visit_statements(list(function.body or []), visible_control=False)
    return candidates


def top_level_bound_locals(function: ast.FunctionDef, *, visible: set[str]) -> set[str]:
    out: set[str] = set()
    for stmt in function.body:
        targets: set[str] = set()
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                targets.update(expression_store_names(target))
        elif isinstance(stmt, ast.AnnAssign):
            targets.update(expression_store_names(stmt.target))
        elif isinstance(stmt, ast.AugAssign):
            targets.update(expression_store_names(stmt.target))
        for name in targets:
            if name.isidentifier() and name not in visible:
                out.add(name)
    return out


def function_has_top_level_valued_return(function: ast.FunctionDef) -> bool:
    return any(isinstance(stmt, ast.Return) and stmt.value is not None for stmt in function.body)


def direct_return_parameter_dependent_locals(prefix: list[str], *, allowed_names: set[str] | None) -> list[str]:
    """Return top-level locals whose completed assignment mentions visible inputs."""

    visible = {str(name) for name in set(allowed_names or set()) if str(name).isidentifier()}
    if not visible:
        return []
    body_text = decode_body_tokens(prefix)
    function = parsed_decode_guard_function(body_text, allowed_names=None)
    if function is not None and function_has_top_level_valued_return(function):
        return []
    out: list[str] = []
    seen: set[str] = set()
    lines, _current_depth, _current_values = prefix_lines_with_depth(prefix)
    for depth, values in lines:
        if depth != 0 or "=" not in values:
            continue
        equals = values.index("=")
        if equals != 1:
            continue
        name = str(values[0])
        if not name.isidentifier() or name in visible or name in {"return", "for", "if", "while", "else", "elif"}:
            continue
        rhs = [str(item) for item in values[equals + 1 :]]
        if not rhs or rhs in (["["], ["{"], ["("]):
            continue
        rhs_names = {item for item in rhs if item.isidentifier()}
        if not (rhs_names & visible):
            continue
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def current_line_expected_closer(values: list[str]) -> str:
    stack: list[str] = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    closers = {")": "(", "]": "[", "}": "{"}
    for value in values:
        if value in pairs:
            stack.append(value)
        elif value in closers and stack and stack[-1] == closers[value]:
            stack.pop()
    return pairs.get(stack[-1], "") if stack else ""


def current_line_tail_needs_operand(values: list[str]) -> bool:
    if not values:
        return True
    return values[-1] in {
        "(",
        "[",
        "{",
        ",",
        ".",
        "=",
        "+",
        "-",
        "*",
        "/",
        "//",
        "%",
        "**",
        "<",
        "<=",
        ">",
        ">=",
        "==",
        "!=",
        "and",
        "or",
        "not",
        "in",
        "is",
        "for",
        "if",
        "elif",
        "while",
        "return",
    }


def current_line_needs_colon_from_values(values: list[str]) -> bool:
    if not values or ":" in values or current_line_expected_closer(values):
        return False
    first = values[0]
    if first == "for":
        return "in" in values and len(values) >= 4 and not current_line_tail_needs_operand(values)
    if first in {"if", "elif", "while", "except", "with"}:
        return len(values) >= 2 and not current_line_tail_needs_operand(values)
    if first in {"else", "try", "finally"}:
        return len(values) == 1
    return False


def expression_closure_line_can_end(values: list[str]) -> bool:
    if not values or current_line_expected_closer(values) or current_line_tail_needs_operand(values):
        return False
    first = values[0]
    if first in {"if", "elif", "while", "for", "else", "try", "except", "finally", "with"}:
        return ":" in values
    if first in {"return", "break", "continue"}:
        return len(values) >= 1
    if "=" in values and values[-1] != "=":
        return True
    if "." in values and "(" in values and values[-1] == ")":
        return True
    if len(values) >= 2 and values[0].isidentifier() and values[-1] == ")":
        return True
    return False


def expression_closure_can_dedent(prefix: list[str], *, body_text: str, expectation: dict[str, Any]) -> bool:
    if not loop_plan_inside_loop(prefix):
        return False
    accumulator = loop_plan_first_assigned_local(body_text)
    if not accumulator:
        return False
    if loop_plan_has_update_call(body_text):
        return True
    if loop_plan_has_local_return(body_text, accumulator=accumulator):
        return True
    if bool(expectation.get("enabled")) and loop_plan_has_loop_over_source(
        body_text,
        source_arg=loop_plan_source_arg(expectation, allowed_names=None),
    ):
        return loop_plan_has_update_call(body_text)
    return False


def source_condition_priority_prefix(
    prefix: list[str],
    expectation: dict[str, Any],
    *,
    enable_operation_value_construction: bool = False,
    allowed_names: set[str] | None = None,
) -> bool:
    if not bool(expectation.get("enabled")):
        return False
    values = tuple(token_values(current_line_tokens(prefix)))
    _lines, current_depth, current_values = prefix_lines_with_depth(prefix)
    operation_tags = {str(item) for item in list(expectation.get("operation_tags") or []) if str(item)}
    if operation_tags and bool(expectation.get("requires_operation_evidence")):
        body_text = decode_body_tokens(prefix)
        if current_depth > 0 and loop_plan_has_update_call(body_text) and not values:
            return False
        operation_evidence = source_condition_operation_evidence_for_body(body_text, expectation)
        if not bool(operation_evidence.get("has_operation_evidence")):
            loop_target = source_condition_prefix_loop_target(prefix)
            operation_value_prefix = source_condition_operation_value_prefix_tokens(
                values,
                operation_tags=operation_tags,
                loop_target=loop_target,
                default_arg=str(expectation.get("default_arg") or ""),
                allowed_names=set(allowed_names or set()),
                enabled=enable_operation_value_construction,
            )
            if operation_value_prefix:
                return True
            operation_prefix = source_condition_operation_prefix_tokens(
                values,
                operation_tags=operation_tags,
                accumulator=loop_plan_first_assigned_local(body_text),
                loop_target=loop_target,
                default_arg=str(expectation.get("default_arg") or ""),
                inside_loop=current_depth > 0,
                has_operation_evidence=False,
            )
            if operation_prefix:
                return True
            if values and current_line_tail_needs_operand(list(values)):
                return True
    if values == ("return",):
        return True
    truthiness_arg = str(expectation.get("truthiness_arg") or "")
    default_arg = str(expectation.get("default_arg") or "")
    progress = source_condition_prefix_progress(prefix, expectation)
    if truthiness_arg and source_condition_prefix_can_add_truthiness(values, truthiness_arg):
        return True
    if truthiness_arg and bool(expectation.get("requires_truthiness_guard")):
        if current_depth <= 0 and not values and not progress.get("has_truthiness_guard"):
            return True
        if values in {
            ("if",),
            ("if", "isinstance"),
            ("if", "isinstance", "("),
            ("if", "isinstance", "(", truthiness_arg),
        }:
            return True
    if truthiness_arg and bool(expectation.get("requires_sequence_type_guard")):
        if values in {
            ("if", "isinstance", "(", truthiness_arg, ","),
            ("if", "isinstance", "(", truthiness_arg, ",", "("),
            ("if", "isinstance", "(", truthiness_arg, ",", "(", "list"),
            ("if", "isinstance", "(", truthiness_arg, ",", "(", "list", ","),
            ("if", "isinstance", "(", truthiness_arg, ",", "(", "list", ",", "tuple"),
            ("if", "isinstance", "(", truthiness_arg, ",", "(", "list", ",", "tuple", ")"),
        }:
            return True
    if truthiness_arg and bool(expectation.get("requires_first_item_return")):
        if current_depth > 0 and not values and not progress.get("has_guarded_first_item_return"):
            return True
        if values in {
            ("return", truthiness_arg),
            ("return", truthiness_arg, "["),
            ("return", truthiness_arg, "[", "0"),
            ("return", truthiness_arg, "[", "0", "]"),
        }:
            return True
    if (
        default_arg
        and bool(expectation.get("requires_default_return"))
        and current_depth > 0
        and not current_values
        and progress.get("has_guarded_first_item_return")
        and not progress.get("has_default_return")
    ):
        return True
    if (
        default_arg
        and bool(expectation.get("requires_default_return"))
        and current_depth <= 0
        and not values
        and not progress.get("has_default_return")
        and (
            not bool(expectation.get("requires_truthiness_guard"))
            or progress.get("has_guarded_first_item_return")
            or progress.get("has_truthiness_guard")
        )
    ):
        return True
    if default_arg and values == ("return", default_arg):
        return True
    return False


def source_condition_preempts_loop_plan(expectation: dict[str, Any]) -> bool:
    """Visible guarded/default-return contracts outrank a loop-plan guess.

    This is a task-blind decode ordering rule. It uses only prompt-derived
    source-condition expectations and prevents a wrong learned plan prefix from
    turning simple safe-head/default contracts into identity-copy loops.
    """
    if not bool(expectation.get("enabled")):
        return False
    required = {str(item) for item in list(expectation.get("required_features") or [])}
    return bool(
        expectation.get("requires_default_return")
        and (
            "guarded_first_item_return" in required
            or "guarded_sequence_return" in required
            or "default_return" in required
        )
    )


def source_condition_prefix_can_add_truthiness(values: tuple[str, ...], arg_name: str) -> bool:
    if not values or values[0] != "if" or ":" in values or "and" in values:
        return False
    if arg_name not in values:
        return False
    if "isinstance" not in values:
        return False
    # Wait until the visible isinstance expression is syntactically closed; this
    # admits `and data` as a condition extension without rendering a body.
    return values.count("(") <= values.count(")")


def source_condition_expectation_from_source_text(source_text: str) -> dict[str, Any]:
    text = str(source_text or "")
    lower = text.lower()
    args = visible_argument_names_from_source_text(text)
    operation_tags = visible_operation_tags_from_source_text(text)
    type_tags = set(visible_tags_from_source_text(text, "visible_type_shape_tags "))
    intent_tags = set(visible_tags_from_source_text(text, "visible_intent_tags "))
    empty_or_default = any(token in lower for token in ["empty", "non-empty", "nonempty", "non-sequence", "nonsequence", "default"])
    graph_visible = bool("shape_graph" in type_tags or "intent_graph" in intent_tags or "op_graph_walk" in operation_tags)
    if (not empty_or_default or len(args) < 2) and not operation_tags and not graph_visible:
        return {
            "enabled": False,
            "policy": "prompt_visible_source_condition_expectation_v2",
            "required_features": [],
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    defaultish = {"default", "fallback", "other", "otherwise", "missing", "empty"}
    default_arg = next((arg for arg in args[1:] if any(token in arg.lower() for token in defaultish)), args[1]) if len(args) >= 2 else ""
    has_empty_default = bool(empty_or_default and len(args) >= 2)
    return {
        "enabled": True,
        "policy": "prompt_visible_source_condition_expectation_v2",
        "truthiness_arg": args[0] if args else "",
        "default_arg": default_arg,
        "expected_sequence_types": ["list", "tuple"] if "sequence" in lower else [],
        "operation_tags": operation_tags,
        "requires_truthiness_guard": has_empty_default and ("empty" in lower or "non-empty" in lower or "nonempty" in lower),
        "requires_default_return": has_empty_default and ("default" in lower or "empty" in lower or "non-sequence" in lower or "nonsequence" in lower),
        "requires_first_item_return": has_empty_default and ("first" in lower and ("item" in lower or "sequence" in lower)),
        "requires_sequence_type_guard": has_empty_default and "sequence" in lower,
        "requires_operation_evidence": bool(operation_tags),
        "requires_graph_walk_evidence": graph_visible,
        "required_features": [
            "truthiness_guard",
            "sequence_type_guard",
            "guarded_sequence_return",
            "guarded_first_item_return",
            "default_return",
            "operation_evidence",
            "graph_walk_evidence",
        ],
        "score_semantics": (
            "Derived only from strict prompt/signature source text. It captures visible empty/default "
            "handling expectations, graph-walk obligations from visible graph intent/type tags, and broad "
            "operation tags for decode search diagnostics and optional ranking; it does not use "
            "tests, solutions, task ids, category labels, verifier output, public benchmark payloads, or "
            "teacher output."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def visible_argument_names_from_source_text(source_text: str) -> list[str]:
    for line in str(source_text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("arguments "):
            return [part for part in stripped.split()[1:] if part.isidentifier()]
    return []


def visible_operation_tags_from_source_text(source_text: str) -> list[str]:
    for line in str(source_text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("prompt_operation_hints ") or stripped.startswith("visible_operation_tags "):
            return [part for part in stripped.split()[1:] if part.startswith("op_")]
    return []


def visible_tags_from_source_text(source_text: str, prefix: str) -> list[str]:
    for line in str(source_text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return [part for part in stripped.split()[1:] if part]
    return []


def source_plan_compatibility_for_plan_token(plan_token: str, source_text: str) -> dict[str, Any]:
    """Score a first plan-token choice against prompt-visible source tags only.

    This is a task-blind decode adequacy signal, not a renderer. It uses broad
    prompt/signature tags already visible to the strict generator and can only
    rerank plan-head choices; it never reads task family labels, tests,
    solutions, return-shape metadata, verifier output, public payloads, or
    teacher output.
    """

    plan = str(plan_token or "").removeprefix("SLOT:PLAN_").upper()
    operation_tags = set(visible_operation_tags_from_source_text(source_text))
    intent_tags = set(visible_tags_from_source_text(source_text, "visible_intent_tags "))
    type_tags = set(visible_tags_from_source_text(source_text, "visible_type_shape_tags "))
    if not plan:
        return {
            "enabled": False,
            "policy": "prompt_visible_source_plan_compatibility_v1",
            "score": 0,
            "candidate_generation_credit": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }

    score = 0
    hits: list[str] = []
    misses: list[str] = []

    def reward(label: str, condition: bool, *, weight: int = 4) -> None:
        nonlocal score
        if condition:
            score += weight
            hits.append(label)
        else:
            score -= max(1, weight // 2)
            misses.append(label)

    def plan_has(*parts: str) -> bool:
        return any(part.upper() in plan for part in parts)

    if "shape_graph" in type_tags or "op_graph_walk" in operation_tags:
        reward("graph_plan", plan_has("GRAPH", "HOPS", "COMPONENT", "BFS", "DFS", "PATH"), weight=8)
        if plan_has("AGGREGATE", "MAX", "MIN", "SUM") and not plan_has("HOPS"):
            score -= 6
            misses.append("graph_prompt_aggregate_plan")
    if "shape_records" in type_tags or {"op_group_by_key", "op_threshold_filter"} & operation_tags:
        reward("record_or_table_plan", plan_has("RECORD", "TABLE", "FIELD", "GROUP", "THRESHOLD", "DICT", "PROJECT", "FILTER"), weight=6)
    if "op_group_by_key" in operation_tags:
        reward("group_by_plan", plan_has("GROUP", "DICT", "SETDEFAULT"), weight=6)
    if "op_frequency_top_k" in operation_tags or "shape_ranked_counts" in type_tags:
        reward("frequency_plan", plan_has("TOP_K", "FREQUENT", "COUNT", "DICT", "SORT"), weight=6)
    if "op_interval_merge" in operation_tags or "shape_intervals" in type_tags:
        reward("interval_plan", plan_has("INTERVAL", "MERGE", "SORT"), weight=6)
    if "op_stack_balance" in operation_tags:
        reward("stack_balance_plan", plan_has("BALANCED", "BRACKET", "STACK", "BOOL"), weight=6)
    if "op_gcd_reduce" in operation_tags:
        reward("gcd_plan", plan_has("GCD", "NUMERIC", "ACCUMULATE"), weight=6)
    if "op_normalize_filter_sort" in operation_tags:
        reward("normalize_filter_sort_plan", plan_has("NORMALIZE", "FILTER", "SORT", "LIST"), weight=6)
    if "op_query_key_value_parse" in operation_tags:
        reward("query_parse_plan", plan_has("QUERY", "PARSE", "TEXT", "DICT"), weight=6)
    if {"op_abs_tolerance_filter", "op_abs_positive_filter", "op_threshold_filter"} & operation_tags:
        reward("filter_or_threshold_plan", plan_has("FILTER", "THRESHOLD", "BRANCH", "LIST", "ACCUMULATE"), weight=5)
    if "op_windowed_delta" in operation_tags:
        reward("window_delta_plan", plan_has("WINDOW", "DELTA", "LIST", "ACCUMULATE"), weight=6)
    if "op_numeric_summary" in operation_tags and not ({"shape_graph", "shape_records"} & type_tags):
        reward("numeric_summary_plan", plan_has("NUMERIC", "AGGREGATE", "MAX", "MIN", "SUM", "AVG", "ROUND"), weight=4)
    if "shape_text" in type_tags and not operation_tags:
        if plan_has("TEXT", "STRING", "JOIN", "SPLIT", "PARSE"):
            score += 3
            hits.append("text_plan")
    if "intent_top_k" in intent_tags:
        reward("top_k_intent_plan", plan_has("TOP_K", "FREQUENT", "COUNT", "SORT"), weight=6)
    if "intent_query_string" in intent_tags:
        reward("query_intent_plan", plan_has("QUERY", "PARSE", "DICT"), weight=6)
    if "intent_run_length" in intent_tags:
        reward("run_length_plan", plan_has("RLE", "RUN", "LENGTH"), weight=6)

    enabled = bool(operation_tags or intent_tags or type_tags)
    return {
        "enabled": enabled,
        "policy": "prompt_visible_source_plan_compatibility_v1",
        "plan_token": plan_token,
        "plan": plan,
        "score": int(score) if enabled else 0,
        "hit_count": len(hits),
        "miss_count": len(misses),
        "hits": hits[:12],
        "misses": misses[:12],
        "operation_tags": sorted(operation_tags),
        "intent_tags": sorted(intent_tags),
        "type_shape_tags": sorted(type_tags),
        "score_semantics": (
            "Prompt/signature-visible first-plan adequacy score. It reranks candidate plan-head tokens "
            "only when explicitly enabled, uses broad visible tags already present in strict source text, "
            "and grants zero learned-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def source_plan_compatibility_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    enabled_rows = [dict_or_empty(row) for row in rows if bool(dict_or_empty(row).get("enabled"))]
    score_counts = Counter(int(row.get("selected_score") or row.get("score") or 0) for row in enabled_rows)
    selected_plans = Counter(str(row.get("selected_plan") or row.get("plan") or "") for row in enabled_rows)
    return {
        "policy": "prompt_visible_source_plan_compatibility_summary_v1",
        "row_count": len(rows),
        "enabled_row_count": len(enabled_rows),
        "score_counts": dict(sorted(score_counts.items())),
        "selected_plan_counts": dict(sorted(selected_plans.items())),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def source_condition_expectations_summary(expectations: list[dict[str, Any]]) -> dict[str, Any]:
    enabled = [row for row in expectations if bool(dict_or_empty(row).get("enabled"))]
    return {
        "policy": "prompt_visible_source_condition_expectation_summary_v2",
        "row_count": len(expectations),
        "enabled_row_count": len(enabled),
        "truthiness_arg_counts": dict(sorted(Counter(str(row.get("truthiness_arg") or "") for row in enabled).items())),
        "default_arg_counts": dict(sorted(Counter(str(row.get("default_arg") or "") for row in enabled).items())),
        "operation_tag_counts": dict(
            sorted(Counter(tag for row in enabled for tag in list(row.get("operation_tags") or [])).items())
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def source_condition_decode_beam_sort_key(
    row: dict[str, Any],
    inverse: dict[int, str],
    *,
    target_mode: str,
    expectation: dict[str, Any],
    prefer: bool,
) -> tuple[Any, ...]:
    base = decode_beam_sort_key(row)
    if not prefer or not bool(expectation.get("enabled")) or not body_like_target_mode(target_mode):
        return (*base,)
    decoded_tokens = [inverse.get(int(idx), "<unk>") for idx in list(row.get("generated") or [])[1:]]
    body_prefix = body_tokens_for_target_mode(decoded_tokens, target_mode=target_mode)
    progress = source_condition_prefix_progress(body_prefix, expectation)
    return (
        progress.get("score", 0),
        progress.get("has_guarded_first_item_return", 0),
        progress.get("has_sequence_type_guard", 0),
        progress.get("has_truthiness_guard", 0),
        progress.get("has_default_return", 0),
        *base,
    )


def learned_prefix_decision_final_decode_beam_sort_key(
    row: dict[str, Any],
    inverse: dict[int, str],
    *,
    target_mode: str,
    expectation: dict[str, Any],
    prefer_source_condition: bool,
    prefer_learned_prefix: bool,
) -> tuple[Any, ...]:
    base = source_condition_final_decode_beam_sort_key(
        row,
        inverse,
        target_mode=target_mode,
        expectation=expectation,
        prefer=prefer_source_condition,
    )
    if not prefer_learned_prefix or not learned_plan_prefix_target_mode(target_mode):
        return (*base,)
    decoded_tokens = [inverse.get(int(idx), "<unk>") for idx in list(row.get("generated") or [])[1:]]
    prefix_meta = split_learned_plan_prefix_tokens(decoded_tokens)[1]
    prefix_tokens = [str(tok) for tok in list(prefix_meta.get("learned_plan_prefix_tokens") or [])]
    learned_expectation = learned_prefix_decision_expectation_from_tokens(prefix_tokens)
    body = decode_body_tokens(body_tokens_for_target_mode(decoded_tokens, target_mode=target_mode))
    source_adequacy = source_condition_adequacy_for_body(body, expectation, allowed_names=None)
    source_operation_rank = source_condition_operation_rank_metrics(source_adequacy)
    adequacy = source_condition_adequacy_for_body(body, learned_expectation, allowed_names=None)
    loop_expectation = learned_prefix_loop_expectation_from_tokens(prefix_tokens)
    loop_adequacy = loop_plan_adequacy_for_body(body, loop_expectation, allowed_names=None)
    exact_slots = [
        tok
        for tok in prefix_tokens
        if tok.startswith("SLOT:COND_")
        or tok.startswith("SLOT:RETURN_GUARDED_")
        or tok.startswith("SLOT:RETURN_DEFAULT_")
    ]
    duplicate_penalty = len(exact_slots) - len(set(exact_slots))
    return (
        int(bool(source_adequacy.get("enabled")) and bool(source_adequacy.get("adequate"))),
        int(source_operation_rank.get("hit_operation_tag_count") or 0),
        -int(source_operation_rank.get("missing_operation_tag_count") or 0),
        int(bool(learned_expectation.get("enabled"))),
        int(bool(loop_expectation.get("enabled"))),
        int(loop_adequacy.get("score") or 0),
        int(bool(loop_adequacy.get("has_loop_over_source"))),
        int(bool(loop_adequacy.get("has_update_call"))),
        int(bool(loop_adequacy.get("returns_local"))),
        -int(loop_adequacy.get("bad_signal_count") or 0),
        len(set(exact_slots)),
        -duplicate_penalty,
        int(any(tok.startswith("SLOT:RETURN_DEFAULT_") for tok in exact_slots)),
        int(bool(adequacy.get("adequate"))),
        int(adequacy.get("satisfied_feature_count") or 0),
        *base,
    )


def learned_prefix_decision_rank_score(
    decoded_tokens: list[str],
    body: str,
    *,
    target_mode: str,
    allowed_names: set[str] | None = None,
    enabled: bool,
) -> float | None:
    if not enabled or not learned_plan_prefix_target_mode(target_mode):
        return None
    prefix_meta = split_learned_plan_prefix_tokens(decoded_tokens)[1]
    prefix_tokens = [str(tok) for tok in list(prefix_meta.get("learned_plan_prefix_tokens") or [])]
    learned_expectation = learned_prefix_decision_expectation_from_tokens(prefix_tokens)
    exact_slots = [
        tok
        for tok in prefix_tokens
        if tok.startswith("SLOT:COND_")
        or tok.startswith("SLOT:RETURN_GUARDED_")
        or tok.startswith("SLOT:RETURN_DEFAULT_")
    ]
    duplicate_penalty = len(exact_slots) - len(set(exact_slots))
    adequacy = source_condition_adequacy_for_body(body, learned_expectation, allowed_names=None)
    loop_expectation = learned_prefix_loop_expectation_from_tokens(prefix_tokens)
    loop_adequacy = loop_plan_adequacy_for_body(body, loop_expectation, allowed_names=allowed_names)
    score = 0.0
    score += 100.0 if bool(learned_expectation.get("enabled")) else 0.0
    score += float(loop_adequacy.get("score") or 0.0)
    score += 10.0 * len(set(exact_slots))
    score -= 5.0 * duplicate_penalty
    score += 20.0 if any(tok.startswith("SLOT:RETURN_DEFAULT_") for tok in exact_slots) else 0.0
    score += 50.0 if bool(adequacy.get("adequate")) else 0.0
    score += float(adequacy.get("satisfied_feature_count") or 0)
    return round(score, 8)


def learned_prefix_loop_plan_adequacy_metadata(
    decoded_tokens: list[str],
    body: str,
    *,
    target_mode: str,
    allowed_names: set[str] | None,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled or not learned_plan_prefix_target_mode(target_mode):
        return {
            "enabled": False,
            "policy": "learned_prefix_loop_plan_adequacy_v1",
            "candidate_generation_credit": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    prefix_tokens = [str(tok) for tok in list(split_learned_plan_prefix_tokens(decoded_tokens)[1].get("learned_plan_prefix_tokens") or [])]
    expectation = learned_prefix_loop_expectation_from_tokens(prefix_tokens)
    adequacy = loop_plan_adequacy_for_body(body, expectation, allowed_names=allowed_names)
    adequacy.update(
        {
            "policy": "learned_prefix_loop_plan_adequacy_v1",
            "score_semantics": (
                "Task-blind AST adequacy check comparing model-generated loop/action prefix slots to the "
                "generated body. It checks structural consistency such as initializer, loop over visible "
                "source argument, update call, and return shape. It does not render code, inspect tests or "
                "solutions, use public data, or grant learned-generation credit."
            ),
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }
    )
    return adequacy


def learned_prefix_loop_expectation_from_tokens(prefix_tokens: list[str]) -> dict[str, Any]:
    slots = [str(tok) for tok in prefix_tokens if str(tok).startswith("SLOT:")]
    plan_token = next((tok for tok in slots if tok.startswith("SLOT:PLAN_")), "")
    plan = plan_token.removeprefix("SLOT:PLAN_")
    if not plan or plan == "SAFE_HEAD_DEFAULT" or PLAN_BODY_START_TOKEN not in slots:
        return {
            "enabled": False,
            "policy": "learned_prefix_loop_expectation_v1",
            "candidate_generation_credit": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    loop_source = "data" if "SLOT:LOOP_SOURCE_DATA" in slots else "other" if "SLOT:LOOP_SOURCE_OTHER" in slots else ""
    init_shapes = sorted(
        token.removeprefix("SLOT:INIT_").lower()
        for token in slots
        if token.startswith("SLOT:INIT_")
    )
    update_ops = sorted(
        token.removeprefix("SLOT:UPDATE_").lower()
        for token in slots
        if token.startswith("SLOT:UPDATE_")
    )
    finalizers = sorted(
        token.removeprefix("SLOT:FINALIZER_").lower()
        for token in slots
        if token.startswith("SLOT:FINALIZER_")
    )
    return_shapes = sorted(
        token.removeprefix("SLOT:RETURN_SHAPE_").lower()
        for token in slots
        if token.startswith("SLOT:RETURN_SHAPE_")
    )
    state_transitions = sorted(
        token.removeprefix("SLOT:STATE_").lower()
        for token in slots
        if token.startswith("SLOT:STATE_")
    )
    operand_bindings = sorted(
        token.removeprefix("SLOT:BIND_").lower()
        for token in slots
        if token.startswith("SLOT:BIND_")
    )
    return {
        "enabled": bool(loop_source or init_shapes or update_ops or finalizers or return_shapes or state_transitions or operand_bindings),
        "policy": "learned_prefix_loop_expectation_v1",
        "source": "model_generated_prefix_tokens",
        "plan": plan,
        "plan_token": plan_token,
        "loop_source_arg": loop_source,
        "init_shapes": init_shapes,
        "update_ops": update_ops,
        "finalizers": finalizers,
        "return_shapes": return_shapes,
        "state_transitions": state_transitions,
        "operand_bindings": operand_bindings,
        "candidate_generation_credit": 0,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
    }


def loop_plan_adequacy_for_body(
    body: str,
    expectation: dict[str, Any],
    *,
    allowed_names: set[str] | None,
) -> dict[str, Any]:
    if not bool(expectation.get("enabled")):
        return {
            "enabled": False,
            "score": 0,
            "candidate_generation_credit": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    source_arg = str(expectation.get("loop_source_arg") or "data")
    init_shapes = set(str(item) for item in list(expectation.get("init_shapes") or []))
    update_ops = set(str(item) for item in list(expectation.get("update_ops") or []))
    return_shapes = set(str(item) for item in list(expectation.get("return_shapes") or []))
    state_transitions = set(str(item) for item in list(expectation.get("state_transitions") or []))
    operand_bindings = set(str(item) for item in list(expectation.get("operand_bindings") or []))
    function = parsed_decode_guard_function(body, allowed_names=allowed_names)
    if function is None:
        return {
            "enabled": True,
            "parse_ok": False,
            "score": -10,
            "bad_signal_count": 1,
            "failures": ["parse_or_function"],
            "expectation": {key: expectation.get(key) for key in ["plan", "loop_source_arg", "init_shapes", "update_ops", "return_shapes", "state_transitions", "operand_bindings"]},
            "candidate_generation_credit": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }

    dependency = decode_guard_return_dependency_summary(body, allowed_names=allowed_names)
    value_quality = expression_value_quality_summary(body, allowed_names=allowed_names)
    init_names: set[str] = set()
    init_shape_hits: set[str] = set()
    repeated_init_count = 0
    for stmt in function.body:
        if isinstance(stmt, ast.Assign):
            target_names = [target.id for target in stmt.targets if isinstance(target, ast.Name)]
            if not target_names:
                continue
            shape = loop_plan_value_shape(stmt.value)
            if shape:
                for name in target_names:
                    if name in init_names:
                        repeated_init_count += 1
                    init_names.add(name)
                init_shape_hits.add(shape)

    loop_count = 0
    loop_over_source_count = 0
    update_call_count = 0
    append_like_update_count = 0
    assignment_transform_count = 0
    branch_count = 0
    control_terminal_count = 0
    augmented_update_count = 0
    shallow_identity_accumulation_count = 0
    self_accumulator_append_count = 0
    append_whole_source_argument_count = 0
    semantic_update_value_count = 0
    early_return_count = 0
    binding_hits: Counter[str] = Counter()
    for node in ast.walk(function):
        if isinstance(node, ast.For):
            loop_count += 1
            loop_target_name = node.target.id if isinstance(node.target, ast.Name) else ""
            loop_target_names = expression_store_names(node.target)
            if loop_iter_mentions_source(node.iter, source_arg):
                loop_over_source_count += 1
            for child in ast.walk(ast.Module(body=list(node.body), type_ignores=[])):
                child_names = expression_load_names(child)
                if loop_target_names and child_names & loop_target_names:
                    binding_hits["loop_target_used"] += 1
                if isinstance(child, ast.If):
                    branch_count += 1
                    test_names = expression_load_names(child.test)
                    if test_names & loop_target_names:
                        binding_hits["branch_uses_loop_target"] += 1
                    if test_names & (init_names | {source_arg, "data", "other"}):
                        binding_hits["branch_uses_source_or_state"] += 1
                if isinstance(child, (ast.Break, ast.Continue)):
                    control_terminal_count += 1
                if isinstance(child, ast.Return):
                    early_return_count += 1
                if isinstance(child, ast.AugAssign):
                    augmented_update_count += 1
                    value_names = expression_load_names(child.value)
                    if value_names & loop_target_names:
                        binding_hits["update_uses_loop_target"] += 1
                    if value_names & init_names:
                        binding_hits["update_uses_accumulator"] += 1
                    if expression_has_constant(child.value):
                        binding_hits["update_uses_constant"] += 1
                if isinstance(child, (ast.Assign, ast.AnnAssign)):
                    value = child.value if isinstance(child, ast.Assign) else child.value
                    value_names = expression_load_names(value)
                    target_names: set[str] = set()
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            target_names.update(expression_store_names(target))
                    else:
                        target_names.update(expression_store_names(child.target))
                    if value_names & loop_target_names:
                        binding_hits["update_uses_loop_target"] += 1
                    if value_names & {source_arg, "data", "other"}:
                        binding_hits["update_uses_source_arg"] += 1
                    if value_names & init_names:
                        binding_hits["update_uses_accumulator"] += 1
                    if target_names & init_names and value_names & init_names:
                        binding_hits["update_assign_uses_previous_state"] += 1
                    if loop_plan_value_has_stateful_transform(value):
                        assignment_transform_count += 1
                        if value_names & loop_target_names:
                            binding_hits["loop_target_transformed"] += 1
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                    if child.func.attr in LOOP_PLAN_MUTATION_METHODS:
                        update_call_count += 1
                        if child.func.attr in LOOP_PLAN_MUTATION_METHODS - {"setdefault"}:
                            append_like_update_count += 1
                        if (
                            child.func.attr in {"append", "add"}
                            and isinstance(child.func.value, ast.Name)
                            and child.func.value.id in init_names
                            and len(child.args) == 1
                            and isinstance(child.args[0], ast.Name)
                            and child.args[0].id == loop_target_name
                        ):
                            shallow_identity_accumulation_count += 1
                        if (
                            child.func.attr in {"append", "add"}
                            and isinstance(child.func.value, ast.Name)
                            and child.func.value.id in init_names
                            and len(child.args) == 1
                            and isinstance(child.args[0], ast.Name)
                            and child.args[0].id == child.func.value.id
                        ):
                            self_accumulator_append_count += 1
                        for arg in child.args:
                            arg_names = expression_load_names(arg)
                            if arg_names & loop_target_names:
                                binding_hits["update_uses_loop_target"] += 1
                            if arg_names & {source_arg, "data", "other"}:
                                binding_hits["update_uses_source_arg"] += 1
                            if arg_names & init_names:
                                binding_hits["update_uses_accumulator"] += 1
                            if isinstance(arg, ast.Name) and arg.id in loop_target_names:
                                binding_hits["update_call_arg_loop_target"] += 1
                            elif loop_plan_value_has_stateful_transform(arg):
                                binding_hits["update_call_arg_transform"] += 1
                            if expression_has_constant(arg):
                                binding_hits["update_uses_constant"] += 1
                            if source_arg and expression_is_direct_source_return(arg, source_arg=source_arg):
                                append_whole_source_argument_count += 1
                            if loop_plan_update_arg_is_semantic(
                                arg,
                                loop_target_names=loop_target_names,
                                init_names=init_names,
                                source_arg=source_arg,
                                mutation_method=child.func.attr,
                            ):
                                semantic_update_value_count += 1

    top_level_returns = [stmt for stmt in function.body if isinstance(stmt, ast.Return)]
    return_local_count = 0
    direct_input_return_count = 0
    inert_return_count = 0
    for stmt in top_level_returns:
        names = expression_load_names(stmt.value)
        if source_arg and expression_is_direct_source_return(stmt.value, source_arg=source_arg):
            direct_input_return_count += 1
        if names & init_names:
            return_local_count += 1
            binding_hits["finalizer_uses_accumulator"] += 1
            if isinstance(stmt.value, ast.Name):
                binding_hits["return_accumulator"] += 1
            elif loop_plan_value_has_stateful_transform(stmt.value):
                binding_hits["return_transformed_accumulator"] += 1
        if return_value_is_none_like(stmt.value) or isinstance(stmt.value, ast.Constant):
            inert_return_count += 1

    score = 0
    failures: list[str] = []
    score += 2 if init_names else -2
    if init_shapes:
        score += 2 if init_shapes & init_shape_hits else -2
        if not (init_shapes & init_shape_hits):
            failures.append("missing_expected_initializer_shape")
    if source_arg:
        score += 4 if loop_over_source_count else -4
        if not loop_over_source_count:
            failures.append("missing_loop_over_source")
    expected_update_call = bool(update_ops & {"append_item", "call"})
    score += 3 if update_call_count else -2
    if expected_update_call and not update_call_count:
        score -= 6
        failures.append("missing_expected_update_call")
    if update_ops and any(op in {"append_item", "call"} for op in update_ops):
        score += 2 if append_like_update_count or update_call_count else -2
    if "has_branch" in state_transitions or "loop_has_branch" in state_transitions:
        score += 3 if branch_count else -3
        if not branch_count:
            failures.append("missing_state_branch")
    if "loop_no_branch" in state_transitions:
        score += 1 if not branch_count else -1
    if "update_augassign" in state_transitions:
        score += 3 if augmented_update_count else -2
        if not augmented_update_count:
            failures.append("missing_augmented_state_update")
    if "update_assign_transform" in state_transitions:
        score += 3 if assignment_transform_count else -2
        if not assignment_transform_count:
            failures.append("missing_assignment_transform_state_update")
    if "update_nonidentity" in state_transitions:
        nonidentity_update_count = max(0, update_call_count + assignment_transform_count + augmented_update_count - shallow_identity_accumulation_count)
        score += 4 if nonidentity_update_count else -4
        if not nonidentity_update_count:
            failures.append("missing_nonidentity_state_update")
    if "update_shallow_identity" in state_transitions:
        score += 1 if shallow_identity_accumulation_count else -1
    score += 4 if return_local_count else -3
    if not return_local_count:
        failures.append("missing_local_result_return")
    top_level_nontrivial_return_count = int(dependency.get("top_level_nontrivial_return_count") or 0)
    parameter_dependent_return_count = int(dependency.get("parameter_dependent_return_count") or 0)
    weak_placeholder_local_return_count = int(dependency.get("weak_placeholder_local_return_count") or 0)
    if top_level_nontrivial_return_count:
        score += 4
    else:
        score -= 5
        failures.append("return_not_nontrivial")
    if parameter_dependent_return_count <= 0:
        score -= 3
        failures.append("return_not_parameter_dependent")
    if weak_placeholder_local_return_count:
        score -= 3 * weak_placeholder_local_return_count
        failures.append("weak_placeholder_local_return")
    if update_call_count and not semantic_update_value_count:
        score -= 4
        failures.append("missing_semantic_update_value")
    if append_whole_source_argument_count:
        score -= 4 * append_whole_source_argument_count
        failures.append("append_whole_source_argument")
    if self_accumulator_append_count:
        score -= 6 * self_accumulator_append_count
        failures.append("append_accumulator_to_itself")
    bad_signal_count = direct_input_return_count + inert_return_count + early_return_count + repeated_init_count
    invalid_value_count = int(value_quality.get("invalid_expression_value_count") or 0)
    if invalid_value_count:
        score -= 6 * invalid_value_count
        failures.append("invalid_expression_value")
        bad_signal_count += invalid_value_count
    score -= 3 * direct_input_return_count
    score -= 2 * inert_return_count
    score -= 2 * early_return_count
    score -= repeated_init_count
    if direct_input_return_count:
        failures.append("direct_input_return")
    if early_return_count:
        failures.append("early_return_inside_loop")
    if repeated_init_count:
        failures.append("repeated_initializer_assignment")
    plan_name = str(expectation.get("plan") or "").lower()
    identity_plan = any(marker in plan_name for marker in ["identity", "copy"])
    if shallow_identity_accumulation_count and not identity_plan:
        score -= 5 * shallow_identity_accumulation_count
        failures.append("shallow_identity_accumulation")
        bad_signal_count += shallow_identity_accumulation_count
    if "list" in return_shapes:
        score += 1 if "list" in init_shape_hits and return_local_count else 0
    for binding in sorted(operand_bindings):
        hit = binding_hits.get(binding, 0)
        if hit:
            score += 2
        else:
            score -= 2
            failures.append(f"missing_operand_binding_{binding}")
    return {
        "enabled": True,
        "parse_ok": True,
        "score": int(score),
        "has_initializer": bool(init_names),
        "initializer_names": sorted(init_names)[:8],
        "initializer_shape_hits": sorted(init_shape_hits),
        "has_expected_initializer_shape": bool(not init_shapes or (init_shapes & init_shape_hits)),
        "loop_count": loop_count,
        "has_loop_over_source": bool(loop_over_source_count),
        "update_call_count": update_call_count,
        "has_update_call": bool(update_call_count),
        "branch_count": branch_count,
        "control_terminal_count": control_terminal_count,
        "assignment_transform_count": assignment_transform_count,
        "augmented_update_count": augmented_update_count,
        "semantic_update_value_count": semantic_update_value_count,
        "append_whole_source_argument_count": append_whole_source_argument_count,
        "self_accumulator_append_count": self_accumulator_append_count,
        "operand_binding_hits": dict(sorted(binding_hits.items())),
        "returns_local": bool(return_local_count),
        "returns_parameter_dependent": bool(parameter_dependent_return_count),
        "returns_nontrivial": bool(top_level_nontrivial_return_count),
        "weak_placeholder_local_return_count": weak_placeholder_local_return_count,
        "expression_value_quality": value_quality,
        "direct_input_return_count": direct_input_return_count,
        "early_return_inside_loop_count": early_return_count,
        "repeated_initializer_assignment_count": repeated_init_count,
        "shallow_identity_accumulation_count": shallow_identity_accumulation_count,
        "bad_signal_count": int(bad_signal_count),
        "failures": failures,
        "expectation": {key: expectation.get(key) for key in ["plan", "loop_source_arg", "init_shapes", "update_ops", "return_shapes", "state_transitions", "operand_bindings"]},
        "candidate_generation_credit": 0,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
    }


def learned_prefix_body_action_trace_metadata(
    decoded_tokens: list[str],
    body: str,
    *,
    target_mode: str,
    allowed_names: set[str] | None,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled or not learned_plan_prefix_target_mode(target_mode):
        return {
            "enabled": False,
            "policy": "strict_generator_body_action_trace_v1",
            "candidate_generation_credit": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    prefix_tokens = [str(tok) for tok in list(split_learned_plan_prefix_tokens(decoded_tokens)[1].get("learned_plan_prefix_tokens") or [])]
    expectation = learned_prefix_loop_expectation_from_tokens(prefix_tokens)
    trace = body_action_trace_for_body(body, expectation, allowed_names=allowed_names)
    trace.update(
        {
            "policy": "strict_generator_body_action_trace_v1",
            "score_semantics": (
                "Task-blind AST action trace over a decoded body. It reports operation families and "
                "prefix/body mismatches for residual mining only. It does not render code, inspect "
                "tests/solutions, use public data, or grant learned-generation credit."
            ),
            "candidate_generation_credit": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    )
    return trace


def body_action_trace_for_body(body: str, expectation: dict[str, Any], *, allowed_names: set[str] | None) -> dict[str, Any]:
    function = parsed_decode_guard_function(body, allowed_names=allowed_names)
    if function is None:
        return {
            "enabled": True,
            "parse_ok": False,
            "operation_counts": {},
            "call_counts": {},
            "mismatch_labels": ["parse_or_function"],
        }
    operation_counts: Counter[str] = Counter()
    call_counts: Counter[str] = Counter()
    assigned_names: set[str] = set()
    loop_targets: set[str] = set()
    return_inside_loop_count = 0
    shallow_identity_accumulation_count = 0
    self_accumulator_append_count = 0
    unreachable_loop_statement_count = 0
    top_level_return_count = sum(1 for stmt in function.body if isinstance(stmt, ast.Return))
    bool_not_local_return_count = 0
    listcomp_return_count = 0
    subscript_assignment_count = 0
    comparison_count = 0
    branch_count = 0
    augmented_update_count = 0
    assignment_transform_count = 0
    append_whole_source_argument_count = 0
    semantic_update_value_count = 0
    source_arg = str(expectation.get("loop_source_arg") or "data")
    dependency = decode_guard_return_dependency_summary(body, allowed_names=allowed_names)
    value_quality = expression_value_quality_summary(body, allowed_names=allowed_names)
    for stmt in function.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                assigned_names.update(expression_store_names(target))
    for node in ast.walk(function):
        if isinstance(node, ast.For):
            operation_counts["for_loop"] += 1
            loop_targets.update(expression_store_names(node.target))
            unreachable_loop_statement_count += unreachable_statement_count_after_control_flow(node.body)
            for child in ast.walk(ast.Module(body=list(node.body), type_ignores=[])):
                if isinstance(child, ast.Return):
                    return_inside_loop_count += 1
                if isinstance(child, (ast.Assign, ast.AnnAssign)):
                    value = child.value if isinstance(child, ast.Assign) else child.value
                    target_names: set[str] = set()
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            target_names.update(expression_store_names(target))
                    else:
                        target_names.update(expression_store_names(child.target))
                    value_names = expression_load_names(value)
                    if target_names & assigned_names and (value_names & (assigned_names | loop_targets) or loop_plan_value_has_stateful_transform(value)):
                        assignment_transform_count += 1
                        operation_counts["assignment_transform"] += 1
        elif isinstance(node, ast.While):
            operation_counts["while_loop"] += 1
            unreachable_loop_statement_count += unreachable_statement_count_after_control_flow(node.body)
        elif isinstance(node, ast.If):
            operation_counts["branch"] += 1
            branch_count += 1
        elif isinstance(node, ast.Assign):
            operation_counts["assignment"] += 1
            if any(isinstance(target, ast.Subscript) for target in node.targets):
                subscript_assignment_count += 1
        elif isinstance(node, ast.AugAssign):
            operation_counts["augmented_update"] += 1
            augmented_update_count += 1
        elif isinstance(node, ast.Compare):
            operation_counts["comparison"] += 1
            comparison_count += 1
        elif isinstance(node, ast.BoolOp):
            operation_counts["bool_op"] += 1
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            operation_counts["not_op"] += 1
        elif isinstance(node, ast.ListComp):
            operation_counts["list_comp"] += 1
        elif isinstance(node, ast.Return):
            operation_counts["return"] += 1
            if isinstance(node.value, ast.UnaryOp) and isinstance(node.value.op, ast.Not):
                names = expression_load_names(node.value.operand)
                if names & assigned_names:
                    bool_not_local_return_count += 1
            if isinstance(node.value, ast.ListComp):
                listcomp_return_count += 1
        elif isinstance(node, ast.Call):
            operation_counts["call"] += 1
            name = action_trace_call_name(node)
            if name:
                call_counts[name] += 1
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"append", "add"}
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in assigned_names
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in loop_targets
            ):
                shallow_identity_accumulation_count += 1
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"append", "add"}
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in assigned_names
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == node.func.value.id
            ):
                self_accumulator_append_count += 1
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in LOOP_PLAN_MUTATION_METHODS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in assigned_names
            ):
                for arg in node.args:
                    if source_arg and expression_is_direct_source_return(arg, source_arg=source_arg):
                        append_whole_source_argument_count += 1
                    if loop_plan_update_arg_is_semantic(
                        arg,
                        loop_target_names=loop_targets,
                        init_names=assigned_names,
                        source_arg=source_arg,
                        mutation_method=node.func.attr,
                    ):
                        semantic_update_value_count += 1
    mismatch_labels: list[str] = []
    plan_fragments: list[str] = []
    for plan_key in ("plan", "source_plan", "semantic_plan", "expected_plan", "plan_tags"):
        value = expectation.get(plan_key)
        if isinstance(value, (list, tuple, set)):
            plan_fragments.extend(str(item) for item in value if str(item))
        elif isinstance(value, dict):
            plan_fragments.extend(f"{key}:{item}" for key, item in value.items() if str(key) or str(item))
        elif value:
            plan_fragments.append(str(value))
    plan = " ".join(plan_fragments).lower()
    update_ops = {str(item) for item in list(expectation.get("update_ops") or [])}
    finalizers = {str(item) for item in list(expectation.get("finalizers") or [])}
    return_shapes = {str(item) for item in list(expectation.get("return_shapes") or [])}
    if operation_counts["for_loop"] and not (
        branch_count
        or comparison_count
        or augmented_update_count
        or assignment_transform_count
        or semantic_update_value_count
        or subscript_assignment_count
        or any(name.rsplit(".", 1)[-1] in {"gcd", "max", "min", "abs", "pop", "discard", "remove"} for name in call_counts)
    ):
        mismatch_labels.append("loop_without_decision_or_state_update")
    if shallow_identity_accumulation_count:
        mismatch_labels.append("shallow_identity_accumulation")
    if self_accumulator_append_count:
        mismatch_labels.append("append_accumulator_to_itself")
    if append_whole_source_argument_count:
        mismatch_labels.append("append_whole_source_argument")
    if operation_counts["for_loop"] and call_counts and not semantic_update_value_count:
        mismatch_labels.append("missing_semantic_update_value")
    if int(value_quality.get("bare_builtin_value_argument_count") or 0):
        mismatch_labels.append("bare_builtin_value_argument")
    if int(value_quality.get("invalid_call_argument_type_count") or 0):
        mismatch_labels.append("invalid_call_argument_type")
    if int(value_quality.get("invalid_expression_value_count") or 0):
        mismatch_labels.append("invalid_expression_value")
    if int(dependency.get("parameter_dependent_return_count") or 0) <= 0:
        mismatch_labels.append("return_not_parameter_dependent")
    if int(dependency.get("top_level_nontrivial_return_count") or 0) <= 0:
        mismatch_labels.append("return_not_nontrivial")
    if int(dependency.get("weak_placeholder_local_return_count") or 0):
        mismatch_labels.append("weak_placeholder_local_return")
    if return_inside_loop_count:
        mismatch_labels.append("early_return_inside_loop")
    if unreachable_loop_statement_count:
        mismatch_labels.append("unreachable_loop_update_after_control_flow")
    if "accumulate_numeric" in update_ops and not augmented_update_count and not assignment_transform_count and "math.gcd" not in call_counts and "gcd" not in call_counts:
        mismatch_labels.append("missing_numeric_accumulation")
    if "bool_not_stack" in finalizers and not bool_not_local_return_count:
        mismatch_labels.append("missing_bool_not_local_finalizer")
    has_list_builder_call = any(
        name in {"append", "extend", "sorted", "list"} or name.endswith(".append") or name.endswith(".extend")
        for name in call_counts
    )
    if "list" in return_shapes and not (listcomp_return_count or has_list_builder_call):
        mismatch_labels.append("missing_list_construction")
    if "gcd" in plan and "math.gcd" not in call_counts and "gcd" not in call_counts:
        mismatch_labels.append("missing_gcd_call")
    if "windowed" in plan and not listcomp_return_count:
        mismatch_labels.append("missing_windowed_finalizer")
    if "rle" in plan and not (subscript_assignment_count or branch_count):
        mismatch_labels.append("missing_rle_branch_or_update")
    graph_plan = any(marker in plan for marker in ("graph", "hops", "path", "component", "bfs", "dfs", "shortest"))
    graph_walk_evidence = source_condition_graph_walk_evidence_for_body(
        body,
        {"requires_graph_walk_evidence": graph_plan},
    )
    if graph_plan and not bool(graph_walk_evidence.get("has_graph_walk_evidence")):
        mismatch_labels.append("missing_graph_walk_evidence")
    return {
        "enabled": True,
        "parse_ok": True,
        "plan": expectation.get("plan"),
        "expected_update_ops": sorted(update_ops),
        "expected_finalizers": sorted(finalizers),
        "expected_return_shapes": sorted(return_shapes),
        "operation_counts": dict(sorted(operation_counts.items())),
        "call_counts": dict(sorted(call_counts.items())),
        "assigned_names": sorted(assigned_names)[:12],
        "loop_targets": sorted(loop_targets)[:12],
        "top_level_return_count": top_level_return_count,
        "return_inside_loop_count": return_inside_loop_count,
        "shallow_identity_accumulation_count": shallow_identity_accumulation_count,
        "unreachable_loop_statement_count": unreachable_loop_statement_count,
        "subscript_assignment_count": subscript_assignment_count,
        "assignment_transform_count": assignment_transform_count,
        "augmented_update_count": augmented_update_count,
        "listcomp_return_count": listcomp_return_count,
        "bool_not_local_return_count": bool_not_local_return_count,
        "semantic_update_value_count": semantic_update_value_count,
        "append_whole_source_argument_count": append_whole_source_argument_count,
        "self_accumulator_append_count": self_accumulator_append_count,
        "graph_walk_evidence": graph_walk_evidence,
        "return_dependency": dependency,
        "expression_value_quality": value_quality,
        "mismatch_labels": mismatch_labels,
    }


def unreachable_statement_count_after_control_flow(statements: list[ast.stmt]) -> int:
    count = 0
    blocked = False
    for stmt in statements:
        if blocked:
            count += 1
        if isinstance(stmt, (ast.Return, ast.Break, ast.Continue)):
            blocked = True
    return count


def expression_has_constant(node: ast.AST | None) -> bool:
    if node is None:
        return False
    return any(isinstance(child, ast.Constant) for child in ast.walk(node))


def loop_plan_value_shape(expr: ast.AST | None) -> str:
    if isinstance(expr, ast.List):
        return "list"
    if isinstance(expr, ast.Dict):
        return "dict"
    if isinstance(expr, ast.Tuple):
        return "tuple"
    if isinstance(expr, ast.Set):
        return "set"
    if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float)):
        return "number"
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
        if expr.func.id in {"list", "dict", "tuple", "set"}:
            return expr.func.id
    return ""


def loop_plan_value_has_stateful_transform(expr: ast.AST | None) -> bool:
    if expr is None:
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
    return any(isinstance(child, semantic_nodes) for child in ast.walk(expr))


def loop_iter_mentions_source(expr: ast.AST | None, source_arg: str) -> bool:
    if not source_arg:
        return False
    if isinstance(expr, ast.Name):
        return expr.id == source_arg
    return source_arg in expression_load_names(expr)


def expression_is_direct_source_return(expr: ast.AST | None, *, source_arg: str) -> bool:
    if not source_arg:
        return False
    if isinstance(expr, ast.Name):
        return expr.id == source_arg
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id in {"list", "tuple", "set", "dict"}:
        return any(isinstance(arg, ast.Name) and arg.id == source_arg for arg in expr.args)
    return False


def loop_plan_update_arg_is_semantic(
    expr: ast.AST | None,
    *,
    loop_target_names: set[str],
    init_names: set[str],
    source_arg: str,
    mutation_method: str = "",
) -> bool:
    if expr is None:
        return False
    if source_arg and expression_is_direct_source_return(expr, source_arg=source_arg):
        return False
    names = expression_load_names(expr)
    semantic_names = names - EXPRESSION_VALUE_BARE_BUILTINS - {"False", "None", "True"}
    if mutation_method in {"discard", "remove", "pop"} and loop_target_names and semantic_names & loop_target_names:
        return True
    if names & init_names:
        if isinstance(expr, ast.Name) and expr.id in init_names:
            return False
        return True
    if loop_target_names and semantic_names & loop_target_names:
        if isinstance(expr, ast.Name):
            return False
        return bool(loop_plan_value_has_stateful_transform(expr) or expression_has_constant(expr))
    return bool(semantic_names) and loop_plan_value_has_stateful_transform(expr) and not expression_value_has_bare_builtin_value(expr)


def loop_plan_exploration_choices(
    arr: Any,
    inverse: dict[int, str],
    prefix: list[str],
    *,
    expectation: dict[str, Any],
    seen: set[int],
    token_policy: str,
    allowed_names: set[str] | None,
    input_type_hints: dict[str, str] | None = None,
    require_nontrivial_return: bool = False,
    enabled: bool,
    enable_loop_progress_guard: bool = False,
    enable_expression_value_guard: bool = False,
) -> list[tuple[int, float]]:
    if not enabled or not bool(expectation.get("enabled")):
        return []
    token_to_id = {text: int(idx) for idx, text in inverse.items()}
    line_values = token_values(current_line_tokens(prefix))
    body_text = decode_body_tokens(prefix)
    accumulator = loop_plan_accumulator_name(expectation, body_text=body_text, inverse=inverse, arr=arr)
    loop_var = loop_plan_loop_var_name(expectation, body_text=body_text, inverse=inverse, arr=arr)
    source_arg = loop_plan_source_arg(expectation, allowed_names=allowed_names)
    shape = loop_plan_primary_init_shape(expectation)
    update_style = loop_plan_update_style(expectation)
    finalizer = loop_plan_return_finalizer(expectation)
    choices: list[str] = []
    has_initializer = loop_plan_has_initializer(body_text)
    has_loop = loop_plan_has_loop_over_source(body_text, source_arg=source_arg)
    inside_loop = loop_plan_inside_loop(prefix)
    has_update = loop_plan_has_update_call(body_text)
    generated_mutation_choices = (
        loop_plan_generated_mutation_continuation(line_values, accumulator=accumulator, loop_var=loop_var)
        if inside_loop and enable_loop_progress_guard
        else []
    )

    if generated_mutation_choices:
        choices = generated_mutation_choices
    elif inside_loop and line_values and line_values[0] == accumulator:
        choices = loop_plan_update_continuation(line_values, accumulator=accumulator, loop_var=loop_var, update_style=update_style)
    elif line_values and line_values[0] == accumulator and ("=" in line_values or not has_initializer):
        choices = loop_plan_initializer_continuation(line_values, accumulator=accumulator, shape=shape)
    elif not has_initializer and not line_values:
        choices = [f"NAME:{accumulator}"]
    elif not has_initializer:
        choices = loop_plan_initializer_continuation(line_values, accumulator=accumulator, shape=shape)
    elif not has_loop:
        choices = loop_plan_loop_header_continuation(line_values, loop_var=loop_var, source_arg=source_arg)
    elif inside_loop and not has_update and not line_values and enable_loop_progress_guard:
        choices = loop_plan_update_continuation(line_values, accumulator=accumulator, loop_var=loop_var, update_style=update_style)
    elif inside_loop and not has_update and not line_values:
        choices = []
    elif inside_loop and has_update and not line_values:
        choices = ["DEDENT:"]
    elif line_values and line_values[0] == "return":
        choices = loop_plan_return_continuation(line_values, accumulator=accumulator, finalizer=finalizer)
    elif not loop_plan_has_local_return(body_text, accumulator=accumulator) and not line_values:
        choices = ["NAME:return"]
    elif not loop_plan_has_local_return(body_text, accumulator=accumulator) and line_values == ["return"]:
        choices = loop_plan_return_continuation(line_values, accumulator=accumulator, finalizer=finalizer)

    out: list[tuple[int, float]] = []
    for token in choices:
        idx = token_to_id.get(token)
        if idx is None or idx in seen:
            continue
        if token_blocked_by_strict_decode_guard(
            prefix,
            token,
            require_nontrivial_return=require_nontrivial_return,
            allowed_names=allowed_names,
            input_type_hints=input_type_hints,
        ):
            continue
        if token_blocked_by_loop_plan(
            prefix,
            token,
            expectation=expectation,
            block_shallow_identity_update=False,
        ):
            continue
        if token_blocked_by_expression_value_guard(
            prefix,
            token,
            expectation=expectation,
            enabled=enable_expression_value_guard,
        ):
            continue
        if not token_allowed_by_policy(prefix, token, policy=token_policy, allowed_names=allowed_names):
            continue
        seen.add(idx)
        out.append((idx, float(max(arr[idx], 1e-9))))
    return out


def loop_plan_priority_prefix(prefix: list[str], expectation: dict[str, Any]) -> bool:
    if not bool(expectation.get("enabled")):
        return False
    body_text = decode_body_tokens(prefix)
    line_values = token_values(current_line_tokens(prefix))
    accumulator = loop_plan_accumulator_name(expectation, body_text=body_text, inverse={}, arr=[])
    source_arg = str(expectation.get("loop_source_arg") or "data")
    has_initializer = loop_plan_has_initializer(body_text)
    if line_values and line_values[0] == accumulator and ("=" in line_values or not has_initializer):
        return True
    if not has_initializer:
        return True
    if not loop_plan_has_loop_over_source(body_text, source_arg=source_arg):
        return True
    if loop_plan_inside_loop(prefix) and line_values and line_values[0] == accumulator:
        return True
    if line_values and line_values[0] == "return":
        return True
    if not loop_plan_has_local_return(body_text, accumulator=accumulator):
        return True
    return False


def token_blocked_by_loop_plan(
    prefix: list[str],
    token: str,
    *,
    expectation: dict[str, Any],
    block_shallow_identity_update: bool = False,
) -> bool:
    if not bool(expectation.get("enabled")):
        return False
    body_text = decode_body_tokens(prefix)
    line_values = token_values(current_line_tokens(prefix))
    if not loop_plan_inside_loop(prefix):
        return False
    if token == "NAME:return" and not line_values:
        return True
    if line_values:
        if block_shallow_identity_update and loop_plan_blocks_shallow_identity_append(
            line_values,
            token,
            expectation=expectation,
            body_text=body_text,
        ):
            return True
        return False
    if loop_plan_has_update_call(body_text):
        return False
    return token in {"NAME:return", "DEDENT:"}


def filter_loop_plan_blocked_choices(
    prefix: list[str],
    choices: list[tuple[int, float]],
    inverse: dict[int, str],
    *,
    expectation: dict[str, Any],
    block_shallow_identity_update: bool = False,
) -> list[tuple[int, float]]:
    if not choices:
        return []
    return [
        (idx, score)
        for idx, score in choices
        if not token_blocked_by_loop_plan(
            prefix,
            inverse.get(int(idx), "<unk>"),
            expectation=expectation,
            block_shallow_identity_update=block_shallow_identity_update,
        )
    ]


def filter_expression_value_blocked_choices(
    prefix: list[str],
    choices: list[tuple[int, float]],
    inverse: dict[int, str],
    *,
    expectation: dict[str, Any],
    enabled: bool = False,
) -> list[tuple[int, float]]:
    if not choices:
        return []
    return [
        (idx, score)
        for idx, score in choices
        if not token_blocked_by_expression_value_guard(
            prefix,
            inverse.get(int(idx), "<unk>"),
            expectation=expectation,
            enabled=enabled,
        )
    ]


def token_blocked_by_expression_value_guard(
    prefix: list[str],
    token: str,
    *,
    expectation: dict[str, Any],
    enabled: bool = False,
) -> bool:
    if not enabled:
        return False
    values = token_values(current_line_tokens(prefix))
    token_value = token_value_for_guard(token)
    if not token_value:
        return False

    if expression_value_would_create_bare_builtin_attribute_chain(values, token_value):
        return True
    if expression_value_has_bare_builtin_attribute_chain(values):
        return True
    if expression_value_tail_is_uncalled_attribute_reference(values) and token_value != "(":
        return True

    if current_return_line_has_invalid_expression_value(values) and not current_return_invalid_value_can_be_repaired(
        values,
        token_value,
    ):
        return True

    # Empty literals remain legal for initializers such as "out = []"; they are
    # blocked only when the prefix is in an expression/update-call value slot.
    if token_value in {")", "]", "}"} and current_line_tail_needs_operand(values):
        if expression_value_allows_empty_initializer(values, token_value):
            return False
        return True

    if expression_value_inside_update_call(values):
        if token_value == "{":
            return True
        if token_value in {")", "]", "}"}:
            arg_values = expression_value_current_call_arg_values(values)
            if not arg_values:
                return True
            if len(arg_values) == 1 and arg_values[0] in EXPRESSION_VALUE_BARE_BUILTINS:
                return True
            if expression_value_arg_is_empty_literal(arg_values):
                return True

    if token_value in {",", ")"} and expression_value_inside_isinstance_call(values):
        arg_values = expression_value_current_call_arg_values(values)
        first_arg = first_call_argument_values(arg_values)
        if token_value == "," and "," not in arg_values and isinstance_first_arg_values_invalid(first_arg):
            return True
        if token_value == ")" and isinstance_first_arg_values_invalid(first_arg):
            return True

    if token_value == "NEWLINE" and values:
        if current_line_tail_needs_operand(values):
            return True
        if current_line_expected_closer(values):
            return True
        if current_return_line_has_invalid_expression_value(values):
            return True
        if expression_value_has_bare_builtin_attribute_chain(values):
            return True
        if current_assignment_line_has_bare_builtin_value(values):
            return True
        if len(values) >= 2 and values[0] == "return" and values[-1] in EXPRESSION_VALUE_BARE_BUILTINS:
            return True
        if expression_value_inside_update_call(values):
            arg_values = expression_value_current_call_arg_values(values)
            if len(arg_values) == 1 and arg_values[0] in EXPRESSION_VALUE_BARE_BUILTINS:
                return True

    plan = str(dict_or_empty(expectation).get("plan") or "").lower()
    if token.startswith("STRING:") and expression_value_inside_update_call(values):
        if any(marker in plan for marker in ["gcd", "numeric", "window", "even", "sum", "count"]):
            return True
    if token_value == "(" and expression_value_has_pathological_open_paren_run(values):
        return True
    if token_value == "not" and expression_value_has_pathological_not_run(values):
        return True
    return False


def expression_value_would_create_bare_builtin_attribute_chain(values: list[str], token_value: str) -> bool:
    """Reject ``max.get``-style attribute chains on builtin/type objects.

    The check is lexical and task-blind so it can fire before the partial
    expression becomes parseable Python.
    """

    if token_value == "." and values and values[-1] in EXPRESSION_VALUE_BARE_BUILTINS:
        return True
    if len(values) >= 2 and values[-2] in EXPRESSION_VALUE_BARE_BUILTINS and values[-1] == ".":
        return True
    return False


def expression_value_has_bare_builtin_attribute_chain(values: list[str]) -> bool:
    for index in range(len(values) - 1):
        if values[index] in EXPRESSION_VALUE_BARE_BUILTINS and values[index + 1] == ".":
            return True
    return False


def expression_value_tail_is_uncalled_attribute_reference(values: list[str]) -> bool:
    if len(values) < 3:
        return False
    if values[-2] != "." or not values[-1].isidentifier():
        return False
    if values[-1] in {"real", "imag"}:
        return False
    return True


def current_return_line_has_invalid_expression_value(values: list[str]) -> bool:
    """Use the existing task-blind value checker before closing a return line."""

    if len(values) < 2 or values[0] != "return":
        return False
    line = " ".join(values)
    summary = expression_value_quality_summary(line, allowed_names=None)
    return bool(summary.get("parse_ok")) and int(summary.get("invalid_expression_value_count") or 0) > 0


def current_assignment_line_has_bare_builtin_value(values: list[str]) -> bool:
    """Reject ``local = max`` style runtime builtin objects as values."""

    if len(values) < 3 or "=" not in values:
        return False
    equals = values.index("=")
    if equals != 1 or not values[0].isidentifier():
        return False
    return values[-1] in EXPRESSION_VALUE_BARE_BUILTINS


def current_return_invalid_value_can_be_repaired(values: list[str], token_value: str) -> bool:
    """Allow a bare builtin return prefix to become a call, not a value."""

    if len(values) != 2 or values[0] != "return" or values[1] not in EXPRESSION_VALUE_BARE_BUILTINS:
        return False
    if token_value != "(":
        return False
    summary = expression_value_quality_summary(" ".join(values), allowed_names=None)
    return int(summary.get("invalid_expression_value_count") or 0) == int(
        summary.get("bare_builtin_runtime_value_count") or 0
    )


def token_value_for_guard(token: str) -> str:
    if token == "NEWLINE:":
        return "NEWLINE"
    if token == "INDENT:":
        return "INDENT"
    if token == "DEDENT:":
        return "DEDENT"
    if ":" not in token:
        return token
    kind, value = token.split(":", 1)
    if kind in {"NAME", "OP", "NUMBER", "STRING"}:
        return value
    return ""


def expression_value_has_pathological_open_paren_run(values: list[str]) -> bool:
    if not values:
        return False
    if values[-1:] == ["("] and expression_value_inside_update_call(values):
        return True
    unmatched = 0
    for value in values:
        if value == "(":
            unmatched += 1
        elif value == ")" and unmatched > 0:
            unmatched -= 1
    if unmatched >= 3:
        return True
    return len(values) >= 3 and values[-3:] == ["(", "(", "("]


def expression_value_has_pathological_not_run(values: list[str]) -> bool:
    """Reject repeated unary-not chains in generated value slots.

    This is task-blind syntax/value hygiene. It only reads the generated prefix
    and never inspects target answers, verifier outputs, tests, solutions, or
    public benchmark payloads.
    """

    if not values:
        return False
    if not expression_value_inside_update_call(values) and not current_line_tail_needs_operand(values):
        return False
    trailing = 0
    for value in reversed(values):
        if value == "not":
            trailing += 1
            continue
        break
    if trailing >= 1:
        return True
    return sum(1 for value in values if value == "not") >= 4


def expression_value_allows_empty_initializer(values: list[str], closer: str) -> bool:
    opener = {")": "(", "]": "[", "}": "{"}.get(closer, "")
    return bool(len(values) == 3 and values[0].isidentifier() and values[1] == "=" and values[2] == opener)


def expression_value_inside_update_call(values: list[str]) -> bool:
    return any(
        values[index : index + 3] in [[".", method, "("] for method in LOOP_PLAN_MUTATION_METHODS]
        for index in range(max(0, len(values) - 8), max(0, len(values) - 2))
    )


def expression_value_inside_isinstance_call(values: list[str]) -> bool:
    for index in range(len(values) - 2, -1, -1):
        if values[index] != "(":
            continue
        if index > 0 and values[index - 1] == "isinstance":
            return True
        return False
    return False


def expression_value_current_call_arg_values(values: list[str]) -> list[str]:
    for index in range(len(values) - 1, -1, -1):
        if values[index] == "(":
            return values[index + 1 :]
    return []


def expression_value_arg_is_empty_literal(arg_values: list[str]) -> bool:
    return arg_values in (["(", ")"], ["[", "]"], ["{", "}"])


def loop_plan_initializer_continuation(line_values: list[str], *, accumulator: str, shape: str) -> list[str]:
    if line_values == [accumulator]:
        return ["OP:="]
    if line_values == [accumulator, "="]:
        if shape == "set":
            return ["NAME:set"]
        if shape == "dict":
            return ["OP:{"]
        if shape == "number":
            return ["NUMBER:0"]
        if shape == "tuple":
            return ["OP:("]
        return ["OP:["]
    if line_values == [accumulator, "=", "set"]:
        return ["OP:("]
    if line_values == [accumulator, "=", "set", "("]:
        return ["OP:)"]
    if line_values == [accumulator, "=", "["]:
        return ["OP:]"]
    if line_values == [accumulator, "=", "{"]:
        return ["OP:}"]
    if line_values == [accumulator, "=", "("]:
        return ["OP:)"]
    if tuple(line_values) in {
        (accumulator, "=", "[", "]"),
        (accumulator, "=", "{", "}"),
        (accumulator, "=", "(", ")"),
        (accumulator, "=", "set", "(", ")"),
        (accumulator, "=", "0"),
    }:
        return ["NEWLINE:"]
    return []


def loop_plan_loop_header_continuation(line_values: list[str], *, loop_var: str, source_arg: str) -> list[str]:
    if not line_values:
        return ["NAME:for"]
    if line_values == ["for"]:
        return [f"NAME:{loop_var}"]
    if line_values == ["for", loop_var]:
        return ["NAME:in"]
    if line_values == ["for", loop_var, "in"]:
        return [f"NAME:{source_arg}"]
    if line_values == ["for", loop_var, "in", source_arg]:
        return ["OP::"]
    return []


def loop_plan_update_continuation(line_values: list[str], *, accumulator: str, loop_var: str, update_style: str) -> list[str]:
    if update_style == "augassign_add":
        if not line_values:
            return [f"NAME:{accumulator}"]
        if line_values == [accumulator]:
            return ["OP:+="]
        if line_values == [accumulator, "+="]:
            return [f"NAME:{loop_var}"]
        if line_values == [accumulator, "+=", loop_var]:
            return ["NEWLINE:"]
        return []
    if update_style == "assign_transform":
        if not line_values:
            return [f"NAME:{accumulator}"]
        if line_values == [accumulator]:
            return ["OP:="]
        if line_values == [accumulator, "="]:
            return [f"NAME:{accumulator}"]
        if line_values == [accumulator, "=", accumulator]:
            return ["OP:+"]
        if line_values == [accumulator, "=", accumulator, "+"]:
            return [f"NAME:{loop_var}"]
        if line_values == [accumulator, "=", accumulator, "+", loop_var]:
            return ["NEWLINE:"]
        return []
    method = "add" if update_style == "method_add" else "append"
    if not line_values:
        return [f"NAME:{accumulator}"]
    if line_values == [accumulator]:
        return ["OP:."]
    if line_values == [accumulator, "."]:
        return [f"NAME:{method}"]
    if line_values == [accumulator, ".", method]:
        return ["OP:("]
    if line_values == [accumulator, ".", method, "("]:
        return [f"NAME:{loop_var}"]
    if line_values == [accumulator, ".", method, "(", loop_var]:
        return ["OP:)"]
    if line_values == [accumulator, ".", method, "(", loop_var, ")"]:
        return ["NEWLINE:"]
    return []


def loop_plan_generated_mutation_continuation(line_values: list[str], *, accumulator: str, loop_var: str) -> list[str]:
    """Finish a generated mutation statement so loop beams can reach return.

    This helper does not select an algorithm or method. It only recognizes that
    the model already emitted an accumulator mutation call and advances the
    syntactic transition needed to close that statement. That keeps learned
    beams from spending the whole budget repeating partial update calls before
    dedent/finalizer search can run.
    """

    if len(line_values) < 3:
        return []
    if line_values[0] != accumulator or line_values[1] != ".":
        return []
    method = line_values[2]
    if method not in LOOP_PLAN_MUTATION_METHODS:
        return []
    if line_values == [accumulator, ".", method]:
        return ["OP:("]
    if line_values == [accumulator, ".", method, "("]:
        if method == "pop":
            return ["OP:)"]
        return [f"NAME:{loop_var}"]
    if line_values == [accumulator, ".", method, "(", loop_var]:
        return ["OP:)"]
    if line_values == [accumulator, ".", method, "(", ")"]:
        return ["NEWLINE:"]
    if line_values == [accumulator, ".", method, "(", loop_var, ")"]:
        return ["NEWLINE:"]
    return []


def loop_plan_return_continuation(line_values: list[str], *, accumulator: str, finalizer: str = "") -> list[str]:
    if finalizer == "sorted":
        if line_values == ["return"]:
            return ["NAME:sorted"]
        if line_values == ["return", "sorted"]:
            return ["OP:("]
        if line_values == ["return", "sorted", "("]:
            return [f"NAME:{accumulator}"]
        if line_values == ["return", "sorted", "(", accumulator]:
            return ["OP:)"]
        if line_values == ["return", "sorted", "(", accumulator, ")"]:
            return ["NEWLINE:"]
        return []
    if line_values == ["return"]:
        return [f"NAME:{accumulator}"]
    if line_values == ["return", accumulator]:
        return ["NEWLINE:"]
    return []


def loop_plan_blocks_shallow_identity_append(
    line_values: list[str],
    token: str,
    *,
    expectation: dict[str, Any],
    body_text: str,
) -> bool:
    if loop_plan_allows_direct_loop_append(expectation):
        return False
    accumulator = loop_plan_first_assigned_local(body_text)
    loop_var = loop_plan_first_loop_var(body_text)
    if not accumulator or not loop_var:
        return False
    if line_values == [accumulator, ".", "append", "("] and token == f"NAME:{loop_var}":
        return True
    if line_values == [accumulator, ".", "add", "("] and token == f"NAME:{loop_var}":
        return True
    return False


def loop_plan_allows_direct_loop_append(expectation: dict[str, Any]) -> bool:
    plan = str(expectation.get("plan") or "").lower()
    update_ops = {str(item) for item in list(expectation.get("update_ops") or [])}
    if "append_item" in update_ops:
        return True
    if any(marker in plan for marker in ["identity", "copy"]):
        return True
    return False


def loop_plan_accumulator_name(expectation: dict[str, Any], *, body_text: str, inverse: dict[int, str], arr: Any) -> str:
    existing = loop_plan_first_assigned_local(body_text)
    if existing:
        return existing
    plan = str(expectation.get("plan") or "").lower()
    candidates = ["out", "result", "items", "values", "acc"]
    if "balanced" in plan:
        candidates = ["stack", "pairs", "out", "result"]
    elif "rle" in plan:
        candidates = ["out", "runs", "result", "items"]
    elif "gcd" in plan or "max" in plan or "length" in plan:
        candidates = ["result", "best", "count", "out"]
    return loop_plan_highest_probability_name(candidates, inverse=inverse, arr=arr)


def loop_plan_loop_var_name(expectation: dict[str, Any], *, body_text: str, inverse: dict[int, str], arr: Any) -> str:
    existing = loop_plan_first_loop_var(body_text)
    if existing:
        return existing
    plan = str(expectation.get("plan") or "").lower()
    candidates = ["item", "value", "x", "row"]
    if "balanced" in plan:
        candidates = ["ch", "item", "value"]
    return loop_plan_highest_probability_name(candidates, inverse=inverse, arr=arr)


def loop_plan_highest_probability_name(candidates: list[str], *, inverse: dict[int, str], arr: Any) -> str:
    if not inverse or len(arr) == 0:
        return candidates[0]
    token_to_id = {text: int(idx) for idx, text in inverse.items()}
    ranked = sorted(
        candidates,
        key=lambda name: float(arr[token_to_id.get(f"NAME:{name}", 0)]) if token_to_id.get(f"NAME:{name}") is not None else -1.0,
        reverse=True,
    )
    return ranked[0] if ranked else candidates[0]


def loop_plan_source_arg(expectation: dict[str, Any], *, allowed_names: set[str] | None) -> str:
    source = str(expectation.get("loop_source_arg") or "")
    allowed = {str(name) for name in set(allowed_names or set()) if str(name).isidentifier()}
    if source and (not allowed or source in allowed):
        return source
    return "data" if not allowed or "data" in allowed else sorted(allowed)[0]


def loop_plan_primary_init_shape(expectation: dict[str, Any]) -> str:
    update_ops = {str(item) for item in list(expectation.get("update_ops") or [])}
    if "set_add" in update_ops:
        return "set"
    shapes = [str(item) for item in list(expectation.get("init_shapes") or [])]
    for shape in ["set", "list", "dict", "number", "tuple"]:
        if shape in shapes:
            return shape
    return "list"


def loop_plan_update_style(expectation: dict[str, Any]) -> str:
    update_ops = {str(item) for item in list(expectation.get("update_ops") or [])}
    state_transitions = {str(item) for item in list(expectation.get("state_transitions") or [])}
    init_shape = loop_plan_primary_init_shape(expectation)
    if "update_augassign" in state_transitions or "accumulate_numeric" in update_ops:
        return "augassign_add"
    if "update_assign_transform" in state_transitions:
        return "assign_transform"
    if "set_add" in update_ops:
        return "method_add"
    if "append_item" in update_ops:
        return "method_append"
    if init_shape == "set":
        return "method_add"
    if init_shape == "number":
        return "assign_transform"
    if "loop_has_assignment" in state_transitions and "update_mutation_call" not in state_transitions:
        return "assign_transform"
    return "method_append"


def loop_plan_update_method(expectation: dict[str, Any]) -> str:
    style = loop_plan_update_style(expectation)
    if style == "method_add":
        return "add"
    if style in {"augassign_add", "assign_transform"}:
        return style
    return "append"


def loop_plan_return_finalizer(expectation: dict[str, Any]) -> str:
    finalizers = {str(item) for item in list(expectation.get("finalizers") or [])}
    update_ops = {str(item) for item in list(expectation.get("update_ops") or [])}
    if "sorted" in finalizers or "sort" in update_ops:
        return "sorted"
    return ""


def loop_plan_has_initializer(body_text: str) -> bool:
    return bool(loop_plan_first_assigned_local(body_text))


def loop_plan_first_assigned_local(body_text: str) -> str:
    for raw_line in str(body_text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("return ", "for ", "if ", "elif ", "else", "while ")):
            continue
        if "=" not in line or line.startswith(("==", "!=", ">=", "<=")):
            continue
        name = line.split("=", 1)[0].strip()
        if name.isidentifier():
            return name
    return ""


def loop_plan_first_loop_var(body_text: str) -> str:
    for raw_line in str(body_text or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("for "):
            continue
        target = line.removeprefix("for ").split(" in ", 1)[0].split(",", 1)[0].strip()
        if target.isidentifier():
            return target
    return ""


def loop_plan_has_loop_over_source(body_text: str, *, source_arg: str) -> bool:
    if not source_arg:
        return False
    for raw_line in str(body_text or "").splitlines():
        line = raw_line.strip()
        if line.startswith("for ") and f" in {source_arg}" in line and line.endswith(":"):
            return True
    return False


def loop_plan_has_update_call(body_text: str) -> bool:
    function = parsed_decode_guard_function(body_text, allowed_names=None)
    if function is not None:
        for loop in [node for node in ast.walk(function) if isinstance(node, (ast.For, ast.While))]:
            body_module = ast.Module(body=list(getattr(loop, "body", []) or []), type_ignores=[])
            for child in ast.walk(body_module):
                if isinstance(child, ast.AugAssign):
                    return True
                if isinstance(child, (ast.Assign, ast.AnnAssign)):
                    value = child.value if isinstance(child, ast.Assign) else child.value
                    if loop_plan_value_has_stateful_transform(value):
                        return True
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                    if child.func.attr in LOOP_PLAN_MUTATION_METHODS:
                        return True
    for raw_line in str(body_text or "").splitlines():
        line = raw_line.strip()
        if not line.endswith(")"):
            continue
        if any(f".{method}(" in line for method in sorted(LOOP_PLAN_MUTATION_METHODS)):
            return True
    return False


def loop_plan_has_local_return(body_text: str, *, accumulator: str) -> bool:
    lines = [line.strip() for line in str(body_text or "").splitlines()]
    return any(line == f"return {accumulator}" or line.startswith(f"return {accumulator}[") or line.startswith(f"return {accumulator} [") for line in lines)


def loop_plan_inside_loop(prefix: list[str]) -> bool:
    depth = 0
    seen_for = False
    for tok in prefix:
        if tok == "NAME:for":
            seen_for = True
        elif tok == "INDENT:":
            depth += 1
        elif tok == "DEDENT:":
            depth = max(0, depth - 1)
    return seen_for and depth > 0


def source_condition_final_decode_beam_sort_key(
    row: dict[str, Any],
    inverse: dict[int, str],
    *,
    target_mode: str,
    expectation: dict[str, Any],
    prefer: bool,
) -> tuple[Any, ...]:
    base = final_decode_beam_sort_key(row, inverse, target_mode=target_mode)
    if not prefer or not bool(expectation.get("enabled")) or not body_like_target_mode(target_mode):
        return (*base,)
    decoded_tokens = [inverse.get(int(idx), "<unk>") for idx in list(row.get("generated") or [])[1:]]
    body = decode_body_tokens(body_tokens_for_target_mode(decoded_tokens, target_mode=target_mode))
    adequacy = source_condition_adequacy_for_body(body, expectation, allowed_names=None)
    operation_rank = source_condition_operation_rank_metrics(adequacy)
    return (
        int(bool(adequacy.get("adequate"))),
        int(adequacy.get("satisfied_feature_count") or 0),
        int(bool(adequacy.get("has_operation_evidence"))),
        int(operation_rank.get("hit_operation_tag_count") or 0),
        -int(operation_rank.get("missing_operation_tag_count") or 0),
        int(bool(adequacy.get("has_guarded_first_item_return"))),
        int(bool(adequacy.get("has_sequence_type_guard"))),
        int(bool(adequacy.get("has_truthiness_guard"))),
        int(bool(adequacy.get("has_guarded_data_return"))),
        int(bool(adequacy.get("has_default_return"))),
        *base,
    )


def source_condition_prefix_progress(prefix: list[str], expectation: dict[str, Any]) -> dict[str, int]:
    text = decode_body_tokens(prefix)
    truthiness_arg = str(expectation.get("truthiness_arg") or "")
    default_arg = str(expectation.get("default_arg") or "")
    lines = [line.strip() for line in text.splitlines()]
    has_truthiness = int(bool(truthiness_arg and any(line.startswith("if ") and f"and {truthiness_arg}" in line for line in lines)))
    has_default = int(bool(default_arg and any(line == f"return {default_arg}" for line in lines)))
    has_sequence_type_guard = int(bool(truthiness_arg and any(f"isinstance({truthiness_arg}, (list, tuple))" in line for line in lines)))
    has_guarded_first_item_return = int(
        bool(
            truthiness_arg
            and any(
                line == f"return {truthiness_arg}[0]"
                or line == f"return {truthiness_arg} [0]"
                for line in lines
            )
        )
    )
    has_guarded_data_return = int(
        bool(
            truthiness_arg
            and any(line.startswith("if ") and f"and {truthiness_arg}" in line for line in lines)
            and any(
                line == f"return {truthiness_arg}"
                or line.startswith(f"return {truthiness_arg}[")
                or line.startswith(f"return {truthiness_arg} [")
                for line in lines
            )
        )
    )
    has_branch = int(bool(truthiness_arg and "if " in text and truthiness_arg in text))
    operation_evidence = source_condition_operation_evidence_for_body(text, expectation)
    has_operation_evidence = int(bool(operation_evidence.get("has_operation_evidence")))
    return {
        "has_branch": has_branch,
        "has_truthiness_guard": has_truthiness,
        "has_default_return": has_default,
        "has_sequence_type_guard": has_sequence_type_guard,
        "has_guarded_first_item_return": has_guarded_first_item_return,
        "has_guarded_data_return": has_guarded_data_return,
        "has_operation_evidence": has_operation_evidence,
        "score": (
            has_branch
            + (2 * has_truthiness)
            + (2 * has_sequence_type_guard)
            + (2 * has_default)
            + (2 * has_guarded_data_return)
            + (3 * has_guarded_first_item_return)
            + (3 * has_operation_evidence)
        ),
    }


def source_condition_operation_rank_metrics(adequacy: dict[str, Any]) -> dict[str, int]:
    operation_evidence = dict_or_empty(adequacy.get("operation_evidence"))
    hit_count = len([item for item in list(operation_evidence.get("hit_operation_tags") or []) if str(item)])
    missing_count = len([item for item in list(operation_evidence.get("missing_operation_tags") or []) if str(item)])
    recognized_count = len([item for item in list(operation_evidence.get("recognized_operation_tags") or []) if str(item)])
    return {
        "hit_operation_tag_count": int(hit_count),
        "missing_operation_tag_count": int(missing_count),
        "recognized_operation_tag_count": int(recognized_count),
    }


def source_condition_adequacy_for_body(
    body: str,
    expectation: dict[str, Any],
    *,
    allowed_names: set[str] | None,
) -> dict[str, Any]:
    if not bool(expectation.get("enabled")):
        return {
            "enabled": False,
            "adequate": True,
            "missing_features": [],
            "candidate_generation_credit": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    truthiness_arg = str(expectation.get("truthiness_arg") or "")
    default_arg = str(expectation.get("default_arg") or "")
    has_truthiness = source_condition_body_has_truthiness_guard(body, truthiness_arg)
    has_default = source_condition_body_has_default_return(body, default_arg)
    expected_sequence_types = [str(item) for item in list(expectation.get("expected_sequence_types") or []) if str(item)]
    has_sequence_type_guard = source_condition_body_has_sequence_type_guard(body, truthiness_arg, expected_sequence_types)
    has_guarded_first_item_return = source_condition_body_has_guarded_first_item_return(body, truthiness_arg)
    has_guarded_data_return = source_condition_body_has_guarded_data_return(body, truthiness_arg)
    operation_evidence = source_condition_operation_evidence_for_body(body, expectation)
    graph_walk_evidence = source_condition_graph_walk_evidence_for_body(body, expectation)
    missing: list[str] = []
    if bool(expectation.get("requires_truthiness_guard")) and not has_truthiness:
        missing.append("truthiness_guard")
    if bool(expectation.get("requires_default_return")) and not has_default:
        missing.append("default_return")
    if bool(expectation.get("requires_sequence_type_guard")) and not has_sequence_type_guard:
        missing.append("sequence_type_guard")
    if bool(expectation.get("requires_truthiness_guard")) and not has_guarded_data_return:
        missing.append("guarded_sequence_return")
    if bool(expectation.get("requires_first_item_return")) and not has_guarded_first_item_return:
        missing.append("guarded_first_item_return")
    if bool(expectation.get("requires_operation_evidence")) and not bool(operation_evidence.get("has_operation_evidence")):
        missing.append("operation_evidence")
    if bool(expectation.get("requires_graph_walk_evidence")) and not bool(graph_walk_evidence.get("has_graph_walk_evidence")):
        missing.append("graph_walk_evidence")
    satisfied = (
        int(bool(has_truthiness))
        + int(bool(has_default))
        + int(bool(has_sequence_type_guard))
        + int(bool(has_guarded_data_return))
        + int(bool(has_guarded_first_item_return))
        + int(bool(operation_evidence.get("has_operation_evidence")))
        + int(bool(graph_walk_evidence.get("has_graph_walk_evidence")))
    )
    return {
        "enabled": True,
        "policy": "prompt_visible_source_condition_adequacy_v2",
        "adequate": not missing,
        "missing_features": missing,
        "satisfied_feature_count": satisfied,
        "has_truthiness_guard": bool(has_truthiness),
        "has_default_return": bool(has_default),
        "has_sequence_type_guard": bool(has_sequence_type_guard),
        "has_guarded_first_item_return": bool(has_guarded_first_item_return),
        "has_guarded_data_return": bool(has_guarded_data_return),
        "has_operation_evidence": bool(operation_evidence.get("has_operation_evidence")),
        "operation_evidence": operation_evidence,
        "has_graph_walk_evidence": bool(graph_walk_evidence.get("has_graph_walk_evidence")),
        "graph_walk_evidence": graph_walk_evidence,
        "truthiness_arg": truthiness_arg,
        "default_arg": default_arg,
        "score_semantics": (
            "Task-blind AST/text adequacy check for prompt-visible empty/default behavior and broad "
            "operation or graph-walk tags. It sees only the decoded body and source-text-derived expectations."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def source_condition_prefix_loop_target(prefix: list[str]) -> str:
    lines, _current_depth, _current_values = prefix_lines_with_depth(prefix)
    for _depth, values in reversed(lines):
        clean = [str(item) for item in values]
        if len(clean) >= 4 and clean[0] == "for" and clean[1].isidentifier() and clean[2] == "in":
            return clean[1]
    return ""


def source_condition_operation_prefix_tokens(
    values: tuple[str, ...],
    *,
    operation_tags: set[str],
    accumulator: str,
    loop_target: str,
    default_arg: str,
    inside_loop: bool,
    has_operation_evidence: bool,
) -> list[str]:
    if not operation_tags or has_operation_evidence:
        return []
    filter_tags = {
        "op_abs_tolerance_filter",
        "op_abs_positive_filter",
        "op_threshold_filter",
        "op_clip_to_range",
    }
    transform_tags = {
        "op_round_values",
        "op_clip_to_range",
        "op_numeric_summary",
        "op_windowed_delta",
    }
    choices: list[str] = []
    if not values:
        if inside_loop and operation_tags & filter_tags:
            choices.append("NAME:if")
        if inside_loop and accumulator and operation_tags & transform_tags:
            choices.append(f"NAME:{accumulator}")
        return list(dict.fromkeys(choices))[:4]
    if accumulator and values == (accumulator,):
        return ["OP:="]
    if accumulator and values == (accumulator, "="):
        if not inside_loop and not loop_target:
            return []
        if "op_round_values" in operation_tags:
            choices.append("NAME:round")
        if "op_clip_to_range" in operation_tags:
            choices.extend(["NAME:min", "NAME:max"])
        if "op_numeric_summary" in operation_tags:
            choices.extend(["NAME:sum", "NAME:max", "NAME:min"])
        return list(dict.fromkeys(choices))[:5]
    if values == ("if",):
        if operation_tags & {"op_abs_tolerance_filter", "op_abs_positive_filter", "op_windowed_delta"}:
            choices.append("NAME:abs")
        if loop_target and operation_tags & {"op_threshold_filter", "op_clip_to_range"}:
            choices.append(f"NAME:{loop_target}")
        return list(dict.fromkeys(choices))[:3]
    if values == ("if", "abs"):
        return ["OP:("]
    if values == ("if", "abs", "(") and loop_target:
        return [f"NAME:{loop_target}"]
    if values == ("if", "abs", "(", loop_target):
        return ["OP:)"]
    if len(values) >= 5 and values[0] == "if" and "abs" in values and values[-1] == ")" and not any(
        item in values for item in {"<", "<=", ">", ">=", "==", "!="}
    ):
        return ["OP:<=", "OP:<"]
    if (
        len(values) >= 6
        and values[0] == "if"
        and "abs" in values
        and values[-1] in {"<", "<=", ">", ">=", "==", "!="}
        and default_arg
    ):
        return [f"NAME:{default_arg}"]
    return []


def source_condition_operation_value_prefix_tokens(
    values: tuple[str, ...],
    *,
    operation_tags: set[str],
    loop_target: str,
    default_arg: str,
    allowed_names: set[str],
    enabled: bool,
) -> list[str]:
    """Continue operation-bearing update expressions from visible prompt tags.

    This is a decode-search helper, not a renderer. It only fires after the
    learned decoder has already opened an update value such as ``out.append(``
    or ``out =``. The helper then offers local expression continuations that
    expose prompt-visible operation evidence, while the token decoder still
    chooses the final candidate and the verifier/integrity layers decide
    whether the body is useful.
    """

    if not enabled or not operation_tags or not loop_target:
        return []
    allowed = {str(name) for name in allowed_names if str(name)}
    default_visible = bool(default_arg and default_arg in allowed)
    choices: list[str] = []

    def unique(items: list[str], limit: int = 5) -> list[str]:
        return list(dict.fromkeys(items))[:limit]

    at_expression_start = bool(
        values
        and (
            values[-1] == "="
            or values[-3:] == (".", "append", "(")
            or values[-2:] == ("append", "(")
        )
    )
    if at_expression_start:
        if "op_round_values" in operation_tags:
            choices.append("NAME:round")
        if operation_tags & {"op_abs_tolerance_filter", "op_abs_positive_filter"}:
            choices.append("NAME:abs")
        if default_visible and operation_tags & {"op_clip_to_range", "op_threshold_filter"}:
            choices.extend(["NAME:min", "NAME:max"])
        choices.append(f"NAME:{loop_target}")
        return unique(choices)

    if values[-1:] in {("round",), ("abs",), ("min",), ("max",)}:
        return ["OP:("]
    if values[-2:] in {("round", "("), ("abs", "(")}:
        if default_visible and "op_clip_to_range" in operation_tags and values[-2:] == ("round", "("):
            return ["NAME:min", "NAME:max", f"NAME:{loop_target}"]
        return [f"NAME:{loop_target}"]
    if default_visible and values[-2:] == ("min", "("):
        return ["NAME:max", f"NAME:{loop_target}"]
    if values[-1:] == (loop_target,):
        open_index = _innermost_value_open_paren(values)
        callee = values[open_index - 1] if open_index > 0 else ""
        if callee in {"round", "abs"}:
            return ["OP:)"]
        if default_visible and callee == "max":
            return ["OP:,"]
        if default_visible and callee == "min":
            return ["OP:,"]
    if default_visible and values[-2:] == ("max", "("):
        return [f"NAME:{loop_target}"]
    if default_visible and values[-3:] == ("max", "(", loop_target):
        return ["OP:,"]
    if default_visible and len(values) >= 4 and values[-1] == "," and values[-3] == "(" and values[-4] == "max":
        return [f"NAME:{default_arg}"]
    if default_visible and values[-1:] == (",",):
        open_index = _innermost_value_open_paren(values)
        callee = values[open_index - 1] if open_index > 0 else ""
        if callee == "min":
            return [f"NAME:{default_arg}"]
    if default_visible and values[-1:] == (default_arg,):
        open_index = _innermost_value_open_paren(values)
        callee = values[open_index - 1] if open_index > 0 else ""
        if callee in {"max", "min"}:
            return ["OP:)"]
    if values[-1:] == (")",):
        if default_visible:
            open_index = _innermost_value_open_paren(values)
            callee = values[open_index - 1] if open_index > 0 else ""
            if callee == "min" and not _top_level_comma_after_open(values, open_index):
                return ["OP:,"]
        if _current_line_open_paren_delta(values) > 0:
            return ["OP:)"]
        return ["NEWLINE:"]
    return []


def _innermost_value_open_paren(values: tuple[str, ...]) -> int:
    depth = 0
    for idx in range(len(values) - 1, -1, -1):
        value = values[idx]
        if value == ")":
            depth += 1
        elif value == "(":
            if depth == 0:
                return idx
            depth -= 1
    return -1


def _current_line_open_paren_delta(values: tuple[str, ...]) -> int:
    depth = 0
    for value in values:
        if value == "(":
            depth += 1
        elif value == ")":
            depth = max(0, depth - 1)
    return depth


def _top_level_comma_after_open(values: tuple[str, ...], open_index: int) -> bool:
    depth = 0
    for value in values[open_index + 1 :]:
        if value == "(":
            depth += 1
        elif value == ")":
            depth = max(0, depth - 1)
        elif value == "," and depth == 0:
            return True
    return False


def source_condition_operation_exploration_tokens(
    values: tuple[str, ...],
    *,
    operation_tags: set[str],
    loop_target: str = "",
    default_arg: str = "",
) -> list[str]:
    if not values or not operation_tags:
        return []
    if not current_line_tail_needs_operand(list(values)):
        return []
    if len(values) >= 2 and values[1] == "=" and not loop_target:
        return []
    choices: list[str] = []
    if "op_gcd_reduce" in operation_tags:
        choices.extend(["NAME:gcd", "NAME:abs"])
    if "op_abs_tolerance_filter" in operation_tags:
        choices.append("NAME:abs")
    if "op_abs_positive_filter" in operation_tags:
        choices.append("NAME:abs")
    if "op_windowed_delta" in operation_tags:
        choices.extend(["NAME:abs", "NAME:min", "NAME:max"])
    if "op_clip_to_range" in operation_tags:
        choices.extend(["NAME:min", "NAME:max"])
    if "op_round_values" in operation_tags:
        choices.append("NAME:round")
    if "op_numeric_summary" in operation_tags:
        choices.extend(["NAME:round", "NAME:abs", "NAME:min", "NAME:max", "NAME:sum"])
    if values and values[0] == "if" and loop_target:
        choices.append(f"NAME:{loop_target}")
    if values and values[0] == "if" and default_arg:
        choices.append(f"NAME:{default_arg}")
    return list(dict.fromkeys(choices))[:5]


def source_condition_graph_walk_evidence_for_body(body: str, expectation: dict[str, Any]) -> dict[str, Any]:
    if not bool(expectation.get("requires_graph_walk_evidence")):
        return {
            "enabled": False,
            "has_graph_walk_evidence": False,
            "candidate_generation_credit": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    function = parsed_decode_guard_function(body, allowed_names=None)
    loop_count = 0
    comparison_count = 0
    branch_count = 0
    subscript_count = 0
    mutation_call_count = 0
    traversal_call_count = 0
    graph_state_name_count = 0
    graph_state_names = {
        "adj",
        "edges",
        "frontier",
        "graph",
        "neighbors",
        "next_nodes",
        "queue",
        "seen",
        "stack",
        "visited",
        "dist",
        "distance",
        "distances",
    }
    call_names: Counter[str] = Counter()
    assigned_names: Counter[str] = Counter()
    if function is not None:
        for node in ast.walk(function):
            if isinstance(node, (ast.For, ast.While, ast.comprehension)):
                loop_count += 1
            elif isinstance(node, ast.Compare):
                comparison_count += 1
            elif isinstance(node, ast.If):
                branch_count += 1
            elif isinstance(node, ast.Subscript):
                subscript_count += 1
            elif isinstance(node, ast.Call):
                call_name = action_trace_call_name(node)
                if call_name:
                    call_names[call_name] += 1
                    basename = call_name.rsplit(".", 1)[-1]
                    if basename in LOOP_PLAN_MUTATION_METHODS:
                        mutation_call_count += 1
                    if basename in {"append", "add", "extend", "get", "items", "pop", "popleft", "setdefault", "update"}:
                        traversal_call_count += 1
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                assigned_names[node.id] += 1
                lowered = node.id.lower()
                if lowered in graph_state_names or any(part in lowered for part in ("queue", "visit", "frontier", "neighbor", "dist")):
                    graph_state_name_count += 1
    has_state = bool(
        graph_state_name_count
        or mutation_call_count
        or traversal_call_count
        or {"get", "items", "pop", "popleft"} & {name.rsplit(".", 1)[-1] for name in call_names}
    )
    has_walk_shape = bool(loop_count and has_state and (branch_count or comparison_count or subscript_count or traversal_call_count))
    return {
        "enabled": True,
        "policy": "prompt_visible_graph_walk_evidence_v1",
        "has_graph_walk_evidence": has_walk_shape,
        "loop_count": loop_count,
        "comparison_count": comparison_count,
        "branch_count": branch_count,
        "subscript_count": subscript_count,
        "mutation_call_count": mutation_call_count,
        "traversal_call_count": traversal_call_count,
        "graph_state_name_count": graph_state_name_count,
        "assigned_graph_state_names": dict(sorted(assigned_names.items())),
        "call_counts": dict(sorted(call_names.items())),
        "score_semantics": (
            "Task-blind graph-walk evidence from decoded candidate AST and prompt-visible graph intent/type "
            "tags only. It requires traversal-shaped control flow plus graph state or queue/visited-style "
            "operations. It is ranking/residual evidence, not a renderer and not candidate-generation credit."
        ),
        "candidate_generation_credit": 0,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
    }


def source_condition_operation_evidence_for_body(body: str, expectation: dict[str, Any]) -> dict[str, Any]:
    operation_tags = {str(item) for item in list(expectation.get("operation_tags") or []) if str(item)}
    if not operation_tags:
        return {
            "enabled": False,
            "operation_tags": [],
            "has_operation_evidence": False,
            "candidate_generation_credit": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    function = parsed_decode_guard_function(body, allowed_names=None)
    call_names: Counter[str] = Counter()
    comparison_count = 0
    arithmetic_count = 0
    branch_count = 0
    loop_count = 0
    subscript_count = 0
    mutation_call_count = 0
    if function is not None:
        for node in ast.walk(function):
            if isinstance(node, ast.Call):
                call_name = action_trace_call_name(node)
                if call_name:
                    call_names[call_name] += 1
                    if call_name.rsplit(".", 1)[-1] in {"add", "append", "extend", "setdefault", "update"}:
                        mutation_call_count += 1
            elif isinstance(node, ast.Compare):
                comparison_count += 1
            elif isinstance(node, ast.BinOp):
                arithmetic_count += 1
            elif isinstance(node, ast.If):
                branch_count += 1
            elif isinstance(node, (ast.For, ast.While, ast.comprehension)):
                loop_count += 1
            elif isinstance(node, ast.Subscript):
                subscript_count += 1
    basenames = {name.rsplit(".", 1)[-1] for name in call_names}
    hit_tags: list[str] = []
    if "op_clip_to_range" in operation_tags and ({"min", "max"} <= basenames or comparison_count or branch_count):
        hit_tags.append("op_clip_to_range")
    if "op_round_values" in operation_tags and "round" in basenames:
        hit_tags.append("op_round_values")
    if "op_numeric_summary" in operation_tags and ({"round", "abs", "min", "max", "sum", "gcd"} & basenames or arithmetic_count):
        hit_tags.append("op_numeric_summary")
    if "op_gcd_reduce" in operation_tags and "gcd" in basenames:
        hit_tags.append("op_gcd_reduce")
    if "op_abs_tolerance_filter" in operation_tags and "abs" in basenames and comparison_count:
        hit_tags.append("op_abs_tolerance_filter")
    if "op_abs_positive_filter" in operation_tags and ("abs" in basenames or comparison_count or branch_count):
        hit_tags.append("op_abs_positive_filter")
    if (
        "op_windowed_delta" in operation_tags
        and (arithmetic_count or "abs" in basenames)
        and (subscript_count or loop_count or {"range", "zip", "enumerate"} & basenames)
    ):
        hit_tags.append("op_windowed_delta")
    if "op_threshold_filter" in operation_tags and (comparison_count or branch_count):
        hit_tags.append("op_threshold_filter")
    if "op_query_key_value_parse" in operation_tags and {"split", "splitlines", "parse_qs"} & basenames:
        hit_tags.append("op_query_key_value_parse")
    if "op_run_length_encode" in operation_tags and loop_count and (comparison_count or branch_count) and mutation_call_count:
        hit_tags.append("op_run_length_encode")
    if "op_graph_walk" in operation_tags:
        graph_expectation = dict(expectation)
        graph_expectation["requires_graph_walk_evidence"] = True
        graph_evidence = source_condition_graph_walk_evidence_for_body(body, graph_expectation)
        if bool(graph_evidence.get("has_graph_walk_evidence")):
            hit_tags.append("op_graph_walk")
    recognized_tags = sorted(
        operation_tags
        & {
            "op_abs_tolerance_filter",
            "op_abs_positive_filter",
            "op_clip_to_range",
            "op_graph_walk",
            "op_gcd_reduce",
            "op_numeric_summary",
            "op_query_key_value_parse",
            "op_round_values",
            "op_run_length_encode",
            "op_threshold_filter",
            "op_windowed_delta",
        }
    )
    hit_tag_set = set(hit_tags)
    missing_tags = [tag for tag in recognized_tags if tag not in hit_tag_set]
    has_required_operation_evidence = bool(recognized_tags) and not missing_tags
    return {
        "enabled": True,
        "policy": "prompt_visible_operation_evidence_v1",
        "operation_tags": sorted(operation_tags),
        "hit_operation_tags": sorted(set(hit_tags)),
        "recognized_operation_tags": recognized_tags,
        "missing_operation_tags": missing_tags,
        "has_operation_evidence": has_required_operation_evidence,
        "call_counts": dict(sorted(call_names.items())),
        "comparison_count": comparison_count,
        "arithmetic_count": arithmetic_count,
        "branch_count": branch_count,
        "loop_count": loop_count,
        "subscript_count": subscript_count,
        "mutation_call_count": mutation_call_count,
        "score_semantics": (
            "Task-blind operation evidence from decoded candidate AST and prompt-visible operation tags "
            "only. This is ranking/residual evidence, not a renderer and not candidate-generation credit."
        ),
        "candidate_generation_credit": 0,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
    }


def source_condition_body_has_truthiness_guard(body: str, arg_name: str) -> bool:
    if not arg_name:
        return False
    try:
        parsed = ast.parse(static_guard_candidate_code(str(body or ""), allowed_names={arg_name}))
    except SyntaxError:
        return False
    function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    if function is None:
        return False
    for node in ast.walk(function):
        if isinstance(node, ast.If) and bool_expr_has_and_name(node.test, arg_name):
            return True
    return False


def source_condition_body_has_default_return(body: str, arg_name: str) -> bool:
    if not arg_name:
        return False
    try:
        parsed = ast.parse(static_guard_candidate_code(str(body or ""), allowed_names={arg_name}))
    except SyntaxError:
        return False
    function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    if function is None:
        return False
    for stmt in function.body:
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name) and stmt.value.id == arg_name:
            return True
    return False


def source_condition_body_has_sequence_type_guard(body: str, arg_name: str, expected_types: list[str]) -> bool:
    if not arg_name or not expected_types:
        return False
    try:
        parsed = ast.parse(static_guard_candidate_code(str(body or ""), allowed_names={arg_name}))
    except SyntaxError:
        return False
    function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    if function is None:
        return False
    expected = set(expected_types)
    for node in ast.walk(function):
        if isinstance(node, ast.If) and isinstance_guard_covers_types(node.test, arg_name, expected):
            return True
    return False


def isinstance_guard_covers_types(node: ast.AST, arg_name: str, expected_types: set[str]) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if not isinstance(child.func, ast.Name) or child.func.id != "isinstance" or len(child.args) < 2:
            continue
        if not isinstance(child.args[0], ast.Name) or child.args[0].id != arg_name:
            continue
        seen: set[str] = set()
        type_node = child.args[1]
        if isinstance(type_node, ast.Name):
            seen.add(type_node.id)
        elif isinstance(type_node, (ast.Tuple, ast.List)):
            for item in type_node.elts:
                if isinstance(item, ast.Name):
                    seen.add(item.id)
        if expected_types.issubset(seen):
            return True
    return False


def source_condition_body_has_guarded_first_item_return(body: str, arg_name: str) -> bool:
    if not arg_name:
        return False
    try:
        parsed = ast.parse(static_guard_candidate_code(str(body or ""), allowed_names={arg_name}))
    except SyntaxError:
        return False
    function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    if function is None:
        return False
    for node in ast.walk(function):
        if not isinstance(node, ast.If) or not bool_expr_has_and_name(node.test, arg_name):
            continue
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Return) and expr_is_first_index_of_name(stmt.value, arg_name):
                return True
    return False


def source_condition_body_has_guarded_data_return(body: str, arg_name: str) -> bool:
    if not arg_name:
        return False
    try:
        parsed = ast.parse(static_guard_candidate_code(str(body or ""), allowed_names={arg_name}))
    except SyntaxError:
        return False
    function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    if function is None:
        return False
    for node in ast.walk(function):
        if not isinstance(node, ast.If) or not bool_expr_has_and_name(node.test, arg_name):
            continue
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Return) and expr_contains_name(stmt.value, arg_name):
                return True
    return False


def expr_is_first_index_of_name(node: ast.AST | None, name: str) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    if not isinstance(node.value, ast.Name) or node.value.id != name:
        return False
    index = node.slice
    if isinstance(index, ast.Constant):
        return index.value == 0
    return False


def bool_expr_has_and_name(node: ast.AST, name: str) -> bool:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        return any(bool_expr_is_direct_name_operand(value, name) for value in node.values)
    return any(bool_expr_has_and_name(child, name) for child in ast.iter_child_nodes(node))


def bool_expr_is_direct_name_operand(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def bool_expr_contains_name(node: ast.AST, name: str) -> bool:
    if isinstance(node, ast.Name) and node.id == name:
        return True
    return any(bool_expr_contains_name(child, name) for child in ast.iter_child_nodes(node))


def expr_contains_name(node: ast.AST | None, name: str) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Name) and node.id == name:
        return True
    return any(expr_contains_name(child, name) for child in ast.iter_child_nodes(node))


def source_condition_candidate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    adequacy_rows = [dict_or_empty(row.get("source_condition_adequacy")) for row in rows]
    enabled = [row for row in adequacy_rows if bool(row.get("enabled"))]
    adequate = [row for row in enabled if bool(row.get("adequate"))]
    missing_counts: Counter[str] = Counter()
    operation_tag_counts: Counter[str] = Counter()
    hit_operation_tag_counts: Counter[str] = Counter()
    missing_operation_tag_counts: Counter[str] = Counter()
    for row in enabled:
        missing_counts.update(str(item) for item in list(row.get("missing_features") or []))
        operation_evidence = dict_or_empty(row.get("operation_evidence"))
        operation_tag_counts.update(str(item) for item in list(operation_evidence.get("operation_tags") or []))
        hit_operation_tag_counts.update(str(item) for item in list(operation_evidence.get("hit_operation_tags") or []))
        missing_operation_tag_counts.update(
            str(item) for item in list(operation_evidence.get("missing_operation_tags") or [])
        )
    return {
        "policy": "prompt_visible_empty_default_condition_candidate_summary_v1",
        "candidate_rows": len(rows),
        "enabled_candidate_rows": len(enabled),
        "adequate_candidate_rows": len(adequate),
        "adequate_candidate_rate": round(len(adequate) / max(1, len(enabled)), 6),
        "missing_feature_counts": dict(sorted(missing_counts.items())),
        "operation_tag_counts": dict(sorted(operation_tag_counts.items())),
        "hit_operation_tag_counts": dict(sorted(hit_operation_tag_counts.items())),
        "missing_operation_tag_counts": dict(sorted(missing_operation_tag_counts.items())),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def body_action_trace_candidate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    traces = [dict_or_empty(row.get("body_action_trace")) for row in rows]
    enabled = [trace for trace in traces if bool(trace.get("enabled"))]
    parse_ok = [trace for trace in enabled if bool(trace.get("parse_ok"))]
    mismatch_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    call_counts: Counter[str] = Counter()
    plan_counts: Counter[str] = Counter()
    expected_update_counts: Counter[str] = Counter()
    expected_finalizer_counts: Counter[str] = Counter()
    expected_return_shape_counts: Counter[str] = Counter()
    assigned_name_counts: Counter[str] = Counter()
    loop_target_counts: Counter[str] = Counter()
    return_inside_loop_count = 0
    shallow_identity_accumulation_count = 0
    unreachable_loop_statement_count = 0
    examples: list[dict[str, Any]] = []
    for row, trace in zip(rows, traces):
        if not bool(trace.get("enabled")):
            continue
        plan = str(trace.get("plan") or "")
        if plan:
            plan_counts[plan] += 1
        for label in list(trace.get("mismatch_labels") or []):
            mismatch_counts[str(label)] += 1
        for key, value in dict_or_empty(trace.get("operation_counts")).items():
            try:
                operation_counts[str(key)] += int(value or 0)
            except (TypeError, ValueError):
                continue
        for key, value in dict_or_empty(trace.get("call_counts")).items():
            try:
                call_counts[str(key)] += int(value or 0)
            except (TypeError, ValueError):
                continue
        expected_update_counts.update(str(item) for item in list(trace.get("expected_update_ops") or []))
        expected_finalizer_counts.update(str(item) for item in list(trace.get("expected_finalizers") or []))
        expected_return_shape_counts.update(str(item) for item in list(trace.get("expected_return_shapes") or []))
        assigned_name_counts.update(str(item) for item in list(trace.get("assigned_names") or []))
        loop_target_counts.update(str(item) for item in list(trace.get("loop_targets") or []))
        try:
            return_inside_loop_count += int(trace.get("return_inside_loop_count") or 0)
        except (TypeError, ValueError):
            pass
        try:
            shallow_identity_accumulation_count += int(trace.get("shallow_identity_accumulation_count") or 0)
        except (TypeError, ValueError):
            pass
        try:
            unreachable_loop_statement_count += int(trace.get("unreachable_loop_statement_count") or 0)
        except (TypeError, ValueError):
            pass
        labels = [str(item) for item in list(trace.get("mismatch_labels") or [])]
        if labels and len(examples) < 8:
            examples.append(
                {
                    "task_id": str(row.get("task_id") or row.get("source_task_id") or ""),
                    "family": str(row.get("category") or row.get("family") or ""),
                    "candidate_sha256": str(row.get("candidate_sha256") or row.get("decoded_token_sha256") or ""),
                    "plan": plan,
                    "mismatch_labels": labels,
                    "operation_counts": dict_or_empty(trace.get("operation_counts")),
                    "call_counts": dict_or_empty(trace.get("call_counts")),
                }
            )
    return {
        "policy": "strict_generator_body_action_trace_candidate_summary_v1",
        "candidate_rows": len(rows),
        "enabled_candidate_rows": len(enabled),
        "parse_ok_candidate_rows": len(parse_ok),
        "parse_ok_candidate_rate": round(len(parse_ok) / max(1, len(enabled)), 6),
        "mismatch_candidate_rows": sum(1 for trace in enabled if list(trace.get("mismatch_labels") or [])),
        "mismatch_label_counts": dict(sorted(mismatch_counts.items())),
        "operation_counts": dict(sorted(operation_counts.items())),
        "call_counts": dict(sorted(call_counts.items())),
        "plan_counts": dict(sorted(plan_counts.items())),
        "expected_update_op_counts": dict(sorted(expected_update_counts.items())),
        "expected_finalizer_counts": dict(sorted(expected_finalizer_counts.items())),
        "expected_return_shape_counts": dict(sorted(expected_return_shape_counts.items())),
        "assigned_name_counts": dict(sorted(assigned_name_counts.items())),
        "loop_target_counts": dict(sorted(loop_target_counts.items())),
        "return_inside_loop_count": return_inside_loop_count,
        "shallow_identity_accumulation_count": shallow_identity_accumulation_count,
        "unreachable_loop_statement_count": unreachable_loop_statement_count,
        "mismatch_examples": examples,
        "score_semantics": (
            "Task-blind AST/action trace summary over already-decoded candidate bodies. It is residual "
            "mining evidence for learned statement-order and loop/finalizer quality, not candidate "
            "generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }
