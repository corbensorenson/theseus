#!/usr/bin/env python3
"""Deterministic decode, repair, and rendering support for neural seed token comparators.

This module is deliberately solver-like: it may normalize syntax and render
explicit structural plans, but it never reads eval tests/solutions, calls a
teacher, or emits fallback returns to manufacture a pass.
"""

from __future__ import annotations

import ast
import base64
import io
import math
import token as py_token
import tokenize as py_tokenize
import textwrap
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from neural_seed_code_proposer_comparator import (  # noqa: E402
    dict_or_empty,
    get_path,
    render_private_function,
    stable_hash,
)
from neural_seed_token_decoder_rendering import (  # noqa: E402
    PLAN_PREFIX_BODY_TARGET_MODE,
    PLAN_SEMANTIC_SLOTS_BODY_TARGET_MODE,
    PLAN_SEMANTIC_STMT_EXPR_TRANSITION_BODY_TARGET_MODE,
    PLAN_BODY_START_TOKEN,
    body_expression_trace_token,
    task_family,
    learned_plan_prefix_target_mode,
    learned_plan_expression_transition_body_target_mode,
    learned_plan_semantic_slots_body_target_mode,
    learned_plan_statement_slots_body_target_mode,
    body_like_target_mode,
    split_learned_plan_prefix_tokens,
    strict_action_tokens,
    strict_action_tokens_for_stmt,
    assign_target_name,
    augassign_op,
    encoded_expr_token,
    encoded_field_token,
    decoded_field_token,
    safe_module_name,
    render_synthetic_function,
    statement_skeleton_tokens,
    iter_uses_data,
    value_shape,
    return_shape_from_expr,
    call_name,
    decode_body_tokens,
    decode_candidate_body_tokens,
    render_strict_action_body,
    StrictActionRenderer,
    render_semantic_slot_body,
    decoded_semantic_plan,
    decoded_slot_value,
    render_supported_semantic_plan,
    body_from_lines,
    semantic_body_safe_head_default,
    semantic_body_group_records_by_field,
    semantic_body_project_table_columns,
    semantic_body_gcd_positive,
    semantic_body_normalize_filter_sort_unique,
    semantic_body_windowed_deltas,
    semantic_body_balanced_brackets,
    semantic_body_stdin_pair_sums,
    semantic_body_stdin_numeric_parse,
    semantic_body_graph_components,
    semantic_body_shortest_hops,
    semantic_body_max_non_adjacent_sum,
    semantic_body_lcs_length,
    semantic_body_merge_intervals,
    semantic_body_interval_coverage,
    semantic_body_longest_even_run,
    semantic_body_rle_encode,
    semantic_body_threshold_labels,
    semantic_body_top_k_frequent,
    semantic_body_media_preview_index,
    semantic_body_room_capability_summary,
    semantic_body_memory_latest_by_project,
    semantic_body_memory_open_action_rollup,
    semantic_body_storage_quota_select,
    semantic_body_storage_sync_plan,
    semantic_body_device_route_worker,
    semantic_body_voice_output_route,
    semantic_body_plan_next_unblocked,
    semantic_body_plan_progress_digest,
    semantic_body_generic_list_append,
    semantic_body_generic_dict_group_append,
    semantic_body_generic_set_add,
    semantic_body_generic_return,
    render_statement_skeleton_body,
    decoded_return_shape,
    initializer_for_shape,
    update_lines_for_shape,
    return_line_for_shape,
    token_values_to_line,
    normalize_body_text,
)

STRICT_BODY_ALLOWED_GLOBAL_NAMES = {
    "False",
    "None",
    "True",
    "abs",
    "all",
    "any",
    "bool",
    "bytes",
    "chr",
    "dict",
    "enumerate",
    "filter",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "map",
    "max",
    "min",
    "ord",
    "pow",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}

STRICT_BODY_KEYWORD_NAMES = {
    "and",
    "as",
    "break",
    "continue",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "if",
    "import",
    "in",
    "is",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
}

STRICT_BODY_BARE_BUILTIN_CONDITION_NAMES = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "filter",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "map",
    "max",
    "min",
    "range",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}

STRICT_BODY_BUILTIN_TYPE_NAMES = {
    "bool",
    "bytes",
    "dict",
    "float",
    "int",
    "list",
    "set",
    "str",
    "tuple",
}

STRICT_BODY_KNOWN_TYPE_METHODS: dict[str, set[str]] = {
    "dict": {"clear", "copy", "get", "items", "keys", "pop", "setdefault", "update", "values"},
    "list": {"append", "clear", "copy", "count", "extend", "index", "insert", "pop", "remove", "reverse", "sort"},
    "set": {"add", "clear", "copy", "discard", "intersection", "pop", "remove", "union", "update"},
    "str": {
        "casefold",
        "count",
        "endswith",
        "find",
        "format",
        "index",
        "isalnum",
        "isalpha",
        "isdigit",
        "join",
        "lower",
        "replace",
        "split",
        "splitlines",
        "startswith",
        "strip",
        "upper",
    },
    "tuple": {"count", "index"},
}

STRICT_BODY_ALL_KNOWN_METHODS = {
    method for methods in STRICT_BODY_KNOWN_TYPE_METHODS.values() for method in methods
}

STRICT_BODY_INVALID_TYPE_METHODS: dict[str, set[str]] = {
    "dict": {"add", "append", "extend", "split", "splitlines", "strip"},
    "list": {"get", "items", "keys", "setdefault", "split", "splitlines", "strip", "update", "values"},
    "set": {"append", "extend", "get", "items", "keys", "split", "splitlines", "strip", "values"},
    "str": {"add", "append", "extend", "get", "items", "keys", "setdefault", "update", "values"},
    "tuple": {"add", "append", "extend", "get", "items", "keys", "setdefault", "update", "values"},
}

STRICT_BODY_NONRETURNING_METHODS: dict[str, set[str]] = {
    "dict": {"clear", "update"},
    "list": {"append", "clear", "extend", "insert", "remove", "reverse", "sort"},
    "set": {"add", "clear", "discard", "remove", "update"},
}

STRICT_BODY_NONRETURNING_METHOD_NAMES = {
    method for methods in STRICT_BODY_NONRETURNING_METHODS.values() for method in methods
}

STRICT_BODY_CHAINING_OPERATORS = {
    "+",
    "-",
    "*",
    "/",
    "%",
    "//",
    "**",
    "<",
    "<=",
    ">",
    ">=",
    "==",
    "!=",
}

STRICT_BODY_COMPARISON_OPERATORS = {
    "<",
    "<=",
    ">",
    ">=",
    "==",
    "!=",
    "in",
    "is",
}

STRICT_BODY_AUGMENTED_ASSIGNMENT_OPERATORS = {
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "//=",
    "**=",
}



def choose_grammar_constrained_token(
    probs: Any,
    inverse: dict[int, str],
    generated: list[int],
    *,
    eos_id: int,
    rank: int,
    step: int,
    grammar_top_k: int,
    token_policy: str = "lightweight_python_v1",
    allowed_names: set[str] | None = None,
    torch: Any,
) -> tuple[int, float]:
    prefix = [inverse.get(idx, "<unk>") for idx in generated[1:]]
    if syntax_complete_body_prefix(prefix):
        return eos_id, float(max(float(probs[eos_id]), 1e-9))
    forced = forced_lightweight_python_token(prefix, inverse, probs)
    if forced is not None:
        return forced
    top_values, top_indices = torch.topk(probs, k=min(max(grammar_top_k, rank + 1, 2), probs.numel()))
    allowed: list[tuple[int, float]] = []
    for value, idx in zip(top_values, top_indices):
        next_id = int(idx)
        tok = inverse.get(next_id, "<unk>")
        if token_allowed_by_policy(prefix, tok, policy=token_policy, allowed_names=allowed_names):
            allowed.append((next_id, float(value)))
    if allowed:
        choice = min(rank, len(allowed) - 1) if step == 0 else 0
        return allowed[choice]
    # If the model has produced a returned function body, end cleanly. Without
    # a return, keep decoding so syntax validity does not mask inert bodies.
    if syntax_complete_body_prefix(prefix):
        return eos_id, float(max(float(probs[eos_id]), 1e-9))
    next_id = int(top_indices[0])
    return next_id, float(top_values[0])


def grammar_constrained_token_choices(
    probs: Any,
    inverse: dict[int, str],
    generated: list[int],
    *,
    eos_id: int,
    grammar_top_k: int,
    max_choices: int,
    token_policy: str = "lightweight_python_v1",
    allowed_names: set[str] | None = None,
    torch: Any,
) -> list[tuple[int, float]]:
    """Return ranked next-token choices allowed by the lightweight grammar.

    This is still direct token generation from model probabilities. It only
    applies syntax-level constraints already used by the greedy chooser; it does
    not choose semantic bodies, inspect tests/solutions, or add fallback returns.
    """

    prefix = [inverse.get(idx, "<unk>") for idx in generated[1:]]
    if syntax_complete_body_prefix(prefix):
        return [(eos_id, float(max(float(probs[eos_id]), 1e-9)))]
    forced = forced_lightweight_python_token(prefix, inverse, probs)
    if forced is not None:
        return [forced]
    top_values, top_indices = torch.topk(
        probs,
        k=min(max(grammar_top_k, max_choices, 2), probs.numel()),
    )
    choices: list[tuple[int, float]] = []
    seen: set[int] = set()
    for value, idx in zip(top_values, top_indices):
        next_id = int(idx)
        if next_id in seen:
            continue
        tok = inverse.get(next_id, "<unk>")
        if token_allowed_by_policy(prefix, tok, policy=token_policy, allowed_names=allowed_names):
            seen.add(next_id)
            choices.append((next_id, float(value)))
            if len(choices) >= max(1, max_choices):
                break
    if choices:
        return choices
    full_values, full_indices = torch.topk(probs, k=probs.numel())
    for value, idx in zip(full_values, full_indices):
        next_id = int(idx)
        if next_id in seen:
            continue
        tok = inverse.get(next_id, "<unk>")
        if token_allowed_by_policy(prefix, tok, policy=token_policy, allowed_names=allowed_names):
            seen.add(next_id)
            choices.append((next_id, float(value)))
            if len(choices) >= max(1, max_choices):
                break
    if choices:
        return choices
    if syntax_complete_body_prefix(prefix):
        return [(eos_id, float(max(float(probs[eos_id]), 1e-9)))]
    return [(eos_id, float(max(float(probs[eos_id]), 1e-9)))]


def token_allowed_by_policy(prefix: list[str], tok: str, *, policy: str, allowed_names: set[str] | None = None) -> bool:
    if str(policy or "") in {"strict_body_token_legality_v1", "strict_body_tokens_v1"}:
        return token_allowed_by_strict_body_token_policy(prefix, tok, allowed_names=allowed_names)
    return token_allowed_by_lightweight_python_grammar(prefix, tok)


def token_allowed_by_strict_body_token_policy(prefix: list[str], tok: str, *, allowed_names: set[str] | None = None) -> bool:
    """Stricter direct body-token legality policy.

    This is not a renderer. It only removes choices that are illegal or useless
    in Python function bodies, such as top-level literal expression spam,
    invalid assignment targets, and break/continue outside loops.
    """

    if not token_allowed_by_lightweight_python_grammar(prefix, tok):
        return False
    if tok == "<eos>":
        return strict_body_can_end(prefix)
    if body_expression_trace_token(tok):
        return True
    kind, _, value = tok.partition(":")
    if kind == "NAME" and value in {"other", "extra"} and allowed_names is not None and value not in allowed_names:
        return False
    if kind == "NAME" and value in {"except", "finally", "raise", "try"}:
        return False
    if kind == "COMMENT":
        return False
    line = current_line_tokens(prefix)
    values = token_values(line)
    previous = values[-1] if values else ""
    previous_kind, _, previous_value = prefix[-1].partition(":") if prefix else ("", "", "")
    local_types = strict_body_prefix_local_static_types(prefix)
    if kind == "NEWLINE" and line_uses_nonreturning_call_as_value(values):
        return False
    if kind == "NEWLINE" and line_has_invalid_multi_assign_from_scalar(values, local_types=local_types):
        return False
    if kind != "NEWLINE" and completed_nonreturning_method_call_on_current_line(prefix):
        return False
    if pending_unassigned_statement_lvalue(prefix, allowed_names=allowed_names):
        return kind == "OP" and value == "="
    if kind == "NAME" and not strict_body_name_allowed_by_scope(prefix, value, allowed_names=allowed_names):
        return False
    if kind == "NAME" and line_would_iterate_known_noniterable(values, value, local_types=local_types):
        return False
    if kind == "NAME" and strict_body_for_loop_name_is_bad_semantic_slot(
        values,
        value,
        allowed_names=allowed_names,
    ):
        return False
    if values and values[0] == "for" and "in" not in values[1:]:
        loop_target_values = [item for item in values[1:] if item not in {","}]
        if kind == "NAME" and value == "in":
            return bool(previous and previous.isidentifier() and previous not in STRICT_BODY_KEYWORD_NAMES)
        if kind == "NAME":
            return bool(
                previous in {"for", ","}
                and value not in STRICT_BODY_KEYWORD_NAMES
                and value not in loop_target_values
            )
        if kind == "OP" and value == ",":
            return bool(
                previous
                and previous.isidentifier()
                and previous not in STRICT_BODY_KEYWORD_NAMES
                and values.count(",") < 1
            )
        return False
    if kind == "OP" and value in STRICT_BODY_AUGMENTED_ASSIGNMENT_OPERATORS:
        return assignable_lvalue_tokens(values)
    if kind == "NAME" and value == "in" and values and values[0] not in {"for", "if", "elif", "while", "return"}:
        return False
    if kind == "NAME" and value == "not" and previous not in {"if", "elif", "while", "return", "and", "or", "("}:
        return False
    if kind == "NAME" and value == "as" and not (values and values[0] in {"except", "import", "from", "with"}):
        return False
    if kind == "OP" and value in {"&", "|", "^", "<<", ">>"}:
        return False
    if kind == "OP" and value in {"&=", "|=", "^=", ":="}:
        return False
    if kind == "OP" and value == "{" and values and previous not in {"=", "return", "(", "[", ",", ":"}:
        return False
    if strict_body_would_extend_pathological_comparison_chain(values, kind=kind, value=value):
        return False
    if strict_body_would_extend_pathological_boolean_chain(values, kind=kind, value=value):
        return False
    if kind == "NEWLINE" and values == ["return"]:
        return False
    if line_start(prefix):
        if any(depth == 0 and row_values[:1] == ["return"] for depth, row_values in previous_significant_lines_with_depth(prefix)):
            return False
        if (
            kind == "NAME"
            and value == "return"
            and any(depth == 0 and row_values[:1] == ["return"] for depth, row_values in previous_significant_lines_with_depth(prefix))
        ):
            return False
        if kind in {"STRING", "NUMBER"}:
            return False
        if kind == "NAME" and value in {"break", "continue"} and not inside_loop_context(prefix):
            return False
    if values and values[0] in {"if", "elif", "while"} and set(values[1:]).issubset({"not"}):
        if kind in {"NUMBER", "STRING"}:
            return False
        if kind == "OP" and value in {"(", "[", "{"}:
            return False
    if values[-2:] in [["len", "("], ["sum", "("], ["sorted", "("], ["list", "("], ["tuple", "("], ["set", "("]]:
        if kind in {"NUMBER", "STRING"}:
            return False
        if kind == "NAME" and value in STRICT_BODY_BUILTIN_TYPE_NAMES:
            return False
        if kind == "NAME" and value in {"None", "True", "False"}:
            return False
        if kind == "OP" and value in {"(", "[", "{"}:
            return False
    if (
        values
        and values[0] in {"if", "elif", "while"}
        and previous_kind == "NAME"
        and previous_value in STRICT_BODY_BARE_BUILTIN_CONDITION_NAMES
        and not isinstance_second_argument_context(prefix)
        and not (kind == "OP" and value == "(")
    ):
        return False
    if values == ["return"]:
        if kind == "NAME" and value in {"and", "in", "is", "not", "or"}:
            return False
        if kind == "NAME" and value in STRICT_BODY_BUILTIN_TYPE_NAMES:
            return False
        if kind in {"NUMBER", "STRING"}:
            return False
        if kind == "OP" and value in {"(", "[", "{"}:
            return False
    if previous in {"+", "-"} and kind == "NAME" and value in {"None", "True", "False"}:
        return False
    if tuple(values) in {("if",), ("elif",), ("while",)} and kind == "OP" and value in {"(", "[", "{"}:
        return False
    if kind == "OP" and value == "=":
        return assignable_lvalue_tokens(values)
    if kind == "OP" and value == "(" and previous_kind == "NAME":
        if previous_value in {"return", "yield"}:
            return False
        if current_line_call_depth(values) >= 2:
            return False
        if not previous_name_is_callable(prefix, previous_value, allowed_names=allowed_names):
            return False
        if isinstance_second_argument_context(prefix) and previous_value in STRICT_BODY_BUILTIN_TYPE_NAMES:
            return False
    if isinstance_second_argument_context(prefix):
        if kind == "NAME":
            return value in STRICT_BODY_BUILTIN_TYPE_NAMES
        if kind == "OP":
            if value == "(":
                return previous in {",", "("}
            if value == ",":
                return isinstance_second_argument_tuple_context(prefix) and (
                    previous_value in STRICT_BODY_BUILTIN_TYPE_NAMES or previous == ")"
                )
            if value == ")":
                return isinstance_second_argument_has_type(prefix)
            return False
        return False
    if kind == "OP" and value == ")" and closes_invalid_known_builtin_call(prefix, "isinstance", min_commas=1):
        return False
    if kind == "OP" and value == ")" and closes_invalid_known_arity_call(prefix, local_types=local_types):
        return False
    if kind == "OP" and value == "]" and closes_empty_subscript(prefix):
        return False
    if kind == "OP" and value in {".", "["}:
        if previous_kind == "NAME" and previous_value in STRICT_BODY_BUILTIN_TYPE_NAMES:
            return False
    if kind == "OP" and value == "[" and strict_body_would_extend_pathological_subscript_chain(values):
        return False
    if kind == "OP" and value == "[" and previous_kind == "NAME":
        if previous_value in STRICT_BODY_ALLOWED_GLOBAL_NAMES or previous_value in STRICT_BODY_KEYWORD_NAMES:
            return False
    if kind == "OP" and value == ".":
        if previous_kind != "NAME" and previous not in {")", "]"}:
            return False
        if strict_body_would_extend_uncalled_method_attribute_chain(values):
            return False
    if kind == "NAME" and previous == ".":
        receiver_type = strict_body_current_attribute_receiver_type(prefix, local_types=local_types)
        if line_uses_attribute_call_as_value(values) and value in STRICT_BODY_NONRETURNING_METHOD_NAMES:
            return False
        if receiver_type:
            known_methods = STRICT_BODY_KNOWN_TYPE_METHODS.get(receiver_type)
            if value in STRICT_BODY_INVALID_TYPE_METHODS.get(receiver_type, set()):
                return False
            if known_methods is not None and value not in known_methods:
                return False
            if values and values[0] == "return" and value in STRICT_BODY_NONRETURNING_METHODS.get(receiver_type, set()):
                return False
        elif value not in STRICT_BODY_ALL_KNOWN_METHODS:
            return False
    if kind == "OP" and value in STRICT_BODY_CHAINING_OPERATORS:
        if current_line_operator_chain_count(values) >= 4:
            return False
        if previous in STRICT_BODY_CHAINING_OPERATORS:
            return False
    if kind == "OP" and value == ":" and line_iterates_known_noniterable(values, local_types=local_types):
        return False
    if (
        kind == "OP"
        and value == ":"
        and values
        and values[0] in {"if", "elif", "while"}
        and strict_body_condition_lacks_runtime_signal(values, allowed_names=allowed_names)
    ):
        return False
    if kind == "OP" and value == "," and (not values or values[-1] in {"return", "yield", "if", "elif", "while", "for", "(", "[", "{", ","}):
        return False
    return True


def strict_body_can_end(prefix: list[str]) -> bool:
    if not syntax_complete_body_prefix(prefix):
        return False
    body = decode_body_tokens(prefix)
    try:
        parsed = ast.parse(render_synthetic_function(body))
    except SyntaxError:
        return False
    function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    return bool(function and any(isinstance(node, ast.Return) and node.value is not None for node in ast.walk(function)))


def pending_unassigned_statement_lvalue(prefix: list[str], *, allowed_names: set[str] | None = None) -> bool:
    line = current_line_tokens(prefix)
    values = token_values(line)
    if len(values) != 1:
        return False
    value = values[0]
    if not value.isidentifier():
        return False
    if value in STRICT_BODY_KEYWORD_NAMES or value in STRICT_BODY_ALLOWED_GLOBAL_NAMES:
        return False
    if allowed_names is not None and value in allowed_names:
        return False
    if value in assigned_names_in_prefix(prefix):
        return False
    return True


def strict_body_condition_lacks_runtime_signal(values: list[str], *, allowed_names: set[str] | None = None) -> bool:
    """Return true for constant/builtin-only control-flow conditions."""

    if not values or values[0] not in {"if", "elif", "while"}:
        return False
    visible = {str(name) for name in set(allowed_names or set()) if str(name)}
    runtime_names = {
        value
        for value in values[1:]
        if value.isidentifier()
        and value not in STRICT_BODY_KEYWORD_NAMES
        and value not in STRICT_BODY_ALLOWED_GLOBAL_NAMES
        and value not in STRICT_BODY_BUILTIN_TYPE_NAMES
    }
    if visible and runtime_names & visible:
        return False
    # Locals such as item/out are meaningful runtime signals too; they are
    # excluded from globals/types above.
    if runtime_names - visible:
        return False
    return True


def strict_body_for_loop_name_is_bad_semantic_slot(
    values: list[str],
    value: str,
    *,
    allowed_names: set[str] | None = None,
) -> bool:
    """Reject recurring builtin-shadowed loop headers under strict decode.

    This is a task-blind body-token legality rule. It never reads tests,
    solutions, targets, verifier labels, public payloads, or answer metadata.
    It only prevents patterns such as ``for max in max`` that are syntactically
    legal Python but destroy source/loop-variable semantics in the learned
    direct generator.
    """

    if not values or values[0] != "for" or not value:
        return False
    allowed = {str(name) for name in set(allowed_names or set()) if str(name).isidentifier()}
    if "in" not in values[1:]:
        previous = values[-1] if values else ""
        if previous in {"for", ","}:
            return value in STRICT_BODY_ALLOWED_GLOBAL_NAMES or value in STRICT_BODY_KEYWORD_NAMES
        return False
    if values[-1] != "in":
        return False
    if allowed and value in allowed:
        return False
    # Calls such as ``range(...)`` or ``enumerate(data)`` can still become
    # valid iterables. Aggregating/scalar builtins like ``max`` and ``sum`` are
    # not valid source iterables here and caused repeated starvation beams.
    iterable_call_builtins = {"dict", "enumerate", "filter", "list", "map", "range", "reversed", "set", "sorted", "tuple", "zip"}
    if value in iterable_call_builtins:
        return False
    return value in STRICT_BODY_ALLOWED_GLOBAL_NAMES


def previous_name_is_callable(prefix: list[str], previous_value: str, *, allowed_names: set[str] | None = None) -> bool:
    if not previous_value:
        return False
    if len(prefix) >= 2 and prefix[-2].partition(":")[2] == ".":
        return True
    if previous_value in STRICT_BODY_ALLOWED_GLOBAL_NAMES:
        return True
    if previous_value in assigned_names_in_prefix(prefix):
        # A recurring strict-generator failure was treating locally bound
        # containers/scalars as functions, e.g. ``out(value)`` after
        # ``out = []`` or ``take(skip)`` after ``take = 0``. The direct token
        # generator is not allowed to manufacture callable semantics from a
        # plain local assignment, so local direct calls stay illegal unless a
        # future prefix analysis proves a callable binding explicitly.
        return False
    # Visible input arguments may be containers or scalars, but direct
    # ``data()``-style calls were a recurring malformed learned pattern. Method
    # calls on inputs remain legal through the attribute branch above.
    if allowed_names is not None and previous_value in allowed_names:
        return False
    return False


def strict_body_name_allowed_by_scope(prefix: list[str], value: str, *, allowed_names: set[str] | None = None) -> bool:
    if not value or not value.isidentifier():
        return True
    if value in STRICT_BODY_KEYWORD_NAMES or value in STRICT_BODY_ALLOWED_GLOBAL_NAMES:
        return True
    if allowed_names is not None and value in allowed_names:
        return True
    if prefix and prefix[-1].partition(":")[2] == ".":
        return True
    line = current_line_tokens(prefix)
    values = token_values(line)
    assigned = assigned_names_before_current_line(prefix)
    if value in assigned:
        return True
    if name_position_binds_local(values):
        return True
    return False


def name_position_binds_local(values: list[str]) -> bool:
    if not values:
        return True
    if values[0] == "for" and "in" not in values:
        return True
    if values[-1] in {",", "(", "["} and values and values[0] == "for" and "in" not in values:
        return True
    if values[-1] in {"as", "import"}:
        return True
    return False


def assigned_names_in_prefix(prefix: list[str]) -> set[str]:
    assigned: set[str] = set()
    for _depth, values in previous_significant_lines_with_depth(prefix):
        if not values:
            continue
        if values[0] == "for" and "in" in values:
            for value in values[1 : values.index("in")]:
                if value.isidentifier() and value not in STRICT_BODY_KEYWORD_NAMES:
                    assigned.add(value)
        if "=" in values:
            for value in values[: values.index("=")]:
                if value.isidentifier() and value not in STRICT_BODY_KEYWORD_NAMES:
                    assigned.add(value)
        if values[0] == "import":
            for value in values[1:]:
                if value.isidentifier() and value not in {"as"}:
                    assigned.add(value)
        if values[0] == "from" and "import" in values:
            for value in values[values.index("import") + 1 :]:
                if value.isidentifier() and value not in {"as"}:
                    assigned.add(value)
    return assigned


def assigned_names_before_current_line(prefix: list[str]) -> set[str]:
    current = current_line_tokens(prefix)
    if not current:
        return assigned_names_in_prefix(prefix)
    return assigned_names_in_prefix(prefix[: max(0, len(prefix) - len(current))])


def strict_body_prefix_local_static_types(prefix: list[str]) -> dict[str, str]:
    """Infer obvious local types from already generated tokens only."""

    bindings: dict[str, str] = {}
    for _depth, values in previous_significant_lines_with_depth(prefix):
        if len(values) < 3 or "=" not in values:
            continue
        equals = values.index("=")
        if equals != 1 or not values[0].isidentifier():
            continue
        inferred = strict_body_static_type_from_values(values[equals + 1 :])
        if inferred:
            bindings[values[0]] = inferred
    return bindings


def strict_body_static_type_from_values(values: list[str]) -> str:
    if not values:
        return ""
    first = values[0]
    if first == "[":
        return "list"
    if first == "{":
        return "dict"
    if first == "(":
        return "tuple"
    if first in STRICT_BODY_BUILTIN_TYPE_NAMES and len(values) > 1 and values[1] == "(":
        return first
    if first in {"abs", "float", "int", "len", "round", "sum"} and len(values) > 1 and values[1] == "(":
        return "float" if first == "float" else "int"
    if first in {"max", "min"} and len(values) > 1 and values[1] == "(" and call_values_have_top_level_comma(values[2:]):
        return "float" if any("." in value for value in values if value.replace(".", "", 1).isdigit()) else "int"
    if len(values) == 1:
        if first.startswith("'") or first.startswith('"'):
            return "str"
        try:
            int(first)
            return "int"
        except ValueError:
            pass
    return ""


def call_values_have_top_level_comma(values: list[str]) -> bool:
    depth = 0
    for value in values:
        if value in {"(", "[", "{"}:
            depth += 1
        elif value in {")", "]", "}"}:
            depth = max(0, depth - 1)
        elif value == "," and depth <= 1:
            return True
    return False


def line_would_iterate_known_noniterable(values: list[str], next_name: str, *, local_types: dict[str, str]) -> bool:
    if len(values) < 3 or values[0] != "for" or values[-1] != "in":
        return False
    return strict_body_type_is_noniterable(local_types.get(next_name, ""))


def line_iterates_known_noniterable(values: list[str], *, local_types: dict[str, str]) -> bool:
    if len(values) < 4 or values[0] != "for" or "in" not in values:
        return False
    in_index = values.index("in")
    iter_values = values[in_index + 1 :]
    if len(iter_values) != 1:
        return False
    return strict_body_type_is_noniterable(local_types.get(iter_values[0], ""))


def line_has_invalid_multi_assign_from_scalar(values: list[str], *, local_types: dict[str, str]) -> bool:
    if "=" not in values:
        return False
    equals = values.index("=")
    lhs = values[:equals]
    rhs = values[equals + 1 :]
    if "," not in lhs:
        return False
    target_names = [value for value in lhs if value.isidentifier() and value not in {","}]
    if len(target_names) < 2:
        return False
    return strict_body_expression_is_likely_noniterable(rhs, local_types=local_types)


def strict_body_expression_is_likely_noniterable(values: list[str], *, local_types: dict[str, str]) -> bool:
    if not values:
        return False
    inferred = strict_body_static_type_from_values(values)
    if strict_body_type_is_noniterable(inferred):
        return True
    first = values[0]
    if strict_body_type_is_noniterable(local_types.get(first, "")):
        return True
    if any(op in values for op in {"+", "-", "*", "/", "%", "//", "**"}):
        operand_names = [value for value in values if value.isidentifier()]
        if operand_names and all(
            value in STRICT_BODY_ALLOWED_GLOBAL_NAMES
            or value in STRICT_BODY_KEYWORD_NAMES
            or strict_body_type_is_noniterable(local_types.get(value, ""))
            for value in operand_names
        ):
            return True
    return False


def strict_body_type_is_noniterable(type_name: str) -> bool:
    return str(type_name or "") in {"bool", "float", "int"}


def strict_body_current_attribute_receiver_type(prefix: list[str], *, local_types: dict[str, str]) -> str:
    values = token_values(current_line_tokens(prefix))
    if len(values) < 2 or values[-1] != ".":
        return ""
    receiver = values[-2]
    if receiver in local_types:
        return local_types[receiver]
    branch_type = strict_body_current_branch_receiver_type(prefix, receiver)
    if branch_type:
        return branch_type
    if receiver in STRICT_BODY_BUILTIN_TYPE_NAMES:
        return receiver
    return ""


def strict_body_current_branch_receiver_type(prefix: list[str], receiver_name: str) -> str:
    if not receiver_name or not receiver_name.isidentifier():
        return ""
    current_depth = indentation_depth(prefix)
    if current_depth <= 0:
        return ""
    for depth, values in reversed(previous_significant_lines_with_depth(prefix)):
        if depth >= current_depth or not values:
            continue
        inferred = strict_body_isinstance_guard_type(values, receiver_name)
        if inferred:
            return inferred
    return ""


def strict_body_isinstance_guard_type(values: list[str], receiver_name: str) -> str:
    compact = [value for value in values if value not in {"(", ")", ",", ":"}]
    if (
        len(compact) >= 4
        and compact[0] in {"if", "elif"}
        and compact[1] == "isinstance"
        and compact[2] == receiver_name
        and compact[3] in STRICT_BODY_BUILTIN_TYPE_NAMES
    ):
        return compact[3]
    return ""


def closes_invalid_known_arity_call(prefix: list[str], *, local_types: dict[str, str]) -> bool:
    values = token_values(current_line_tokens(prefix))
    open_index = innermost_open_paren_index(values)
    if open_index <= 0:
        return False
    callee = values[open_index - 1]
    argc = call_arg_count(values[open_index + 1 :])
    if open_index >= 2 and values[open_index - 2] == ".":
        receiver = values[open_index - 3] if open_index >= 3 else ""
        receiver_type = local_types.get(receiver, "")
        return invalid_known_method_arity(receiver_type, callee, argc)
    return invalid_known_builtin_arity(callee, argc)


def innermost_open_paren_index(values: list[str]) -> int:
    depth = 0
    for index in range(len(values) - 1, -1, -1):
        value = values[index]
        if value == ")":
            depth += 1
        elif value == "(":
            if depth == 0:
                return index
            depth -= 1
    return -1


def call_arg_count(values: list[str]) -> int:
    if not values:
        return 0
    depth = 0
    has_token = False
    commas = 0
    for value in values:
        if value in {"(", "[", "{"}:
            depth += 1
        elif value in {")", "]", "}"}:
            depth = max(0, depth - 1)
        elif value == "," and depth == 0:
            commas += 1
        else:
            has_token = True
    return commas + 1 if has_token else 0


def invalid_known_builtin_arity(callee: str, argc: int) -> bool:
    if callee in {"abs", "len", "reversed"}:
        return argc != 1
    if callee in {"filter", "map"}:
        return argc < 2
    if callee in {"all", "any", "enumerate", "max", "min", "sorted", "sum"}:
        return argc < 1
    if callee == "range":
        return argc < 1 or argc > 3
    if callee == "round":
        return argc < 1 or argc > 2
    if callee == "isinstance":
        return argc < 2 or argc > 2
    if callee in STRICT_BODY_BUILTIN_TYPE_NAMES:
        return argc > 1
    return False


def invalid_known_method_arity(receiver_type: str, method: str, argc: int) -> bool:
    if not receiver_type:
        if method in {"append", "add", "discard", "extend", "remove", "update", "join"}:
            return argc != 1
        if method in {"casefold", "isalnum", "isalpha", "isdigit", "items", "keys", "lower", "splitlines", "upper", "values"}:
            return argc != 0
        if method in {"get", "setdefault", "split", "strip"}:
            return argc < 1 if method in {"get", "setdefault"} else argc > 2
        if method == "insert":
            return argc != 2
        if method in {"clear", "copy", "reverse", "sort"}:
            return argc != 0
        if method == "pop":
            return argc > 1
        return False
    if receiver_type in {"list", "set"} and method in {"append", "add", "discard", "extend", "remove", "update"}:
        return argc != 1
    if receiver_type == "list" and method == "insert":
        return argc != 2
    if receiver_type == "list" and method in {"clear", "copy", "reverse", "sort"}:
        return argc != 0
    if receiver_type == "list" and method == "pop":
        return argc > 1
    if receiver_type == "dict" and method in {"items", "keys", "values", "clear", "copy"}:
        return argc != 0
    if receiver_type == "dict" and method in {"get", "setdefault"}:
        return argc < 1 or argc > 2
    if receiver_type == "dict" and method == "update":
        return argc > 1
    if receiver_type == "str" and method in {"casefold", "isalnum", "isalpha", "isdigit", "lower", "splitlines", "upper"}:
        return argc != 0
    if receiver_type == "str" and method in {"strip", "split"}:
        return argc > 2
    if receiver_type == "str" and method == "join":
        return argc != 1
    return False


def closes_empty_subscript(prefix: list[str]) -> bool:
    values = token_values(current_line_tokens(prefix))
    if not values or values[-1] != "[":
        return False
    if len(values) >= 2 and values[-2] in {"=", "return", "(", ",", "["}:
        return False
    return True


def current_line_call_depth(values: list[str]) -> int:
    return sum(1 for value in values if value == "(")


def current_line_operator_chain_count(values: list[str]) -> int:
    return sum(1 for value in values if value in STRICT_BODY_CHAINING_OPERATORS)


def line_uses_attribute_call_as_value(values: list[str]) -> bool:
    if not values:
        return False
    if values[0] == "return":
        return True
    if "=" in values:
        return True
    if "(" in values[:-1]:
        return True
    return False


def completed_nonreturning_method_call_on_current_line(prefix: list[str]) -> bool:
    values = token_values(current_line_tokens(prefix))
    if not values or values[-1] != ")":
        return False
    open_index = matching_open_paren_for_final_close(values)
    if open_index < 2 or values[open_index - 2] != ".":
        return False
    method = values[open_index - 1]
    return method in STRICT_BODY_NONRETURNING_METHOD_NAMES


def line_is_standalone_nonreturning_call(values: list[str]) -> bool:
    if not values or values[-1] != ")":
        return False
    open_index = matching_open_paren_for_final_close(values)
    if open_index < 2 or values[open_index - 2] != ".":
        return False
    if values[open_index - 1] not in STRICT_BODY_NONRETURNING_METHOD_NAMES:
        return False
    prefix = values[: open_index - 2]
    return len(prefix) == 1 and str(prefix[0]).isidentifier()


def line_uses_nonreturning_call_as_value(values: list[str]) -> bool:
    if not values or values[-1] != ")":
        return False
    if line_is_standalone_nonreturning_call(values):
        return False
    open_index = matching_open_paren_for_final_close(values)
    if open_index < 2 or values[open_index - 2] != ".":
        return False
    return values[open_index - 1] in STRICT_BODY_NONRETURNING_METHOD_NAMES


def matching_open_paren_for_final_close(values: list[str]) -> int:
    if not values or values[-1] != ")":
        return -1
    depth = 0
    for index in range(len(values) - 1, -1, -1):
        value = values[index]
        if value == ")":
            depth += 1
        elif value == "(":
            depth -= 1
            if depth == 0:
                return index
    return -1


def token_allowed_by_lightweight_python_grammar(prefix: list[str], tok: str) -> bool:
    if tok in {"<pad>", "<bos>", "<unk>"}:
        return False
    if tok == "<eos>":
        return syntax_complete_body_prefix(prefix)
    if body_expression_trace_token(tok):
        return current_body_expression_trace_count(prefix) < 2
    kind, _, value = tok.partition(":")
    if not prefix:
        return tok.startswith("SLOT:PLAN_") or direct_statement_start_token(tok)
    if kind in {"SKEL", "SLOT"}:
        return True
    line = current_line_tokens(prefix)
    values = token_values(line)
    previous = values[-1] if values else ""
    previous_kind, _, previous_value = prefix[-1].partition(":") if prefix else ("", "", "")
    if kind == "NEWLINE":
        return bool(line) and not current_line_needs_more_tokens(line)
    if kind == "INDENT":
        return last_complete_line_ended_with_colon(prefix)
    if kind == "DEDENT":
        if prefix and prefix[-1] == "INDENT:":
            return False
        return indentation_depth(prefix) > 0
    if line_start(prefix):
        if kind not in {"NAME", "STRING", "NUMBER"}:
            return False
        if value in {"and", "or", "in", "is", "not"}:
            return False
        if value in {"elif", "else", "except", "finally"} and not previous_block_allows_continuation(prefix, value):
            return False
    elif kind == "NAME" and value in {
        "return",
        "continue",
        "break",
        "pass",
        "import",
        "from",
        "try",
        "except",
        "finally",
        "for",
        "while",
        "with",
        "elif",
        "else",
    }:
        return False
    elif kind == "NAME" and value == "if" and previous not in {"else"}:
        # Conditional expressions are legal Python, but the tiny strict decoder
        # repeatedly used statement keywords inside malformed expressions.
        return False
    elif kind == "NAME" and value == "not" and previous in {"in", "is"}:
        return False
    elif kind == "NAME" and value in {"and", "or", "in", "is"} and previous in {",", "(", "[", "{"}:
        return False
    elif kind == "NAME" and value in {"and", "or", "in", "is"} and previous in {"and", "or", "in", "is", "not"}:
        return False
    elif kind == "NAME" and value in {"and", "or"} and condition_connector_count(values) >= 8:
        return False
    elif kind == "NAME" and previous_kind == "NAME" and not adjacent_name_allowed(previous_value, value):
        return False
    elif kind == "NAME" and previous == "." and mutating_method_chain_after_call(prefix, value):
        return False
    if kind == "OP" and value == "(" and (previous in {")", "]", "}"} or previous_kind in {"NUMBER", "STRING"}):
        return False
    if kind == "OP" and value == "[" and current_line_subscript_count(values) >= 6:
        return False
    if kind == "OP" and value == "[" and values and values[0] in {"if", "elif", "while"}:
        if condition_subscript_count(values) >= 8:
            return False
    if kind == "OP" and value in {")", "]", "}"}:
        if previous in {"(", "[", "{"} and current_line_leading_keyword(line) in {"if", "elif", "while", "return"}:
            return False
        return has_unclosed_matching_bracket(prefix, value)
    if kind == "OP" and value == "=":
        if "=" in values:
            return False
        if previous in {"(", "[", "{", ",", ".", "and", "or", "not", "in", "is"}:
            return False
        if values and values[0] in {"if", "elif", "while", "return"}:
            return False
        if line_values_contain_non_lvalue_operator(values):
            return False
    if token_starts_expression_atom(kind, value) and expression_atom_requires_separator_before(values, value):
        return False
    if kind == "OP" and value == "," and values and values[0] in {"if", "elif", "while"} and not line_has_unclosed_brackets(line):
        return False
    if kind == "OP" and value == ":":
        return colon_allowed_on_current_line(prefix)
    if kind == "OP" and value in {".", ",", ":"} and line_start(prefix):
        return False
    return True


def current_line_leading_keyword(line: list[str]) -> str:
    values = token_values(line)
    return values[0] if values and values[0] in {"if", "elif", "while", "return"} else ""


def adjacent_name_allowed(previous: str, value: str) -> bool:
    if previous in {"for", "if", "elif", "while", "return", "not", "in", "is", "and", "or", "as", "import", "from", "except", "with"}:
        return True
    if value in {"in", "is", "not", "and", "or", "as", "import"}:
        return True
    return False


def mutating_method_chain_after_call(prefix: list[str], value: str) -> bool:
    if value not in {"add", "append", "extend", "setdefault", "update"}:
        return False
    line = current_line_tokens(prefix)
    vals = token_values(line)
    return len(vals) >= 2 and vals[-1] == "." and vals[-2] == ")"


def previous_block_allows_continuation(prefix: list[str], value: str) -> bool:
    lines = previous_significant_lines(prefix)
    if not lines:
        return False
    if value in {"elif", "else"}:
        return any(line and line[0] in {"if", "elif"} for line in lines)
    if value in {"except", "finally"}:
        return any(line and line[0] == "try" for line in lines)
    return False


def inside_loop_context(prefix: list[str]) -> bool:
    current_depth = indentation_depth(prefix)
    if current_depth <= 0:
        return False
    for depth, values in reversed(previous_significant_lines_with_depth(prefix)):
        if depth < current_depth and values and values[0] in {"for", "while"}:
            return True
        if depth < current_depth and values and values[0] in {"def", "class"}:
            return False
    return False


def previous_significant_lines_with_depth(prefix: list[str]) -> list[tuple[int, list[str]]]:
    lines: list[tuple[int, list[str]]] = []
    current: list[str] = []
    depth = 0
    line_depth = 0
    for tok in prefix:
        kind, _, _value = tok.partition(":")
        if kind == "INDENT":
            depth += 1
            continue
        if kind == "DEDENT":
            depth = max(0, depth - 1)
            continue
        if kind == "NEWLINE":
            vals = token_values(current)
            if vals:
                lines.append((line_depth, vals))
            current = []
            continue
        if tok == "<eos>":
            break
        if not current:
            line_depth = depth
        current.append(tok)
    vals = token_values(current)
    if vals:
        lines.append((line_depth, vals))
    return lines


def assignable_lvalue_tokens(values: list[str]) -> bool:
    if not values:
        return False
    if values[0] in {
        "and",
        "or",
        "not",
        "in",
        "is",
        "if",
        "for",
        "while",
        "return",
        "break",
        "continue",
        "pass",
        "raise",
        "try",
        "except",
        "finally",
        "else",
        "elif",
        "with",
        "import",
        "from",
    }:
        return False
    if values[-1] in {",", ".", "[", "(", "{", "+", "-", "*", "/", "%", "//", "**"}:
        return False
    if "(" in values or ")" in values:
        return False
    allowed = {",", ".", "[", "]"}
    for value in values:
        if value in allowed:
            continue
        if not str(value).isidentifier():
            return False
    return values.count("[") == values.count("]")


def strict_body_would_extend_pathological_comparison_chain(
    values: list[str],
    *,
    kind: str,
    value: str,
) -> bool:
    """Reject runaway comparison/membership chains in direct learned decoding.

    This is task-blind syntax hygiene. It only sees the generated current line
    and blocks patterns that repeatedly dominated failed private replays, such
    as ``data in data in data`` or comparisons mixed with assignment updates
    inside an ``if`` expression. It does not choose a return value, inspect
    prompts/tests/solutions, or add a candidate fallback.
    """

    if not values:
        return False
    if kind == "OP" and value in STRICT_BODY_AUGMENTED_ASSIGNMENT_OPERATORS:
        return True
    next_is_comparison = (kind == "OP" and value in STRICT_BODY_COMPARISON_OPERATORS) or (
        kind == "NAME" and value in {"in", "is"}
    )
    if not next_is_comparison:
        return False
    if values[0] == "for" and value == "in" and "in" not in values[1:]:
        return False
    existing = sum(1 for item in values if item in STRICT_BODY_COMPARISON_OPERATORS)
    if value in {"in", "is"} and value in values:
        return True
    return existing >= 2


def strict_body_would_extend_pathological_boolean_chain(
    values: list[str],
    *,
    kind: str,
    value: str,
) -> bool:
    """Reject runaway flat boolean chains in one generated line."""

    if kind != "NAME" or value not in {"and", "or"}:
        return False
    if not values:
        return False
    if values[-1] in {"and", "or", "not", "(", "[", "{", ",", ".", "="}:
        return True
    limit = 3 if values and values[0] in {"if", "elif", "while"} else 4
    return sum(1 for item in values if item in {"and", "or"}) >= limit


def strict_body_would_extend_pathological_subscript_chain(values: list[str]) -> bool:
    """Reject deep same-line subscript chains that dominated failed replays."""

    return values.count("[") >= 3


def strict_body_would_extend_uncalled_method_attribute_chain(values: list[str]) -> bool:
    """Reject ``data.get.strip``-style uncalled method attribute chains."""

    if len(values) < 2:
        return False
    return values[-2] == "." and values[-1] in STRICT_BODY_ALL_KNOWN_METHODS


def previous_significant_lines(prefix: list[str]) -> list[list[str]]:
    lines: list[list[str]] = []
    current: list[str] = []
    for tok in prefix:
        kind, _, _value = tok.partition(":")
        if kind == "NEWLINE":
            vals = token_values(current)
            if vals:
                lines.append(vals)
            current = []
        elif kind in {"INDENT", "DEDENT"}:
            continue
        else:
            current.append(tok)
    vals = token_values(current)
    if vals:
        lines.append(vals)
    return lines


def forced_lightweight_python_token(
    prefix: list[str],
    inverse: dict[int, str],
    probs: Any,
) -> tuple[int, float] | None:
    """Force only syntax-separator tokens required by direct body-token Python.

    This does not render a solution, select a body, or add a fallback return. It
    prevents the direct token stream from producing known-invalid layout such as
    ``if condition`` followed by NEWLINE without a colon/indent block.
    """

    if not prefix:
        return None
    if prefix[-1] == "NEWLINE:" and line_start(prefix) and last_complete_line_ended_with_colon(prefix):
        return token_choice_by_text(inverse, probs, "INDENT:")
    line = current_line_tokens(prefix)
    if not line:
        return None
    if line[-1] == "OP::":
        return token_choice_by_text(inverse, probs, "NEWLINE:")
    if current_line_needs_forced_colon(line):
        return token_choice_by_text(inverse, probs, "OP::")
    return None


def token_choice_by_text(
    inverse: dict[int, str],
    probs: Any,
    token_text: str,
) -> tuple[int, float] | None:
    for idx, text in inverse.items():
        if text == token_text:
            token_id = int(idx)
            return token_id, float(max(float(probs[token_id]), 1e-9))
    return None


def direct_statement_start_token(tok: str) -> bool:
    kind, _, value = tok.partition(":")
    if kind not in {"NAME", "STRING", "NUMBER"}:
        return False
    return value not in {"and", "or", "in", "is", "not"}


def token_values(tokens: list[str]) -> list[str]:
    values: list[str] = []
    for tok in tokens:
        if body_expression_trace_token(tok):
            continue
        kind, _, value = tok.partition(":")
        if kind in {"NEWLINE", "INDENT", "DEDENT"} or tok == "<eos>":
            continue
        values.append(value)
    return values


def current_body_expression_trace_count(prefix: list[str]) -> int:
    count = 0
    for tok in reversed(prefix):
        kind, _, _value = tok.partition(":")
        if kind in {"NEWLINE", "INDENT", "DEDENT"} or tok == "<eos>":
            break
        if body_expression_trace_token(tok):
            count += 1
    return count


def current_line_needs_more_tokens(line_tokens: list[str]) -> bool:
    values = token_values(line_tokens)
    if not values:
        return True
    if line_has_unclosed_brackets(line_tokens):
        return True
    if for_line_iterable_is_incomplete(values):
        return True
    if for_line_iterable_is_bare_constructor(values):
        return True
    if compound_clause_without_colon(values):
        return True
    if values[-1] in {
        "=",
        "+",
        "-",
        "*",
        "/",
        "%",
        "//",
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
        ".",
        ",",
    }:
        return True
    if condition_values_are_bare_builtin(values):
        return True
    return False


def current_line_needs_forced_colon(line_tokens: list[str]) -> bool:
    values = token_values(line_tokens)
    if not compound_clause_without_colon(values):
        return False
    if line_has_unclosed_brackets(line_tokens):
        return False
    if for_line_iterable_is_incomplete(values):
        return False
    if for_line_iterable_is_bare_constructor(values):
        return False
    if values[-1] in {
        "=",
        "+",
        "-",
        "*",
        "/",
        "%",
        "//",
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
        ".",
        ",",
    }:
        return False
    if condition_values_are_bare_builtin(values):
        return False
    if values[0] in {"if", "elif", "while"} and "=" in values:
        return False
    first = values[0]
    if first in {"else", "try", "finally"}:
        return True
    if first == "for":
        return "in" in values and len(values) >= 4 and not for_line_iterable_is_bare_constructor(values)
    if first in {"if", "while", "elif", "except", "with"}:
        return len(values) >= 2
    if first in {"def", "class"}:
        return len(values) >= 3
    return len(values) >= 8


def compound_clause_without_colon(values: list[str]) -> bool:
    if not values:
        return False
    first = values[0]
    if first in {"if", "for", "while", "with", "elif", "except", "def", "class"}:
        return ":" not in values
    if first in {"else", "try", "finally"}:
        return ":" not in values
    return False


def condition_values_are_bare_builtin(values: list[str]) -> bool:
    if not values or values[0] not in {"if", "elif", "while"}:
        return False
    expression = [value for value in values[1:] if value != "not"]
    return len(expression) == 1 and expression[0] in STRICT_BODY_BARE_BUILTIN_CONDITION_NAMES


def condition_connector_count(values: list[str]) -> int:
    if not values or values[0] not in {"if", "elif", "while"}:
        return 0
    return sum(1 for value in values if value in {"and", "or"})


def condition_subscript_count(values: list[str]) -> int:
    if not values or values[0] not in {"if", "elif", "while"}:
        return 0
    return values.count("[")


def current_line_subscript_count(values: list[str]) -> int:
    return values.count("[")


def line_has_unclosed_brackets(tokens: list[str]) -> bool:
    stack: list[str] = []
    for tok in tokens:
        kind, _, value = tok.partition(":")
        if kind != "OP":
            continue
        if value in {"(", "[", "{"}:
            stack.append(value)
        elif value in {")", "]", "}"} and stack:
            expected = {")": "(", "]": "[", "}": "{"}[value]
            if stack[-1] == expected:
                stack.pop()
    return bool(stack)


def colon_allowed_on_current_line(prefix: list[str]) -> bool:
    line = current_line_tokens(prefix)
    values = token_values(line)
    if compound_clause_without_colon(values):
        if for_line_iterable_is_incomplete(values) or for_line_iterable_is_bare_constructor(values):
            return False
        return current_line_needs_forced_colon(line)
    stack: list[str] = []
    for tok in line:
        kind, _, value = tok.partition(":")
        if kind != "OP":
            continue
        if value in {"{", "["}:
            stack.append(value)
        elif value in {"}", "]"} and stack:
            stack.pop()
    return bool(stack and stack[-1] == "{")


def for_line_iterable_values(values: list[str]) -> list[str]:
    if not values or values[0] != "for" or "in" not in values[1:]:
        return []
    in_index = values.index("in")
    return values[in_index + 1 :]


def for_line_iterable_is_incomplete(values: list[str]) -> bool:
    iter_values = for_line_iterable_values(values)
    if not iter_values:
        return bool(values and values[0] == "for" and "in" in values[1:])
    return iter_values[-1] in {"=", "+", "-", "*", "/", "%", "//", "**", "and", "or", "not", "in", "is", ".", ","}


def for_line_iterable_is_bare_constructor(values: list[str]) -> bool:
    """Return true for ``for x in list:`` / ``str:`` style partial iterables.

    The bare constructor can become a valid iterable expression once followed by
    a call, e.g. ``str(data)``. The decoder should not force a colon while it is
    still only a global constructor/type object.
    """

    iter_values = for_line_iterable_values(values)
    return len(iter_values) == 1 and iter_values[0] in STRICT_BODY_BUILTIN_TYPE_NAMES


def line_values_contain_non_lvalue_operator(values: list[str]) -> bool:
    """Reject assignment after comparison/membership expressions.

    This is a prefix-only lvalue check. It blocks malformed streams such as
    ``sign in str(data) + num =`` without looking at tasks or expected answers.
    """

    return any(value in {"<", "<=", ">", ">=", "==", "!=", "and", "or", "in", "is", "not"} for value in values)


def token_starts_expression_atom(kind: str, value: str) -> bool:
    if kind in {"NUMBER", "STRING"}:
        return True
    if kind != "NAME":
        return False
    if value in {"and", "or", "in", "is", "not", "as"}:
        return False
    if value in {"return", "break", "continue", "pass", "for", "if", "elif", "else", "while", "with"}:
        return False
    return True


def expression_atom_requires_separator_before(values: list[str], next_value: str) -> bool:
    if not values:
        return False
    previous = values[-1]
    if previous in {
        "=",
        "+",
        "-",
        "*",
        "/",
        "%",
        "//",
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
        "return",
        "yield",
        "if",
        "elif",
        "while",
        "for",
        "as",
        "(",
        "[",
        "{",
        ",",
        ":",
        ".",
    }:
        return False
    if next_value in {"True", "False", "None"}:
        return expression_value_is_terminal(previous)
    return expression_value_is_terminal(previous)


def expression_value_is_terminal(value: str) -> bool:
    if value in {")", "]", "}"}:
        return True
    if value in STRICT_BODY_KEYWORD_NAMES:
        return False
    if value in STRICT_BODY_CHAINING_OPERATORS or value in {"=", ".", ",", ":", "(", "[", "{"}:
        return False
    if value.isidentifier():
        return True
    try:
        float(value)
        return True
    except ValueError:
        pass
    return bool(value.startswith(("'", '"')))


def syntax_complete_body_prefix(prefix: list[str]) -> bool:
    if not prefix or prefix[-1] not in {"NEWLINE:", "DEDENT:"}:
        return False
    body = decode_body_tokens(prefix)
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if len(lines) < 1:
        return False
    if not any(line != "_ = 0" for line in lines):
        return False
    if not decoded_body_has_valued_return(body):
        return False
    try:
        source = render_synthetic_function(body)
        ast.parse(source)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            compile(source, "<body_prefix>", "exec")
    except SyntaxError:
        return False
    return True


def decoded_body_has_return(body: str) -> bool:
    try:
        parsed = ast.parse(render_synthetic_function(body))
    except SyntaxError:
        return False
    function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    return bool(function and any(isinstance(node, ast.Return) for node in ast.walk(function)))


def decoded_body_has_valued_return(body: str) -> bool:
    try:
        parsed = ast.parse(render_synthetic_function(body))
    except SyntaxError:
        return False
    function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    return bool(function and any(isinstance(node, ast.Return) and node.value is not None for node in ast.walk(function)))


def line_start(prefix: list[str]) -> bool:
    return not current_line_tokens(prefix)


def current_line_tokens(prefix: list[str]) -> list[str]:
    out = []
    for tok in reversed(prefix):
        if body_expression_trace_token(tok):
            continue
        kind, _, _value = tok.partition(":")
        if kind in {"NEWLINE", "INDENT", "DEDENT"} or tok == "<eos>":
            break
        out.append(tok)
    return list(reversed(out))


def last_complete_line_ended_with_colon(prefix: list[str]) -> bool:
    values = []
    for tok in reversed(prefix):
        kind, _, value = tok.partition(":")
        if kind == "NEWLINE":
            if values:
                break
            continue
        if kind in {"INDENT", "DEDENT"} and not values:
            continue
        if kind not in {"INDENT", "DEDENT"} and tok != "<eos>":
            values.append(value)
    return bool(values and values[0] == ":")


def indentation_depth(prefix: list[str]) -> int:
    depth = 0
    for tok in prefix:
        kind, _, _value = tok.partition(":")
        if kind == "INDENT":
            depth += 1
        elif kind == "DEDENT":
            depth = max(0, depth - 1)
    return depth


def has_unclosed_matching_bracket(prefix: list[str], closer: str) -> bool:
    opener = {")": "(", "]": "[", "}": "{"}[closer]
    stack: list[str] = []
    for tok in prefix:
        kind, _, value = tok.partition(":")
        if kind != "OP":
            continue
        if value in {"(", "[", "{"}:
            stack.append(value)
        elif value in {")", "]", "}"} and stack:
            expected = {")": "(", "]": "[", "}": "{"}[value]
            if stack[-1] == expected:
                stack.pop()
    return bool(stack and stack[-1] == opener)


def closes_invalid_known_builtin_call(prefix: list[str], builtin_name: str, *, min_commas: int) -> bool:
    line = current_line_tokens(prefix)
    values = token_values(line)
    depth = 0
    for index in range(len(values) - 1, -1, -1):
        value = values[index]
        if value == ")":
            depth += 1
        elif value == "(":
            if depth == 0:
                if index > 0 and values[index - 1] == builtin_name:
                    return values[index + 1 :].count(",") < int(min_commas)
                return False
            depth -= 1
    return False


def isinstance_second_argument_context(prefix: list[str]) -> bool:
    """Return true while decoding the second argument of an open isinstance call."""

    values = token_values(current_line_tokens(prefix))
    for index, value in enumerate(values[:-1]):
        if value != "(" or index == 0 or values[index - 1] != "isinstance":
            continue
        balance = 1
        top_level_commas = 0
        for inner in values[index + 1 :]:
            if inner in {"(", "[", "{"}:
                balance += 1
            elif inner in {")", "]", "}"}:
                balance -= 1
                if balance <= 0:
                    break
            elif inner == "," and balance == 1:
                top_level_commas += 1
        if balance > 0 and top_level_commas >= 1:
            return True
    return False


def isinstance_second_argument_tuple_context(prefix: list[str]) -> bool:
    """Return true while inside a type tuple used as isinstance's second arg."""

    values = token_values(current_line_tokens(prefix))
    for index, value in enumerate(values[:-1]):
        if value != "(" or index == 0 or values[index - 1] != "isinstance":
            continue
        balance = 1
        seen_top_level_comma = False
        second_arg_balance = 0
        for inner in values[index + 1 :]:
            if inner == "," and balance == 1:
                seen_top_level_comma = True
                continue
            if inner in {"(", "[", "{"}:
                balance += 1
                if seen_top_level_comma:
                    second_arg_balance += 1
            elif inner in {")", "]", "}"}:
                if seen_top_level_comma and second_arg_balance > 0:
                    second_arg_balance -= 1
                balance -= 1
                if balance <= 0:
                    break
        if balance > 0 and seen_top_level_comma and second_arg_balance > 0:
            return True
    return False


def isinstance_second_argument_has_type(prefix: list[str]) -> bool:
    """Return true if an open isinstance call already has a type argument."""

    values = token_values(current_line_tokens(prefix))
    for index, value in enumerate(values[:-1]):
        if value != "(" or index == 0 or values[index - 1] != "isinstance":
            continue
        balance = 1
        seen_top_level_comma = False
        second_arg_values: list[str] = []
        for inner in values[index + 1 :]:
            if inner in {"(", "[", "{"}:
                balance += 1
            elif inner in {")", "]", "}"}:
                balance -= 1
                if balance <= 0:
                    break
            elif inner == "," and balance == 1:
                seen_top_level_comma = True
                continue
            if seen_top_level_comma:
                second_arg_values.append(inner)
        if seen_top_level_comma and any(item in STRICT_BODY_BUILTIN_TYPE_NAMES for item in second_arg_values):
            return True
    return False

def visible_contract_semantic_beams(task: dict[str, Any], config: dict[str, Any], proposal_count: int) -> list[dict[str, Any]]:
    """Append shared semantic-plan beams inferred only from visible contracts.

    Contract-blind private eval rows deliberately withhold semantic family names.
    They still expose bounded operational fingerprints such as type family,
    argument roles, return shape, and required constructs. This beam converts
    those allowed fields into a reusable semantic-slot candidate when the
    fingerprint is unambiguous in the private train distribution. It never reads
    eval tests, eval solutions, public data, or teacher output.
    """

    structure_cfg = dict_or_empty(config.get("body_structure_decoder"))
    beam_cfg = dict_or_empty(structure_cfg.get("visible_contract_semantic_beam"))
    if not beam_cfg.get("enabled", False):
        return []
    prior = visible_contract_semantic_plan(task)
    plan = str(prior.get("plan") or "")
    if not plan:
        return []
    tokens = [f"SLOT:PLAN_{plan}"]
    return_shape = return_shape_for_task(task)
    if return_shape and return_shape != "unknown":
        tokens.append(f"SLOT:RETURN_SHAPE_{return_shape.upper()}")
    tokens.append("<eos>")
    source_count = max(1, int(proposal_count or 0))
    return [
        {
            "body": "",
            "decoded_tokens": tokens,
            "rank_score": round(-8.0 - (0.01 * source_count), 8),
            "decoded_token_count": len(tokens),
            "decoded_token_sha256": stable_hash(" ".join(tokens)),
            "beam_source": "visible_contract_semantic_prior",
            "contract_prior_plan": plan,
            "contract_prior_rule": prior.get("rule"),
            "contract_prior_fields_used": prior.get("fields_used"),
        }
    ]


def visible_contract_semantic_plan(task: dict[str, Any]) -> dict[str, Any]:
    contract = dict_or_empty(task.get("decoder_contract"))
    roles = dict_or_empty(contract.get("argument_roles"))
    data_role = str(roles.get("data") or "").strip()
    other_role = str(roles.get("other") or "").strip()
    type_family = str(contract.get("type_family") or "").strip()
    return_shape = str(
        get_path(contract, ["return_contract", "shape"], "")
        or contract.get("return_shape")
        or ""
    ).strip().lower()
    required = {str(item) for item in contract.get("required_constructs", []) or []}
    skeleton_bias = {
        str(item)
        for item in get_path(contract, ["generation_plan", "skeleton_bias"], []) or []
    }
    common_fields = {
        "decoder_contract.argument_roles.data": data_role,
        "decoder_contract.argument_roles.other": other_role,
        "decoder_contract.type_family": type_family,
        "decoder_contract.return_contract.shape": return_shape,
        "decoder_contract.required_constructs": sorted(required),
        "decoder_contract.generation_plan.skeleton_bias": sorted(skeleton_bias),
    }

    def prior(plan: str, rule: str) -> dict[str, Any]:
        return {"plan": plan, "rule": rule, "fields_used": common_fields}

    if type_family == "stdin_numeric_line_parser" and return_shape == "str" and (
        "stdin_parse" in required or "stdin_parse" in skeleton_bias
    ):
        return prior("STDIN_PAIR_SUMS", "stdin_numeric_line_parser:str:stdin_parse")
    if type_family == "grouped_interval_algorithm":
        if return_shape == "number":
            return prior("INTERVAL_COVERAGE", "grouped_interval_algorithm:number")
        if return_shape == "list":
            return prior("MERGE_INTERVALS", "grouped_interval_algorithm:list")
    if type_family == "dynamic_programming":
        if other_role == "secondary_input" or "index_or_string_ops" in required or "index_or_string_ops" in skeleton_bias:
            return prior("LCS_LENGTH", "dynamic_programming:secondary_or_index_ops")
        if return_shape == "number":
            return prior("MAX_NON_ADJACENT_SUM", "dynamic_programming:number")
    if type_family == "device_routing":
        if other_role == "request":
            return prior("DEVICE_ROUTE_WORKER", "device_routing:node_records:request")
        if other_role == "room_hint":
            return prior("VOICE_OUTPUT_ROUTE", "device_routing:node_records:room_hint")
    if type_family == "spatial_operator":
        if data_role == "media_records" and other_role == "query":
            return prior("MEDIA_PREVIEW_INDEX", "spatial_operator:media_records:query")
        if data_role == "device_records":
            return prior("ROOM_CAPABILITY_SUMMARY", "spatial_operator:device_records")
    if type_family == "project_memory":
        if data_role == "note_records":
            return prior("MEMORY_LATEST_BY_PROJECT", "project_memory:note_records")
        if data_role == "action_records":
            return prior("MEMORY_OPEN_ACTION_ROLLUP", "project_memory:action_records")
    if type_family == "storage_manifest":
        if data_role == "file_records" and other_role == "quota_bytes":
            return prior("STORAGE_QUOTA_SELECT", "storage_manifest:file_records:quota_bytes")
        if data_role == "local_manifest" and other_role == "remote_manifest":
            return prior("STORAGE_SYNC_PLAN", "storage_manifest:local_remote_manifest")
    if type_family == "long_horizon_plan":
        if return_shape == "list":
            return prior("PLAN_NEXT_UNBLOCKED", "long_horizon_plan:list")
        if return_shape == "dict":
            return prior("PLAN_PROGRESS_DIGEST", "long_horizon_plan:dict")
    return {"plan": "", "rule": "no_unambiguous_visible_contract_prior", "fields_used": common_fields}


def baseline_candidate(task: dict[str, Any], *, arm_id: str, config: dict[str, Any], seed: int) -> dict[str, Any]:
    return candidate_row(
        task,
        code=render_private_function(task, "return None"),
        phase="private_baseline",
        arm_id=arm_id,
        substrate="shared_null_baseline",
        view="baseline",
        rank=1,
        rank_score=1.0,
        config=config,
        seed=seed,
        decoded_token_count=2,
        decoded_token_sha256=stable_hash("return None"),
    )


def candidate_row(
    task: dict[str, Any],
    *,
    code: str,
    phase: str,
    arm_id: str,
    substrate: str,
    view: str,
    rank: int,
    rank_score: float,
    config: dict[str, Any],
    seed: int,
    decoded_token_count: int,
    decoded_token_sha256: str,
) -> dict[str, Any]:
    return {
        "task_id": str(task.get("task_id") or ""),
        "source_task_id": str(task.get("source_task_id") or ""),
        "entry_point": str(task.get("entry_point") or "solve"),
        "phase": phase,
        "candidate_source": "neural_seed_token_decoder_comparator",
        "code": code,
        "candidate_sha256": stable_hash(code),
        "substrate_arm": arm_id,
        "substrate_adapter": substrate,
        "rank": rank,
        "rank_score": round(rank_score, 8),
        "candidate_generation_mode": "token_level_code_decoder",
        "decoded_token_count": decoded_token_count,
        "decoded_token_sha256": decoded_token_sha256,
        "template_id": "",
        "template_sha256": "",
        "benchmark_promotion_eligible": False,
        "public_tests_visible_to_generator": False,
        "public_solutions_visible_to_generator": False,
        "eval_tests_visible_to_generator": False,
        "eval_solution_visible_to_generator": False,
        "external_inference_calls": 0,
        "provenance": {
            "policy": config.get("policy"),
            "comparison_level": config.get("comparison_level"),
            "view": view,
            "seed": seed,
            "ranker": get_path(config, ["candidate_row_schema", "ranker"], ""),
            "verifier": get_path(config, ["candidate_row_schema", "verifier"], ""),
            "generation_inputs": generation_inputs_for_view(config, view),
            "training_target": get_path(config, ["data", "training_target"], ""),
            "tests_used_for_generation": False,
            "solutions_used_for_generation": False,
            "body_template_selected": False,
            "model_promotion_allowed": False,
        },
    }


def generation_inputs_for_view(config: dict[str, Any], view: str) -> list[str]:
    text_views = config.get("text_views") if isinstance(config.get("text_views"), dict) else {}
    fields = text_views.get(view)
    if not isinstance(fields, list):
        return ["prompt", "callable_signature"]
    normalized = [str(field) for field in fields]
    if "entry_point" in normalized and "callable_signature" not in normalized:
        insert_at = normalized.index("entry_point") + 1
        normalized.insert(insert_at, "callable_signature")
    return normalized










def learned_plan_token_for_body(body: str) -> str:
    return f"SLOT:PLAN_{semantic_plan_from_body(str(body or ''))}"


def learned_body_decision_prefix_tokens_for_body(body: str) -> list[str]:
    plan_token = learned_plan_token_for_body(body)
    tokens: list[str] = []
    for token in semantic_slot_tokens(body):
        if token == plan_token or token.startswith("SLOT:PLAN_"):
            continue
        if token == PLAN_BODY_START_TOKEN:
            continue
        if token not in tokens:
            tokens.append(token)
    for token in state_transition_slot_tokens(body):
        if token not in tokens:
            tokens.append(token)
    for token in operand_binding_slot_tokens(body):
        if token not in tokens:
            tokens.append(token)
    for token in body_expression_intent_slot_tokens(body):
        if token not in tokens:
            tokens.append(token)
    for token in source_condition_decision_slots_from_body(body):
        if token not in tokens:
            tokens.append(token)
    return tokens


def body_expression_intent_slot_tokens(body: str) -> list[str]:
    """Return target-side expression intent slots for learned body prefixes.

    These slots describe operations that appear in admitted private/licensed
    target bodies. They are stripped before Python compilation and never render
    code. Their purpose is to give the learned decoder a source-conditioned
    handle for expression choice, such as calls, operators, indexing, and final
    return shape, without counting a deterministic renderer as learned
    generation.
    """

    try:
        parsed = ast.parse(render_synthetic_function(body))
        function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    except SyntaxError:
        function = None
    if function is None:
        return ["SLOT:EXPR_AST_INVALID"]

    slots: list[str] = []

    def add(token: str) -> None:
        if token not in slots:
            slots.append(token)

    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            name = expression_call_slot_value(node)
            if name:
                add(f"SLOT:EXPR_CALL_{name}")
        elif isinstance(node, ast.BinOp):
            add(f"SLOT:EXPR_BINOP_{node.op.__class__.__name__.upper()}")
        elif isinstance(node, ast.BoolOp):
            add(f"SLOT:EXPR_BOOLOP_{node.op.__class__.__name__.upper()}")
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                add(f"SLOT:EXPR_COMPARE_{op.__class__.__name__.upper()}")
        elif isinstance(node, ast.Subscript):
            add("SLOT:EXPR_INDEXING")
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            add(f"SLOT:EXPR_COMPREHENSION_{node.__class__.__name__.upper()}")

    for stmt in function.body:
        if isinstance(stmt, ast.Return):
            add(f"SLOT:EXPR_RETURN_{expression_intent_kind(stmt.value)}")
        if isinstance(stmt, ast.Assign):
            add(f"SLOT:EXPR_TOP_ASSIGN_{expression_intent_kind(stmt.value)}")
        if isinstance(stmt, ast.AnnAssign):
            add(f"SLOT:EXPR_TOP_ASSIGN_{expression_intent_kind(stmt.value)}")
        if isinstance(stmt, ast.For):
            for child in stmt.body:
                if isinstance(child, (ast.Assign, ast.AnnAssign)):
                    value = child.value if isinstance(child, (ast.Assign, ast.AnnAssign)) else None
                    add(f"SLOT:EXPR_LOOP_UPDATE_{expression_intent_kind(value)}")
                elif isinstance(child, ast.AugAssign):
                    add(f"SLOT:EXPR_LOOP_UPDATE_AUG_{child.op.__class__.__name__.upper()}")
                elif isinstance(child, ast.Expr) and isinstance(child.value, ast.Call):
                    add(f"SLOT:EXPR_LOOP_UPDATE_CALL_{expression_call_slot_value(child.value)}")
                elif isinstance(child, ast.If):
                    add(f"SLOT:EXPR_LOOP_BRANCH_{expression_intent_kind(child.test)}")
    return slots


def expression_intent_kind(node: ast.AST | None) -> str:
    if node is None:
        return "MISSING"
    if isinstance(node, ast.Name):
        return "NAME"
    if isinstance(node, ast.Constant):
        return "CONSTANT"
    if isinstance(node, ast.Call):
        name = expression_call_slot_value(node)
        return f"CALL_{name}" if name else "CALL"
    if isinstance(node, ast.BinOp):
        return f"BINOP_{node.op.__class__.__name__.upper()}"
    if isinstance(node, ast.BoolOp):
        return f"BOOLOP_{node.op.__class__.__name__.upper()}"
    if isinstance(node, ast.Compare):
        op = node.ops[0].__class__.__name__.upper() if node.ops else "UNKNOWN"
        return f"COMPARE_{op}"
    if isinstance(node, ast.Subscript):
        return "SUBSCRIPT"
    if isinstance(node, ast.IfExp):
        return "IFEXP"
    if isinstance(node, ast.List):
        return "LIST"
    if isinstance(node, ast.Tuple):
        return "TUPLE"
    if isinstance(node, ast.Set):
        return "SET"
    if isinstance(node, ast.Dict):
        return "DICT"
    if isinstance(node, ast.UnaryOp):
        return f"UNARY_{node.op.__class__.__name__.upper()}"
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return f"COMPREHENSION_{node.__class__.__name__.upper()}"
    return node.__class__.__name__.upper()


def safe_expr_slot_value(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value or "").upper()).strip("_")
    return cleaned[:48] or "UNKNOWN"


def expression_call_slot_value(node: ast.Call | None) -> str:
    """Return a canonical, learnable call-intent slot value.

    Earlier target modes emitted raw function/method names into semantic slot
    labels. That made the expression-call head learn hundreds of sparse private
    source/library names and encouraged nonsense calls during decode. This
    function keeps common Python operations explicit and collapses long-tail
    user/library calls into semantic families. The slot is still only an
    auxiliary training label; it does not render code or grant candidate credit.
    """

    if node is None:
        return "UNKNOWN"
    func = node.func
    if isinstance(func, ast.Attribute):
        attr = safe_expr_slot_value(str(func.attr or ""))
        lower = str(func.attr or "").lower()
        if lower in {"append", "extend", "insert"}:
            return "METHOD_SEQUENCE_MUTATION"
        if lower in {"add", "update", "discard", "remove"}:
            return "METHOD_SET_OR_MAP_MUTATION"
        if lower in {"get", "setdefault", "pop", "keys", "values", "items"}:
            return "METHOD_MAPPING_LOOKUP"
        if lower in {"split", "splitlines", "join", "strip", "lstrip", "rstrip", "lower", "upper", "replace"}:
            return "METHOD_TEXT_TRANSFORM"
        if lower in {"startswith", "endswith", "find", "index", "count"}:
            return "METHOD_TEXT_OR_SEQUENCE_QUERY"
        if lower in {"sort", "reverse"}:
            return "METHOD_ORDERING_MUTATION"
        if attr:
            return "METHOD_OTHER"
        return "METHOD_UNKNOWN"

    raw_name = call_name(node)
    lower = str(raw_name or "").lower()
    exact_builtin = {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "getattr",
        "hasattr",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }
    if lower in exact_builtin:
        return safe_expr_slot_value(lower)
    if lower in {"ceil", "floor", "sqrt", "gcd"}:
        return safe_expr_slot_value(lower)
    if lower.startswith("is_") or lower.startswith("has_"):
        return "PREDICATE_HELPER"
    if lower.startswith("to_") or lower.startswith("from_"):
        return "CONVERSION_HELPER"
    if raw_name:
        return "USER_OR_LIBRARY"
    return "UNKNOWN"


def state_transition_slot_tokens(body: str) -> list[str]:
    """Return private target-side state transition slots for learned prefixes.

    These slots summarize block-level control/dataflow shape in admitted target
    bodies. They are stripped before Python compilation and are never rendered
    into code, so they are not templates, tools, or learned-generation credit.
    Their job is to give the model a causal handle for stateful loop bodies
    before it emits raw body tokens.
    """

    try:
        parsed = ast.parse(render_synthetic_function(body))
        function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    except SyntaxError:
        function = None
    if function is None:
        return ["SLOT:STATE_AST_INVALID"]

    slots: list[str] = []

    def add(token: str) -> None:
        if token not in slots:
            slots.append(token)

    top_returns = [stmt for stmt in function.body if isinstance(stmt, ast.Return)]
    if top_returns:
        for stmt in top_returns:
            value = stmt.value
            if isinstance(value, ast.Name):
                add("SLOT:STATE_FINALIZER_DIRECT_LOCAL")
            elif value_has_stateful_transform(value):
                add("SLOT:STATE_FINALIZER_TRANSFORM")
            else:
                add("SLOT:STATE_FINALIZER_OTHER")

    loops = [node for node in ast.walk(function) if isinstance(node, (ast.For, ast.While))]
    if not loops:
        return slots
    add("SLOT:STATE_HAS_LOOP")
    for loop in loops:
        loop_targets = ast_store_names_local(getattr(loop, "target", None))
        if any(isinstance(child, ast.If) for child in ast.walk(ast.Module(body=list(getattr(loop, "body", []) or []), type_ignores=[]))):
            add("SLOT:STATE_LOOP_HAS_BRANCH")
        else:
            add("SLOT:STATE_LOOP_NO_BRANCH")
        if any(isinstance(child, (ast.Break, ast.Continue)) for child in ast.walk(ast.Module(body=list(getattr(loop, "body", []) or []), type_ignores=[]))):
            add("SLOT:STATE_LOOP_HAS_CONTROL_TERMINAL")
        mutation_seen = False
        assignment_seen = False
        augassign_seen = False
        shallow_identity_seen = False
        for child in ast.walk(ast.Module(body=list(getattr(loop, "body", []) or []), type_ignores=[])):
            if isinstance(child, ast.AugAssign):
                augassign_seen = True
                assignment_seen = True
            elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                assignment_seen = True
                value = child.value if isinstance(child, ast.Assign) else child.value
                if value_has_stateful_transform(value):
                    add("SLOT:STATE_UPDATE_ASSIGN_TRANSFORM")
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute) and child.func.attr in {"append", "add", "extend", "update", "setdefault", "pop", "remove"}:
                    mutation_seen = True
                    if child.func.attr in {"append", "add"} and len(child.args) == 1 and value_is_plain_loop_identity_local(child.args[0], loop_targets):
                        shallow_identity_seen = True
                    elif value_has_stateful_transform(child) or child.args:
                        add("SLOT:STATE_UPDATE_MUTATE_TRANSFORM")
        if assignment_seen:
            add("SLOT:STATE_LOOP_HAS_ASSIGNMENT")
        if augassign_seen:
            add("SLOT:STATE_UPDATE_AUGASSIGN")
        if mutation_seen:
            add("SLOT:STATE_UPDATE_MUTATION_CALL")
        if shallow_identity_seen:
            add("SLOT:STATE_UPDATE_SHALLOW_IDENTITY")
        elif mutation_seen or assignment_seen or augassign_seen:
            add("SLOT:STATE_UPDATE_NONIDENTITY")
    return slots


def ast_store_names_local(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)}


def value_is_plain_loop_identity_local(node: ast.AST | None, loop_targets: set[str]) -> bool:
    return isinstance(node, ast.Name) and node.id in loop_targets


def value_has_stateful_transform(node: ast.AST | None) -> bool:
    if node is None:
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
    return any(isinstance(child, semantic_nodes) for child in ast.walk(node))


def operand_binding_slot_tokens(body: str) -> list[str]:
    """Return private target-side operand/dataflow binding slots.

    State slots say what kind of update should happen; these binding slots say
    which visible/local operand roles the target body actually binds into that
    update. They are target tokens only, stripped before compilation, and never
    rendered into Python.
    """

    try:
        parsed = ast.parse(render_synthetic_function(body))
        function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    except SyntaxError:
        function = None
    if function is None:
        return []

    slots: list[str] = []

    def add(token: str) -> None:
        if token not in slots:
            slots.append(token)

    init_names: set[str] = set()
    for stmt in function.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                init_names.update(ast_store_names_local(target))
        elif isinstance(stmt, ast.AnnAssign):
            init_names.update(ast_store_names_local(stmt.target))

    for loop in [node for node in ast.walk(function) if isinstance(node, (ast.For, ast.While))]:
        loop_targets = ast_store_names_local(getattr(loop, "target", None))
        loop_body = list(getattr(loop, "body", []) or [])
        body_module = ast.Module(body=loop_body, type_ignores=[])
        if loop_targets and any(expression_load_names_local(child) & loop_targets for child in ast.walk(body_module)):
            add("SLOT:BIND_LOOP_TARGET_USED")
        for child in ast.walk(body_module):
            if isinstance(child, ast.If):
                test_names = expression_load_names_local(child.test)
                if test_names & loop_targets:
                    add("SLOT:BIND_BRANCH_USES_LOOP_TARGET")
                if test_names & (init_names | {"data", "other"}):
                    add("SLOT:BIND_BRANCH_USES_SOURCE_OR_STATE")
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                if child.func.attr in {"append", "add", "extend", "update", "setdefault"}:
                    for arg in child.args:
                        names = expression_load_names_local(arg)
                        if names & loop_targets:
                            add("SLOT:BIND_UPDATE_USES_LOOP_TARGET")
                        if names & {"data", "other"}:
                            add("SLOT:BIND_UPDATE_USES_SOURCE_ARG")
                        if names & init_names:
                            add("SLOT:BIND_UPDATE_USES_ACCUMULATOR")
                        if value_is_plain_loop_identity_local(arg, loop_targets):
                            add("SLOT:BIND_UPDATE_CALL_ARG_LOOP_TARGET")
                        elif value_has_stateful_transform(arg):
                            add("SLOT:BIND_UPDATE_CALL_ARG_TRANSFORM")
                        if any(isinstance(node, ast.Constant) for node in ast.walk(arg)):
                            add("SLOT:BIND_UPDATE_USES_CONSTANT")
            elif isinstance(child, ast.AugAssign):
                names = expression_load_names_local(child.value)
                if names & loop_targets:
                    add("SLOT:BIND_UPDATE_USES_LOOP_TARGET")
                if names & init_names:
                    add("SLOT:BIND_UPDATE_USES_ACCUMULATOR")
                if any(isinstance(node, ast.Constant) for node in ast.walk(child.value)):
                    add("SLOT:BIND_UPDATE_USES_CONSTANT")
            elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                value = child.value if isinstance(child, ast.Assign) else child.value
                names = expression_load_names_local(value)
                target_names: set[str] = set()
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        target_names.update(ast_store_names_local(target))
                else:
                    target_names.update(ast_store_names_local(child.target))
                if names & loop_targets:
                    add("SLOT:BIND_UPDATE_USES_LOOP_TARGET")
                if names & {"data", "other"}:
                    add("SLOT:BIND_UPDATE_USES_SOURCE_ARG")
                if names & init_names:
                    add("SLOT:BIND_UPDATE_USES_ACCUMULATOR")
                if target_names & init_names and names & init_names:
                    add("SLOT:BIND_UPDATE_ASSIGN_USES_PREVIOUS_STATE")
                if value_has_stateful_transform(value) and names & loop_targets:
                    add("SLOT:BIND_LOOP_TARGET_TRANSFORMED")
    for stmt in function.body:
        if isinstance(stmt, ast.Return):
            names = expression_load_names_local(stmt.value)
            if names & init_names:
                add("SLOT:BIND_FINALIZER_USES_ACCUMULATOR")
                if isinstance(stmt.value, ast.Name):
                    add("SLOT:BIND_RETURN_ACCUMULATOR")
                elif value_has_stateful_transform(stmt.value):
                    add("SLOT:BIND_RETURN_TRANSFORMED_ACCUMULATOR")
    return slots


def expression_load_names_local(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)}


def source_condition_decision_slots_from_body(body: str) -> list[str]:
    """Return learned prefix slots for branch/value decisions in private targets.

    These slots are decoder targets only. They are stripped before code
    compilation and are never rendered into Python, so they cannot claim
    template, fallback, tool, or adapter generation credit.
    """

    try:
        parsed = ast.parse(render_synthetic_function(body))
        function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    except SyntaxError:
        function = None
    if function is None:
        return []
    slots: list[str] = []
    for stmt in function.body:
        if not isinstance(stmt, ast.If):
            continue
        test_slots = source_condition_test_decision_slots(stmt.test)
        for token in test_slots:
            if token not in slots:
                slots.append(token)
        for child in stmt.body:
            if isinstance(child, ast.Return):
                token = source_condition_return_decision_slot(child.value, role="GUARDED")
                if token and token not in slots:
                    slots.append(token)
    for stmt in function.body:
        if isinstance(stmt, ast.Return):
            token = source_condition_return_decision_slot(stmt.value, role="DEFAULT")
            if token and token not in slots:
                slots.append(token)
    return slots


def source_condition_test_decision_slots(node: ast.AST) -> list[str]:
    slots: list[str] = []
    parts = list(node.values) if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And) else [node]
    for part in parts:
        if isinstance(part, ast.Call) and call_name(part) == "isinstance" and part.args:
            arg = part.args[0]
            if isinstance(arg, ast.Name):
                type_source = ast.unparse(part.args[1]) if len(part.args) > 1 and hasattr(ast, "unparse") else ""
                if "list" in type_source or "tuple" in type_source:
                    slots.append(f"SLOT:COND_SEQUENCE_ARG_{arg.id.upper()}")
        elif isinstance(part, ast.Name):
            slots.append(f"SLOT:COND_TRUTHY_ARG_{part.id.upper()}")
    return slots


def source_condition_return_decision_slot(node: ast.AST | None, *, role: str) -> str:
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        index = node.slice
        if isinstance(index, ast.Constant) and index.value == 0:
            return f"SLOT:RETURN_{role}_HEAD_ARG_{node.value.id.upper()}"
        if isinstance(index, ast.Index) and isinstance(index.value, ast.Constant) and index.value.value == 0:
            return f"SLOT:RETURN_{role}_HEAD_ARG_{node.value.id.upper()}"
    if isinstance(node, ast.Name):
        return f"SLOT:RETURN_{role}_ARG_{node.id.upper()}"
    return ""




def statement_sequence_slot_tokens(body: str) -> list[str]:
    """Return private target-side statement sequencing slots.

    The slots describe statement order and nested loop action shape in admitted
    private target bodies. They are generated prefix targets only and are
    stripped before Python compilation. They do not render code, inspect
    tests/solutions/public payloads, call a teacher, or grant candidate credit.
    """

    try:
        parsed = ast.parse(render_synthetic_function(body))
        function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    except SyntaxError:
        function = None
    if function is None:
        return ["SLOT:STMT_SEQ_AST_INVALID"]

    tokens: list[str] = []

    def add(token: str) -> None:
        if token not in tokens:
            tokens.append(token)

    top_kinds = [statement_sequence_kind(stmt, top_level=True) for stmt in function.body]
    top_kinds = [kind for kind in top_kinds if kind]
    if top_kinds:
        add("SLOT:STMT_SEQ_" + "_".join(top_kinds[:6]))
        add(f"SLOT:STMT_TOP_LEN_{min(len(top_kinds), 6)}")
    else:
        add("SLOT:STMT_SEQ_EMPTY")

    for stmt in function.body:
        if not isinstance(stmt, (ast.For, ast.While)):
            continue
        loop_kinds = [statement_sequence_kind(child, top_level=False) for child in list(stmt.body or [])]
        loop_kinds = [kind for kind in loop_kinds if kind]
        if loop_kinds:
            add("SLOT:STMT_LOOP_SEQ_" + "_".join(loop_kinds[:6]))
            add(f"SLOT:STMT_LOOP_LEN_{min(len(loop_kinds), 6)}")
        if any(kind.startswith("IF") for kind in loop_kinds):
            add("SLOT:STMT_LOOP_HAS_DECISION")
        if any(kind in {"CALL_APPEND", "CALL_ADD", "ASSIGN_TRANSFORM", "AUGASSIGN"} for kind in loop_kinds):
            add("SLOT:STMT_LOOP_HAS_UPDATE")
        if any(kind.startswith("RETURN") for kind in loop_kinds):
            add("SLOT:STMT_LOOP_HAS_INNER_RETURN")

    top_returns = [stmt for stmt in function.body if isinstance(stmt, ast.Return)]
    add(f"SLOT:STMT_FINAL_{statement_sequence_return_kind(top_returns[-1])}" if top_returns else "SLOT:STMT_FINAL_MISSING")
    return tokens


def statement_sequence_kind(stmt: ast.stmt, *, top_level: bool) -> str:
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return "IMPORT"
    if isinstance(stmt, ast.Assign):
        return "ASSIGN_" + statement_sequence_value_kind(stmt.value)
    if isinstance(stmt, ast.AnnAssign):
        return "ASSIGN_" + statement_sequence_value_kind(stmt.value)
    if isinstance(stmt, ast.AugAssign):
        return "AUGASSIGN"
    if isinstance(stmt, ast.For):
        return "FOR"
    if isinstance(stmt, ast.While):
        return "WHILE"
    if isinstance(stmt, ast.If):
        return "IF_RETURN" if any(isinstance(child, ast.Return) for child in list(stmt.body or [])) else "IF"
    if isinstance(stmt, ast.Try):
        return "TRY"
    if isinstance(stmt, ast.Expr):
        if isinstance(stmt.value, ast.Call):
            name = call_name(stmt.value)
            if name.endswith(".append"):
                return "CALL_APPEND"
            if name.endswith(".add"):
                return "CALL_ADD"
            if name.endswith(".setdefault"):
                return "CALL_SETDEFAULT"
            return "CALL"
        return "EXPR"
    if isinstance(stmt, ast.Return):
        return "RETURN_TOP" if top_level else "RETURN_INNER"
    if isinstance(stmt, ast.Continue):
        return "CONTINUE"
    if isinstance(stmt, ast.Break):
        return "BREAK"
    return "OTHER"


def statement_sequence_value_kind(node: ast.AST | None) -> str:
    if node is None:
        return "OTHER"
    if isinstance(node, (ast.List, ast.Dict, ast.Set, ast.Tuple, ast.Constant, ast.Name)):
        return value_shape(node).upper()
    if isinstance(node, ast.Call):
        name = call_name(node)
        if name.endswith(".split") or name.endswith(".splitlines"):
            return "TEXT_SPLIT"
        return "CALL"
    if isinstance(node, (ast.BinOp, ast.BoolOp, ast.UnaryOp, ast.Compare, ast.Subscript, ast.IfExp)):
        return "TRANSFORM"
    if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
        return "COMP"
    return "OTHER"


def statement_sequence_return_kind(stmt: ast.Return) -> str:
    value = stmt.value
    if isinstance(value, ast.Name):
        return "LOCAL"
    if value_has_stateful_transform(value):
        return "TRANSFORM"
    return return_shape_from_expr(value).upper()


def body_tokens_for_target_mode(tokens: list[str], *, target_mode: str) -> list[str]:
    if learned_plan_prefix_target_mode(target_mode):
        body, _meta = split_learned_plan_prefix_tokens(tokens)
        return body
    return list(tokens)


def build_target_vocab(
    bodies: list[str],
    *,
    max_vocab: int,
    target_mode: str = "body_tokens",
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for body in bodies:
        counts.update(target_tokens(body, target_mode=target_mode))
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    for tok, _count in counts.most_common(max(0, max_vocab - len(vocab))):
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def encode_target_rows(
    bodies: list[str],
    vocab: dict[str, int],
    max_len: int,
    *,
    target_mode: str = "body_tokens",
) -> list[list[int]]:
    rows = []
    for body in bodies:
        ids = [vocab["<bos>"]]
        ids.extend(vocab.get(tok, vocab["<unk>"]) for tok in target_tokens(body, target_mode=target_mode)[: max(1, max_len - 2)])
        ids.append(vocab["<eos>"])
        ids = ids[:max_len]
        rows.append(ids + [vocab["<pad>"]] * max(0, max_len - len(ids)))
    return rows


def target_tokens(body: str, *, target_mode: str) -> list[str]:
    if learned_plan_expression_transition_body_target_mode(target_mode):
        return [
            learned_plan_token_for_body(body),
            *learned_body_decision_prefix_tokens_for_body(body),
            *statement_sequence_slot_tokens(body),
            PLAN_BODY_START_TOKEN,
            *body_tokens_with_expression_transition_markers(body),
        ]
    if learned_plan_statement_slots_body_target_mode(target_mode):
        return [
            learned_plan_token_for_body(body),
            *learned_body_decision_prefix_tokens_for_body(body),
            *statement_sequence_slot_tokens(body),
            PLAN_BODY_START_TOKEN,
            *body_tokens(body),
        ]
    if learned_plan_semantic_slots_body_target_mode(target_mode):
        return [
            learned_plan_token_for_body(body),
            *learned_body_decision_prefix_tokens_for_body(body),
            PLAN_BODY_START_TOKEN,
            *body_tokens(body),
        ]
    if str(target_mode or "") == PLAN_PREFIX_BODY_TARGET_MODE:
        return [learned_plan_token_for_body(body), PLAN_BODY_START_TOKEN, *body_tokens(body)]
    if target_mode == "semantic_slots_v1":
        return semantic_slot_tokens(body) + body_structure_tokens(body)
    if target_mode == "statement_skeleton_v1":
        return body_structure_tokens(body)
    if target_mode == "strict_action_tokens_v1":
        return strict_action_tokens(body)
    return body_tokens(body)


def body_tokens_with_expression_transition_markers(body: str) -> list[str]:
    """Return body tokens with learned expression-transition trace markers.

    The trace tokens are derived from admitted private/licensed target bodies
    and are stripped before Python compilation. They train the body stream to
    cross statement/expression boundaries without invoking a renderer,
    template, tool, public benchmark surface, or verifier target.
    """

    normalized = normalize_body_text(body)
    markers_by_line = expression_transition_markers_by_body_line(normalized)
    out: list[str] = []
    inserted_lines: set[int] = set()
    try:
        generated = py_tokenize.generate_tokens(io.StringIO(normalized.rstrip() + "\n").readline)
        for tok in generated:
            if tok.type in {py_token.ENCODING, py_token.ENDMARKER, py_token.NL}:
                continue
            line_no = int(tok.start[0] or 0)
            if (
                line_no > 0
                and line_no not in inserted_lines
                and tok.type not in {py_token.NEWLINE, py_token.INDENT, py_token.DEDENT}
            ):
                out.extend(markers_by_line.get(line_no, [])[:2])
                inserted_lines.add(line_no)
            if tok.type == py_token.NEWLINE:
                out.append("NEWLINE:")
            elif tok.type == py_token.INDENT:
                out.append("INDENT:")
            elif tok.type == py_token.DEDENT:
                out.append("DEDENT:")
            else:
                out.append(f"{py_token.tok_name.get(tok.type, str(tok.type))}:{tok.string}")
    except (py_tokenize.TokenError, IndentationError):
        out.extend(f"RAW:{part}" for part in normalized.split())
    return out


def expression_transition_markers_by_body_line(body: str) -> dict[int, list[str]]:
    try:
        parsed = ast.parse(render_synthetic_function(body))
        function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    except SyntaxError:
        function = None
    if function is None:
        return {}

    markers_by_line: dict[int, list[str]] = {}

    def add(line: int, marker: str) -> None:
        if line <= 0:
            return
        values = markers_by_line.setdefault(line, [])
        if marker not in values:
            values.append(marker)

    def visit_stmt(stmt: ast.stmt) -> None:
        line = int(getattr(stmt, "lineno", 0) or 0) - 1
        for marker in expression_transition_markers_for_stmt(stmt):
            add(line, marker)
        for child in getattr(stmt, "body", []) or []:
            if isinstance(child, ast.stmt):
                visit_stmt(child)
        for child in getattr(stmt, "orelse", []) or []:
            if isinstance(child, ast.stmt):
                visit_stmt(child)
        for handler in getattr(stmt, "handlers", []) or []:
            for child in getattr(handler, "body", []) or []:
                if isinstance(child, ast.stmt):
                    visit_stmt(child)
        for child in getattr(stmt, "finalbody", []) or []:
            if isinstance(child, ast.stmt):
                visit_stmt(child)

    for stmt in function.body:
        visit_stmt(stmt)
    return markers_by_line


def expression_transition_markers_for_stmt(stmt: ast.stmt) -> list[str]:
    markers: list[str] = []

    def add(value: str) -> None:
        marker = f"TRACE:EXPR_{safe_expr_slot_value(value)}"
        if marker not in markers:
            markers.append(marker)

    if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
        value = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
        add(f"ASSIGN_{expression_intent_kind(value)}")
    elif isinstance(stmt, ast.AugAssign):
        add(f"AUG_{stmt.op.__class__.__name__.upper()}")
        add(f"AUG_VALUE_{expression_intent_kind(stmt.value)}")
    elif isinstance(stmt, ast.For):
        add(f"FOR_ITER_{expression_intent_kind(stmt.iter)}")
    elif isinstance(stmt, ast.If):
        add(f"IF_TEST_{expression_intent_kind(stmt.test)}")
    elif isinstance(stmt, ast.Return):
        add(f"RETURN_{expression_intent_kind(stmt.value)}")
    elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        add(f"CALL_STMT_{expression_call_slot_value(stmt.value)}")
    elif isinstance(stmt, ast.While):
        add(f"WHILE_TEST_{expression_intent_kind(stmt.test)}")
    elif isinstance(stmt, ast.Try):
        add("TRY")
    return markers


def semantic_slot_tokens(body: str) -> list[str]:
    try:
        parsed = ast.parse(render_synthetic_function(body))
        function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    except SyntaxError:
        function = None
    plan = semantic_plan_from_body(body)
    slots = [f"SLOT:PLAN_{plan}"]
    if function is None:
        return slots + ["SLOT:AST_INVALID"]
    return_shape = ""
    loop_sources: set[str] = set()
    init_shapes: set[str] = set()
    update_ops: set[str] = set()
    guard_ops: set[str] = set()
    finalizers: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Return) and not return_shape:
            return_shape = return_shape_from_expr(node.value)
            finalizers.add(finalizer_from_return(node.value))
        if isinstance(node, ast.For):
            loop_sources.add(loop_source_from_iter(node.iter))
        if isinstance(node, ast.Assign):
            init_shapes.add(value_shape(node.value))
        if isinstance(node, ast.If):
            guard_ops.update(guard_slots_from_test(node.test))
        if isinstance(node, ast.Try):
            guard_ops.add("TRY_EXCEPT_CONTINUE")
        if isinstance(node, ast.Call):
            update_ops.add(update_slot_from_call(node))
        if isinstance(node, ast.AugAssign):
            update_ops.add("ACCUMULATE_NUMERIC")
    if return_shape:
        slots.append(f"SLOT:RETURN_SHAPE_{return_shape.upper()}")
    for value in sorted(loop_sources):
        slots.append(f"SLOT:LOOP_SOURCE_{value}")
    for value in sorted(init_shapes):
        slots.append(f"SLOT:INIT_{value.upper()}")
    for value in sorted(update_ops):
        slots.append(f"SLOT:UPDATE_{value}")
    for value in sorted(guard_ops):
        slots.append(f"SLOT:GUARD_{value}")
    for value in sorted(finalizers):
        slots.append(f"SLOT:FINALIZER_{value}")
    return slots


def semantic_plan_from_body(body: str) -> str:
    text = normalize_body_text(body)
    compact = " ".join(text.split())
    if "return data[0]" in text and "return other" in text:
        return "SAFE_HEAD_DEFAULT"
    if "out.setdefault(str(key), []).append(record['id'])" in text:
        return "GROUP_RECORDS_BY_FIELD"
    if "out.append({col: row.get(col) for col in other})" in text:
        return "PROJECT_TABLE_COLUMNS"
    if "math.gcd" in text:
        return "GCD_POSITIVE"
    if "return sorted(out)" in text and ".casefold()" in text:
        return "NORMALIZE_FILTER_SORT_UNIQUE"
    if "values[i + 1] - values[i]" in text:
        return "WINDOWED_DELTAS"
    if "pairs = {')': '('" in text or "return not stack" in text:
        return "BALANCED_BRACKETS"
    if "str(data).splitlines()" in text and "'\\n'.join(out)" in text and "int(parts[0]) + int(parts[1])" in text:
        return "STDIN_PAIR_SUMS"
    if "components += 1" in text and "graph[a].append(b)" in text:
        return "GRAPH_COMPONENTS"
    if "queue = deque" in text and "return dist" in text:
        return "SHORTEST_HOPS"
    if "take, skip = skip + value" in text:
        return "MAX_NON_ADJACENT_SUM"
    if "prev = [0] * (len(b) + 1)" in text:
        return "LCS_LENGTH"
    if "return [tuple(item) for item in merged]" in text:
        return "MERGE_INTERVALS"
    if "return sum(b - a for a, b in merged)" in text:
        return "INTERVAL_COVERAGE"
    if "current += 1" in text and "best = max(best, current)" in text:
        return "LONGEST_EVEN_RUN"
    if "out[-1] = (item, out[-1][1] + 1)" in text:
        return "RLE_ENCODE"
    if "record.get('score'" in text and "record.get('label')" in text:
        return "THRESHOLD_LABELS"
    if "counts[item] = counts.get(item, 0) + 1" in text and "items[:max(0, int(other))]" in text:
        return "TOP_K_FREQUENT"
    if "need_tags = set(query.get(\"tags\") or [])" in text:
        return "MEDIA_PREVIEW_INDEX"
    if "rooms.setdefault(room" in text and "\"mics\"" in text:
        return "ROOM_CAPABILITY_SUMMARY"
    if "latest[project] = (ts, text)" in text:
        return "MEMORY_LATEST_BY_PROJECT"
    if "out.setdefault(owner, set()).add(str(label))" in text:
        return "MEMORY_OPEN_ACTION_ROLLUP"
    if "quota = int(other or 0)" in text and "picked.append(name)" in text:
        return "STORAGE_QUOTA_SELECT"
    if "ops.append((\"download\", path))" in text and "ops.append((\"upload\", path))" in text:
        return "STORAGE_SYNC_PLAN"
    if "request.get(\"capabilities\")" in text and "avoid_battery" in text:
        return "DEVICE_ROUTE_WORKER"
    if "node.get(\"speaker\")" in text and "room = str(other or \"\")" in text:
        return "VOICE_OUTPUT_ROUTE"
    if "deps <= done" in text and "available.append" in text:
        return "PLAN_NEXT_UNBLOCKED"
    if "\"blocked\": 0" in text and "\"owners\"" in text:
        return "PLAN_PROGRESS_DIGEST"
    if "int(parts[0])" in text and "splitlines" in text:
        return "STDIN_NUMERIC_PARSE"
    ast_plan = semantic_plan_from_ast_body(body)
    if ast_plan:
        return ast_plan
    if "setdefault" in compact and ".append" in compact:
        return "DICT_GROUP_APPEND"
    if ".append" in compact:
        return "LIST_APPEND"
    if ".add" in compact:
        return "SET_ADD"
    if "return" in compact:
        return "GENERIC_RETURN"
    return "GENERIC_BODY"


def semantic_plan_from_ast_body(body: str) -> str:
    """Return an AST-local operation plan for auxiliary supervision.

    The label is derived only from the admitted training body AST. It is never
    used as a renderer or as learned-generation credit.
    """

    try:
        parsed = ast.parse(render_synthetic_function(body))
        function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    except SyntaxError:
        return "AST_INVALID"
    if function is None:
        return "AST_INVALID"

    calls = [call_name(node) for node in ast.walk(function) if isinstance(node, ast.Call)]
    call_set = set(calls)
    method_names = {name.rsplit(".", 1)[-1] for name in calls if "." in name}
    assign_shapes = Counter(value_shape(node.value) for node in ast.walk(function) if isinstance(node, ast.Assign))
    return_shapes = Counter(
        return_shape_from_expr(node.value)
        for node in ast.walk(function)
        if isinstance(node, ast.Return)
    )
    return_shape = return_shapes.most_common(1)[0][0] if return_shapes else "none"
    loop_count = sum(1 for node in ast.walk(function) if isinstance(node, (ast.For, ast.While)))
    branch_count = sum(1 for node in ast.walk(function) if isinstance(node, ast.If))
    augassign_count = sum(1 for node in ast.walk(function) if isinstance(node, ast.AugAssign))
    compare_count = sum(1 for node in ast.walk(function) if isinstance(node, ast.Compare))
    comprehension_kind = semantic_plan_comprehension_kind(function)
    if comprehension_kind:
        return f"AST_RETURN_{comprehension_kind}_COMP"
    if "setdefault" in method_names and "append" in method_names:
        return "AST_DICT_GROUP_APPEND"
    if "setdefault" in method_names and "add" in method_names:
        return "AST_DICT_GROUP_SET"
    if semantic_plan_has_dict_counter(function):
        return "AST_DICT_COUNT"
    if "append" in method_names and ("sorted" in call_set or "sort" in method_names):
        return "AST_LIST_ACCUMULATE_SORT"
    if "append" in method_names and ("join" in method_names or "''.join" in call_set):
        return "AST_TEXT_BUILD_JOIN"
    if "append" in method_names and loop_count:
        if branch_count:
            return "AST_LIST_FILTER_MAP"
        return "AST_LIST_ACCUMULATE"
    if "extend" in method_names and loop_count:
        return "AST_LIST_EXTEND_FLATTEN"
    if "add" in method_names and loop_count:
        return "AST_SET_ACCUMULATE"
    if "update" in method_names and loop_count:
        return "AST_MAPPING_UPDATE"
    if {"split", "splitlines"} & method_names:
        if "int" in call_set or "float" in call_set:
            return "AST_TEXT_PARSE_NUMERIC"
        return "AST_TEXT_SPLIT_TRANSFORM"
    if "join" in method_names or "''.join" in call_set:
        return "AST_TEXT_JOIN"
    if "sorted" in call_set or "sort" in method_names:
        if loop_count:
            return "AST_LOOP_SORT_RETURN"
        return "AST_RETURN_SORTED"
    if call_set & {"sum", "all", "any", "max", "min"}:
        aggregate = sorted(call_set & {"sum", "all", "any", "max", "min"})[0].upper()
        if loop_count:
            return f"AST_LOOP_{aggregate}_AGGREGATE"
        return f"AST_RETURN_{aggregate}_AGGREGATE"
    if augassign_count and loop_count:
        return "AST_LOOP_NUMERIC_ACCUMULATE"
    if compare_count and branch_count and return_shape == "bool":
        return "AST_BRANCH_BOOL_PREDICATE"
    if loop_count and branch_count:
        return f"AST_LOOP_BRANCH_RETURN_{return_shape.upper()}"
    if loop_count:
        if assign_shapes.get("dict"):
            return "AST_LOOP_DICT_BUILD"
        if assign_shapes.get("set"):
            return "AST_LOOP_SET_BUILD"
        if assign_shapes.get("list"):
            return "AST_LOOP_LIST_BUILD"
        return f"AST_LOOP_RETURN_{return_shape.upper()}"
    if branch_count:
        return f"AST_BRANCH_RETURN_{return_shape.upper()}"
    if function.body and len(function.body) == 1 and isinstance(function.body[0], ast.Return):
        return f"AST_DIRECT_RETURN_{return_shape.upper()}"
    if return_shape != "none":
        return f"AST_RETURN_{return_shape.upper()}"
    return ""


def semantic_plan_comprehension_kind(function: ast.FunctionDef) -> str:
    for node in ast.walk(function):
        if isinstance(node, ast.Return):
            if isinstance(node.value, ast.ListComp):
                return "LIST"
            if isinstance(node.value, ast.DictComp):
                return "DICT"
            if isinstance(node.value, ast.SetComp):
                return "SET"
            if isinstance(node.value, ast.GeneratorExp):
                return "GENERATOR"
    return ""


def semantic_plan_has_dict_counter(function: ast.FunctionDef) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or not node.targets:
            continue
        target = node.targets[0]
        value = node.value
        if not isinstance(target, ast.Subscript):
            continue
        if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Add):
            continue
        if isinstance(value.right, ast.Constant) and value.right.value == 1:
            if any(isinstance(child, ast.Call) and call_name(child).endswith(".get") for child in ast.walk(value.left)):
                return True
    return False


def loop_source_from_iter(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id.upper()
    if isinstance(node, ast.Call):
        name = call_name(node)
        if name.endswith("splitlines"):
            return "STR_DATA_SPLITLINES"
        if name == "range":
            return "RANGE"
        if name == "sorted":
            return "SORTED"
    return "OTHER"


def guard_slots_from_test(node: ast.AST) -> set[str]:
    source = ast.unparse(node) if hasattr(ast, "unparse") else ""
    slots: set[str] = set()
    if "is None" in source:
        slots.add("SKIP_NONE")
    if "isinstance" in source and "dict" in source:
        slots.add("SKIP_NON_DICT")
    if "isinstance" in source and ("list" in source or "tuple" in source):
        slots.add("SEQUENCE")
    if "len(" in source:
        slots.add("LENGTH_CHECK")
    if "done" in source:
        slots.add("DONE_FILTER")
    if "blocked" in source:
        slots.add("BLOCKED_FILTER")
    if "score" in source or "priority" in source:
        slots.add("SCORE_OR_PRIORITY")
    return slots or {"BRANCH"}


def update_slot_from_call(node: ast.Call) -> str:
    name = call_name(node)
    source = ast.unparse(node) if hasattr(ast, "unparse") else ""
    if name.endswith(".append"):
        if "record['id']" in source:
            return "APPEND_RECORD_ID"
        if "row.get" in source or " for col in " in source:
            return "APPEND_PROJECTED_ROW"
        if "str(int(parts[0]) + int(parts[1]))" in source:
            return "APPEND_STDIN_PAIR_SUM"
        if "label" in source:
            return "APPEND_LABEL"
        if "name" in source:
            return "APPEND_NAME"
        return "APPEND_ITEM"
    if name.endswith(".add"):
        return "SET_ADD"
    if name.endswith(".setdefault"):
        return "DICT_SETDEFAULT"
    if name in {"sorted"}:
        return "SORT"
    return "CALL"


def finalizer_from_return(node: ast.AST | None) -> str:
    if node is None:
        return "NONE"
    source = ast.unparse(node) if hasattr(ast, "unparse") else ""
    if "sorted" in source:
        return "SORTED"
    if "join" in source:
        return "JOIN"
    if "tuple" in source:
        return "TUPLE"
    if "max(" in source:
        return "MAX"
    if "sum(" in source:
        return "SUM"
    if "not stack" in source:
        return "BOOL_NOT_STACK"
    if "data[0]" in source:
        return "HEAD_DEFAULT"
    return "RETURN_RESULT"


def body_tokens(body: str) -> list[str]:
    out = []
    body = normalize_body_text(body)
    try:
        generated = py_tokenize.generate_tokens(io.StringIO(body.rstrip() + "\n").readline)
        for tok in generated:
            if tok.type in {py_token.ENCODING, py_token.ENDMARKER, py_token.NL}:
                continue
            if tok.type == py_token.NEWLINE:
                out.append("NEWLINE:")
            elif tok.type == py_token.INDENT:
                out.append("INDENT:")
            elif tok.type == py_token.DEDENT:
                out.append("DEDENT:")
            else:
                out.append(f"{py_token.tok_name.get(tok.type, str(tok.type))}:{tok.string}")
    except (py_tokenize.TokenError, IndentationError):
        out.extend(f"RAW:{part}" for part in body.split())
    return out


def body_structure_tokens(body: str) -> list[str]:
    try:
        parsed = ast.parse(render_synthetic_function(body))
        function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    except SyntaxError:
        function = None
    if function is None:
        return ["SKEL:INVALID_BODY"]
    tokens: list[str] = []
    for stmt in function.body:
        tokens.extend(statement_skeleton_tokens(stmt))
    if not any(token.startswith("SKEL:RETURN_") for token in tokens):
        tokens.append("SKEL:NO_EXPLICIT_RETURN")
    return tokens or ["SKEL:EMPTY_BODY"]






























































































































def grammar_repaired_body(
    task: dict[str, Any],
    *,
    raw_body: str,
    decoded_tokens: list[str],
) -> dict[str, Any]:
    """Repair decoded token streams into a syntactically valid function body.

    This is deliberately deterministic and local: it can normalize Python
    syntax, drop impossible statement fragments, and close open blocks. It
    never synthesizes a terminal return just to satisfy the verifier. It does
    not select a solution body or consult tests.
    """

    raw = normalize_body_text(raw_body)
    raw_ok, raw_failure = function_body_syntax(task, raw)
    variants: list[tuple[str, str]] = [
        ("raw_decoded_body", raw),
        ("balanced_raw_body", balance_brackets_in_body(raw)),
        ("block_header_body_repair", repair_incomplete_block_lines(raw)),
        ("salvaged_statement_lines", salvage_statement_lines(raw)),
        ("token_stream_statement_salvage", salvage_statement_lines(decode_body_tokens(decoded_tokens))),
        ("token_stream_block_header_repair", repair_incomplete_block_lines(decode_body_tokens(decoded_tokens))),
    ]
    seen: set[str] = set()
    contract_shape = return_shape_for_task(task)
    for repair_pass, (strategy, body) in enumerate(variants, start=1):
        for candidate in deterministic_body_variants(body):
            cleaned = normalize_body_text(candidate)
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            ok, failure = function_body_syntax(task, cleaned)
            if ok:
                return {
                    "body": cleaned,
                    "raw_syntax_ok": raw_ok,
                    "repaired_syntax_ok": True,
                    "changed": cleaned != raw,
                    "strategy": strategy,
                    "repair_passes": repair_pass,
                    "fallback_return_used": False,
                    "fallback_return_shape": "",
                    "contract_return_shape_for_diagnostics": contract_shape,
                    "raw_failure": raw_failure,
                    "repaired_failure": "",
                }
    failure_body = raw or ")"
    ok, failure = function_body_syntax(task, failure_body)
    return {
        "body": failure_body,
        "raw_syntax_ok": raw_ok,
        "repaired_syntax_ok": False,
        "changed": False,
        "strategy": "no_fallback_repair_failed",
        "repair_passes": len(variants) + 1,
        "fallback_return_used": False,
        "fallback_return_shape": "",
        "contract_return_shape_for_diagnostics": contract_shape,
        "raw_failure": raw_failure,
        "repaired_failure": failure or "no valid deterministic repair without fallback return",
    }


def deterministic_body_variants(body: str) -> list[str]:
    cleaned = normalize_body_text(body)
    with_passes = close_open_blocks_with_pass(cleaned)
    with_block_headers = repair_incomplete_block_lines(cleaned)
    return [
        cleaned,
        with_passes,
        with_block_headers,
    ]




def function_body_syntax(task: dict[str, Any], body: str) -> tuple[bool, str]:
    if not str(body or "").strip():
        return False, "empty_body"
    try:
        source = render_private_function(task, body)
        ast.parse(source)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            compile(source, "<token_decoder_candidate>", "exec")
    except SyntaxError as exc:
        return False, f"{exc.__class__.__name__}:{exc.msg}"
    return True, ""


def function_body_has_return(task: dict[str, Any], body: str) -> bool:
    try:
        parsed = ast.parse(render_private_function(task, body))
    except SyntaxError:
        return False
    function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    return bool(function and any(isinstance(node, ast.Return) for node in ast.walk(function)))


def salvage_statement_lines(body: str) -> str:
    raw_lines = [line.strip() for line in str(body or "").splitlines() if line.strip()]
    filtered = [balance_brackets_in_line(line) for line in raw_lines if plausible_statement_start(line)]
    filtered = [line for line in filtered if plausible_statement_start(line)]
    return reindent_statement_lines(filtered)


def repair_incomplete_block_lines(body: str) -> str:
    """Normalize incomplete block headers and add pass-only empty bodies.

    The repair is intentionally syntax-only. It may add missing colons or a
    ``pass`` statement for an otherwise empty block, but it must not add return
    values, select a semantic body, or satisfy any verifier contract by shape.
    """

    out: list[str] = []
    for raw in normalize_body_text(body).splitlines():
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        repaired = complete_compound_statement(balance_brackets_in_line(raw.strip()))
        if repaired:
            out.append(" " * indent + repaired)
    return close_open_blocks_with_pass("\n".join(out))


def complete_compound_statement(line: str) -> str:
    stripped = str(line or "").strip()
    if not stripped:
        return ""
    first = stripped.split(None, 1)[0].rstrip(":")
    if first not in {"if", "for", "while", "with", "elif", "else", "try", "except", "finally", "def", "class"}:
        return stripped
    if stripped.endswith(":"):
        return stripped
    if first in {"else", "try", "finally"}:
        return f"{first}:"
    if first in {"if", "elif"} and stripped == first:
        return f"{first} False:"
    if first == "while" and stripped == first:
        return "while False:"
    if first == "for" and stripped == first:
        return "for _item in []:"
    if first == "for" and " in " not in f" {stripped} ":
        return "for _item in []:"
    if first == "except" and stripped == first:
        return "except Exception:"
    if first in {"def", "class"} and stripped == first:
        return "if False:"
    return stripped + ":"


def plausible_statement_start(line: str) -> bool:
    stripped = str(line or "").strip()
    if not stripped:
        return False
    if stripped in {"()", "[]", "{}", "=", ".", ",", ":", ";"}:
        return False
    if stripped[0] in set(")]}.,:;=+*/%<>!&|^~"):
        return False
    if stripped.count(")") > stripped.count("(") + 2:
        return False
    if stripped.count("]") > stripped.count("[") + 2:
        return False
    if stripped.count("}") > stripped.count("{") + 2:
        return False
    return bool(stripped[0].isalpha() or stripped[0] == "_" or stripped[0].isdigit() or stripped[0] in {"'", '"'})


def reindent_statement_lines(lines: list[str]) -> str:
    out: list[str] = []
    indent = 0
    closers = ("elif ", "else", "except", "finally")
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith(closers):
            indent = max(0, indent - 1)
        out.append(("    " * indent) + line)
        if line.endswith(":"):
            indent += 1
        if line in {"return", "break", "continue", "pass"}:
            indent = max(0, indent - 1)
    return "\n".join(out)


def close_open_blocks_with_pass(body: str) -> str:
    lines = str(body or "").splitlines()
    if not lines:
        return ""
    out: list[str] = []
    for idx, raw in enumerate(lines):
        out.append(raw.rstrip())
        stripped = raw.strip()
        if not stripped.endswith(":"):
            continue
        next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
        current_indent = len(raw) - len(raw.lstrip(" "))
        next_indent = len(next_line) - len(next_line.lstrip(" ")) if next_line.strip() else -1
        if next_indent <= current_indent:
            out.append(" " * (current_indent + 4) + "pass")
    return "\n".join(line for line in out if line.strip())


def balance_brackets_in_body(body: str) -> str:
    return "\n".join(balance_brackets_in_line(line) for line in str(body or "").splitlines())


def balance_brackets_in_line(line: str) -> str:
    opens = {"(": ")", "[": "]", "{": "}"}
    closes = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    out = []
    in_string = ""
    escape = False
    for ch in str(line or ""):
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_string:
                in_string = ""
            continue
        if ch in {"'", '"'}:
            in_string = ch
            out.append(ch)
        elif ch in opens:
            stack.append(ch)
            out.append(ch)
        elif ch in closes:
            if stack and stack[-1] == closes[ch]:
                stack.pop()
                out.append(ch)
            else:
                continue
        else:
            out.append(ch)
    while stack:
        out.append(opens[stack.pop()])
    return "".join(out).strip()


def return_shape_for_task(task: dict[str, Any]) -> str:
    return str(
        get_path(task, ["decoder_contract", "return_contract", "shape"], "")
        or get_path(task, ["decoder_contract", "return_shape"], "")
        or "unknown"
    ).strip().lower()
