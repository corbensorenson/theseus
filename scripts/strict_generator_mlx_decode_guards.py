#!/usr/bin/env python3
"""Task-blind decode guards for strict-generator MLX body-token sampling.

These helpers only inspect the generated token prefix, callable signature
surface, and prompt-visible type hints. They do not inspect tests, solutions,
public benchmark labels, answer templates, or verifier outcomes.
"""

from __future__ import annotations

from typing import Any

from neural_seed_code_proposer_comparator import dict_or_empty, get_path
from neural_seed_token_decoder_support import (
    call_arg_count,
    current_line_tokens,
    body_expression_trace_token,
    innermost_open_paren_index,
    invalid_known_builtin_arity,
    invalid_known_method_arity,
    strict_body_expression_is_likely_noniterable,
    strict_body_prefix_local_static_types,
    strict_body_static_type_from_values,
    token_values,
)


def token_blocked_by_strict_decode_guard(
    prefix: list[str],
    tok: str,
    *,
    require_nontrivial_return: bool,
    allowed_names: set[str] | None,
    input_type_hints: dict[str, str] | None = None,
) -> bool:
    if not require_nontrivial_return:
        return False
    kind, _, value = tok.partition(":")
    _lines, _current_depth, current_values = prefix_lines_with_depth(prefix)
    values = tuple(current_values)
    builtin_type_names = {"bool", "bytes", "dict", "float", "int", "list", "set", "str", "tuple"}
    if kind == "INDENT":
        if current_prefix_control_depth(prefix) + 1 > 12:
            return True
        if repeated_condition_chain_on_indent(prefix, values) >= 4:
            return True
    if kind == "NAME" and value in {"self", "cls"} and value not in set(allowed_names or set()):
        return True
    if values == ("return",) and kind == "NAME" and value in {"None", "True", "False", "not"}:
        return True
    if values == ("return",) and kind == "NAME" and value in builtin_type_names:
        return True
    if values == ("return",) and kind == "NAME" and value in set(allowed_names or set()):
        return True
    if values[:1] == ("return",) and values[-1:] in {("+",), ("-",)} and kind == "NUMBER":
        return True
    if values[-1:] in {("+",), ("-",)} and kind == "NAME" and value in {"None", "True", "False"}:
        return True
    if len(values) >= 2 and values[-2:] in {(name, "(") for name in builtin_type_names}:
        if kind == "NAME" and value in builtin_type_names:
            return True
    if kind == "NAME" and value in builtin_type_names and isinstance_guard_type_token_redundant_or_contradictory(
        prefix,
        values,
        value,
    ):
        return True
    if kind == "NAME" and method_receiver_known_invalid(prefix, values, value):
        return True
    if isinstance_first_arg_continuation_invalid(values, tok, allowed_names=allowed_names):
        return True
    if bare_builtin_type_value_argument_invalid(values, tok):
        return True
    if known_call_prefix_would_be_invalid(prefix, tok, input_type_hints=input_type_hints):
        return True
    if condition_prefix_is_only_negation(values):
        if kind == "NAME" and value in {"None", "True", "False", "not"}:
            return True
        if kind in {"NUMBER", "STRING"}:
            return True
    if values[-2:] in {("len", "("), ("sum", "("), ("sorted", "("), ("list", "("), ("tuple", "("), ("set", "(")}:
        if kind in {"NUMBER", "STRING"}:
            return True
        if kind == "NAME" and value in {"None", "True", "False"}:
            return True
        if kind == "OP" and value in {"(", "[", "{"}:
            return True
    if values[-1:] == (".",) and kind != "NAME":
        return True
    if values[-1:] == (".",) and kind == "NAME" and value.startswith("__"):
        return True
    return False


def bare_builtin_type_value_argument_invalid(values: tuple[str, ...], tok: str) -> bool:
    """Reject bare type descriptors where a runtime value argument is needed."""

    kind, _, value = tok.partition(":")
    builtin_type_names = {"bool", "bytes", "dict", "float", "int", "list", "set", "str", "tuple"}
    if kind != "NAME" or value not in builtin_type_names:
        return False
    line_values = list(values)
    open_index = innermost_open_paren_index(line_values)
    if open_index <= 0:
        if line_values[:1] == ["return"]:
            return True
        return False
    callee = line_values[open_index - 1]
    is_method = open_index >= 2 and line_values[open_index - 2] == "."
    current_args = line_values[open_index + 1 :]
    if callee == "isinstance" and top_level_comma_seen(current_args):
        return False
    if callee in builtin_type_names and not is_method:
        # Constructors like list(data) are legal; their own invalid nested type
        # argument is handled by existing call-prefix checks.
        return False
    return True


def isinstance_first_arg_continuation_invalid(
    values: tuple[str, ...],
    tok: str,
    *,
    allowed_names: set[str] | None,
) -> bool:
    """Reject runaway boolean/arithmetic first arguments to isinstance().

    The strict generator often tries to turn ``isinstance(`` into
    ``isinstance((data) and ...``. That is grammar-shaped noise, not a useful
    learned branch. This guard is task-blind: it only constrains the first
    argument position of an already-generated ``isinstance`` call.
    """

    kind, _, value = tok.partition(":")
    line_values = list(values)
    open_index = innermost_open_paren_index(line_values)
    if open_index <= 0 or line_values[open_index - 1] != "isinstance":
        return False
    current_args = line_values[open_index + 1 :]
    if top_level_comma_seen(current_args):
        return False
    visible_names = {str(name) for name in set(allowed_names or set()) if str(name)}
    if not current_args:
        if kind == "OP" and value in {"(", "{", "["}:
            return True
        if kind == "NAME" and value in {"None", "True", "False", "list", "tuple", "dict", "set", "str", "int", "float", "bool"}:
            return True
        return False
    if kind == "NAME" and value in {"and", "or", "not", "in", "is"}:
        return True
    if kind == "OP" and value in {
        "+",
        "-",
        "*",
        "/",
        "//",
        "%",
        "&",
        "|",
        "^",
        "<<",
        ">>",
        "<",
        "<=",
        ">",
        ">=",
        "==",
        "!=",
    }:
        return True
    if kind == "OP" and value == "(":
        return True
    if kind == "NAME" and visible_names and value not in visible_names and not any(arg in {".", "["} for arg in current_args):
        # The first argument should be a visible object or a simple access from
        # one. Type names and builtins belong after the comma.
        builtin_names = {
            "abs",
            "all",
            "any",
            "bool",
            "bytes",
            "dict",
            "enumerate",
            "filter",
            "float",
            "int",
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
        }
        return value in builtin_names
    return False


def top_level_comma_seen(values: list[str]) -> bool:
    depth = 0
    for value in values:
        if value in {"(", "[", "{"}:
            depth += 1
        elif value in {")", "]", "}"}:
            depth = max(0, depth - 1)
        elif value == "," and depth == 0:
            return True
    return False


def known_call_prefix_would_be_invalid(
    prefix: list[str],
    tok: str,
    *,
    input_type_hints: dict[str, str] | None = None,
) -> bool:
    """Reject task-blind builtin/method call continuations that are already invalid."""

    kind, _, value = tok.partition(":")
    if kind != "OP" or value not in {",", ")"}:
        return False
    candidate_tokens = list(prefix)
    local_types = strict_body_prefix_local_static_types(candidate_tokens)
    line_values = token_values(current_line_tokens(candidate_tokens))
    open_index = innermost_open_paren_index(line_values)
    if open_index <= 0:
        return False
    callee = line_values[open_index - 1]
    is_method = open_index >= 2 and line_values[open_index - 2] == "."
    receiver = line_values[open_index - 3] if is_method and open_index >= 3 else ""
    for name, type_name in dict_or_empty(input_type_hints).items():
        if name and type_name and name not in local_types:
            local_types[name] = str(type_name)
    receiver_type = local_types.get(receiver, "") if receiver else ""
    current_args = line_values[open_index + 1 :]
    argc_before = call_arg_count(current_args)
    if value == ",":
        if callee == "isinstance" and isinstance_first_arg_values_invalid(first_call_argument_values(current_args)):
            return True
        max_args = known_positional_max_args(callee, is_method=is_method, receiver_type=receiver_type)
        if max_args is not None and argc_before >= max_args:
            return True
        return False
    argc_after = call_arg_count(current_args)
    if is_method:
        if invalid_known_method_arity(receiver_type, callee, argc_after):
            return True
    elif invalid_known_builtin_arity(callee, argc_after):
        return True
    return known_builtin_first_arg_type_invalid(callee, current_args, local_types=local_types)


def known_positional_max_args(callee: str, *, is_method: bool, receiver_type: str) -> int | None:
    if is_method:
        if callee in {"append", "add", "discard", "extend", "join", "remove", "update"}:
            return 1
        if callee in {
            "casefold",
            "clear",
            "copy",
            "isalnum",
            "isalpha",
            "isdigit",
            "items",
            "keys",
            "lower",
            "reverse",
            "sort",
            "splitlines",
            "upper",
            "values",
        }:
            return 0
        if callee in {"get", "setdefault", "split", "strip"}:
            return 2
        if callee == "insert":
            return 2
        if callee == "pop":
            return 1
        return None
    if callee in {
        "abs",
        "all",
        "any",
        "bool",
        "bytes",
        "dict",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "reversed",
        "set",
        "sorted",
        "str",
        "tuple",
    }:
        return 1
    if callee in {"filter", "isinstance", "map", "round", "sum"}:
        return 2
    if callee == "range":
        return 3
    return None


def known_builtin_first_arg_type_invalid(
    callee: str,
    arg_values: list[str],
    *,
    local_types: dict[str, str],
) -> bool:
    if not callee or not arg_values:
        return False
    first_arg = first_call_argument_values(arg_values)
    if not first_arg:
        return False
    inferred = strict_body_static_type_from_values(first_arg)
    if not inferred and len(first_arg) == 1:
        inferred = local_types.get(first_arg[0], "")
    if not inferred and strict_body_expression_is_likely_noniterable(first_arg, local_types=local_types):
        inferred = "int"
    if callee == "isinstance":
        return isinstance_first_arg_values_invalid(first_arg)
    iterable_builtins = {"all", "any", "enumerate", "len", "list", "max", "min", "reversed", "set", "sorted", "sum", "tuple"}
    if callee in iterable_builtins:
        return inferred in {"bool", "float", "int"}
    numeric_scalar_builtins = {"abs", "round"}
    if callee in numeric_scalar_builtins:
        return inferred in {"dict", "list", "set", "str", "tuple"}
    return False


def isinstance_first_arg_values_invalid(values: list[str]) -> bool:
    """Reject bool/comparison expressions as the object tested by isinstance.

    This is generated-prefix hygiene only. A learned body can still choose
    which object to test from visible prompt/signature context, but
    `isinstance(data in xs and ..., list)` is almost always a syntax-shaped
    hallucination that passes parsing while destroying branch semantics.
    """

    if not values:
        return False
    depth = 0
    comparison_tokens = {"<", "<=", ">", ">=", "==", "!=", "is", "in"}
    bool_tokens = {"and", "or", "not"}
    for value in values:
        if value in {"(", "[", "{"}:
            depth += 1
            continue
        if value in {")", "]", "}"}:
            depth = max(0, depth - 1)
            continue
        if depth == 0 and value in comparison_tokens | bool_tokens:
            return True
    return False


def first_call_argument_values(values: list[str]) -> list[str]:
    out: list[str] = []
    depth = 0
    for value in values:
        if value == "," and depth == 0:
            break
        out.append(value)
        if value in {"(", "[", "{"}:
            depth += 1
        elif value in {")", "]", "}"}:
            depth = max(0, depth - 1)
    return out


def isinstance_guard_type_token_redundant_or_contradictory(
    prefix: list[str],
    values: tuple[str, ...],
    type_name: str,
) -> bool:
    compact = [value for value in values if value not in {"(", ")", ",", ":"}]
    if len(compact) < 3 or compact[0] not in {"if", "elif"} or compact[1] != "isinstance":
        return False
    receiver_name = compact[2]
    if not receiver_name.isidentifier():
        return False
    active_type = current_branch_receiver_type(prefix, receiver_name)
    if active_type and active_type != type_name:
        return True
    if active_type and active_type == type_name:
        return True
    return current_else_excludes_receiver_type(prefix, receiver_name, type_name)


def method_receiver_known_invalid(prefix: list[str], values: tuple[str, ...], method_name: str) -> bool:
    if not values or values[-1] != ".":
        return False
    receiver = values[:-1]
    receiver_name = receiver[-1] if receiver and str(receiver[-1]).isidentifier() else ""
    dict_only_methods = {"get", "items", "keys", "popitem", "setdefault", "update", "values"}
    string_only_methods = {
        "casefold",
        "endswith",
        "format",
        "join",
        "lower",
        "replace",
        "split",
        "splitlines",
        "startswith",
        "strip",
        "upper",
    }
    mutating_sequence_methods = {"append", "add", "clear", "extend", "insert", "pop", "remove", "sort"}
    receiver_kind = infer_receiver_tail_kind(receiver)
    local_types = strict_body_prefix_local_static_types(prefix)
    local_kind = local_types.get(receiver_name, "") if receiver_name else ""
    if local_kind:
        receiver_kind = {
            "bool": "scalar",
            "float": "scalar",
            "int": "scalar",
            "str": "string",
        }.get(local_kind, local_kind)
    if receiver_kind == "scalar":
        return not method_name.startswith("__")
    if receiver_kind == "none":
        return True
    if receiver_kind == "string":
        return method_name not in string_only_methods and not method_name.startswith("__")
    if receiver_kind == "list":
        list_methods = {"append", "clear", "copy", "count", "extend", "index", "insert", "pop", "remove", "reverse", "sort"}
        return method_name not in list_methods and not method_name.startswith("__")
    if receiver_kind == "tuple":
        tuple_methods = {"count", "index"}
        return method_name not in tuple_methods and not method_name.startswith("__")
    if receiver_kind == "set":
        set_methods = {"add", "clear", "copy", "difference", "discard", "intersection", "pop", "remove", "union", "update"}
        return method_name not in set_methods and not method_name.startswith("__")
    if receiver_kind == "dict":
        dict_methods = {"clear", "copy", "get", "items", "keys", "pop", "popitem", "setdefault", "update", "values"}
        return method_name not in dict_methods and not method_name.startswith("__")
    branch_kind = current_branch_receiver_type(prefix, receiver_name) if receiver_name else ""
    if branch_kind == "str":
        return method_name not in string_only_methods and not method_name.startswith("__")
    if branch_kind in {"list", "tuple", "set"}:
        return method_name in dict_only_methods or method_name in string_only_methods
    if branch_kind == "dict":
        return method_name in string_only_methods
    if receiver_name and method_name in string_only_methods and current_else_excludes_receiver_type(prefix, receiver_name, "str"):
        return True
    if receiver_name and method_name in dict_only_methods and current_else_excludes_receiver_type(prefix, receiver_name, "dict"):
        return True
    if method_name in dict_only_methods and receiver_tail_is_known_non_dict(receiver):
        return True
    if method_name in string_only_methods and receiver_tail_is_known_non_string(receiver):
        return True
    if method_name in mutating_sequence_methods and receiver_tail_is_temporary_expression(receiver):
        return True
    return False


def current_branch_receiver_type(prefix: list[str], receiver_name: str) -> str:
    lines, current_depth, _current_values = prefix_lines_with_depth(prefix)
    if not receiver_name or current_depth <= 0:
        return ""
    for depth, values in reversed(lines):
        if depth >= current_depth or not values:
            continue
        inferred = isinstance_guard_type(values, receiver_name)
        if inferred:
            return inferred
    return ""


def current_else_excludes_receiver_type(prefix: list[str], receiver_name: str, type_name: str) -> bool:
    lines, current_depth, current_values = prefix_lines_with_depth(prefix)
    if not current_values or current_values[-1:] != ["."]:
        return False
    if receiver_name not in current_values:
        return False
    else_depth = current_depth - 1
    if else_depth < 0:
        return False
    previous_same_or_parent = [
        (depth, values)
        for depth, values in lines
        if values and depth <= else_depth
    ]
    last_else_index = -1
    for index, (depth, values) in enumerate(previous_same_or_parent):
        if depth == else_depth and values[:1] == ["else"]:
            last_else_index = index
    if last_else_index < 0:
        return False
    for depth, values in reversed(previous_same_or_parent[:last_else_index]):
        if depth != else_depth:
            continue
        if values[:2] not in (["if", "isinstance"], ["elif", "isinstance"]):
            return False
        return isinstance_guard_matches(values, receiver_name, type_name)
    return False


def isinstance_guard_matches(values: list[str], receiver_name: str, type_name: str) -> bool:
    return isinstance_guard_type(values, receiver_name) == type_name


def isinstance_guard_type(values: list[str], receiver_name: str) -> str:
    compact = [value for value in values if value not in {"(", ")", ",", ":"}]
    builtin_type_names = {"bool", "bytes", "dict", "float", "int", "list", "set", "str", "tuple"}
    if (
        len(compact) >= 4
        and compact[0] in {"if", "elif"}
        and compact[1] == "isinstance"
        and compact[2] == receiver_name
        and compact[3] in builtin_type_names
    ):
        return compact[3]
    return ""


def prefix_lines_with_depth(prefix: list[str]) -> tuple[list[tuple[int, list[str]]], int, list[str]]:
    lines: list[tuple[int, list[str]]] = []
    depth = 0
    current: list[str] = []
    current_depth = 0
    for tok in prefix:
        if body_expression_trace_token(tok):
            continue
        kind, _, value = tok.partition(":")
        if kind == "DEDENT":
            if current:
                lines.append((current_depth, current))
                current = []
            depth = max(0, depth - 1)
            current_depth = depth
            continue
        if kind == "INDENT":
            if current:
                lines.append((current_depth, current))
                current = []
            depth += 1
            current_depth = depth
            continue
        if kind == "NEWLINE":
            if current:
                lines.append((current_depth, current))
                current = []
            current_depth = depth
            continue
        if tok == "<eos>":
            continue
        if not current:
            current_depth = depth
        current.append(value)
    return lines, current_depth, current


def infer_receiver_tail_kind(values: tuple[str, ...]) -> str:
    call_name = receiver_tail_call_name(values)
    if call_name in {"abs", "all", "any", "bool", "float", "int", "len", "max", "min", "round", "sum"}:
        return "scalar"
    if call_name in {"str", "casefold", "format", "join", "lower", "replace", "strip", "upper"}:
        return "string"
    if call_name in {"split", "splitlines"}:
        return "list"
    if call_name in {"list", "sorted"}:
        return "list"
    if call_name == "tuple":
        return "tuple"
    if call_name == "set":
        return "set"
    if call_name == "dict":
        return "dict"
    tail = expression_tail_values(values)
    if not tail:
        return ""
    if tail[0] == "[":
        return "list"
    if tail[0] == "(":
        return "tuple"
    if tail[0] == "{":
        return "dict_or_set"
    if tail[0] in {"0", "1", "True", "False"} or tail[0].replace(".", "", 1).isdigit():
        return "scalar"
    if tail[0] in {"None"}:
        return "none"
    if len(tail) == 1 and tail[0].startswith(("'", '"')):
        return "string"
    return ""


def receiver_tail_is_known_non_dict(values: tuple[str, ...]) -> bool:
    receiver_kind = infer_receiver_tail_kind(values)
    if receiver_kind in {"scalar", "none", "string", "list", "tuple", "set"}:
        return True
    call_name = receiver_tail_call_name(values)
    if call_name in {
        "abs",
        "all",
        "any",
        "bool",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
    }:
        return True
    tail = expression_tail_values(values)
    return bool(tail and tail[0] in {"[", "(", "{", "0", "1", "None", "True", "False"})


def receiver_tail_is_known_non_string(values: tuple[str, ...]) -> bool:
    receiver_kind = infer_receiver_tail_kind(values)
    if receiver_kind in {"scalar", "none", "dict", "list", "tuple", "set"}:
        return True
    call_name = receiver_tail_call_name(values)
    if call_name in {"abs", "all", "any", "bool", "dict", "float", "int", "len", "list", "max", "min", "round", "set", "sorted", "sum", "tuple"}:
        return True
    tail = expression_tail_values(values)
    return bool(tail and tail[0] in {"[", "(", "{", "0", "1", "None", "True", "False"})


def receiver_tail_is_temporary_expression(values: tuple[str, ...]) -> bool:
    call_name = receiver_tail_call_name(values)
    if call_name in {"dict", "list", "set", "sorted", "tuple"}:
        return True
    tail = expression_tail_values(values)
    return bool(tail and tail[0] in {"[", "(", "{"})


def receiver_tail_call_name(values: tuple[str, ...]) -> str:
    if not values or values[-1] != ")":
        return ""
    depth = 0
    for index in range(len(values) - 1, -1, -1):
        value = values[index]
        if value == ")":
            depth += 1
        elif value == "(":
            depth -= 1
            if depth == 0:
                return values[index - 1] if index > 0 and values[index - 1].isidentifier() else ""
    return ""


def expression_tail_values(values: tuple[str, ...]) -> tuple[str, ...]:
    boundaries = {"return", "=", "if", "elif", "while", "for", "in", "and", "or", ","}
    depth = 0
    for index in range(len(values) - 1, -1, -1):
        value = values[index]
        if value in {")", "]", "}"}:
            depth += 1
        elif value in {"(", "[", "{"}:
            depth = max(0, depth - 1)
        if depth == 0 and value in boundaries:
            return values[index + 1 :]
    return values


def condition_prefix_is_only_negation(values: tuple[str, ...]) -> bool:
    if not values or values[0] not in {"if", "elif", "while"}:
        return False
    return all(value == "not" for value in values[1:])


def current_prefix_control_depth(prefix: list[str]) -> int:
    _lines, current_depth, _current_values = prefix_lines_with_depth(prefix)
    return int(current_depth)


def normalized_condition_values(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if not values or values[0] not in {"if", "elif", "while"}:
        return ()
    return tuple(value for value in list(values)[1:] if value not in {":"})


def repeated_condition_chain_on_indent(prefix: list[str], values: tuple[str, ...]) -> int:
    condition = normalized_condition_values(values)
    if not condition:
        return 0
    lines, current_depth, _current_values = prefix_lines_with_depth(prefix)
    repeats = 1
    expected_depth = current_depth - 1
    for depth, line_values in reversed(lines):
        if depth > expected_depth:
            continue
        if depth < expected_depth:
            break
        ancestor_condition = normalized_condition_values(line_values)
        if ancestor_condition != condition:
            break
        repeats += 1
        expected_depth -= 1
        if expected_depth < 0:
            break
    return repeats


def current_body_line_values(prefix: list[str]) -> tuple[str, ...]:
    current: list[str] = []
    for tok in prefix:
        kind, _, value = tok.partition(":")
        if kind == "NEWLINE":
            current = []
            continue
        if kind in {"INDENT", "DEDENT"} or tok == "<eos>":
            continue
        current.append(value)
    return tuple(current)


def prefix_mentions_allowed_parameter(prefix: list[str], *, allowed_names: set[str]) -> bool:
    for tok in prefix:
        kind, _, value = tok.partition(":")
        if kind == "NAME" and value in allowed_names:
            return True
    return False


def allowed_signature_names_for_task(task: dict[str, Any]) -> set[str]:
    argc = int(get_path(task, ["decoder_contract", "visible_arg_count_hint"], 1) or 1)
    names = {"data"}
    if argc >= 2:
        names.add("other")
    if argc > 2:
        names.add("extra")
    return names
