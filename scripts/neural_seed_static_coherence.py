#!/usr/bin/env python3
"""Task-blind static coherence checks for decoded code candidates.

This module is a mechanical extraction from
``neural_seed_token_decoder_comparator.py``. It intentionally uses only the
candidate code plus visible prompt/signature-derived constraints supplied by
callers; it does not inspect tests, solutions, categories, or answer metadata.
"""

from __future__ import annotations

import ast
from collections import Counter
from typing import Any

from neural_seed_code_proposer_comparator import dict_or_empty, get_path


STATIC_COHERENCE_ALLOWED_GLOBALS = {
    "abs",
    "all",
    "any",
    "bool",
    "bytes",
    "chr",
    "collections",
    "dict",
    "enumerate",
    "Exception",
    "False",
    "filter",
    "float",
    "functools",
    "int",
    "isinstance",
    "itertools",
    "len",
    "list",
    "map",
    "math",
    "max",
    "min",
    "None",
    "ord",
    "pow",
    "print",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "True",
    "ValueError",
    "zip",
}

STATIC_COHERENCE_BUILTIN_TYPE_OBJECTS = {
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


def static_coherence_ranker_enabled(config: dict[str, Any]) -> bool:
    cfg = get_path(config, ["body_structure_decoder", "static_coherence_ranker"], {})
    if not isinstance(cfg, dict):
        return False
    return bool(cfg.get("enabled", False))


def static_coherence_candidate_pool_size(config: dict[str, Any], budget: dict[str, Any]) -> int:
    fanout = int(budget.get("fanout_top_k") or 1)
    if not static_coherence_ranker_enabled(config):
        return fanout
    cfg = dict_or_empty(get_path(config, ["body_structure_decoder", "static_coherence_ranker"], {}))
    configured = int(cfg.get("candidate_pool_size") or budget.get("static_coherence_candidate_pool_size") or 0)
    return max(fanout, configured)


def allowed_signature_names_for_task(task: dict[str, Any]) -> set[str]:
    argc = int(get_path(task, ["decoder_contract", "visible_arg_count_hint"], 1) or 1)
    names = {"data"}
    if argc >= 2:
        names.add("other")
    if argc > 2:
        names.add("extra")
    return names


def candidate_static_coherence(code: str) -> dict[str, Any]:
    """Task-blind static hygiene for already decoded candidate code.

    The analyzer never sees tests, solutions, families, categories, or answer
    metadata. It only checks whether the rendered callable refers to names that
    are not defined by its own signature/body or by the verifier's fixed Python
    prelude. This is a linter-style rank signal, not a semantic renderer.
    """

    try:
        parsed = ast.parse(str(code or ""))
    except SyntaxError as exc:
        return {
            "policy": "prompt_signature_static_coherence_v1",
            "parse_ok": False,
            "has_function": False,
            "has_return": False,
            "valued_return_count": 0,
            "bare_return_count": 0,
            "trivial_return_count": 0,
            "nontrivial_return_count": 0,
            "top_level_valued_return_count": 0,
            "nested_return_count": 999,
            "return_uses_parameter_count": 0,
            "literal_only_return_count": 0,
            "parameter_free_literal_expression_return_count": 0,
            "parameter_load_count": 0,
            "used_parameter_count": 0,
            "body_statement_count": 0,
            "control_flow_count": 0,
            "assignment_count": 0,
            "call_count": 0,
            "invalid_receiver_count": 999,
            "builtin_type_descriptor_receiver_count": 999,
            "bare_builtin_condition_count": 999,
            "invalid_known_builtin_arity_count": 999,
            "invalid_known_local_receiver_count": 999,
            "invalid_known_local_call_count": 999,
            "invalid_known_local_iter_count": 999,
            "invalid_multi_assign_from_scalar_count": 999,
            "mutating_method_return_value_count": 999,
            "ignored_pure_call_expression_count": 999,
            "comprehension_count": 0,
            "inert_stub": True,
            "undefined_name_count": 999,
            "undefined_names": [],
            "unexpected_signature_name_count": 0,
            "unexpected_signature_names": [],
            "score": -999.0,
            "failure": f"{exc.__class__.__name__}: {exc.msg}",
            "uses_tests_or_solutions": False,
            "uses_public_data": False,
        }
    function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    if function is None:
        return {
            "policy": "prompt_signature_static_coherence_v1",
            "parse_ok": True,
            "has_function": False,
            "has_return": False,
            "valued_return_count": 0,
            "bare_return_count": 0,
            "trivial_return_count": 0,
            "nontrivial_return_count": 0,
            "top_level_valued_return_count": 0,
            "nested_return_count": 999,
            "return_uses_parameter_count": 0,
            "literal_only_return_count": 0,
            "parameter_free_literal_expression_return_count": 0,
            "parameter_load_count": 0,
            "used_parameter_count": 0,
            "body_statement_count": 0,
            "control_flow_count": 0,
            "assignment_count": 0,
            "call_count": 0,
            "invalid_receiver_count": 999,
            "builtin_type_descriptor_receiver_count": 999,
            "bare_builtin_condition_count": 999,
            "invalid_known_builtin_arity_count": 999,
            "invalid_known_local_receiver_count": 999,
            "invalid_known_local_call_count": 999,
            "invalid_known_local_iter_count": 999,
            "invalid_multi_assign_from_scalar_count": 999,
            "mutating_method_return_value_count": 999,
            "ignored_pure_call_expression_count": 999,
            "comprehension_count": 0,
            "inert_stub": True,
            "undefined_name_count": 999,
            "undefined_names": [],
            "unexpected_signature_name_count": 0,
            "unexpected_signature_names": [],
            "score": -500.0,
            "failure": "missing_function_def",
            "uses_tests_or_solutions": False,
            "uses_public_data": False,
        }

    params = function_arg_names(function)
    assigned = set(params)
    assigned.add(function.name)
    loads: Counter[str] = Counter()
    stores: Counter[str] = Counter()
    imports: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                loads[node.id] += 1
            elif isinstance(node.ctx, (ast.Store, ast.Del)):
                assigned.add(node.id)
                stores[node.id] += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assigned.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            assigned.add(str(node.name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                local = alias.asname or str(alias.name).split(".", maxsplit=1)[0]
                assigned.add(local)
                imports.add(local)

    undefined = sorted(
        name
        for name in loads
        if name not in assigned and name not in STATIC_COHERENCE_ALLOWED_GLOBALS
    )
    unexpected_signature_names = sorted(
        name
        for name in {"other", "extra"}
        if loads.get(name, 0) > 0 and name not in params and name not in stores
    )
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    has_return = bool(returns)
    param_set = set(params)
    primary_parameter = params[0] if params else ""
    auxiliary_parameters = set(params[1:]) if len(params) > 1 else set()
    parameter_load_count = sum(int(loads.get(param, 0)) for param in params)
    used_parameter_names = sorted(param for param in params if int(loads.get(param, 0)) > 0)
    primary_parameter_load_count = int(loads.get(primary_parameter, 0)) if primary_parameter else 0
    valued_return_count = sum(1 for node in returns if node.value is not None)
    bare_return_count = sum(1 for node in returns if node.value is None)
    trivial_return_count = sum(1 for node in returns if return_value_is_trivial(node.value))
    top_level_valued_return_count = sum(
        1
        for node in function.body
        if isinstance(node, ast.Return) and node.value is not None and not return_value_is_trivial(node.value)
    )
    nested_return_count = sum(1 for node in returns if not any(node is top for top in function.body))
    return_uses_parameter_count = sum(
        1
        for node in returns
        if node.value is not None and expression_uses_any_name(node.value, param_set)
    )
    primary_parameter_return_count = sum(
        1
        for node in returns
        if primary_parameter and node.value is not None and expression_uses_any_name(node.value, {primary_parameter})
    )
    auxiliary_parameter_return_count = sum(
        1
        for node in returns
        if auxiliary_parameters and node.value is not None and expression_uses_any_name(node.value, auxiliary_parameters)
    )
    return_only_uses_auxiliary_parameter_count = sum(
        1
        for node in returns
        if auxiliary_parameters
        and node.value is not None
        and expression_uses_any_name(node.value, auxiliary_parameters)
        and not (primary_parameter and expression_uses_any_name(node.value, {primary_parameter}))
    )
    literal_only_return_count = sum(
        1
        for node in returns
        if node.value is not None and expression_is_static_literal_only(node.value)
    )
    parameter_free_literal_expression_return_count = sum(
        1
        for node in returns
        if node.value is not None
        and not expression_uses_any_name(node.value, param_set)
        and expression_is_parameter_free_literal_expression(node.value)
    )
    nontrivial_return_count = sum(
        1
        for node in returns
        if node.value is not None
        and (
            expression_uses_any_name(node.value, param_set)
            or expression_complexity_score(node.value) >= 2
        )
        and not expression_is_static_literal_only(node.value)
    )
    local_static_types = local_static_type_bindings(function)
    expression_hygiene = expression_hygiene_counts(function, local_static_types=local_static_types)
    self_dependent_assignment_count = self_dependent_assignment_load_count(function, initial_bound=set(params))
    repeated_identical_condition_chain_count = max_repeated_identical_condition_chain(function)
    literal_call_count = int(expression_hygiene.get("literal_call_count", 0))
    invalid_receiver_count = int(expression_hygiene.get("invalid_receiver_count", 0))
    builtin_type_descriptor_receiver_count = int(expression_hygiene.get("builtin_type_descriptor_receiver_count", 0))
    bare_builtin_condition_count = int(expression_hygiene.get("bare_builtin_condition_count", 0))
    invalid_known_builtin_arity_count = int(expression_hygiene.get("invalid_known_builtin_arity_count", 0))
    invalid_known_local_receiver_count = int(expression_hygiene.get("invalid_known_local_receiver_count", 0))
    invalid_known_local_call_count = int(expression_hygiene.get("invalid_known_local_call_count", 0))
    invalid_known_local_iter_count = int(expression_hygiene.get("invalid_known_local_iter_count", 0))
    invalid_multi_assign_from_scalar_count = int(expression_hygiene.get("invalid_multi_assign_from_scalar_count", 0))
    mutating_method_return_value_count = int(expression_hygiene.get("mutating_method_return_value_count", 0))
    no_effect_expression_count = int(expression_hygiene.get("no_effect_expression_count", 0))
    ignored_pure_call_expression_count = int(expression_hygiene.get("ignored_pure_call_expression_count", 0))
    undefined_count = len(undefined)
    unexpected_count = len(unexpected_signature_names)
    body_statement_count = len(function.body)
    local_store_name_count = len(stores)
    control_flow_count = sum(1 for node in ast.walk(function) if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)))
    assignment_count = sum(1 for node in ast.walk(function) if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)))
    call_count = sum(1 for node in ast.walk(function) if isinstance(node, ast.Call))
    comprehension_count = sum(
        1
        for node in ast.walk(function)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
    )
    inert_stub = (
        not has_return
        or valued_return_count == 0
        or (bool(returns) and trivial_return_count == len(returns))
        or (
            body_statement_count <= 2
            and nontrivial_return_count == 0
            and local_store_name_count == 0
            and assignment_count == 0
            and call_count == 0
            and comprehension_count == 0
        )
    )
    score = 1.0
    score += 1.25 * nontrivial_return_count
    score += 0.6 * valued_return_count
    score += 1.0 if top_level_valued_return_count > 0 else 0.0
    if nested_return_count > 0 and top_level_valued_return_count == 0:
        score -= 1.25 * nested_return_count
    elif nested_return_count > 0:
        score -= 0.35 * nested_return_count
    score -= 1.5 if valued_return_count == 0 else 0.0
    score -= 1.25 * bare_return_count
    score -= 1.0 * trivial_return_count
    score += min(1.0, 0.35 * len(used_parameter_names) + 0.05 * parameter_load_count)
    if params and not used_parameter_names:
        score -= 1.0
    if primary_parameter and primary_parameter_load_count == 0:
        score -= 1.25
    if primary_parameter and primary_parameter_return_count == 0 and return_only_uses_auxiliary_parameter_count > 0:
        score -= 1.5 * return_only_uses_auxiliary_parameter_count
    score += min(0.9, 0.2 * local_store_name_count + 0.15 * (assignment_count + control_flow_count + call_count + comprehension_count))
    score += min(0.4, 0.08 * body_statement_count)
    score -= 1.0 * undefined_count
    score -= 1.75 * self_dependent_assignment_count
    score -= 1.5 * unexpected_count
    score -= 1.25 * literal_call_count
    score -= 1.25 * invalid_receiver_count
    score -= 1.5 * invalid_known_local_receiver_count
    score -= 1.25 * builtin_type_descriptor_receiver_count
    score -= 1.0 * bare_builtin_condition_count
    score -= 1.0 * invalid_known_builtin_arity_count
    score -= 1.5 * mutating_method_return_value_count
    score -= 1.75 * invalid_known_local_call_count
    score -= 1.75 * invalid_known_local_iter_count
    score -= 1.75 * invalid_multi_assign_from_scalar_count
    score -= 0.75 * no_effect_expression_count
    score -= 0.75 * ignored_pure_call_expression_count
    score -= 0.75 * literal_only_return_count
    score -= 1.0 * parameter_free_literal_expression_return_count
    score -= 0.75 * max(0, repeated_identical_condition_chain_count - 1)
    if inert_stub:
        score -= 2.0
    return {
        "policy": "prompt_signature_static_coherence_v1",
        "parse_ok": True,
        "has_function": True,
        "has_return": has_return,
        "valued_return_count": valued_return_count,
        "bare_return_count": bare_return_count,
        "trivial_return_count": trivial_return_count,
        "nontrivial_return_count": nontrivial_return_count,
        "top_level_valued_return_count": top_level_valued_return_count,
        "nested_return_count": nested_return_count,
        "return_uses_parameter_count": return_uses_parameter_count,
        "primary_parameter_name": primary_parameter,
        "primary_parameter_load_count": primary_parameter_load_count,
        "primary_parameter_return_count": primary_parameter_return_count,
        "auxiliary_parameter_return_count": auxiliary_parameter_return_count,
        "return_only_uses_auxiliary_parameter_count": return_only_uses_auxiliary_parameter_count,
        "literal_only_return_count": literal_only_return_count,
        "parameter_free_literal_expression_return_count": parameter_free_literal_expression_return_count,
        "literal_call_count": literal_call_count,
        "invalid_receiver_count": invalid_receiver_count,
        "builtin_type_descriptor_receiver_count": builtin_type_descriptor_receiver_count,
        "bare_builtin_condition_count": bare_builtin_condition_count,
        "invalid_known_builtin_arity_count": invalid_known_builtin_arity_count,
        "invalid_known_local_receiver_count": invalid_known_local_receiver_count,
        "invalid_known_local_call_count": invalid_known_local_call_count,
        "invalid_known_local_iter_count": invalid_known_local_iter_count,
        "invalid_multi_assign_from_scalar_count": invalid_multi_assign_from_scalar_count,
        "mutating_method_return_value_count": mutating_method_return_value_count,
        "ignored_pure_call_expression_count": ignored_pure_call_expression_count,
        "no_effect_expression_count": no_effect_expression_count,
        "expression_statement_count": int(expression_hygiene.get("expression_statement_count", 0)),
        "self_dependent_assignment_count": self_dependent_assignment_count,
        "max_repeated_identical_condition_chain": repeated_identical_condition_chain_count,
        "local_static_type_bindings": local_static_types,
        "undefined_name_count": undefined_count,
        "undefined_names": undefined[:12],
        "unexpected_signature_name_count": unexpected_count,
        "unexpected_signature_names": unexpected_signature_names,
        "parameter_names": params,
        "parameter_load_count": parameter_load_count,
        "used_parameter_count": len(used_parameter_names),
        "used_parameter_names": used_parameter_names,
        "local_store_name_count": local_store_name_count,
        "import_name_count": len(imports),
        "body_statement_count": body_statement_count,
        "control_flow_count": control_flow_count,
        "assignment_count": assignment_count,
        "call_count": call_count,
        "comprehension_count": comprehension_count,
        "inert_stub": inert_stub,
        "score": round(score, 6),
        "failure": "",
        "uses_tests_or_solutions": False,
        "uses_public_data": False,
    }


def expression_hygiene_counts(function: ast.FunctionDef, *, local_static_types: dict[str, str]) -> dict[str, int]:
    literal_call_count = 0
    invalid_receiver_count = 0
    builtin_type_descriptor_receiver_count = 0
    bare_builtin_condition_count = 0
    invalid_known_builtin_arity_count = 0
    invalid_known_local_receiver_count = 0
    invalid_known_local_call_count = 0
    invalid_known_local_iter_count = 0
    invalid_multi_assign_from_scalar_count = 0
    mutating_method_return_value_count = 0
    no_effect_expression_count = 0
    ignored_pure_call_expression_count = 0
    expression_statement_count = 0

    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Set)):
            literal_call_count += 1
        elif isinstance(node, ast.Call) and expression_has_invalid_literal_receiver(node.func):
            literal_call_count += 1
        if isinstance(node, ast.Attribute) and expression_has_invalid_method_receiver(node):
            invalid_receiver_count += 1
        if isinstance(node, ast.Attribute) and expression_has_builtin_type_descriptor_receiver(node):
            builtin_type_descriptor_receiver_count += 1
        if isinstance(node, (ast.If, ast.While)) and expression_has_bare_builtin_condition(node.test):
            bare_builtin_condition_count += 1
        if isinstance(node, ast.Call) and (
            expression_has_invalid_known_builtin_arity(node)
            or expression_has_invalid_known_builtin_arg_type(node)
            or expression_has_invalid_known_method_arity(node, local_static_types=local_static_types)
        ):
            invalid_known_builtin_arity_count += 1
        if isinstance(node, ast.Attribute) and expression_has_invalid_known_local_receiver(
            node, local_static_types=local_static_types
        ):
            invalid_known_local_receiver_count += 1
        if isinstance(node, ast.Call) and expression_has_invalid_known_local_call(
            node, local_static_types=local_static_types
        ):
            invalid_known_local_call_count += 1
        if isinstance(node, ast.For) and expression_has_invalid_known_local_iter(
            node, local_static_types=local_static_types
        ):
            invalid_known_local_iter_count += 1
        if isinstance(node, ast.Assign) and assignment_has_invalid_multi_target_scalar_unpack(
            node, local_static_types=local_static_types
        ):
            invalid_multi_assign_from_scalar_count += 1
        if isinstance(node, ast.Return) and node.value is not None:
            mutating_method_return_value_count += mutating_method_return_value_calls(
                node.value, local_static_types=local_static_types
            )

    for parent in ast.walk(function):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        for index, stmt in enumerate(body):
            if not isinstance(stmt, ast.Expr):
                continue
            expression_statement_count += 1
            if (
                parent is function
                and index == 0
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ):
                continue
            if expression_is_no_effect(stmt.value):
                no_effect_expression_count += 1
            if expression_is_ignored_pure_call(stmt.value):
                ignored_pure_call_expression_count += 1

    return {
        "literal_call_count": literal_call_count,
        "invalid_receiver_count": invalid_receiver_count,
        "builtin_type_descriptor_receiver_count": builtin_type_descriptor_receiver_count,
        "bare_builtin_condition_count": bare_builtin_condition_count,
        "invalid_known_builtin_arity_count": invalid_known_builtin_arity_count,
        "invalid_known_local_receiver_count": invalid_known_local_receiver_count,
        "invalid_known_local_call_count": invalid_known_local_call_count,
        "invalid_known_local_iter_count": invalid_known_local_iter_count,
        "invalid_multi_assign_from_scalar_count": invalid_multi_assign_from_scalar_count,
        "mutating_method_return_value_count": mutating_method_return_value_count,
        "no_effect_expression_count": no_effect_expression_count,
        "ignored_pure_call_expression_count": ignored_pure_call_expression_count,
        "expression_statement_count": expression_statement_count,
    }


def self_dependent_assignment_load_count(function: ast.FunctionDef, *, initial_bound: set[str]) -> int:
    """Count obvious ``x = x[...]`` style reads before a local is bound.

    This is intentionally shallow and task-blind. It only inspects the candidate
    Python body and callable signature, catching a common learned-generator
    artifact without using tests, solutions, verifier outcomes, or task labels.
    """

    bound = set(initial_bound)
    count = 0
    for statement in function.body:
        if isinstance(statement, ast.Assign):
            targets = assignment_target_names(statement.targets)
            loaded = expression_loaded_names(statement.value)
            count += len((targets & loaded) - bound)
            bound.update(targets)
        elif isinstance(statement, ast.AnnAssign):
            targets = assignment_target_names([statement.target])
            loaded = expression_loaded_names(statement.value)
            count += len((targets & loaded) - bound)
            bound.update(targets)
        elif isinstance(statement, ast.AugAssign):
            targets = assignment_target_names([statement.target])
            loaded = expression_loaded_names(statement.target) | expression_loaded_names(statement.value)
            count += len((targets & loaded) - bound)
            bound.update(targets)
        elif isinstance(statement, (ast.For, ast.AsyncFor)):
            bound.update(assignment_target_names([statement.target]))
        elif isinstance(statement, (ast.Import, ast.ImportFrom)):
            for alias in statement.names:
                bound.add(alias.asname or str(alias.name).split(".", maxsplit=1)[0])
    return int(count)


def assignment_target_names(targets: list[ast.AST]) -> set[str]:
    names: set[str] = set()
    for target in targets:
        for node in ast.walk(target):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                names.add(node.id)
    return names


def expression_loaded_names(expr: ast.AST | None) -> set[str]:
    if expr is None:
        return set()
    return {
        node.id
        for node in ast.walk(expr)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def max_repeated_identical_condition_chain(function: ast.FunctionDef) -> int:
    max_depth = 0

    def visit(statement: ast.stmt, parent_key: str = "", depth: int = 0) -> None:
        nonlocal max_depth
        if isinstance(statement, (ast.If, ast.While)):
            key = ast.dump(statement.test, annotate_fields=False, include_attributes=False)
            next_depth = depth + 1 if key == parent_key else 1
            max_depth = max(max_depth, next_depth)
            for child in list(statement.body) + list(statement.orelse):
                visit(child, key, next_depth)
            return
        for child in ast.iter_child_nodes(statement):
            if isinstance(child, ast.stmt):
                visit(child, "", 0)

    for stmt in function.body:
        visit(stmt)
    return int(max_depth)


def expression_has_invalid_literal_receiver(expr: ast.AST) -> bool:
    """Detect method calls rooted in impossible literal receivers.

    This is task-blind syntax/semantic hygiene. It rejects artifacts such as
    ``True.get(1)`` and ``None.args[0].get(1)`` without using verifier tests,
    solution bodies, task families, return shapes, or any answer metadata.
    String literal methods such as ``"\\n".join(items)`` are intentionally
    allowed because they are common valid Python idioms.
    """

    if isinstance(expr, ast.Attribute):
        value = expr.value
        if isinstance(value, ast.Constant):
            return value.value is None or isinstance(value.value, (bool, int, float, complex, bytes))
        return expression_has_invalid_literal_receiver(value)
    if isinstance(expr, ast.Subscript):
        return expression_has_invalid_literal_receiver(expr.value)
    if isinstance(expr, ast.Call):
        return expression_has_invalid_literal_receiver(expr.func)
    return False


TEMPORARY_CONSTRUCTOR_INVALID_METHODS: dict[str, set[str]] = {
    "str": {"get", "items", "keys", "values", "append", "extend", "add", "setdefault", "__name__"},
    "list": {"get", "items", "keys", "values", "split", "strip", "__name__"},
    "tuple": {"get", "items", "keys", "values", "append", "extend", "add", "__name__"},
    "set": {"get", "items", "keys", "values", "append", "extend", "__name__"},
    "dict": {"append", "extend", "add", "split", "strip", "__name__"},
    "bool": {"get", "items", "keys", "values", "append", "extend", "add", "__name__"},
    "int": {"get", "items", "keys", "values", "append", "extend", "add", "__name__"},
    "float": {"get", "items", "keys", "values", "append", "extend", "add", "__name__"},
}

KNOWN_TYPE_METHODS: dict[str, set[str]] = {
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
        "startswith",
        "strip",
        "upper",
    },
    "tuple": {"count", "index"},
}

NONRETURNING_MUTATING_METHODS: dict[str, set[str]] = {
    "dict": {"clear", "update"},
    "list": {"append", "clear", "extend", "insert", "remove", "reverse", "sort"},
    "set": {"add", "clear", "discard", "remove", "update"},
}

PURE_CALL_METHODS = {
    "casefold",
    "copy",
    "count",
    "endswith",
    "find",
    "format",
    "get",
    "index",
    "isalnum",
    "isalpha",
    "isdigit",
    "items",
    "join",
    "keys",
    "lower",
    "replace",
    "split",
    "startswith",
    "strip",
    "upper",
    "values",
}


def expression_has_builtin_type_descriptor_receiver(attribute: ast.Attribute) -> bool:
    """Detect direct descriptor chains on built-in type objects.

    The direct generator often learns artifacts such as ``bytes.path`` or
    ``bytes.split(data)``. Those are not task-specific reasoning; they are
    brittle class-object descriptor chains. Calling constructors such as
    ``bytes(data)`` or instance methods such as ``data.split()`` stays legal.
    """

    return isinstance(attribute.value, ast.Name) and attribute.value.id in STATIC_COHERENCE_BUILTIN_TYPE_OBJECTS


def expression_has_invalid_method_receiver(attribute: ast.Attribute) -> bool:
    """Detect impossible method/attribute receivers independent of the task.

    This catches model artifacts such as ``str(data).get(...)`` and
    ``list(data).__name__``. It intentionally leaves ``other.get(...)`` legal
    because the prompt/signature alone may not reveal whether an argument is
    mapping-like.
    """

    value = attribute.value
    if isinstance(value, ast.Constant):
        return value.value is None or isinstance(value.value, (bool, int, float, complex, bytes))
    if isinstance(value, ast.Call):
        callee = expression_direct_name(value.func)
        return bool(callee and attribute.attr in TEMPORARY_CONSTRUCTOR_INVALID_METHODS.get(callee, set()))
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)) and attribute.attr in {"get", "items", "keys", "values", "__name__"}:
        return True
    if isinstance(value, ast.Dict) and attribute.attr in {"append", "extend", "add", "split", "strip", "__name__"}:
        return True
    return False


def expression_has_invalid_known_local_receiver(
    attribute: ast.Attribute, *, local_static_types: dict[str, str]
) -> bool:
    value = attribute.value
    if not isinstance(value, ast.Name):
        return False
    local_type = str(local_static_types.get(value.id) or "")
    if not local_type:
        return False
    if attribute.attr in TEMPORARY_CONSTRUCTOR_INVALID_METHODS.get(local_type, set()):
        return True
    known_methods = KNOWN_TYPE_METHODS.get(local_type)
    return bool(known_methods is not None and attribute.attr not in known_methods)


def expression_has_invalid_known_local_call(call: ast.Call, *, local_static_types: dict[str, str]) -> bool:
    """Detect direct calls to locals already bound as obvious non-callables.

    This catches generated artifacts such as ``out(value)`` after ``out = []``
    or ``take(skip)`` after ``take = 0``. The check is task-blind and uses only
    bindings created by the candidate itself.
    """

    if not isinstance(call.func, ast.Name):
        return False
    local_type = str(local_static_types.get(call.func.id) or "")
    return local_type in {"bool", "float", "int", "number", "dict", "list", "set", "str", "tuple"}


def expression_has_invalid_known_local_iter(for_node: ast.For, *, local_static_types: dict[str, str]) -> bool:
    iter_type = expression_static_type(for_node.iter)
    if not iter_type and isinstance(for_node.iter, ast.Name):
        iter_type = str(local_static_types.get(for_node.iter.id) or "")
    return iter_type in {"bool", "float", "int", "number"}


def assignment_has_invalid_multi_target_scalar_unpack(
    assign: ast.Assign, *, local_static_types: dict[str, str]
) -> bool:
    if not any(isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) >= 2 for target in assign.targets):
        return False
    return expression_is_likely_noniterable(assign.value, local_static_types=local_static_types)


def expression_is_likely_noniterable(expr: ast.AST | None, *, local_static_types: dict[str, str]) -> bool:
    expr_type = expression_static_type(expr)
    if expr_type in {"bool", "float", "int", "number"}:
        return True
    if isinstance(expr, ast.Name):
        return str(local_static_types.get(expr.id) or "") in {"bool", "float", "int", "number"}
    if isinstance(expr, ast.BinOp):
        left = expression_is_likely_noniterable(expr.left, local_static_types=local_static_types)
        right = expression_is_likely_noniterable(expr.right, local_static_types=local_static_types)
        return left or right
    if isinstance(expr, ast.UnaryOp):
        return expression_is_likely_noniterable(expr.operand, local_static_types=local_static_types)
    return False


def mutating_method_return_value_calls(expr: ast.AST, *, local_static_types: dict[str, str]) -> int:
    count = 0
    for node in ast.walk(expr):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        receiver = node.func.value
        receiver_type = ""
        if isinstance(receiver, ast.Name):
            receiver_type = str(local_static_types.get(receiver.id) or "")
        elif isinstance(receiver, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            receiver_type = expression_static_type(receiver)
        elif isinstance(receiver, ast.Call):
            receiver_type = expression_direct_name(receiver.func)
        if node.func.attr in NONRETURNING_MUTATING_METHODS.get(receiver_type, set()):
            count += 1
    return count


def expression_has_invalid_known_method_arity(call: ast.Call, *, local_static_types: dict[str, str]) -> bool:
    if not isinstance(call.func, ast.Attribute):
        return False
    method = call.func.attr
    receiver = call.func.value
    receiver_type = ""
    if isinstance(receiver, ast.Name):
        receiver_type = str(local_static_types.get(receiver.id) or "")
    elif isinstance(receiver, (ast.List, ast.Tuple, ast.Set, ast.Dict, ast.Constant)):
        receiver_type = expression_static_type(receiver)
    elif isinstance(receiver, ast.Call):
        receiver_type = expression_direct_name(receiver.func)
    argc = len(call.args)
    if receiver_type in {"list", "set"} and method in {"append", "add", "discard", "extend", "insert", "remove", "update"}:
        if method == "insert":
            return argc != 2
        return argc != 1
    if receiver_type == "list" and method in {"clear", "copy", "pop", "reverse", "sort"}:
        if method == "pop":
            return argc > 1
        return argc != 0
    if receiver_type == "dict" and method == "get":
        return argc < 1 or argc > 2
    if receiver_type == "dict" and method == "setdefault":
        return argc < 1 or argc > 2
    if receiver_type == "dict" and method in {"items", "keys", "values", "clear", "copy"}:
        return argc != 0
    if receiver_type == "dict" and method == "update":
        return argc > 1
    if receiver_type == "str" and method in {"casefold", "isalnum", "isalpha", "isdigit", "lower", "upper"}:
        return argc != 0
    if receiver_type == "str" and method in {"strip", "split"}:
        return argc > 2
    if receiver_type == "str" and method == "join":
        return argc != 1
    return False


def expression_is_ignored_pure_call(expr: ast.AST) -> bool:
    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Attribute):
        return False
    return expr.func.attr in PURE_CALL_METHODS


def expression_direct_name(expr: ast.AST) -> str:
    if isinstance(expr, ast.Name):
        return expr.id
    return ""


def local_static_type_bindings(function: ast.FunctionDef) -> dict[str, str]:
    """Infer obvious local container/scalar bindings from candidate code only.

    This deliberately avoids task metadata and type-shape labels. It only learns
    facts like ``out = []`` or ``counts = dict()`` so the ranker can penalize
    Python-impossible artifacts such as ``out.get(...)`` when ``out`` is visibly
    a list in the candidate itself.
    """

    bindings: dict[str, str] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            inferred = expression_static_type(node.value)
            if not inferred:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = inferred
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            inferred = expression_static_type(node.value) if node.value is not None else ""
            if inferred:
                bindings[node.target.id] = inferred
    return bindings


def expression_static_type(expr: ast.AST | None) -> str:
    if isinstance(expr, ast.List):
        return "list"
    if isinstance(expr, ast.Tuple):
        return "tuple"
    if isinstance(expr, ast.Set):
        return "set"
    if isinstance(expr, ast.Dict):
        return "dict"
    if isinstance(expr, ast.Constant):
        value = expr.value
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, bytes):
            return "bytes"
        if isinstance(value, str):
            return "str"
    if isinstance(expr, ast.Call):
        callee = expression_direct_name(expr.func)
        if callee in STATIC_COHERENCE_BUILTIN_TYPE_OBJECTS and len(expr.args) <= 1 and not expr.keywords:
            return callee
        if callee == "len" and len(expr.args) == 1 and not expr.keywords:
            return "int"
        if callee in {"all", "any", "bool", "isinstance"} and not expr.keywords:
            return "bool"
        if callee in {"abs", "round", "sum", "min", "max"} and not expr.keywords:
            return "number"
    return ""


BARE_BUILTIN_CONDITION_NAMES = {
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


def expression_has_bare_builtin_condition(expr: ast.AST) -> bool:
    if isinstance(expr, ast.Name):
        return expr.id in BARE_BUILTIN_CONDITION_NAMES
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return expression_has_bare_builtin_condition(expr.operand)
    if isinstance(expr, ast.BoolOp):
        return any(expression_has_bare_builtin_condition(value) for value in expr.values)
    if isinstance(expr, ast.Compare):
        return expression_contains_bare_builtin_reference(expr)
    return False


def expression_contains_bare_builtin_reference(expr: ast.AST) -> bool:
    ignored_call_func_ids = {
        id(node.func)
        for node in ast.walk(expr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    return any(
        isinstance(node, ast.Name)
        and id(node) not in ignored_call_func_ids
        and node.id in BARE_BUILTIN_CONDITION_NAMES
        for node in ast.walk(expr)
    )


def expression_is_parameter_free_literal_expression(expr: ast.AST) -> bool:
    """Detect literal/builtin-only computed returns such as ``return -1 / 2``."""

    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and node.id not in STATIC_COHERENCE_ALLOWED_GLOBALS:
            return False
        if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return False
    return not any(
        isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(getattr(node, "value", None), ast.Name)
        for node in ast.walk(expr)
    )


def expression_has_invalid_known_builtin_arity(call: ast.Call) -> bool:
    callee = expression_direct_name(call.func)
    if callee == "isinstance":
        return len(call.args) < 2 or len(call.args) > 2
    if callee in {"abs", "len", "reversed"}:
        return len(call.args) != 1
    if callee in {"filter", "map"}:
        return len(call.args) < 2
    if callee in {"all", "any", "enumerate", "sorted", "sum"}:
        return len(call.args) < 1
    if callee in {"max", "min"}:
        return len(call.args) < 1
    if callee == "range":
        return len(call.args) < 1 or len(call.args) > 3
    if callee == "round":
        return len(call.args) < 1 or len(call.args) > 2
    if callee in {"bool", "bytes", "dict", "float", "int", "list", "set", "str", "tuple"}:
        return len(call.args) > 1
    return False


def expression_has_invalid_known_builtin_arg_type(call: ast.Call) -> bool:
    """Task-blind builtin type hygiene for obvious scalar/iterable mistakes."""

    callee = expression_direct_name(call.func)
    if not callee or not call.args:
        return False
    first_type = expression_static_type(call.args[0])
    if callee == "len":
        return first_type in {"bool", "int", "float", "number"}
    if callee in {"max", "min"}:
        return len(call.args) == 1 and first_type in {"bool", "int", "float", "number"}
    if callee in {"all", "any", "enumerate", "list", "reversed", "set", "sorted", "sum", "tuple"}:
        return first_type in {"bool", "int", "float", "number"}
    return False


def return_value_is_trivial(expr: ast.AST | None) -> bool:
    if expr is None:
        return True
    if isinstance(expr, ast.Constant):
        return expr.value in {None, False, True, 0, 1, "", b""}
    if isinstance(expr, (ast.List, ast.Tuple, ast.Set)) and not expr.elts:
        return True
    if isinstance(expr, ast.Dict) and not expr.keys:
        return True
    return False


def expression_is_parameter_copy_call(expr: ast.AST | None, params: set[str]) -> bool:
    """Detect direct ``param.copy()`` style pass-through calls.

    This is used only for AST-local corpus quality accounting. It does not read
    eval rows, tests, solutions, or answer metadata.
    """

    if not params or not isinstance(expr, ast.Call) or expr.args or expr.keywords:
        return False
    if not isinstance(expr.func, ast.Attribute) or expr.func.attr != "copy":
        return False
    return isinstance(expr.func.value, ast.Name) and expr.func.value.id in params


def expression_is_parameter_identity_copy(expr: ast.AST | None, params: set[str]) -> bool:
    """Detect direct parameter/copy pass-through return expressions."""

    if not params or expr is None:
        return False
    if isinstance(expr, ast.Name):
        return expr.id in params
    if isinstance(expr, ast.Attribute) and expr.attr == "copy":
        return isinstance(expr.value, ast.Name) and expr.value.id in params
    return expression_is_parameter_copy_call(expr, params)


def expression_uses_any_name(expr: ast.AST, names: set[str]) -> bool:
    if not names:
        return False
    return any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in names for node in ast.walk(expr))


def expression_is_static_literal_only(expr: ast.AST) -> bool:
    if isinstance(expr, ast.Constant):
        return True
    if isinstance(expr, (ast.Tuple, ast.List, ast.Set)):
        return all(expression_is_static_literal_only(item) for item in expr.elts)
    if isinstance(expr, ast.Dict):
        return all(
            (key is None or expression_is_static_literal_only(key)) and expression_is_static_literal_only(value)
            for key, value in zip(expr.keys, expr.values)
        )
    return False


def expression_complexity_score(expr: ast.AST) -> int:
    return sum(
        1
        for node in ast.walk(expr)
        if isinstance(
            node,
            (
                ast.BinOp,
                ast.BoolOp,
                ast.UnaryOp,
                ast.Compare,
                ast.Call,
                ast.Subscript,
                ast.IfExp,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
            ),
        )
    )


def expression_is_no_effect(expr: ast.AST) -> bool:
    if isinstance(expr, (ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Set)):
        return True
    if isinstance(expr, ast.Call) and isinstance(expr.func, (ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Set)):
        return True
    if isinstance(expr, (ast.BinOp, ast.BoolOp, ast.UnaryOp, ast.Compare, ast.IfExp, ast.Subscript, ast.Slice)):
        return True
    return False


def function_arg_names(function: ast.FunctionDef) -> list[str]:
    args = list(function.args.posonlyargs) + list(function.args.args) + list(function.args.kwonlyargs)
    names = [arg.arg for arg in args]
    if function.args.vararg is not None:
        names.append(function.args.vararg.arg)
    if function.args.kwarg is not None:
        names.append(function.args.kwarg.arg)
    return names


def static_coherence_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    coherence = dict_or_empty(item.get("static_coherence"))
    proposal = dict_or_empty(item.get("proposal"))
    undefined_count = int(coherence.get("undefined_name_count", 999))
    unexpected_count = int(coherence.get("unexpected_signature_name_count", 999))
    literal_call_count = int(coherence.get("literal_call_count", 999))
    invalid_receiver_count = int(coherence.get("invalid_receiver_count", 999))
    builtin_type_descriptor_receiver_count = int(coherence.get("builtin_type_descriptor_receiver_count", 999))
    bare_builtin_condition_count = int(coherence.get("bare_builtin_condition_count", 999))
    invalid_known_builtin_arity_count = int(coherence.get("invalid_known_builtin_arity_count", 999))
    invalid_known_local_receiver_count = int(coherence.get("invalid_known_local_receiver_count", 999))
    mutating_method_return_value_count = int(coherence.get("mutating_method_return_value_count", 999))
    ignored_pure_call_expression_count = int(coherence.get("ignored_pure_call_expression_count", 999))
    no_effect_expression_count = int(coherence.get("no_effect_expression_count", 999))
    valued_return_count = int(coherence.get("valued_return_count", 0))
    nontrivial_return_count = int(coherence.get("nontrivial_return_count", 0))
    top_level_valued_return_count = int(coherence.get("top_level_valued_return_count", 0))
    nested_return_count = int(coherence.get("nested_return_count", 999))
    bare_return_count = int(coherence.get("bare_return_count", 999))
    used_parameter_count = int(coherence.get("used_parameter_count", 0))
    primary_parameter_load_count = int(coherence.get("primary_parameter_load_count", 0))
    primary_parameter_return_count = int(coherence.get("primary_parameter_return_count", 0))
    return_only_uses_auxiliary_count = int(coherence.get("return_only_uses_auxiliary_parameter_count", 999))
    self_dependent_assignment_count = int(coherence.get("self_dependent_assignment_count", 999))
    repeated_condition_chain = int(coherence.get("max_repeated_identical_condition_chain", 999))
    local_store_name_count = int(coherence.get("local_store_name_count", 0))
    assignment_count = int(coherence.get("assignment_count", 0))
    call_count = int(coherence.get("call_count", 0))
    comprehension_count = int(coherence.get("comprehension_count", 0))
    literal_only_return_count = int(coherence.get("literal_only_return_count", 999))
    parameter_free_literal_return_count = int(coherence.get("parameter_free_literal_expression_return_count", 999))
    inert_stub = bool(coherence.get("inert_stub", True))
    return (
        int(bool(coherence.get("parse_ok")) and bool(coherence.get("has_function"))),
        int(literal_call_count == 0),
        int(invalid_receiver_count == 0),
        int(builtin_type_descriptor_receiver_count == 0),
        int(bare_builtin_condition_count == 0),
        int(invalid_known_builtin_arity_count == 0),
        int(invalid_known_local_receiver_count == 0),
        int(mutating_method_return_value_count == 0),
        int(ignored_pure_call_expression_count == 0),
        int(parameter_free_literal_return_count == 0),
        int(no_effect_expression_count == 0),
        int(self_dependent_assignment_count == 0),
        -self_dependent_assignment_count,
        int(repeated_condition_chain < 4),
        -repeated_condition_chain,
        int(undefined_count == 0),
        int(unexpected_count == 0),
        int(nontrivial_return_count > 0),
        int(valued_return_count > 0),
        int(top_level_valued_return_count > 0),
        int(nested_return_count == 0 or top_level_valued_return_count > 0),
        int(bare_return_count == 0),
        int(used_parameter_count > 0),
        int(primary_parameter_load_count > 0),
        int(primary_parameter_return_count > 0),
        int(return_only_uses_auxiliary_count == 0),
        -return_only_uses_auxiliary_count,
        int(not inert_stub),
        int(local_store_name_count + assignment_count + call_count + comprehension_count > 0),
        int(literal_only_return_count == 0),
        float(coherence.get("score") or -999.0),
        float(proposal.get("rank_score") or -999.0),
        -int(proposal.get("decoded_token_count") or 0),
    )
