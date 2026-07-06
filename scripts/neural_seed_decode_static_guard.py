#!/usr/bin/env python3
"""Task-blind static decode guards for strict learned code generation."""

from __future__ import annotations

import ast
import keyword
from typing import Any

from neural_seed_token_decoder_comparator import candidate_static_coherence  # noqa: E402
from neural_seed_token_decoder_support import (  # noqa: E402
    body_tokens_for_target_mode,
    decode_body_tokens,
    syntax_complete_body_prefix,
)
from neural_seed_expression_value_guard import expression_value_quality_summary  # noqa: E402


def decode_guard_parameter_names(allowed_names: set[str] | None) -> list[str]:
    names = sorted(
        str(name)
        for name in set(allowed_names or {"data"})
        if str(name).isidentifier() and not keyword.iskeyword(str(name))
    )
    return names or ["data"]


def static_guard_candidate_code(body: str, *, allowed_names: set[str] | None) -> str:
    params = ", ".join(decode_guard_parameter_names(allowed_names))
    lines = str(body or "").splitlines() or ["pass"]
    return f"def _candidate({params}):\n" + "\n".join(f"    {line}" if line.strip() else "" for line in lines) + "\n"


def parsed_decode_guard_function(body: str, *, allowed_names: set[str] | None) -> ast.FunctionDef | None:
    try:
        parsed = ast.parse(static_guard_candidate_code(body, allowed_names=allowed_names))
    except SyntaxError:
        return None
    function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    return function


def expression_complexity_for_decode_guard(expr: ast.AST | None) -> int:
    if expr is None:
        return 0
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


def expression_is_direct_parameter_identity(expr: ast.AST | None, *, params: set[str]) -> bool:
    if expr is None or not params:
        return False
    if isinstance(expr, ast.Name):
        return expr.id in params
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and not expr.args and not expr.keywords:
        return expr.func.attr == "copy" and isinstance(expr.func.value, ast.Name) and expr.func.value.id in params
    return False


def expression_load_names(expr: ast.AST | None) -> set[str]:
    if expr is None:
        return set()
    return {
        node.id
        for node in ast.walk(expr)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def expression_store_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)}


def action_trace_call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            return f"{func.value.id}.{func.attr}"
        return func.attr
    return ""


def return_value_is_none_like(expr: ast.AST | None) -> bool:
    return expr is None or (isinstance(expr, ast.Constant) and expr.value is None)


def ast_text_for_decode_guard(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ast.dump(node, annotate_fields=False, include_attributes=False)


def assigned_name_targets(target: ast.AST | None) -> set[str]:
    if target is None:
        return set()
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in target.elts:
            names.update(assigned_name_targets(item))
        return names
    return set()


def definitely_bound_return_summary(body: str, *, allowed_names: set[str] | None) -> dict[str, Any]:
    """Find top-level returns of locals that may not be assigned on all paths."""

    function = parsed_decode_guard_function(body, allowed_names=allowed_names)
    if function is None:
        return {
            "parse_ok": False,
            "top_level_local_return_count": 0,
            "top_level_return_local_not_definitely_bound_count": 0,
            "not_definitely_bound_return_names": [],
        }
    params = set(decode_guard_parameter_names(allowed_names))
    not_bound: list[str] = []
    top_level_local_returns = 0

    def bind_from_stmt(stmt: ast.stmt, bound: set[str]) -> set[str]:
        updated = set(bound)
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                updated.update(assigned_name_targets(target))
        elif isinstance(stmt, ast.AnnAssign):
            updated.update(assigned_name_targets(stmt.target))
        elif isinstance(stmt, ast.AugAssign):
            # AugAssign reads before writing. It is definitely bound only when
            # the target was already definitely bound.
            names = assigned_name_targets(stmt.target)
            updated.update(name for name in names if name in bound)
        elif isinstance(stmt, ast.If):
            before = set(bound)
            body_bound = bind_from_statements(list(stmt.body), set(bound))
            else_bound = bind_from_statements(list(stmt.orelse), set(bound)) if stmt.orelse else before
            updated = before | ((body_bound - before) & (else_bound - before))
        elif isinstance(stmt, ast.Try):
            before = set(bound)
            body_bound = bind_from_statements(list(stmt.body), set(bound))
            else_bound = bind_from_statements(list(stmt.orelse), set(body_bound)) if stmt.orelse else body_bound
            final_bound = bind_from_statements(list(stmt.finalbody), set(before))
            updated = before | ((else_bound - before) & (final_bound - before))
        return updated

    def bind_from_statements(statements: list[ast.stmt], bound: set[str]) -> set[str]:
        current = set(bound)
        for stmt in statements:
            if isinstance(stmt, ast.Return):
                break
            current = bind_from_stmt(stmt, current)
        return current

    bound_names: set[str] = set()
    for stmt in function.body:
        if isinstance(stmt, ast.Return):
            if isinstance(stmt.value, ast.Name) and stmt.value.id not in params:
                top_level_local_returns += 1
                if stmt.value.id not in bound_names:
                    not_bound.append(stmt.value.id)
            continue
        # Loops and with-statements are not guaranteed to execute, so locals
        # first created inside them are not definitely available afterward.
        if isinstance(stmt, (ast.For, ast.While, ast.With)):
            continue
        bound_names = bind_from_stmt(stmt, bound_names)

    return {
        "parse_ok": True,
        "top_level_local_return_count": top_level_local_returns,
        "top_level_return_local_not_definitely_bound_count": len(not_bound),
        "not_definitely_bound_return_names": sorted(set(not_bound))[:12],
        "definitely_bound_top_level_names": sorted(bound_names)[:24],
    }


def control_flow_pathology_summary(body: str, *, allowed_names: set[str] | None) -> dict[str, Any]:
    """Task-blind structural hygiene for generated bodies.

    This rejects generated bodies that satisfy syntax by walking down repeated
    identical guards, a common failure mode of undertrained token generators.
    """

    function = parsed_decode_guard_function(body, allowed_names=allowed_names)
    if function is None:
        return {
            "parse_ok": False,
            "max_control_flow_depth": 0,
            "max_repeated_identical_condition_chain": 0,
            "repeated_condition_examples": [],
        }
    max_depth = 0
    max_repeated = 0
    examples: list[dict[str, Any]] = []

    def visit_statements(
        statements: list[ast.stmt],
        *,
        depth: int,
        condition_stack: list[tuple[str, int]],
    ) -> None:
        nonlocal max_depth, max_repeated
        for stmt in statements:
            if isinstance(stmt, (ast.If, ast.While)):
                test_text = ast_text_for_decode_guard(stmt.test)
                repeat_count = condition_stack[-1][1] + 1 if condition_stack and condition_stack[-1][0] == test_text else 1
                current_depth = depth + 1
                max_depth = max(max_depth, current_depth)
                max_repeated = max(max_repeated, repeat_count)
                if repeat_count >= 4 and len(examples) < 6:
                    examples.append(
                        {
                            "condition": test_text[:160],
                            "repeat_count": repeat_count,
                            "depth": current_depth,
                        }
                    )
                next_stack = condition_stack + [(test_text, repeat_count)]
                visit_statements(list(stmt.body), depth=current_depth, condition_stack=next_stack)
                visit_statements(list(stmt.orelse), depth=current_depth, condition_stack=next_stack)
            elif isinstance(stmt, ast.For):
                iter_text = f"for:{ast_text_for_decode_guard(stmt.target)} in {ast_text_for_decode_guard(stmt.iter)}"
                current_depth = depth + 1
                max_depth = max(max_depth, current_depth)
                next_stack = condition_stack + [(iter_text, 1)]
                visit_statements(list(stmt.body), depth=current_depth, condition_stack=next_stack)
                visit_statements(list(stmt.orelse), depth=current_depth, condition_stack=next_stack)
            elif isinstance(stmt, ast.Try):
                visit_statements(list(stmt.body), depth=depth + 1, condition_stack=condition_stack)
                for handler in stmt.handlers:
                    visit_statements(list(handler.body), depth=depth + 1, condition_stack=condition_stack)
                visit_statements(list(stmt.orelse), depth=depth + 1, condition_stack=condition_stack)
                visit_statements(list(stmt.finalbody), depth=depth + 1, condition_stack=condition_stack)
            elif isinstance(stmt, ast.With):
                visit_statements(list(stmt.body), depth=depth + 1, condition_stack=condition_stack)

    visit_statements(list(function.body), depth=0, condition_stack=[])
    return {
        "parse_ok": True,
        "max_control_flow_depth": max_depth,
        "max_repeated_identical_condition_chain": max_repeated,
        "repeated_condition_examples": examples,
    }


def decode_guard_return_dependency_summary(body: str, *, allowed_names: set[str] | None) -> dict[str, Any]:
    """Task-blind dataflow summary for final strict-generator decode guards.

    This intentionally sees only the generated body plus visible signature
    names. It rejects placeholder-local returns without rejecting ordinary
    accumulator algorithms whose returned local is mutated from parameter data.
    """

    function = parsed_decode_guard_function(body, allowed_names=allowed_names)
    if function is None:
        return {
            "parse_ok": False,
            "valued_return_count": 0,
            "parameter_dependent_return_count": 0,
            "nontrivial_return_count": 0,
            "top_level_valued_return_count": 0,
            "top_level_nontrivial_return_count": 0,
            "weak_placeholder_local_return_count": 0,
            "parameter_dependent_locals": [],
            "nontrivial_locals": [],
        }
    params = set(decode_guard_parameter_names(allowed_names))
    dependent_locals: set[str] = set()
    nontrivial_locals: set[str] = set()
    weak_placeholder_local_return_count = 0
    valued_return_count = 0
    parameter_dependent_return_count = 0
    nontrivial_return_count = 0
    top_level_valued_return_count = 0
    top_level_nontrivial_return_count = 0

    def expr_depends_on_parameter(expr: ast.AST | None) -> bool:
        names = expression_load_names(expr)
        return bool(names & params or names & dependent_locals)

    def expr_is_nontrivial_parameter_expression(expr: ast.AST | None, *, context_dependent: bool) -> bool:
        if return_value_is_none_like(expr):
            return False
        if isinstance(expr, ast.Name):
            if expr.id in nontrivial_locals:
                return True
            if expr.id in params:
                return bool(context_dependent)
            return False
        if context_dependent and not return_value_is_none_like(expr):
            return True
        if not expr_depends_on_parameter(expr):
            return False
        if expression_is_direct_parameter_identity(expr, params=params):
            return False
        if ast_expr_is_trivial_constant_like(expr, allowed_names=params):
            return False
        return expression_complexity_for_decode_guard(expr) > 0

    def bind_target(target: ast.AST, *, dependent: bool, nontrivial: bool) -> None:
        if not isinstance(target, ast.Name):
            return
        if dependent:
            dependent_locals.add(target.id)
        if nontrivial:
            nontrivial_locals.add(target.id)

    def handle_assignment(targets: list[ast.AST], value: ast.AST | None, *, context_dependent: bool) -> None:
        dependent = bool(context_dependent or expr_depends_on_parameter(value))
        nontrivial = bool(
            dependent
            and not return_value_is_none_like(value)
            and not expression_is_direct_parameter_identity(value, params=params)
            and (
                context_dependent
                or expr_is_nontrivial_parameter_expression(value, context_dependent=context_dependent)
                or expression_complexity_for_decode_guard(value) > 0
            )
        )
        for target in targets:
            bind_target(target, dependent=dependent, nontrivial=nontrivial)

    def mark_mutated_receiver(expr: ast.AST, *, context_dependent: bool) -> None:
        if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Attribute):
            return
        receiver = expr.func.value
        if not isinstance(receiver, ast.Name):
            return
        mutating_methods = {
            "add",
            "append",
            "clear",
            "discard",
            "extend",
            "insert",
            "pop",
            "remove",
            "setdefault",
            "sort",
            "update",
        }
        if expr.func.attr not in mutating_methods:
            return
        dependent = bool(
            context_dependent
            or receiver.id in dependent_locals
            or any(expr_depends_on_parameter(arg) for arg in expr.args)
            or any(expr_depends_on_parameter(keyword.value) for keyword in expr.keywords)
        )
        if dependent:
            dependent_locals.add(receiver.id)
            nontrivial_locals.add(receiver.id)

    def visit_statements(statements: list[ast.stmt], *, context_dependent: bool, top_level: bool) -> None:
        nonlocal valued_return_count
        nonlocal parameter_dependent_return_count
        nonlocal nontrivial_return_count
        nonlocal top_level_valued_return_count
        nonlocal top_level_nontrivial_return_count
        nonlocal weak_placeholder_local_return_count
        for stmt in statements:
            if isinstance(stmt, ast.Assign):
                handle_assignment(list(stmt.targets), stmt.value, context_dependent=context_dependent)
            elif isinstance(stmt, ast.AnnAssign):
                handle_assignment([stmt.target], stmt.value, context_dependent=context_dependent)
            elif isinstance(stmt, ast.AugAssign):
                dependent = bool(context_dependent or expr_depends_on_parameter(stmt.value) or expr_depends_on_parameter(stmt.target))
                bind_target(stmt.target, dependent=dependent, nontrivial=dependent)
            elif isinstance(stmt, ast.Expr):
                mark_mutated_receiver(stmt.value, context_dependent=context_dependent)
            elif isinstance(stmt, ast.For):
                iter_dependent = bool(context_dependent or expr_depends_on_parameter(stmt.iter))
                bind_target(stmt.target, dependent=iter_dependent, nontrivial=iter_dependent)
                visit_statements(list(stmt.body), context_dependent=iter_dependent, top_level=False)
                visit_statements(list(stmt.orelse), context_dependent=context_dependent, top_level=False)
            elif isinstance(stmt, ast.While):
                loop_dependent = bool(context_dependent or expr_depends_on_parameter(stmt.test))
                visit_statements(list(stmt.body), context_dependent=loop_dependent, top_level=False)
                visit_statements(list(stmt.orelse), context_dependent=context_dependent, top_level=False)
            elif isinstance(stmt, ast.If):
                branch_dependent = bool(context_dependent or expr_depends_on_parameter(stmt.test))
                visit_statements(list(stmt.body), context_dependent=branch_dependent, top_level=False)
                visit_statements(list(stmt.orelse), context_dependent=branch_dependent, top_level=False)
            elif isinstance(stmt, ast.Try):
                visit_statements(list(stmt.body), context_dependent=context_dependent, top_level=False)
                for handler in stmt.handlers:
                    visit_statements(list(handler.body), context_dependent=context_dependent, top_level=False)
                visit_statements(list(stmt.orelse), context_dependent=context_dependent, top_level=False)
                visit_statements(list(stmt.finalbody), context_dependent=context_dependent, top_level=False)
            elif isinstance(stmt, ast.With):
                with_dependent = context_dependent or any(expr_depends_on_parameter(item.context_expr) for item in stmt.items)
                visit_statements(list(stmt.body), context_dependent=with_dependent, top_level=False)
            elif isinstance(stmt, ast.Return):
                if stmt.value is None:
                    continue
                valued_return_count += 1
                if top_level:
                    top_level_valued_return_count += 1
                dependent = bool(context_dependent or expr_depends_on_parameter(stmt.value))
                nontrivial = expr_is_nontrivial_parameter_expression(stmt.value, context_dependent=context_dependent)
                if isinstance(stmt.value, ast.Name) and stmt.value.id not in params and stmt.value.id not in nontrivial_locals:
                    weak_placeholder_local_return_count += 1
                if dependent:
                    parameter_dependent_return_count += 1
                if nontrivial:
                    nontrivial_return_count += 1
                    if top_level:
                        top_level_nontrivial_return_count += 1

    visit_statements(list(function.body), context_dependent=False, top_level=True)
    return {
        "parse_ok": True,
        "valued_return_count": valued_return_count,
        "parameter_dependent_return_count": parameter_dependent_return_count,
        "nontrivial_return_count": nontrivial_return_count,
        "top_level_valued_return_count": top_level_valued_return_count,
        "top_level_nontrivial_return_count": top_level_nontrivial_return_count,
        "weak_placeholder_local_return_count": weak_placeholder_local_return_count,
        "parameter_dependent_locals": sorted(dependent_locals)[:12],
        "nontrivial_locals": sorted(nontrivial_locals)[:12],
    }


def constant_isinstance_guard_count(body: str, *, allowed_names: set[str] | None) -> int:
    function = parsed_decode_guard_function(body, allowed_names=allowed_names)
    if function is None:
        return 0
    count = 0
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "isinstance" or not node.args:
            continue
        receiver = node.args[0]
        if isinstance(receiver, ast.Constant):
            count += 1
        elif isinstance(receiver, ast.Name) and receiver.id in {"None", "True", "False"}:
            count += 1
    return count


def decode_static_guard(
    body: str,
    *,
    allowed_names: set[str] | None,
    require_parameter_use: bool,
    require_nontrivial_return: bool,
    require_top_level_return: bool,
) -> dict[str, Any]:
    code = static_guard_candidate_code(body, allowed_names=allowed_names)
    coherence = candidate_static_coherence(code)
    dependency = decode_guard_return_dependency_summary(body, allowed_names=allowed_names)
    definite = definitely_bound_return_summary(body, allowed_names=allowed_names)
    control_flow = control_flow_pathology_summary(body, allowed_names=allowed_names)
    constant_isinstance_count = constant_isinstance_guard_count(body, allowed_names=allowed_names)
    value_quality = expression_value_quality_summary(body, allowed_names=allowed_names)
    failures: list[str] = []
    parse_ok = bool(coherence.get("parse_ok")) and bool(coherence.get("has_function"))
    if not parse_ok:
        failures.append("parse_or_function")
    else:
        for key in [
            "undefined_name_count",
            "unexpected_signature_name_count",
            "literal_call_count",
            "invalid_receiver_count",
            "builtin_type_descriptor_receiver_count",
            "bare_builtin_condition_count",
            "invalid_known_builtin_arity_count",
            "invalid_known_local_receiver_count",
            "invalid_known_local_call_count",
            "invalid_known_local_iter_count",
            "invalid_multi_assign_from_scalar_count",
            "mutating_method_return_value_count",
        ]:
            if int(coherence.get(key, 0) or 0) > 0:
                failures.append(key)
    if require_parameter_use and int(dependency.get("parameter_dependent_return_count", 0) or 0) <= 0:
        failures.append("return_not_parameter_dependent")
    if require_nontrivial_return and int(dependency.get("nontrivial_return_count", 0) or 0) <= 0:
        failures.append("return_not_nontrivial")
    if require_top_level_return and int(dependency.get("top_level_valued_return_count", 0) or 0) <= 0:
        failures.append("missing_top_level_valued_return")
    if (
        require_top_level_return
        and require_nontrivial_return
        and int(dependency.get("top_level_nontrivial_return_count", 0) or 0) <= 0
    ):
        failures.append("top_level_return_not_nontrivial")
    if require_nontrivial_return and int(dependency.get("weak_placeholder_local_return_count", 0) or 0) > 0:
        failures.append("weak_placeholder_local_return")
    if int(definite.get("top_level_return_local_not_definitely_bound_count", 0) or 0) > 0:
        failures.append("top_level_return_local_not_definitely_bound")
    if int(control_flow.get("max_repeated_identical_condition_chain", 0) or 0) >= 4:
        failures.append("repeated_identical_condition_chain")
    if int(control_flow.get("max_control_flow_depth", 0) or 0) > 12:
        failures.append("excessive_control_flow_depth")
    if constant_isinstance_count > 0:
        failures.append("constant_isinstance_receiver")
    if int(value_quality.get("bare_builtin_value_argument_count") or 0) > 0:
        failures.append("bare_builtin_value_argument")
    if int(value_quality.get("invalid_call_argument_type_count") or 0) > 0:
        failures.append("invalid_call_argument_type")
    return {
        "passed": not failures,
        "failures": failures,
        "coherence": {
            "parse_ok": bool(coherence.get("parse_ok")),
            "undefined_name_count": int(coherence.get("undefined_name_count", 999) or 0),
            "invalid_receiver_count": int(coherence.get("invalid_receiver_count", 999) or 0),
            "invalid_known_local_call_count": int(coherence.get("invalid_known_local_call_count", 999) or 0),
            "inert_stub": bool(coherence.get("inert_stub", True)),
            "score": coherence.get("score"),
        },
        "dependency": dependency,
        "definite_assignment": definite,
        "control_flow_pathology": control_flow,
        "constant_isinstance_receiver_count": constant_isinstance_count,
        "expression_value_quality": value_quality,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
    }


def completion_ready(
    prefix: list[str],
    *,
    target_mode: str = "body_tokens",
    allowed_names: set[str] | None,
    require_parameter_use: bool,
    require_nontrivial_return: bool,
    require_top_level_return: bool,
) -> bool:
    body_prefix = body_tokens_for_target_mode(prefix, target_mode=target_mode)
    if not syntax_complete_body_prefix(body_prefix):
        return False
    body = decode_body_tokens(body_prefix)
    guard = decode_static_guard(
        body,
        allowed_names=allowed_names,
        require_parameter_use=require_parameter_use,
        require_nontrivial_return=require_nontrivial_return,
        require_top_level_return=require_top_level_return,
    )
    if not bool(guard.get("passed")):
        return False
    return True


def body_uses_allowed_parameter(body: str, *, allowed_names: set[str] | None) -> bool:
    names = set(allowed_names or {"data"})
    if not names:
        return True
    try:
        tree = ast.parse("def _candidate(data, other=None, *extra):\n" + "\n".join(f"    {line}" for line in body.splitlines()) + "\n")
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in names for node in ast.walk(tree))


def body_has_nontrivial_return(body: str, *, allowed_names: set[str] | None = None) -> bool:
    try:
        tree = ast.parse("def _candidate(data, other=None, *extra):\n" + "\n".join(f"    {line}" for line in body.splitlines()) + "\n")
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        value = node.value
        if ast_expr_is_trivial_constant_like(value, allowed_names=allowed_names):
            continue
        return True
    return False


def ast_expr_is_trivial_constant_like(value: ast.AST, *, allowed_names: set[str] | None = None) -> bool:
    trivial_name_values = {
        "None",
        "True",
        "False",
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
    if isinstance(value, ast.Name):
        return value.id in set(allowed_names or set()) or value.id in trivial_name_values
    if isinstance(value, ast.Constant):
        return value.value in {None, False, True, 0, 1, 0.0, 1.0, "", b""}
    if isinstance(value, ast.UnaryOp):
        return isinstance(value.op, (ast.Not, ast.USub, ast.UAdd)) and ast_expr_is_trivial_constant_like(value.operand, allowed_names=allowed_names)
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        return all(ast_expr_is_trivial_constant_like(item, allowed_names=allowed_names) for item in value.elts)
    if isinstance(value, ast.Dict):
        items = [item for pair in zip(value.keys, value.values) for item in pair if item is not None]
        return all(ast_expr_is_trivial_constant_like(item, allowed_names=allowed_names) for item in items)
    if isinstance(value, ast.Call):
        # Calls such as max(int, int) or tuple(other) are syntactically valid
        # but use builtin objects as data. They repeatedly pass the old
        # "nontrivial" check while carrying no task semantics.
        call_func_ids = {id(child.func) for child in ast.walk(value) if isinstance(child, ast.Call)}
        return any(
            isinstance(child, ast.Name)
            and not isinstance(child.ctx, ast.Store)
            and id(child) not in call_func_ids
            and child.id in trivial_name_values
            for child in ast.walk(value)
        )
    return False


def body_has_top_level_valued_return(body: str) -> bool:
    try:
        tree = ast.parse("def _candidate(data, other=None, *extra):\n" + "\n".join(f"    {line}" for line in body.splitlines()) + "\n")
    except SyntaxError:
        return False
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
    if function is None:
        return False
    return any(isinstance(node, ast.Return) and node.value is not None for node in function.body)
