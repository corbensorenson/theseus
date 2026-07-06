#!/usr/bin/env python3
"""Task-blind expression value hygiene for strict learned code generation."""

from __future__ import annotations

import ast
import keyword
from typing import Any


EXPRESSION_VALUE_BARE_BUILTINS = {
    "abs",
    "bool",
    "dict",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "range",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
}


def expression_value_quality_summary(body: str, *, allowed_names: set[str] | None) -> dict[str, Any]:
    function = parsed_expression_value_function(body, allowed_names=allowed_names)
    if function is None:
        return {
            "parse_ok": False,
            "invalid_expression_value_count": 0,
            "bare_builtin_value_argument_count": 0,
            "invalid_call_argument_type_count": 0,
            "examples": [],
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    local_types = inferred_ast_local_types(function)
    examples: list[dict[str, Any]] = []
    bare_builtin_value_argument_count = 0
    invalid_call_argument_type_count = 0
    invalid_attribute_receiver_count = 0
    invalid_literal_subscript_count = 0
    invalid_membership_container_count = 0
    for node in ast.walk(function):
        if isinstance(node, ast.Attribute) and expression_is_bare_builtin_name(node.value):
            invalid_attribute_receiver_count += 1
            if len(examples) < 8:
                examples.append(
                    {
                        "label": "invalid_attribute_receiver",
                        "receiver": ast_text(node.value)[:80],
                        "attribute": str(node.attr or ""),
                        "expression": ast_text(node)[:160],
                    }
                )
        if isinstance(node, ast.Subscript) and expression_is_literal_subscript_base(node.value):
            invalid_literal_subscript_count += 1
            if len(examples) < 8:
                examples.append(
                    {
                        "label": "invalid_literal_subscript",
                        "expression": ast_text(node)[:160],
                    }
                )
        if isinstance(node, ast.Compare):
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)) and expression_is_noniterable_literal(comparator):
                    invalid_membership_container_count += 1
                    if len(examples) < 8:
                        examples.append(
                            {
                                "label": "invalid_membership_container",
                                "expression": ast_text(node)[:160],
                            }
                        )
        if not isinstance(node, ast.Call):
            continue
        callee = expression_call_name(node)
        bare_names: set[str] = set()
        for index, arg in enumerate(node.args):
            if callee == "isinstance" and index >= 1:
                continue
            bare_names.update(expression_bare_builtin_value_names(arg))
        for kw in node.keywords:
            bare_names.update(expression_bare_builtin_value_names(kw.value))
        if bare_names:
            bare_builtin_value_argument_count += len(bare_names)
            if len(examples) < 8:
                examples.append(
                    {
                        "label": "bare_builtin_value_argument",
                        "callee": callee,
                        "names": sorted(bare_names),
                        "expression": ast_text(node)[:160],
                    }
                )
        if ast_call_first_arg_type_invalid(node, local_types=local_types):
            invalid_call_argument_type_count += 1
            if len(examples) < 8:
                examples.append(
                    {
                        "label": "invalid_call_argument_type",
                        "callee": callee,
                        "expression": ast_text(node)[:160],
                    }
                )
    invalid_expression_value_count = (
        bare_builtin_value_argument_count
        + invalid_call_argument_type_count
        + invalid_attribute_receiver_count
        + invalid_literal_subscript_count
        + invalid_membership_container_count
    )
    return {
        "parse_ok": True,
        "invalid_expression_value_count": invalid_expression_value_count,
        "bare_builtin_value_argument_count": bare_builtin_value_argument_count,
        "invalid_call_argument_type_count": invalid_call_argument_type_count,
        "invalid_attribute_receiver_count": invalid_attribute_receiver_count,
        "invalid_literal_subscript_count": invalid_literal_subscript_count,
        "invalid_membership_container_count": invalid_membership_container_count,
        "examples": examples,
        "score_semantics": (
            "Task-blind AST value hygiene over generated expressions. It rejects bare builtin/type "
            "objects used as runtime values, obvious call argument type errors, builtin/type objects used "
            "as attribute receivers, literal constant/container subscripts, impossible membership tests "
            "against non-iterable literals, and boolean/comparison expressions used as the object passed "
            "to isinstance. It does not inspect tests, solutions, public benchmark payloads, or hidden "
            "target metadata."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def expression_value_has_bare_builtin_value(expr: ast.AST | None) -> bool:
    return bool(expression_bare_builtin_value_names(expr))


def expression_is_bare_builtin_name(expr: ast.AST | None) -> bool:
    return isinstance(expr, ast.Name) and isinstance(expr.ctx, ast.Load) and expr.id in EXPRESSION_VALUE_BARE_BUILTINS


def expression_is_literal_value(expr: ast.AST | None) -> bool:
    if isinstance(expr, ast.Constant):
        return True
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.operand, ast.Constant):
        return True
    if isinstance(expr, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
        return True
    return False


def expression_is_literal_subscript_base(expr: ast.AST | None) -> bool:
    if expression_is_literal_value(expr):
        return True
    if isinstance(expr, ast.Subscript):
        return expression_is_literal_subscript_base(expr.value)
    return False


def expression_is_noniterable_literal(expr: ast.AST | None) -> bool:
    if isinstance(expr, ast.Constant):
        return not isinstance(expr.value, (str, bytes))
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.operand, ast.Constant):
        return not isinstance(expr.operand.value, (str, bytes))
    return False


def expression_bare_builtin_value_names(expr: ast.AST | None) -> set[str]:
    names: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            for arg in node.args:
                self.visit(arg)
            for kw in node.keywords:
                self.visit(kw.value)

        def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
            if isinstance(node.ctx, ast.Load) and node.id in EXPRESSION_VALUE_BARE_BUILTINS:
                names.add(node.id)

    if expr is not None:
        Visitor().visit(expr)
    return names


def ast_call_first_arg_type_invalid(node: ast.Call, *, local_types: dict[str, str]) -> bool:
    if not isinstance(node.func, ast.Name) or not node.args:
        return False
    callee = node.func.id
    first = node.args[0]
    if expression_value_has_bare_builtin_value(first):
        return True
    if callee == "isinstance" and ast_expr_is_bool_or_comparison(first):
        return True
    inferred = ast_static_type(first, local_types=local_types)
    if callee in {"max", "min"}:
        return len(node.args) == 1 and inferred in {"bool", "float", "int"}
    iterable_builtins = {"all", "any", "enumerate", "len", "list", "reversed", "set", "sorted", "sum", "tuple"}
    if callee in iterable_builtins:
        return inferred in {"bool", "float", "int"}
    if callee in {"abs", "round"}:
        return inferred in {"dict", "list", "set", "str", "tuple"}
    if callee == "range":
        return any(
            ast_static_type(arg, local_types=local_types) in {"dict", "list", "set", "str", "tuple"}
            or expression_value_has_bare_builtin_value(arg)
            for arg in node.args
        )
    return False


def ast_expr_is_bool_or_comparison(expr: ast.AST | None) -> bool:
    """Return true for boolean/test expressions used as runtime values."""

    if expr is None:
        return False
    if isinstance(expr, (ast.BoolOp, ast.Compare)):
        return True
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return True
    if isinstance(expr, ast.Constant) and isinstance(expr.value, bool):
        return True
    return False


def inferred_ast_local_types(function: ast.FunctionDef) -> dict[str, str]:
    local_types: dict[str, str] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            inferred = ast_static_type(node.value, local_types=local_types)
            if not inferred:
                continue
            for target in node.targets:
                for name in expression_store_names(target):
                    local_types[name] = inferred
        elif isinstance(node, ast.AnnAssign):
            inferred = ast_static_type(node.value, local_types=local_types)
            if inferred:
                for name in expression_store_names(node.target):
                    local_types[name] = inferred
    return local_types


def ast_static_type(expr: ast.AST | None, *, local_types: dict[str, str]) -> str:
    if expr is None:
        return ""
    if isinstance(expr, ast.Constant):
        if isinstance(expr.value, bool):
            return "bool"
        if isinstance(expr.value, int):
            return "int"
        if isinstance(expr.value, float):
            return "float"
        if isinstance(expr.value, str):
            return "str"
        return ""
    if isinstance(expr, ast.List):
        return "list"
    if isinstance(expr, ast.Dict):
        return "dict"
    if isinstance(expr, ast.Set):
        return "set"
    if isinstance(expr, ast.Tuple):
        return "tuple"
    if isinstance(expr, ast.Name):
        return local_types.get(expr.id, "")
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
        callee = expr.func.id
        if callee in {"bool", "dict", "float", "int", "list", "set", "str", "tuple"}:
            return callee
        if callee in {"abs", "len", "round", "sum"}:
            return "int"
        if callee in {"max", "min"}:
            return "float" if any(ast_static_type(arg, local_types=local_types) == "float" for arg in expr.args) else "int"
    return ""


def expression_store_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for elt in node.elts:
            names.update(expression_store_names(elt))
        return names
    return set()


def parsed_expression_value_function(body: str, *, allowed_names: set[str] | None) -> ast.FunctionDef | None:
    names = [
        str(name)
        for name in sorted(set(allowed_names or {"data"}))
        if str(name).isidentifier() and not keyword.iskeyword(str(name))
    ] or ["data"]
    params = ", ".join(names)
    lines = str(body or "").splitlines() or ["pass"]
    code = f"def _candidate({params}):\n" + "\n".join(f"    {line}" if line.strip() else "" for line in lines) + "\n"
    try:
        parsed = ast.parse(code)
    except SyntaxError:
        return None
    return next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)


def expression_call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        base = node.func.value
        if isinstance(base, ast.Name):
            return f"{base.id}.{node.func.attr}"
        return node.func.attr
    return ""


def ast_text(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ast.dump(node, annotate_fields=False, include_attributes=False)
