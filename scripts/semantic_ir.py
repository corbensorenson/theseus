#!/usr/bin/env python3
"""Canonical typed semantic IR for Theseus Python program bodies.

The IR is deliberately generic. It serializes Python AST nodes and fields rather
than selecting task-family templates, and it round-trips through an explicit
parser that fails with typed faults. Deterministic compilation is an assisted
tool surface and never receives learned-generation credit.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import textwrap
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable


POLICY = "project_theseus_typed_semantic_ir_v1"
TARGET_MODE = "typed_semantic_ir_tokens_v1"
PLAN_BODY_TARGET_MODE = "typed_semantic_ir_plan_body_tokens_v1"
PLAN_BEGIN = "IRP:BEGIN"
PLAN_END = "IRP:END"
PLAN_PREFIX = "IRP:"
PLAN_MAX_TOKENS = 48
PLAN_MAX_STEPS = 8
PLAN_DEPTH_BUCKETS = tuple(range(8))
PLAN_COUNT_BUCKETS = ("0", "1", "2", "M")
PLAN_STATEMENT_KINDS = (
    "assert",
    "bind",
    "branch",
    "break",
    "continue",
    "dependency",
    "effect",
    "failure_boundary",
    "iterate",
    "iterate_condition",
    "nested_definition",
    "pass",
    "raise",
    "resource_scope",
    "return",
    "statement",
    "update",
)
PLAN_SEMANTIC_INTENTS = (
    "binding_reference",
    "branch_constraint",
    "collection_construction",
    "control_finalizer",
    "effect_expression",
    "literal_value",
    "return_closure",
    "semantic_operation",
    "state_update",
    "traversal",
    "value_expression",
    "verification_or_failure_boundary",
)
PLAN_FEATURES = (
    "call_convert",
    "call_inspect",
    "call_iterate",
    "call_mapping",
    "call_mutate",
    "call_numeric",
    "call_order",
    "call_other",
    "call_text",
    "comprehension",
    "container_literal",
    "op_arithmetic",
    "op_boolean",
    "op_compare",
    "op_identity",
    "op_membership",
    "op_other",
)
PLAN_DATA_ROLES = (
    "NONE",
    "ARG0",
    "ARG1",
    "ARG2",
    "ARG3",
    "ARGN",
    "LOCAL0",
    "LOCAL1",
    "LOCAL2",
    "LOCAL3",
    "LOCALN",
    "MULTI",
)
PLAN_VALUE_KINDS = (
    "none",
    "name",
    "number",
    "text",
    "boolean",
    "container",
    "call",
    "binary_operation",
    "boolean_operation",
    "comparison",
    "subscript",
    "attribute",
    "conditional",
    "comprehension",
    "unary_operation",
    "other",
)
PROGRAM_BEGIN = "IR:PROGRAM"
PROGRAM_END = "IR:PROGRAM_END"  # Reserved so older fault fixtures fail explicitly.
NODE_END = "IR:NODE_END"  # Reserved; compact v1 uses AST field arity.
LIST_BEGIN = "IR:LIST"
LIST_END = "IR:LIST_END"  # Reserved; compact v1 uses counted lists.
SCALAR_END = "IR:SCALAR_END"  # Reserved; compact v1 uses byte lengths.
OMITTED_AST_FIELDS = {"ctx", "kind", "type_comment"}
NO_CHEAT = {
    "candidate_generation_credit": 0,
    "deterministic_compiler_credit": 0,
    "public_training_rows_written": 0,
    "external_inference_calls": 0,
    "fallback_return_count": 0,
}


class SemanticIRFault(ValueError):
    """A typed, fail-closed semantic-IR fault."""

    def __init__(self, code: str, detail: str, *, token_index: int = -1) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.token_index = token_index

    def record(self) -> dict[str, Any]:
        return {
            "fault_type": self.code,
            "detail": self.detail,
            "token_index": self.token_index,
            "failure_behavior": "reject_without_fallback",
        }


@dataclass(frozen=True)
class SemanticIRProgram:
    body: tuple[ast.stmt, ...]
    tokens: tuple[str, ...]


def semantic_ir_target_mode(value: str) -> bool:
    return str(value or "") == TARGET_MODE


def semantic_ir_plan_body_target_mode(value: str) -> bool:
    return str(value or "") == PLAN_BODY_TARGET_MODE


@lru_cache(maxsize=1)
def plan_protocol_tokens() -> tuple[str, ...]:
    """Return the closed, target-independent vocabulary for compact plans."""

    values = [PLAN_BEGIN, PLAN_END]
    values.extend(
        f"IRP:STEP:D{depth}:{kind}"
        for depth in PLAN_DEPTH_BUCKETS
        for kind in PLAN_STATEMENT_KINDS
    )
    values.extend(f"IRP:SEM:{intent}" for intent in PLAN_SEMANTIC_INTENTS)
    values.extend(
        f"IRP:FLOW:R{read_count}:W{write_count}"
        for read_count in PLAN_COUNT_BUCKETS
        for write_count in PLAN_COUNT_BUCKETS
    )
    values.extend(
        f"IRP:DATA:R{read_role}:W{write_role}"
        for read_role in PLAN_DATA_ROLES
        for write_role in PLAN_DATA_ROLES
    )
    values.extend(f"IRP:VALUE:{kind}" for kind in PLAN_VALUE_KINDS)
    values.extend(f"IRP:FEATURE:{feature}" for feature in PLAN_FEATURES)
    return tuple(values)


@lru_cache(maxsize=1)
def plan_protocol_token_set() -> frozenset[str]:
    return frozenset(plan_protocol_tokens())


@lru_cache(maxsize=1)
def plan_obligation_features() -> tuple[str, ...]:
    """Return the fixed multi-label feature space for learned planning.

    Begin/end framing is excluded because it carries no semantic information.
    The remaining closed protocol contains only generic AST operation, intent,
    flow, and feature atoms; it cannot reconstruct identifiers or code.
    """

    return tuple(
        token for token in plan_protocol_tokens() if token not in {PLAN_BEGIN, PLAN_END}
    )


def body_to_plan_obligation_labels(body: str) -> tuple[int, ...]:
    """Compile an admitted training body into a fixed multi-hot plan target."""

    active = set(body_to_plan_tokens(body))
    return tuple(int(feature in active) for feature in plan_obligation_features())


def plan_prefix_token_allowed(prefix: Iterable[str], token: str, *, body_start_token: str) -> bool:
    """Validate one learned plan-prefix transition without reading a target body."""

    stream = [str(value) for value in prefix]
    value = str(token)
    if body_start_token in stream:
        return False
    if not stream:
        return value == PLAN_BEGIN
    if stream[0] != PLAN_BEGIN or len(stream) >= PLAN_MAX_TOKENS + 1:
        return False
    previous = stream[-1]
    if previous == PLAN_END:
        return value == body_start_token
    if value in {PLAN_BEGIN, body_start_token} or value not in plan_protocol_token_set():
        return False
    if previous == PLAN_BEGIN:
        return value.startswith("IRP:STEP:")
    if previous.startswith("IRP:FEATURE:"):
        return value == PLAN_END or value.startswith("IRP:STEP:")
    if previous.startswith("IRP:STEP:"):
        return value.startswith("IRP:SEM:")
    if previous.startswith("IRP:SEM:"):
        return value.startswith("IRP:FLOW:")
    if previous.startswith("IRP:FLOW:"):
        return value.startswith("IRP:DATA:")
    if previous.startswith("IRP:DATA:"):
        return value.startswith("IRP:VALUE:")
    if previous.startswith("IRP:VALUE:"):
        return value == PLAN_END or value.startswith(("IRP:FEATURE:", "IRP:STEP:"))
    return False


def body_to_plan_tokens(body: str, *, max_tokens: int = PLAN_MAX_TOKENS) -> list[str]:
    """Encode an ordered generic semantic plan that cannot reconstruct code.

    This is an autoregressive scratchpad target for the existing strict
    generator. It records AST-local operations and state flow in execution
    order, but omits task-family labels, tests, solutions, and renderer actions.
    The model must still emit the complete Python body after this plan.
    """

    function = parse_body(body)
    budget = max(4, int(max_tokens or PLAN_MAX_TOKENS))
    name_roles = _plan_name_roles(function.body)
    content: list[str] = []
    for step in _plan_statement_groups(function.body, name_roles=name_roles):
        if len(content) + len(step) > budget - 2:
            break
        content.extend(step)
        if sum(token.startswith("IRP:STEP:") for token in content) >= PLAN_MAX_STEPS:
            break
    return [PLAN_BEGIN, *content, PLAN_END]


def _plan_statement_groups(
    statements: Iterable[ast.stmt],
    *,
    name_roles: dict[str, str],
    depth: int = 0,
) -> Iterable[list[str]]:
    """Yield compact, complete plan steps in execution order.

    The former plan used five to ten tokens per statement and routinely spent
    the entire target prefix describing a few AST nodes. These steps use a
    closed, generic vocabulary and never include identifiers or exact call
    names, so the model gets a short planning bottleneck without a renderer or
    task-family catalog.
    """

    for statement in statements:
        step = [
            f"IRP:STEP:D{min(max(0, int(depth)), 7)}:{_plan_statement_kind(statement)}",
            f"IRP:SEM:{semantic_intent(statement)}",
            _plan_flow_token(statement),
            _plan_data_token(statement, name_roles),
            f"IRP:VALUE:{_plan_value_kind(_plan_value_node(statement))}",
        ]
        feature = _plan_primary_feature(statement)
        if feature:
            step.append(f"IRP:FEATURE:{feature}")
        yield step
        yield from _plan_statement_groups(
            _nested_statements(statement), name_roles=name_roles, depth=depth + 1
        )


def dropout_plan_tokens(tokens: Iterable[str]) -> list[str]:
    """Remove semantic information while preserving plan framing and token mass."""

    dropped: list[str] = []
    for raw in tokens:
        token = str(raw)
        if token in {PLAN_BEGIN, PLAN_END}:
            dropped.append(token)
        elif token.startswith("IRP:STEP:"):
            dropped.append("IRP:STEP:D0:statement")
        elif token.startswith("IRP:SEM:"):
            dropped.append("IRP:SEM:semantic_operation")
        elif token.startswith("IRP:FLOW:"):
            dropped.append("IRP:FLOW:R0:W0")
        elif token.startswith("IRP:DATA:"):
            dropped.append("IRP:DATA:RNONE:WNONE")
        elif token.startswith("IRP:VALUE:"):
            dropped.append("IRP:VALUE:other")
        elif token.startswith("IRP:FEATURE:"):
            dropped.append("IRP:FEATURE:call_other")
        else:
            raise SemanticIRFault("IR_PLAN_DROPOUT_UNKNOWN_TOKEN", token)
    return dropped


def _plan_name_roles(statements: Iterable[ast.stmt]) -> dict[str, str]:
    stored: list[str] = []
    loaded: list[str] = []
    for statement in statements:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Name):
                continue
            target = stored if isinstance(node.ctx, ast.Store) else loaded
            if node.id not in target:
                target.append(node.id)
    local_set = set(stored)
    external = [
        name
        for name in loaded
        if name not in local_set and not hasattr(builtins, name)
    ]
    roles: dict[str, str] = {}
    for index, name in enumerate(external):
        roles[name] = f"ARG{index}" if index < 4 else "ARGN"
    for index, name in enumerate(stored):
        roles[name] = f"LOCAL{index}" if index < 4 else "LOCALN"
    return roles


def _plan_data_token(statement: ast.stmt, name_roles: dict[str, str]) -> str:
    reads: list[str] = []
    writes: list[str] = []
    for node in _walk_direct_statement_scope(statement):
        if not isinstance(node, ast.Name):
            continue
        role = name_roles.get(node.id)
        if role is None:
            continue
        target = writes if isinstance(node.ctx, ast.Store) else reads
        if role not in target:
            target.append(role)
    return f"IRP:DATA:R{_plan_role_summary(reads)}:W{_plan_role_summary(writes)}"


def _plan_role_summary(roles: list[str]) -> str:
    if not roles:
        return "NONE"
    return roles[0] if len(roles) == 1 else "MULTI"


def _plan_value_node(statement: ast.stmt) -> ast.AST | None:
    if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        return statement.value
    if isinstance(statement, (ast.For, ast.AsyncFor)):
        return statement.iter
    if isinstance(statement, (ast.If, ast.While, ast.Assert)):
        return statement.test
    if isinstance(statement, (ast.Return, ast.Raise, ast.Expr)):
        return getattr(statement, "value", None) or getattr(statement, "exc", None)
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        return statement.items[0].context_expr if statement.items else None
    return None


def _plan_value_kind(node: ast.AST | None) -> str:
    if node is None:
        return "none"
    if isinstance(node, ast.Name):
        return "name"
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "boolean"
        if isinstance(node.value, (int, float, complex)):
            return "number"
        if isinstance(node.value, (str, bytes)):
            return "text"
        return "other"
    if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
        return "container"
    if isinstance(node, ast.Call):
        return "call"
    if isinstance(node, ast.BinOp):
        return "binary_operation"
    if isinstance(node, ast.BoolOp):
        return "boolean_operation"
    if isinstance(node, ast.Compare):
        return "comparison"
    if isinstance(node, ast.Subscript):
        return "subscript"
    if isinstance(node, ast.Attribute):
        return "attribute"
    if isinstance(node, ast.IfExp):
        return "conditional"
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return "comprehension"
    if isinstance(node, ast.UnaryOp):
        return "unary_operation"
    return "other"


def _plan_statement_kind(statement: ast.stmt) -> str:
    if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
        return "bind"
    if isinstance(statement, ast.AugAssign):
        return "update"
    if isinstance(statement, (ast.For, ast.AsyncFor)):
        return "iterate"
    if isinstance(statement, ast.While):
        return "iterate_condition"
    if isinstance(statement, ast.If):
        return "branch"
    if isinstance(statement, ast.Return):
        return "return"
    if isinstance(statement, ast.Expr):
        return "effect"
    if isinstance(statement, (ast.Try, getattr(ast, "TryStar", ast.Try))):
        return "failure_boundary"
    if isinstance(statement, ast.Raise):
        return "raise"
    if isinstance(statement, ast.Assert):
        return "assert"
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        return "resource_scope"
    if isinstance(statement, (ast.Break, ast.Continue, ast.Pass)):
        return type(statement).__name__.lower()
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        return "dependency"
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return "nested_definition"
    match_type = getattr(ast, "Match", ())
    if isinstance(match_type, type) and isinstance(statement, match_type):
        return "branch"
    return "statement"


def _plan_flow_token(statement: ast.stmt) -> str:
    return f"IRP:FLOW:R{_plan_count_bucket(_loaded_names(statement))}:W{_plan_count_bucket(_stored_names(statement))}"


def _plan_count_bucket(values: set[str]) -> str:
    count = len(values)
    return str(count) if count < 3 else "M"


def _plan_primary_feature(node: ast.AST) -> str:
    calls: list[str] = []
    operators: list[ast.AST] = []
    containers: list[ast.AST] = []
    for child in _walk_direct_statement_scope(node):
        if isinstance(child, ast.Call):
            calls.append(call_name(child.func).rsplit(".", 1)[-1])
        elif isinstance(child, (ast.operator, ast.boolop, ast.unaryop, ast.cmpop)):
            operators.append(child)
        elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            containers.append(child)
        elif isinstance(child, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            containers.append(child)
    if calls:
        return _plan_call_family(calls[0])
    if operators:
        return _plan_operator_family(operators[0])
    if containers:
        return "comprehension" if isinstance(
            containers[0], (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ) else "container_literal"
    return ""


def _plan_call_family(name: str) -> str:
    value = str(name or "").lower()
    groups = (
        ({"all", "any", "isinstance", "len"}, "inspect"),
        ({"sorted", "reversed"}, "order"),
        ({"abs", "max", "min", "pow", "round", "sum"}, "numeric"),
        ({"enumerate", "filter", "map", "range", "zip"}, "iterate"),
        ({"bool", "bytes", "dict", "float", "int", "list", "set", "str", "tuple"}, "convert"),
        ({"add", "append", "extend", "insert", "setdefault", "update"}, "mutate"),
        ({"get", "items", "keys", "pop", "values"}, "mapping"),
        ({"endswith", "join", "lower", "replace", "split", "startswith", "strip", "upper"}, "text"),
    )
    for members, family in groups:
        if value in members:
            return f"call_{family}"
    return "call_other"


def _plan_operator_family(operator: ast.AST) -> str:
    if isinstance(operator, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.MatMult)):
        return "op_arithmetic"
    if isinstance(operator, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
        return "op_compare"
    if isinstance(operator, (ast.And, ast.Or, ast.Not)):
        return "op_boolean"
    if isinstance(operator, (ast.In, ast.NotIn)):
        return "op_membership"
    if isinstance(operator, (ast.Is, ast.IsNot)):
        return "op_identity"
    return "op_other"


def _plan_expression_features(node: ast.AST) -> list[str]:
    features: list[str] = []
    calls: list[str] = []
    operators: list[str] = []
    containers: list[str] = []
    for child in _walk_direct_statement_scope(node):
        if isinstance(child, ast.Call):
            name = call_name(child.func).rsplit(".", 1)[-1]
            if name:
                calls.append(_plan_symbol(name))
        elif isinstance(child, (ast.operator, ast.boolop, ast.unaryop, ast.cmpop)):
            operators.append(type(child).__name__)
        elif isinstance(child, (ast.List, ast.Tuple, ast.Set, ast.Dict, ast.ListComp, ast.SetComp, ast.DictComp)):
            containers.append(type(child).__name__)
    for name in _stable_unique(calls)[:3]:
        features.append(f"IRP:CALL:{name}")
    for name in _stable_unique(operators)[:3]:
        features.append(f"IRP:OP:{name}")
    for name in _stable_unique(containers)[:2]:
        features.append(f"IRP:CONTAINER:{name}")
    if isinstance(node, ast.Return):
        features.append(f"IRP:RETURN:{infer_node_type(node.value, {})}")
    if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        features.append("IRP:CONTROL:LOOP")
    elif isinstance(node, ast.If):
        features.append("IRP:CONTROL:BRANCH")
    elif isinstance(node, ast.Try):
        features.append("IRP:CONTROL:FAILURE_BOUNDARY")
    return features


def _nested_statements(node: ast.AST) -> list[ast.stmt]:
    nested: list[ast.stmt] = []
    for field in ("body", "orelse", "finalbody"):
        values = getattr(node, field, None)
        if isinstance(values, list):
            nested.extend(value for value in values if isinstance(value, ast.stmt))
    handlers = getattr(node, "handlers", None)
    if isinstance(handlers, list):
        for handler in handlers:
            nested.extend(value for value in getattr(handler, "body", []) if isinstance(value, ast.stmt))
    return nested


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        item.id
        for item in _walk_direct_statement_scope(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
    }


def _stored_names(node: ast.AST) -> set[str]:
    return {
        item.id
        for item in _walk_direct_statement_scope(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
    }


def _walk_direct_statement_scope(root: ast.AST) -> Iterable[ast.AST]:
    """Walk one statement's expressions without folding nested statements in."""

    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        children = list(ast.iter_child_nodes(node))
        for child in reversed(children):
            if child is not root and isinstance(child, ast.stmt):
                continue
            stack.append(child)


def _plan_count_token(kind: str, values: set[str]) -> str:
    count = len(values)
    bucket = str(count) if count < 4 else "4PLUS"
    return f"IRP:{kind}:{bucket}"


def _plan_symbol(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in str(value or ""))
    return cleaned[:48] or "UNKNOWN"


def _stable_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def body_to_tokens(body: str) -> list[str]:
    function = parse_body(body)
    return statements_to_tokens(function.body)


def code_to_tokens(code: str) -> list[str]:
    function = parse_single_function(code)
    return statements_to_tokens(function.body)


def statements_to_tokens(statements: Iterable[ast.stmt]) -> list[str]:
    body = list(statements)
    tokens = [f"{PROGRAM_BEGIN}:{len(body)}"]
    for stmt in body:
        tokens.extend(encode_value(stmt))
    return tokens


def tokens_to_program(tokens: Iterable[str]) -> SemanticIRProgram:
    stream = [str(token) for token in tokens if str(token) not in {"<pad>", "<bos>"}]
    if "<eos>" in stream:
        stream = stream[: stream.index("<eos>")]
    parser = _Parser(stream)
    header = parser.pop()
    if not header.startswith(f"{PROGRAM_BEGIN}:"):
        raise parser.fault("IR_TOKEN_EXPECTED", f"expected counted program header, received {header!r}")
    try:
        statement_count = int(header.rsplit(":", 1)[1])
    except ValueError as exc:
        raise parser.fault("IR_PROGRAM_COUNT_INVALID", header) from exc
    if statement_count <= 0:
        raise parser.fault("IR_EMPTY_PROGRAM", "program body has no statements")
    body: list[ast.stmt] = []
    for _ in range(statement_count):
        if not parser.peek():
            raise parser.fault("IR_TRUNCATED_PROGRAM", "fewer statements than declared")
        node = parser.parse_value()
        if not isinstance(node, ast.stmt):
            raise parser.fault("IR_TOP_LEVEL_NOT_STATEMENT", type(node).__name__)
        body.append(node)
    if parser.peek():
        raise parser.fault("IR_TRAILING_TOKENS", parser.peek())
    normalize_contexts(body)
    wrapper = ast.Module(
        body=[
            ast.FunctionDef(
                name="_theseus_semantic_ir",
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg="data"), ast.arg(arg="other")],
                    vararg=ast.arg(arg="extra"),
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[ast.Constant(value=None)],
                ),
                body=body,
                decorator_list=[],
                returns=None,
                type_comment=None,
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(wrapper)
    try:
        compile(wrapper, "<theseus-semantic-ir>", "exec")
    except (SyntaxError, TypeError, ValueError) as exc:
        raise SemanticIRFault("IR_COMPILE_REJECTED", f"{type(exc).__name__}: {exc}") from exc
    return SemanticIRProgram(body=tuple(body), tokens=tuple(stream))


def compile_body_tokens(tokens: Iterable[str]) -> tuple[str, dict[str, Any]]:
    stream = [str(token) for token in tokens]
    try:
        program = tokens_to_program(stream)
        body = "\n".join(ast.unparse(stmt) for stmt in program.body).strip()
        reparsed = parse_body(body)
        roundtrip = statements_to_tokens(reparsed.body)
        canonical_equal = canonical_ast(program.body) == canonical_ast(tuple(reparsed.body))
        if not canonical_equal:
            raise SemanticIRFault("IR_ROUNDTRIP_MISMATCH", "compiled body changed canonical AST")
        return body, {
            "policy": POLICY,
            "state": "READY",
            "typed_faults": [],
            "token_count": len(program.tokens),
            "token_sha256": stable_hash(list(program.tokens)),
            "program_sha256": stable_hash(canonical_ast(program.body)),
            "roundtrip_token_sha256": stable_hash(roundtrip),
            "roundtrip_ast_equal": True,
            "rendered_from_typed_semantic_ir": True,
            "failure_behavior": "typed_fault_no_fallback",
            **NO_CHEAT,
            "non_claims": [
                "deterministic semantic-IR compilation is not learned generation",
                "syntax reconstruction does not establish intended behavior",
            ],
        }
    except SemanticIRFault as exc:
        return "", {
            "policy": POLICY,
            "state": "FAULT",
            "typed_faults": [exc.record()],
            "token_count": len(stream),
            "token_sha256": stable_hash(stream),
            "program_sha256": "",
            "roundtrip_ast_equal": False,
            "rendered_from_typed_semantic_ir": False,
            "failure_behavior": "typed_fault_no_fallback",
            **NO_CHEAT,
            "non_claims": ["no body was emitted after semantic-IR rejection"],
        }


def candidate_receipt(
    code: str,
    *,
    prompt: str = "",
    callable_signature: str = "",
    learned_prefix_tokens: Iterable[str] = (),
    vcm_context_ref: str = "",
    residual_lineage: Iterable[str] = (),
    include_graph: bool = False,
) -> dict[str, Any]:
    try:
        function = parse_single_function(code)
        tokens = statements_to_tokens(function.body)
        graph = build_program_graph(
            function,
            prompt=prompt,
            callable_signature=callable_signature,
            learned_prefix_tokens=learned_prefix_tokens,
            vcm_context_ref=vcm_context_ref,
            residual_lineage=residual_lineage,
        )
        compiled, compile_receipt = compile_body_tokens(tokens)
        source_body = "\n".join(ast.unparse(stmt) for stmt in function.body).strip()
        canonical_equal = canonical_ast(tuple(function.body)) == canonical_body(source_body)
        faults = list(compile_receipt.get("typed_faults") or [])
        if not canonical_equal:
            faults.append(
                {
                    "fault_type": "IR_SOURCE_ROUNDTRIP_MISMATCH",
                    "detail": "source body changed canonical AST",
                    "token_index": -1,
                    "failure_behavior": "reject_without_fallback",
                }
            )
        receipt = {
            "policy": POLICY,
            "state": "READY" if not faults else "FAULT",
            "entry_point": function.name,
            "program_sha256": stable_hash(canonical_function(function)),
            "body_ast_sha256": stable_hash(canonical_ast(tuple(function.body))),
            "actual_signature_sha256": stable_hash(canonical_signature(function)),
            "token_sha256": stable_hash(tokens),
            "token_count": len(tokens),
            "token_scope": "function_body_only_signature_bound_separately",
            "atom_count": len(graph["atoms"]),
            "dependency_edge_count": len(graph["dependency_edges"]),
            "operation_counts": graph["operation_counts"],
            "input_bindings": graph["input_bindings"],
            "output_types": graph["output_types"],
            "open_obligation_count": len(graph["open_obligations"]),
            "open_obligation_types": sorted({row["obligation_type"] for row in graph["open_obligations"]}),
            "typed_faults": faults,
            "roundtrip_ast_equal": canonical_equal and bool(compile_receipt.get("roundtrip_ast_equal")),
            "compiled_body_sha256": stable_hash(compiled) if compiled else "",
            "learned_prefix_sha256": stable_hash(list(learned_prefix_tokens)),
            "vcm_context_ref": vcm_context_ref,
            "residual_lineage": sorted({str(item) for item in residual_lineage if str(item)}),
            "generation_boundary": {
                "allowed_inputs": ["prompt", "callable_signature", "model_generated_candidate", "model_generated_prefix"],
                "prompt_sha256": stable_hash(prompt),
                "callable_signature_sha256": stable_hash(callable_signature),
                "uses_eval_tests_or_solutions": False,
                "uses_answer_metadata": False,
                "uses_public_data": False,
            },
            "failure_behavior": "typed_fault_no_fallback",
            **NO_CHEAT,
            "non_claims": [
                "candidate receipt does not grant learned-generation credit",
                "semantic-IR loadability does not establish intended behavior",
            ],
        }
        if include_graph:
            receipt["program_graph"] = graph
        return receipt
    except (SemanticIRFault, SyntaxError) as exc:
        fault = exc.record() if isinstance(exc, SemanticIRFault) else {
            "fault_type": "IR_SOURCE_SYNTAX_ERROR",
            "detail": f"{type(exc).__name__}: {exc}",
            "token_index": -1,
            "failure_behavior": "reject_without_fallback",
        }
        return {
            "policy": POLICY,
            "state": "FAULT",
            "entry_point": "",
            "program_sha256": "",
            "body_ast_sha256": "",
            "actual_signature_sha256": "",
            "token_sha256": "",
            "token_count": 0,
            "token_scope": "function_body_only_signature_bound_separately",
            "atom_count": 0,
            "dependency_edge_count": 0,
            "operation_counts": {},
            "input_bindings": [],
            "output_types": [],
            "open_obligation_count": 1,
            "open_obligation_types": [fault["fault_type"]],
            "typed_faults": [fault],
            "roundtrip_ast_equal": False,
            "compiled_body_sha256": "",
            "learned_prefix_sha256": stable_hash(list(learned_prefix_tokens)),
            "vcm_context_ref": vcm_context_ref,
            "residual_lineage": sorted({str(item) for item in residual_lineage if str(item)}),
            "generation_boundary": {
                "allowed_inputs": ["prompt", "callable_signature", "model_generated_candidate", "model_generated_prefix"],
                "prompt_sha256": stable_hash(prompt),
                "callable_signature_sha256": stable_hash(callable_signature),
                "uses_eval_tests_or_solutions": False,
                "uses_answer_metadata": False,
                "uses_public_data": False,
            },
            "failure_behavior": "typed_fault_no_fallback",
            **NO_CHEAT,
            "non_claims": ["no semantic program was admitted"],
        }


def build_program_graph(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    prompt: str,
    callable_signature: str,
    learned_prefix_tokens: Iterable[str],
    vcm_context_ref: str,
    residual_lineage: Iterable[str],
) -> dict[str, Any]:
    atoms: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    producers: dict[str, str] = {}
    env: dict[str, str] = {arg.arg: "unknown" for arg in function.args.args}
    env.update({arg.arg: "unknown" for arg in function.args.kwonlyargs})
    if function.args.vararg:
        env[function.args.vararg.arg] = "tuple"
    if function.args.kwarg:
        env[function.args.kwarg.arg] = "dict"
    residual_refs = sorted({str(item) for item in residual_lineage if str(item)})

    def visit(node: ast.AST, parent: str = "") -> str:
        atom_id = f"atom-{len(atoms):05d}-{stable_hash(ast.dump(node, include_attributes=False))[:12]}"
        loaded = sorted({item.id for item in ast.walk(node) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)})
        stored = sorted({item.id for item in ast.walk(node) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)})
        dependencies = sorted({producers[name] for name in loaded if name in producers})
        if parent:
            dependencies.append(parent)
        dependencies = sorted(set(dependencies))
        output_type = infer_node_type(node, env)
        atom = {
            "record_type": "semantic_atom",
            "atom_id": atom_id,
            "intent": semantic_intent(node),
            "node_type": type(node).__name__,
            "typed_inputs": [{"binding": name, "type": env.get(name, "unknown")} for name in loaded],
            "typed_outputs": [{"binding": name, "type": output_type} for name in stored],
            "constraints": semantic_constraints(node),
            "dependencies": dependencies,
            "authority_required": ["local_private_candidate_analysis"],
            "validator": "python_ast_compile_plus_private_verifier",
            "target": "strict_generator_or_private_verifier",
            "repair_scope": [atom_id],
            "residual_lineage": residual_refs,
            "source_span": {
                "line": int(getattr(node, "lineno", 0) or 0),
                "end_line": int(getattr(node, "end_lineno", 0) or 0),
            },
            "vcm_context_ref": vcm_context_ref,
            "support_state": "model_output_observed",
            **NO_CHEAT,
        }
        atoms.append(atom)
        for dependency in dependencies:
            edges.append({"from_atom": dependency, "to_atom": atom_id, "edge_kind": "data_or_control_dependency"})
        for name in stored:
            producers[name] = atom_id
            env[name] = output_type
        for child in ast.iter_child_nodes(node):
            visit(child, atom_id)
        return atom_id

    for stmt in function.body:
        visit(stmt)
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    output_types = sorted({infer_node_type(node.value, env) for node in returns if node.value is not None})
    open_obligations: list[dict[str, Any]] = []
    if not returns:
        open_obligations.append(obligation("missing_return_closure", atoms[-1]["atom_id"] if atoms else "function"))
    undefined = undefined_load_names(function)
    for name in undefined:
        open_obligations.append(obligation("undefined_binding", producers.get(name, "function"), binding=name))
    return {
        "policy": POLICY,
        "program_id": f"program-{stable_hash(canonical_function(function))[:20]}",
        "entry_point": function.name,
        "prompt_sha256": stable_hash(prompt),
        "callable_signature_sha256": stable_hash(callable_signature),
        "learned_prefix_sha256": stable_hash(list(learned_prefix_tokens)),
        "input_bindings": [{"binding": name, "type": kind} for name, kind in sorted(env.items()) if name in argument_names(function)],
        "output_types": output_types or ["unknown"],
        "atoms": atoms,
        "dependency_edges": edges,
        "operation_counts": dict(sorted(Counter(atom["intent"] for atom in atoms).items())),
        "open_obligations": open_obligations,
        "vcm_context_ref": vcm_context_ref,
        "generation_boundary": "prompt_signature_and_model_output_only",
        "uses_eval_tests_or_solutions": False,
        "uses_answer_metadata": False,
        "uses_public_data": False,
        **NO_CHEAT,
    }


def encode_value(value: Any) -> list[str]:
    if isinstance(value, ast.AST):
        tokens = [f"IR:NODE:{type(value).__name__}"]
        for field in value._fields:
            if field in OMITTED_AST_FIELDS:
                continue
            tokens.extend(encode_value(getattr(value, field, None)))
        return tokens
    if isinstance(value, list):
        tokens = [f"{LIST_BEGIN}:{len(value)}"]
        for item in value:
            tokens.extend(encode_value(item))
        return tokens
    if value is None:
        return ["IR:NONE"]
    kind, payload = scalar_payload(value)
    return [f"IR:SCALAR:{kind}:{len(payload)}", *[f"IR:BYTE:{byte:02x}" for byte in payload]]


def scalar_payload(value: Any) -> tuple[str, bytes]:
    if value is Ellipsis:
        return "ellipsis", b""
    if isinstance(value, bool):
        return "bool", b"1" if value else b"0"
    if isinstance(value, int):
        return "int", str(value).encode("ascii")
    if isinstance(value, float):
        return "float", repr(value).encode("ascii")
    if isinstance(value, complex):
        return "complex", repr(value).encode("ascii")
    if isinstance(value, str):
        return "str", value.encode("utf-8")
    if isinstance(value, bytes):
        return "bytes", value
    raise SemanticIRFault("IR_UNSUPPORTED_SCALAR", type(value).__name__)


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.index = 0

    def peek(self) -> str:
        return self.tokens[self.index] if self.index < len(self.tokens) else ""

    def pop(self) -> str:
        token = self.peek()
        if token:
            self.index += 1
        return token

    def expect(self, expected: str) -> None:
        actual = self.pop()
        if actual != expected:
            raise self.fault("IR_TOKEN_EXPECTED", f"expected {expected!r}, received {actual!r}")

    def fault(self, code: str, detail: str) -> SemanticIRFault:
        return SemanticIRFault(code, detail, token_index=self.index)

    def parse_value(self) -> Any:
        token = self.peek()
        if token == "IR:NONE":
            self.pop()
            return None
        if token.startswith(f"{LIST_BEGIN}:"):
            header = self.pop()
            try:
                count = int(header.rsplit(":", 1)[1])
            except ValueError as exc:
                raise self.fault("IR_LIST_COUNT_INVALID", header) from exc
            if count < 0:
                raise self.fault("IR_LIST_COUNT_INVALID", header)
            return [self.parse_value() for _ in range(count)]
        if token.startswith("IR:SCALAR:"):
            return self.parse_scalar()
        if token.startswith("IR:NODE:"):
            return self.parse_node()
        raise self.fault("IR_VALUE_TOKEN_INVALID", token or "end_of_stream")

    def parse_scalar(self) -> Any:
        header = self.pop()
        parts = header.split(":")
        if len(parts) != 4:
            raise self.fault("IR_SCALAR_HEADER_INVALID", header)
        kind = parts[2]
        try:
            count = int(parts[3])
        except ValueError as exc:
            raise self.fault("IR_SCALAR_LENGTH_INVALID", header) from exc
        if count < 0:
            raise self.fault("IR_SCALAR_LENGTH_INVALID", header)
        payload = bytearray()
        for _ in range(count):
            token = self.pop()
            if not token:
                raise self.fault("IR_TRUNCATED_SCALAR", f"expected {count} bytes for {kind}")
            if not token.startswith("IR:BYTE:"):
                raise self.fault("IR_SCALAR_BYTE_INVALID", token)
            try:
                payload.append(int(token.removeprefix("IR:BYTE:"), 16))
            except ValueError as exc:
                raise self.fault("IR_SCALAR_BYTE_INVALID", token) from exc
        try:
            if kind == "ellipsis":
                return Ellipsis
            if kind == "bool":
                return bytes(payload) == b"1"
            if kind == "int":
                return int(bytes(payload).decode("ascii"))
            if kind == "float":
                return float(bytes(payload).decode("ascii"))
            if kind == "complex":
                return complex(bytes(payload).decode("ascii"))
            if kind == "str":
                return bytes(payload).decode("utf-8")
            if kind == "bytes":
                return bytes(payload)
        except (UnicodeDecodeError, ValueError) as exc:
            raise self.fault("IR_SCALAR_DECODE_FAILED", f"{kind}: {exc}") from exc
        raise self.fault("IR_SCALAR_KIND_UNSUPPORTED", kind)

    def parse_node(self) -> ast.AST:
        name = self.pop().removeprefix("IR:NODE:")
        cls = allowed_ast_classes().get(name)
        if cls is None:
            raise self.fault("IR_NODE_KIND_UNSUPPORTED", name)
        kwargs: dict[str, Any] = {}
        for field in cls._fields:
            if field == "ctx":
                kwargs[field] = ast.Load()
            elif field in {"kind", "type_comment"}:
                kwargs[field] = None
            else:
                kwargs[field] = self.parse_value()
        try:
            return cls(**kwargs)
        except (TypeError, ValueError) as exc:
            raise self.fault("IR_NODE_CONSTRUCTION_FAILED", f"{name}: {exc}") from exc


def allowed_ast_classes() -> dict[str, type[ast.AST]]:
    classes: dict[str, type[ast.AST]] = {}
    for name in dir(ast):
        value = getattr(ast, name)
        if isinstance(value, type) and issubclass(value, ast.AST):
            classes[name] = value
    for blocked in {"Module", "Interactive", "Expression", "FunctionType", "Suite"}:
        classes.pop(blocked, None)
    return classes


def parse_body(body: str) -> ast.FunctionDef:
    normalized = textwrap.dedent(str(body or "")).strip("\n")
    if not normalized.strip():
        raise SemanticIRFault("IR_EMPTY_SOURCE_BODY", "body is empty")
    source = "def _theseus_semantic_ir(data=None, other=None):\n" + "\n".join(
        f"    {line}" if line else "" for line in normalized.splitlines()
    ) + "\n"
    return parse_single_function(source)


def parse_single_function(code: str) -> ast.FunctionDef:
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError as exc:
        raise SemanticIRFault("IR_SOURCE_SYNTAX_ERROR", f"{exc.msg} at line {exc.lineno}") from exc
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    non_functions = [node for node in tree.body if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(functions) != 1 or non_functions:
        raise SemanticIRFault("IR_SINGLE_FUNCTION_REQUIRED", f"functions={len(functions)} non_functions={len(non_functions)}")
    function = functions[0]
    if not isinstance(function, ast.FunctionDef):
        raise SemanticIRFault("IR_ASYNC_FUNCTION_UNSUPPORTED", function.name)
    return function


def canonical_ast(body: tuple[ast.stmt, ...] | list[ast.stmt]) -> str:
    module = ast.Module(body=list(body), type_ignores=[])
    return ast.dump(module, annotate_fields=True, include_attributes=False)


def canonical_function(function: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    return ast.dump(function, annotate_fields=True, include_attributes=False)


def canonical_signature(function: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    payload = ast.FunctionDef(
        name=function.name,
        args=function.args,
        body=[ast.Pass()],
        decorator_list=function.decorator_list,
        returns=function.returns,
        type_comment=getattr(function, "type_comment", None),
    )
    return ast.dump(payload, annotate_fields=True, include_attributes=False)


def normalize_contexts(body: list[ast.stmt]) -> None:
    module = ast.Module(body=body, type_ignores=[])
    for node in ast.walk(module):
        if isinstance(node, ast.Name):
            node.ctx = ast.Load()
        elif isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred, ast.List, ast.Tuple)):
            node.ctx = ast.Load()
    for node in ast.walk(module):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                set_target_context(target, ast.Store)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            set_target_context(node.target, ast.Store)
        elif isinstance(node, ast.comprehension):
            set_target_context(node.target, ast.Store)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    set_target_context(item.optional_vars, ast.Store)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                set_target_context(target, ast.Del)


def set_target_context(node: ast.AST, context: type[ast.expr_context]) -> None:
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript, ast.Starred, ast.List, ast.Tuple)):
        node.ctx = context()
    if isinstance(node, (ast.Starred, ast.List, ast.Tuple)):
        for child in ast.iter_child_nodes(node):
            set_target_context(child, context)


def canonical_body(body: str) -> str:
    return canonical_ast(tuple(parse_body(body).body))


def semantic_intent(node: ast.AST) -> str:
    if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        return "traversal"
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        return "state_update"
    if isinstance(node, ast.Return):
        return "return_closure"
    match_type = getattr(ast, "Match", ())
    if isinstance(node, (ast.If, ast.IfExp) + ((match_type,) if isinstance(match_type, type) else ())):
        return "branch_constraint"
    if isinstance(node, (ast.Call, ast.Attribute, ast.Subscript, ast.BinOp, ast.BoolOp, ast.Compare, ast.UnaryOp)):
        return "value_expression"
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension)):
        return "collection_construction"
    if isinstance(node, (ast.Try, ast.Raise, ast.Assert)):
        return "verification_or_failure_boundary"
    if isinstance(node, (ast.Break, ast.Continue)):
        return "control_finalizer"
    if isinstance(node, ast.Expr):
        return "effect_expression"
    if isinstance(node, ast.Name):
        return "binding_reference"
    if isinstance(node, ast.Constant):
        return "literal_value"
    return "semantic_operation"


def semantic_constraints(node: ast.AST) -> list[str]:
    constraints = ["python_ast_well_formed"]
    if isinstance(node, ast.Return):
        constraints.append("return_value_must_be_defined")
    if isinstance(node, (ast.For, ast.AsyncFor)):
        constraints.extend(["iterable_source_required", "loop_target_bound_before_body"])
    if isinstance(node, ast.Call):
        constraints.append("call_target_and_arguments_must_be_runtime_valid")
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        constraints.append("state_transition_type_compatible")
    if isinstance(node, (ast.If, ast.While, ast.IfExp)):
        constraints.append("condition_must_be_truth_evaluable")
    return constraints


def infer_node_type(node: ast.AST | None, env: dict[str, str]) -> str:
    if node is None:
        return "none"
    if isinstance(node, ast.Constant):
        return type(node.value).__name__
    if isinstance(node, ast.Name):
        return env.get(node.id, "unknown")
    if isinstance(node, (ast.List, ast.ListComp)):
        return "list"
    if isinstance(node, (ast.Tuple,)):
        return "tuple"
    if isinstance(node, (ast.Set, ast.SetComp)):
        return "set"
    if isinstance(node, (ast.Dict, ast.DictComp)):
        return "dict"
    if isinstance(node, (ast.Compare, ast.BoolOp)):
        return "bool"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return "bool"
    if isinstance(node, ast.IfExp):
        left = infer_node_type(node.body, env)
        right = infer_node_type(node.orelse, env)
        return left if left == right else "union"
    if isinstance(node, ast.Call):
        name = call_name(node.func)
        return {
            "bool": "bool", "bytes": "bytes", "dict": "dict", "float": "float", "int": "int",
            "list": "list", "set": "set", "sorted": "list", "str": "str", "tuple": "tuple",
            "len": "int", "sum": "number", "max": "unknown", "min": "unknown", "round": "number",
        }.get(name, "unknown")
    if isinstance(node, ast.BinOp):
        left = infer_node_type(node.left, env)
        right = infer_node_type(node.right, env)
        return left if left == right else "unknown"
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        return infer_node_type(node.value, env)
    if isinstance(node, ast.AugAssign):
        return infer_node_type(node.value, env)
    if isinstance(node, ast.Return):
        return infer_node_type(node.value, env)
    return "unknown"


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def argument_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names = {arg.arg for arg in function.args.args + function.args.kwonlyargs}
    if function.args.vararg:
        names.add(function.args.vararg.arg)
    if function.args.kwarg:
        names.add(function.args.kwarg.arg)
    return names


def undefined_load_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    builtins = {
        "False", "None", "True", "abs", "all", "any", "bool", "bytes", "dict", "enumerate",
        "filter", "float", "int", "isinstance", "len", "list", "map", "max", "min", "ord",
        "pow", "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
    }
    defined = argument_names(function)
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split(".", 1)[0])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not function:
            defined.add(node.name)
        if isinstance(node, ast.ExceptHandler) and isinstance(node.name, str):
            defined.add(node.name)
    loaded = {node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    return sorted(loaded - defined - builtins)


def obligation(kind: str, atom_id: str, *, binding: str = "") -> dict[str, Any]:
    return {
        "record_type": "semantic_obligation",
        "obligation_id": f"obligation-{stable_hash([kind, atom_id, binding])[:16]}",
        "obligation_type": kind,
        "dependent_atom_ids": [atom_id],
        "binding": binding,
        "repair_scope": [atom_id],
        "validator": "python_ast_compile_plus_private_verifier",
        "support_state": "OPEN",
        "failure_behavior": "typed_fault_no_fallback",
    }


def stable_hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
