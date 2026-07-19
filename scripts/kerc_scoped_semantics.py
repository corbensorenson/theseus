#!/usr/bin/env python3
"""Canonical scoped-semantic graph contract for Kernel English.

This is a deterministic ABI and compiler, not a semantic parser. It makes
proposition ownership and operator scope explicit so learned components cannot
collapse distinctions such as POSSIBLE(NEGATED(P)) and
NEGATED(POSSIBLE(P)). It writes no training rows and earns no learned credit.
"""

from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any, Mapping

from kernel_english_protocol import (
    NO_CHEAT,
    KernelProtocolFault,
    canonical_json,
    stable_hash,
    validate_kernel_program,
)


POLICY = "project_theseus_kerc_scoped_semantic_graph_v1"
COMPILER_POLICY = "project_theseus_kerc_scoped_semantic_compiler_v1"
SCOPE_OPERATORS: dict[str, dict[str, tuple[int, int | None]]] = {
    "ASSERTION": {"BODY": (1, 1)},
    "NEGATION": {"BODY": (1, 1)},
    "POSSIBILITY": {"BODY": (1, 1)},
    "NECESSITY": {"BODY": (1, 1)},
    "QUESTION": {"BODY": (1, 1)},
    "CONJUNCTION": {"MEMBER": (2, None)},
    "ALTERNATION": {"MEMBER": (2, None)},
    "CONDITION": {"ANTECEDENT": (1, 1), "CONSEQUENT": (1, 1)},
    "CONSEQUENCE": {"CAUSE": (1, 1), "RESULT": (1, 1)},
    "CONTRAST": {"LEFT": (1, 1), "RIGHT": (1, 1)},
    "CONTINUATION": {"PREVIOUS": (1, 1), "NEXT": (1, 1)},
    "EXPLANATION": {"CLAIM": (1, 1), "EVIDENCE": (1, 1)},
    "ATTRIBUTION": {"CONTENT": (1, 1)},
    "QUOTATION": {"CONTENT": (1, 1)},
}
ATTRIBUTED_OPERATORS = {"ATTRIBUTION", "QUOTATION"}


class ScopedSemanticFault(ValueError):
    def __init__(self, code: str, detail: str, *, path: str = "") -> None:
        self.code = code
        self.detail = detail
        self.path = path
        suffix = f" at {path}" if path else ""
        super().__init__(f"{code}{suffix}: {detail}")


def _check_spans(spans: Any, *, source_length: int, path: str) -> list[list[int]]:
    if not isinstance(spans, list):
        raise ScopedSemanticFault("KERC_SCOPE_SPANS_INVALID", canonical_json(spans), path=path)
    normalized: list[list[int]] = []
    for index, span in enumerate(spans):
        if (
            not isinstance(span, list)
            or len(span) != 2
            or isinstance(span[0], bool)
            or isinstance(span[1], bool)
        ):
            raise ScopedSemanticFault("KERC_SCOPE_SPAN_INVALID", canonical_json(span), path=f"{path}[{index}]")
        start, end = int(span[0]), int(span[1])
        if not 0 <= start < end <= source_length:
            raise ScopedSemanticFault("KERC_SCOPE_SPAN_INVALID", f"{start}:{end}", path=f"{path}[{index}]")
        normalized.append([start, end])
    return normalized


def _reject_cycle(edges: Mapping[str, list[str]]) -> None:
    active: set[str] = set()
    complete: set[str] = set()

    def visit(node: str) -> None:
        if node in active:
            raise ScopedSemanticFault("KERC_SCOPE_CYCLE", node, path="scopes")
        if node in complete:
            return
        active.add(node)
        for child in edges.get(node, []):
            if child in edges:
                visit(child)
        active.remove(node)
        complete.add(node)

    for node in sorted(edges):
        visit(node)


def validate_scoped_semantic_graph(spec: dict[str, Any], *, source_length: int) -> dict[str, Any]:
    if not isinstance(spec, dict) or spec.get("policy") != POLICY:
        raise ScopedSemanticFault("KERC_SCOPE_POLICY_INVALID", str(spec.get("policy") if isinstance(spec, dict) else type(spec)), path="policy")
    if set(spec) != {"policy", "roots", "scopes", "propositions"}:
        raise ScopedSemanticFault("KERC_SCOPE_SCHEMA_INVALID", canonical_json(sorted(spec)), path="graph")
    roots = spec.get("roots")
    scopes = spec.get("scopes")
    propositions = spec.get("propositions")
    if not isinstance(roots, list) or not roots or len(roots) != len(set(roots)):
        raise ScopedSemanticFault("KERC_SCOPE_ROOTS_INVALID", canonical_json(roots), path="roots")
    if not isinstance(scopes, list) or not scopes or not isinstance(propositions, list) or not propositions:
        raise ScopedSemanticFault("KERC_SCOPE_CONTENT_MISSING", "scopes and propositions must be non-empty")

    scope_by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(scopes):
        path = f"scopes[{index}]"
        if not isinstance(raw, dict) or set(raw) != {"scope_id", "operator", "targets", "arguments", "source_spans"}:
            raise ScopedSemanticFault("KERC_SCOPE_ENTRY_SCHEMA_INVALID", canonical_json(raw), path=path)
        scope_id = str(raw.get("scope_id") or "")
        operator = str(raw.get("operator") or "")
        if not re.fullmatch(r"s[0-9]+", scope_id) or scope_id in scope_by_id:
            raise ScopedSemanticFault("KERC_SCOPE_ID_INVALID", scope_id, path=f"{path}.scope_id")
        if operator not in SCOPE_OPERATORS:
            raise ScopedSemanticFault("KERC_SCOPE_OPERATOR_INVALID", operator, path=f"{path}.operator")
        targets = raw.get("targets")
        arguments = raw.get("arguments")
        if not isinstance(targets, list) or not isinstance(arguments, list):
            raise ScopedSemanticFault("KERC_SCOPE_ARGUMENTS_INVALID", canonical_json(raw), path=path)
        counts: Counter[str] = Counter()
        normalized_targets = []
        for target_index, target in enumerate(targets):
            target_path = f"{path}.targets[{target_index}]"
            if not isinstance(target, dict) or set(target) != {"role", "target_id"}:
                raise ScopedSemanticFault("KERC_SCOPE_TARGET_SCHEMA_INVALID", canonical_json(target), path=target_path)
            role = str(target.get("role") or "")
            target_id = str(target.get("target_id") or "")
            if role not in SCOPE_OPERATORS[operator] or not re.fullmatch(r"[ps][0-9]+", target_id):
                raise ScopedSemanticFault("KERC_SCOPE_TARGET_INVALID", canonical_json(target), path=target_path)
            counts[role] += 1
            normalized_targets.append({"role": role, "target_id": target_id})
        for role, (minimum, maximum) in SCOPE_OPERATORS[operator].items():
            count = counts[role]
            if count < minimum or (maximum is not None and count > maximum):
                raise ScopedSemanticFault("KERC_SCOPE_ARITY_INVALID", f"{operator}:{role}:{count}", path=f"{path}.targets")
        if set(counts) != set(SCOPE_OPERATORS[operator]):
            raise ScopedSemanticFault("KERC_SCOPE_ROLE_SET_INVALID", canonical_json(sorted(counts)), path=f"{path}.targets")
        if operator in ATTRIBUTED_OPERATORS and not any(str(arg.get("role") or "") in {"SOURCE", "HOLDER", "SPEAKER"} for arg in arguments if isinstance(arg, dict)):
            raise ScopedSemanticFault("KERC_SCOPE_ATTRIBUTION_SOURCE_MISSING", operator, path=f"{path}.arguments")
        normalized_arguments = []
        for arg_index, argument in enumerate(arguments):
            arg_path = f"{path}.arguments[{arg_index}]"
            if (
                not isinstance(argument, dict)
                or set(argument) != {"role", "value"}
                or not re.fullmatch(r"[A-Z][A-Z0-9_]*", str(argument.get("role") or ""))
                or (isinstance(argument.get("value"), dict) and argument["value"].get("type") == "node_ref")
            ):
                raise ScopedSemanticFault("KERC_SCOPE_ATTRIBUTE_INVALID", canonical_json(argument), path=arg_path)
            normalized_arguments.append(copy.deepcopy(argument))
        scope_by_id[scope_id] = {
            "scope_id": scope_id,
            "operator": operator,
            "targets": normalized_targets,
            "arguments": normalized_arguments,
            "source_spans": _check_spans(raw["source_spans"], source_length=source_length, path=f"{path}.source_spans"),
        }

    proposition_by_id: dict[str, dict[str, Any]] = {}
    proposition_fields = {"proposition_id", "predicate", "modality", "polarity", "quantifier", "confidence", "derivation", "arguments", "source_spans"}
    for index, raw in enumerate(propositions):
        path = f"propositions[{index}]"
        if not isinstance(raw, dict) or set(raw) != proposition_fields:
            raise ScopedSemanticFault("KERC_PROPOSITION_SCHEMA_INVALID", canonical_json(raw), path=path)
        proposition_id = str(raw.get("proposition_id") or "")
        if not re.fullmatch(r"p[0-9]+", proposition_id) or proposition_id in proposition_by_id:
            raise ScopedSemanticFault("KERC_PROPOSITION_ID_INVALID", proposition_id, path=f"{path}.proposition_id")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", str(raw.get("predicate") or "")):
            raise ScopedSemanticFault("KERC_PROPOSITION_PREDICATE_INVALID", str(raw.get("predicate")), path=f"{path}.predicate")
        proposition_by_id[proposition_id] = {
            **copy.deepcopy(raw),
            "source_spans": _check_spans(raw["source_spans"], source_length=source_length, path=f"{path}.source_spans"),
        }

    all_ids = set(scope_by_id) | set(proposition_by_id)
    if len(all_ids) != len(scope_by_id) + len(proposition_by_id):
        raise ScopedSemanticFault("KERC_SCOPE_PROPOSITION_ID_COLLISION", canonical_json(sorted(all_ids)))
    if not set(roots) <= set(scope_by_id):
        raise ScopedSemanticFault("KERC_SCOPE_ROOT_UNKNOWN", canonical_json(roots), path="roots")
    parent_count: Counter[str] = Counter()
    edges: dict[str, list[str]] = {}
    for scope_id, scope in scope_by_id.items():
        targets = [target["target_id"] for target in scope["targets"]]
        unknown = sorted(set(targets) - all_ids)
        if unknown:
            raise ScopedSemanticFault("KERC_SCOPE_TARGET_UNKNOWN", canonical_json(unknown), path=f"scopes.{scope_id}.targets")
        edges[scope_id] = targets
        parent_count.update(targets)
    _reject_cycle(edges)
    expected_roots = sorted(scope_id for scope_id in scope_by_id if parent_count[scope_id] == 0)
    if sorted(roots) != expected_roots:
        raise ScopedSemanticFault("KERC_SCOPE_ROOT_OWNERSHIP_MISMATCH", canonical_json({"declared": sorted(roots), "expected": expected_roots}), path="roots")
    multiply_owned = sorted(node for node, count in parent_count.items() if count != 1)
    missing_propositions = sorted(set(proposition_by_id) - set(parent_count))
    if multiply_owned or missing_propositions:
        raise ScopedSemanticFault("KERC_SCOPE_OWNERSHIP_INVALID", canonical_json({"not_exactly_one": multiply_owned, "unowned_propositions": missing_propositions}), path="scopes")
    reachable: set[str] = set()
    pending = list(roots)
    while pending:
        current = pending.pop()
        if current in reachable:
            continue
        reachable.add(current)
        pending.extend(edges.get(current, []))
    if reachable != all_ids:
        raise ScopedSemanticFault("KERC_SCOPE_GRAPH_DISCONNECTED", canonical_json(sorted(all_ids - reachable)), path="scopes")
    canonical = {
        "policy": POLICY,
        "roots": sorted(roots),
        "scopes": [scope_by_id[key] for key in sorted(scope_by_id, key=lambda value: int(value[1:]))],
        "propositions": [proposition_by_id[key] for key in sorted(proposition_by_id, key=lambda value: int(value[1:]))],
    }
    canonical["graph_sha256"] = stable_hash(canonical)
    return canonical


def compile_scoped_semantic_graph(
    spec: dict[str, Any],
    *,
    protected_objects: dict[str, dict[str, Any]],
    concept_capsules: dict[str, dict[str, Any]],
    source_length: int,
) -> dict[str, Any]:
    canonical = validate_scoped_semantic_graph(spec, source_length=source_length)
    proposition_ids = [row["proposition_id"] for row in canonical["propositions"]]
    scope_ids = [row["scope_id"] for row in canonical["scopes"]]
    node_id = {identifier: f"k{index}" for index, identifier in enumerate(proposition_ids + scope_ids)}
    nodes = []
    for proposition in canonical["propositions"]:
        nodes.append(
            {
                "node_id": node_id[proposition["proposition_id"]],
                "operator": proposition["predicate"],
                "modality": proposition["modality"],
                "polarity": proposition["polarity"],
                "quantifier": proposition["quantifier"],
                "confidence": proposition["confidence"],
                "derivation": proposition["derivation"],
                "source_spans": proposition["source_spans"],
                "arguments": copy.deepcopy(proposition["arguments"]),
            }
        )
    for scope in canonical["scopes"]:
        target_arguments = [
            {"role": target["role"], "value": {"type": "node_ref", "value": node_id[target["target_id"]]}}
            for target in scope["targets"]
        ]
        nodes.append(
            {
                "node_id": node_id[scope["scope_id"]],
                "operator": f"SCOPE_{scope['operator']}",
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "derivation": "preserved",
                "source_spans": scope["source_spans"],
                "arguments": target_arguments + copy.deepcopy(scope["arguments"]),
            }
        )
    validated = validate_kernel_program(
        {"roots": [node_id[root] for root in canonical["roots"]], "nodes": nodes},
        protected_objects=protected_objects,
        concept_capsules=concept_capsules,
        source_character_length=source_length,
    )
    receipt = {
        "policy": COMPILER_POLICY,
        "scoped_graph_sha256": canonical["graph_sha256"],
        "program": validated["canonical_program"],
        "identity_map": node_id,
        "scope_count": len(scope_ids),
        "proposition_count": len(proposition_ids),
        "scope_ownership_exact": True,
        "semantic_equivalence_claimed": False,
        "truth_verified": False,
        "learned_competence_claimed": False,
        "public_training_rows_written": 0,
        **NO_CHEAT,
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    return receipt
