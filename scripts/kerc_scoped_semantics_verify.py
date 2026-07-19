#!/usr/bin/env python3
"""Independent structural verifier for the KERC scoped-semantic ABI.

This module does not import or execute the producer. It reconstructs the
expected Kernel topology from the frozen graph contract and compares every
field. The shared Kernel protocol is used only as the final ABI validator.
"""

from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any

from kernel_english_protocol import NO_CHEAT, canonical_json, stable_hash, validate_kernel_program


POLICY = "project_theseus_kerc_scoped_semantic_graph_v1"
VERIFIER_POLICY = "project_theseus_kerc_scoped_semantic_independent_verifier_v1"
ROLE_RULES = {
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


class ScopedSemanticVerificationFault(ValueError):
    pass


def independently_verify_scoped_semantic_program(
    spec: dict[str, Any],
    compiled: dict[str, Any],
    *,
    protected_objects: dict[str, dict[str, Any]],
    concept_capsules: dict[str, dict[str, Any]],
    source_length: int,
) -> dict[str, Any]:
    if not isinstance(spec, dict) or spec.get("policy") != POLICY:
        raise ScopedSemanticVerificationFault("independent graph policy mismatch")
    scopes = spec.get("scopes")
    propositions = spec.get("propositions")
    roots = spec.get("roots")
    if not isinstance(scopes, list) or not isinstance(propositions, list) or not isinstance(roots, list):
        raise ScopedSemanticVerificationFault("independent graph shape invalid")
    scope_by_id = {}
    parent_count: Counter[str] = Counter()
    for row in scopes:
        if not isinstance(row, dict) or set(row) != {"scope_id", "operator", "targets", "arguments", "source_spans"}:
            raise ScopedSemanticVerificationFault("independent scope schema invalid")
        scope_id = str(row["scope_id"])
        operator = str(row["operator"])
        if not re.fullmatch(r"s[0-9]+", scope_id) or scope_id in scope_by_id or operator not in ROLE_RULES:
            raise ScopedSemanticVerificationFault("independent scope identity invalid")
        counts = Counter(str(target.get("role")) for target in row["targets"] if isinstance(target, dict))
        if set(counts) != set(ROLE_RULES[operator]):
            raise ScopedSemanticVerificationFault("independent scope role set invalid")
        for role, (minimum, maximum) in ROLE_RULES[operator].items():
            if counts[role] < minimum or (maximum is not None and counts[role] > maximum):
                raise ScopedSemanticVerificationFault("independent scope arity invalid")
        scope_by_id[scope_id] = copy.deepcopy(row)
        parent_count.update(str(target["target_id"]) for target in row["targets"])
    proposition_by_id = {}
    for row in propositions:
        proposition_id = str(row.get("proposition_id") if isinstance(row, dict) else "")
        if not re.fullmatch(r"p[0-9]+", proposition_id) or proposition_id in proposition_by_id:
            raise ScopedSemanticVerificationFault("independent proposition identity invalid")
        proposition_by_id[proposition_id] = copy.deepcopy(row)
    all_ids = set(scope_by_id) | set(proposition_by_id)
    expected_roots = sorted(scope_id for scope_id in scope_by_id if parent_count[scope_id] == 0)
    if sorted(roots) != expected_roots or set(parent_count) != all_ids - set(roots) or any(count != 1 for count in parent_count.values()):
        raise ScopedSemanticVerificationFault("independent ownership reconstruction failed")
    reachable: set[str] = set()
    active: set[str] = set()

    def visit(identifier: str) -> None:
        if identifier in active:
            raise ScopedSemanticVerificationFault("independent scope cycle detected")
        if identifier in reachable:
            return
        active.add(identifier)
        if identifier in scope_by_id:
            for target in scope_by_id[identifier]["targets"]:
                target_id = str(target["target_id"])
                if target_id not in all_ids:
                    raise ScopedSemanticVerificationFault("independent unknown scope target")
                visit(target_id)
        active.remove(identifier)
        reachable.add(identifier)

    for root in roots:
        visit(str(root))
    if reachable != all_ids:
        raise ScopedSemanticVerificationFault("independent scope graph disconnected")
    proposition_ids = sorted(proposition_by_id, key=lambda value: int(value[1:]))
    scope_ids = sorted(scope_by_id, key=lambda value: int(value[1:]))
    identity = {value: f"k{index}" for index, value in enumerate(proposition_ids + scope_ids)}
    expected_nodes = []
    for proposition_id in proposition_ids:
        row = proposition_by_id[proposition_id]
        expected_nodes.append(
            {
                "node_id": identity[proposition_id],
                "operator": row["predicate"],
                "modality": row["modality"],
                "polarity": row["polarity"],
                "quantifier": row["quantifier"],
                "confidence": float(row["confidence"]),
                "derivation": row["derivation"],
                "source_spans": copy.deepcopy(row["source_spans"]),
                "arguments": copy.deepcopy(row["arguments"]),
            }
        )
    for scope_id in scope_ids:
        row = scope_by_id[scope_id]
        expected_nodes.append(
            {
                "node_id": identity[scope_id],
                "operator": f"SCOPE_{row['operator']}",
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "derivation": "preserved",
                "source_spans": copy.deepcopy(row["source_spans"]),
                "arguments": [
                    {"role": target["role"], "value": {"type": "node_ref", "value": identity[target["target_id"]]}}
                    for target in row["targets"]
                ] + copy.deepcopy(row["arguments"]),
            }
        )
    expected = validate_kernel_program(
        {"roots": [identity[root] for root in sorted(roots)], "nodes": expected_nodes},
        protected_objects=protected_objects,
        concept_capsules=concept_capsules,
        source_character_length=source_length,
    )["canonical_program"]
    observed = compiled.get("program") if isinstance(compiled, dict) else None
    failures = []
    if observed != expected:
        failures.append({"fault": "program_mismatch", "expected_sha256": expected["program_sha256"], "observed_sha256": stable_hash(observed)})
    if compiled.get("identity_map") != identity:
        failures.append({"fault": "identity_map_mismatch"})
    canonical_spec = {
        "policy": POLICY,
        "roots": sorted(str(root) for root in roots),
        "scopes": [scope_by_id[key] for key in scope_ids],
        "propositions": [proposition_by_id[key] for key in proposition_ids],
    }
    if compiled.get("scoped_graph_sha256") != stable_hash(canonical_spec):
        failures.append({"fault": "graph_hash_mismatch"})
    return {
        "policy": VERIFIER_POLICY,
        "passes": not failures,
        "failures": failures,
        "expected_program_sha256": expected["program_sha256"],
        "scope_count": len(scope_ids),
        "proposition_count": len(proposition_ids),
        "producer_imported": False,
        "truth_verified": False,
        "semantic_equivalence_claimed": False,
        "learned_competence_claimed": False,
        **NO_CHEAT,
    }
