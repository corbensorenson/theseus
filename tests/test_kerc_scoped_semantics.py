from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import kernel_english_protocol as kernel  # noqa: E402
import kerc_scoped_semantics as scoped  # noqa: E402
from kerc_scoped_semantics_verify import (  # noqa: E402
    independently_verify_scoped_semantic_program,
)


SOURCE = "The treatment may not reduce risk."


def proposition() -> dict:
    return {
        "proposition_id": "p0",
        "predicate": "REDUCE",
        "modality": "ASSERTED",
        "polarity": "AFFIRMED",
        "quantifier": "NONE",
        "confidence": 1.0,
        "derivation": "preserved",
        "source_spans": [[0, len(SOURCE)]],
        "arguments": [
            {"role": "AG", "value": {"type": "concept", "value": "medical.treatment"}},
            {"role": "OBJ", "value": {"type": "concept", "value": "risk"}},
        ],
    }


def may_not_graph() -> dict:
    return {
        "policy": scoped.POLICY,
        "roots": ["s1"],
        "scopes": [
            {
                "scope_id": "s0",
                "operator": "NEGATION",
                "targets": [{"role": "BODY", "target_id": "p0"}],
                "arguments": [],
                "source_spans": [[18, 21]],
            },
            {
                "scope_id": "s1",
                "operator": "POSSIBILITY",
                "targets": [{"role": "BODY", "target_id": "s0"}],
                "arguments": [],
                "source_spans": [[14, 17]],
            },
        ],
        "propositions": [proposition()],
    }


def not_possible_graph() -> dict:
    graph = may_not_graph()
    graph["scopes"] = [
        {
            "scope_id": "s0",
            "operator": "POSSIBILITY",
            "targets": [{"role": "BODY", "target_id": "p0"}],
            "arguments": [],
            "source_spans": [[14, 17]],
        },
        {
            "scope_id": "s1",
            "operator": "NEGATION",
            "targets": [{"role": "BODY", "target_id": "s0"}],
            "arguments": [],
            "source_spans": [[18, 21]],
        },
    ]
    return graph


def compile_graph(graph: dict) -> dict:
    return scoped.compile_scoped_semantic_graph(
        graph,
        protected_objects={},
        concept_capsules={},
        source_length=len(SOURCE),
    )


def test_nested_scope_order_is_causal_and_independently_replayed() -> None:
    may_not = compile_graph(may_not_graph())
    not_possible = compile_graph(not_possible_graph())
    assert may_not["program"]["program_sha256"] != not_possible["program"]["program_sha256"]
    may_ops = {row["node_id"]: row for row in may_not["program"]["nodes"]}
    assert may_ops[may_not["program"]["roots"][0]]["operator"] == "SCOPE_POSSIBILITY"
    assert independently_verify_scoped_semantic_program(
        may_not_graph(),
        may_not,
        protected_objects={},
        concept_capsules={},
        source_length=len(SOURCE),
    )["passes"] is True
    assert independently_verify_scoped_semantic_program(
        not_possible_graph(),
        not_possible,
        protected_objects={},
        concept_capsules={},
        source_length=len(SOURCE),
    )["passes"] is True


def test_independent_verifier_rejects_scope_role_and_topology_mutations() -> None:
    graph = may_not_graph()
    compiled = compile_graph(graph)
    mutated = copy.deepcopy(compiled)
    root = next(row for row in mutated["program"]["nodes"] if row["operator"] == "SCOPE_POSSIBILITY")
    root["operator"] = "SCOPE_NECESSITY"
    root["program_sha256"] = kernel.stable_hash(root)
    receipt = independently_verify_scoped_semantic_program(
        graph,
        mutated,
        protected_objects={},
        concept_capsules={},
        source_length=len(SOURCE),
    )
    assert receipt["passes"] is False
    assert receipt["failures"][0]["fault"] == "program_mismatch"


def test_scope_contract_rejects_cycles_multiple_ownership_and_bad_arity() -> None:
    cycle = may_not_graph()
    cycle["scopes"][0]["targets"][0]["target_id"] = "s1"
    with pytest.raises(scoped.ScopedSemanticFault, match="KERC_SCOPE_CYCLE"):
        compile_graph(cycle)

    multiply_owned = may_not_graph()
    multiply_owned["roots"] = ["s0", "s1"]
    multiply_owned["scopes"][1]["targets"][0]["target_id"] = "p0"
    with pytest.raises(scoped.ScopedSemanticFault, match="KERC_SCOPE_OWNERSHIP_INVALID"):
        compile_graph(multiply_owned)

    bad_arity = may_not_graph()
    bad_arity["scopes"][0]["targets"].append({"role": "BODY", "target_id": "p0"})
    with pytest.raises(scoped.ScopedSemanticFault, match="KERC_SCOPE_ARITY_INVALID"):
        compile_graph(bad_arity)


def test_attribution_requires_explicit_source_or_holder() -> None:
    graph = may_not_graph()
    graph["scopes"] = [
        {
            "scope_id": "s0",
            "operator": "ATTRIBUTION",
            "targets": [{"role": "CONTENT", "target_id": "p0"}],
            "arguments": [],
            "source_spans": [[0, len(SOURCE)]],
        }
    ]
    graph["roots"] = ["s0"]
    with pytest.raises(scoped.ScopedSemanticFault, match="KERC_SCOPE_ATTRIBUTION_SOURCE_MISSING"):
        compile_graph(graph)
    graph["scopes"][0]["arguments"] = [
        {"role": "SOURCE", "value": {"type": "concept", "value": "speaker.unknown"}}
    ]
    assert compile_graph(graph)["scope_ownership_exact"] is True


def test_typed_values_and_full_program_serialization_roundtrip_exactly() -> None:
    program = {
        "roots": ["k0"],
        "nodes": [
            {
                "node_id": "k0",
                "operator": "OBSERVE",
                "modality": "PROBABLE",
                "polarity": "AFFIRMED",
                "quantifier": "EXACT",
                "confidence": 0.875,
                "derivation": "tool_evidence",
                "source_spans": [[0, len(SOURCE)]],
                "arguments": [
                    {
                        "role": "VALUE",
                        "value": {
                            "type": "quantity",
                            "value": {
                                "kind": "LENGTH",
                                "relation": "APPROX",
                                "value": "6",
                                "lower": None,
                                "upper": None,
                                "unit_concept": "unit.foot",
                                "approximate": True,
                            },
                        },
                    },
                    {
                        "role": "TIME",
                        "value": {
                            "type": "temporal",
                            "value": {
                                "kind": "RELATIVE",
                                "relation": "AFTER",
                                "value": "next-thursday",
                                "anchor": "time.now",
                                "calendar": None,
                            },
                        },
                    },
                    {"role": "LABEL", "value": {"type": "text", "value": {"text": "six feet", "language": "en"}}},
                    {"role": "STATUS", "value": {"type": "symbol", "value": "observed/value"}},
                    {
                        "role": "ALTERNATIVES",
                        "value": {
                            "type": "ambiguity",
                            "value": [
                                {"probability": 1.0 / 3.0, "evidence": "private_fixture", "value": {"type": "concept", "value": f"option.{index}"}}
                                for index in range(3)
                            ],
                        },
                    },
                ],
            }
        ],
    }
    canonical = kernel.validate_kernel_program(
        program,
        protected_objects={},
        concept_capsules={},
        source_character_length=len(SOURCE),
    )["canonical_program"]
    serialization = kernel.serialize_kernel_program(canonical)
    replay = kernel.deserialize_kernel_program(
        serialization,
        protected_objects={},
        concept_capsules={},
        source_character_length=len(SOURCE),
    )
    assert replay["canonical_program"] == canonical
    assert replay["exact_program_roundtrip"] is True

    root_mutation = copy.deepcopy(serialization)
    for stream in (root_mutation["expanded_tokens"], root_mutation["compact_tokens"]):
        next(row for row in stream if row["token"].startswith("ROOT:"))["token"] = "ROOT:k99"
    root_mutation["expanded_sha256"] = kernel.stable_hash(root_mutation["expanded_tokens"])
    root_mutation["compact_sha256"] = kernel.stable_hash(root_mutation["compact_tokens"])
    with pytest.raises(kernel.KernelProtocolFault, match="KERC_ROOT_REFERENCE_UNKNOWN"):
        kernel.deserialize_kernel_program(
            root_mutation,
            protected_objects={},
            concept_capsules={},
            source_character_length=len(SOURCE),
        )


@pytest.mark.parametrize(
    "value,fault",
    [
        (
            {"type": "quantity", "value": {"kind": "LENGTH", "relation": "BETWEEN", "value": None, "lower": "9", "upper": "2", "unit_concept": "unit.meter", "approximate": False}},
            "KERC_QUANTITY_BOUNDS_REVERSED",
        ),
        (
            {"type": "quantity", "value": {"kind": "COUNT", "relation": "EXACT", "value": "2.0", "lower": None, "upper": None, "unit_concept": None, "approximate": False}},
            "KERC_DECIMAL_VALUE_NONCANONICAL",
        ),
        (
            {"type": "temporal", "value": {"kind": "DATE", "relation": "AT", "value": "", "anchor": None, "calendar": "ISO8601"}},
            "KERC_TEMPORAL_VALUE_INVALID",
        ),
        (
            {"type": "temporal", "value": {"kind": "DATE", "relation": "AT", "value": "next-thursday", "anchor": None, "calendar": "GREGORIAN"}},
            "KERC_TEMPORAL_DATE_INVALID",
        ),
        (
            {"type": "number", "value": {"value": True}},
            "KERC_NUMBER_INVALID",
        ),
    ],
)
def test_typed_value_mutations_fail_closed(value: dict, fault: str) -> None:
    row = proposition()
    row["arguments"] = [{"role": "VALUE", "value": value}]
    graph = may_not_graph()
    graph["propositions"] = [row]
    with pytest.raises(kernel.KernelProtocolFault, match=fault):
        compile_graph(graph)


@pytest.mark.parametrize(
    "operator,roles",
    [
        ("ASSERTION", ["BODY"]),
        ("NEGATION", ["BODY"]),
        ("POSSIBILITY", ["BODY"]),
        ("NECESSITY", ["BODY"]),
        ("QUESTION", ["BODY"]),
        ("CONJUNCTION", ["MEMBER", "MEMBER"]),
        ("ALTERNATION", ["MEMBER", "MEMBER"]),
        ("CONDITION", ["ANTECEDENT", "CONSEQUENT"]),
        ("CONSEQUENCE", ["CAUSE", "RESULT"]),
        ("CONTRAST", ["LEFT", "RIGHT"]),
        ("CONTINUATION", ["PREVIOUS", "NEXT"]),
        ("EXPLANATION", ["CLAIM", "EVIDENCE"]),
        ("ATTRIBUTION", ["CONTENT"]),
        ("QUOTATION", ["CONTENT"]),
    ],
)
def test_every_declared_scope_operator_has_an_independently_verified_route(
    operator: str, roles: list[str]
) -> None:
    propositions = [proposition()]
    propositions[0]["proposition_id"] = "p0"
    if len(roles) > 1:
        second = copy.deepcopy(proposition())
        second["proposition_id"] = "p1"
        second["predicate"] = "INCREASE"
        propositions.append(second)
    targets = [
        {"role": role, "target_id": f"p{min(index, len(propositions) - 1)}"}
        for index, role in enumerate(roles)
    ]
    arguments = []
    if operator in {"ATTRIBUTION", "QUOTATION"}:
        arguments.append(
            {"role": "SOURCE", "value": {"type": "concept", "value": "speaker.unknown"}}
        )
    graph = {
        "policy": scoped.POLICY,
        "roots": ["s0"],
        "scopes": [
            {
                "scope_id": "s0",
                "operator": operator,
                "targets": targets,
                "arguments": arguments,
                "source_spans": [[0, len(SOURCE)]],
            }
        ],
        "propositions": propositions,
    }
    compiled = compile_graph(graph)
    receipt = independently_verify_scoped_semantic_program(
        graph,
        compiled,
        protected_objects={},
        concept_capsules={},
        source_length=len(SOURCE),
    )
    assert receipt["passes"] is True
    assert compiled["program"]["nodes"][-1]["operator"] == f"SCOPE_{operator}"


def test_quantifier_and_condition_direction_interventions_change_program_identity() -> None:
    universal = may_not_graph()
    universal["propositions"][0]["quantifier"] = "FORALL"
    existential = copy.deepcopy(universal)
    existential["propositions"][0]["quantifier"] = "EXISTS"
    assert compile_graph(universal)["program"]["program_sha256"] != compile_graph(existential)["program"]["program_sha256"]

    condition = {
        "policy": scoped.POLICY,
        "roots": ["s0"],
        "scopes": [
            {
                "scope_id": "s0",
                "operator": "CONDITION",
                "targets": [
                    {"role": "ANTECEDENT", "target_id": "p0"},
                    {"role": "CONSEQUENT", "target_id": "p1"},
                ],
                "arguments": [],
                "source_spans": [[0, len(SOURCE)]],
            }
        ],
        "propositions": [proposition(), {**copy.deepcopy(proposition()), "proposition_id": "p1", "predicate": "INCREASE"}],
    }
    reversed_condition = copy.deepcopy(condition)
    reversed_condition["scopes"][0]["targets"][0]["target_id"] = "p1"
    reversed_condition["scopes"][0]["targets"][1]["target_id"] = "p0"
    assert compile_graph(condition)["program"]["program_sha256"] != compile_graph(reversed_condition)["program"]["program_sha256"]


def test_noncanonical_input_order_replays_to_the_same_canonical_graph() -> None:
    graph = may_not_graph()
    graph["scopes"].reverse()
    compiled = compile_graph(graph)
    receipt = independently_verify_scoped_semantic_program(
        graph,
        compiled,
        protected_objects={},
        concept_capsules={},
        source_length=len(SOURCE),
    )
    assert receipt["passes"] is True
