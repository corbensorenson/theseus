#!/usr/bin/env python3
"""Run the bounded KERC scoped-semantic ABI adequacy gate."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from kernel_english_protocol import (
    NO_CHEAT,
    deserialize_kernel_program,
    serialize_kernel_program,
    stable_hash,
)
from kerc_scoped_semantics import POLICY, SCOPE_OPERATORS, compile_scoped_semantic_graph
from kerc_scoped_semantics_verify import independently_verify_scoped_semantic_program


SOURCE = "A condition may not produce a result."


def proposition(identifier: str, predicate: str) -> dict[str, Any]:
    return {
        "proposition_id": identifier,
        "predicate": predicate,
        "modality": "ASSERTED",
        "polarity": "AFFIRMED",
        "quantifier": "NONE",
        "confidence": 1.0,
        "derivation": "preserved",
        "source_spans": [[0, len(SOURCE)]],
        "arguments": [
            {"role": "SUBJECT", "value": {"type": "concept", "value": "fixture.subject"}}
        ],
    }


def graph_for_operator(operator: str) -> dict[str, Any]:
    roles = []
    for role, (minimum, _maximum) in SCOPE_OPERATORS[operator].items():
        roles.extend([role] * minimum)
    propositions = [proposition(f"p{index}", "OBSERVE" if index == 0 else "CHANGE") for index in range(len(roles))]
    arguments = []
    if operator in {"ATTRIBUTION", "QUOTATION"}:
        arguments.append({"role": "SOURCE", "value": {"type": "concept", "value": "speaker.unknown"}})
    return {
        "policy": POLICY,
        "roots": ["s0"],
        "scopes": [
            {
                "scope_id": "s0",
                "operator": operator,
                "targets": [
                    {"role": role, "target_id": f"p{index}"}
                    for index, role in enumerate(roles)
                ],
                "arguments": arguments,
                "source_spans": [[0, len(SOURCE)]],
            }
        ],
        "propositions": propositions,
    }


def nested_graph(outer: str, inner: str) -> dict[str, Any]:
    return {
        "policy": POLICY,
        "roots": ["s1"],
        "scopes": [
            {"scope_id": "s0", "operator": inner, "targets": [{"role": "BODY", "target_id": "p0"}], "arguments": [], "source_spans": [[0, len(SOURCE)]]},
            {"scope_id": "s1", "operator": outer, "targets": [{"role": "BODY", "target_id": "s0"}], "arguments": [], "source_spans": [[0, len(SOURCE)]]},
        ],
        "propositions": [proposition("p0", "PRODUCE")],
    }


def compile_and_verify(graph: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    compiled = compile_scoped_semantic_graph(
        graph,
        protected_objects={},
        concept_capsules={},
        source_length=len(SOURCE),
    )
    verification = independently_verify_scoped_semantic_program(
        graph,
        compiled,
        protected_objects={},
        concept_capsules={},
        source_length=len(SOURCE),
    )
    return compiled, verification


def run_gate() -> dict[str, Any]:
    operator_rows = []
    for operator in sorted(SCOPE_OPERATORS):
        compiled, verification = compile_and_verify(graph_for_operator(operator))
        serialization = serialize_kernel_program(compiled["program"])
        replay = deserialize_kernel_program(
            serialization,
            protected_objects={},
            concept_capsules={},
            source_character_length=len(SOURCE),
        )
        operator_rows.append(
            {
                "operator": operator,
                "independent_verification_passes": verification["passes"],
                "serialization_exact": replay["canonical_program"] == compiled["program"],
                "program_sha256": compiled["program"]["program_sha256"],
            }
        )

    interventions = []
    for outer, inner in (
        ("POSSIBILITY", "NEGATION"),
        ("NECESSITY", "NEGATION"),
        ("QUESTION", "NEGATION"),
    ):
        first, first_verify = compile_and_verify(nested_graph(outer, inner))
        second, second_verify = compile_and_verify(nested_graph(inner, outer))
        interventions.append(
            {
                "pair": [outer, inner],
                "both_independently_verified": first_verify["passes"] and second_verify["passes"],
                "causal_identity_distinguished": first["program"]["program_sha256"] != second["program"]["program_sha256"],
            }
        )

    condition = graph_for_operator("CONDITION")
    reversed_condition = copy.deepcopy(condition)
    first_target, second_target = reversed_condition["scopes"][0]["targets"]
    first_target["target_id"], second_target["target_id"] = (
        second_target["target_id"],
        first_target["target_id"],
    )
    condition_a, condition_a_verify = compile_and_verify(condition)
    condition_b, condition_b_verify = compile_and_verify(reversed_condition)
    interventions.append(
        {
            "pair": ["CONDITION_ANTECEDENT", "CONDITION_CONSEQUENT"],
            "both_independently_verified": condition_a_verify["passes"] and condition_b_verify["passes"],
            "causal_identity_distinguished": condition_a["program"]["program_sha256"] != condition_b["program"]["program_sha256"],
        }
    )

    universal = nested_graph("ASSERTION", "NEGATION")
    universal["propositions"][0]["quantifier"] = "FORALL"
    existential = copy.deepcopy(universal)
    existential["propositions"][0]["quantifier"] = "EXISTS"
    universal_compiled, universal_verify = compile_and_verify(universal)
    existential_compiled, existential_verify = compile_and_verify(existential)
    interventions.append(
        {
            "pair": ["QUANTIFIER_FORALL", "QUANTIFIER_EXISTS"],
            "both_independently_verified": universal_verify["passes"] and existential_verify["passes"],
            "causal_identity_distinguished": universal_compiled["program"]["program_sha256"] != existential_compiled["program"]["program_sha256"],
        }
    )

    negative_mutations = []
    base_graph = nested_graph("POSSIBILITY", "NEGATION")
    base_compiled, _ = compile_and_verify(base_graph)
    for name, mutate in (
        ("operator", lambda value: value["program"]["nodes"][-1].update(operator="SCOPE_NECESSITY")),
        ("target_role", lambda value: value["program"]["nodes"][-1]["arguments"][0].update(role="CONTENT")),
        ("root", lambda value: value["program"]["roots"].__setitem__(0, "k0")),
        ("identity_map", lambda value: value["identity_map"].update(s1="k0")),
    ):
        candidate = copy.deepcopy(base_compiled)
        mutate(candidate)
        receipt = independently_verify_scoped_semantic_program(
            base_graph,
            candidate,
            protected_objects={},
            concept_capsules={},
            source_length=len(SOURCE),
        )
        negative_mutations.append({"mutation": name, "rejected": not receipt["passes"]})

    passes = (
        all(row["independent_verification_passes"] and row["serialization_exact"] for row in operator_rows)
        and all(row["both_independently_verified"] and row["causal_identity_distinguished"] for row in interventions)
        and all(row["rejected"] for row in negative_mutations)
    )
    report = {
        "policy": "project_theseus_kerc_scoped_semantic_adequacy_gate_v1",
        "status": "GREEN" if passes else "RED",
        "operator_coverage": operator_rows,
        "intervention_controls": interventions,
        "negative_mutations": negative_mutations,
        "metrics": {
            "operator_count": len(operator_rows),
            "operator_pass_count": sum(row["independent_verification_passes"] and row["serialization_exact"] for row in operator_rows),
            "intervention_count": len(interventions),
            "distinguished_intervention_count": sum(row["causal_identity_distinguished"] for row in interventions),
            "negative_mutation_count": len(negative_mutations),
            "negative_mutation_rejection_count": sum(row["rejected"] for row in negative_mutations),
        },
        "claim_ceiling": "Fixture-backed scoped-semantic ABI, exact serialization, causal intervention, and independent structural replay mechanics only. No semantic parsing, truth, completeness, learned competence, public benchmark, utility, efficiency, SOTA, AGI, or ASI claim.",
        "remaining_wall": "Licensed non-benchmark proposition-level supervision and a learned source-to-Kernel compiler remain absent. PMB remains calibration-only and writes zero training rows.",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        **NO_CHEAT,
    }
    report["report_sha256"] = stable_hash(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/kerc_scoped_semantics_k1k.json")
    args = parser.parse_args()
    report = run_gate()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
