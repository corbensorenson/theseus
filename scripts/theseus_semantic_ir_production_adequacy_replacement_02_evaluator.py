#!/usr/bin/env python3
"""Independently qualify the hidden evaluator for adequacy Task 2 replacement."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import theseus_assistant_p2a as p2a


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPORT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_replacement_02_source.json"
SOURCE_REPORT_SHA256 = "b6a92bd9c8d73eae5f48f1ddf47eac8abce80a9b0dba926bcb18d3df27382680"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_replacement_02_evaluator.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_replacement_evaluator_v1"
EXPECTED_PATH = "business_logic_test.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    args = parser.parse_args()
    report = qualify()
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def qualify() -> dict[str, Any]:
    faults: list[str] = []
    if p2a.sha256_file(SOURCE_REPORT) != SOURCE_REPORT_SHA256:
        faults.append("replacement_source_binding_invalid")
    source_report = p2a.read_json(SOURCE_REPORT)
    if (
        source_report.get("trigger_state") != "GREEN"
        or source_report.get("state") != "SOURCE_PAIR_FROZEN_BEFORE_EVALUATOR_QUALIFICATION"
        or source_report.get("source_pair_admitted") is not True
        or source_report.get("candidate_packet_materialized") is not False
    ):
        faults.append("replacement_source_not_admitted")
    archives = p2a.mapping(source_report.get("archives"))
    parent = archive_sources(p2a.mapping(archives.get("parent"))) if not faults else {}
    target = archive_sources(p2a.mapping(archives.get("target"))) if not faults else {}
    parent_source = parent.get(EXPECTED_PATH, "")
    target_source = target.get(EXPECTED_PATH, "")
    controls: dict[str, dict[str, Any]] = {}

    def observe(name: str, sources: dict[str, str]) -> bool:
        try:
            passed = evaluate(sources, (EXPECTED_PATH,))
            controls[name] = {"passed": passed, "fault": None}
            return passed
        except Exception as exc:
            controls[name] = {
                "passed": False,
                "fault": f"{type(exc).__name__}:{exc}"[:1000],
            }
            return False

    parent_passed = observe("parent", parent)
    target_passed = observe("target", target)
    benign = dict(target)
    benign[EXPECTED_PATH] = target_source + "\n# independent evaluator benign probe\n"
    benign_passed = observe("benign_equivalent", benign)
    mechanism = dict(target)
    mechanism[EXPECTED_PATH] = replace_assert_expected_with_literal(target_source)
    mechanism_passed = observe("required_mechanism_mutation", mechanism)
    dataflow = dict(target)
    dataflow[EXPECTED_PATH] = target_source.replace(
        "expected_discount = self.fixture_ctx.get_expected_fixed_discount_reduction()",
        "expected_discount = 500",
        1,
    )
    dataflow_passed = observe("required_dataflow_mutation", dataflow)
    missing_passed = observe("missing_required_path", {})
    unauthorized = dict(target)
    unauthorized["unauthorized/effect.py"] = "value = 1\n"
    unauthorized_passed = observe("unauthorized_path", unauthorized)
    checks = {
        "parent_negative": parent_passed is False,
        "target_positive": target_passed is True,
        "benign_equivalent_positive": benign_passed is True,
        "required_mechanism_mutation_rejected": mechanism_passed is False,
        "required_dataflow_mutation_rejected": dataflow_passed is False,
        "missing_required_path_rejected": missing_passed is False,
        "unauthorized_path_rejected": unauthorized_passed is False,
    }
    faults.extend(name for name, passed in checks.items() if not passed)
    green = not faults
    return {
        "policy": POLICY,
        "trigger_state": "GREEN" if green else "RED",
        "state": "QUALIFIED_BEFORE_CANDIDATE_PACKET_MATERIALIZATION" if green else "INVALID_EVALUATOR",
        "faults": sorted(set(faults)),
        "source_report": {"path": p2a.rel(SOURCE_REPORT), "sha256": p2a.sha256_file(SOURCE_REPORT)},
        "evaluator_owner": {"path": p2a.rel(Path(__file__).resolve()), "sha256": p2a.sha256_file(Path(__file__).resolve())},
        "opaque_evaluator_id": "semantic-ir-adequacy-evaluator-02r1",
        "selected_source_paths": [EXPECTED_PATH],
        "verification_kind": "independent_ast_dataflow_and_assertion_binding",
        "checks": checks,
        "observations": controls,
        "candidate_packet_materialized": False,
        "candidate_or_model_exposure_authorized": False,
        "counters": {
            "parent_target_evaluator_executions": 7,
            "candidate_or_control_calls": 0,
            "local_model_calls": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "training_rows_written": 0,
            "D1_cases_consumed": 0,
            "D2_cases_consumed": 0,
        },
        "maximum_inference": (
            "A GREEN report establishes only that the replacement evaluator rejects the frozen "
            "parent, accepts the frozen target and a benign variant, and rejects mechanism, "
            "dataflow, missing-path, and unauthorized-path controls. It does not establish "
            "candidate competence, Semantic-IR adequacy or treatment effect, D1, D2, training "
            "value, serving, or book support."
        ),
    }


def evaluate(sources: dict[str, str], expected_paths: tuple[str, ...]) -> bool:
    if set(sources) != set(expected_paths):
        return False
    source = sources.get(EXPECTED_PATH, "")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "test_fixed_amount_discount"
    ]
    if len(methods) != 1:
        return False
    method = methods[0]
    assignments = [
        node
        for node in method.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "expected_discount" for target in node.targets)
    ]
    if len(assignments) != 1 or not is_expected_discount_source(assignments[0].value):
        return False
    matches = [
        call
        for call in (node for node in ast.walk(method) if isinstance(node, ast.Call))
        if is_self_method(call.func, "assertEqual")
        and len(call.args) >= 2
        and is_applied_amount(call.args[0])
    ]
    return len(matches) == 1 and isinstance(matches[0].args[1], ast.Name) and matches[0].args[1].id == "expected_discount"


def is_expected_discount_source(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and not node.args
        and not node.keywords
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_expected_fixed_discount_reduction"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "fixture_ctx"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "self"
    )


def is_self_method(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == name
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def is_applied_amount(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "amount"
        and isinstance(node.value, ast.Subscript)
        and isinstance(node.value.slice, ast.Constant)
        and node.value.slice.value == 0
        and isinstance(node.value.value, ast.Attribute)
        and node.value.value.attr == "applied"
        and isinstance(node.value.value.value, ast.Name)
        and node.value.value.value.id == "discounts_obj"
    )


def replace_assert_expected_with_literal(source: str) -> str:
    tree = ast.parse(source)
    replacement = copy.deepcopy(tree)
    changed = 0
    for node in ast.walk(replacement):
        if (
            isinstance(node, ast.Call)
            and is_self_method(node.func, "assertEqual")
            and len(node.args) >= 2
            and is_applied_amount(node.args[0])
        ):
            node.args[1] = ast.Constant(value=500)
            changed += 1
    if changed != 1:
        raise AssertionError(f"expected one assertion mutation, found {changed}")
    ast.fix_missing_locations(replacement)
    return ast.unparse(replacement) + "\n"


def archive_sources(receipt: dict[str, Any]) -> dict[str, str]:
    path = p2a.resolve(str(receipt.get("path") or ""))
    if not path.is_file() or p2a.sha256_file(path) != receipt.get("sha256"):
        raise AssertionError("archive binding invalid")
    root = PurePosixPath(str(receipt.get("root") or ""))
    result: dict[str, str] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = PurePosixPath(member.name)
            if not member.isfile() or member_path.parent != root or member_path.name == "LICENSE":
                continue
            handle = archive.extractfile(member)
            if handle is not None:
                result[member_path.name] = handle.read().decode("utf-8")
    return result


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report.get("trigger_state"),
        "state": report.get("state"),
        "checks": report.get("checks"),
        "faults": report.get("faults"),
        "counters": report.get("counters"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
