#!/usr/bin/env python3
"""Independently qualify the hidden evaluator for adequacy Task 4 replacement."""

from __future__ import annotations

import argparse
import ast
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import theseus_assistant_p2a as p2a


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPORT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_replacement_04_source.json"
SOURCE_REPORT_SHA256 = "579bb48e49e1ce2c777d97a3e87e2a029fcd51b668e6aee49d26423bfd6a91a4"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_replacement_04_evaluator.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_replacement_04_evaluator_v1"
EXPECTED_PATH = "skbio/alignment/_pair.py"


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
    if not SOURCE_REPORT.is_file() or p2a.sha256_file(SOURCE_REPORT) != SOURCE_REPORT_SHA256:
        faults.append("replacement_source_binding_invalid")
    source_report = p2a.read_json(SOURCE_REPORT) if SOURCE_REPORT.is_file() else {}
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
    target_source = target.get(EXPECTED_PATH, "")
    controls: dict[str, dict[str, Any]] = {}

    def observe(name: str, sources: dict[str, str]) -> bool:
        try:
            passed = evaluate(sources, (EXPECTED_PATH,))
            controls[name] = {"passed": passed, "fault": None}
            return passed
        except Exception as exc:
            controls[name] = {"passed": False, "fault": f"{type(exc).__name__}:{exc}"[:1000]}
            return False

    parent_passed = observe("parent", parent)
    target_passed = observe("target", target)
    benign = dict(target)
    benign[EXPECTED_PATH] = target_source + "\n# independent evaluator benign probe\n"
    benign_passed = observe("benign_equivalent", benign)
    truthiness = dict(target)
    truthiness[EXPECTED_PATH] = target_source.replace("if atol is None:", "if not atol:", 1)
    truthiness_passed = observe("truthiness_mutation", truthiness)
    missing_cast = dict(target)
    missing_cast[EXPECTED_PATH] = target_source.replace("    atol = dtype(atol)\n", "", 1)
    missing_cast_passed = observe("missing_dtype_cast", missing_cast)
    wrong_default = dict(target)
    wrong_default[EXPECTED_PATH] = target_source.replace("        atol = 0.0\n", "        atol = 0.5\n", 1)
    wrong_default_passed = observe("wrong_none_default", wrong_default)
    wrong_dataflow = dict(target)
    wrong_dataflow[EXPECTED_PATH] = target_source.replace(
        "    atol = dtype(atol)\n", "    atol = dtype(0.0)\n", 1
    )
    wrong_dataflow_passed = observe("wrong_dtype_dataflow", wrong_dataflow)
    wrong_order = dict(target)
    block = "    if atol is None:\n        atol = 0.0\n    atol = dtype(atol)\n\n"
    anchor = "    gap_open, gap_extend = prep_gapcost(gap_cost, dtype=dtype)\n"
    wrong_order[EXPECTED_PATH] = target_source.replace(block, "", 1).replace(
        anchor, anchor + block, 1
    )
    wrong_order_passed = observe("wrong_order", wrong_order)
    missing_passed = observe("missing_required_path", {})
    unauthorized = dict(target)
    unauthorized["unauthorized/effect.py"] = "value = 1\n"
    unauthorized_passed = observe("unauthorized_path", unauthorized)
    checks = {
        "parent_negative": parent_passed is False,
        "target_positive": target_passed is True,
        "benign_equivalent_positive": benign_passed is True,
        "truthiness_mutation_rejected": truthiness_passed is False,
        "missing_dtype_cast_rejected": missing_cast_passed is False,
        "wrong_none_default_rejected": wrong_default_passed is False,
        "wrong_dtype_dataflow_rejected": wrong_dataflow_passed is False,
        "wrong_order_rejected": wrong_order_passed is False,
        "missing_required_path_rejected": missing_passed is False,
        "unauthorized_path_rejected": unauthorized_passed is False,
    }
    faults.extend(name for name, passed in checks.items() if not passed)
    green = not faults
    counters = zero_counters()
    counters["parent_target_evaluator_executions"] = len(controls)
    return {
        "policy": POLICY,
        "trigger_state": "GREEN" if green else "RED",
        "state": "QUALIFIED_BEFORE_CANDIDATE_PACKET_MATERIALIZATION" if green else "INVALID_EVALUATOR",
        "faults": sorted(set(faults)),
        "source_report": {"path": p2a.rel(SOURCE_REPORT), "sha256": p2a.sha256_file(SOURCE_REPORT)},
        "evaluator_owner": {
            "path": p2a.rel(Path(__file__).resolve()),
            "sha256": p2a.sha256_file(Path(__file__).resolve()),
        },
        "opaque_evaluator_id": "semantic-ir-adequacy-evaluator-04r1",
        "selected_source_paths": [EXPECTED_PATH],
        "verification_kind": "independent_ast_none_branch_dtype_dataflow_and_order",
        "checks": checks,
        "observations": controls,
        "candidate_packet_materialized": False,
        "hidden_from_candidate_generation": True,
        "counters": counters,
        "maximum_inference": (
            "A GREEN report establishes only that the replacement evaluator rejects the frozen "
            "parent, accepts the frozen target and a benign variant, and rejects truthiness, "
            "default, dtype-dataflow, ordering, missing-path, and unauthorized-path controls. "
            "It does not establish candidate competence, Semantic-IR adequacy or treatment "
            "effect, D1, D2, training value, serving, or book support."
        ),
    }


def evaluate(sources: dict[str, str], expected_paths: tuple[str, ...]) -> bool:
    if set(sources) != set(expected_paths):
        return False
    try:
        tree = ast.parse(sources.get(EXPECTED_PATH, ""))
    except SyntaxError:
        return False
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "pair_align"
    ]
    if len(functions) != 1:
        return False
    body = functions[0].body
    none_branches = [
        (index, node)
        for index, node in enumerate(body)
        if isinstance(node, ast.If) and is_atol_is_none(node.test)
    ]
    if len(none_branches) != 1:
        return False
    branch_index, branch = none_branches[0]
    if len(branch.body) != 1 or not is_assign_constant(branch.body[0], "atol", 0.0) or branch.orelse:
        return False
    casts = [
        (index, node)
        for index, node in enumerate(body)
        if isinstance(node, ast.Assign) and is_atol_dtype_cast(node)
    ]
    gaps = [
        (index, node)
        for index, node in enumerate(body)
        if isinstance(node, ast.Assign) and is_gapcost_assignment(node)
    ]
    if len(casts) != 1 or len(gaps) != 1 or not (branch_index < casts[0][0] < gaps[0][0]):
        return False
    mode_calls = [
        node for node in ast.walk(functions[0]) if isinstance(node, ast.Call) and call_name(node) == "_prep_mode"
    ]
    atol_loads = [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Name) and node.id == "atol" and isinstance(node.ctx, ast.Load)
    ]
    return len(mode_calls) == 1 and len(atol_loads) >= 3


def is_atol_is_none(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "atol"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Is)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value is None
    )


def assignment_name(node: ast.Assign) -> str | None:
    return node.targets[0].id if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) else None


def is_assign_constant(node: ast.stmt, name: str, value: float) -> bool:
    return (
        isinstance(node, ast.Assign)
        and assignment_name(node) == name
        and isinstance(node.value, ast.Constant)
        and node.value.value == value
    )


def is_atol_dtype_cast(node: ast.Assign) -> bool:
    return (
        assignment_name(node) == "atol"
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "dtype"
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == "atol"
        and not node.value.keywords
    )


def is_gapcost_assignment(node: ast.Assign) -> bool:
    return (
        len(node.targets) == 1
        and isinstance(node.targets[0], ast.Tuple)
        and [item.id for item in node.targets[0].elts if isinstance(item, ast.Name)]
        == ["gap_open", "gap_extend"]
        and isinstance(node.value, ast.Call)
        and call_name(node.value) == "prep_gapcost"
    )


def call_name(node: ast.Call) -> str | None:
    return node.func.id if isinstance(node.func, ast.Name) else None


def archive_sources(receipt: dict[str, Any]) -> dict[str, str]:
    path = p2a.resolve(str(receipt.get("path") or ""))
    if not path.is_file() or p2a.sha256_file(path) != receipt.get("sha256"):
        raise AssertionError("archive binding invalid")
    root = PurePosixPath(str(receipt.get("root") or ""))
    result: dict[str, str] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = PurePosixPath(member.name)
            if not member.isfile() or not member_path.is_relative_to(root):
                continue
            relative = member_path.relative_to(root)
            if str(relative) == "LICENSE.txt":
                continue
            handle = archive.extractfile(member)
            if handle is not None:
                result[str(relative)] = handle.read().decode("utf-8")
    return result


def zero_counters() -> dict[str, int]:
    return {
        "D1_cases_consumed": 0,
        "D2_cases_consumed": 0,
        "candidate_or_control_calls": 0,
        "external_inference_calls": 0,
        "hidden_evaluator_executions": 0,
        "local_model_calls": 0,
        "network_source_calls": 0,
        "parent_target_evaluator_executions": 0,
        "teacher_calls": 0,
        "training_rows_written": 0,
    }


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
