#!/usr/bin/env python3
"""Independently qualify four v6 evaluators before candidate packet creation."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import re
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_semantic_ir_production_adequacy_fresh_v4_evaluator as archive_owner


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPORT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v6_sources.json"
SOURCE_REPORT_SHA256 = "229af6cd74682117be4f50b179620ad79c1c070e0b63eca37a2a881b991b1403"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v6_evaluator.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_fresh_v6_evaluator_v1"


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
        faults.append("source_report_binding_invalid")
    source_report = p2a.read_json(SOURCE_REPORT)
    if (
        source_report.get("trigger_state") != "GREEN"
        or source_report.get("state")
        != "FOUR_SOURCE_PAIRS_FROZEN_BEFORE_V6_EVALUATOR_QUALIFICATION"
        or source_report.get("source_pairs_admitted") is not True
        or source_report.get("candidate_packet_materialized") is not False
    ):
        faults.append("source_report_not_admitted")
    rows: list[dict[str, Any]] = []
    if not faults:
        for source_row in p2a.dicts(source_report.get("rows")):
            row = qualify_row(source_row)
            rows.append(row)
            faults.extend(
                f"task_{int(source_row.get('index') or 0):02d}:{fault}"
                for fault in p2a.strings(row.get("faults"))
            )
    green = (
        not faults
        and len(rows) == 4
        and all(row.get("trigger_state") == "GREEN" for row in rows)
    )
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if green else "RED",
        "state": "QUALIFIED_BEFORE_FRESH_V6_CANDIDATE_PACKET_MATERIALIZATION"
        if green
        else "INVALID_EVALUATOR",
        "source_report": artifact(SOURCE_REPORT),
        "evaluator_owner": artifact(Path(__file__).resolve()),
        "qualification_method": "Independent AST and dependency-stubbed causal behavior over the named source slice; upstream tests, target byte equality, and candidate-visible answer metadata are not used.",
        "task_count": len(rows),
        "green_task_count": sum(row.get("trigger_state") == "GREEN" for row in rows),
        "candidate_packet_materialized": False,
        "candidate_or_model_exposure_authorized": False,
        "rows": rows,
        "faults": sorted(set(faults)),
        "counters": {
            "parent_target_evaluator_executions": len(rows) * 2,
            "evaluator_control_executions": len(rows) * 4,
            "candidate_or_control_calls": 0,
            "local_model_calls": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "training_rows_written": 0,
            "D1_cases_consumed": 0,
            "D2_cases_consumed": 0,
        },
        "maximum_inference": "A GREEN report establishes only that four independent evaluators distinguish parent, target, benign, mechanism-loss, missing-path, and unauthorized-path controls before packet creation. It provides no candidate, implementation-adequacy, subsystem-effect, D1, D2, training, serving, or book-support evidence.",
    }


def qualify_row(row: dict[str, Any]) -> dict[str, Any]:
    index = int(row.get("index") or 0)
    selected = tuple(p2a.strings(row.get("selected_source_paths")))
    archives = p2a.mapping(row.get("archives"))
    parent = archive_owner.archive_sources(
        p2a.mapping(archives.get("parent")), selected
    )
    target = archive_owner.archive_sources(
        p2a.mapping(archives.get("target")), selected
    )
    observations: dict[str, dict[str, Any]] = {}

    def observe(name: str, sources: dict[str, str]) -> bool:
        try:
            passed = evaluate(index, sources, selected)
            observations[name] = {"passed": passed, "fault": None}
            return passed
        except Exception as exc:
            observations[name] = {
                "passed": False,
                "fault": f"{type(exc).__name__}:{exc}"[:1000],
            }
            return False

    parent_passed = observe("parent", parent)
    target_passed = observe("target", target)
    benign = dict(target)
    benign[selected[0]] += "\n# independent evaluator benign-equivalence probe\n"
    benign_passed = observe("benign_equivalent", benign)
    mutation_passed = observe(
        "required_mechanism_mutation",
        mutate_required_mechanism(index, target, selected[0]),
    )
    missing_passed = observe("missing_required_path", {})
    unauthorized = dict(target)
    unauthorized["unauthorized/effect.py"] = "value = 1\n"
    unauthorized_passed = observe("unauthorized_path", unauthorized)
    checks = {
        "parent_negative": parent_passed is False,
        "target_positive": target_passed is True,
        "benign_equivalent_positive": benign_passed is True,
        "required_mechanism_mutation_rejected": mutation_passed is False,
        "missing_required_path_rejected": missing_passed is False,
        "unauthorized_path_rejected": unauthorized_passed is False,
    }
    faults = [name for name, passed in checks.items() if not passed]
    return {
        "index": index,
        "opaque_evaluator_id": f"semantic-ir-adequacy-evaluator-{index:02d}r4",
        "selected_source_paths": list(selected),
        "verification_kind": "independent_causal_slice_ast_and_behavior",
        "trigger_state": "GREEN" if not faults else "RED",
        "checks": checks,
        "observations": observations,
        "faults": faults,
    }


def evaluate(
    index: int, sources: dict[str, str], expected_paths: tuple[str, ...]
) -> bool:
    if len(expected_paths) != 1 or set(sources) != set(expected_paths):
        return False
    try:
        tree = ast.parse(sources[expected_paths[0]])
    except SyntaxError:
        return False
    evaluator = {
        1: evaluate_lightllm,
        2: evaluate_translation_finder,
        3: evaluate_feu,
        4: evaluate_statsmodels,
    }.get(index)
    return bool(evaluator and evaluator(tree))


def evaluate_lightllm(tree: ast.AST) -> bool:
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "select_experts_and_quant_input"
    ]
    if len(functions) != 1:
        return False
    assignments = [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "_select_experts"
    ]
    if len(assignments) != 1 or len(assignments[0].targets) != 1:
        return False
    target = assignments[0].targets[0]
    if not isinstance(target, ast.Tuple) or len(target.elts) != 3:
        return False
    return [elt.id for elt in target.elts if isinstance(elt, ast.Name)] == [
        "topk_weights",
        "topk_idx",
        "_",
    ]


def evaluate_translation_finder(tree: ast.AST) -> bool:
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "LARAVEL_BYTES_RE"
            for target in node.targets
        )
    ]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Call):
        return False
    call = assignments[0].value
    if not call.args or not isinstance(call.args[0], ast.Constant):
        return False
    pattern = call.args[0].value
    if not isinstance(pattern, bytes):
        return False
    if not pattern.startswith(b"^(?>") or b"[^\\n]" not in pattern:
        return False
    namespace = {"re": re}
    expression = ast.Expression(body=copy.deepcopy(call))
    ast.fix_missing_locations(expression)
    compiled = eval(compile(expression, "<translation-finder-evaluator>", "eval"), namespace)
    cases = (
        (b"'apples' => 'one|many'", True),
        (b"'key|context' => 'value'", False),
        (b"'key' => 'value'\n'one|many'", False),
        (b"'key' =>\n'one|many'", False),
        (b"return [" + b"=>" * 64_000, False),
    )
    return all((compiled.search(value) is not None) is expected for value, expected in cases)


def evaluate_feu(tree: ast.AST) -> bool:
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_release_date"
    ]
    if len(functions) != 1:
        return False
    normalizers = [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fromisoformat"
        and node.args
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Attribute)
        and node.args[0].func.attr == "replace"
        and len(node.args[0].args) == 2
        and all(isinstance(argument, ast.Constant) for argument in node.args[0].args)
        and [argument.value for argument in node.args[0].args] == ["Z", "+00:00"]
    ]
    if len(normalizers) != 1:
        return False
    node = archive_owner.AnnotationStripper().visit(copy.deepcopy(functions[0]))
    ast.fix_missing_locations(node)
    namespace: dict[str, Any] = {"date": date, "datetime": datetime}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<feu-evaluator>", "exec"), namespace)
    function = namespace["_release_date"]
    values = [
        {"upload_time_iso_8601": "2026-08-03T05:00:00Z"},
        {"upload_time_iso_8601": "2026-08-02T04:00:00Z"},
    ]
    return (
        function(None) is None
        and function([]) is None
        and function(values) == date(2026, 8, 2)
    )


def evaluate_statsmodels(tree: ast.AST) -> bool:
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "multipletests"
    ]
    if len(functions) != 1:
        return False
    body = functions[0].body
    ntests_index = next(
        (
            index
            for index, node in enumerate(body)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "ntests" for target in node.targets)
        ),
        -1,
    )
    if ntests_index < 0 or ntests_index + 1 >= len(body):
        return False
    ntests = copy.deepcopy(body[ntests_index])
    branch = copy.deepcopy(body[ntests_index + 1])
    if not isinstance(branch, ast.If) or not branch.orelse:
        return False
    probe = ast.FunctionDef(
        name="probe",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="pvals"), ast.arg(arg="alphaf")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            ntests,
            branch,
            ast.Return(
                value=ast.Tuple(
                    elts=[
                        ast.Name(id="alphacSidak", ctx=ast.Load()),
                        ast.Name(id="alphacBonf", ctx=ast.Load()),
                    ],
                    ctx=ast.Load(),
                )
            ),
        ],
        decorator_list=[],
    )
    ast.fix_missing_locations(probe)
    np_stub = SimpleNamespace(power=pow, nan=math.nan)
    namespace: dict[str, Any] = {"np": np_stub}
    exec(compile(ast.Module(body=[probe], type_ignores=[]), "<statsmodels-evaluator>", "exec"), namespace)
    empty_sidak, empty_bonf = namespace["probe"]([], 0.05)
    sidak, bonf = namespace["probe"]([0.1, 0.2], 0.05)
    return (
        math.isnan(empty_sidak)
        and math.isnan(empty_bonf)
        and math.isclose(sidak, 1 - (1 - 0.05) ** 0.5)
        and math.isclose(bonf, 0.025)
    )


def mutate_required_mechanism(
    index: int, sources: dict[str, str], path: str
) -> dict[str, str]:
    mutated = dict(sources)
    replacements = {
        1: (
            "topk_weights, topk_idx, _ = self._select_experts(",
            "topk_weights, topk_idx = self._select_experts(",
        ),
        2: ("rb\"^(?>", "rb\"^(?:"),
        3: (
            "datetime.fromisoformat(t.replace(\"Z\", \"+00:00\"))",
            "datetime.fromisoformat(t)",
        ),
        4: ("if ntests > 0:", "if ntests >= 0:"),
    }
    old, new = replacements[index]
    if index == 1:
        head, separator, tail = mutated[path].rpartition(old)
        mutated[path] = head + (new if separator else "") + tail
    else:
        mutated[path] = mutated[path].replace(old, new, 1)
    return mutated


def artifact(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report.get("trigger_state"),
        "state": report.get("state"),
        "green_task_count": report.get("green_task_count"),
        "faults": report.get("faults"),
        "counters": report.get("counters"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
