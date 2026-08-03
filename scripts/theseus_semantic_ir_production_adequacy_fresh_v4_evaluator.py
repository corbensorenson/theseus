#!/usr/bin/env python3
"""Independently qualify hidden evaluators for fresh adequacy tasks 1-4."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import tarfile
from datetime import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import theseus_assistant_p2a as p2a


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPORT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v4_sources.json"
SOURCE_REPORT_SHA256 = "fb8bbe4446dec871db53c04aae0e83a53057df39988e102c707dc9ac27496b37"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v4_evaluator.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_fresh_v4_evaluator_v1"


class AnnotationStripper(ast.NodeTransformer):
    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.annotation = None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.returns = None
        node.decorator_list = []
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        node.decorator_list = []
        return node


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
        or source_report.get("state") != "FOUR_SOURCE_PAIRS_FROZEN_BEFORE_EVALUATOR_QUALIFICATION"
        or source_report.get("source_pairs_admitted") is not True
        or source_report.get("candidate_packet_materialized") is not False
    ):
        faults.append("source_report_not_admitted")
    rows: list[dict[str, Any]] = []
    if not faults:
        for row in p2a.dicts(source_report.get("rows")):
            result = qualify_row(row)
            rows.append(result)
            faults.extend(f"task_{int(row.get('index') or 0):02d}:{fault}" for fault in p2a.strings(result.get("faults")))
    green = not faults and len(rows) == 4 and all(row.get("trigger_state") == "GREEN" for row in rows)
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if green else "RED",
        "state": "QUALIFIED_BEFORE_FRESH_V4_CANDIDATE_PACKET_MATERIALIZATION" if green else "INVALID_EVALUATOR",
        "source_report": artifact(SOURCE_REPORT),
        "evaluator_owner": artifact(Path(__file__).resolve()),
        "qualification_method": "Independent AST extraction plus dependency-stubbed behavior for each named causal slice; no upstream test, target byte equality, or candidate-visible oracle is used.",
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
            "D2_cases_consumed": 0
        },
        "maximum_inference": "A GREEN report establishes only that the four source-disjoint evaluators distinguish parent, target, benign, mechanism-loss, missing-path, and unauthorized-path controls before candidate packet creation. It provides no candidate, implementation-adequacy, subsystem-effect, D1, D2, training, serving, or book-support evidence."
    }


def qualify_row(row: dict[str, Any]) -> dict[str, Any]:
    index = int(row.get("index") or 0)
    selected = tuple(p2a.strings(row.get("selected_source_paths")))
    archives = p2a.mapping(row.get("archives"))
    parent = archive_sources(p2a.mapping(archives.get("parent")), selected)
    target = archive_sources(p2a.mapping(archives.get("target")), selected)
    observations: dict[str, dict[str, Any]] = {}

    def observe(name: str, sources: dict[str, str]) -> bool:
        try:
            passed = evaluate(index, sources, selected)
            observations[name] = {"passed": passed, "fault": None}
            return passed
        except Exception as exc:
            observations[name] = {"passed": False, "fault": f"{type(exc).__name__}:{exc}"[:1000]}
            return False

    parent_passed = observe("parent", parent)
    target_passed = observe("target", target)
    benign = dict(target)
    benign[selected[0]] += "\n# independent evaluator benign-equivalence probe\n"
    benign_passed = observe("benign_equivalent", benign)
    mutation_passed = observe("required_mechanism_mutation", mutate_required_mechanism(index, target, selected[0]))
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
        "opaque_evaluator_id": f"semantic-ir-adequacy-evaluator-{index:02d}r2",
        "selected_source_paths": list(selected),
        "verification_kind": "independent_causal_slice_behavior_and_ast",
        "trigger_state": "GREEN" if not faults else "RED",
        "checks": checks,
        "observations": observations,
        "faults": faults,
    }


def evaluate(index: int, sources: dict[str, str], expected_paths: tuple[str, ...]) -> bool:
    if set(sources) != set(expected_paths):
        return False
    try:
        tree = ast.parse(sources[expected_paths[0]])
    except SyntaxError:
        return False
    return bool({1: evaluate_sklearn, 2: evaluate_django, 3: evaluate_networkx, 4: evaluate_black}.get(index, lambda _: False)(tree))


def function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise AssertionError(f"expected one function {name}, found {len(matches)}")
    node = AnnotationStripper().visit(copy.deepcopy(matches[0]))
    ast.fix_missing_locations(node)
    return node


def evaluate_sklearn(tree: ast.AST) -> bool:
    node = function_node(tree, "_make_indexable")
    namespace: dict[str, Any] = {
        "sp": SimpleNamespace(issparse=lambda value: getattr(value, "sparse", False)),
        "_nw_is_into_df_or_series": lambda value: getattr(value, "dataframe", False),
        "nw": SimpleNamespace(from_native=lambda value, allow_series=False: ("wrapped", value, allow_series)),
        "np": SimpleNamespace(array=lambda value: ("array", tuple(value))),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<sklearn-evaluator>", "exec"), namespace)
    function = namespace["_make_indexable"]
    sparse = SimpleNamespace(sparse=True, tocsr=lambda: "csr")
    dataframe = SimpleNamespace(dataframe=True)
    indexable = {"a": 1}
    return (
        function(sparse) == "csr"
        and function(dataframe) is dataframe
        and function(indexable) is indexable
        and function(None) is None
        and function(iter([1, 2])) == ("array", (1, 2))
    )


def evaluate_django(tree: ast.AST) -> bool:
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "OFTTime"]
    if len(matches) != 1:
        return False
    class DummyField:
        def as_datetime(self):
            if getattr(self, "error", None) is not None:
                raise self.error
            return self.payload
    class GDALException(Exception):
        pass
    node = AnnotationStripper().visit(copy.deepcopy(matches[0]))
    node.bases = [ast.Name(id="DummyField", ctx=ast.Load())]
    ast.fix_missing_locations(node)
    namespace: dict[str, Any] = {"DummyField": DummyField, "GDALException": GDALException, "time": time}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<django-evaluator>", "exec"), namespace)
    cls = namespace["OFTTime"]
    values = [SimpleNamespace(value=value) for value in (0, 0, 0, 12, 34, 5.25, 0)]
    valid = cls()
    valid.payload = values
    valid.error = None
    if valid.value() != time(12, 34, 5, 250000):
        return False
    for error in (TypeError("null"), ValueError("bad"), GDALException("bad")):
        item = cls()
        item.error = error
        if item.value() is not None:
            return False
    return True


def evaluate_networkx(tree: ast.AST) -> bool:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "reverse"]
    if len(matches) != 1:
        return False
    calls = [decorator for decorator in matches[0].decorator_list if isinstance(decorator, ast.Call)]
    dispatch = [call for call in calls if isinstance(call.func, ast.Attribute) and call.func.attr == "_dispatchable"]
    if len(dispatch) != 1:
        return False
    keywords = {keyword.arg: keyword.value for keyword in dispatch[0].keywords if keyword.arg}
    return (
        isinstance(keywords.get("preserve_all_attrs"), ast.Constant)
        and keywords["preserve_all_attrs"].value is True
        and isinstance(keywords.get("returns_graph"), ast.Constant)
        and keywords["returns_graph"].value is True
    )


def evaluate_black(tree: ast.AST) -> bool:
    node = function_node(tree, "is_docstring")
    class Leaf:
        def __init__(self, value: str, node_type: int = 1):
            self.value = value
            self.type = node_type
            grandparent = SimpleNamespace(type=10)
            self.parent = SimpleNamespace(type=11, prev_sibling=None, parent=grandparent)
    token = SimpleNamespace(STRING=1, NEWLINE=2, INDENT=3, COLON=4)
    syms = SimpleNamespace(simple_stmt=11, file_input=10, parameters=12)
    def get_string_prefix(value: str) -> str:
        return value.split('"', 1)[0].split("'", 1)[0]
    namespace: dict[str, Any] = {
        "Leaf": Leaf,
        "token": token,
        "syms": syms,
        "get_string_prefix": get_string_prefix,
        "prev_siblings_are": lambda *_: False,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<black-evaluator>", "exec"), namespace)
    function = namespace["is_docstring"]
    return (
        function(Leaf('"ordinary"')) is True
        and function(Leaf('r"ordinary"')) is True
        and function(Leaf('b"bytes"')) is False
        and function(Leaf('F"formatted"')) is False
        and function(Leaf('t"template"')) is False
        and function(Leaf('T"template"')) is False
    )


def mutate_required_mechanism(index: int, sources: dict[str, str], path: str) -> dict[str, str]:
    mutated = dict(sources)
    source = mutated[path]
    replacements = {
        1: ("return iterable\n    elif hasattr(iterable, \"__getitem__\"):", "return nw.from_native(iterable, allow_series=True)\n    elif hasattr(iterable, \"__getitem__\"):") ,
        2: ("except (TypeError, ValueError, GDALException):\n            return None\n\n\nclass OFTInteger64", "except (ValueError, GDALException):\n            return None\n\n\nclass OFTInteger64"),
        3: ("@nx._dispatchable(preserve_all_attrs=True, returns_graph=True)", "@nx._dispatchable(returns_graph=True)"),
        4: ('intersection("bBfFtT")', 'intersection("bBfF")'),
    }
    old, new = replacements[index]
    if old not in source:
        raise AssertionError(f"mechanism mutation anchor absent for task {index}")
    mutated[path] = source.replace(old, new, 1)
    return mutated


def archive_sources(receipt: dict[str, Any], selected: tuple[str, ...]) -> dict[str, str]:
    path = p2a.resolve(str(receipt.get("path") or ""))
    if not path.is_file() or p2a.sha256_file(path) != receipt.get("sha256"):
        raise AssertionError("archive binding invalid")
    root = PurePosixPath(str(receipt.get("root") or ""))
    result: dict[str, str] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = PurePosixPath(member.name)
            if not member.isfile() or member_path.parent == root and member_path.name not in {PurePosixPath(value).name for value in selected}:
                continue
            relative = str(member_path.relative_to(root)) if root in member_path.parents else ""
            if relative not in selected:
                continue
            handle = archive.extractfile(member)
            if handle is not None:
                result[relative] = handle.read().decode("utf-8")
    return result


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
