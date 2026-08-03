#!/usr/bin/env python3
"""Independently qualify the hidden evaluator for fresh v5 Task 1."""

from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_semantic_ir_production_adequacy_fresh_v4_evaluator as v4


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPORT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v5_sources.json"
SOURCE_REPORT_SHA256 = "af9b65f34a7f153f5491e5e1ed4550b7a2616d68b5040084e008ed44784085e9"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v5_evaluator.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_fresh_v5_evaluator_v1"


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
        or source_report.get("state") != "ONE_SOURCE_PAIR_FROZEN_BEFORE_V5_EVALUATOR_QUALIFICATION"
        or source_report.get("source_pairs_admitted") is not True
        or source_report.get("candidate_packet_materialized") is not False
    ):
        faults.append("source_report_not_admitted")
    rows: list[dict[str, Any]] = []
    if not faults:
        source_rows = p2a.dicts(source_report.get("rows"))
        if len(source_rows) != 1 or int(source_rows[0].get("index") or 0) != 1:
            faults.append("source_row_identity_invalid")
        else:
            result = qualify_row(source_rows[0])
            rows.append(result)
            faults.extend(f"task_01:{fault}" for fault in p2a.strings(result.get("faults")))
    green = not faults and len(rows) == 1 and rows[0].get("trigger_state") == "GREEN"
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if green else "RED",
        "state": "QUALIFIED_BEFORE_FRESH_V5_CANDIDATE_PACKET_MATERIALIZATION" if green else "INVALID_EVALUATOR",
        "source_report": artifact(SOURCE_REPORT),
        "evaluator_owner": artifact(Path(__file__).resolve()),
        "qualification_method": "Independent AST extraction plus dependency-stubbed async behavior for the named exception-translation slice; no upstream test, target byte equality, or candidate-visible oracle is used.",
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
        "maximum_inference": "A GREEN report establishes only that the replacement evaluator distinguishes parent, target, benign, mechanism-loss, missing-path, and unauthorized-path controls before candidate packet creation. It provides no candidate, implementation-adequacy, subsystem-effect, D1, D2, training, serving, or book-support evidence."
    }


def qualify_row(row: dict[str, Any]) -> dict[str, Any]:
    selected = tuple(p2a.strings(row.get("selected_source_paths")))
    archives = p2a.mapping(row.get("archives"))
    parent = v4.archive_sources(p2a.mapping(archives.get("parent")), selected)
    target = v4.archive_sources(p2a.mapping(archives.get("target")), selected)
    observations: dict[str, dict[str, Any]] = {}

    def observe(name: str, sources: dict[str, str]) -> bool:
        try:
            passed = evaluate(sources, selected)
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
    mutation = dict(target)
    mutation[selected[0]] = mutation[selected[0]].replace(
        "except (ClientError, TimeoutError) as ex:",
        "except ClientError as ex:",
        1,
    )
    mutation_passed = observe("required_mechanism_mutation", mutation)
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
        "index": 1,
        "opaque_evaluator_id": "semantic-ir-adequacy-evaluator-01r3",
        "selected_source_paths": list(selected),
        "verification_kind": "independent_exception_translation_behavior_and_ast",
        "trigger_state": "GREEN" if not faults else "RED",
        "checks": checks,
        "observations": observations,
        "faults": faults,
    }


def evaluate(sources: dict[str, str], expected_paths: tuple[str, ...]) -> bool:
    if set(sources) != set(expected_paths) or len(expected_paths) != 1:
        return False
    try:
        tree = ast.parse(sources[expected_paths[0]])
    except SyntaxError:
        return False
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "_send_prepared_command"]
    if len(matches) != 1:
        return False
    node = copy.deepcopy(matches[0])
    node.decorator_list = []
    node.returns = None
    for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
        argument.annotation = None
    ast.fix_missing_locations(node)

    class ClientError(Exception):
        pass

    class NotConnectedError(Exception):
        pass

    class RPCError(Exception):
        pass

    class Logger:
        @staticmethod
        def debug(*_args: Any, **_kwargs: Any) -> None:
            return None

    namespace: dict[str, Any] = {
        "Any": Any,
        "ClientError": ClientError,
        "TimeoutError": TimeoutError,
        "NotConnectedError": NotConnectedError,
        "RPCError": RPCError,
        "_LOGGER": Logger(),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<pytboss-evaluator>", "exec"), namespace)
    function = namespace["_send_prepared_command"]

    class Response:
        status = 200

        async def json(self, **_kwargs: Any) -> dict[str, int]:
            return {"id": 7}

    class Context:
        async def __aenter__(self) -> Response:
            return Response()

        async def __aexit__(self, *_args: Any) -> None:
            return None

    class Session:
        closed = False

        def __init__(self, error: BaseException | None = None) -> None:
            self.error = error

        def post(self, *_args: Any, **_kwargs: Any) -> Context:
            if self.error is not None:
                raise self.error
            return Context()

    class Owner:
        def __init__(self, error: BaseException | None = None) -> None:
            self._session = Session(error)
            self._url = "http://grill.invalid"
            self._timeout = 0.1
            self._connected = True
            self.observed: list[dict[str, int]] = []

        async def _on_command_response(self, payload: dict[str, int]) -> None:
            self.observed.append(payload)

    async def exercise() -> bool:
        for error in (ClientError("refused"), TimeoutError("silent")):
            owner = Owner(error)
            try:
                await function(owner, {"id": 7})
            except NotConnectedError as exc:
                if owner._connected is not False or exc.__cause__ is not error:
                    return False
            except BaseException:
                return False
            else:
                return False
        success = Owner()
        await function(success, {"id": 7})
        return success._connected is True and success.observed == [{"id": 7}]

    return asyncio.run(exercise())


def artifact(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report.get("trigger_state"),
        "state": report.get("state"),
        "task_count": report.get("task_count"),
        "green_task_count": report.get("green_task_count"),
        "faults": report.get("faults"),
        "counters": report.get("counters"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
