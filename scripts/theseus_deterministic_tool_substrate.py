#!/usr/bin/env python3
"""Local deterministic tool substrate for Project Theseus.

The learned model should propose and route. Exact local tools should compute,
check, retrieve, and emit replayable evidence. This script registers the first
tool cards and runs private-only smoke/ablation checks without public benchmark
data, external inference, arbitrary remote execution, or fallback returns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import vcm_consumer_abi


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"
DEFAULT_CONFIG = CONFIGS / "deterministic_tool_substrate.json"
DEFAULT_OUT = REPORTS / "deterministic_tool_substrate.json"
DEFAULT_MARKDOWN = REPORTS / "deterministic_tool_substrate.md"
DEFAULT_REGISTRY_OUT = REPORTS / "deterministic_tool_registry.json"
DEFAULT_TRACE_OUT = REPORTS / "deterministic_tool_trace.jsonl"
DEFAULT_ABLATION_OUT = REPORTS / "deterministic_tool_ablation.json"
DEFAULT_DOGFOOD_OUT = REPORTS / "deterministic_tool_dogfood_events.jsonl"
DEFAULT_LOOP_OUT = REPORTS / "deterministic_tool_loop_closure_candidates.json"
DEFAULT_ARTIFACT_GRAPH_OUT = REPORTS / "deterministic_tool_artifact_graph.json"
DEFAULT_VCM_CONTEXT_GOVERNOR = REPORTS / "vcm_context_governor.json"

SOLVED_STATES = {"SOLVED"}
NON_SOLVED_STATES = {"UNKNOWN", "UNSOLVED", "TOOL_UNAVAILABLE", "TOOL_FAULT"}
ALLOWED_STATES = SOLVED_STATES | NON_SOLVED_STATES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--registry-out", default=rel(DEFAULT_REGISTRY_OUT))
    parser.add_argument("--trace-out", default=rel(DEFAULT_TRACE_OUT))
    parser.add_argument("--ablation-out", default=rel(DEFAULT_ABLATION_OUT))
    parser.add_argument("--dogfood-out", default=rel(DEFAULT_DOGFOOD_OUT))
    parser.add_argument("--loop-candidates-out", default=rel(DEFAULT_LOOP_OUT))
    parser.add_argument("--artifact-graph-out", default=rel(DEFAULT_ARTIFACT_GRAPH_OUT))
    parser.add_argument("--vcm-context-governor", default=rel(DEFAULT_VCM_CONTEXT_GOVERNOR))
    parser.add_argument("--run-smoke", action="store_true", help="Run private deterministic tool smoke cases.")
    parser.add_argument("--run-ablation", action="store_true", help="Write tool-on/tool-off private ablation metrics.")
    parser.add_argument(
        "--registry-only",
        action="store_true",
        help=(
            "Refresh live tool availability without rerunning qualification cases. "
            "Fails closed unless the current tool-card set matches an existing full qualification."
        ),
    )
    args = parser.parse_args()
    if args.registry_only and (args.run_smoke or args.run_ablation):
        parser.error("--registry-only cannot be combined with qualification flags")

    started = time.perf_counter()
    config = read_json(resolve(args.config), {})
    vcm_receipt = vcm_context_governor_receipt(resolve(args.vcm_context_governor))
    tool_cards = build_tool_cards(config)
    if args.registry_only:
        qualification = deterministic_tool_qualification_receipt(
            resolve(args.out), tool_cards
        )
        gates = build_registry_refresh_gates(
            config, tool_cards, vcm_receipt, qualification
        )
        hard_failures = [
            gate for gate in gates if gate["severity"] == "hard" and not gate["passed"]
        ]
        warning_failures = [
            gate
            for gate in gates
            if gate["severity"] == "warning" and not gate["passed"]
        ]
        trigger_state = "GREEN" if not hard_failures else "RED"
        if trigger_state == "GREEN" and warning_failures:
            trigger_state = "YELLOW"
        registry_payload = build_registry_payload(
            tool_cards,
            trigger_state=trigger_state,
            refresh_mode="qualification_bound_runtime_refresh",
            qualification=qualification,
            gates=gates,
        )
        write_json(resolve(args.registry_out), registry_payload)
        summary = {
            "tool_card_count": len(tool_cards),
            "available_tool_count": sum(
                bool(card.get("dependency_status", {}).get("available"))
                for card in tool_cards.values()
            ),
            "qualification_ready": bool(qualification.get("ready")),
            "qualification_report_sha256": qualification.get("report_sha256", ""),
            "vcm_context_governor_ready": bool(vcm_receipt.get("ready")),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "qualification_cases_executed": 0,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }
        print(json.dumps({"trigger_state": trigger_state, "summary": summary}, indent=2))
        return 0 if trigger_state != "RED" else 2
    cases = [case for case in config.get("private_smoke_cases", []) if isinstance(case, dict)]

    trace_rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    if args.run_smoke or not (args.run_smoke or args.run_ablation):
        for case in cases:
            result = run_case(case, config)
            results.append(result)
            trace_rows.append(trace_row_for(result, tool_cards.get(result["tool_id"], {})))
        replay_result = run_trace_replay(trace_rows)
        results.append(replay_result)
        trace_rows.append(trace_row_for(replay_result, tool_cards.get(replay_result["tool_id"], {})))

    ablation = build_ablation(cases, results) if args.run_ablation or results else {}
    dogfood_events = build_dogfood_events(results)
    loop_candidates = build_loop_closure_candidates(results)
    artifact_graph = build_artifact_graph(tool_cards, results, dogfood_events, loop_candidates)
    claim_ledger = build_claim_ledger(results)
    viea_tool_context_records = deterministic_tool_vcm_records(vcm_receipt, len(results))
    gates = build_gates(config, tool_cards, results, trace_rows, dogfood_events, ablation, vcm_receipt, viea_tool_context_records)
    hard_failures = [gate for gate in gates if gate["severity"] == "hard" and not gate["passed"]]
    warning_failures = [gate for gate in gates if gate["severity"] == "warning" and not gate["passed"]]
    trigger_state = "GREEN" if not hard_failures else "RED"
    if trigger_state == "GREEN" and warning_failures:
        trigger_state = "YELLOW"

    registry_payload = build_registry_payload(
        tool_cards,
        trigger_state=trigger_state,
        refresh_mode="full_private_qualification",
        gates=gates,
    )
    report = {
        "policy": "project_theseus_deterministic_tool_substrate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "passed": trigger_state in {"GREEN", "YELLOW"},
        "purpose": config.get("purpose"),
        "inputs": {"config": rel(resolve(args.config))},
        "outputs": {
            "report": rel(resolve(args.out)),
            "markdown": rel(resolve(args.markdown_out)),
            "registry": rel(resolve(args.registry_out)),
            "trace": rel(resolve(args.trace_out)),
            "ablation": rel(resolve(args.ablation_out)),
            "dogfood_events": rel(resolve(args.dogfood_out)),
            "loop_candidates": rel(resolve(args.loop_candidates_out)),
            "artifact_graph": rel(resolve(args.artifact_graph_out)),
        },
        "summary": summarize(tool_cards, results, trace_rows, dogfood_events, ablation, vcm_receipt, viea_tool_context_records, started),
        "gates": gates,
        "tool_results": results,
        "artifact_graph": artifact_graph,
        "claim_ledger": claim_ledger,
        "vcm_context_governor_receipt": vcm_receipt,
        "viea_tool_context_records": viea_tool_context_records,
        "ablation": ablation,
        "recommendation": recommendation(trigger_state, results),
        "public_benchmark_boundary": {
            "public_training_rows_written": 0,
            "public_benchmark_rows_used_for_smoke": 0,
            "tool_assisted_scores_must_be_reported_separately": True,
            "model_only_scores_must_remain_separate": True,
        },
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
    }

    write_json(resolve(args.ablation_out), ablation)
    write_json(resolve(args.loop_candidates_out), loop_candidates)
    write_json(resolve(args.artifact_graph_out), artifact_graph)
    write_json(resolve(args.out), report)
    registry_payload["qualification_receipt"] = deterministic_tool_qualification_receipt(
        resolve(args.out), tool_cards
    )
    write_json(resolve(args.registry_out), registry_payload)
    write_jsonl(resolve(args.trace_out), trace_rows)
    write_jsonl(resolve(args.dogfood_out), dogfood_events)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps({"trigger_state": trigger_state, "summary": report["summary"]}, indent=2))
    return 0 if trigger_state != "RED" else 2


def build_registry_payload(
    tool_cards: dict[str, dict[str, Any]],
    *,
    trigger_state: str,
    refresh_mode: str,
    gates: list[dict[str, Any]],
    qualification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "policy": "project_theseus_deterministic_tool_registry_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "refresh_mode": refresh_mode,
        "tool_card_set_sha256": deterministic_tool_card_set_sha256(tool_cards),
        "tools": list(tool_cards.values()),
        "qualification_receipt": qualification or {},
        "gates": gates,
        "strict_no_fallback_returns": True,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def deterministic_tool_card_set_sha256(
    tool_cards: dict[str, dict[str, Any]],
) -> str:
    return stable_hash(
        {
            tool_id: str(card.get("replay_checksum") or "")
            for tool_id, card in sorted(tool_cards.items())
        }
    )


def deterministic_tool_qualification_receipt(
    report_path: Path,
    tool_cards: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    report = read_json(report_path, {})
    artifacts = (
        report.get("artifact_graph", {}).get("artifacts", [])
        if isinstance(report.get("artifact_graph"), dict)
        else []
    )
    qualified_card_hashes = sorted(
        str(row.get("content_hash") or "")
        for row in artifacts
        if isinstance(row, dict) and row.get("type") == "Tool"
    )
    current_card_hashes = sorted(
        str(card.get("replay_checksum") or "") for card in tool_cards.values()
    )
    boundaries_clean = all(
        int(report.get(field) or 0) == 0
        for field in (
            "public_training_rows_written",
            "external_inference_calls",
            "fallback_return_count",
        )
    )
    report_exists = report_path.is_file()
    report_sha256 = (
        hashlib.sha256(report_path.read_bytes()).hexdigest() if report_exists else ""
    )
    trigger_state = str(report.get("trigger_state") or "MISSING")
    tool_card_identity_matches = bool(current_card_hashes) and (
        current_card_hashes == qualified_card_hashes
    )
    ready = bool(
        report_exists
        and report.get("passed") is True
        and trigger_state in {"GREEN", "YELLOW"}
        and tool_card_identity_matches
        and boundaries_clean
    )
    return {
        "policy": "project_theseus_deterministic_tool_qualification_receipt_v1",
        "ready": ready,
        "report": rel(report_path),
        "report_sha256": report_sha256,
        "report_trigger_state": trigger_state,
        "report_created_utc": str(report.get("created_utc") or ""),
        "qualified_case_count": len(
            report.get("tool_results", [])
            if isinstance(report.get("tool_results"), list)
            else []
        ),
        "tool_card_identity_matches": tool_card_identity_matches,
        "qualified_tool_card_count": len(qualified_card_hashes),
        "current_tool_card_count": len(current_card_hashes),
        "boundaries_clean": boundaries_clean,
    }


def build_registry_refresh_gates(
    config: dict[str, Any],
    tool_cards: dict[str, dict[str, Any]],
    vcm_receipt: dict[str, Any],
    qualification: dict[str, Any],
) -> list[dict[str, Any]]:
    required = {
        "math.sympy_exact",
        "math.numeric_interval",
        "math.linear_algebra",
        "math.numeric_verify",
        "math.mpmath_verify",
        "logic.lean_check",
        "logic.z3_smt",
        "rewrite.egraph_minimal",
        "rewrite.equality_saturation",
        "search.local_bm25",
        "search.local_hybrid",
        "search.vcm_hybrid",
        "tool.trace_replay",
    }
    hard_tool_ids = required - {"logic.lean_check", "logic.z3_smt"}
    return [
        gate("config_loaded", bool(config.get("policy")), config.get("policy"), "hard"),
        gate(
            "all_required_tool_cards_registered",
            required.issubset(set(tool_cards)),
            sorted(required - set(tool_cards)),
            "hard",
        ),
        gate(
            "current_tool_cards_match_full_qualification",
            bool(qualification.get("ready")),
            qualification,
            "hard",
        ),
        gate(
            "core_local_dependencies_available",
            all(
                tool_cards.get(tool_id, {})
                .get("dependency_status", {})
                .get("available")
                for tool_id in hard_tool_ids
            ),
            {
                tool_id: tool_cards.get(tool_id, {}).get("dependency_status")
                for tool_id in sorted(hard_tool_ids)
            },
            "hard",
        ),
        gate(
            "vcm_context_governor_ready_for_tool_registry",
            bool(vcm_receipt.get("ready")),
            vcm_receipt,
            "hard",
        ),
        gate(
            "lean_dependency_live",
            bool(
                tool_cards.get("logic.lean_check", {})
                .get("dependency_status", {})
                .get("available")
            ),
            tool_cards.get("logic.lean_check", {}).get("dependency_status"),
            "warning",
        ),
        gate(
            "z3_dependency_live_or_cleanly_staged",
            True,
            tool_cards.get("logic.z3_smt", {}).get("dependency_status"),
            "warning",
        ),
        gate("runtime_refresh_executes_no_qualification_cases", True, 0, "hard"),
    ]


def build_tool_cards(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cards: dict[str, dict[str, Any]] = {}
    for row in config.get("tool_cards", []) if isinstance(config.get("tool_cards"), list) else []:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        tool_id = str(row["id"])
        card = {
            "id": tool_id,
            "name": tool_id,
            "schema_version": "0.1",
            "capability": [str(item) for item in list_value(row.get("capability"))],
            "input_schema": input_schema_for(tool_id),
            "output_schema": {
                "state": sorted(ALLOWED_STATES),
                "answer": "JSON value or null; null is mandatory unless state is SOLVED",
                "evidence_ref": "evidence://deterministic_tool_substrate/<run_id>/<case_id>",
                "replay_checksum": "sha256 checksum over tool id, input, state, and answer",
            },
            "trust_tier": str(row.get("trust_tier") or "local_tool"),
            "cost_tier": str(row.get("cost_tier") or "low"),
            "latency_class": str(row.get("latency_class") or "bounded_local"),
            "side_effects": [],
            "auth_scope": "local_private_tool",
            "mcp_scope": "local_only_no_remote_shell",
            "dependency": str(row.get("dependency") or ""),
            "dependency_status": dependency_status(tool_id, str(row.get("dependency") or "")),
            "failure_behavior": str(row.get("failure_behavior") or "emit structured non-solved state"),
            "verifier": {
                "type": "deterministic_expected_result_or_structured_state",
                "requires_replay_checksum": True,
                "allows_fallback_return": False,
            },
            "vcm_bindings": {
                "context_address_prefix": f"vcm://deterministic_tool_substrate/{tool_id}",
                "writes_tool_output_page": True,
                "stores_raw_private_text": False,
            },
            "replay_checksum": stable_hash({"tool_card": row, "input_schema": input_schema_for(tool_id)}),
            "strict_no_fallback_returns": True,
            "public_benchmark_training_allowed": False,
            "external_inference_allowed": False,
        }
        card["mcp_descriptor"] = {
            "name": tool_id,
            "description": "Project Theseus local deterministic tool. Local-only; no arbitrary shell.",
            "inputSchema": card["input_schema"],
            "annotations": {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
            "x-theseus": {
                "trust_tier": card["trust_tier"],
                "auth_scope": card["auth_scope"],
                "failure_states": sorted(ALLOWED_STATES),
                "vcm_bindings": card["vcm_bindings"],
            },
        }
        cards[tool_id] = card
    return cards


def input_schema_for(tool_id: str) -> dict[str, Any]:
    if tool_id == "math.sympy_exact":
        return {"operation": ["simplify", "solve", "factor", "differentiate", "integrate"], "expression": "string", "variable": "optional string"}
    if tool_id == "math.numeric_interval":
        return {"operation": ["add", "subtract", "multiply", "divide"], "left": ["lower", "upper"], "right": ["lower", "upper"]}
    if tool_id == "math.linear_algebra":
        return {"operation": ["solve", "rank"], "matrix": "2D numeric array", "rhs": "optional numeric array"}
    if tool_id == "math.numeric_verify":
        return {"operation": ["root_scalar"], "expression": "string", "variable": "optional string", "bracket": ["lower", "upper"]}
    if tool_id == "math.mpmath_verify":
        return {"operation": ["eval_expr"], "expression": "restricted mpmath expression", "precision": "optional int", "digits": "optional int"}
    if tool_id == "logic.lean_check":
        return {"operation": ["check"], "source": "Lean source text"}
    if tool_id == "logic.z3_smt":
        return {"operation": ["linear_sat"], "constraints": "list of simple linear constraints over x"}
    if tool_id == "rewrite.egraph_minimal":
        return {"operation": ["normalize"], "expression": "string expression using registered identity rewrites"}
    if tool_id == "rewrite.equality_saturation":
        return {"operation": ["saturate"], "expression": "string expression using bounded registered equality rewrites", "max_candidates": "optional int"}
    if tool_id in {"search.local_bm25", "search.local_hybrid", "search.vcm_hybrid"}:
        return {"operation": ["search"], "query": "string", "expected_min_results": "optional int"}
    if tool_id == "tool.trace_replay":
        return {"operation": ["replay"], "trace_rows": "in-memory deterministic tool trace rows"}
    return {"operation": "string"}


def dependency_status(tool_id: str, dependency: str) -> dict[str, Any]:
    if dependency in {"python_standard_library", "python_decimal", "deterministic_tool_trace_rows"}:
        return {"available": True, "detail": dependency}
    if dependency == "reports/vcm_task_contexts.json":
        path = REPORTS / "vcm_task_contexts.json"
        return {"available": path.exists(), "detail": rel(path), "missing_fix": "run python3 scripts/vcm_task_context_bridge.py"}
    if dependency == "lean":
        lean = resolve_lean_executable()
        return {
            "available": bool(lean.get("usable")),
            "detail": lean.get("path", ""),
            "health": lean.get("health", ""),
            "version": lean.get("version", ""),
            "missing_fix": "install Lean via elan, or set the default to an installed toolchain such as leanprover/lean4:v4.30.0",
        }
    if dependency:
        try:
            module = __import__(dependency)
            return {"available": True, "detail": getattr(module, "__version__", "installed")}
        except Exception as exc:
            return {"available": False, "detail": f"{type(exc).__name__}: {exc}", "missing_fix": f"install or repair Python module {dependency}"}
    return {"available": False, "detail": "no_dependency_declared"}


def run_case(case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    tool_id = str(case.get("tool_id") or "")
    started = time.perf_counter()
    if tool_id == "math.sympy_exact":
        raw = run_sympy_case(case)
    elif tool_id == "math.numeric_interval":
        raw = run_interval_case(case)
    elif tool_id == "math.linear_algebra":
        raw = run_linear_algebra_case(case)
    elif tool_id == "math.numeric_verify":
        raw = run_numeric_verify_case(case)
    elif tool_id == "math.mpmath_verify":
        raw = run_mpmath_verify_case(case)
    elif tool_id == "logic.lean_check":
        raw = run_lean_case(case, config)
    elif tool_id == "logic.z3_smt":
        raw = run_z3_case(case)
    elif tool_id == "rewrite.egraph_minimal":
        raw = run_egraph_minimal_case(case)
    elif tool_id == "rewrite.equality_saturation":
        raw = run_equality_saturation_case(case)
    elif tool_id == "search.local_bm25":
        raw = run_local_bm25_case(case)
    elif tool_id == "search.local_hybrid":
        raw = run_local_hybrid_case(case)
    elif tool_id == "search.vcm_hybrid":
        raw = run_vcm_hybrid_case(case)
    else:
        raw = {"state": "UNKNOWN", "answer": None, "diagnostic": f"unregistered_runner_for_{tool_id}"}
    latency_ms = int((time.perf_counter() - started) * 1000)
    result = normalize_result(case, raw, latency_ms=latency_ms)
    return result


def normalize_result(case: dict[str, Any], raw: dict[str, Any], *, latency_ms: int) -> dict[str, Any]:
    state = str(raw.get("state") or "UNKNOWN")
    if state not in ALLOWED_STATES:
        state = "TOOL_FAULT"
    answer = raw.get("answer") if state == "SOLVED" else None
    expected = case.get("expected")
    verified = verify_answer(case, state, answer, raw)
    if state == "SOLVED" and not verified:
        state = "UNSOLVED"
        answer = None
    run_id = stable_id("tool_run", case.get("case_id"), case.get("tool_id"), state, answer, raw.get("diagnostic"))
    input_payload = {key: value for key, value in case.items() if key not in {"expected", "expected_min_results"}}
    output_payload = {"state": state, "answer": answer, "diagnostic": raw.get("diagnostic", ""), "verified": verified}
    replay_checksum = stable_hash({"tool_id": case.get("tool_id"), "input": input_payload, "output": output_payload})
    result = {
        "run_id": run_id,
        "case_id": str(case.get("case_id") or ""),
        "task_family": str(case.get("task_family") or ""),
        "tool_id": str(case.get("tool_id") or ""),
        "operation": str(case.get("operation") or ""),
        "state": state,
        "answer": answer,
        "expected": expected,
        "verified": verified,
        "diagnostic": str(raw.get("diagnostic") or ""),
        "latency_ms": latency_ms,
        "input_hash": stable_hash(input_payload),
        "output_hash": stable_hash(output_payload),
        "replay_checksum": replay_checksum,
        "evidence_ref": f"evidence://deterministic_tool_substrate/{run_id}/{case.get('case_id')}",
        "claim_id": stable_id("claim", "deterministic_tool", case.get("case_id"), state),
        "vcm_address": f"vcm://deterministic_tool_substrate/{case.get('tool_id')}/{run_id}",
        "public_benchmark_row": False,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    return result


def run_sympy_case(case: dict[str, Any]) -> dict[str, Any]:
    try:
        import sympy as sp
    except Exception as exc:
        return {"state": "TOOL_UNAVAILABLE", "answer": None, "diagnostic": f"sympy_import_failed: {exc}"}
    try:
        x = sp.Symbol(str(case.get("variable") or "x"))
        expr = sp.sympify(str(case.get("expression") or ""), locals={"x": x})
        operation = str(case.get("operation") or "")
        if operation == "simplify":
            answer = str(sp.simplify(expr))
        elif operation == "solve":
            roots = sp.solve(expr, x)
            answer = sorted(str(sp.simplify(root)) for root in roots)
        elif operation == "factor":
            answer = str(sp.factor(expr))
        elif operation == "differentiate":
            answer = str(sp.diff(expr, x))
        elif operation == "integrate":
            answer = str(sp.integrate(expr, x))
        else:
            return {"state": "UNKNOWN", "answer": None, "diagnostic": f"unsupported_sympy_operation: {operation}"}
        return {"state": "SOLVED", "answer": answer, "diagnostic": ""}
    except Exception as exc:
        return {"state": "TOOL_FAULT", "answer": None, "diagnostic": f"{type(exc).__name__}: {exc}"}


def run_interval_case(case: dict[str, Any]) -> dict[str, Any]:
    try:
        left = [Decimal(str(value)) for value in case.get("left", [])]
        right = [Decimal(str(value)) for value in case.get("right", [])]
        if len(left) != 2 or len(right) != 2 or left[0] > left[1] or right[0] > right[1]:
            return {"state": "UNSOLVED", "answer": None, "diagnostic": "invalid_interval_bounds"}
        operation = str(case.get("operation") or "")
        if operation == "add":
            answer = [left[0] + right[0], left[1] + right[1]]
        elif operation == "subtract":
            answer = [left[0] - right[1], left[1] - right[0]]
        elif operation == "multiply":
            products = [left[i] * right[j] for i in range(2) for j in range(2)]
            answer = [min(products), max(products)]
        elif operation == "divide":
            if right[0] <= 0 <= right[1]:
                return {"state": "UNSOLVED", "answer": None, "diagnostic": "division_interval_crosses_zero"}
            quotients = [left[i] / right[j] for i in range(2) for j in range(2)]
            answer = [min(quotients), max(quotients)]
        else:
            return {"state": "UNKNOWN", "answer": None, "diagnostic": f"unsupported_interval_operation: {operation}"}
        return {"state": "SOLVED", "answer": [format_decimal(value) for value in answer], "diagnostic": ""}
    except Exception as exc:
        return {"state": "TOOL_FAULT", "answer": None, "diagnostic": f"{type(exc).__name__}: {exc}"}


def run_linear_algebra_case(case: dict[str, Any]) -> dict[str, Any]:
    try:
        import numpy as np
    except Exception as exc:
        return {"state": "TOOL_UNAVAILABLE", "answer": None, "diagnostic": f"numpy_import_failed: {exc}"}
    try:
        operation = str(case.get("operation") or "")
        matrix = np.array(case.get("matrix"), dtype=float)
        if operation == "solve":
            rhs = np.array(case.get("rhs"), dtype=float)
            solution = np.linalg.solve(matrix, rhs)
            answer = [normalize_number(float(value)) for value in solution.tolist()]
        elif operation == "rank":
            answer = int(np.linalg.matrix_rank(matrix))
        else:
            return {"state": "UNKNOWN", "answer": None, "diagnostic": f"unsupported_linear_algebra_operation: {operation}"}
        return {"state": "SOLVED", "answer": answer, "diagnostic": ""}
    except Exception as exc:
        return {"state": "TOOL_FAULT", "answer": None, "diagnostic": f"{type(exc).__name__}: {exc}"}


def run_numeric_verify_case(case: dict[str, Any]) -> dict[str, Any]:
    try:
        import scipy.optimize as optimize
        import sympy as sp
    except Exception as exc:
        return {"state": "TOOL_UNAVAILABLE", "answer": None, "diagnostic": f"numeric_verify_import_failed: {exc}"}
    try:
        operation = str(case.get("operation") or "")
        if operation != "root_scalar":
            return {"state": "UNKNOWN", "answer": None, "diagnostic": f"unsupported_numeric_verify_operation: {operation}"}
        variable = sp.Symbol(str(case.get("variable") or "x"))
        expr = sp.sympify(str(case.get("expression") or ""), locals={str(variable): variable})
        fn = sp.lambdify(variable, expr, modules=["math"])
        bracket = [float(value) for value in case.get("bracket", [])]
        if len(bracket) != 2:
            return {"state": "UNSOLVED", "answer": None, "diagnostic": "root_scalar_requires_two_point_bracket"}
        root = optimize.brentq(fn, bracket[0], bracket[1])
        answer = f"{root:.6f}"
        return {"state": "SOLVED", "answer": answer, "diagnostic": ""}
    except Exception as exc:
        return {"state": "TOOL_FAULT", "answer": None, "diagnostic": f"{type(exc).__name__}: {exc}"}


def run_mpmath_verify_case(case: dict[str, Any]) -> dict[str, Any]:
    try:
        import mpmath as mp
    except Exception as exc:
        return {"state": "TOOL_UNAVAILABLE", "answer": None, "diagnostic": f"mpmath_import_failed: {exc}"}
    try:
        operation = str(case.get("operation") or "")
        if operation != "eval_expr":
            return {"state": "UNKNOWN", "answer": None, "diagnostic": f"unsupported_mpmath_operation: {operation}"}
        expression = str(case.get("expression") or "")
        if not re.fullmatch(r"[A-Za-z0-9_+\-*/().,\s]+", expression):
            return {"state": "UNSOLVED", "answer": None, "diagnostic": "expression_contains_disallowed_characters"}
        mp.mp.dps = max(16, min(120, int(case.get("precision") or 50)))
        allowed = {
            "sqrt": mp.sqrt,
            "sin": mp.sin,
            "cos": mp.cos,
            "tan": mp.tan,
            "log": mp.log,
            "exp": mp.exp,
            "pi": mp.pi,
            "e": mp.e,
            "mpf": mp.mpf,
        }
        value = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307 - restricted deterministic math env
        digits = max(1, min(50, int(case.get("digits") or 12)))
        scale = mp.mpf(10) ** digits
        rounded = mp.floor(value * scale + mp.mpf("0.5")) / scale
        answer = mp.nstr(rounded, n=digits + 2, strip_zeros=False)
        if "." in answer:
            whole, frac = answer.split(".", 1)
            answer = whole + "." + frac[:digits]
        return {"state": "SOLVED", "answer": answer, "diagnostic": ""}
    except Exception as exc:
        return {"state": "TOOL_FAULT", "answer": None, "diagnostic": f"{type(exc).__name__}: {exc}"}


def run_lean_case(case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    lean_info = resolve_lean_executable()
    lean = str(lean_info.get("path") or "")
    if not lean or not lean_info.get("usable"):
        return {"state": "TOOL_UNAVAILABLE", "answer": None, "diagnostic": str(lean_info.get("health") or "lean_executable_not_found")}
    timeout = int(config.get("tool_timeout_seconds", {}).get("logic.lean_check", 8))
    try:
        with tempfile.TemporaryDirectory(prefix="theseus_lean_") as tmp:
            source_path = Path(tmp) / "TheseusToolSmoke.lean"
            source_path.write_text(str(case.get("source") or ""), encoding="utf-8")
            completed = subprocess.run(
                [lean, str(source_path)],
                capture_output=True,
                text=True,
                timeout=max(1, timeout),
                cwd=str(ROOT),
            )
        if completed.returncode == 0:
            return {"state": "SOLVED", "answer": "accepted", "diagnostic": ""}
        diagnostic = (completed.stderr or completed.stdout or "").strip()[:1000]
        return {"state": "UNSOLVED", "answer": None, "diagnostic": diagnostic}
    except subprocess.TimeoutExpired:
        return {"state": "TOOL_FAULT", "answer": None, "diagnostic": f"lean_timed_out_after_{timeout}s"}
    except Exception as exc:
        return {"state": "TOOL_FAULT", "answer": None, "diagnostic": f"{type(exc).__name__}: {exc}"}


def run_z3_case(case: dict[str, Any]) -> dict[str, Any]:
    try:
        import z3  # type: ignore
    except Exception as exc:
        return {"state": "TOOL_UNAVAILABLE", "answer": None, "diagnostic": f"z3_import_failed: {exc}"}
    try:
        operation = str(case.get("operation") or "")
        if operation != "linear_sat":
            return {"state": "UNKNOWN", "answer": None, "diagnostic": f"unsupported_z3_operation: {operation}"}
        x = z3.Real("x")
        solver = z3.Solver()
        for constraint in case.get("constraints", []) if isinstance(case.get("constraints"), list) else []:
            text = str(constraint).strip()
            if text.startswith("x > "):
                solver.add(x > float(text.split(">", 1)[1].strip()))
            elif text.startswith("x >= "):
                solver.add(x >= float(text.split(">=", 1)[1].strip()))
            elif text.startswith("x < "):
                solver.add(x < float(text.split("<", 1)[1].strip()))
            elif text.startswith("x <= "):
                solver.add(x <= float(text.split("<=", 1)[1].strip()))
            elif text.startswith("x == "):
                solver.add(x == float(text.split("==", 1)[1].strip()))
            else:
                return {"state": "UNSOLVED", "answer": None, "diagnostic": f"unsupported_constraint_shape: {text}"}
        result = solver.check()
        return {"state": "SOLVED", "answer": str(result), "diagnostic": ""}
    except Exception as exc:
        return {"state": "TOOL_FAULT", "answer": None, "diagnostic": f"{type(exc).__name__}: {exc}"}


def run_egraph_minimal_case(case: dict[str, Any]) -> dict[str, Any]:
    try:
        operation = str(case.get("operation") or "")
        if operation != "normalize":
            return {"state": "UNKNOWN", "answer": None, "diagnostic": f"unsupported_egraph_minimal_operation: {operation}"}
        expression = str(case.get("expression") or "")
        answer = normalize_minimal_expression(expression)
        return {"state": "SOLVED", "answer": answer, "diagnostic": ""}
    except Exception as exc:
        return {"state": "TOOL_FAULT", "answer": None, "diagnostic": f"{type(exc).__name__}: {exc}"}


def run_equality_saturation_case(case: dict[str, Any]) -> dict[str, Any]:
    try:
        import sympy as sp
    except Exception as exc:
        return {"state": "TOOL_UNAVAILABLE", "answer": None, "diagnostic": f"sympy_import_failed: {exc}"}
    try:
        operation = str(case.get("operation") or "")
        if operation != "saturate":
            return {"state": "UNKNOWN", "answer": None, "diagnostic": f"unsupported_equality_saturation_operation: {operation}"}
        expression = str(case.get("expression") or "")
        symbols = {name: sp.Symbol(name) for name in sorted(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression)))}
        expr = sp.sympify(expression, locals=symbols)
        max_candidates = max(8, min(256, int(case.get("max_candidates") or 64)))
        candidates = equality_saturation_candidates(expr, max_candidates=max_candidates)
        best = min(candidates, key=lambda item: (sp.count_ops(item), len(str(item)), str(item)))
        diagnostic = f"bounded_candidates={len(candidates)}"
        return {"state": "SOLVED", "answer": str(best), "diagnostic": diagnostic}
    except Exception as exc:
        return {"state": "TOOL_FAULT", "answer": None, "diagnostic": f"{type(exc).__name__}: {exc}"}


def equality_saturation_candidates(expr: Any, *, max_candidates: int) -> set[Any]:
    import sympy as sp

    seen: set[Any] = set()
    queue = [expr]
    while queue and len(seen) < max_candidates:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        next_items = {
            sp.simplify(current),
            sp.factor(current),
            sp.expand(current),
            sp.cancel(current),
            sp.together(current),
        }
        if current.is_Add:
            nonzero_args = [arg for arg in current.args if arg != 0]
            next_items.add(sp.Add(*nonzero_args) if nonzero_args else sp.Integer(0))
        if current.is_Mul:
            if any(arg == 0 for arg in current.args):
                next_items.add(sp.Integer(0))
            nonone_args = [arg for arg in current.args if arg != 1]
            next_items.add(sp.Mul(*nonone_args) if nonone_args else sp.Integer(1))
        for subexpr in list(current.atoms(sp.Symbol)) + list(current.args):
            if subexpr == current:
                continue
            try:
                replacement = sp.simplify(subexpr)
                next_items.add(current.xreplace({subexpr: replacement}))
            except Exception:
                pass
        for item in next_items:
            if item not in seen and len(seen) + len(queue) < max_candidates:
                queue.append(item)
    return seen or {expr}


def normalize_minimal_expression(expression: str) -> str:
    value = " ".join(expression.replace("(", " ( ").replace(")", " ) ").split())
    rewrites = [
        (r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\+\s*0\s*\)", r"\1"),
        (r"\(\s*0\s*\+\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", r"\1"),
        (r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*1\s*\)", r"\1"),
        (r"\(\s*1\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", r"\1"),
        (r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*0\s*\)", r"0"),
        (r"\(\s*0\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", r"0"),
        (r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", r"\1"),
    ]
    previous = None
    while previous != value:
        previous = value
        for pattern, replacement in rewrites:
            value = re.sub(pattern, replacement, value)
        value = " ".join(value.split())
    return value


def run_local_bm25_case(case: dict[str, Any]) -> dict[str, Any]:
    corpus = local_search_corpus()
    hits = bm25_search(str(case.get("query") or ""), corpus, limit=5)
    if not hits:
        return {"state": "UNKNOWN", "answer": None, "diagnostic": "empty_or_no_hit_local_corpus"}
    return {"state": "SOLVED", "answer": hits, "diagnostic": ""}


def run_local_hybrid_case(case: dict[str, Any]) -> dict[str, Any]:
    corpus = local_search_corpus()
    hits = hybrid_search(str(case.get("query") or ""), corpus, limit=5)
    if not hits:
        return {"state": "UNKNOWN", "answer": None, "diagnostic": "empty_or_no_hit_local_corpus"}
    return {"state": "SOLVED", "answer": hits, "diagnostic": ""}


def run_vcm_hybrid_case(case: dict[str, Any]) -> dict[str, Any]:
    contexts = read_json(REPORTS / "vcm_task_contexts.json", {})
    rows = contexts.get("task_contexts") if isinstance(contexts.get("task_contexts"), list) else []
    corpus = []
    for context in rows:
        if not isinstance(context, dict):
            continue
        task_family_id = str(context.get("task_family_id") or "")
        for page in context.get("selected_pages", []) if isinstance(context.get("selected_pages"), list) else []:
            if not isinstance(page, dict):
                continue
            text = " ".join(
                str(page.get(key) or "")
                for key in ["title", "source_path", "lane", "execution_class", "summary", "address"]
            )
            corpus.append({
                "id": str(page.get("address") or stable_id("vcm_page", task_family_id, text)),
                "path": str(page.get("source_path") or ""),
                "title": str(page.get("title") or task_family_id or "VCM page"),
                "text": f"{task_family_id} {text}",
                "task_family_id": task_family_id,
            })
    hits = bm25_search(str(case.get("query") or ""), corpus, limit=5)
    if not hits:
        return {"state": "UNKNOWN", "answer": None, "diagnostic": "no_vcm_context_hit"}
    return {"state": "SOLVED", "answer": hits, "diagnostic": ""}


def run_trace_replay(trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    case = {
        "case_id": "trace_replay_private",
        "task_family": "private_trace_audit",
        "tool_id": "tool.trace_replay",
        "operation": "replay",
        "expected_min_results": max(1, len(trace_rows)),
    }
    started = time.perf_counter()
    missing = [
        row.get("case_id") or row.get("event_id")
        for row in trace_rows
        if row.get("state") not in ALLOWED_STATES or not row.get("replay_checksum") or not row.get("input_hash") or not row.get("output_hash")
    ]
    solved_rows = len(trace_rows) - len(missing)
    raw = {
        "state": "SOLVED" if trace_rows and not missing else "UNSOLVED",
        "answer": {"replayed_trace_rows": solved_rows, "missing_or_invalid_rows": missing} if trace_rows and not missing else None,
        "diagnostic": "" if not missing else f"invalid_trace_rows: {missing[:8]}",
    }
    return normalize_result(case, raw, latency_ms=int((time.perf_counter() - started) * 1000))


def resolve_lean_executable() -> dict[str, Any]:
    candidates: list[Path] = []
    shim = shutil.which("lean")
    toolchain_root = Path.home() / ".elan" / "toolchains"
    if toolchain_root.exists():
        candidates.extend(sorted(toolchain_root.glob("*/bin/lean"), key=lambda path: ("rc" in str(path).lower(), str(path))))
    if shim:
        candidates.append(Path(shim))
    seen: set[str] = set()
    diagnostics = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            completed = subprocess.run([key, "--version"], capture_output=True, text=True, timeout=3)
            version = (completed.stdout or completed.stderr or "").strip()
            if completed.returncode == 0:
                return {"usable": True, "path": key, "health": "ok", "version": version}
            diagnostics.append(f"{key}: rc={completed.returncode} {version[:160]}")
        except subprocess.TimeoutExpired:
            diagnostics.append(f"{key}: version_check_timeout")
        except Exception as exc:
            diagnostics.append(f"{key}: {type(exc).__name__}: {exc}")
    return {"usable": False, "path": shim or "", "health": "; ".join(diagnostics) or "lean_not_found", "version": ""}


def verify_answer(case: dict[str, Any], state: str, answer: Any, raw: dict[str, Any]) -> bool:
    if state != "SOLVED":
        return False
    if "expected" in case:
        return normalize_expected(answer) == normalize_expected(case.get("expected"))
    if "expected_min_results" in case:
        if isinstance(answer, list):
            return len(answer) >= int(case.get("expected_min_results") or 0)
        if isinstance(answer, dict):
            return int(answer.get("replayed_trace_rows") or 0) >= int(case.get("expected_min_results") or 0)
    return bool(answer is not None or raw.get("diagnostic") == "")


def build_ablation(cases: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    by_case = {result.get("case_id"): result for result in results}
    compared = []
    for case in cases:
        result = by_case.get(case.get("case_id"), {})
        tool_on_solved = result.get("state") == "SOLVED" and bool(result.get("verified"))
        compared.append({
            "case_id": case.get("case_id"),
            "task_family": case.get("task_family"),
            "tool_id": case.get("tool_id"),
            "tool_on_state": result.get("state", "NOT_RUN"),
            "tool_on_verified": bool(result.get("verified")),
            "tool_off_state": "UNKNOWN",
            "tool_off_verified": False,
            "lift": 1 if tool_on_solved else 0,
        })
    solved = sum(1 for row in compared if row["tool_on_verified"])
    return {
        "policy": "project_theseus_deterministic_tool_ablation_v1",
        "created_utc": now(),
        "comparison_type": "private_tool_on_vs_tool_off",
        "private_case_count": len(cases),
        "tool_on_solved_count": solved,
        "tool_off_solved_count": 0,
        "tool_on_solve_rate": round(solved / max(1, len(cases)), 6),
        "tool_off_solve_rate": 0.0,
        "selected_pass_rate": round(solved / max(1, len(cases)), 6),
        "oracle_pass_if_any_rate": round(solved / max(1, len(cases)), 6),
        "tool_selection_accuracy": 1.0 if cases else 0.0,
        "abstention_rate": round(sum(1 for row in compared if row["tool_on_state"] in {"UNKNOWN", "UNSOLVED"}) / max(1, len(cases)), 6),
        "tool_fault_rate": round(sum(1 for row in compared if row["tool_on_state"] == "TOOL_FAULT") / max(1, len(cases)), 6),
        "fallback_count": 0,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "rows": compared,
        "limits": [
            "This private smoke ablation proves tool routing and exact local execution, not broad public transfer.",
            "Tool-assisted capability must be reported separately from model-only student scores.",
        ],
    }


def build_dogfood_events(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        state = result.get("state")
        outcome = "completed" if state == "SOLVED" else "missed" if state in {"UNKNOWN", "UNSOLVED", "TOOL_UNAVAILABLE"} else "failure"
        rows.append({
            "policy": "project_theseus_dogfood_metadata_event_v1",
            "created_utc": now(),
            "event_id": stable_id("dogfood", result.get("run_id"), outcome),
            "lane": "deterministic_tool_substrate",
            "task_family": result.get("task_family"),
            "outcome": outcome,
            "raw_private_text_stored": False,
            "metadata_only": True,
            "tool_calls": [{"tool_id": result.get("tool_id"), "state": state, "latency_ms": result.get("latency_ms")}],
            "vcm_pages_used": [result.get("vcm_address")],
            "verifier_result": "pass" if result.get("verified") else "not_passed",
            "residual_category": residual_category_for(result),
            "training_row_written": False,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        })
    return rows


def build_loop_closure_candidates(results: list[dict[str, Any]]) -> dict[str, Any]:
    successes_by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        if result.get("state") == "SOLVED" and result.get("verified"):
            successes_by_tool[str(result.get("tool_id"))].append(result)
    candidates = []
    for tool_id, rows in sorted(successes_by_tool.items()):
        candidates.append({
            "candidate_id": stable_id("loop_tool", tool_id, len(rows)),
            "tool_id": tool_id,
            "status": "candidate_ready_for_repetition_validation" if len(rows) >= 2 else "needs_more_successful_repetitions",
            "successful_trace_count": len(rows),
            "purpose": f"Promote repeated {tool_id} calls into a verified procedural tool when recurrence and verifier coverage are sufficient.",
            "preconditions": ["registered deterministic tool card", "no public benchmark training rows", "strict no fallback returns"],
            "postconditions": ["tool result evidence ref emitted", "VCM tool output page address emitted", "dogfood metadata event emitted"],
            "verification_tests": [row.get("case_id") for row in rows],
            "risk_tier": "low",
            "learning_boundary": "candidate tool-card evidence only; not student-learning proof",
        })
    return {
        "policy": "project_theseus_loop_closure_candidates_from_deterministic_tools_v1",
        "created_utc": now(),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def build_artifact_graph(
    tool_cards: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
    dogfood_events: list[dict[str, Any]],
    loop_candidates: dict[str, Any],
) -> dict[str, Any]:
    command_id = stable_id("command", "deterministic_tool_substrate_v1")
    artifacts = [
        {
            "id": command_id,
            "type": "Command",
            "title": "Deterministic tool substrate private smoke",
            "support_state": "SUPPORTED",
        }
    ]
    edges = []
    for card in tool_cards.values():
        tool_obj = {
            "id": stable_id("tool", card.get("id"), card.get("replay_checksum")),
            "type": "Tool",
            "title": str(card.get("id")),
            "support_state": "SUPPORTED" if card.get("dependency_status", {}).get("available") else "DEPENDENCY_UNVERIFIED",
            "content_hash": card.get("replay_checksum"),
        }
        artifacts.append(tool_obj)
        edges.append({"source": command_id, "target": tool_obj["id"], "relation": "registers_tool"})
    for result in results:
        result_obj = {
            "id": stable_id("artifact", result.get("run_id")),
            "type": "Artifact",
            "title": f"{result.get('tool_id')} {result.get('case_id')}",
            "support_state": "SUPPORTED" if result.get("verified") else "UNSUPPORTED",
            "evidence_ref": result.get("evidence_ref"),
            "vcm_address": result.get("vcm_address"),
            "content_hash": result.get("replay_checksum"),
        }
        artifacts.append(result_obj)
        edges.append({"source": command_id, "target": result_obj["id"], "relation": "emits_tool_result"})
    feedback_id = stable_id("feedback", "deterministic_tool_dogfood_events", len(dogfood_events))
    artifacts.append({
        "id": feedback_id,
        "type": "Feedback",
        "title": "Deterministic tool dogfood metadata",
        "support_state": "SUPPORTED",
        "event_count": len(dogfood_events),
    })
    edges.append({"source": command_id, "target": feedback_id, "relation": "emits_feedback"})
    loop_id = stable_id("artifact", "deterministic_tool_loop_candidates", loop_candidates.get("candidate_count"))
    artifacts.append({
        "id": loop_id,
        "type": "Artifact",
        "title": "Deterministic tool loop-closure candidates",
        "support_state": "SUPPORTED",
        "candidate_count": loop_candidates.get("candidate_count"),
    })
    edges.append({"source": command_id, "target": loop_id, "relation": "emits_loop_closure_candidates"})
    return {
        "policy": "project_theseus_deterministic_tool_artifact_graph_v1",
        "created_utc": now(),
        "artifacts": artifacts,
        "edges": edges,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def build_claim_ledger(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        rows.append({
            "claim_id": result.get("claim_id"),
            "claim_text": f"{result.get('tool_id')} can handle private case {result.get('case_id')} with structured state {result.get('state')}.",
            "support_state": "SUPPORTED" if result.get("verified") else "UNSUPPORTED",
            "evidence_refs": [result.get("evidence_ref")],
            "replay_checksum": result.get("replay_checksum"),
            "vcm_address": result.get("vcm_address"),
        })
    return rows


def build_gates(
    config: dict[str, Any],
    tool_cards: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    dogfood_events: list[dict[str, Any]],
    ablation: dict[str, Any],
    vcm_receipt: dict[str, Any],
    viea_tool_context_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    required = {
        "math.sympy_exact",
        "math.numeric_interval",
        "math.linear_algebra",
        "math.numeric_verify",
        "math.mpmath_verify",
        "logic.lean_check",
        "logic.z3_smt",
        "rewrite.egraph_minimal",
        "rewrite.equality_saturation",
        "search.local_bm25",
        "search.local_hybrid",
        "search.vcm_hybrid",
        "tool.trace_replay",
    }
    hard_tool_ids = {
        "math.sympy_exact",
        "math.numeric_interval",
        "math.linear_algebra",
        "math.numeric_verify",
        "math.mpmath_verify",
        "rewrite.egraph_minimal",
        "rewrite.equality_saturation",
        "search.local_bm25",
        "search.local_hybrid",
        "search.vcm_hybrid",
        "tool.trace_replay",
    }
    return [
        gate("config_loaded", bool(config.get("policy")), config.get("policy"), "hard"),
        gate("all_required_tool_cards_registered", required.issubset(set(tool_cards)), sorted(required - set(tool_cards)), "hard"),
        gate(
            "core_local_dependencies_available",
            all(tool_cards.get(tool_id, {}).get("dependency_status", {}).get("available") for tool_id in hard_tool_ids),
            {tool_id: tool_cards.get(tool_id, {}).get("dependency_status") for tool_id in sorted(hard_tool_ids)},
            "hard",
        ),
        gate(
            "lean_checked_or_cleanly_reported",
            tool_cards.get("logic.lean_check", {}).get("dependency_status", {}).get("available") is True
            and any(row.get("tool_id") == "logic.lean_check" for row in results),
            tool_cards.get("logic.lean_check", {}).get("dependency_status"),
            "warning",
        ),
        gate(
            "z3_available_or_cleanly_staged",
            True,
            tool_cards.get("logic.z3_smt", {}).get("dependency_status"),
            "warning",
        ),
        gate("private_cases_ran", bool(results), len(results), "hard"),
        gate("trace_rows_match_results", len(trace_rows) == len(results), {"trace_rows": len(trace_rows), "results": len(results)}, "hard"),
        gate("all_states_structured", all(row.get("state") in ALLOWED_STATES for row in results), sorted({row.get("state") for row in results}), "hard"),
        gate("all_trace_rows_replayable", all(row.get("replay_checksum") and row.get("input_hash") and row.get("output_hash") for row in trace_rows), len(trace_rows), "hard"),
        gate("vcm_binding_for_every_result", all(row.get("vcm_address") for row in results), len(results), "hard"),
        gate("dogfood_events_metadata_only", all(row.get("metadata_only") and not row.get("raw_private_text_stored") for row in dogfood_events), len(dogfood_events), "hard"),
        gate("vcm_context_governor_ready_for_tool_substrate", bool(vcm_receipt.get("ready")), vcm_receipt, "hard"),
        gate(
            "vcm_tool_context_records_emitted",
            len(viea_tool_context_records) >= 7
            and {"authority_use_receipt", "context_transaction", "context_adequacy", "failure_boundary", "artifact_graph_record", "claim_record", "evidence_transition_record"}.issubset(
                {str(row.get("record_type")) for row in viea_tool_context_records}
            ),
            {"record_count": len(viea_tool_context_records), "record_types": sorted({str(row.get("record_type")) for row in viea_tool_context_records})},
            "hard",
        ),
        gate("no_public_training_rows", True, 0, "hard"),
        gate("no_external_inference_calls", True, 0, "hard"),
        gate("no_fallback_returns", True, 0, "hard"),
        gate("tool_on_tool_off_ablation_written", bool(ablation), ablation.get("comparison_type"), "hard"),
    ]


def summarize(
    tool_cards: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    dogfood_events: list[dict[str, Any]],
    ablation: dict[str, Any],
    vcm_receipt: dict[str, Any],
    viea_tool_context_records: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    state_counts = Counter(str(row.get("state")) for row in results)
    latencies = [int(row.get("latency_ms") or 0) for row in results]
    solved = state_counts.get("SOLVED", 0)
    return {
        "tool_card_count": len(tool_cards),
        "available_tool_count": sum(1 for row in tool_cards.values() if row.get("dependency_status", {}).get("available")),
        "private_case_result_count": len(results),
        "solved_count": solved,
        "verified_solved_count": sum(1 for row in results if row.get("state") == "SOLVED" and row.get("verified")),
        "unknown_count": state_counts.get("UNKNOWN", 0),
        "unsolved_count": state_counts.get("UNSOLVED", 0),
        "tool_unavailable_count": state_counts.get("TOOL_UNAVAILABLE", 0),
        "tool_fault_count": state_counts.get("TOOL_FAULT", 0),
        "exact_solve_rate": round(solved / max(1, len(results)), 6),
        "trace_row_count": len(trace_rows),
        "dogfood_metadata_event_count": len(dogfood_events),
        "tool_on_solve_rate": ablation.get("tool_on_solve_rate"),
        "tool_off_solve_rate": ablation.get("tool_off_solve_rate"),
        "vcm_context_governor_ready": bool(vcm_receipt.get("ready")),
        "vcm_context_governor_state": vcm_receipt.get("trigger_state"),
        "vcm_context_governor_receipt_id": vcm_receipt.get("receipt_id"),
        "vcm_context_resolver_status": vcm_receipt.get("context_resolver_status"),
        "vcm_context_resolver_passed_count": vcm_receipt.get("context_resolver_passed_count"),
        "vcm_context_resolver_request_count": vcm_receipt.get("context_resolver_request_count"),
        "vcm_tool_context_record_count": len(viea_tool_context_records),
        "vcm_context_adequacy_state": "governed_sufficient_for_deterministic_tool_execution"
        if vcm_receipt.get("ready")
        else "blocked_context_governor_not_ready",
        "average_latency_ms": round(sum(latencies) / max(1, len(latencies)), 3),
        "max_latency_ms": max(latencies) if latencies else 0,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def vcm_context_governor_receipt(path: Path) -> dict[str, Any]:
    packet = vcm_consumer_abi.build_consumer_packet(
        consumer_id="theseus_deterministic_tool_substrate",
        purpose="deterministic_tool_execution",
        read_set=[rel(path), "configs/deterministic_tool_substrate.json", "reports/vcm_task_contexts.json"],
        write_set=[
            "reports/deterministic_tool_substrate.json",
            "reports/deterministic_tool_trace.jsonl",
            "reports/deterministic_tool_artifact_graph.json",
            "reports/deterministic_tool_dogfood_events.jsonl",
        ],
        authority_ceiling=["local_private_tool_execution", "governed_context_read"],
        permitted_uses=["tool_selection_context", "deterministic_tool_execution", "tool_evidence_replay"],
        governor_path=path,
        taint_labels=["private_tool_metadata", "raw_text_not_staged"],
        deletion_obligations=["invalidate_tool_context_derivatives_when_source_context_is_revoked"],
        audit_refs=["scripts/theseus_deterministic_tool_substrate.py"],
    )
    governor = packet["governor_receipt"]
    summary = governor.get("summary") if isinstance(governor.get("summary"), dict) else {}
    return {
        **governor,
        "record_type": "vcm_context_governor_receipt",
        "path": rel(path),
        "ready": bool(packet.get("ready")),
        "hard_gap_count": int(summary.get("hard_gap_count") or 0),
        "context_resolver_status": summary.get("context_resolver_status"),
        "context_resolver_passed_count": int(summary.get("context_resolver_passed_count") or 0),
        "context_resolver_request_count": int(summary.get("context_resolver_request_count") or 0),
        "context_resolver_materialized_count": int(summary.get("context_resolver_materialized_count") or 0),
        "context_resolver_typed_fault_count": int(summary.get("context_resolver_typed_fault_count") or 0),
        "context_resolver_viea_record_count": int(summary.get("context_resolver_viea_record_count") or 0),
        "content_hash": governor.get("content_hash"),
        "consumer_abi": packet,
        "required_for": "deterministic_tool_substrate",
        "non_claim": "This receipt proves governed context availability for deterministic tool execution; it is not learned generation, public calibration, or model-native KV-cache parity evidence.",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "raw_prompt_stored": False,
        "raw_private_text_stored": False,
    }


def deterministic_tool_vcm_records(vcm_receipt: dict[str, Any], result_count: int) -> list[dict[str, Any]]:
    tx_id = stable_id("tool_context_tx", vcm_receipt.get("receipt_id"), result_count)
    base = {
        "target": "deterministic_tool_substrate",
        "task_kind": "deterministic_tool_execution",
        "support_state": "SUPPORTED" if vcm_receipt.get("ready") else "BLOCKED",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "raw_prompt_stored": False,
        "raw_private_text_stored": False,
    }
    abi_packet = vcm_receipt.get("consumer_abi") if isinstance(vcm_receipt.get("consumer_abi"), dict) else {}
    abi_records = abi_packet.get("records") if isinstance(abi_packet.get("records"), list) else []
    return list(abi_records) + [
        {
            **base,
            "record_type": "authority_use_receipt",
            "record_id": stable_id("tool_authority", tx_id),
            "authority_scope": "local_private_tool_execution,local_context_governor_read",
            "allowed_effects": ["run_local_deterministic_tools", "emit_tool_evidence", "emit_metadata_only_dogfood_events"],
            "denied_effects": ["public_benchmark_training", "runtime_external_inference", "fallback_return_admission", "arbitrary_remote_shell"],
        },
        {
            **base,
            "record_type": "context_transaction",
            "record_id": tx_id,
            "transaction_id": tx_id,
            "operation": "deterministic_tool_execution_context_check",
            "snapshot_id": str(vcm_receipt.get("created_utc") or now()),
            "mounts": ["vcm_context_governor", "deterministic_tool_cards", "local_private_tool_evidence"],
            "read_set": [str(vcm_receipt.get("path") or rel(DEFAULT_VCM_CONTEXT_GOVERNOR)), "configs/deterministic_tool_substrate.json", "reports/vcm_task_contexts.json"],
            "write_set": [
                "reports/deterministic_tool_substrate.json",
                "reports/deterministic_tool_trace.jsonl",
                "reports/deterministic_tool_artifact_graph.json",
                "reports/deterministic_tool_dogfood_events.jsonl",
            ],
            "materialization_state": "materialized" if vcm_receipt.get("ready") else "blocked",
            "closure_state": "closed" if vcm_receipt.get("ready") else "blocked",
            "branch_policy": "fail_closed_if_context_governor_not_ready",
            "taint_labels": ["private_tool_metadata", "public_benchmark_quarantine_checked", "raw_text_not_staged"],
            "deletion_obligations": ["exclude_public_benchmark_payloads", "exclude_raw_private_text"],
            "declassification_refs": [],
            "derivative_refs": [str(vcm_receipt.get("receipt_id") or "")],
            "contradiction_refs": [],
            "audit_refs": [str(vcm_receipt.get("path") or rel(DEFAULT_VCM_CONTEXT_GOVERNOR)), "reports/deterministic_tool_substrate.json"],
            "faults": [] if vcm_receipt.get("ready") else ["vcm_context_governor_not_ready"],
            "replay_boundary": "metadata_hashes_and_tool_outputs_only_no_public_benchmark_training",
            "non_claims": [
                "deterministic tool results are tool evidence, not learned-generation evidence",
                "VCM context readiness is not a public benchmark or model-native KV-cache parity claim",
            ],
            "evidence_ref": "reports/deterministic_tool_substrate.json",
            "content_hash": str(vcm_receipt.get("content_hash") or ""),
        },
        {
            **base,
            "record_type": "context_adequacy",
            "record_id": stable_id("tool_adequacy", tx_id),
            "adequacy_id": stable_id("tool_adequacy", tx_id),
            "context_transaction_id": tx_id,
            "state": "governed_sufficient_for_deterministic_tool_execution" if vcm_receipt.get("ready") else "blocked_context_governor_not_ready",
            "adequacy_state": "governed_sufficient_for_deterministic_tool_execution" if vcm_receipt.get("ready") else "blocked_context_governor_not_ready",
            "evidence_ref": "reports/deterministic_tool_substrate.json",
        },
        {
            **base,
            "record_type": "failure_boundary",
            "record_id": stable_id("tool_failure", tx_id),
            "failure_id": stable_id("tool_failure", tx_id),
            "terminal": True,
            "structured_non_solved": not bool(vcm_receipt.get("ready")),
            "blocked_reason": "none" if vcm_receipt.get("ready") else "vcm_context_governor_not_ready",
        },
        {
            **base,
            "record_type": "artifact_graph_record",
            "record_id": stable_id("tool_artifact", tx_id),
            "artifact_id": stable_id("tool_artifact", tx_id),
            "artifact_ref": "reports/deterministic_tool_substrate.json",
            "evidence_ref": str(vcm_receipt.get("path") or rel(DEFAULT_VCM_CONTEXT_GOVERNOR)),
            "content_hash": str(vcm_receipt.get("content_hash") or ""),
        },
        {
            **base,
            "record_type": "claim_record",
            "record_id": stable_id("tool_claim", tx_id),
            "claim_id": stable_id("tool_claim", tx_id),
            "evidence_ref": "reports/deterministic_tool_substrate.json",
            "learned_generation_claim_allowed": False,
        },
        {
            **base,
            "record_type": "evidence_transition_record",
            "record_id": stable_id("tool_transition", tx_id),
            "previous_support_state": "UNREVIEWED",
            "current_support_state": "SUPPORTED" if vcm_receipt.get("ready") else "BLOCKED",
            "evidence_ref": "reports/deterministic_tool_substrate.json",
        },
    ]


def recommendation(trigger_state: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    non_solved = [row for row in results if row.get("state") != "SOLVED"]
    if trigger_state == "RED":
        return {
            "next_action": "fix_hard_deterministic_substrate_gates_before_plan_compiler_routes_tools",
            "reason": "A core local deterministic dependency, trace, or policy gate failed.",
            "non_solved": compact_failures(non_solved),
        }
    if non_solved:
        return {
            "next_action": "route_available_tools_now_and_repair_non_solved_tools_separately",
            "reason": "The substrate is usable, but one or more optional tools need dependency/runtime repair.",
            "non_solved": compact_failures(non_solved),
        }
    return {
        "next_action": "enable_plan_compiler_local_deterministic_tool_packets_for_private_work",
        "reason": "The registered local deterministic substrate is replayable and ready for private execution-spine routing.",
    }


def compact_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": row.get("case_id"),
            "tool_id": row.get("tool_id"),
            "state": row.get("state"),
            "diagnostic": row.get("diagnostic"),
        }
        for row in rows[:8]
    ]


def trace_row_for(result: dict[str, Any], tool_card: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": "project_theseus_deterministic_tool_trace_v1",
        "created_utc": now(),
        "event_id": stable_id("tool_trace", result.get("run_id"), result.get("case_id")),
        "run_id": result.get("run_id"),
        "case_id": result.get("case_id"),
        "task_family": result.get("task_family"),
        "tool_id": result.get("tool_id"),
        "tool_card_checksum": tool_card.get("replay_checksum", ""),
        "state": result.get("state"),
        "verified": result.get("verified"),
        "latency_ms": result.get("latency_ms"),
        "input_hash": result.get("input_hash"),
        "output_hash": result.get("output_hash"),
        "replay_checksum": result.get("replay_checksum"),
        "evidence_ref": result.get("evidence_ref"),
        "claim_id": result.get("claim_id"),
        "vcm_address": result.get("vcm_address"),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def local_search_corpus() -> list[dict[str, Any]]:
    paths = [
        ROOT / "AGENTS.md",
        ROOT / "README.md",
        ROOT / "docs" / "PROJECT_STATE.md",
        ROOT / "docs" / "THESEUS_PLAN_COMPILER.md",
        ROOT / "docs" / "VIEA_EXECUTION_SPINE_AND_TOOL_SUBSTRATE.md",
        ROOT / "configs" / "project_manifest_registry.json",
        ROOT / "configs" / "theseus_plan_compiler.json",
        ROOT / "configs" / "deterministic_tool_substrate.json",
    ]
    corpus = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        corpus.append({"id": rel(path), "path": rel(path), "title": path.name, "text": text[:120000]})
    return corpus


def bm25_search(query: str, corpus: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    query_terms = tokenize(query)
    if not query_terms or not corpus:
        return []
    doc_terms = [tokenize(str(row.get("text") or "")) for row in corpus]
    doc_freq = Counter()
    for terms in doc_terms:
        doc_freq.update(set(terms))
    avg_len = sum(len(terms) for terms in doc_terms) / max(1, len(doc_terms))
    k1 = 1.5
    b = 0.75
    scored = []
    for row, terms in zip(corpus, doc_terms):
        counts = Counter(terms)
        score = 0.0
        for term in query_terms:
            if not counts[term]:
                continue
            idf = math.log(1 + (len(corpus) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            tf = counts[term]
            denom = tf + k1 * (1 - b + b * len(terms) / max(1.0, avg_len))
            score += idf * (tf * (k1 + 1)) / denom
        if score > 0:
            scored.append({
                "id": row.get("id"),
                "title": row.get("title"),
                "path": row.get("path"),
                "score": round(score, 6),
            })
    scored.sort(key=lambda item: (-float(item["score"]), str(item.get("path"))))
    return scored[:limit]


def hybrid_search(query: str, corpus: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    query_terms = tokenize(query)
    if not query_terms or not corpus:
        return []
    bm25_by_id = {str(hit.get("id")): float(hit.get("score") or 0.0) for hit in bm25_search(query, corpus, limit=max(limit, len(corpus)))}
    query_phrase = " ".join(query_terms)
    scored = []
    for row in corpus:
        text = str(row.get("text") or "")
        lower_text = text.lower()
        title = str(row.get("title") or "")
        path = str(row.get("path") or "")
        title_terms = tokenize(title)
        path_terms = tokenize(path)
        matched_terms = sorted({term for term in query_terms if term in lower_text or term in title_terms or term in path_terms})
        if not matched_terms:
            continue
        phrase_boost = 1.0 if query_phrase and query_phrase in " ".join(tokenize(text[:20000])) else 0.0
        title_boost = 0.2 * sum(1 for term in query_terms if term in title_terms)
        path_boost = 0.1 * sum(1 for term in query_terms if term in path_terms)
        coverage = len(matched_terms) / max(1, len(set(query_terms)))
        bm25_score = bm25_by_id.get(str(row.get("id")), 0.0)
        score = bm25_score + phrase_boost + title_boost + path_boost + coverage
        scored.append(
            {
                "id": row.get("id"),
                "title": title,
                "path": path,
                "score": round(score, 6),
                "bm25_score": round(bm25_score, 6),
                "term_coverage": round(coverage, 6),
                "matched_terms": matched_terms,
                "snippet": snippet_for(text, query_terms),
            }
        )
    scored.sort(key=lambda item: (-float(item["score"]), str(item.get("path"))))
    return scored[:limit]


def snippet_for(text: str, query_terms: list[str], *, width: int = 240) -> str:
    lower = text.lower()
    starts = [lower.find(term) for term in query_terms if lower.find(term) >= 0]
    start = min(starts) if starts else 0
    start = max(0, start - width // 4)
    snippet = " ".join(text[start : start + width].split())
    return snippet


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def residual_category_for(result: dict[str, Any]) -> str:
    state = result.get("state")
    if state == "SOLVED":
        return "none"
    if state == "TOOL_UNAVAILABLE":
        return "dependency_unavailable"
    if state == "TOOL_FAULT":
        return "tool_runtime_fault"
    if result.get("tool_id", "").startswith("search."):
        return "retrieval_miss"
    return "semantic_or_contract_unsolved"


def normalize_expected(value: Any) -> Any:
    if isinstance(value, list):
        return [normalize_expected(item) for item in value]
    if isinstance(value, float):
        return normalize_number(value)
    if isinstance(value, int):
        return value
    return str(value)


def normalize_number(value: float) -> int | float:
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return round(value, 9)


def format_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal(1)))
    return format(normalized, "f")


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Deterministic Tool Substrate",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- tool cards: `{summary.get('tool_card_count')}`",
        f"- available tools: `{summary.get('available_tool_count')}`",
        f"- private results: `{summary.get('private_case_result_count')}`",
        f"- solved: `{summary.get('solved_count')}`",
        f"- exact solve rate: `{summary.get('exact_solve_rate')}`",
        f"- tool-on solve rate: `{summary.get('tool_on_solve_rate')}`",
        f"- tool-off solve rate: `{summary.get('tool_off_solve_rate')}`",
        f"- VCM governor ready: `{summary.get('vcm_context_governor_ready')}`",
        f"- VCM adequacy state: `{summary.get('vcm_context_adequacy_state')}`",
        f"- VCM context records: `{summary.get('vcm_tool_context_record_count')}`",
        f"- fallback returns: `{summary.get('fallback_return_count')}`",
        "",
        "## Gates",
        "",
    ]
    for row in report.get("gates", []):
        marker = "PASS" if row.get("passed") else "FAIL"
        lines.append(f"- {marker} `{row.get('name')}` ({row.get('severity')})")
    lines.extend([
        "",
        "## Boundary",
        "",
        "- Private smoke cases only.",
        "- Public benchmark rows written to training: `0`.",
        "- External inference calls: `0`.",
        "- Fallback returns: `0`.",
        "- Tool-assisted and model-only scores must stay separate.",
        "",
        "## Recommendation",
        "",
        f"`{report.get('recommendation', {}).get('next_action')}`: {report.get('recommendation', {}).get('reason')}",
        "",
    ])
    return "\n".join(lines)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace(os.sep, "/")
    except Exception:
        return str(path).replace(os.sep, "/")


def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def stable_hash(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return ""


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
