#!/usr/bin/env python3
"""Fixture-level simulation fidelity receipt suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "simulation_fidelity_receipt_suite.json"
DEFAULT_REPORT = ROOT / "reports" / "simulation_fidelity_receipt_suite.json"
DEFAULT_PLAN_DAGS = ROOT / "reports" / "theseus_plan_compiled_dags.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--plan-dags", default=rel(DEFAULT_PLAN_DAGS))
    parser.add_argument("--out", default=rel(DEFAULT_REPORT))
    args = parser.parse_args()

    started = time.perf_counter()
    config_path = resolve(args.config)
    plan_dags_path = resolve(args.plan_dags)
    config = read_json(config_path)
    plan_dags = read_json(plan_dags_path)
    report = build_report(config_path, config, plan_dags_path, plan_dags, started)
    write_json(resolve(args.out), report)
    print(json.dumps(gate_view(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(config_path: Path, config: dict[str, Any], plan_dags_path: Path, plan_dags: dict[str, Any], started: float) -> dict[str, Any]:
    fixtures = [audit_fixture(row) for row in list_dicts(config.get("fixtures"))]
    required = sorted(str(row) for row in list_values(config.get("required_scenarios")))
    seen = sorted({str(row.get("scenario")) for row in fixtures})
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    missing_scenarios = sorted(set(required) - set(seen))
    if missing_scenarios:
        hard_gaps.append({"id": "required_simulation_scenarios_missing", "missing": missing_scenarios})
    no_cheat = dict_value(config.get("no_cheat"))
    for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count"):
        if int_or(no_cheat.get(key), -1) != 0:
            hard_gaps.append({"id": "no_cheat_config_counter_fault", "counter": key, "value": no_cheat.get(key)})
    for row in fixtures:
        hard_gaps.extend(row["hard_gaps"])
        warnings.extend(row["warnings"])

    records = build_records(fixtures)
    planning_adapter = build_planning_adapter(plan_dags_path, plan_dags)
    hard_gaps.extend(planning_adapter["hard_gaps"])
    warnings.extend(planning_adapter["warnings"])
    for field in (
        "simulation_contract_records",
        "fidelity_records",
        "counterfactual_trace_records",
        "world_adapter_receipts",
        "failure_boundary_records",
        "claim_records",
        "evidence_transition_records",
    ):
        records[field].extend(planning_adapter["records"].get(field, []))
    trigger_state = "RED" if hard_gaps else ("YELLOW" if warnings else "GREEN")
    return {
        "policy": "project_theseus_simulation_fidelity_receipt_suite_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "config": rel(config_path),
        "summary": {
            "fixture_count": len(fixtures),
            "passed_fixture_count": sum(1 for row in fixtures if row["passed"]),
            "real_planning_adapter_count": 1 if planning_adapter["passed"] else 0,
            "planning_adapter_passed": planning_adapter["passed"],
            "planning_adapter_goal_id": planning_adapter.get("goal_id"),
            "planning_adapter_node_count": planning_adapter.get("node_count", 0),
            "planning_adapter_edge_count": planning_adapter.get("edge_count", 0),
            "required_scenario_count": len(required),
            "blocked_transfer_count": sum(1 for row in fixtures if row.get("transfer_decision") == "block_transfer"),
            "downgraded_claim_count": sum(1 for row in fixtures if row.get("transfer_decision") == "downgrade_claim"),
            "scenario_only_count": sum(1 for row in fixtures if row.get("transfer_decision") == "scenario_only_no_promotion"),
            "simulation_contract_record_count": len(records["simulation_contract_records"]),
            "fidelity_record_count": len(records["fidelity_records"]),
            "counterfactual_trace_count": len(records["counterfactual_trace_records"]),
            "world_adapter_receipt_count": len(records["world_adapter_receipts"]),
            "failure_boundary_record_count": len(records["failure_boundary_records"]),
            "claim_record_count": len(records["claim_records"]),
            "evidence_transition_record_count": len(records["evidence_transition_records"]),
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "hard_gap_count": len(hard_gaps),
            "warning_count": len(warnings),
        },
        "fixtures": fixtures,
        "planning_simulation_adapters": [planning_adapter["adapter_report"]],
        **records,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "non_claims": [
            "Simulation receipt fixtures prove claim-boundary accounting only.",
            "The planning world adapter proves bounded compile-time plan/resource/fidelity accounting only.",
            "This is not a physical feasibility result, benchmark-transfer result, live simulator result, deployment result, or learned-generation claim.",
        ],
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def build_planning_adapter(plan_dags_path: Path, plan_dags: dict[str, Any]) -> dict[str, Any]:
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    goals = list_dicts(plan_dags.get("compiled_goals"))
    if not plan_dags_path.exists():
        hard_gaps.append({"id": "planning_dag_report_missing", "path": rel(plan_dags_path)})
    if plan_dags.get("trigger_state") != "GREEN":
        hard_gaps.append({"id": "planning_dag_report_not_green", "trigger_state": plan_dags.get("trigger_state")})
    if not goals:
        hard_gaps.append({"id": "compiled_goals_missing", "path": rel(plan_dags_path)})
    for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count"):
        if int_or(plan_dags.get(key), 0) != 0:
            hard_gaps.append({"id": "planning_dag_no_cheat_counter_fault", "counter": key, "value": plan_dags.get(key)})
    goal = first_real_goal(goals)
    nodes = list_dicts(goal.get("nodes")) if goal else []
    if goal and not nodes:
        hard_gaps.append({"id": "planning_dag_nodes_missing", "goal_id": goal.get("goal_id")})
    edge_count = int_or(goal.get("edge_count"), 0) if goal else 0
    estimated_makespan = int_or(goal.get("estimated_makespan_seconds"), 0) if goal else 0
    goal_id = str(goal.get("goal_id") or "missing") if goal else "missing"
    simulation_id = stable_id("planning_world_adapter", goal_id, len(nodes), edge_count, estimated_makespan)
    passed = not hard_gaps
    support_state = "SUPPORTED" if passed else "BLOCKED"
    adapter_report = {
        **clean_counters(
            {
                "adapter_id": "planforge_compiled_dag_resource_world_adapter_v1",
                "simulation_id": simulation_id,
                "goal_id": goal_id,
                "source_report": rel(plan_dags_path),
                "node_count": len(nodes),
                "edge_count": edge_count,
                "estimated_makespan_seconds": estimated_makespan,
                "fidelity_level": "bounded_compile_time_plan_graph",
                "supported_claim_boundary": "plan shape, declared dependencies, route-local resource estimate, and evidence-presence accounting only",
                "transfer_decision": "scenario_only_no_promotion",
                "live_environment_used": False,
                "deployment_evidence": False,
                "passed": passed,
                "hard_gaps": hard_gaps,
                "warnings": warnings,
            }
        )
    }
    records = planning_adapter_records(adapter_report, plan_dags_path)
    return {
        "passed": passed,
        "goal_id": goal_id,
        "node_count": len(nodes),
        "edge_count": edge_count,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "adapter_report": adapter_report,
        "records": records,
    }


def first_real_goal(goals: list[dict[str, Any]]) -> dict[str, Any]:
    for goal in goals:
        if str(goal.get("goal_id") or "").startswith("procedural_"):
            continue
        if list_dicts(goal.get("nodes")):
            return goal
    return goals[0] if goals else {}


def planning_adapter_records(adapter: dict[str, Any], plan_dags_path: Path) -> dict[str, list[dict[str, Any]]]:
    simulation_id = str(adapter.get("simulation_id") or stable_id("planning_world_adapter"))
    goal_id = str(adapter.get("goal_id") or "missing")
    support_state = "SUPPORTED" if adapter.get("passed") else "BLOCKED"
    base = clean_counters(
        {
            "scenario": "real_bounded_planning_world_adapter",
            "target": "simulation_fidelity_receipt_suite",
            "support_state": support_state,
            "goal_id": goal_id,
            "evidence_ref": rel(plan_dags_path),
        }
    )
    blocked_reason = ";".join(gap["id"] for gap in list_dicts(adapter.get("hard_gaps"))) or "scenario_only_no_promotion"
    return {
        "simulation_contract_records": [
            {
                **base,
                "record_id": stable_id("simulation_contract", simulation_id),
                "record_type": "simulation_contract_record",
                "simulation_id": simulation_id,
                "claim_id": stable_id("simulation_claim", "real_bounded_planning_world_adapter"),
                "scope": "current Theseus compiled planning DAG",
                "fidelity_standard": adapter.get("fidelity_level"),
                "temporal_semantics": "compile_time_snapshot_only",
                "demand_estimate": {
                    "node_count": adapter.get("node_count"),
                    "edge_count": adapter.get("edge_count"),
                    "estimated_makespan_seconds": adapter.get("estimated_makespan_seconds"),
                },
                "capacity_bottlenecks": ["executor receipts still required before deployment/runtime claims"],
                "approximation_liberties": ["does not execute nodes", "does not model external latency", "does not model hidden runtime failures"],
                "supported_claim_boundary": adapter.get("supported_claim_boundary"),
                "residual_risks": ["plan may pass compile-time checks while runtime executor fails"],
                "evidence_refs": [rel(plan_dags_path)],
            }
        ],
        "fidelity_records": [
            {
                **base,
                "record_id": stable_id("fidelity_record", simulation_id),
                "record_type": "fidelity_record",
                "simulation_id": simulation_id,
                "claim_class": "compile_time_planning_resource_projection",
                "transfer_decision": "scenario_only_no_promotion",
                "instrumentation_cost_recorded": True,
                "support_boundary": adapter.get("supported_claim_boundary"),
                "fidelity_level": adapter.get("fidelity_level"),
            }
        ],
        "counterfactual_trace_records": [
            {
                **base,
                "record_id": stable_id("counterfactual_trace", simulation_id),
                "record_type": "counterfactual_trace",
                "simulation_id": simulation_id,
                "counterfactual": "claim_runtime_completion_from_compile_time_plan_shape_without_executor_receipts",
                "blocked_or_downgraded": True,
                "counterfactual_decision": "downgrade_to_compile_time_plan_evidence_only",
            }
        ],
        "world_adapter_receipts": [
            {
                **base,
                "record_id": stable_id("world_adapter_receipt", simulation_id),
                "record_type": "world_adapter_receipt",
                "simulation_id": simulation_id,
                "adapter_id": adapter.get("adapter_id"),
                "allowed_claim_boundary": adapter.get("supported_claim_boundary"),
                "live_environment_used": False,
                "source_report": rel(plan_dags_path),
            }
        ],
        "failure_boundary_records": [
            {
                **base,
                "record_id": stable_id("failure_boundary", simulation_id),
                "record_type": "failure_boundary",
                "failure_id": stable_id("simulation_failure", simulation_id),
                "blocked_reason": blocked_reason,
                "terminal": not adapter.get("passed"),
                "structured_non_solved": True,
            }
        ],
        "claim_records": [
            {
                **base,
                "record_id": stable_id("claim", simulation_id),
                "record_type": "claim_record",
                "claim_id": stable_id("simulation_claim", "real_bounded_planning_world_adapter"),
                "evidence_ref": rel(plan_dags_path),
                "support_state": support_state,
                "learned_generation_claim_allowed": False,
            }
        ],
        "evidence_transition_records": [
            {
                **base,
                "record_id": stable_id("evidence_transition", simulation_id),
                "record_type": "evidence_transition_record",
                "previous_support_state": "FIXTURE_ONLY",
                "current_support_state": support_state,
                "evidence_ref": rel(plan_dags_path),
            }
        ],
    }


def audit_fixture(raw: dict[str, Any]) -> dict[str, Any]:
    row = dict(raw)
    scenario = str(row.get("scenario") or "")
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for field in (
        "simulation_id",
        "scope",
        "fidelity_standard",
        "temporal_semantics",
        "demand_estimate",
        "capacity_bottlenecks",
        "approximation_liberties",
        "supported_claim_boundary",
        "transfer_decision",
        "world_adapter",
    ):
        value = row.get(field)
        if value in (None, "", [], {}):
            hard_gaps.append({"id": f"{field}_missing", "scenario": scenario})
    if row.get("instrumentation_cost_recorded") is not True:
        hard_gaps.append({"id": "instrumentation_cost_not_recorded", "scenario": scenario})
    decision = str(row.get("transfer_decision") or "")
    if scenario == "blocked_transfer_missing_bottleneck" and decision != "block_transfer":
        hard_gaps.append({"id": "missing_bottleneck_must_block_transfer", "scenario": scenario})
    if scenario == "downgraded_claim_after_instrumentation_cost" and decision != "downgrade_claim":
        hard_gaps.append({"id": "instrumentation_cost_must_downgrade_claim", "scenario": scenario})
    if scenario == "scenario_only_result" and decision != "scenario_only_no_promotion":
        hard_gaps.append({"id": "scenario_only_must_not_promote", "scenario": scenario})
    if not list_values(row.get("evidence_refs")):
        hard_gaps.append({"id": "evidence_refs_missing", "scenario": scenario})
    row.update(
        {
            "passed": not hard_gaps,
            "hard_gaps": hard_gaps,
            "warnings": warnings,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }
    )
    return row


def build_records(fixtures: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    simulation_contract_records = []
    fidelity_records = []
    counterfactual_trace_records = []
    world_adapter_receipts = []
    failure_boundary_records = []
    claim_records = []
    evidence_transition_records = []
    for row in fixtures:
        scenario = str(row.get("scenario") or "")
        simulation_id = str(row.get("simulation_id") or stable_id("simulation", scenario))
        support_state = "SUPPORTED" if row["passed"] else "BLOCKED"
        base = clean_counters({"scenario": scenario, "target": "simulation_fidelity_receipt_suite", "support_state": support_state})
        simulation_contract_records.append(
            {
                **base,
                "record_id": stable_id("simulation_contract", simulation_id),
                "record_type": "simulation_contract_record",
                "simulation_id": simulation_id,
                "claim_id": stable_id("simulation_claim", scenario),
                "scope": row.get("scope"),
                "fidelity_standard": row.get("fidelity_standard"),
                "temporal_semantics": row.get("temporal_semantics"),
                "demand_estimate": row.get("demand_estimate"),
                "capacity_bottlenecks": row.get("capacity_bottlenecks"),
                "approximation_liberties": row.get("approximation_liberties"),
                "supported_claim_boundary": row.get("supported_claim_boundary"),
                "residual_risks": row.get("capacity_bottlenecks"),
                "evidence_refs": row.get("evidence_refs"),
            }
        )
        fidelity_records.append(
            {
                **base,
                "record_id": stable_id("fidelity_record", simulation_id),
                "record_type": "fidelity_record",
                "simulation_id": simulation_id,
                "claim_class": row.get("claim_class"),
                "transfer_decision": row.get("transfer_decision"),
                "instrumentation_cost_recorded": row.get("instrumentation_cost_recorded"),
                "support_boundary": row.get("supported_claim_boundary"),
            }
        )
        counterfactual_trace_records.append(
            {
                **base,
                "record_id": stable_id("counterfactual_trace", simulation_id),
                "record_type": "counterfactual_trace",
                "simulation_id": simulation_id,
                "counterfactual": "claim_attempt_without_declared_fidelity_or_bottleneck",
                "blocked_or_downgraded": row.get("transfer_decision") in {"block_transfer", "downgrade_claim", "scenario_only_no_promotion"},
            }
        )
        world_adapter_receipts.append(
            {
                **base,
                "record_id": stable_id("world_adapter_receipt", simulation_id),
                "record_type": "world_adapter_receipt",
                "simulation_id": simulation_id,
                "adapter_id": row.get("world_adapter"),
                "allowed_claim_boundary": row.get("supported_claim_boundary"),
                "live_environment_used": False,
            }
        )
        failure_boundary_records.append(
            {
                **base,
                "record_id": stable_id("failure_boundary", simulation_id),
                "record_type": "failure_boundary",
                "failure_id": stable_id("simulation_failure", simulation_id),
                "blocked_reason": ";".join(gap["id"] for gap in row["hard_gaps"]) or ("none" if row.get("transfer_decision") not in {"block_transfer", "downgrade_claim"} else row.get("transfer_decision")),
                "terminal": row.get("transfer_decision") == "block_transfer" or not row["passed"],
                "structured_non_solved": row.get("transfer_decision") in {"block_transfer", "downgrade_claim", "scenario_only_no_promotion"} or not row["passed"],
            }
        )
        claim_records.append(
            {
                **base,
                "record_id": stable_id("claim", simulation_id),
                "record_type": "claim_record",
                "claim_id": stable_id("simulation_claim", scenario),
                "evidence_ref": "reports/simulation_fidelity_receipt_suite.json",
                "learned_generation_claim_allowed": False,
            }
        )
        evidence_transition_records.append(
            {
                **base,
                "record_id": stable_id("evidence_transition", simulation_id),
                "record_type": "evidence_transition_record",
                "previous_support_state": "UNREVIEWED",
                "current_support_state": support_state,
                "evidence_ref": "reports/simulation_fidelity_receipt_suite.json",
            }
        )
    return {
        "simulation_contract_records": simulation_contract_records,
        "fidelity_records": fidelity_records,
        "counterfactual_trace_records": counterfactual_trace_records,
        "world_adapter_receipts": world_adapter_receipts,
        "failure_boundary_records": failure_boundary_records,
        "claim_records": claim_records,
        "evidence_transition_records": evidence_transition_records,
    }


def gate_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report["trigger_state"],
        "policy": report["policy"],
        "summary": report["summary"],
        "hard_gaps": report["hard_gaps"],
        "warnings": report["warnings"],
    }


def clean_counters(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **payload,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "raw_prompt_stored": False,
        "raw_private_text_stored": False,
    }


def stable_id(*parts: Any) -> str:
    return stable_hash(parts)[:16]


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def list_values(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in list_values(value) if isinstance(row, dict)]


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
