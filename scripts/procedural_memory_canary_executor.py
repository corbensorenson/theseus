#!/usr/bin/env python3
"""Execute replay-backed procedural-memory canaries in bounded metadata mode.

This is not a default-route adopter and not learned-generation evidence. It
turns a replay-passed procedural candidate into a local route-packet execution
receipt, then records the measured metadata deltas and VIEA records needed by
the shared spine gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "procedural_memory_toolification.json"
DEFAULT_PROCEDURAL_REPORT = ROOT / "reports" / "procedural_memory_toolification.json"
DEFAULT_COMPILED_DAGS = ROOT / "reports" / "theseus_plan_compiled_dags.json"
DEFAULT_OUT = ROOT / "reports" / "procedural_memory_canary_execution.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "procedural_memory_canary_execution.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--procedural-report", default=rel(DEFAULT_PROCEDURAL_REPORT))
    parser.add_argument("--compiled-dags", default=rel(DEFAULT_COMPILED_DAGS))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(
        config_path=resolve(args.config),
        procedural_report_path=resolve(args.procedural_report),
        compiled_dags_path=resolve(args.compiled_dags),
        started=started,
    )
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(gate_view(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(*, config_path: Path, procedural_report_path: Path, compiled_dags_path: Path, started: float) -> dict[str, Any]:
    config = read_json(config_path)
    procedural = read_json(procedural_report_path)
    dags = read_json(compiled_dags_path)
    schema = read_json(resolve(str(get_path(config, ["inputs", "assistant_trace_schema"], ""))))
    events = read_assistant_trace_events(dict_value(config.get("inputs")))

    candidates = {
        str(row.get("id") or ""): row
        for row in list_dicts(procedural.get("procedural_tool_candidates"))
    }
    config_fixtures = {
        str(row.get("id") or ""): row
        for row in list_dicts(config.get("assistant_trace_replay_fixtures"))
    }
    replay_results = {
        str(row.get("id") or ""): row
        for row in list_dicts(procedural.get("assistant_trace_replay_results"))
    }
    fixtures = {
        fixture_id: {**config_fixtures.get(fixture_id, {}), **replay}
        for fixture_id, replay in replay_results.items()
    }
    goals = {
        str(row.get("goal_id") or ""): row
        for row in list_dicts(dags.get("compiled_goals"))
    }

    executions: list[dict[str, Any]] = []
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for route in list_dicts(procedural.get("canary_routes")):
        if route.get("canary_route_eligible") is not True:
            continue
        execution = execute_canary_route(route, candidates, fixtures, goals, events, schema)
        executions.append(execution)
        records.extend(execution.get("viea_canary_execution_records", []))
        hard_gaps.extend(execution.get("hard_gaps", []))
        warnings.extend(execution.get("warnings", []))

    if not executions:
        hard_gaps.append(gap("procedural_memory_canary", "no_eligible_canary_routes", {}))
    if str(procedural.get("trigger_state") or "") != "GREEN":
        hard_gaps.append(gap("procedural_memory_canary", "procedural_memory_gate_not_green", {"state": procedural.get("trigger_state")}))
    if str(dags.get("trigger_state") or "") != "GREEN":
        hard_gaps.append(gap("procedural_memory_canary", "compiled_dags_not_green", {"state": dags.get("trigger_state")}))

    trigger_state = "GREEN"
    if hard_gaps:
        trigger_state = "RED"
    elif warnings:
        trigger_state = "YELLOW"

    summary = {
        "config": rel(config_path),
        "procedural_report": rel(procedural_report_path),
        "compiled_dags": rel(compiled_dags_path),
        "eligible_canary_route_count": len(executions),
        "executed_canary_route_count": sum(1 for row in executions if row.get("executed")),
        "default_route_adopted_count": sum(1 for row in executions if row.get("default_route_adopted")),
        "learned_generation_claim_count": sum(1 for row in executions if row.get("learned_generation_claim_allowed")),
        "matched_event_count": sum(int_or(row.get("matched_event_count"), 0) for row in executions),
        "actual_duplicate_work_delta_total": sum(int_or(get_path(row, ["metrics", "actual_duplicate_work_delta"], 0), 0) for row in executions),
        "metadata_verification_cost_delta_total": sum(int_or(get_path(row, ["metrics", "metadata_verification_cost_delta"], 0), 0) for row in executions),
        "viea_canary_execution_record_count": len(records),
        "hard_gap_count": len(hard_gaps),
        "warning_count": len(warnings),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }
    return {
        "policy": "project_theseus_procedural_memory_canary_execution_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": summary,
        "executions": executions,
        "viea_canary_execution_records": records,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "non_claims": [
            "Canary execution is not default procedural tool adoption.",
            "Canary execution is not learned model capability.",
            "Metadata verification deltas are bounded local execution evidence, not wall-clock throughput claims.",
        ],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def execute_canary_route(
    route: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    fixtures: dict[str, dict[str, Any]],
    goals: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
    schema: dict[str, Any],
) -> dict[str, Any]:
    route_id = str(route.get("id") or "")
    candidate_id = str(route.get("candidate_id") or "")
    fixture_id = str(route.get("replay_fixture_id") or "")
    candidate = candidates.get(candidate_id, {})
    fixture = fixtures.get(fixture_id, {})
    goal_id = f"procedural_memory_canary_{slug(route_id)}"
    goal = goals.get(goal_id, {})
    expected_nodes = {
        f"{goal_id}.validate_replay_fixture",
        f"{goal_id}.compile_canary_route_packet",
        f"{goal_id}.measure_duplicate_and_verification_delta",
    }
    observed_nodes = {str(row.get("node_id") or "") for row in list_dicts(goal.get("nodes"))}
    fixture_checks = list_dicts(fixture.get("checks"))
    fixture_passed = bool(fixture.get("passed")) and all(row.get("passed") is True for row in fixture_checks)
    matched_events = matching_events(events, schema, fixture)
    repeated_trace_count = len(matched_events)
    recurrence = int_or(candidate.get("recurrence_count"), 0)

    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not candidate:
        hard_gaps.append(gap(route_id, "candidate_missing", {"candidate_id": candidate_id}))
    if not fixture:
        hard_gaps.append(gap(route_id, "replay_fixture_missing", {"fixture_id": fixture_id}))
    if not fixture_passed:
        hard_gaps.append(gap(route_id, "replay_fixture_not_passed", {"fixture_id": fixture_id}))
    if route.get("default_route_allowed") is not False:
        hard_gaps.append(gap(route_id, "default_route_allowed_for_canary", {"default_route_allowed": route.get("default_route_allowed")}))
    if not goal:
        hard_gaps.append(gap(route_id, "compiled_canary_goal_missing", {"goal_id": goal_id}))
    if expected_nodes - observed_nodes:
        hard_gaps.append(gap(route_id, "compiled_canary_nodes_missing", {"missing": sorted(expected_nodes - observed_nodes)}))
    if repeated_trace_count != recurrence:
        hard_gaps.append(gap(route_id, "matched_event_count_differs_from_candidate_recurrence", {"matched_event_count": repeated_trace_count, "candidate_recurrence": recurrence}))
    if any(event.get("raw_user_text") or event.get("raw_prompt") or event.get("prompt_text") for event in matched_events):
        hard_gaps.append(gap(route_id, "raw_private_text_present_in_matched_events", {}))
    if any(no_cheat_fault(event) for event in matched_events):
        hard_gaps.append(gap(route_id, "no_cheat_counter_fault_in_matched_events", {}))

    executed = not hard_gaps
    emitted_route_packet_count = 1 if executed else 0
    metadata_obligations_per_trace = 4
    before_verification_obligations = repeated_trace_count * metadata_obligations_per_trace
    after_verification_obligations = len(fixture_checks)
    metrics = {
        "matched_event_count": repeated_trace_count,
        "candidate_recurrence_count": recurrence,
        "emitted_route_packet_count": emitted_route_packet_count,
        "actual_duplicate_work_delta": emitted_route_packet_count - repeated_trace_count,
        "metadata_obligations_per_trace": metadata_obligations_per_trace,
        "before_metadata_verification_obligations": before_verification_obligations,
        "after_replay_fixture_check_count": after_verification_obligations,
        "metadata_verification_cost_delta": after_verification_obligations - before_verification_obligations,
        "metric_semantics": "Executed local metadata canary over current event stream and replay fixture; not wall-clock throughput evidence.",
    }
    route_packet = {
        "packet_id": stable_id("procedural_canary_packet", route_id, candidate_id, fixture_id, goal_id),
        "route_id": route_id,
        "candidate_id": candidate_id,
        "fixture_id": fixture_id,
        "goal_id": goal_id,
        "planner_mode": route.get("planner_mode"),
        "default_route_allowed": False,
        "default_route_adopted": False,
        "owner_surface": route.get("owner_surface"),
        "compiled_node_ids": sorted(observed_nodes.intersection(expected_nodes)),
        "matched_event_count": repeated_trace_count,
        "fixture_check_count": len(fixture_checks),
        "packet_hash": "",
    }
    route_packet["packet_hash"] = stable_hash(route_packet)
    receipts = [
        {
            "node_id": f"{goal_id}.validate_replay_fixture",
            "status": "passed" if fixture_passed and executed else "blocked",
            "evidence": {"fixture_id": fixture_id, "check_count": len(fixture_checks)},
        },
        {
            "node_id": f"{goal_id}.compile_canary_route_packet",
            "status": "passed" if executed else "blocked",
            "evidence": {"packet_id": route_packet["packet_id"], "default_route_adopted": False},
        },
        {
            "node_id": f"{goal_id}.measure_duplicate_and_verification_delta",
            "status": "passed" if executed else "blocked",
            "evidence": metrics,
        },
    ]
    return {
        "id": stable_id("procedural_canary_execution", route_id, route_packet["packet_hash"]),
        "route_id": route_id,
        "candidate_id": candidate_id,
        "replay_fixture_id": fixture_id,
        "goal_id": goal_id,
        "executed": executed,
        "default_route_adopted": False,
        "learned_generation_claim_allowed": False,
        "matched_event_count": repeated_trace_count,
        "route_packet": route_packet,
        "execution_receipts": receipts,
        "metrics": metrics,
        "viea_canary_execution_records": build_viea_records(route, candidate, fixture, goal_id, route_packet, metrics, executed),
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def build_viea_records(
    route: dict[str, Any],
    candidate: dict[str, Any],
    fixture: dict[str, Any],
    goal_id: str,
    route_packet: dict[str, Any],
    metrics: dict[str, Any],
    executed: bool,
) -> list[dict[str, Any]]:
    route_id = str(route.get("id") or "")
    candidate_id = str(route.get("candidate_id") or "")
    created = now()
    support_state = "canary_executed_not_default_route" if executed else "canary_execution_blocked"
    base = {
        "created_utc": created,
        "route_id": route_id,
        "candidate_id": candidate_id,
        "goal_id": goal_id,
        "support_state": support_state,
        "default_route_adopted": False,
        "learned_generation_claim_allowed": False,
        "raw_private_text_stored": False,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    claim_id = f"claim-{stable_id('procedural_canary_claim', route_id)}"
    artifact_id = f"artifact-{stable_id('procedural_canary_artifact', route_packet.get('packet_hash'))}"
    return [
        {
            **base,
            "record_id": f"procedural_tool_record-{stable_id(route_id, 'procedural_tool')}",
            "record_type": "procedural_tool_record",
            "lifecycle_state": "canary_executed_needs_registry_adoption" if executed else "canary_blocked",
            "source_family": candidate.get("source_family"),
            "recurrence_count": candidate.get("recurrence_count"),
            "replay_fixture_id": fixture.get("id"),
        },
        {
            **base,
            "record_id": claim_id,
            "record_type": "claim_record",
            "claim": "procedural memory canary route packet executed in bounded metadata mode",
            "support_state": support_state,
            "non_claims": ["not default adoption", "not learned generation", "not public transfer"],
        },
        {
            **base,
            "record_id": f"authority_use_receipt-{stable_id(route_id, 'authority')}",
            "record_type": "authority_use_receipt",
            "principal": "procedural_memory_canary_executor",
            "authority_ceiling": "local_T0_metadata_replay_only",
            "allowed_effect": "write_reports_only",
            "default_route_authority_granted": False,
        },
        {
            **base,
            "record_id": f"runtime_adapter_invocation-{stable_id(route_id, 'runtime')}",
            "record_type": "runtime_adapter_invocation",
            "adapter_id": "procedural_memory_canary_executor.local_metadata_replay",
            "status": "executed" if executed else "blocked",
            "packet_hash": route_packet.get("packet_hash"),
        },
        {
            **base,
            "record_id": f"resource_budget-{stable_id(route_id, 'resource')}",
            "record_type": "resource_budget",
            "capacity_pool": "local_T0",
            "matched_event_count": metrics.get("matched_event_count"),
            "after_replay_fixture_check_count": metrics.get("after_replay_fixture_check_count"),
        },
        {
            **base,
            "record_id": f"costed_route-{stable_id(route_id, 'cost')}",
            "record_type": "costed_route",
            "selected_route": "registry_gated_canary",
            "actual_duplicate_work_delta": metrics.get("actual_duplicate_work_delta"),
            "metadata_verification_cost_delta": metrics.get("metadata_verification_cost_delta"),
            "promotion_candidate": False,
        },
        {
            **base,
            "record_id": f"generation_mode-{stable_id(route_id, 'generation')}",
            "record_type": "generation_mode",
            "generation_mode": "procedural_memory_metadata_route_packet",
            "learned_generation_claim_allowed": False,
            "quality_or_pass_result": "not_applicable_metadata_canary",
        },
        {
            **base,
            "record_id": f"failure_boundary-{stable_id(route_id, 'failure')}",
            "record_type": "failure_boundary",
            "protected_invariant": "canary execution cannot adopt a default route or claim learned generation",
            "containment_action": "block_default_route_and_emit_residual",
            "residual_risk": "needs_executor_canary_review_before_registry_adoption" if executed else "canary_blocked",
        },
        {
            **base,
            "record_id": artifact_id,
            "record_type": "artifact_graph_record",
            "artifact_type": "procedural_memory_canary_route_packet",
            "source_refs": ["reports/procedural_memory_toolification.json", "reports/theseus_plan_compiled_dags.json"],
            "artifact_refs": [route_packet.get("packet_id"), route_packet.get("packet_hash")],
            "claim_refs": [claim_id],
        },
        {
            **base,
            "record_id": f"evidence_transition_record-{stable_id(route_id, 'evidence')}",
            "record_type": "evidence_transition_record",
            "claim_id": claim_id,
            "old_support_state": "registry_gated_canary_route_packet_ready",
            "new_support_state": support_state,
            "evidence_ref": "reports/procedural_memory_canary_execution.json",
            "verification_result": "passed" if executed else "blocked",
        },
    ]


def matching_events(events: list[dict[str, Any]], schema: dict[str, Any], fixture: dict[str, Any]) -> list[dict[str, Any]]:
    expected_surface = clean_label(fixture.get("expected_surface") or "")
    expected_lane = clean_label(fixture.get("expected_assistant_lane") or "")
    expected_intent = clean_label(fixture.get("expected_intent_bucket") or "")
    rows = []
    for event in events:
        if not assistant_event_schema_clean(event, schema):
            continue
        surface = clean_label(event.get("surface") or "local_assistant")
        lane = clean_label(event.get("assistant_lane") or "assistant")
        intent = assistant_intent_bucket(event)
        if surface == expected_surface and lane == expected_lane and intent == expected_intent:
            rows.append(event)
    return rows


def read_assistant_trace_events(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in list_values(inputs.get("assistant_trace_events")):
        path = resolve(str(raw_path))
        for row in read_jsonl(path):
            row = dict(row)
            row.setdefault("source_path", rel(path))
            rows.append(row)
    return rows


def assistant_event_schema_clean(event: dict[str, Any], schema: dict[str, Any]) -> bool:
    allowed = set(allowed_outcomes(schema))
    outcome = str(event.get("outcome") or event.get("feedback") or "")
    if allowed and outcome not in allowed:
        return False
    return not no_cheat_fault(event) and not (event.get("raw_user_text") or event.get("raw_prompt") or event.get("prompt_text"))


def assistant_intent_bucket(event: dict[str, Any]) -> str:
    explicit = str(event.get("intent") or "").strip()
    if explicit:
        return clean_label(explicit.split(";", 1)[0])
    summary = str(event.get("intent_summary_redacted") or "")
    if summary.startswith("intent="):
        summary = summary.removeprefix("intent=").split(";", 1)[0]
    elif "intent=" in summary:
        summary = summary.split("intent=", 1)[1].split(";", 1)[0]
    elif summary:
        summary = summary.split(":", 1)[0].split("_metadata", 1)[0]
    return clean_label(summary or "general")


def allowed_outcomes(schema: dict[str, Any]) -> list[str]:
    outcomes = schema.get("allowed_outcomes")
    if isinstance(outcomes, list):
        return [str(item) for item in outcomes]
    outcome_policy = dict_value(schema.get("outcome_policy"))
    outcomes = outcome_policy.get("allowed_outcomes")
    if isinstance(outcomes, list):
        return [str(item) for item in outcomes]
    return ["accepted", "missed", "ignored", "corrected", "completed"]


def no_cheat_fault(payload: dict[str, Any]) -> bool:
    return (
        int_or(payload.get("public_training_rows_written"), 0) != 0
        or int_or(payload.get("external_inference_calls"), 0) != 0
        or int_or(payload.get("fallback_return_count"), 0) != 0
    )


def gate_view(report: dict[str, Any]) -> dict[str, Any]:
    summary = dict_value(report.get("summary"))
    return {
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "summary": summary,
        "hard_gaps": report.get("hard_gaps", []),
        "warnings": report.get("warnings", []),
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = dict_value(report.get("summary"))
    lines = [
        "# Procedural Memory Canary Execution",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- eligible routes: `{summary.get('eligible_canary_route_count')}`",
        f"- executed routes: `{summary.get('executed_canary_route_count')}`",
        f"- default routes adopted: `{summary.get('default_route_adopted_count')}`",
        f"- learned-generation claims: `{summary.get('learned_generation_claim_count')}`",
        f"- matched event count: `{summary.get('matched_event_count')}`",
        f"- actual duplicate-work delta: `{summary.get('actual_duplicate_work_delta_total')}`",
        f"- metadata verification-cost delta: `{summary.get('metadata_verification_cost_delta_total')}`",
        f"- VIEA records: `{summary.get('viea_canary_execution_record_count')}`",
        f"- hard gaps: `{summary.get('hard_gap_count')}`",
        "",
        "## Boundary",
        "",
        "- This is a local metadata replay canary.",
        "- It does not adopt a default route.",
        "- It does not support learned-generation or public-transfer claims.",
    ]
    return "\n".join(lines) + "\n"


def gap(item_id: str, kind: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {"id": item_id, "kind": kind, "passed": False, "severity": "hard", "evidence": evidence}


def clean_label(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    cleaned = []
    for char in text:
        cleaned.append(char if char.isalnum() else "_")
    while "__" in "".join(cleaned):
        text = "".join(cleaned).replace("__", "_")
        cleaned = list(text)
    return "".join(cleaned).strip("_")[:80] or "unknown"


def slug(value: str) -> str:
    return clean_label(value)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def list_values(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def int_or(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_path(payload: Any, path: list[str], default: Any = None) -> Any:
    cur = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def stable_id(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
