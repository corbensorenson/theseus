#!/usr/bin/env python3
"""Procedural memory and toolification gate for repeated Theseus traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from procedural_memory_assets import append_lifecycle_ledger, build_assets


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "procedural_memory_toolification.json"
DEFAULT_REPORT = ROOT / "reports" / "procedural_memory_toolification.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "procedural_memory_toolification.md"
DEFAULT_CANDIDATES = ROOT / "reports" / "procedural_tool_candidates.jsonl"
DEFAULT_ROUTE_DECISIONS = ROOT / "reports" / "procedural_tool_route_decisions.jsonl"
DEFAULT_ASSETS = ROOT / "reports" / "procedural_memory_assets.json"
DEFAULT_LIFECYCLE_LEDGER = ROOT / "runtime" / "procedural_memory" / "lifecycle_ledger.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=rel(DEFAULT_REPORT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--candidates-out", default=rel(DEFAULT_CANDIDATES))
    parser.add_argument("--route-decisions-out", default=rel(DEFAULT_ROUTE_DECISIONS))
    parser.add_argument("--assets-out", default=rel(DEFAULT_ASSETS))
    parser.add_argument("--lifecycle-ledger", default=rel(DEFAULT_LIFECYCLE_LEDGER))
    args = parser.parse_args()

    started = time.perf_counter()
    config_path = resolve(args.config)
    config = read_json(config_path)
    report = build_report(config_path, config, started)
    appended = append_lifecycle_ledger(
        resolve(args.lifecycle_ledger),
        list_dicts(get_path(report, ["procedural_assets", "lifecycle_receipts"], [])),
    )
    report["summary"]["lifecycle_ledger"] = rel(resolve(args.lifecycle_ledger))
    report["summary"]["lifecycle_ledger_appended_count"] = appended
    write_json(resolve(args.out), report)
    write_json(resolve(args.assets_out), dict_value(report.get("procedural_assets")))
    write_jsonl(resolve(args.candidates_out), report["procedural_tool_candidates"])
    write_jsonl(resolve(args.route_decisions_out), report["route_decisions"])
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(gate_view(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(config_path: Path, config: dict[str, Any], started: float) -> dict[str, Any]:
    inputs = dict_value(config.get("inputs"))
    harvester = read_json(resolve(str(inputs.get("loop_closure_harvester") or "")))
    promoter = read_json(resolve(str(inputs.get("loop_closure_tool_promoter") or "")))
    viea = read_json(resolve(str(inputs.get("viea_verified_procedural_tools") or "")))
    deterministic = read_json(resolve(str(inputs.get("deterministic_tool_loop_closure_candidates") or "")))
    assistant_trace_schema = read_json(resolve(str(inputs.get("assistant_trace_schema") or "")))
    assistant_trace_events = read_assistant_trace_events(inputs)
    assistant_viea = read_assistant_viea_traces(inputs)
    boundaries = audit_boundaries(dict_value(config.get("hard_boundaries")))
    assistant_trace_audit = audit_assistant_trace_inputs(assistant_trace_schema, assistant_trace_events, assistant_viea)
    candidates = build_candidates(config, harvester, promoter, viea, deterministic, assistant_trace_schema, assistant_trace_events, assistant_viea)
    replay_results = evaluate_replay_fixtures(config, candidates, assistant_trace_audit)
    apply_replay_results(candidates, replay_results)
    route_decisions = [route_decision(config, candidate) for candidate in candidates]
    regression_decisions = [regression_fixture_decision(row) for row in list_dicts(config.get("regression_fixtures"))]
    route_decisions.extend(regression_decisions)
    canary_routes = build_canary_routes(config, candidates, route_decisions, replay_results)
    created_utc = now()
    procedural_assets = build_assets(
        candidates=candidates,
        replay_results=replay_results,
        canary_routes=canary_routes,
        events=assistant_trace_events,
        lifecycle_policy=dict_value(config.get("procedural_lifecycle")),
        created_utc=created_utc,
    )
    bind_procedural_assets(candidates, canary_routes, route_decisions, procedural_assets)
    viea_records = build_viea_procedural_tool_records(candidates, route_decisions, assistant_trace_audit, canary_routes)

    hard_gaps = [gate for gate in boundaries if gate["severity"] == "hard" and not gate["passed"]]
    hard_gaps.extend(assistant_trace_audit["hard_gaps"])
    warnings = []
    if not candidates:
        warnings.append(item_gap("procedural_memory", "no_repeated_tool_candidates", {}, severity="warning"))
    for candidate in candidates:
        if candidate["field_gap_count"] > 0:
            hard_gaps.append(item_gap(candidate["id"], "procedural_tool_field_gaps", {"missing": candidate["missing_fields"]}))
        if candidate["recurrence_count"] < int(config.get("minimum_recurrence") or 3):
            hard_gaps.append(item_gap(candidate["id"], "recurrence_below_minimum", {"recurrence_count": candidate["recurrence_count"]}))
        if candidate["benchmark_or_eval_blocked"] and candidate["route_eligible"]:
            hard_gaps.append(item_gap(candidate["id"], "benchmark_eval_workflow_route_eligible", {}))
    for decision in route_decisions:
        if decision["regression_passed"] is False and decision["route_eligible"] is True:
            hard_gaps.append(item_gap(decision["id"], "failed_regression_allowed_route", decision))
    for result in replay_results:
        if result.get("required_for_canary") and not result.get("passed"):
            hard_gaps.append(item_gap(str(result.get("id") or "assistant_trace_replay"), "required_canary_replay_fixture_failed", result))
    if procedural_assets.get("trigger_state") != "GREEN":
        hard_gaps.append(item_gap("procedural_memory_assets", "procedural_memory_assets_not_green", dict_value(procedural_assets.get("summary"))))
    required_diversity = int_or(get_path(config, ["procedural_lifecycle", "minimum_active_diverse_workflows"], 2), 2)
    observed_diversity = int_or(get_path(procedural_assets, ["summary", "diverse_binding_count"], 0), 0)
    if observed_diversity < required_diversity:
        hard_gaps.append(
            item_gap(
                "procedural_memory_assets",
                "active_workflow_diversity_below_minimum",
                {"observed": observed_diversity, "required": required_diversity},
            )
        )
    trigger_state = "GREEN"
    if hard_gaps:
        trigger_state = "RED"
    elif warnings:
        trigger_state = "YELLOW"
    summary = {
        "config": rel(config_path),
        "harvester_candidate_count": int(dict_value(harvester.get("summary")).get("candidates") or len(list_dicts(harvester.get("candidates")))),
        "assistant_trace_event_count": assistant_trace_audit["event_count"],
        "assistant_trace_schema_ready": assistant_trace_audit["schema_ready"],
        "assistant_trace_candidate_count": sum(1 for row in candidates if row.get("source_family") == "assistant_trace"),
        "assistant_viea_trace_session_count": assistant_trace_audit["viea_session_count"],
        "assistant_trace_raw_private_text_count": assistant_trace_audit["raw_private_text_count"],
        "assistant_trace_no_cheat_counter_fault_count": assistant_trace_audit["no_cheat_counter_fault_count"],
        "assistant_trace_replay_fixture_count": len(replay_results),
        "assistant_trace_replay_fixture_passed_count": sum(1 for row in replay_results if row.get("passed")),
        "canary_route_eligible_count": sum(1 for row in canary_routes if row.get("canary_route_eligible")),
        "viea_procedural_tool_record_count": len(viea_records),
        "procedural_asset_count": int_or(get_path(procedural_assets, ["summary", "asset_count"], 0), 0),
        "active_procedural_asset_count": int_or(get_path(procedural_assets, ["summary", "active_asset_count"], 0), 0),
        "retired_procedural_asset_count": int_or(get_path(procedural_assets, ["summary", "retired_asset_count"], 0), 0),
        "procedural_asset_diverse_binding_count": observed_diversity,
        "procedural_lookahead_fixture_count": int_or(get_path(procedural_assets, ["summary", "lookup_fixture_count"], 0), 0),
        "procedural_lookahead_selected_count": int_or(get_path(procedural_assets, ["summary", "lookahead_selected_count"], 0), 0),
        "procedural_lookahead_negative_control_rejected_count": int_or(get_path(procedural_assets, ["summary", "negative_control_rejected_count"], 0), 0),
        "procedural_tool_candidate_count": len(candidates),
        "route_eligible_count": sum(1 for row in route_decisions if row["route_eligible"]),
        "route_blocked_count": sum(1 for row in route_decisions if not row["route_eligible"]),
        "failed_regression_blocks_route_count": sum(1 for row in route_decisions if row["regression_passed"] is False and not row["route_eligible"]),
        "benchmark_eval_blocked_count": sum(1 for row in candidates if row["benchmark_or_eval_blocked"]),
        "public_training_rows_written": 0,
        "external_inference_calls": int(harvester.get("external_inference_calls") or 0) + int(promoter.get("external_inference_calls") or 0) + int(viea.get("external_inference_calls") or 0) + int(deterministic.get("external_inference_calls") or 0),
        "fallback_return_count": 0,
        "hard_gap_count": len(hard_gaps),
        "warning_count": len(warnings),
    }
    return {
        "policy": "project_theseus_procedural_memory_toolification_gate_v1",
        "created_utc": created_utc,
        "trigger_state": trigger_state,
        "summary": summary,
        "boundary_gates": boundaries,
        "assistant_trace_audit": assistant_trace_audit,
        "assistant_trace_replay_results": replay_results,
        "procedural_tool_candidates": candidates,
        "route_decisions": route_decisions,
        "canary_routes": canary_routes,
        "procedural_assets": procedural_assets,
        "viea_procedural_tool_records": viea_records,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "rules": {
            "recurrence": "A one-off successful trace cannot become a default tool.",
            "regression": "A failed regression blocks routable promotion.",
            "capability": "Toolification is workflow compression, not learned model capability.",
            "benchmark": "Benchmark/eval workflows become residuals or training pressure, not answer tools.",
            "lifecycle": "Every tool candidate needs monitoring and retirement criteria.",
            "lookahead": "Verified procedural assets may guide workflow routing but receive no learned-generation credit.",
        },
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def audit_boundaries(boundaries: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        gate("one_off_default_forbidden", boundaries.get("one_off_trace_can_be_default_tool") is False, "hard", boundaries.get("one_off_trace_can_be_default_tool")),
        gate("toolification_not_learned_capability", boundaries.get("toolification_counts_as_learned_model_capability") is False, "hard", boundaries.get("toolification_counts_as_learned_model_capability")),
        gate("benchmark_answer_tools_forbidden", boundaries.get("benchmark_eval_workflows_can_be_distilled_to_answer_tools") is False, "hard", boundaries.get("benchmark_eval_workflows_can_be_distilled_to_answer_tools")),
        gate("retirement_required", boundaries.get("route_without_retirement_criteria_allowed") is False, "hard", boundaries.get("route_without_retirement_criteria_allowed")),
        gate("regression_required", boundaries.get("route_without_regression_check_allowed") is False, "hard", boundaries.get("route_without_regression_check_allowed")),
        gate("external_inference_forbidden", boundaries.get("external_inference_allowed") is False, "hard", boundaries.get("external_inference_allowed")),
        gate("raw_private_text_forbidden", boundaries.get("raw_private_text_allowed") is False, "hard", boundaries.get("raw_private_text_allowed")),
    ]


def build_candidates(
    config: dict[str, Any],
    harvester: dict[str, Any],
    promoter: dict[str, Any],
    viea: dict[str, Any],
    deterministic: dict[str, Any],
    assistant_trace_schema: dict[str, Any],
    assistant_trace_events: list[dict[str, Any]],
    assistant_viea: dict[str, Any],
) -> list[dict[str, Any]]:
    required = set(str(x) for x in list_values(config.get("required_tool_fields")))
    candidates = []
    for item in list_dicts(harvester.get("candidates")):
        if int(item.get("recurrence_count") or 0) < int(config.get("minimum_recurrence") or 3):
            continue
        candidates.append(enrich_harvester_candidate(item, required))
    for item in list_dicts(viea.get("tools")):
        candidates.append(enrich_external_candidate(item, "viea_verified_procedural_tools", required))
    for item in list_dicts(deterministic.get("candidates")):
        candidates.append(enrich_external_candidate(item, "deterministic_tool_loop_closure", required))
    candidates.extend(enrich_assistant_trace_candidates(config, assistant_trace_schema, assistant_trace_events, assistant_viea, required))
    # Keep stable deterministic order and avoid duplicate ids.
    seen = set()
    unique = []
    for candidate in candidates:
        if candidate["id"] in seen:
            continue
        seen.add(candidate["id"])
        unique.append(candidate)
    return unique[:50]


def enrich_harvester_candidate(item: dict[str, Any], required: set[str]) -> dict[str, Any]:
    tool_name = str(item.get("tool_name") or "unnamed_tool")
    benchmark_blocked = str(item.get("status") or "") == "blocked_benchmark_or_eval_task_not_tool_distillable" or not bool(item.get("tool_distillation_allowed", True))
    regression_passed = float(item.get("success_rate") or 0.0) >= 0.8 and not benchmark_blocked
    route_eligible = regression_passed and bool(item.get("verification_plan")) and not benchmark_blocked
    record = {
        "id": f"procedural.{tool_name}",
        "source_traces": [{"source_workflow": item.get("source_workflow"), "recurrence_count": item.get("recurrence_count"), "success_rate": item.get("success_rate")}],
        "invariant_structure": str(item.get("source_workflow") or ""),
        "parameters": list_values(item.get("parameters_to_discover")),
        "preconditions": list_values(item.get("preconditions")),
        "postconditions": ["typed report emitted", "external_inference_calls=0", "regression gate unchanged or improved"],
        "verification_result": {"status": item.get("status"), "success_rate": item.get("success_rate"), "verification_plan": list_values(item.get("verification_plan"))},
        "risk_tier": str(item.get("risk_tier") or "medium"),
        "runtime_tier": str(item.get("runtime_tier") or "E1"),
        "monitoring": ["schema check output report", "runtime_ms trend", "external inference audit", "regression failure disables route"],
        "residuals": [str(item.get("blocked_reason") or "")] if item.get("blocked_reason") else [],
        "regressions": [{"name": "historical_success_rate_threshold", "passed": regression_passed}],
        "lifecycle_state": "candidate" if route_eligible else "blocked_candidate",
        "retirement_criteria": ["verification plan fails twice consecutively", "canonical surface supersedes workflow", "external inference or benchmark-answer-tool boundary is violated"],
        "non_claims": ["Not learned model capability.", "Not public benchmark transfer.", "Not a default route without registry eligibility."],
        "recurrence_count": int(item.get("recurrence_count") or 0),
        "benchmark_or_eval_blocked": benchmark_blocked,
        "route_eligible": route_eligible,
    }
    return finalize_record(record, required)


def enrich_external_candidate(item: dict[str, Any], source: str, required: set[str]) -> dict[str, Any]:
    name = str(item.get("tool_name") or item.get("id") or source)
    record = {
        "id": f"procedural.{name}",
        "source_traces": [{"source": source, "raw_id": item.get("id") or item.get("tool_name")}],
        "invariant_structure": str(item.get("source_workflow") or item.get("description") or source),
        "parameters": list_values(item.get("parameters") or item.get("parameters_to_discover")),
        "preconditions": list_values(item.get("preconditions")),
        "postconditions": list_values(item.get("postconditions")) or ["typed report emitted"],
        "verification_result": {"status": item.get("status") or item.get("trigger_state") or "imported"},
        "risk_tier": str(item.get("risk_tier") or "medium"),
        "runtime_tier": str(item.get("runtime_tier") or "E1"),
        "monitoring": ["schema check output report", "external inference audit"],
        "residuals": list_values(item.get("residuals")),
        "regressions": [{"name": "import_requires_future_regression", "passed": False}],
        "lifecycle_state": "candidate_needs_regression",
        "retirement_criteria": ["regression fixture missing", "surface superseded"],
        "non_claims": ["Not learned model capability.", "Imported tool candidate is not default-routable."],
        "recurrence_count": int(item.get("recurrence_count") or 3),
        "benchmark_or_eval_blocked": False,
        "route_eligible": False,
    }
    return finalize_record(record, required)


def enrich_assistant_trace_candidates(
    config: dict[str, Any],
    schema: dict[str, Any],
    events: list[dict[str, Any]],
    assistant_viea: dict[str, Any],
    required: set[str],
) -> list[dict[str, Any]]:
    minimum = int(config.get("minimum_recurrence") or 3)
    allowed_success = {"accepted", "completed"}
    allowed_repair = {"corrected"}
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for event in events:
        if not assistant_event_schema_clean(event, schema):
            continue
        surface = clean_label(event.get("surface") or "local_assistant")
        lane = clean_label(event.get("assistant_lane") or "assistant")
        intent = assistant_intent_bucket(event)
        groups.setdefault((surface, lane, intent), []).append(event)

    candidates: list[dict[str, Any]] = []
    viea_sessions = set(assistant_viea.get("sessions") or [])
    for (surface, lane, intent), rows in sorted(groups.items()):
        recurrence = len(rows)
        if recurrence < minimum:
            continue
        outcomes = [str(row.get("outcome") or row.get("feedback") or "") for row in rows]
        accepted = sum(1 for outcome in outcomes if outcome in allowed_success)
        corrected = sum(1 for outcome in outcomes if outcome in allowed_repair)
        success_rate = accepted / recurrence if recurrence else 0.0
        useful_rate = (accepted + corrected) / recurrence if recurrence else 0.0
        session_ids = sorted({str(row.get("session_id") or "") for row in rows if row.get("session_id")})
        viea_bound = bool(viea_sessions.intersection(session_ids)) if session_ids else bool(assistant_viea.get("session_count"))
        route_eligible = False
        record = {
            "id": f"procedural.assistant_trace.{stable_id(surface, lane, intent)}",
            "source_family": "assistant_trace",
            "source_traces": [
                {
                    "source": "schema_bound_dogfood_event",
                    "surface": surface,
                    "assistant_lane": lane,
                    "intent_bucket": intent,
                    "recurrence_count": recurrence,
                    "accepted_or_completed_count": accepted,
                    "corrected_count": corrected,
                    "session_ids": session_ids[:10],
                    "viea_bound": viea_bound,
                }
            ],
            "invariant_structure": f"{surface}:{lane}:{intent}",
            "parameters": ["intent_bucket", "surface", "assistant_lane", "vcm_task_family", "artifact_refs"],
            "preconditions": [
                "assistant trace schema ready",
                "prompt persisted as hash only",
                "raw private text absent",
                "public_training_rows_written=0",
                "external_inference_calls=0",
                "fallback_return_count=0",
                "VIEA receipt present before default routing",
            ],
            "postconditions": [
                "metadata-only dogfood event emitted",
                "VIEA trace or receipt linked when available",
                "accepted/missed/ignored/corrected/completed outcome preserved",
                "procedural candidate remains non-default until replay regression exists",
            ],
            "verification_result": {
                "status": "candidate_from_schema_bound_assistant_trace",
                "success_rate": success_rate,
                "useful_or_corrected_rate": useful_rate,
                "schema_policy": schema.get("policy"),
                "viea_bound": viea_bound,
                "verification_plan": [
                    "replay command contract with hash-only prompt metadata",
                    "check VCM adequacy and route-validator receipt",
                    "run assistant e2e regression for the lane",
                    "disable route on any raw text, public-training, external-inference, or fallback counter fault",
                ],
            },
            "risk_tier": "medium" if lane in {"code_assistant", "tool_assistant"} else "low",
            "runtime_tier": "E1" if lane in {"chat_checkpoint", "planning_assistant"} else "E2",
            "monitoring": [
                "assistant_trace_schema_sha256",
                "outcome distribution",
                "VIEA receipt coverage",
                "raw_private_text counter",
                "public/external/fallback counters",
                "assistant e2e regression status",
            ],
            "residuals": assistant_trace_residuals(outcomes, viea_bound),
            "regressions": [
                {
                    "name": "assistant_trace_replay_required_before_route",
                    "passed": False,
                    "reason": "candidate is harvested from real traces but no replay regression fixture has approved default routing",
                }
            ],
            "lifecycle_state": "candidate_needs_replay_regression",
            "retirement_criteria": [
                "assistant e2e regression fails twice consecutively",
                "schema hash changes without steward review",
                "raw private text appears in a candidate event",
                "public-training, runtime-external-inference, or fallback counter becomes nonzero",
                "lane is superseded by a registry-owned assistant route",
            ],
            "non_claims": [
                "Not learned model capability.",
                "Not public benchmark transfer.",
                "Not a default tool route until replay regression and registry eligibility pass.",
                "Does not store raw private prompt text.",
            ],
            "recurrence_count": recurrence,
            "benchmark_or_eval_blocked": False,
            "route_eligible": route_eligible,
        }
        candidates.append(finalize_record(record, required))
    return candidates


def assistant_trace_residuals(outcomes: list[str], viea_bound: bool) -> list[str]:
    residuals: list[str] = []
    counts = {outcome: outcomes.count(outcome) for outcome in sorted(set(outcomes))}
    if counts.get("missed"):
        residuals.append(f"missed_count={counts['missed']}")
    if counts.get("ignored"):
        residuals.append(f"ignored_count={counts['ignored']}")
    if counts.get("corrected"):
        residuals.append(f"corrected_count={counts['corrected']}")
    if not viea_bound:
        residuals.append("viea_receipt_not_bound_to_event_sessions")
    return residuals


def evaluate_replay_fixtures(config: dict[str, Any], candidates: list[dict[str, Any]], audit: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {str(candidate.get("id") or ""): candidate for candidate in candidates}
    results: list[dict[str, Any]] = []
    for fixture in list_dicts(config.get("assistant_trace_replay_fixtures")):
        candidate_id = str(fixture.get("candidate_id") or "")
        candidate = by_id.get(candidate_id)
        traces = list_dicts(candidate.get("source_traces")) if candidate else []
        trace = traces[0] if traces else {}
        verification = dict_value(candidate.get("verification_result")) if candidate else {}
        checks = []

        def add_check(name: str, passed: bool, evidence: Any) -> None:
            checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

        add_check("candidate_present", candidate is not None, candidate_id)
        add_check("surface_matches", str(trace.get("surface") or "") == str(fixture.get("expected_surface") or ""), trace.get("surface"))
        add_check("assistant_lane_matches", str(trace.get("assistant_lane") or "") == str(fixture.get("expected_assistant_lane") or ""), trace.get("assistant_lane"))
        add_check("intent_bucket_matches", str(trace.get("intent_bucket") or "") == str(fixture.get("expected_intent_bucket") or ""), trace.get("intent_bucket"))
        add_check("minimum_recurrence_met", int_or(candidate.get("recurrence_count") if candidate else 0) >= int_or(fixture.get("minimum_recurrence"), 3), candidate.get("recurrence_count") if candidate else 0)
        add_check("minimum_success_rate_met", float_or(verification.get("success_rate")) >= float_or(fixture.get("minimum_success_rate"), 1.0), verification.get("success_rate"))
        add_check("minimum_useful_rate_met", float_or(verification.get("useful_or_corrected_rate")) >= float_or(fixture.get("minimum_useful_or_corrected_rate"), 1.0), verification.get("useful_or_corrected_rate"))
        add_check("risk_tier_matches", str(candidate.get("risk_tier") if candidate else "") == str(fixture.get("required_risk_tier") or ""), candidate.get("risk_tier") if candidate else "")
        add_check("runtime_tier_matches", str(candidate.get("runtime_tier") if candidate else "") == str(fixture.get("required_runtime_tier") or ""), candidate.get("runtime_tier") if candidate else "")
        add_check("viea_bound", bool(trace.get("viea_bound")) is bool(fixture.get("require_viea_bound", True)), trace.get("viea_bound"))
        add_check("residuals_clear", not list_values(candidate.get("residuals") if candidate else []) if fixture.get("require_no_residuals", False) else True, candidate.get("residuals") if candidate else [])
        add_check("raw_private_text_absent", int_or(audit.get("raw_private_text_count"), 0) == int_or(fixture.get("expected_raw_private_text_count"), 0), audit.get("raw_private_text_count"))
        add_check("no_cheat_counters_clean", int_or(audit.get("no_cheat_counter_fault_count"), 0) == 0, audit.get("no_cheat_counter_fault_count"))
        add_check("route_not_default", candidate is not None and not bool(candidate.get("route_eligible")), candidate.get("route_eligible") if candidate else None)
        passed = all(check["passed"] for check in checks)
        results.append(
            {
                "id": str(fixture.get("id") or stable_id("assistant_trace_replay", candidate_id)),
                "candidate_id": candidate_id,
                "passed": passed,
                "required_for_canary": bool(fixture.get("required_for_canary")),
                "expected_canary_route_eligible": bool(fixture.get("expected_canary_route_eligible")),
                "checks": checks,
                "replay_contract": {
                    "prompt_replay_mode": "hash_only_metadata",
                    "raw_private_text_allowed": False,
                    "public_training_rows_written": 0,
                    "external_inference_calls": 0,
                    "fallback_return_count": 0,
                    "default_route_allowed": False,
                    "canary_only": True,
                },
            }
        )
    return results


def apply_replay_results(candidates: list[dict[str, Any]], replay_results: list[dict[str, Any]]) -> None:
    by_id = {str(candidate.get("id") or ""): candidate for candidate in candidates}
    for result in replay_results:
        candidate = by_id.get(str(result.get("candidate_id") or ""))
        if not candidate:
            continue
        regressions = list_dicts(candidate.get("regressions"))
        regressions.append(
            {
                "name": str(result.get("id") or "assistant_trace_replay_fixture"),
                "passed": bool(result.get("passed")),
                "reason": "metadata-only replay fixture validates canary eligibility, not default route adoption",
            }
        )
        candidate["regressions"] = regressions
        candidate["replay_fixture_ids"] = [str(row.get("id") or "") for row in replay_results if row.get("candidate_id") == candidate.get("id")]
        if result.get("passed") and result.get("expected_canary_route_eligible"):
            candidate["canary_route_eligible"] = True
            candidate["lifecycle_state"] = "canary_candidate_needs_registry_planner_execution"
        else:
            candidate["canary_route_eligible"] = False


def build_canary_routes(config: dict[str, Any], candidates: list[dict[str, Any]], route_decisions: list[dict[str, Any]], replay_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(candidate.get("id") or ""): candidate for candidate in candidates}
    decisions = {str(decision.get("candidate_id") or ""): decision for decision in route_decisions}
    replay_by_id = {str(result.get("id") or ""): result for result in replay_results}
    routes: list[dict[str, Any]] = []
    for row in list_dicts(config.get("canary_routes")):
        candidate_id = str(row.get("candidate_id") or "")
        candidate = by_id.get(candidate_id)
        replay = replay_by_id.get(str(row.get("replay_fixture_id") or ""))
        decision = decisions.get(candidate_id, {})
        recurrence = int_or(candidate.get("recurrence_count") if candidate else 0)
        verification_checks_before = max(0, recurrence * 4)
        verification_checks_after = len(list_dicts(replay.get("checks"))) if replay else 0
        canary_eligible = bool(candidate and candidate.get("canary_route_eligible") and replay and replay.get("passed") and row.get("registry_required") is True and row.get("default_route_allowed") is False)
        routes.append(
            {
                "id": str(row.get("id") or stable_id("canary", candidate_id)),
                "candidate_id": candidate_id,
                "owner_surface": str(row.get("owner_surface") or ""),
                "planner_mode": str(row.get("planner_mode") or "registry_gated_canary"),
                "canary_route_eligible": canary_eligible,
                "default_route_allowed": bool(row.get("default_route_allowed")),
                "registry_required": bool(row.get("registry_required")),
                "replay_fixture_id": str(row.get("replay_fixture_id") or ""),
                "replay_fixture_passed": bool(replay and replay.get("passed")),
                "route_decision": decision.get("decision"),
                "default_route_eligible": bool(decision.get("route_eligible")),
                "metrics": {
                    "before_duplicate_work_instances": recurrence,
                    "after_canary_route_instances": 1 if canary_eligible else 0,
                    "estimated_duplicate_work_delta": (1 - recurrence) if canary_eligible else 0,
                    "before_verification_check_instances": verification_checks_before,
                    "after_replay_fixture_check_instances": verification_checks_after,
                    "estimated_verification_cost_delta": verification_checks_after - verification_checks_before if canary_eligible else 0,
                    "metric_semantics": "planning estimate from repeated metadata traces; not execution throughput evidence",
                },
                "non_claims": [
                    "Canary route is not a default procedural tool.",
                    "Canary route is not learned model capability.",
                    "Duplicate-work and verification-cost deltas are planning estimates until an executor canary runs.",
                ],
                "public_training_rows_written": 0,
                "external_inference_calls": 0,
                "fallback_return_count": 0,
            }
        )
    return routes


def bind_procedural_assets(
    candidates: list[dict[str, Any]],
    canary_routes: list[dict[str, Any]],
    route_decisions: list[dict[str, Any]],
    asset_report: dict[str, Any],
) -> None:
    assets = {
        str(row.get("candidate_id") or ""): row
        for row in list_dicts(asset_report.get("assets"))
    }
    receipts = {
        str(row.get("candidate_id") or ""): row
        for row in list_dicts(asset_report.get("lifecycle_receipts"))
    }
    for candidate in candidates:
        candidate_id = str(candidate.get("id") or "")
        asset = assets.get(candidate_id, {})
        receipt = receipts.get(candidate_id, {})
        if not asset:
            continue
        candidate["procedural_asset_id"] = asset.get("id")
        candidate["procedural_asset_sha256"] = asset.get("asset_sha256")
        candidate["lifecycle_receipt_id"] = receipt.get("receipt_id")
        candidate["lifecycle_state"] = receipt.get("lifecycle_state")
        if receipt.get("lifecycle_state") != "active":
            candidate["canary_route_eligible"] = False
            candidate["route_eligible"] = False
    for route in canary_routes:
        candidate_id = str(route.get("candidate_id") or "")
        asset = assets.get(candidate_id, {})
        receipt = receipts.get(candidate_id, {})
        if not asset:
            continue
        route["procedural_asset_id"] = asset.get("id")
        route["procedural_asset_sha256"] = asset.get("asset_sha256")
        route["lifecycle_receipt_id"] = receipt.get("receipt_id")
        route["lifecycle_state"] = receipt.get("lifecycle_state")
        route["lookahead_tokens"] = asset.get("lookahead_tokens")
        if receipt.get("lifecycle_state") != "active":
            route["canary_route_eligible"] = False
    for decision in route_decisions:
        receipt = receipts.get(str(decision.get("candidate_id") or ""), {})
        if not receipt:
            continue
        decision["lifecycle_receipt_id"] = receipt.get("receipt_id")
        decision["lifecycle_state"] = receipt.get("lifecycle_state")
        if receipt.get("lifecycle_state") != "active":
            decision["route_eligible"] = False
            decision["canary_route_eligible"] = False
            decision["decision"] = "retired_or_blocked_by_lifecycle"


def assistant_event_schema_clean(event: dict[str, Any], schema: dict[str, Any]) -> bool:
    allowed = set(allowed_outcomes(schema))
    outcome = str(event.get("outcome") or event.get("feedback") or "")
    if allowed and outcome not in allowed:
        return False
    if event.get("raw_user_text") or event.get("raw_prompt") or event.get("prompt_text"):
        return False
    if int_or(event.get("public_training_rows_written"), 0) != 0:
        return False
    if int_or(event.get("external_inference_calls"), 0) != 0:
        return False
    if int_or(event.get("fallback_return_count"), 0) != 0:
        return False
    return True


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


def clean_label(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    cleaned = []
    for char in text:
        cleaned.append(char if char.isalnum() else "_")
    while "__" in "".join(cleaned):
        text = "".join(cleaned).replace("__", "_")
        cleaned = list(text)
    return "".join(cleaned).strip("_")[:80] or "unknown"


def audit_assistant_trace_inputs(schema: dict[str, Any], events: list[dict[str, Any]], assistant_viea: dict[str, Any]) -> dict[str, Any]:
    hard_gaps: list[dict[str, Any]] = []
    schema_ready = str(schema.get("policy") or "") == "project_theseus_assistant_trace_schema_v1"
    allowed = set(allowed_outcomes(schema))
    raw_private_text_count = sum(1 for row in events if row.get("raw_user_text") or row.get("raw_prompt") or row.get("prompt_text"))
    no_cheat_fault_count = sum(
        1
        for row in events
        if int_or(row.get("public_training_rows_written"), 0) != 0
        or int_or(row.get("external_inference_calls"), 0) != 0
        or int_or(row.get("fallback_return_count"), 0) != 0
    )
    invalid_outcome_count = sum(1 for row in events if allowed and str(row.get("outcome") or row.get("feedback") or "") not in allowed)
    if not schema_ready:
        hard_gaps.append(item_gap("assistant_trace", "assistant_trace_schema_not_ready", {"policy": schema.get("policy")}))
    if raw_private_text_count:
        hard_gaps.append(item_gap("assistant_trace", "raw_private_text_present", {"count": raw_private_text_count}))
    if no_cheat_fault_count:
        hard_gaps.append(item_gap("assistant_trace", "no_cheat_counter_fault", {"count": no_cheat_fault_count}))
    if invalid_outcome_count:
        hard_gaps.append(item_gap("assistant_trace", "invalid_outcome_values", {"count": invalid_outcome_count, "allowed": sorted(allowed)}))
    return {
        "schema_ready": schema_ready,
        "schema_policy": schema.get("policy"),
        "allowed_outcomes": sorted(allowed),
        "event_count": len(events),
        "viea_session_count": assistant_viea.get("session_count", 0),
        "viea_record_count": assistant_viea.get("record_count", 0),
        "raw_private_text_count": raw_private_text_count,
        "no_cheat_counter_fault_count": no_cheat_fault_count,
        "invalid_outcome_count": invalid_outcome_count,
        "hard_gaps": hard_gaps,
    }


def allowed_outcomes(schema: dict[str, Any]) -> list[str]:
    outcomes = schema.get("allowed_outcomes")
    if isinstance(outcomes, list):
        return [str(item) for item in outcomes]
    outcome_policy = dict_value(schema.get("outcome_policy"))
    outcomes = outcome_policy.get("allowed_outcomes")
    if isinstance(outcomes, list):
        return [str(item) for item in outcomes]
    return ["accepted", "missed", "ignored", "corrected", "completed"]


def read_assistant_trace_events(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in list_values(inputs.get("assistant_trace_events")):
        path = resolve(str(raw_path))
        for row in read_jsonl(path):
            if isinstance(row, dict):
                row = dict(row)
                row.setdefault("source_path", rel(path))
                rows.append(row)
    return rows


def read_assistant_viea_traces(inputs: dict[str, Any]) -> dict[str, Any]:
    sessions: set[str] = set()
    record_count = 0
    paths: list[str] = []
    record_types: dict[str, int] = {}
    for raw_path in list_values(inputs.get("assistant_viea_traces")):
        path = resolve(str(raw_path))
        if path.exists():
            paths.append(rel(path))
        for row in read_jsonl(path):
            record_count += 1
            if row.get("session_id"):
                sessions.add(str(row.get("session_id")))
            rtype = str(row.get("record_type") or "unknown")
            record_types[rtype] = record_types.get(rtype, 0) + 1
    return {
        "paths": paths,
        "sessions": sorted(sessions),
        "session_count": len(sessions),
        "record_count": record_count,
        "record_types": record_types,
    }


def finalize_record(record: dict[str, Any], required: set[str]) -> dict[str, Any]:
    missing = sorted(required - set(record))
    record["missing_fields"] = missing
    record["field_gap_count"] = len(missing)
    return record


def route_decision(config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    regressions = list_dicts(candidate.get("regressions"))
    regression_passed = all(bool(row.get("passed")) for row in regressions) if regressions else False
    retirement_ok = bool(list_values(candidate.get("retirement_criteria")))
    route_eligible = bool(candidate.get("route_eligible")) and regression_passed and retirement_ok and not bool(candidate.get("benchmark_or_eval_blocked"))
    return {
        "id": f"route.{candidate['id']}",
        "candidate_id": candidate["id"],
        "regression_passed": regression_passed,
        "retirement_criteria_present": retirement_ok,
        "benchmark_or_eval_blocked": bool(candidate.get("benchmark_or_eval_blocked")),
        "route_eligible": route_eligible,
        "canary_route_eligible": bool(candidate.get("canary_route_eligible")),
        "default_route_allowed": route_eligible,
        "decision": "route_eligible_candidate" if route_eligible else "blocked_or_candidate_only",
    }


def build_viea_procedural_tool_records(candidates: list[dict[str, Any]], route_decisions: list[dict[str, Any]], audit: dict[str, Any], canary_routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decision_by_candidate = {str(row.get("candidate_id") or ""): row for row in route_decisions}
    records: list[dict[str, Any]] = []
    created = now()
    for candidate in candidates:
        cid = str(candidate.get("id") or "")
        decision = decision_by_candidate.get(cid, {})
        base = {
            "candidate_id": cid,
            "record_id": "",
            "created_utc": created,
            "source_family": candidate.get("source_family") or "loop_closure",
            "lifecycle_state": candidate.get("lifecycle_state"),
                    "route_eligible": bool(decision.get("route_eligible")),
                    "canary_route_eligible": bool(decision.get("canary_route_eligible")),
                    "recurrence_count": candidate.get("recurrence_count"),
            "risk_tier": candidate.get("risk_tier"),
            "runtime_tier": candidate.get("runtime_tier"),
            "procedural_asset_id": candidate.get("procedural_asset_id", ""),
            "procedural_asset_sha256": candidate.get("procedural_asset_sha256", ""),
            "lifecycle_receipt_id": candidate.get("lifecycle_receipt_id", ""),
            "support_state": "candidate_only_not_capability_claim",
            "raw_private_text_stored": False,
            "learned_generation_claim_allowed": False,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }
        records.extend(
            [
                {
                    **base,
                    "record_id": f"procedural_tool_record-{stable_id(cid, 'tool')}",
                    "record_type": "procedural_tool_record",
                    "tool_id": cid,
                    "preconditions": candidate.get("preconditions"),
                    "postconditions": candidate.get("postconditions"),
                    "monitoring": candidate.get("monitoring"),
                    "retirement_criteria": candidate.get("retirement_criteria"),
                    "non_claims": candidate.get("non_claims"),
                    "verification_result": candidate.get("verification_result"),
                },
                {
                    **base,
                    "record_id": f"claim_record-{stable_id(cid, 'claim')}",
                    "record_type": "claim_record",
                    "claim_id": f"claim.procedural_tool_candidate.{stable_id(cid)}",
                    "claim": "This repeated workflow is a procedural-memory candidate only.",
                    "support_state": "candidate_supported_by_metadata_trace_not_default_route",
                    "evidence_ref": "reports/procedural_memory_toolification.json",
                },
                {
                    **base,
                    "record_id": f"artifact_graph_record-{stable_id(cid, 'artifact')}",
                    "record_type": "artifact_graph_record",
                    "artifact_ref": "reports/procedural_tool_candidates.jsonl",
                    "content_hash": stable_id(cid, candidate.get("source_traces"), candidate.get("verification_result")),
                },
                {
                    **base,
                    "record_id": f"authority_use_receipt-{stable_id(cid, 'authority')}",
                    "record_type": "authority_use_receipt",
                    "authority_scope": "candidate_only_no_default_route_without_replay_regression_and_registry_eligibility",
                    "status": "approved_for_review_only",
                },
                {
                    **base,
                    "record_id": f"generation_mode-{stable_id(cid, 'generation')}",
                    "record_type": "generation_mode",
                    "candidate_generation_credit": "none",
                    "blocked_reason": "procedural toolification is workflow compression, not learned generation",
                },
                {
                    **base,
                    "record_id": f"failure_boundary-{stable_id(cid, 'failure')}",
                    "record_type": "failure_boundary",
                    "failure_id": f"failure.procedural_tool_candidate.{stable_id(cid)}",
                    "blocked_reason": "; ".join(str(item) for item in list_values(candidate.get("residuals"))) or "replay_regression_required_before_route",
                    "terminal": False,
                    "fallback_return_used": False,
                },
                {
                    **base,
                    "record_id": f"evidence_transition_record-{stable_id(cid, 'evidence')}",
                    "record_type": "evidence_transition_record",
                    "previous_support_state": "metadata_trace_observed",
                    "current_support_state": "procedural_candidate_review_queued",
                    "evidence_ref": "reports/procedural_memory_toolification.json",
                },
            ]
        )
    records.append(
        {
            "record_id": f"policy_optimization_record-{stable_id('procedural_memory_toolification', audit.get('event_count'), audit.get('viea_record_count'))}",
            "record_type": "policy_optimization_record",
            "support_state": "assistant_trace_harvest_policy_active",
            "event_count": audit.get("event_count"),
            "assistant_viea_record_count": audit.get("viea_record_count"),
            "raw_private_text_stored": False,
            "learned_generation_claim_allowed": False,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }
    )
    for route in canary_routes:
        rid = str(route.get("id") or "")
        base = {
            "record_id": "",
            "created_utc": created,
            "route_id": rid,
            "candidate_id": route.get("candidate_id"),
            "procedural_asset_id": route.get("procedural_asset_id", ""),
            "procedural_asset_sha256": route.get("procedural_asset_sha256", ""),
            "lifecycle_receipt_id": route.get("lifecycle_receipt_id", ""),
            "lifecycle_state": route.get("lifecycle_state", ""),
            "support_state": "registry_gated_canary_not_default_route",
            "route_phase": "canary",
            "route_validator_ready": bool(route.get("canary_route_eligible")),
            "learned_generation_claim_allowed": False,
            "raw_private_text_stored": False,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }
        records.extend(
            [
                {
                    **base,
                    "record_id": f"costed_route-{stable_id(rid, 'cost')}",
                    "record_type": "costed_route",
                    "task_kind": "procedural_memory_canary_route",
                    "estimated_latency_ms": int_or(get_path(route, ["metrics", "after_replay_fixture_check_instances"], 0)),
                    "task_fit": "metadata_replay_canary",
                    "blocked_reason": "" if route.get("canary_route_eligible") else "replay_or_registry_canary_not_ready",
                },
                {
                    **base,
                    "record_id": f"resource_budget-{stable_id(rid, 'resource')}",
                    "record_type": "resource_budget",
                    "worker_limit": "local_T0_metadata_replay_only",
                    "task_fit": "low_risk_assistant_trace_replay",
                },
                {
                    **base,
                    "record_id": f"runtime_adapter_invocation-{stable_id(rid, 'adapter')}",
                    "record_type": "runtime_adapter_invocation",
                    "adapter_id": "theseus_plan_compiler.procedural_memory_canary",
                    "status": "canary_route_packet_emitted" if route.get("canary_route_eligible") else "blocked_or_candidate_only",
                },
                {
                    **base,
                    "record_id": f"evidence_transition_record-{stable_id(rid, 'route_evidence')}",
                    "record_type": "evidence_transition_record",
                    "previous_support_state": "procedural_candidate_review_queued",
                    "current_support_state": "registry_gated_canary_route_packet_ready" if route.get("canary_route_eligible") else "canary_route_blocked",
                    "evidence_ref": "reports/procedural_memory_toolification.json",
                },
            ]
        )
    return records


def regression_fixture_decision(row: dict[str, Any]) -> dict[str, Any]:
    regression_passed = bool(row.get("regression_passed"))
    expected = bool(row.get("expected_route_eligible"))
    return {
        "id": str(row.get("id") or ""),
        "candidate_id": str(row.get("candidate_tool_name") or ""),
        "regression_passed": regression_passed,
        "retirement_criteria_present": True,
        "benchmark_or_eval_blocked": False,
        "route_eligible": expected if regression_passed else False,
        "decision": "fixture_blocks_route" if not regression_passed else "fixture_allows_route",
        "reason": str(row.get("reason") or ""),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Procedural Memory Toolification",
        "",
        f"- trigger_state: `{report['trigger_state']}`",
        f"- candidates: `{report['summary']['procedural_tool_candidate_count']}`",
        f"- route eligible: `{report['summary']['route_eligible_count']}` blocked: `{report['summary']['route_blocked_count']}`",
        f"- failed regression blocks route: `{report['summary']['failed_regression_blocks_route_count']}`",
        f"- benchmark/eval blocked: `{report['summary']['benchmark_eval_blocked_count']}`",
        f"- active procedural assets: `{report['summary']['active_procedural_asset_count']}` diverse bindings: `{report['summary']['procedural_asset_diverse_binding_count']}`",
        f"- lookahead selections: `{report['summary']['procedural_lookahead_selected_count']}/{report['summary']['procedural_lookahead_fixture_count']}` negative controls rejected: `{report['summary']['procedural_lookahead_negative_control_rejected_count']}`",
        f"- hard gaps: `{report['summary']['hard_gap_count']}` warnings: `{report['summary']['warning_count']}`",
        "",
        "## Candidates",
        "",
    ]
    for candidate in report["procedural_tool_candidates"][:20]:
        lines.append(f"- `{candidate['id']}` recurrence=`{candidate['recurrence_count']}` lifecycle=`{candidate['lifecycle_state']}` route_eligible=`{candidate['route_eligible']}`")
    lines.extend(["", "## Hard Gaps", ""])
    if report["hard_gaps"]:
        for item in report["hard_gaps"]:
            lines.append(f"- `{item['id']}` `{item['kind']}`: `{json.dumps(item['evidence'], sort_keys=True)}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Rules", ""])
    for key, value in report["rules"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def gate_view(report: dict[str, Any]) -> dict[str, Any]:
    return {"policy": report["policy"], "created_utc": report["created_utc"], "trigger_state": report["trigger_state"], "summary": report["summary"], "hard_gaps": report["hard_gaps"], "warnings": report["warnings"]}


def gate(name: str, passed: bool, severity: str, evidence: Any) -> dict[str, Any]:
    return {"id": name, "kind": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def item_gap(item_id: str, kind: str, evidence: dict[str, Any], severity: str = "hard") -> dict[str, Any]:
    return {"id": item_id, "kind": kind, "passed": False, "severity": severity, "evidence": evidence}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
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
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def list_values(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def float_or(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def int_or(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def stable_id(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def get_path(payload: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
