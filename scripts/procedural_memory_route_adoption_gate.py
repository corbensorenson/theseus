#!/usr/bin/env python3
"""Adopt replay-proven procedural-memory routes through a registry transaction.

This is the step after canary execution. It can mark a procedural route as a
default local metadata route only when replay, canary execution, registry, and
steward evidence all remain clean. It never creates learned-generation credit.
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
DEFAULT_TOOLIFICATION = ROOT / "reports" / "procedural_memory_toolification.json"
DEFAULT_CANARY = ROOT / "reports" / "procedural_memory_canary_execution.json"
DEFAULT_REGISTRY = ROOT / "reports" / "theseus_project_registry.json"
DEFAULT_STEWARD = ROOT / "configs" / "project_steward.json"
DEFAULT_OUT = ROOT / "reports" / "procedural_memory_route_adoption.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "procedural_memory_route_adoption.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--toolification", default=rel(DEFAULT_TOOLIFICATION))
    parser.add_argument("--canary-execution", default=rel(DEFAULT_CANARY))
    parser.add_argument("--registry", default=rel(DEFAULT_REGISTRY))
    parser.add_argument("--steward", default=rel(DEFAULT_STEWARD))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(
        config_path=resolve(args.config),
        toolification_path=resolve(args.toolification),
        canary_path=resolve(args.canary_execution),
        registry_path=resolve(args.registry),
        steward_path=resolve(args.steward),
        started=started,
    )
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(gate_view(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(
    *,
    config_path: Path,
    toolification_path: Path,
    canary_path: Path,
    registry_path: Path,
    steward_path: Path,
    started: float,
) -> dict[str, Any]:
    config = read_json(config_path)
    toolification = read_json(toolification_path)
    canary = read_json(canary_path)
    registry = read_json(registry_path)
    steward = read_json(steward_path)
    policy = dict_value(config.get("default_route_adoption_policy"))

    transaction = evaluate_policy(policy, toolification, canary, registry, steward)
    default_routes = [transaction["default_route"]] if transaction.get("default_route") else []
    records = build_viea_records(transaction)
    replacement_kernel = build_replacement_transaction_kernel(
        transaction=transaction,
        default_routes=default_routes,
        toolification=toolification,
        canary=canary,
        registry=registry,
        steward=steward,
    )
    hard_gaps = list_dicts(transaction.get("hard_gaps"))
    if replacement_kernel["state"] != "GREEN":
        hard_gaps.append(gap("A2_replacement_transaction_kernel", "replacement_transaction_kernel_not_green", replacement_kernel))
    warnings = list_dicts(transaction.get("warnings"))
    trigger_state = "GREEN"
    if hard_gaps:
        trigger_state = "RED"
    elif warnings:
        trigger_state = "YELLOW"
    summary = {
        "config": rel(config_path),
        "toolification": rel(toolification_path),
        "canary_execution": rel(canary_path),
        "registry": rel(registry_path),
        "steward": rel(steward_path),
        "transaction_count": 1 if transaction else 0,
        "default_route_adopted_count": sum(1 for row in default_routes if row.get("default_route_adopted")),
        "default_route_guarded_count": sum(1 for row in default_routes if row.get("continued_regression_guard", {}).get("armed")),
        "a2_replacement_transaction_kernel_state": replacement_kernel["state"],
        "a2_replacement_transaction_kernel_support_state": replacement_kernel["support_state"],
        "learned_generation_claim_count": sum(1 for row in default_routes if row.get("learned_generation_claim_allowed")),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "hard_gap_count": len(hard_gaps),
        "warning_count": len(warnings),
        "viea_route_adoption_record_count": len(records),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }
    return {
        "policy": "project_theseus_procedural_memory_route_adoption_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": summary,
        "adoption_transaction": transaction,
        "replacement_transaction_kernel": replacement_kernel,
        "default_routes": default_routes,
        "viea_route_adoption_records": records,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "non_claims": [
            "Default procedural routes are workflow compression, not learned model generation.",
            "Tool/router/procedural route behavior must be reported separately from learned code generation.",
            "This adoption gate does not train on public benchmarks or serve external inference.",
        ],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def evaluate_policy(
    policy: dict[str, Any],
    toolification: dict[str, Any],
    canary: dict[str, Any],
    registry: dict[str, Any],
    steward: dict[str, Any],
) -> dict[str, Any]:
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    policy_id = str(policy.get("id") or "")
    candidate_id = str(policy.get("candidate_id") or "")
    route_id = str(policy.get("source_canary_route_id") or "")
    fixture_id = str(policy.get("required_replay_fixture_id") or "")
    route_binding = dict_value(policy.get("route_binding_contract"))
    candidate = find_by_id(toolification.get("procedural_tool_candidates"), candidate_id)
    replay = find_by_id(toolification.get("assistant_trace_replay_results"), fixture_id)
    canary_execution = find_execution(canary, route_id, candidate_id, fixture_id)
    work_contract = find_by_id(steward.get("project_work_contracts"), str(policy.get("required_steward_work_contract") or ""))

    checks = [
        check("adoption_policy_present", bool(policy_id), policy_id),
        check("toolification_gate_green", toolification.get("trigger_state") == "GREEN", toolification.get("trigger_state")),
        check("canary_execution_green", canary.get("trigger_state") == "GREEN", canary.get("trigger_state")),
        check("registry_gate_green", registry.get("trigger_state") == "GREEN", registry.get("trigger_state")),
        check("route_validator_ready", get_path(registry, ["summary", "route_validator_viea_spine_view_ready"], False) is True, get_path(registry, ["summary", "route_validator_viea_spine_record_count"], 0)),
        check("steward_contract_present", bool(work_contract), policy.get("required_steward_work_contract")),
        check("steward_contract_blocks_unqualified_default_route", "no_default_route_without_registry_eligibility" in str(work_contract.get("authority_ceiling") or ""), work_contract.get("authority_ceiling")),
        check("candidate_present", bool(candidate), candidate_id),
        check("candidate_canary_eligible", candidate.get("canary_route_eligible") is True, candidate.get("canary_route_eligible")),
        check("candidate_risk_allowed", risk_rank(candidate.get("risk_tier")) <= risk_rank(policy.get("maximum_risk_tier") or "low"), candidate.get("risk_tier")),
        check("candidate_runtime_tier_matches", str(candidate.get("runtime_tier") or "") == str(policy.get("required_runtime_tier") or ""), candidate.get("runtime_tier")),
        check("candidate_retirement_criteria_present", bool(candidate.get("retirement_criteria")), len(list_values(candidate.get("retirement_criteria")))),
        check("candidate_residuals_clear", not list_values(candidate.get("residuals")), candidate.get("residuals")),
        check("replay_fixture_passed", replay.get("passed") is True, replay.get("passed")),
        check("route_binding_contract_present", bool(route_binding), route_binding),
        check("route_binding_has_selection_keys", bool(list_values(route_binding.get("selection_keys"))), route_binding.get("selection_keys")),
        check("route_binding_has_runtime_consumers", bool(list_values(route_binding.get("runtime_consumers"))), route_binding.get("runtime_consumers")),
        check("route_binding_includes_assistant_runtime", "theseus_assistant_runtime" in {str(item) for item in list_values(route_binding.get("runtime_consumers"))}, route_binding.get("runtime_consumers")),
        check("route_binding_includes_plan_compiler", "theseus_plan_compiler" in {str(item) for item in list_values(route_binding.get("runtime_consumers"))}, route_binding.get("runtime_consumers")),
        check("route_binding_matches_candidate_trace", route_binding_matches_candidate(route_binding, candidate), route_binding_evidence(route_binding, candidate)),
        check("route_binding_matches_replay_fixture", route_binding_matches_replay(route_binding, replay), route_binding_evidence(route_binding, replay)),
        check("canary_execution_present", bool(canary_execution), route_id),
        check("canary_executed", canary_execution.get("executed") is True, canary_execution.get("executed")),
        check("canary_default_not_already_adopted", canary_execution.get("default_route_adopted") is False, canary_execution.get("default_route_adopted")),
        check("canary_no_learned_generation_claim", canary_execution.get("learned_generation_claim_allowed") is False, canary_execution.get("learned_generation_claim_allowed")),
        check("matched_event_floor_met", int_or(canary_execution.get("matched_event_count"), 0) >= int_or(policy.get("minimum_matched_event_count"), 1), canary_execution.get("matched_event_count")),
        check("duplicate_work_delta_useful", int_or(get_path(canary_execution, ["metrics", "actual_duplicate_work_delta"], 0), 0) < int_or(policy.get("require_actual_duplicate_work_delta_below"), 0), get_path(canary_execution, ["metrics", "actual_duplicate_work_delta"], 0)),
        check("verification_cost_delta_useful", int_or(get_path(canary_execution, ["metrics", "metadata_verification_cost_delta"], 0), 0) < int_or(policy.get("require_metadata_verification_cost_delta_below"), 0), get_path(canary_execution, ["metrics", "metadata_verification_cost_delta"], 0)),
        check("rollback_criteria_present", bool(policy.get("rollback_criteria")), len(list_values(policy.get("rollback_criteria")))),
        check("no_public_training_rows", no_cheat_counter(toolification, "public_training_rows_written") == 0 and no_cheat_counter(canary, "public_training_rows_written") == 0, {"toolification": no_cheat_counter(toolification, "public_training_rows_written"), "canary": no_cheat_counter(canary, "public_training_rows_written")}),
        check("no_external_inference_calls", no_cheat_counter(toolification, "external_inference_calls") == 0 and no_cheat_counter(canary, "external_inference_calls") == 0, {"toolification": no_cheat_counter(toolification, "external_inference_calls"), "canary": no_cheat_counter(canary, "external_inference_calls")}),
        check("no_fallback_returns", no_cheat_counter(toolification, "fallback_return_count") == 0 and no_cheat_counter(canary, "fallback_return_count") == 0, {"toolification": no_cheat_counter(toolification, "fallback_return_count"), "canary": no_cheat_counter(canary, "fallback_return_count")}),
    ]
    for row in checks:
        if not row["passed"]:
            hard_gaps.append(gap(policy_id or "procedural_memory_route_adoption", row["name"], {"evidence": row.get("evidence")}))

    default_route_adopted = bool(policy.get("allow_default_route_when_all_gates_pass")) and not hard_gaps
    guard = {
        "armed": default_route_adopted,
        "guard_id": stable_id("procedural_route_guard", policy_id, candidate_id, route_id),
        "revalidate_command": "python3 scripts/procedural_memory_toolification_gate.py && python3 scripts/theseus_plan_compiler.py && python3 scripts/procedural_memory_canary_executor.py && python3 scripts/procedural_memory_route_adoption_gate.py",
        "rollback_criteria": list_values(policy.get("rollback_criteria")),
        "dependency_hashes": {
            "toolification": stable_hash(compact_dependency(toolification)),
            "canary_execution": stable_hash(compact_dependency(canary)),
            "registry": stable_hash({
                "trigger_state": registry.get("trigger_state"),
                "route_validator_ready": get_path(registry, ["summary", "route_validator_viea_spine_view_ready"], False),
                "route_validator_records": get_path(registry, ["summary", "route_validator_viea_spine_record_count"], 0),
            }),
        },
    }
    default_route = None
    if default_route_adopted:
        default_route = {
            "id": policy_id,
            "candidate_id": candidate_id,
            "source_canary_route_id": route_id,
            "replay_fixture_id": fixture_id,
            "owner_surface": str(policy.get("owner_surface") or ""),
            "planner_mode": str(policy.get("planner_mode") or ""),
            "route_scope": str(route_binding.get("scope") or ""),
            "route_binding_contract": route_binding,
            "assistant_surfaces": [str(item) for item in list_values(route_binding.get("assistant_surfaces"))],
            "assistant_intents": [str(item) for item in list_values(route_binding.get("assistant_intents"))],
            "assistant_lanes": [str(item) for item in list_values(route_binding.get("assistant_lanes"))],
            "vcm_task_families": [str(item) for item in list_values(route_binding.get("vcm_task_families"))],
            "runtime_consumers": [str(item) for item in list_values(route_binding.get("runtime_consumers"))],
            "selection_keys": [str(item) for item in list_values(route_binding.get("selection_keys"))],
            "lifecycle_state": "default_route_adopted_regression_guarded",
            "default_route_adopted": True,
            "default_route_authority_ceiling": str(policy.get("default_route_authority_ceiling") or ""),
            "learned_generation_claim_allowed": False,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "continued_regression_guard": guard,
            "metrics": dict_value(canary_execution.get("metrics")),
            "non_claims": list_values(policy.get("non_claims")),
        }
    return {
        "transaction_id": stable_id("procedural_default_route_adoption", policy_id, candidate_id, route_id),
        "policy_id": policy_id,
        "candidate_id": candidate_id,
        "source_canary_route_id": route_id,
        "replay_fixture_id": fixture_id,
        "decision": "adopt_default_route" if default_route_adopted else "blocked",
        "default_route": default_route,
        "checks": checks,
        "continued_regression_guard": guard,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def build_viea_records(transaction: dict[str, Any]) -> list[dict[str, Any]]:
    route = dict_value(transaction.get("default_route"))
    policy_id = str(transaction.get("policy_id") or "")
    candidate_id = str(transaction.get("candidate_id") or "")
    route_id = str(transaction.get("source_canary_route_id") or "")
    adopted = bool(route.get("default_route_adopted"))
    support_state = "default_route_adopted_regression_guarded" if adopted else "default_route_adoption_blocked"
    created = now()
    claim_id = f"claim-{stable_id('procedural_route_adoption_claim', policy_id, candidate_id)}"
    artifact_id = f"artifact-{stable_id('procedural_route_adoption', transaction.get('transaction_id'))}"
    base = {
        "created_utc": created,
        "transaction_id": transaction.get("transaction_id"),
        "route_id": policy_id,
        "source_canary_route_id": route_id,
        "candidate_id": candidate_id,
        "route_scope": route.get("route_scope", ""),
        "route_binding_contract": route.get("route_binding_contract", {}),
        "assistant_intents": route.get("assistant_intents", []),
        "assistant_lanes": route.get("assistant_lanes", []),
        "assistant_surfaces": route.get("assistant_surfaces", []),
        "runtime_consumers": route.get("runtime_consumers", []),
        "support_state": support_state,
        "default_route_adopted": adopted,
        "learned_generation_claim_allowed": False,
        "raw_private_text_stored": False,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    return [
        {
            **base,
            "record_id": f"registry_adoption_record-{stable_id(policy_id, 'adoption')}",
            "record_type": "registry_adoption_record",
            "decision": transaction.get("decision"),
            "guard_id": get_path(transaction, ["continued_regression_guard", "guard_id"], ""),
        },
        {
            **base,
            "record_id": f"procedural_tool_record-{stable_id(policy_id, 'procedural_tool')}",
            "record_type": "procedural_tool_record",
            "lifecycle_state": support_state,
        },
        {
            **base,
            "record_id": claim_id,
            "record_type": "claim_record",
            "claim": "procedural memory route adopted as local default metadata route under regression guard",
            "non_claims": ["not learned generation", "not public transfer", "not external runtime inference"],
        },
        {
            **base,
            "record_id": f"authority_use_receipt-{stable_id(policy_id, 'authority')}",
            "record_type": "authority_use_receipt",
            "principal": "procedural_memory_route_adoption_gate",
            "authority_ceiling": route.get("default_route_authority_ceiling", "none"),
            "allowed_effect": "registry_default_route_metadata_packet_only" if adopted else "none",
        },
        {
            **base,
            "record_id": f"runtime_adapter_invocation-{stable_id(policy_id, 'runtime')}",
            "record_type": "runtime_adapter_invocation",
            "adapter_id": "theseus_plan_compiler.procedural_memory_default_route",
            "status": "default_route_adopted" if adopted else "blocked",
        },
        {
            **base,
            "record_id": f"resource_budget-{stable_id(policy_id, 'resource')}",
            "record_type": "resource_budget",
            "capacity_pool": "local_T0_metadata_route",
            "risk_class": "low",
        },
        {
            **base,
            "record_id": f"costed_route-{stable_id(policy_id, 'cost')}",
            "record_type": "costed_route",
            "selected_route": policy_id if adopted else "",
            "actual_duplicate_work_delta": get_path(route, ["metrics", "actual_duplicate_work_delta"], 0),
            "metadata_verification_cost_delta": get_path(route, ["metrics", "metadata_verification_cost_delta"], 0),
            "promotion_candidate": False,
        },
        {
            **base,
            "record_id": f"generation_mode-{stable_id(policy_id, 'generation')}",
            "record_type": "generation_mode",
            "generation_mode": "procedural_memory_default_metadata_route",
            "learned_generation_claim_allowed": False,
        },
        {
            **base,
            "record_id": f"failure_boundary-{stable_id(policy_id, 'failure')}",
            "record_type": "failure_boundary",
            "protected_invariant": "default procedural route cannot claim learned generation or survive failed guard checks",
            "rollback_criteria": get_path(transaction, ["continued_regression_guard", "rollback_criteria"], []),
        },
        {
            **base,
            "record_id": artifact_id,
            "record_type": "artifact_graph_record",
            "artifact_type": "procedural_memory_default_route_transaction",
            "source_refs": ["reports/procedural_memory_toolification.json", "reports/procedural_memory_canary_execution.json", "reports/theseus_project_registry.json"],
            "claim_refs": [claim_id],
        },
        {
            **base,
            "record_id": f"evidence_transition_record-{stable_id(policy_id, 'evidence')}",
            "record_type": "evidence_transition_record",
            "claim_id": claim_id,
            "old_support_state": "canary_executed_not_default_route",
            "new_support_state": support_state,
            "evidence_ref": "reports/procedural_memory_route_adoption.json",
            "verification_result": "passed" if adopted else "blocked",
        },
        {
            **base,
            "record_id": f"policy_optimization_record-{stable_id(policy_id, 'policy')}",
            "record_type": "policy_optimization_record",
            "policy_decision": "adopt_default_route_with_regression_guard" if adopted else "block_default_route",
            "feedback_loop": "revalidate replay/canary/registry/steward evidence before route remains default",
        },
    ]


def build_replacement_transaction_kernel(
    *,
    transaction: dict[str, Any],
    default_routes: list[dict[str, Any]],
    toolification: dict[str, Any],
    canary: dict[str, Any],
    registry: dict[str, Any],
    steward: dict[str, Any],
) -> dict[str, Any]:
    checks = list_dicts(transaction.get("checks"))
    route = default_routes[0] if default_routes else {}
    guard = dict_value(transaction.get("continued_regression_guard"))
    failed_checks = [row for row in checks if row.get("passed") is not True]
    rollback_criteria = [str(item) for item in list_values(guard.get("rollback_criteria"))]
    dependency_hashes = dict_value(guard.get("dependency_hashes"))
    hard_gaps = []
    if transaction.get("decision") != "adopt_default_route":
        hard_gaps.append({"kind": "transaction_not_adopted", "decision": transaction.get("decision")})
    if failed_checks:
        hard_gaps.append({"kind": "failed_prechecks", "failed_checks": failed_checks})
    if not route.get("default_route_adopted"):
        hard_gaps.append({"kind": "default_route_not_adopted"})
    if route.get("learned_generation_claim_allowed") is not False:
        hard_gaps.append({"kind": "learned_generation_claim_not_disabled", "value": route.get("learned_generation_claim_allowed")})
    if guard.get("armed") is not True:
        hard_gaps.append({"kind": "regression_guard_not_armed", "guard": guard})
    if not rollback_criteria:
        hard_gaps.append({"kind": "rollback_criteria_missing"})
    if not dependency_hashes:
        hard_gaps.append({"kind": "dependency_hashes_missing"})
    if any(no_cheat_counter(payload, key) != 0 for payload in [toolification, canary] for key in ["public_training_rows_written", "external_inference_calls", "fallback_return_count"]):
        hard_gaps.append({"kind": "no_cheat_counter_fault"})

    precheck_receipts = [
        {
            "name": row.get("name"),
            "passed": bool(row.get("passed")),
            "evidence_summary": compact_public_evidence(row.get("evidence")),
        }
        for row in checks
    ]
    independent_evaluators = [
        {
            "surface": "procedural_memory_toolification_gate",
            "path": "reports/procedural_memory_toolification.json",
            "trigger_state": toolification.get("trigger_state"),
            "role": "candidate, replay, route-block, and no-cheat evaluator",
        },
        {
            "surface": "procedural_memory_canary_executor",
            "path": "reports/procedural_memory_canary_execution.json",
            "trigger_state": canary.get("trigger_state"),
            "role": "bounded canary execution and regression-cost evaluator",
        },
        {
            "surface": "theseus_project_registry",
            "path": "reports/theseus_project_registry.json",
            "trigger_state": registry.get("trigger_state"),
            "role": "registered surface, SCF, route-validator, and steward coverage evaluator",
        },
        {
            "surface": "project_steward",
            "path": "configs/project_steward.json",
            "trigger_state": "config",
            "role": "authority ceiling and work-contract evaluator",
        },
    ]
    residual_escrow = {
        "known_residuals": [
            "adopted route is local metadata workflow compression only",
            "adopted route is not learned generation, public transfer, model quality, or ASI evidence",
            "future procedural candidates must repeat the same regression-guarded transaction",
        ],
        "transaction_hard_gap_count": len(list_dicts(transaction.get("hard_gaps"))),
        "transaction_warning_count": len(list_dicts(transaction.get("warnings"))),
        "failed_regression_blocked_count": int_or(get_path(toolification, ["summary", "failed_regression_blocks_route_count"], 0)),
        "blocked_route_count": int_or(get_path(toolification, ["summary", "route_blocked_count"], 0)),
        "non_claims": [
            "replacement transaction supports only guarded route adoption",
            "procedural route compression cannot support learned-generation claims",
        ],
    }
    rollback_receipt = {
        "guard_armed": guard.get("armed") is True,
        "guard_id": guard.get("guard_id"),
        "rollback_criteria": rollback_criteria,
        "revalidate_command": guard.get("revalidate_command"),
        "dependency_hashes": dependency_hashes,
        "rollback_or_no_rollback_state": "rollback_guard_available" if rollback_criteria and guard.get("armed") else "rollback_guard_missing",
    }
    support_state_transition = {
        "claim": "procedural planning-assistant metadata route may become a guarded default route",
        "from_state": "canary_executed_not_default_route",
        "to_state": "default_route_adopted_regression_guarded" if route.get("default_route_adopted") else "blocked",
        "support_state": "prototype-backed" if not hard_gaps else "unsupported",
        "evidence_refs": [
            "reports/procedural_memory_toolification.json",
            "reports/procedural_memory_canary_execution.json",
            "reports/theseus_project_registry.json",
            "reports/procedural_memory_route_adoption.json",
        ],
    }
    expected_invalid_controls = [
        {
            "control": "failed_precheck_blocks_adoption",
            "rejected": bool(failed_checks) is False and all(row.get("passed") is True for row in checks),
            "reason": "all real prechecks must pass before default adoption",
        },
        {
            "control": "missing_rollback_blocks_adoption",
            "rejected": bool(rollback_criteria and guard.get("armed")),
            "reason": "default adoption requires armed rollback/regression criteria",
        },
        {
            "control": "learned_generation_overclaim_blocks_adoption",
            "rejected": route.get("learned_generation_claim_allowed") is False,
            "reason": "procedural route adoption cannot claim learned generation",
        },
        {
            "control": "no_cheat_counter_fault_blocks_adoption",
            "rejected": all(no_cheat_counter(payload, key) == 0 for payload in [toolification, canary] for key in ["public_training_rows_written", "external_inference_calls", "fallback_return_count"]),
            "reason": "adoption requires public/external/fallback counters at zero",
        },
        {
            "control": "residuals_not_erased",
            "rejected": bool(residual_escrow["known_residuals"]),
            "reason": "known residuals are preserved as escrow/non-claims instead of hidden",
        },
    ]
    if any(not row["rejected"] for row in expected_invalid_controls):
        hard_gaps.append({"kind": "expected_invalid_control_not_rejected", "controls": expected_invalid_controls})

    state = "GREEN" if not hard_gaps else "RED"
    return {
        "policy": "project_theseus_a2_replacement_transaction_kernel_v1",
        "state": state,
        "support_state": "prototype-backed" if state == "GREEN" else "unsupported",
        "slice_id": "A2_replacement_transaction_kernel",
        "transaction_id": transaction.get("transaction_id"),
        "decision": transaction.get("decision"),
        "precheck_receipts": precheck_receipts,
        "independent_evaluators": independent_evaluators,
        "regression_guard": rollback_receipt,
        "residual_escrow": residual_escrow,
        "support_state_transition": support_state_transition,
        "expected_invalid_controls": expected_invalid_controls,
        "hard_gaps": hard_gaps,
        "non_claims": [
            "A2 prototype-backed means one guarded route replacement transaction passed; it is not model-quality or ASI evidence.",
            "The adopted route is local metadata workflow compression only.",
            "The transaction cannot support learned-generation claims, public-transfer claims, or external inference serving.",
        ],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def compact_public_evidence(value: Any) -> Any:
    if isinstance(value, dict):
        allowed = {}
        for key, item in value.items():
            if key in {"trigger_state", "ready", "record_count", "matched_event_count", "actual_duplicate_work_delta", "metadata_verification_cost_delta"}:
                allowed[key] = item
            elif isinstance(item, (str, int, float, bool)) and len(str(item)) < 120:
                allowed[key] = item
        return allowed
    if isinstance(value, list):
        return [compact_public_evidence(item) for item in value[:8]]
    return value


def check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def find_by_id(rows: Any, target: str) -> dict[str, Any]:
    for row in list_dicts(rows):
        if str(row.get("id") or "") == target:
            return row
    return {}


def find_execution(report: dict[str, Any], route_id: str, candidate_id: str, fixture_id: str) -> dict[str, Any]:
    for row in list_dicts(report.get("executions")):
        if (
            str(row.get("route_id") or "") == route_id
            and str(row.get("candidate_id") or "") == candidate_id
            and str(row.get("replay_fixture_id") or "") == fixture_id
        ):
            return row
    return {}


def route_binding_matches_candidate(binding: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if not binding or not candidate:
        return False
    traces = list_dicts(candidate.get("source_traces"))
    if not traces:
        return False
    return any(route_binding_matches_trace(binding, trace) for trace in traces)


def route_binding_matches_replay(binding: dict[str, Any], replay: dict[str, Any]) -> bool:
    if not binding or not replay:
        return False
    check_values = {
        str(row.get("name") or ""): row.get("evidence")
        for row in list_dicts(replay.get("checks"))
        if row.get("passed") is True
    }
    trace = {
        "surface": check_values.get("surface_matches"),
        "assistant_lane": check_values.get("assistant_lane_matches"),
        "intent_bucket": check_values.get("intent_bucket_matches"),
    }
    return route_binding_matches_trace(binding, trace)


def route_binding_matches_trace(binding: dict[str, Any], trace: dict[str, Any]) -> bool:
    surfaces = {str(item) for item in list_values(binding.get("assistant_surfaces"))}
    intents = {str(item) for item in list_values(binding.get("assistant_intents"))}
    lanes = {str(item) for item in list_values(binding.get("assistant_lanes"))}
    surface = str(trace.get("surface") or "")
    intent = str(trace.get("intent_bucket") or trace.get("intent") or "")
    lane = str(trace.get("assistant_lane") or "")
    return (
        bool(surface and intent and lane)
        and (not surfaces or surface in surfaces)
        and (not intents or intent in intents)
        and (not lanes or lane in lanes)
    )


def route_binding_evidence(binding: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "binding": {
            "assistant_surfaces": list_values(binding.get("assistant_surfaces")),
            "assistant_intents": list_values(binding.get("assistant_intents")),
            "assistant_lanes": list_values(binding.get("assistant_lanes")),
            "vcm_task_families": list_values(binding.get("vcm_task_families")),
            "runtime_consumers": list_values(binding.get("runtime_consumers")),
        },
        "candidate_source_traces": payload.get("source_traces") if isinstance(payload.get("source_traces"), list) else [],
        "replay_checks": payload.get("checks") if isinstance(payload.get("checks"), list) else [],
    }


def risk_rank(value: Any) -> int:
    ranks = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    return ranks.get(str(value or "").lower(), 99)


def no_cheat_counter(payload: dict[str, Any], key: str) -> int:
    return int_or(payload.get(key), int_or(get_path(payload, ["summary", key], 0), 0))


def compact_dependency(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": payload.get("policy"),
        "trigger_state": payload.get("trigger_state"),
        "summary": payload.get("summary"),
    }


def gate_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "summary": report.get("summary"),
        "hard_gaps": report.get("hard_gaps", []),
        "warnings": report.get("warnings", []),
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = dict_value(report.get("summary"))
    lines = [
        "# Procedural Memory Route Adoption",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- transactions: `{summary.get('transaction_count')}`",
        f"- default routes adopted: `{summary.get('default_route_adopted_count')}`",
        f"- guarded default routes: `{summary.get('default_route_guarded_count')}`",
        f"- learned-generation claims: `{summary.get('learned_generation_claim_count')}`",
        f"- VIEA adoption records: `{summary.get('viea_route_adoption_record_count')}`",
        f"- hard gaps: `{summary.get('hard_gap_count')}`",
        "",
        "## Boundary",
        "",
        "- Adoption is local metadata route adoption only.",
        "- It is not learned-generation evidence.",
        "- Public benchmark training, runtime external inference, and fallback returns stay forbidden.",
    ]
    return "\n".join(lines) + "\n"


def gap(item_id: str, kind: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {"id": item_id, "kind": kind, "passed": False, "severity": "hard", "evidence": evidence}


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
